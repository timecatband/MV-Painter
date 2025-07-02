"""
MVPainter API Server
A FastAPI server that provides texture generation for 3D models using MVPainter pipeline.
Compatible with Hunyuan API but specialized for MVPainter workflow.
"""
import argparse
import asyncio
import base64
import logging
import logging.handlers
import os
import sys
import tempfile
import threading
import traceback
import uuid
import shutil
import subprocess
import time
import gc
from io import BytesIO
from glob import glob

import torch
import trimesh
import uvicorn
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# MVPainter specific imports
from mvpainter.mvpainter_pipeline import unscale_latents, unscale_image, unscale_image_2, MVPainter_Pipeline
from mvpainter.controlnet import ControlNetModel_Union
from mvpainter.bake_pipeline import BakePipeline
from scripts.remesh_reduce_blender_script import reduce_mesh
from src.utils.infer_util import remove_background, resize_foreground
import rembg

# Optional imports
try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False
    print("Warning: OpenEXR not available. Some depth processing features may not work.")

try:
    from pytorch_lightning import seed_everything
except ImportError:
    # Fallback seed function
    def seed_everything(seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

LOGDIR = '.'
SAVE_DIR = 'mvpainter_cache'
RENDER_TEMP_DIR = 'render_temp'

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RENDER_TEMP_DIR, exist_ok=True)

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger("mvpainter_controller", f"{SAVE_DIR}/mvpainter_controller.log")

# Global variables
model_semaphore = None


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def depth_exr_to_png(exr_file, png_file, depth_channel='R', depth_scale=6.0):
    """Convert EXR depth image to PNG format"""
    if not HAS_OPENEXR:
        logger.warning("OpenEXR not available, skipping depth conversion")
        # Create a dummy depth image if OpenEXR is not available
        dummy_depth = np.ones((512, 512), dtype=np.uint16) * 32767
        Image.fromarray(dummy_depth).save(png_file)
        return
        
    exr_image = OpenEXR.InputFile(exr_file)
    header = exr_image.header()
    size = (header['displayWindow'].max.x + 1, header['displayWindow'].max.y + 1)
    depth_channel = exr_image.channel(depth_channel, Imath.PixelType(Imath.PixelType.FLOAT))

    depth_array = np.frombuffer(depth_channel, dtype=np.float32).copy()
    depth_array = depth_array.reshape((size[1], size[0]))

    invalid_mask = depth_array == 1.0
    depth_array /= depth_scale
    depth_array_uint8 = (depth_array * 65535).astype(np.uint16)
    depth_array_uint8[invalid_mask] = 65535

    depth_image = Image.fromarray(depth_array_uint8)
    depth_image.save(png_file)


class MVPainterWorker:
    def __init__(self,
                 unet_ckpt='/home/racarr/v29_25000.ckpt',
                 blender_path='/home/racarr/blender/blender',
                 device='cuda',
                 seed=12,
                 limit_model_concurrency=5):
        self.worker_id = worker_id
        self.device = device
        self.blender_path = blender_path
        self.seed = seed
        self.limit_model_concurrency = limit_model_concurrency
        
        # Set random seeds
        seed_everything(seed)
        
        logger.info(f"Loading MVPainter model on worker {worker_id} ...")

        # Initialize MVPainter pipeline
        total_start_time = time.time()
        logger.info('Model loading started')

        pipeline_path = 'shaomq/MVPainter'
        t0 = time.time()
        self.pipeline = MVPainter_Pipeline.from_pretrained(
            pipeline_path,
            torch_dtype=torch.float16,
        )
        logger.info(f'MVPainter_Pipeline loaded in {time.time() - t0:.2f} seconds')

        t0 = time.time()
        controlnet = ControlNetModel_Union.from_unet(self.pipeline.unet).to(
            dtype=torch.float16, device=self.pipeline.device
        )
        self.pipeline.add_controlnet(controlnet, conditioning_scale=1.0)
        logger.info(f'ControlNetModel_Union loaded and added in {time.time() - t0:.2f} seconds')

        # Load custom unet checkpoint
        logger.info('Loading custom unet ...')
        t0 = time.time()
        ckpt = torch.load(unet_ckpt, map_location="cpu")["state_dict"]
        new_ckpt = {k[5:]: v for k, v in ckpt.items() if "unet" in k}
        del ckpt
        load_result = self.pipeline.unet.load_state_dict(new_ckpt)
        self.pipeline.unet = self.pipeline.unet.half()
        logger.info(f"Load result: {load_result}")
        logger.info(f'Custom unet loaded in {time.time() - t0:.2f} seconds')

        self.pipeline.vae.enable_slicing()
        self.pipeline.vae.enable_tiling()
        self.pipeline = self.pipeline.to(torch.device('cuda'))
        
        # Initialize background remover
        self.rembg_session = rembg.new_session()
        
        # Initialize bake pipeline
        self.bake_pipeline = None  # Will be initialized with specific parameters
        
        logger.info(f'Total model loading time: {time.time() - total_start_time:.2f} seconds')

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return self.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate(self, uid, params):
        """
        Main generation function that runs the full MVPainter pipeline
        """
        try:
            # Parse parameters
            if 'image' in params:
                reference_image = params["image"]
                reference_image = load_image_from_base64(reference_image)
            else:
                raise ValueError("No input image provided")

            if 'mesh' in params:
                mesh_data = base64.b64decode(params["mesh"])
                mesh_type = params.get('mesh_type', 'glb').lower()
                print("Mesh type:", mesh_type)
                # Save mesh to temporary file
                with tempfile.NamedTemporaryFile(suffix=f'.{mesh_type}', delete=False) as temp_file:
                    temp_file.write(mesh_data)
                    obj_path = temp_file.name
            else:
                raise ValueError("No input mesh provided")
            print("Wrote mesh to temporary file:", obj_path)

            # Get parameters with defaults
            geo_rotation = params.get('geo_rotation', -90)
            diffusion_steps = params.get('diffusion_steps', 75)
            use_pbr = params.get('use_pbr', False)
            no_rembg = params.get('no_rembg', False)
            
            object_uid = str(uid)
            
            # Step 1: Run multiview inference
            logger.info(f"Starting multiview inference for {object_uid}")
            mv_output_dir = self.run_multiview_inference(
                obj_path, reference_image, object_uid, geo_rotation, 
                diffusion_steps, no_rembg
            )
            
            # Step 2: Run painting pipeline
            logger.info(f"Starting painting pipeline for {object_uid}")
            final_glb_path = self.run_painting_pipeline(
                mv_output_dir, object_uid, geo_rotation, use_pbr
            )
            
            # Clean up temporary files
            if os.path.exists(obj_path):
                os.unlink(obj_path)
            
            torch.cuda.empty_cache()
            gc.collect()
            
            return final_glb_path, uid
            
        except Exception as e:
            logger.error(f"Error in generate: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def run_multiview_inference(self, obj_path, reference_image, object_uid, 
                               geo_rotation, diffusion_steps, no_rembg):
        """Run the multiview inference pipeline"""
        
        # Create output directory
        output_dir = os.path.join(SAVE_DIR, 'outputs', 'mvpainter', object_uid)
        os.makedirs(output_dir, exist_ok=True)
        
        # Run Blender rendering
        logger.info("Starting Blender rendering...")
        t0 = time.time()
        # Create render_temp subdirectory for this object_uid
        blender_obj_dir = os.path.join(RENDER_TEMP_DIR, object_uid)
        os.makedirs(blender_obj_dir, exist_ok=True)
        obj_extension = os.path.splitext(obj_path)[1].lower()
        blender_obj_path = os.path.join(blender_obj_dir, f"{object_uid}.{obj_extension}")
        shutil.copy(obj_path, blender_obj_path)
        # Call Blender script, forcing output_dir to RENDER_TEMP_DIR
        cmds = [
            self.blender_path, '--background', '-Y',
            '--python', 'scripts/blender_render_ortho.py', '--',
            '--object_path', blender_obj_path,
            '--output_dir', RENDER_TEMP_DIR,
            '--geo_rotation', str(geo_rotation),
        ]
        subprocess.run(cmds, check=True)
        logger.info(f'Blender rendering finished in {time.time() - t0:.2f} seconds')
        
        # Process depth images
        self.process_depth_images(object_uid)
        
        # Generate multiview images using diffusion
        self.generate_multiview_images(
            object_uid, reference_image, output_dir, 
            diffusion_steps, no_rembg
        )
        
        return output_dir

    def process_depth_images(self, object_uid):
        """Convert EXR depth images to PNG format"""
        depth_exr_dir = os.path.join(RENDER_TEMP_DIR, object_uid, "depth")
        depth_png_dir = os.path.join(RENDER_TEMP_DIR, object_uid, "depth_png")
        os.makedirs(depth_png_dir, exist_ok=True)
        
        # Handle actual .exr files produced by Blender (e.g., '0050001.exr')
        exr_files = [f for f in os.listdir(depth_exr_dir) if f.lower().endswith('.exr')]
        for fname in sorted(exr_files):
            # use first three characters as view index
            view_idx = fname[:3]
            depth_exr_path = os.path.join(depth_exr_dir, fname)
            depth_png_path = os.path.join(depth_png_dir, f"{view_idx}.png")
            try:
                depth_exr_to_png(depth_exr_path, depth_png_path)
            except Exception as e:
                logger.warning(f"Failed to convert {depth_exr_path}: {e}")

    def generate_multiview_images(self, object_uid, reference_image, output_dir, 
                                 diffusion_steps, no_rembg):
        """Generate multiview images using the diffusion pipeline"""
        
        filenames = ['000.png', '005.png', '001.png', '004.png', '002.png', '003.png']
        
        # Load and process images
        normal_path = os.path.join(RENDER_TEMP_DIR, object_uid, "normal")
        depth_png_dir = os.path.join(RENDER_TEMP_DIR, object_uid, "depth_png")
        
        # Load depth and normal images
        depth_images = [
            cv2.resize(cv2.imread(os.path.join(depth_png_dir, filename), cv2.IMREAD_UNCHANGED), (512, 512)) 
            for filename in filenames
        ]
        images = [
            cv2.resize(cv2.imread(os.path.join(normal_path, filename)), (512, 512)) 
            for filename in filenames
        ]
        
        # Combine images
        height, width, channel = images[0].shape
        combined_image = np.zeros((height * 3, width * 2, channel), dtype=np.uint8)
        combined_image_depth = np.ones((height * 3, width * 2), dtype=np.uint16) * 65535
        
        for index, image in enumerate(images):
            x_offset = (index % 2) * width
            y_offset = (index // 2) * height
            combined_image[y_offset:y_offset + height, x_offset:x_offset + width] = image
            combined_image_depth[y_offset:y_offset + height, x_offset:x_offset + width] = depth_images[index]
        
        # Normalize depth
        valid_mask = combined_image_depth < 65535
        combined_image_depth = combined_image_depth.astype(np.float32) / 65535.0
        combined_image_depth[valid_mask] = 1.0 / combined_image_depth[valid_mask]
        min_depth = np.min(combined_image_depth[valid_mask])
        max_depth = np.max(combined_image_depth[valid_mask])
        combined_image_depth[valid_mask] = (combined_image_depth[valid_mask] - min_depth) / (max_depth - min_depth)
        combined_image_depth = np.repeat(combined_image_depth[:, :, np.newaxis], 3, axis=2)
        
        # Save combined images
        cv2.imwrite(os.path.join(output_dir, 'combined_image_0.png'), combined_image)
        cv2.imwrite(os.path.join(output_dir, 'combined_image_0_depth.png'), 
                   (combined_image_depth * 255).astype(np.uint8))
        
        # Process reference image
        if not no_rembg:
            reference_image = remove_background(reference_image, self.rembg_session)
            reference_image = resize_foreground(reference_image, 0.85)
        
        # Load condition images
        depth = Image.open(os.path.join(output_dir, 'combined_image_0.png'))
        depth_2 = Image.open(os.path.join(output_dir, 'combined_image_0_depth.png'))
        
        # Run diffusion
        logger.info("Running diffusion pipeline...")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            latents = self.pipeline(
                reference_image,
                depth_image=depth,
                depth_image_2=depth_2,
                num_inference_steps=diffusion_steps,
                output_type="latent",
            )
        
        # Decode latents
        gc.collect()
        torch.cuda.empty_cache()
        
        scaled = unscale_latents(latents) / self.pipeline.vae.config.scaling_factor
        decoded = self.pipeline.vae.decode(scaled, return_dict=False)[0]
        decoded = unscale_image(unscale_image_2(decoded)).clamp(0, 1)
        
        # Convert to PIL and save
        view_img = Image.fromarray(
            (decoded[0] * 255 + 0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .astype("uint8")
        )
        view_img.save(os.path.join(output_dir, 'reference_0_6view.png'))
        
        # Save individual view images
        combine_image = cv2.imread(os.path.join(output_dir, 'reference_0_6view.png'))
        for index, image_name in enumerate(filenames):
            x_offset = (index % 2) * width
            y_offset = (index // 2) * height
            cv2.imwrite(os.path.join(output_dir, f"result_{index + 1}.png"),
                       combine_image[y_offset:y_offset + height + 1, x_offset:x_offset + width + 1])

    def run_painting_pipeline(self, mv_output_dir, object_uid, geo_rotation, use_pbr):
        """Run the painting pipeline to generate textured GLB"""
        
        # Create mesh reduction if needed
        render_obj_path = os.path.join(RENDER_TEMP_DIR, object_uid, "blender.obj")
        render_obj_low_path = os.path.join(RENDER_TEMP_DIR, object_uid, "blender_low.obj")
        
        if not os.path.exists(render_obj_low_path):
            reduce_mesh(render_obj_path, target_vertices=30000, output_path=render_obj_low_path)
        
        # Prepare output directory
        results_dir = os.path.join(SAVE_DIR, 'results', object_uid)
        os.makedirs(results_dir, exist_ok=True)
        
        if not use_pbr:
            # Simple texture baking
            glb_output_dir = os.path.join(results_dir, 'glbs')
            os.makedirs(glb_output_dir, exist_ok=True)
            
            # Initialize bake pipeline
            if self.bake_pipeline is None:
                self.bake_pipeline = BakePipeline(offset=geo_rotation + 90)
            
            # Run baking
            img_paths = [
                os.path.join(mv_output_dir, f'result_{i}.png') 
                for i in [1, 3, 5, 6, 4, 2]
            ]
            
            mesh = trimesh.load(render_obj_low_path)
            textured_mesh = self.bake_pipeline.call_bake(mesh, img_paths)
            
            final_path = os.path.join(glb_output_dir, f'{object_uid}.glb')
            textured_mesh.export(final_path)
            
            return final_path
        else:
            # PBR pipeline (simplified version)
            # For now, we'll just do basic texture baking
            # Full PBR implementation would require additional steps
            glb_output_dir = os.path.join(results_dir, 'glbs')
            os.makedirs(glb_output_dir, exist_ok=True)
            
            if self.bake_pipeline is None:
                self.bake_pipeline = BakePipeline(offset=geo_rotation + 90)
            
            img_paths = [
                os.path.join(mv_output_dir, f'result_{i}.png') 
                for i in [1, 3, 5, 6, 4, 2]
            ]
            
            mesh = trimesh.load(render_obj_low_path)
            textured_mesh = self.bake_pipeline.call_bake(mesh, img_paths)
            
            final_path = os.path.join(glb_output_dir, f'{object_uid}.glb')
            textured_mesh.export(final_path)
            
            return final_path


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate")
async def generate(request: Request):
    """Generate textured GLB from input mesh and reference image"""
    logger.info("Worker generating...")
    params = await request.json()
    uid = uuid.uuid4()
    try:
        file_path, uid = worker.generate(uid, params)
        return FileResponse(file_path)
    except ValueError as e:
        traceback.print_exc()
        logger.error("Caught ValueError:", e)
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=400)
    except Exception as e:
        logger.error("Caught Unknown Error", e)
        traceback.print_exc()
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=500)


@app.post("/send")
async def send_generate(request: Request):
    """Start generation in background and return job ID"""
    logger.info("Worker send...")
    params = await request.json()
    uid = uuid.uuid4()
    threading.Thread(target=worker.generate, args=(uid, params,)).start()
    ret = {"uid": str(uid)}
    return JSONResponse(ret, status_code=200)


@app.get("/status/{uid}")
async def status(uid: str):
    """Check generation status and retrieve result"""
    # Check for GLB file in multiple possible locations
    possible_paths = [
        os.path.join(SAVE_DIR, 'results', uid, 'glbs', f'{uid}.glb'),
        os.path.join(SAVE_DIR, f'{uid}.glb'),
    ]
    
    save_file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            save_file_path = path
            break
    
    if save_file_path is None:
        response = {'status': 'processing'}
        return JSONResponse(response, status_code=200)
    else:
        with open(save_file_path, 'rb') as f:
            base64_str = base64.b64encode(f.read()).decode()
        response = {'status': 'completed', 'model_base64': base64_str}
        return JSONResponse(response, status_code=200)


@app.get("/healthcheck")
async def healthcheck():
    """Health check endpoint"""
    return JSONResponse({"status": "ok"}, status_code=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--unet_ckpt", type=str, default="/home/racarr/v29_25000.ckpt")
    parser.add_argument("--blender_path", type=str, default="/home/racarr/blender/blender")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()
    logger.info(f"args: {args}")

    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)

    worker = MVPainterWorker(
        unet_ckpt=args.unet_ckpt,
        blender_path=args.blender_path,
        device=args.device,
        seed=args.seed,
        limit_model_concurrency=args.limit_model_concurrency
    )
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

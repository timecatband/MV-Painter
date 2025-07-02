# MVPainter API Server

A FastAPI server that provides texture generation for 3D models using the MVPainter pipeline. This API server is compatible with the texture generation portion of the Hunyuan API but is specifically designed for the MVPainter workflow.

## Features

- **Multiview Image Generation**: Generates multiple view images from a single reference image and 3D mesh
- **Texture Baking**: Bakes generated multiview images onto the 3D mesh to create a textured GLB file
- **Background Removal**: Automatically removes background from reference images (optional)
- **PBR Support**: Basic PBR material support (can be extended)
- **Asynchronous Processing**: Supports both synchronous and asynchronous generation modes
- **Compatible API**: Compatible with Hunyuan API format for easy integration

## Installation

1. Make sure you have all the MVPainter dependencies installed:
```bash
pip install -r requirements.txt
```

2. Ensure you have Blender installed and accessible. Update the `--blender_path` parameter accordingly.

3. Download the required model checkpoints:
   - MVPainter pipeline from HuggingFace: `shaomq/MVPainter`
   - Custom UNet checkpoint (specify path with `--unet_ckpt`)

## Usage

### Starting the Server

```bash
python api_server_mvpainter.py \
    --host 0.0.0.0 \
    --port 8082 \
    --unet_ckpt /path/to/your/unet_checkpoint.ckpt \
    --blender_path /path/to/blender/executable \
    --device cuda \
    --limit-model-concurrency 5
```

### API Endpoints

#### 1. Health Check
```
GET /healthcheck
```
Returns server status.

#### 2. Synchronous Generation
```
POST /generate
```

**Request Body:**
```json
{
    "mesh": "base64_encoded_glb_or_obj_file",
    "image": "base64_encoded_reference_image",
    "geo_rotation": -90,
    "diffusion_steps": 75,
    "use_pbr": false,
    "no_rembg": false
}
```

**Parameters:**
- `mesh` (required): Base64-encoded 3D mesh file (GLB or OBJ format)
- `image` (required): Base64-encoded reference image (PNG, JPG, etc.)
- `geo_rotation` (optional): Rotation applied to geometry (default: -90)
- `diffusion_steps` (optional): Number of diffusion denoising steps (default: 75)
- `use_pbr` (optional): Enable PBR material generation (default: false)
- `no_rembg` (optional): Skip background removal from reference image (default: false)

**Response:** 
Direct download of the generated textured GLB file.

#### 3. Asynchronous Generation
```
POST /send
```

Same request body as synchronous generation.

**Response:**
```json
{
    "uid": "unique_job_id"
}
```

#### 4. Check Generation Status
```
GET /status/{uid}
```

**Response (Processing):**
```json
{
    "status": "processing"
}
```

**Response (Completed):**
```json
{
    "status": "completed",
    "model_base64": "base64_encoded_textured_glb"
}
```

### Testing the API

Use the provided test script:

```bash
# Test synchronous generation
python test_mvpainter_api.py \
    --mesh_path /path/to/input.glb \
    --image_path /path/to/reference.png \
    --server_url http://localhost:8082

# Test asynchronous generation
python test_mvpainter_api.py \
    --mesh_path /path/to/input.glb \
    --image_path /path/to/reference.png \
    --server_url http://localhost:8082 \
    --test_async

# Test with custom parameters
python test_mvpainter_api.py \
    --mesh_path /path/to/input.glb \
    --image_path /path/to/reference.png \
    --geo_rotation -90 \
    --diffusion_steps 50 \
    --use_pbr
```

### Client Example

Here's a simple Python client example:

```python
import requests
import base64

def generate_texture(mesh_path, image_path, server_url="http://localhost:8082"):
    # Encode files
    with open(mesh_path, 'rb') as f:
        mesh_b64 = base64.b64encode(f.read()).decode()
    
    with open(image_path, 'rb') as f:
        image_b64 = base64.b64encode(f.read()).decode()
    
    # Make request
    payload = {
        "mesh": mesh_b64,
        "image": image_b64,
        "geo_rotation": -90,
        "diffusion_steps": 75,
        "use_pbr": False
    }
    
    response = requests.post(f"{server_url}/generate", json=payload)
    
    if response.status_code == 200:
        # Save result
        with open("textured_output.glb", "wb") as f:
            f.write(response.content)
        print("Success! Textured GLB saved.")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Usage
generate_texture("input.glb", "reference.png")
```

## Configuration

### Command Line Arguments

- `--host`: Server host (default: "0.0.0.0")
- `--port`: Server port (default: 8082)
- `--unet_ckpt`: Path to custom UNet checkpoint (required)
- `--blender_path`: Path to Blender executable (default: "/home/racarr/blender/blender")
- `--device`: Device for inference (default: "cuda")
- `--limit-model-concurrency`: Maximum concurrent model operations (default: 5)
- `--seed`: Random seed for generation (default: 12)

### Directory Structure

The server creates the following directories:
- `mvpainter_cache/`: Cache directory for temporary files and outputs
- `render_temp/`: Temporary directory for Blender rendering outputs

## Pipeline Overview

1. **Input Processing**: Receive and decode mesh and reference image
2. **Blender Rendering**: Generate orthogonal views and depth maps using Blender
3. **Depth Processing**: Convert EXR depth images to PNG format
4. **Multiview Generation**: Use MVPainter diffusion pipeline to generate textured views
5. **Mesh Reduction**: Reduce mesh complexity for efficient texture baking
6. **Texture Baking**: Bake multiview images onto mesh UV coordinates
7. **Output**: Return textured GLB file

## Compatibility

This API is designed to be compatible with the texture generation portion of the Hunyuan 3D API, allowing for easy migration or comparison between the two systems. The main differences are:

- MVPainter-specific parameters like `geo_rotation`
- Enhanced multiview generation pipeline
- Specialized texture baking process

## Performance Tips

- Use GPU acceleration for faster inference
- Adjust `diffusion_steps` based on quality vs. speed requirements
- Consider reducing mesh complexity for faster processing
- Use background removal (`no_rembg=false`) for better results with complex backgrounds

## Troubleshooting

### Common Issues

1. **Blender not found**: Make sure the `--blender_path` points to a valid Blender executable
2. **CUDA out of memory**: Reduce `limit-model-concurrency` or use a smaller model
3. **Slow processing**: Reduce `diffusion_steps` or use a faster GPU
4. **Import errors**: Ensure all dependencies are installed and paths are correct

### Logs

Server logs are saved to `mvpainter_cache/mvpainter_controller.log` for debugging.

## License

This project follows the same licensing terms as the original MVPainter project.

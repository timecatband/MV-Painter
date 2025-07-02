#!/usr/bin/env python3
"""
Test script for MVPainter API Server
"""
import requests
import base64
import json
import time
import argparse


def encode_file_to_base64(file_path):
    """Encode file to base64 string"""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def test_mvpainter_api(server_url, mesh_path, image_path, 
                      geo_rotation=-90, diffusion_steps=75, use_pbr=False):
    """Test the MVPainter API"""
    
    # Prepare the payload
    mesh_base64 = encode_file_to_base64(mesh_path)
    image_base64 = encode_file_to_base64(image_path)
    mesh_type = mesh_path.split('.')[-1].lower()
    
    payload = {
        "mesh": mesh_base64,
        "image": image_base64,
        "mesh_type": mesh_type,
        "geo_rotation": geo_rotation,
        "diffusion_steps": diffusion_steps,
        "use_pbr": use_pbr,
        "no_rembg": False
    }
    
    print(f"Testing MVPainter API at {server_url}")
    print(f"Mesh: {mesh_path}")
    print(f"Image: {image_path}")
    print(f"Parameters: geo_rotation={geo_rotation}, diffusion_steps={diffusion_steps}, use_pbr={use_pbr}")
    
    # Test synchronous generation
    print("\n=== Testing synchronous generation ===")
    try:
        response = requests.post(f"{server_url}/generate", 
                               json=payload, 
                               timeout=300)  # 5 minute timeout
        
        if response.status_code == 200:
            # Save the returned GLB file
            output_path = f"test_output_{int(time.time())}.glb"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Success! Generated GLB saved to: {output_path}")
            return output_path
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_async_api(server_url, mesh_path, image_path, 
                  geo_rotation=-90, diffusion_steps=75, use_pbr=False):
    """Test the asynchronous API"""
    
    # Prepare the payload
    mesh_base64 = encode_file_to_base64(mesh_path)
    image_base64 = encode_file_to_base64(image_path)
    mesh_type = mesh_path.split('.')[-1].lower()
    
    payload = {
        "mesh": mesh_base64,
        "image": image_base64,
        "mesh_type": mesh_type,
        "geo_rotation": geo_rotation,
        "diffusion_steps": diffusion_steps,
        "use_pbr": use_pbr,
        "no_rembg": False
    }
    
    print("\n=== Testing asynchronous generation ===")
    
    # Start generation
    try:
        response = requests.post(f"{server_url}/send", json=payload)
        if response.status_code != 200:
            print(f"❌ Failed to start generation: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
        data = response.json()
        uid = data['uid']
        print(f"✅ Generation started with UID: {uid}")
        
        # Poll for completion
        max_attempts = 60  # 5 minutes with 5-second intervals
        for attempt in range(max_attempts):
            time.sleep(5)
            status_response = requests.get(f"{server_url}/status/{uid}")
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                if status_data['status'] == 'completed':
                    # Save the result
                    model_data = base64.b64decode(status_data['model_base64'])
                    output_path = f"test_async_output_{int(time.time())}.glb"
                    with open(output_path, 'wb') as f:
                        f.write(model_data)
                    print(f"✅ Async generation completed! GLB saved to: {output_path}")
                    return output_path
                elif status_data['status'] == 'processing':
                    print(f"🔄 Still processing... (attempt {attempt + 1}/{max_attempts})")
                else:
                    print(f"❌ Unknown status: {status_data['status']}")
                    return None
            else:
                print(f"❌ Status check failed: {status_response.status_code}")
                return None
        
        print("❌ Generation timed out")
        return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_health_check(server_url):
    """Test the health check endpoint"""
    print("\n=== Testing health check ===")
    try:
        response = requests.get(f"{server_url}/healthcheck")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MVPainter API Server")
    parser.add_argument("--server_url", type=str, default="http://localhost:8082",
                       help="Server URL")
    parser.add_argument("--mesh_path", type=str, required=True,
                       help="Path to input mesh file (GLB/OBJ)")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to reference image")
    parser.add_argument("--geo_rotation", type=int, default=-90,
                       help="Geometry rotation")
    parser.add_argument("--diffusion_steps", type=int, default=75,
                       help="Diffusion steps")
    parser.add_argument("--use_pbr", action="store_true",
                       help="Use PBR rendering")
    parser.add_argument("--test_async", action="store_true",
                       help="Test async API instead of sync")
    
    args = parser.parse_args()
    
    # Test health check first
    if not test_health_check(args.server_url):
        print("❌ Server health check failed. Is the server running?")
        exit(1)
    
    # Test the API
    if args.test_async:
        result = test_async_api(
            args.server_url, args.mesh_path, args.image_path,
            args.geo_rotation, args.diffusion_steps, args.use_pbr
        )
    else:
        result = test_mvpainter_api(
            args.server_url, args.mesh_path, args.image_path,
            args.geo_rotation, args.diffusion_steps, args.use_pbr
        )
    
    if result:
        print(f"\n🎉 Test completed successfully! Output: {result}")
    else:
        print("\n💥 Test failed!")
        exit(1)

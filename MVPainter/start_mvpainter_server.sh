#!/bin/bash

# MVPainter API Server Startup Script

# Default values
HOST="0.0.0.0"
PORT="8082"
UNET_CKPT="/home/racarr/v29_25000.ckpt"
BLENDER_PATH="/home/racarr/blender/blender"
DEVICE="cuda"
CONCURRENCY="5"
SEED="12"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --unet_ckpt)
            UNET_CKPT="$2"
            shift 2
            ;;
        --blender_path)
            BLENDER_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help|-h)
            echo "MVPainter API Server Startup Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host HOST                Server host (default: $HOST)"
            echo "  --port PORT                Server port (default: $PORT)"
            echo "  --unet_ckpt PATH           Path to UNet checkpoint (default: $UNET_CKPT)"
            echo "  --blender_path PATH        Path to Blender executable (default: $BLENDER_PATH)"
            echo "  --device DEVICE            Device for inference (default: $DEVICE)"
            echo "  --concurrency NUM          Max concurrent operations (default: $CONCURRENCY)"
            echo "  --seed SEED                Random seed (default: $SEED)"
            echo "  --help, -h                 Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --port 8083 --device cpu"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if required files exist
if [ ! -f "$UNET_CKPT" ]; then
    echo "Error: UNet checkpoint not found at $UNET_CKPT"
    echo "Please update the --unet_ckpt parameter or create a symlink"
    exit 1
fi

if [ ! -f "$BLENDER_PATH" ] && [ ! -x "$(command -v $BLENDER_PATH)" ]; then
    echo "Warning: Blender not found at $BLENDER_PATH"
    echo "Please update the --blender_path parameter if you encounter issues"
fi

# Create necessary directories
mkdir -p mvpainter_cache
mkdir -p render_temp

echo "Starting MVPainter API Server..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "UNet checkpoint: $UNET_CKPT"
echo "Blender path: $BLENDER_PATH"
echo "Device: $DEVICE"
echo "Max concurrency: $CONCURRENCY"
echo "Random seed: $SEED"
echo ""

# Start the server
python api_server_mvpainter.py \
    --host "$HOST" \
    --port "$PORT" \
    --unet_ckpt "$UNET_CKPT" \
    --blender_path "$BLENDER_PATH" \
    --device "$DEVICE" \
    --limit-model-concurrency "$CONCURRENCY" \
    --seed "$SEED"

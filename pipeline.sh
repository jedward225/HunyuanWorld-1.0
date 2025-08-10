#!/bin/bash

# Complete pipeline from text/image to 3D world rendering
# Usage: ./pipeline.sh --mode [text|image|existing] --input "input_content" --scene scene_name [--gpu gpu_id]

set -e  # Exit on any error

# Default values
MODE="text"
INPUT=""
SCENE="demo"
GPU_ID=7
RESIZE_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --input)
            INPUT="$2"
            shift 2
            ;;
        --scene)
            SCENE="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --resize-only)
            RESIZE_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 --mode [text|image|existing] --input \"content\" --scene scene_name [options]"
            echo ""
            echo "Modes:"
            echo "  text     Generate panorama from text prompt"
            echo "  image    Generate panorama from input image"
            echo "  existing Use existing panorama image (provide path)"
            echo ""
            echo "Options:"
            echo "  --input TEXT/PATH    Text prompt, image path, or panorama path"
            echo "  --scene NAME         Scene name for output (default: demo)"
            echo "  --gpu ID             GPU ID to use (default: 7)"
            echo "  --resize-only        Only resize existing panorama to (960,1920)"
            echo "  --help               Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --mode text --input \"a beautiful street scene\" --scene street"
            echo "  $0 --mode existing --input \"/path/to/panorama.png\" --scene custom --resize-only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate inputs
if [ -z "$INPUT" ]; then
    echo "Error: --input is required"
    exit 1
fi

# Set paths
OUTPUT_DIR="test_results/${SCENE}"
PANO_PATH="${OUTPUT_DIR}/panorama.png"
DEPTH_PATH="${OUTPUT_DIR}/depth"
PC_PATH="${OUTPUT_DIR}/pointcloud"

echo "========================================="
echo "HunyuanWorld-FlexWorld Complete Pipeline"
echo "========================================="
echo "Mode: $MODE"
echo "Input: $INPUT"
echo "Scene: $SCENE"
echo "GPU: $GPU_ID"
echo "Output Directory: $OUTPUT_DIR"
echo "Resize Only: $RESIZE_ONLY"
echo "========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Generate or process panorama
if [ "$MODE" = "text" ]; then
    echo "🎨 Step 1: Generating panorama from text prompt..."
    python3 demo_panogen_local.py \
        --prompt "$INPUT" \
        --output_path "$OUTPUT_DIR" \
        --seed 42 \
        --use_local
    
elif [ "$MODE" = "image" ]; then
    echo "🎨 Step 1: Generating panorama from input image..."
    python3 demo_panogen_local.py \
        --image_path "$INPUT" \
        --output_path "$OUTPUT_DIR" \
        --seed 42 \
        --use_local
    
elif [ "$MODE" = "existing" ]; then
    echo "📁 Step 1: Processing existing panorama..."
    if [ ! -f "$INPUT" ]; then
        echo "Error: Input file not found: $INPUT"
        exit 1
    fi
    
    if [ "$RESIZE_ONLY" = true ]; then
        echo "🔄 Resizing panorama to (960, 1920)..."
        python3 -c "
import cv2
import sys
img = cv2.imread('$INPUT')
if img is None:
    print('Error: Could not load image $INPUT')
    sys.exit(1)
resized = cv2.resize(img, (1920, 960), interpolation=cv2.INTER_LINEAR)
cv2.imwrite('$PANO_PATH', resized)
print('Resized panorama saved to: $PANO_PATH')
"
    else
        cp "$INPUT" "$PANO_PATH"
        echo "Copied panorama to: $PANO_PATH"
    fi
else
    echo "Error: Invalid mode. Use text, image, or existing"
    exit 1
fi

# Check if panorama was created
if [ ! -f "$PANO_PATH" ]; then
    echo "Error: Panorama generation failed"
    exit 1
fi

echo "✅ Step 1 completed: $PANO_PATH"

# Step 2: Generate depth map
echo "🧠 Step 2: Generating depth map..."
CUDA_VISIBLE_DEVICES=$GPU_ID python3 generate_pano_depth.py \
    --image_path "$PANO_PATH" \
    --output_path "$DEPTH_PATH" \
    --verbose

DEPTH_NPY="${DEPTH_PATH}/panorama_depth.npy"
if [ ! -f "$DEPTH_NPY" ]; then
    echo "Error: Depth generation failed"
    exit 1
fi

echo "✅ Step 2 completed: $DEPTH_NPY"

# Step 3: Generate point cloud
echo "☁️  Step 3: Generating point cloud..."
python3 generate_pano_pointcloud.py \
    --rgb_path "$PANO_PATH" \
    --depth_path "$DEPTH_NPY" \
    --output_path "$PC_PATH" \
    --verbose

PLY_PATH="${PC_PATH}/panorama_pointcloud.ply"
if [ ! -f "$PLY_PATH" ]; then
    echo "Error: Point cloud generation failed"
    exit 1
fi

echo "✅ Step 3 completed: $PLY_PATH"

# Step 4: Render 3D world
echo "🎬 Step 4: Rendering 3D world and generating panorama..."

# Update ljj.py with the correct PLY path (use absolute path)
ABS_PLY_PATH="$(realpath $PLY_PATH)"
sed -i "s|pcd=PcdMgr(ply_file_path=f'.*')|pcd=PcdMgr(ply_file_path=f'$ABS_PLY_PATH')|g" FlexWorld/ljj.py
echo "Updated ljj.py to use: $ABS_PLY_PATH"

# Run FlexWorld rendering
cd FlexWorld
python ljj.py
cd ..

echo "✅ Step 4 completed"

# Final summary
echo ""
echo "🎉 Pipeline completed successfully!"
echo "========================================="
echo "Generated files:"
echo "  📸 Panorama: $PANO_PATH"
echo "  🧠 Depth: $DEPTH_NPY"
echo "  ☁️  Point cloud: $PLY_PATH"
echo "  🎬 Orbit video: FlexWorld/testOutput/test_video.mp4"
echo "  🌍 Final panorama: FlexWorld/testOutput/panorama_output/pano.png"
echo "========================================="
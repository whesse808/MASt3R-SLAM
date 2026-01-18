#!/bin/bash
# Run MASt3R-SLAM on an Insta360 video
#
# Usage: ./run.sh /path/to/video.insv [additional args]
#
# Examples:
#   ./run.sh /data/recording.insv
#   ./run.sh /data/recording.insv --no-viz
#   ./run.sh "insta360:/data/recording.insv?start=10&duration=30"

set -e

if [[ -z "$1" ]]; then
    echo "Usage: $0 <video_path_or_dataset_string> [additional args]"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/video.insv"
    echo "  $0 /path/to/video.insv --no-viz"
    echo "  $0 \"insta360:/path/to/video.insv?start=10&duration=30&fps=5\""
    echo ""
    echo "Parameters (in dataset string):"
    echo "  start    - Start time in seconds (default: 0)"
    echo "  duration - Duration in seconds (default: 60)"
    echo "  fov      - Output FOV in degrees (default: 90)"
    echo "  fps      - Perspectives per second (default: 10)"
    exit 1
fi

VIDEO_INPUT="$1"
shift

# Determine if input is a file path or a dataset string
if [[ "$VIDEO_INPUT" == insta360:* ]]; then
    DATASET="$VIDEO_INPUT"
elif [[ -f "$VIDEO_INPUT" ]]; then
    DATASET="$VIDEO_INPUT"
else
    echo "Error: File not found: $VIDEO_INPUT"
    exit 1
fi

# Get directory containing the video for mounting
if [[ "$VIDEO_INPUT" == insta360:* ]]; then
    # Extract path from dataset string
    VIDEO_PATH=$(echo "$VIDEO_INPUT" | sed 's/^insta360://' | sed 's/?.*//')
    VIDEO_DIR=$(dirname "$VIDEO_PATH")
else
    VIDEO_PATH="$VIDEO_INPUT"
    VIDEO_DIR=$(dirname "$VIDEO_PATH")
fi

VIDEO_NAME=$(basename "$VIDEO_PATH")
VIDEO_DIR=$(cd "$VIDEO_DIR" && pwd)

echo "Processing: $VIDEO_NAME"
echo "Video directory: $VIDEO_DIR"

# Update dataset path for container
if [[ "$DATASET" == insta360:* ]]; then
    # Replace local path with container path
    DATASET=$(echo "$DATASET" | sed "s|$VIDEO_DIR|/data|")
else
    DATASET="/data/$VIDEO_NAME"
fi

echo "Container dataset: $DATASET"
echo ""

docker run --rm -it \
    --gpus all \
    -v "$VIDEO_DIR":/data:ro \
    -v "$(pwd)/output":/workspace/MASt3R-SLAM/logs \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    mast3r-slam-insta360:latest \
    python main.py \
    --dataset "$DATASET" \
    --config config/base.yaml \
    "$@"

echo ""
echo "Results saved to: $(pwd)/output/"

#!/bin/bash
# Build the MASt3R-SLAM Insta360 Docker image
#
# Usage: ./build.sh [--no-cache]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

BUILD_ARGS=""
if [[ "$1" == "--no-cache" ]]; then
    BUILD_ARGS="--no-cache"
fi

echo "Building MASt3R-SLAM Insta360 Docker image..."
echo "Project root: $PROJECT_ROOT"

docker build $BUILD_ARGS \
    -t mast3r-slam-insta360:latest \
    -f docker/insta360/Dockerfile \
    .

echo ""
echo "Build complete!"
echo ""
echo "To run on an Insta360 video:"
echo "  docker run --gpus all -v /path/to/videos:/data mast3r-slam-insta360 \\"
echo "    python main.py --dataset /data/video.insv --config config/base.yaml --no-viz"

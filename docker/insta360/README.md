# MASt3R-SLAM with Insta360 X5 Support

This Docker setup enables running MASt3R-SLAM directly on Insta360 X5 dual-fisheye videos without intermediate disk storage. The pipeline streams 512x512 perspective frames from the fisheye video directly into the SLAM system.

## System Requirements

- NVIDIA GPU with compute capability 8.0+ (RTX 30xx, 40xx, 50xx, A100, etc.)
- NVIDIA Driver 535+ with CUDA 12.x
- Docker with NVIDIA Container Toolkit
- ~16GB GPU memory recommended (works with 8GB in headless mode)

## Quick Start

### Option 1: Build locally

```bash
cd docker/insta360
./build.sh

# Run on your video
./run.sh /path/to/your_video.insv --no-viz
```

### Option 2: Docker Compose

```bash
cd docker/insta360
mkdir -p data output

# Copy your video
cp /path/to/video.insv data/

# Run
docker-compose run mast3r-slam python main.py \
    --dataset /data/video.insv \
    --config config/base.yaml \
    --no-viz
```

## Usage

### Basic Usage

```bash
# Process first 60 seconds at 10 fps
docker run --gpus all -v /path/to/videos:/data mast3r-slam-insta360 \
    python main.py --dataset /data/video.insv --config config/base.yaml --no-viz
```

### Advanced Parameters

Use the `insta360:` URL scheme for fine-grained control:

```bash
# Start at 10 seconds, process 30 seconds, 5 perspectives/second
docker run --gpus all -v /path/to/videos:/data mast3r-slam-insta360 \
    python main.py \
    --dataset "insta360:/data/video.insv?start=10&duration=30&fps=5" \
    --config config/base.yaml \
    --no-viz
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `start` | 0 | Start time in seconds |
| `duration` | 60 | Duration to process in seconds |
| `fov` | 90 | Output perspective FOV in degrees |
| `fps` | 10 | Perspectives per second |

### With Visualization (requires X11 forwarding)

```bash
docker run --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /path/to/videos:/data \
    mast3r-slam-insta360 \
    python main.py --dataset /data/video.insv --config config/base.yaml
```

## Output

Results are saved to `/workspace/MASt3R-SLAM/logs/` inside the container:

- `{video_name}.ply` - 3D point cloud reconstruction
- `{video_name}.txt` - Camera trajectory (keyframe poses)
- `keyframes/{video_name}/` - Extracted keyframe images

Mount the logs directory to access results:

```bash
docker run --gpus all \
    -v /path/to/videos:/data \
    -v /path/to/output:/workspace/MASt3R-SLAM/logs \
    mast3r-slam-insta360 \
    python main.py --dataset /data/video.insv --config config/base.yaml --no-viz
```

## RunPod Deployment

### Using RunPod Template

1. Create a new Pod with:
   - GPU: RTX 4090, A100, or similar (16GB+ VRAM)
   - Container Image: Build and push to your registry, or build on-pod
   - Volume: Mount persistent storage for videos and results

2. Build on-pod:
```bash
git clone --recursive https://github.com/whesse808/MASt3R-SLAM.git
cd MASt3R-SLAM/docker/insta360
./build.sh
```

3. Upload video and run:
```bash
./run.sh /path/to/video.insv --no-viz
```

### Pre-built Image (if available)

```bash
docker pull your-registry/mast3r-slam-insta360:latest
```

## Pipeline Architecture

```
Insta360 X5 (.insv)
    │
    ├─► Front fisheye (3840x3840)  ─┐
    │                                │ NVDEC decode to 1024x1024
    └─► Back fisheye (3840x3840)   ─┘
                │
                ▼
    GPU Perspective Extraction (PyTorch)
    - Equidistant fisheye projection
    - Forward-facing view (yaw=0°)
    - Output: 512x512 @ 90° FOV
                │
                ▼
    MASt3R-SLAM Processing
    - Feature extraction (ViT-Large)
    - Frame tracking
    - Global optimization
    - Loop closure
                │
                ▼
    Output: 3D Point Cloud + Trajectory
```

## Performance

Tested on RTX 5070 Ti:
- ~2.7 FPS end-to-end (SLAM + perspective extraction)
- 60 seconds of video → 600 frames → ~3.7 minutes processing
- Typical output: 2-3 million point cloud

## Troubleshooting

### Out of GPU Memory

Reduce duration or fps:
```bash
--dataset "insta360:/data/video.insv?duration=30&fps=5"
```

### Tracking Failures / Stuck in RELOC

The forward-facing perspective works best for scenes with:
- Moderate camera motion
- Sufficient visual texture
- Overlapping features between frames

For challenging scenes, try:
- Lower fps (more time between frames = larger motion)
- Different start time in the video

### OpenGL Errors

For headless operation, always use `--no-viz`.

### FFmpeg / Video Decoding Issues

Ensure the video file is a valid Insta360 .insv file with two video streams.

## Files Modified for Insta360 Support

- `mast3r_slam/insta360_loader.py` - New file: Dataset class for streaming
- `mast3r_slam/dataloader.py` - Added Insta360 dataset detection
- `thirdparty/mast3r/mast3r/retrieval/processor.py` - PyTorch 2.5+ compatibility fix

## License

MASt3R-SLAM: See original repository license
Insta360 integration: Same license as MASt3R-SLAM

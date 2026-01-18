# MASt3R-SLAM + 3D Gaussian Splatting with Insta360 X5 Support
# nfoldROI-4D360 Pipeline
# Optimized for RunPod with NVIDIA GPUs (sm_80+)
#
# Build: docker build -t nfoldroi-4d360 .
# Run:   docker run --gpus all -v /path/to/videos:/data -v /path/to/output:/output nfoldroi-4d360

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    libglfw3 \
    libglfw3-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Create workspace
WORKDIR /workspace

# Install PyTorch with CUDA 12.4 support
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install lietorch (requires PyTorch first)
RUN pip install lietorch

# Install core dependencies
RUN pip install \
    numpy==1.26.4 \
    opencv-python==4.10.0.84 \
    open3d==0.18.0 \
    einops==0.8.0 \
    scipy \
    matplotlib \
    tqdm \
    pyyaml \
    natsort \
    roma \
    gradio \
    huggingface_hub \
    pyglet==1.5.29 \
    trimesh \
    pillow \
    plyfile \
    pyrealsense2 || true

# Install PyNvVideoCodec for GPU video decoding
RUN pip install PyNvVideoCodec

# ============================================================
# MASt3R-SLAM
# ============================================================

# Clone MASt3R-SLAM with Insta360 support
RUN git clone --recursive https://github.com/whesse808/MASt3R-SLAM.git /workspace/MASt3R-SLAM

WORKDIR /workspace/MASt3R-SLAM

# Install thirdparty dependencies
RUN cd thirdparty/mast3r && pip install -e .
RUN cd thirdparty/mast3r/dust3r && pip install -e .
RUN cd thirdparty/mast3r/dust3r/croco/models/curope && pip install -e .
RUN cd thirdparty/mast3r/asmk && pip install -e .
RUN cd thirdparty/in3d && pip install -e .

# Build MASt3R-SLAM CUDA backends
RUN pip install -e .

# Create checkpoints directory and download models
RUN mkdir -p checkpoints && \
    cd checkpoints && \
    wget -q --show-progress https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth && \
    wget -q --show-progress https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth && \
    wget -q --show-progress https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl

# ============================================================
# 3D Gaussian Splatting
# ============================================================

WORKDIR /workspace

# Clone 3D Gaussian Splatting
RUN git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git /workspace/gaussian-splatting

WORKDIR /workspace/gaussian-splatting

# Fix missing cstdint include for CUDA 12.4 compatibility
RUN sed -i '1i #include <cstdint>' submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h

# Install 3DGS submodules (diff-gaussian-rasterization and simple-knn)
RUN pip install --no-build-isolation submodules/diff-gaussian-rasterization
RUN pip install --no-build-isolation submodules/simple-knn

# ============================================================
# Utility Scripts
# ============================================================

WORKDIR /workspace

# Create scripts directory
RUN mkdir -p /workspace/scripts

# Convert MASt3R-SLAM output to COLMAP format
COPY scripts/convert_mast3r_to_colmap.py /workspace/scripts/

# Visualization scripts
COPY scripts/create_4way_comparison.py /workspace/scripts/
COPY scripts/create_3way_comparison.py /workspace/scripts/
COPY scripts/cuda_ply_flythrough.py /workspace/scripts/
COPY scripts/merge_ply_and_visualize.py /workspace/scripts/
COPY scripts/subsample_ply.py /workspace/scripts/
COPY scripts/video_viewer.py /workspace/scripts/

# Create data directories
RUN mkdir -p /data /output

# Set environment variables
ENV PYTHONPATH=/workspace/MASt3R-SLAM:/workspace/gaussian-splatting:/workspace/scripts:$PYTHONPATH

# Default command shows pipeline info
CMD ["bash", "-c", "echo '========================================' && \
    echo 'nfoldROI-4D360 Pipeline' && \
    echo '========================================' && \
    echo '' && \
    echo 'Components:' && \
    echo '  - MASt3R-SLAM: Monocular SLAM with dense point clouds' && \
    echo '  - 3D Gaussian Splatting: Novel view synthesis' && \
    echo '  - Insta360 X5: GPU-accelerated fisheye processing' && \
    echo '' && \
    echo 'Commands:' && \
    echo '  MASt3R-SLAM:' && \
    echo '    python /workspace/MASt3R-SLAM/main.py --help' && \
    echo '' && \
    echo '  3DGS Train:' && \
    echo '    python /workspace/gaussian-splatting/train.py -s <colmap_data> -m <output>' && \
    echo '' && \
    echo '  3DGS Render:' && \
    echo '    python /workspace/gaussian-splatting/render.py -m <model_path>' && \
    echo '' && \
    echo '  Convert SLAM to COLMAP:' && \
    echo '    python /workspace/scripts/convert_mast3r_to_colmap.py --help' && \
    echo '' && \
    echo 'Volumes:' && \
    echo '  /data   - Mount input videos here' && \
    echo '  /output - Mount output directory here' && \
    echo ''"]

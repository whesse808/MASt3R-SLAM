#!/bin/bash
# RunPod Quick Setup Script
# Run this on a fresh RunPod instance with NVIDIA GPU
#
# Usage: curl -sL https://raw.githubusercontent.com/whesse808/MASt3R-SLAM/main/docker/insta360/runpod_setup.sh | bash

set -e

echo "=========================================="
echo "MASt3R-SLAM + Insta360 RunPod Setup"
echo "=========================================="
echo ""

# Check for NVIDIA GPU
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: NVIDIA GPU not detected. Please use a GPU-enabled RunPod instance."
    exit 1
fi

echo "Detected GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Install system dependencies
echo "[1/7] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq git ffmpeg libgl1-mesa-glx libglib2.0-0 > /dev/null

# Install Miniconda if not present
if ! command -v conda &> /dev/null; then
    echo "[2/7] Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /root/miniconda3
    rm /tmp/miniconda.sh
    eval "$(/root/miniconda3/bin/conda shell.bash hook)"
    conda init bash
else
    echo "[2/7] Miniconda already installed"
    eval "$(conda shell.bash hook)"
fi

# Create conda environment
echo "[3/7] Creating conda environment..."
conda create -n mast3r python=3.11 -y -q
conda activate mast3r

# Install PyTorch
echo "[4/7] Installing PyTorch with CUDA..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install lietorch
echo "[5/7] Installing lietorch..."
pip install -q lietorch

# Clone and setup MASt3R-SLAM
echo "[6/7] Cloning MASt3R-SLAM..."
cd /workspace
if [ ! -d "MASt3R-SLAM" ]; then
    git clone --recursive https://github.com/whesse808/MASt3R-SLAM.git
fi
cd MASt3R-SLAM

# Install dependencies
pip install -q numpy==1.26.4 opencv-python open3d einops scipy matplotlib tqdm pyyaml natsort roma gradio huggingface_hub trimesh PyNvVideoCodec

# Install thirdparty modules
cd thirdparty/mast3r && pip install -q -e . && cd ../..
cd thirdparty/mast3r/dust3r && pip install -q -e . && cd ../../..
cd thirdparty/mast3r/dust3r/croco/models/curope && pip install -q -e . && cd ../../../../../..
cd thirdparty/mast3r/asmk && pip install -q -e . && cd ../../..
cd thirdparty/in3d && pip install -q -e . && cd ../..

# Build CUDA backends
echo "[7/7] Building CUDA backends..."
pip install -e .

# Download model checkpoints
echo "Downloading model checkpoints (this may take a while)..."
mkdir -p checkpoints
cd checkpoints
if [ ! -f "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" ]; then
    wget -q --show-progress https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
fi
if [ ! -f "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth" ]; then
    wget -q --show-progress https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
fi
if [ ! -f "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl" ]; then
    wget -q --show-progress https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl
fi
cd ..

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To run MASt3R-SLAM on an Insta360 video:"
echo ""
echo "  conda activate mast3r"
echo "  cd /workspace/MASt3R-SLAM"
echo "  python main.py --dataset /path/to/video.insv --config config/base.yaml --no-viz"
echo ""
echo "With parameters:"
echo "  python main.py --dataset \"insta360:/path/to/video.insv?start=10&duration=30&fps=5\" \\"
echo "                 --config config/base.yaml --no-viz"
echo ""
echo "Results will be saved to: /workspace/MASt3R-SLAM/logs/"

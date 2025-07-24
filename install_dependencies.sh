#!/bin/bash
echo "Installing dependencies for Audix Preprocessing..."

echo ""
echo "Step 1: Installing PyTorch with CUDA support..."
pip install torch==2.1.0+cu121 torchaudio==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Step 2: Installing basic dependencies..."
pip install numpy==1.24.0 scipy librosa einops julius lameenc openunmix hydra-core hydra-colorlog flashy retrying noisereduce sounddevice omegaconf submitit cloudpickle treetable

echo ""
echo "Step 3: Installing Demucs (without dependencies)..."
pip install git+https://github.com/facebookresearch/demucs.git --no-deps

echo ""
echo "Step 4: Installing Dora (without dependencies)..."
pip install git+https://github.com/facebookresearch/dora.git --no-deps

echo ""
echo "Installation completed!"

#!/bin/bash

# First install ubuntu packages via apt-get
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    python3-dev \
    swig

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment and install dependencies using uv
uv venv .venv --python 3.12

# Install main dependencies from pyproject.toml
uv pip install -e .

# Install PyTorch with CUDA 11.8
# Change this depending on your CUDA version: https://pytorch.org/get-started/locally/
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install git dependencies
uv pip install "git+https://github.com/pybox2d/pybox2d#egg=Box2D"
uv pip install "git+https://github.com/nacansino/gym-bandits"
uv pip install "git+https://github.com/nacansino/gym-walk"
uv pip install "git+https://github.com/nacansino/gym-aima"

echo ""
echo "Virtual environment created successfully in .venv"
echo "To activate it, run: source .venv/bin/activate"

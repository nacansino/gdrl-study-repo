#!/bin/bash

# First install ubuntu packages via apt-get
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    python3-dev python3-venv python3-pip \
    swig

# Create virtual environment in the current directory
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install requirements
pip install --upgrade pip setuptools wheel

# ---- Jupyter + tooling ----
# Modern JupyterLab no longer requires node for most extensions, but we keep node/npm installed
# since your original image used lab extensions and other JS tools.
pip install \
    jupyterlab notebook jupyterhub \
    tensorboard \
    tqdm numpy scipy pandas scikit-learn pyyaml tabulate \
    pyglet \
    tornado==6.* \
    matplotlib \
    ipywidgets==7.7.1

# ---- RL / Gym stack (closest equivalent to your original) ----
pip install \
    gymnasium \
    pygame \
    pybullet \
    asciinema \
    && pip install "gymnasium[atari]" \
    && pip install "git+https://github.com/pybox2d/pybox2d#egg=Box2D" \
    && pip install "git+https://github.com/nacansino/gym-bandits" \
    && pip install "git+https://github.com/nacansino/gym-walk" \
    && pip install "git+https://github.com/nacansino/gym-aima"

# ---- PyTorch (CUDA 12.9 / cu129) ----
# PyTorch publishes a cu129 index; if stable wheels ever lag, use nightly as noted below.
pip install --index-url https://download.pytorch.org/whl/cu129 \
    torch torchvision

echo "Virtual environment created successfully in .venv"
echo "To activate it, run: source .venv/bin/activate"

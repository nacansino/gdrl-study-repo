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

# Sync all dependencies from pyproject.toml
uv sync

echo ""
echo "Virtual environment created successfully in .venv"
echo "Run commands with: uv run <command>"
#!/bin/bash
set -e

# Get absolute path to project root (one level up from installers)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if Python path is provided as argument
if [ -z "$1" ]; then
    PYTHON_CMD="python"
    echo "Using system Python..."
else
    PYTHON_CMD="$1"
    echo "Using Python from: $1"
fi

# Check if uv is installed
echo "Checking for uv package manager..."
if ! "$PYTHON_CMD" -m uv --version >/dev/null 2>&1; then
    echo "uv not found. Installing uv..."
    "$PYTHON_CMD" -m pip install uv
    if [ $? -ne 0 ]; then
        echo "Failed to install uv. Please install it manually: pip install uv"
        exit 1
    fi
    echo "uv installed successfully!"
else
    echo "uv is already installed."
fi

echo
echo "Installing dependencies for CUDA..."
echo "Project root: $PROJECT_ROOT"
echo

echo "Installing PyTorch with CUDA support..."
"$PYTHON_CMD" -m uv pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu128
"$PYTHON_CMD" -m uv pip install torchvision>=0.16.0 --index-url https://download.pytorch.org/whl/cu128

echo "Installing TensorRT optimization (optional, may fail if TensorRT not available)..."
if ! "$PYTHON_CMD" -m uv pip install torch-tensorrt>=2.1.0 --index-url https://download.pytorch.org/whl/cu128; then
    echo "Warning: torch-tensorrt installation failed. This is optional and won't affect basic functionality."
    echo "To use TensorRT optimizations, install NVIDIA TensorRT separately."
    echo "Continuing with installation..."
fi

echo
echo "Installing remaining dependencies..."
"$PYTHON_CMD" -m uv pip install -r "$PROJECT_ROOT/requirements/requirements_cuda.txt"

echo
echo "Installation completed successfully!" 
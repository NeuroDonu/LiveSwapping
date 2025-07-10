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
echo "Installing dependencies for CPU..."
echo "Project root: $PROJECT_ROOT"
echo

echo "Installing PyTorch with CPU support..."
"$PYTHON_CMD" -m uv pip install torch>=2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
"$PYTHON_CMD" -m uv pip install torchvision>=0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu

echo
echo "Installing remaining dependencies..."
"$PYTHON_CMD" -m uv pip install -r "$PROJECT_ROOT/requirements/requirements_cpu.txt"

echo
echo "Installation completed successfully!" 
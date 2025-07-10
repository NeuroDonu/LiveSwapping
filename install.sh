#!/bin/bash
#
# LiveSwapping Installation Script for Linux/macOS
# 
# Usage:
#   ./install.sh                                   - Use system Python
#   ./install.sh "/path/to/python"                 - Use specific Python
#   ./install.sh "../python311/bin/python"        - Use relative path
#   ./install.sh "/usr/bin/python3.11"            - Use absolute path
#
# This script automatically:
#   - Detects your GPU type (NVIDIA/AMD/Intel)
#   - Suggests optimal configuration
#   - Installs uv package manager if needed
#   - Handles PyTorch CUDA version selection
#

set -e

echo "===================================================="
echo "          LiveSwapping - Installation"
echo "===================================================="

# Handle Python path argument  
if [ -z "$1" ]; then
    PYTHON_PATH="python3"
    echo "Using system Python..."
else
    # Convert relative path to absolute if needed
    PYTHON_PATH="$(realpath "$1")"
    echo "Using Python from: $PYTHON_PATH"
fi

check_python() {
    if ! "$PYTHON_PATH" --version &> /dev/null; then
        echo "ERROR: Python not found at: $PYTHON_PATH"
        echo "Please check the path or install Python 3.8+"
        exit 1
    fi
    
    python_version=$("$PYTHON_PATH" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo "Python version: $python_version"
    
    if ! "$PYTHON_PATH" -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
        echo "ERROR: Python 3.8 or newer required"
        exit 1
    fi
}

detect_gpu() {
    echo "Detecting GPU type..."
    
    if command -v nvidia-smi &> /dev/null; then
        nvidia_output=$(nvidia-smi --query-gpu=name --format=csv 2>/dev/null | tail -n +2 || echo "")
        if [ ! -z "$nvidia_output" ]; then
            echo "NVIDIA GPU detected: $nvidia_output"
            return 0  # CUDA
        fi
    fi
    
    if lspci 2>/dev/null | grep -i nvidia &> /dev/null; then
        echo "NVIDIA GPU detected (via lspci)"
        echo "WARNING: nvidia-smi not found - ensure NVIDIA drivers are installed"
        return 0  # CUDA
    fi
    
    if lspci 2>/dev/null | grep -i "vga.*amd\|vga.*radeon\|vga.*ati" &> /dev/null; then
        echo "AMD GPU detected"
        echo "WARNING: AMD GPU only supported in CPU mode on Linux"
        return 1  # CPU
    fi
    
    if lspci 2>/dev/null | grep -i "vga.*intel" &> /dev/null; then
        echo "Intel GPU detected"
        return 2  # Intel/OpenVINO
    fi
    
    echo "WARNING: No GPU detected or unknown type - using CPU mode"
    return 1  # CPU
}



main() {
    check_python
    
    cd "$(dirname "$0")"
    
    detect_gpu
    gpu_type=$?
    
    echo ""
    echo "Configuration options:"
    
    case $gpu_type in
        0)  # NVIDIA/CUDA
            echo "1. CUDA (NVIDIA GPU) - recommended"
            echo "2. OpenVINO (Intel optimization)"
            echo "3. CPU only"
            echo ""
            read -p "Select mode (1-3, default 1): " choice
            choice=${choice:-1}
            
            case $choice in
                1)
                    echo "Launching CUDA installer..."
                    ./installers/install_cuda.sh "$PYTHON_PATH"
                    ;;
                2)
                    echo "Launching OpenVINO installer..."
                    ./installers/install_openvino.sh "$PYTHON_PATH"
                    ;;
                3)
                    echo "Launching CPU installer..."
                    ./installers/install_cpu.sh "$PYTHON_PATH"
                    ;;
                *)
                    echo "ERROR: Invalid choice"
                    exit 1
                    ;;
            esac
            ;;
        1)  # CPU/AMD
            echo "1. CPU only - recommended"
            echo "2. OpenVINO (Intel optimization)"
            echo ""
            read -p "Select mode (1-2, default 1): " choice
            choice=${choice:-1}
            
            case $choice in
                1)
                    echo "Launching CPU installer..."
                    ./installers/install_cpu.sh "$PYTHON_PATH"
                    ;;
                2)
                    echo "Launching OpenVINO installer..."
                    ./installers/install_openvino.sh "$PYTHON_PATH"
                    ;;
                *)
                    echo "ERROR: Invalid choice"
                    exit 1
                    ;;
            esac
            ;;
        2)  # Intel
            echo "1. OpenVINO (Intel GPU) - recommended"
            echo "2. CPU only"
            echo ""
            read -p "Select mode (1-2, default 1): " choice
            choice=${choice:-1}
            
            case $choice in
                1)
                    echo "Launching OpenVINO installer..."
                    ./installers/install_openvino.sh "$PYTHON_PATH"
                    ;;
                2)
                    echo "Launching CPU installer..."
                    ./installers/install_cpu.sh "$PYTHON_PATH"
                    ;;
                *)
                    echo "ERROR: Invalid choice"
                    exit 1
                    ;;
            esac
            ;;
    esac
    
    echo ""
    echo "[SUCCESS] Ready! You can now run:"
    if [ "$PYTHON_PATH" = "python3" ]; then
        echo "   python3 run.py"
    else
        echo "   \"$PYTHON_PATH\" run.py"
    fi
    echo ""
    echo "[INFO] Installation used uv package manager for faster downloads"
    echo "[INFO] For direct access to installers, check the 'installers' folder"
}

main "$@" 
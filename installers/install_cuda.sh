#!/bin/bash

echo "===================================================="
echo "        LiveSwapping CUDA Installation"
echo "===================================================="
echo
echo "Please select your CUDA version:"
echo
echo "1. CUDA 12.1 (older, more compatible)"
echo "2. CUDA 12.8 (newer, latest features)"
echo
read -p "Enter your choice (1 or 2): " choice

case $choice in
    1)
        echo
        echo "Installing for CUDA 12.1..."
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        "$SCRIPT_DIR/install_cuda121.sh" "$1"
        ;;
    2)
        echo
        echo "Installing for CUDA 12.8..."
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        "$SCRIPT_DIR/install_cuda128.sh" "$1"
        ;;
    *)
        echo
        echo "Invalid choice. Please run the script again and select 1 or 2."
        exit 1
        ;;
esac

echo
echo "Installation script finished."
case $choice in
    1) echo "Installed with CUDA 12.1 support." ;;
    2) echo "Installed with CUDA 12.8 support." ;;
esac 
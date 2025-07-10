# 📥 Installation Guide

**Complete installation guide for LiveSwapping with multiple installation options**

> 🌍 **English version** | 🇷🇺 **[Русская версия](../Installation)**

---

## 📚 Table of Contents

1. [System Requirements](#-system-requirements)
2. [Automatic Installation (Recommended)](#-automatic-installation-recommended)
3. [Manual Installation](#-manual-installation)
4. [CUDA Installation](#-cuda-installation)
5. [Installation Verification](#-installation-verification)
6. [Troubleshooting](#-troubleshooting)

---

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.15+
- **Python**: 3.8 - 3.11
- **RAM**: 8GB+
- **Storage**: 10GB free space
- **CPU**: Intel i5+ or AMD Ryzen 5+

### Recommended Requirements
- **GPU**: NVIDIA RTX 20-series or newer
- **VRAM**: 6GB+ (4GB minimum)
- **RAM**: 16GB+
- **CPU**: Intel i7+ or AMD Ryzen 7+

### Supported GPUs

| GPU Series | Status | Performance |
|------------|--------|-------------|
| **RTX 40-series** | ✅ Excellent | 40-50 FPS |
| **RTX 30-series** | ✅ Excellent | 25-40 FPS |
| **RTX 20-series** | ✅ Good | 15-25 FPS |
| **GTX 16-series** | ✅ Basic | 10-15 FPS |
| **Intel Arc** | ⚠️ Limited | 5-10 FPS |
| **AMD GPUs** | ❌ Not supported | - |

---

## 🚀 Automatic Installation (Recommended)

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/NeuroDonu/LiveSwapping.git
cd LiveSwapping
```

### Step 2: Run Installer

#### Windows
```cmd
# Run installer (requires PowerShell)
.\install.bat

# Or manual PowerShell execution
powershell -ExecutionPolicy Bypass -File scripts/install.ps1
```

#### Linux/macOS
```bash
# Make installer executable
chmod +x install.sh

# Run installer
./install.sh
```

### What the Installer Does

The automatic installer will:
1. ✅ **Check system compatibility**
2. ✅ **Install Miniconda** (if not present)
3. ✅ **Create isolated environment**
4. ✅ **Install Python dependencies**
5. ✅ **Configure GPU support** (CUDA/OpenVINO)
6. ✅ **Download AI models**
7. ✅ **Run system tests**

### Installer Options

```bash
# GPU-specific installation
./install.sh --cuda        # NVIDIA GPUs
./install.sh --openvino    # Intel GPUs
./install.sh --cpu         # CPU only

# Custom Python version
./install.sh --python 3.10

# Minimal installation (no models)
./install.sh --minimal
```

---

## 🔧 Manual Installation

### Step 1: Environment Setup

#### Option A: Conda (Recommended)
```bash
# Install Miniconda if not present
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create environment
conda create -n liveswapping python=3.10 -y
conda activate liveswapping
```

#### Option B: Python venv
```bash
# Create virtual environment
python -m venv liveswapping
source liveswapping/bin/activate  # Linux/macOS
liveswapping\Scripts\activate     # Windows
```

### Step 2: Core Dependencies

```bash
# Install core package
pip install -e .

# Or install from requirements
pip install -r requirements/requirements_base.txt
```

### Step 3: GPU Support

#### NVIDIA GPUs (CUDA)
```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ONNX Runtime GPU
pip install onnxruntime-gpu

# Install CUDA dependencies
pip install -r requirements/requirements_cuda.txt
```

#### Intel GPUs (OpenVINO)
```bash
# Install OpenVINO
pip install openvino[pytorch]

# Install Intel dependencies
pip install -r requirements/requirements_openvino.txt
```

#### CPU Only
```bash
# CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CPU ONNX Runtime
pip install onnxruntime

# CPU dependencies
pip install -r requirements/requirements_cpu.txt
```

### Step 4: Download Models

```bash
# Download all models (6GB)
python scripts/download_models.py

# Download specific model
python scripts/download_models.py --model reswapper128

# List available models
python scripts/download_models.py --list
```

---

## 🚀 CUDA Installation

For optimal performance with NVIDIA GPUs, follow our comprehensive CUDA setup guide:

**[🔗 Complete CUDA Installation Guide](CUDA-Installation-Guide)**

Quick CUDA setup:
```bash
# Install CUDA via conda (recommended)
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -y
conda install conda-forge::cudnn -y

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install TensorRT for 3x speedup
pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## ✅ Installation Verification

### Basic System Check

```bash
# Activate environment
conda activate liveswapping

# Run system diagnostics
python scripts/check_installation.py
```

### Expected Output
```
🔍 LIVESWAPPING INSTALLATION CHECK
=======================================
✅ Python: 3.10.12
✅ PyTorch: 2.1.0 (CUDA 12.1)
✅ ONNX Runtime: 1.16.1 (GPU)
✅ OpenCV: 4.8.1
✅ Models: 3/3 downloaded
✅ GPU: NVIDIA GeForce RTX 4090 (24GB)

🎉 Installation successful!
```

### Performance Test

```bash
# Quick performance benchmark
python scripts/benchmark.py --quick

# Full benchmark (takes 5 minutes)
python scripts/benchmark.py --full
```

### GUI Test

```bash
# Launch GUI to verify everything works
python run.py
```

---

## 🐞 Troubleshooting

### Common Issues

#### 1. "No module named 'torch'"
```bash
# Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 2. "CUDA out of memory"
```bash
# Reduce model size or use CPU
python run.py --device cpu
python run.py --model reswapper128  # Smaller model
```

#### 3. "No such file or directory: nvcc"
```bash
# Install CUDA toolkit
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -y

# Or download from NVIDIA website
# https://developer.nvidia.com/cuda-downloads
```

#### 4. Models not downloading
```bash
# Manual model download
mkdir -p models
wget https://github.com/NeuroDonu/LiveSwapping/releases/download/v1.0/reswapper128.pth -O models/reswapper128.pth
```

#### 5. Low FPS performance
```bash
# Check GPU usage
nvidia-smi

# Enable optimizations
python run.py --use_tensorrt --fp16

# Check CUDA is working
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### System-Specific Issues

#### Windows Issues
```cmd
# Long path support
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1

# PowerShell execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Linux Issues
```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Install build essentials
sudo apt install build-essential cmake

# Fix library issues
sudo apt install libgl1-mesa-glx libglib2.0-0
```

#### macOS Issues
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# For M1/M2 Macs (CPU only)
pip install torch torchvision torchaudio
```

---

## 🔄 Update Installation

### Update LiveSwapping
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Update models if needed
python scripts/download_models.py --update
```

### Update CUDA Stack
```bash
# Update PyTorch
pip install torch torchvision torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu121

# Update ONNX Runtime
pip install onnxruntime-gpu --upgrade
```

---

## 🗑️ Uninstallation

### Remove Environment
```bash
# Conda environment
conda env remove -n liveswapping

# Python venv
rm -rf liveswapping  # Linux/macOS
rmdir /s liveswapping  # Windows
```

### Clean System
```bash
# Remove downloaded models
rm -rf models/

# Remove cache
rm -rf ~/.cache/torch/
rm -rf ~/.cache/huggingface/
```

---

## 📊 Installation Sizes

| Component | Size | Description |
|-----------|------|-------------|
| **Base Package** | ~500MB | Core dependencies |
| **CUDA Support** | ~2GB | PyTorch CUDA + tools |
| **AI Models** | ~6GB | All face swap models |
| **Enhancers** | ~2GB | GFPGAN, RealESRGAN |
| **Total (Full)** | ~10GB | Complete installation |

---

## 🌍 Language Selection

- 🌍 **English** (current)
- 🇷🇺 **[Русский](../Installation)**

---

*[⬅️ Quick Start](Quick-Start) | [🏠 Home](Home) | [CUDA Guide ➡️](CUDA-Installation-Guide)*
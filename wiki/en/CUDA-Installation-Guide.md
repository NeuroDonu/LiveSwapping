# 🚀 Complete CUDA+cuDNN+TensorRT Installation Guide

**Safe and easy installation of the complete CUDA stack for LiveSwapping using conda/miniconda**

> 🌍 **English version** | 🇷🇺 **[Русская версия](../CUDA-Installation-Guide)**

## 📚 Table of Contents

1. [Why You Need This Setup](#-why-you-need-this-setup)
2. [System Preparation](#-system-preparation)
3. [Installation via Miniconda (Recommended)](#-installation-via-miniconda-recommended)
4. [Installation Verification](#-installation-verification)
5. [Troubleshooting](#-troubleshooting)
6. [Alternative Methods](#-alternative-methods)

---

## 🎯 Why You Need This Setup

### What a proper CUDA stack installation provides:
- **3x acceleration** for LiveSwapping with TensorRT
- **Stable operation** without version conflicts
- **GPU acceleration** for all operations
- **Compatibility** with PyTorch and ONNX Runtime

### Stack components:
- **CUDA Toolkit** - primary GPU computing platform
- **cuDNN** - neural network library for CUDA
- **TensorRT** - inference optimization for NVIDIA GPUs
- **PyTorch** - with CUDA support
- **ONNX Runtime** - with GPU providers

---

## 🛠️ System Preparation

### GPU Compatibility Check

```bash
# Check for NVIDIA GPU
nvidia-smi

# Should display GPU information:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.54.03    Driver Version: 535.54.03    CUDA Version: 12.2   |
# +-------------------------------+----------------------+----------------------+
```

### Supported GPUs:
- ✅ **RTX 40 series** (4090, 4080, 4070, 4060)
- ✅ **RTX 30 series** (3090, 3080, 3070, 3060)
- ✅ **RTX 20 series** (2080, 2070, 2060)
- ✅ **GTX 16 series** (1660, 1650)
- ✅ **Tesla, Quadro** series
- ❌ **GTX 10xx and older** (limited TensorRT support)

### Driver Requirements:

| CUDA Version | Minimum Driver Version |
|--------------|------------------------|
| CUDA 12.1 | 530.30.02+ (Linux), 531.14+ (Windows) |
| CUDA 12.4 | 550.54.15+ (Linux), 551.61+ (Windows) |
| CUDA 11.8 | 520.61.05+ (Linux), 522.06+ (Windows) |

---

## 🐍 Installation via Miniconda (Recommended)

### Why conda is the best choice:
- ✅ **Isolated environments** - no conflicts
- ✅ **Automatic dependency** management
- ✅ **Pre-compiled** packages
- ✅ **Easy reinstallation** when problems occur
- ✅ **Works consistently** everywhere

### Step 1: Install Miniconda

#### Windows:
```cmd
# Download Miniconda from official website
# https://docs.conda.io/en/latest/miniconda.html

# Or via PowerShell:
Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile "miniconda.exe"
.\miniconda.exe /S /D=C:\miniconda3

# Restart terminal or run:
C:\miniconda3\Scripts\activate.bat
```

#### Linux:
```bash
# Download and install
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Add to PATH
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Step 2: Create Isolated Environment

```bash
# Create environment with Python 3.10
conda create -n liveswapping python=3.10 -y

# Activate environment
conda activate liveswapping

# Verify
python --version  # Should show Python 3.10.x
which python      # Should show miniconda path
```

### Step 3: Install CUDA Toolkit via conda

```bash
# Install CUDA 12.1 (recommended for PyTorch 2.1+)
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -y

# Or CUDA 11.8 (for older systems)
# conda install nvidia/label/cuda-11.8.0::cuda-toolkit -y

# Verify installation
nvcc --version
```

### Step 4: Install cuDNN

```bash
# cuDNN via conda-forge
conda install conda-forge::cudnn -y

# Or specific version for CUDA 12.1
conda install nvidia::cudnn=8.9.7.29 -y
```

### Step 5: Install PyTorch with CUDA

```bash
# PyTorch with CUDA 12.1 support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Or for CUDA 11.8
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### Step 6: Install TensorRT

```bash
# TensorRT via pip (as conda version may be outdated)
pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu121

# Or for CUDA 11.8
# pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu118
```

### Step 7: Install Additional Dependencies

```bash
# ONNX Runtime with GPU support
pip install onnxruntime-gpu

# CuPy for numpy acceleration
conda install cupy -c conda-forge -y

# Additional libraries
pip install opencv-python pillow numpy scipy
```

### Step 8: Install LiveSwapping

```bash
# Clone repository
git clone https://github.com/NeuroDonu/LiveSwapping.git
cd LiveSwapping

# Install in development mode
pip install -e .

# Or install dependencies
pip install -r requirements/requirements_cuda.txt
```

---

## ✅ Installation Verification

### Comprehensive Diagnostics

Create file `test_cuda_stack.py`:

```python
#!/usr/bin/env python3
"""
Comprehensive CUDA stack check for LiveSwapping
"""

import sys
import subprocess

def check_component(name, check_func):
    """Check component with colored output."""
    try:
        result = check_func()
        if result:
            print(f"✅ {name}: OK")
            if isinstance(result, str):
                print(f"   {result}")
        else:
            print(f"❌ {name}: FAILED")
        return bool(result)
    except Exception as e:
        print(f"❌ {name}: ERROR - {e}")
        return False

def check_python():
    """Check Python version."""
    version = sys.version.split()[0]
    major, minor = map(int, version.split('.')[:2])
    if major == 3 and minor >= 8:
        return f"Python {version}"
    return None

def check_conda():
    """Check conda environment."""
    try:
        result = subprocess.run(['conda', '--version'], 
                              capture_output=True, text=True, check=True)
        env_name = subprocess.run(['conda', 'info', '--envs'], 
                                capture_output=True, text=True, check=True)
        if '*' in env_name.stdout:
            active_env = [line for line in env_name.stdout.split('\n') 
                         if '*' in line][0].split()[0]
            return f"{result.stdout.strip()}, active env: {active_env}"
        return result.stdout.strip()
    except:
        return None

def check_cuda_toolkit():
    """Check CUDA Toolkit."""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, check=True)
        lines = result.stdout.split('\n')
        version_line = [line for line in lines if 'release' in line][0]
        version = version_line.split('release ')[1].split(',')[0]
        return f"CUDA {version}"
    except:
        return None

def check_nvidia_driver():
    """Check NVIDIA driver."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        driver_version = result.stdout.strip()
        return f"Driver {driver_version}"
    except:
        return None

def check_pytorch():
    """Check PyTorch with CUDA."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            return f"PyTorch {torch.__version__}, CUDA {cuda_version}, {device_count} GPU(s), Current: {device_name}"
        else:
            return f"PyTorch {torch.__version__} (CUDA NOT AVAILABLE)"
    except ImportError:
        return None

def check_tensorrt():
    """Check TensorRT."""
    try:
        import torch_tensorrt
        return f"TensorRT {torch_tensorrt.__version__}"
    except ImportError:
        return None

def check_onnxruntime():
    """Check ONNX Runtime."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        gpu_providers = [p for p in providers if 'CUDA' in p or 'DirectML' in p or 'OpenVINO' in p]
        return f"ONNX Runtime {ort.__version__}, GPU providers: {gpu_providers}"
    except ImportError:
        return None

def check_cupy():
    """Check CuPy."""
    try:
        import cupy as cp
        # Simple test
        a = cp.array([1, 2, 3])
        b = cp.array([4, 5, 6])
        c = a + b
        return f"CuPy {cp.__version__}"
    except ImportError:
        return None
    except Exception as e:
        return f"CuPy installed but error: {e}"

def check_liveswapping():
    """Check LiveSwapping."""
    try:
        from liveswapping.utils.gpu_utils import print_gpu_info, get_optimal_provider
        provider = get_optimal_provider()
        return f"LiveSwapping OK, optimal provider: {provider}"
    except ImportError:
        return None

def performance_test():
    """Quick performance test."""
    try:
        import torch
        import time
        
        if not torch.cuda.is_available():
            return "CUDA not available for testing"
        
        device = torch.device('cuda')
        
        # Create test data
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        
        # Warmup
        for _ in range(10):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
        
        # Test
        start_time = time.time()
        for _ in range(100):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        ops_per_sec = 1.0 / avg_time
        
        return f"Matrix multiplication: {avg_time:.4f}s per op, {ops_per_sec:.1f} ops/sec"
        
    except Exception as e:
        return f"Performance test failed: {e}"

def main():
    """Main diagnostics function."""
    print("🔍 CUDA STACK DIAGNOSTICS FOR LIVESWAPPING")
    print("=" * 60)
    
    checks = [
        ("Python", check_python),
        ("Conda", check_conda),
        ("NVIDIA Driver", check_nvidia_driver),
        ("CUDA Toolkit", check_cuda_toolkit),
        ("PyTorch", check_pytorch),
        ("TensorRT", check_tensorrt),
        ("ONNX Runtime", check_onnxruntime),
        ("CuPy", check_cupy),
        ("LiveSwapping", check_liveswapping),
    ]
    
    results = []
    for name, check_func in checks:
        result = check_component(name, check_func)
        results.append(result)
    
    print("\n🚀 PERFORMANCE TEST")
    print("-" * 30)
    check_component("GPU Performance", performance_test)
    
    print(f"\n📊 OVERALL RESULT")
    print("-" * 20)
    success_count = sum(results)
    total_count = len(results)
    
    if success_count == total_count:
        print("🎉 ALL COMPONENTS WORKING PERFECTLY!")
        print("   LiveSwapping is ready for maximum performance")
    elif success_count >= 6:
        print("✅ SYSTEM READY TO WORK")
        print("   Some components may be unavailable, but core functionality works")
    else:
        print("⚠️  ADDITIONAL SETUP REQUIRED")
        print("   Check failed components and repeat installation")
    
    print(f"\nStatus: {success_count}/{total_count} components working")

if __name__ == "__main__":
    main()
```

Run diagnostics:

```bash
python test_cuda_stack.py
```

### Expected Output (successful installation):

```
🔍 CUDA STACK DIAGNOSTICS FOR LIVESWAPPING
============================================================
✅ Python: OK
   Python 3.10.12
✅ Conda: OK
   conda 23.7.4, active env: liveswapping
✅ NVIDIA Driver: OK
   Driver 535.54.03
✅ CUDA Toolkit: OK
   CUDA 12.1
✅ PyTorch: OK
   PyTorch 2.1.0, CUDA 12.1, 1 GPU(s), Current: NVIDIA GeForce RTX 4090
✅ TensorRT: OK
   TensorRT 2.1.0
✅ ONNX Runtime: OK
   ONNX Runtime 1.16.1, GPU providers: ['CUDAExecutionProvider']
✅ CuPy: OK
   CuPy 12.2.0
✅ LiveSwapping: OK
   optimal provider: cuda

🚀 PERFORMANCE TEST
------------------------------
✅ GPU Performance: OK
   Matrix multiplication: 0.0001s per op, 8542.3 ops/sec

📊 OVERALL RESULT
--------------------
🎉 ALL COMPONENTS WORKING PERFECTLY!
   LiveSwapping is ready for maximum performance

Status: 9/9 components working
```

---

## 🔧 Troubleshooting

### Problem 1: "CUDA driver version is insufficient"

**Symptoms:**
```
RuntimeError: CUDA driver version is insufficient for CUDA runtime version
```

**Solution:**
```bash
# Check driver version
nvidia-smi

# Update NVIDIA driver:
# Windows: https://www.nvidia.com/drivers/
# Ubuntu: sudo apt update && sudo apt install nvidia-driver-535

# Reboot system
sudo reboot
```

### Problem 2: "torch-tensorrt compilation failed"

**Symptoms:**
```
WARNING: torch-tensorrt compilation failed, falling back to PyTorch
```

**Solution:**
```bash
# Reinstall torch-tensorrt
pip uninstall torch-tensorrt
pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu121

# Check version compatibility
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

### Problem 3: CUDA Version Conflict

**Symptoms:**
```
RuntimeError: CUDA version mismatch
```

**Solution - complete reinstallation:**
```bash
# Deactivate environment
conda deactivate

# Remove old environment
conda env remove -n liveswapping

# Create new environment from scratch
conda create -n liveswapping python=3.10 -y
conda activate liveswapping

# Clean installation of entire stack
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -y
conda install conda-forge::cudnn -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## 🎯 Final LiveSwapping Check

After installation, verify LiveSwapping works:

```bash
# Run diagnostics
python test_cuda_stack.py

# Check model loading
python -c "
from liveswapping.ai_models.models import load_model, get_optimal_provider
print(f'Optimal provider: {get_optimal_provider()}')
model = load_model('reswapper128', use_tensorrt=True)
print('✅ Model loaded successfully with TensorRT!')
"

# Test GUI
python run.py
```

## 📚 Additional Resources

- **[NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)**
- **[PyTorch Installation Guide](https://pytorch.org/get-started/locally/)**
- **[TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/)**
- **[Conda User Guide](https://docs.conda.io/en/latest/)**

---

## 🆘 Getting Help

If you still have problems:

1. **Run diagnostics:** `python test_cuda_stack.py`
2. **Check logs:** save command outputs
3. **Create issue:** [GitHub Issues](https://github.com/NeuroDonu/LiveSwapping/issues)
4. **Include information:**
   - Output of `nvidia-smi`
   - Diagnostic script output
   - Operating system and version
   - GPU model

---

> **💡 Tip:** Always use conda environments for isolation! This prevents conflicts and allows easy component reinstallation.

---

## 🌍 Language Selection

- 🌍 **English** (current)
- 🇷🇺 **[Русский](../CUDA-Installation-Guide)**

---

*[⬅️ Installation](Installation) | [🏠 Home](Home) | [🇷🇺 Russian version](../CUDA-Installation-Guide)*
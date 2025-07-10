# Installation Scripts

These scripts help you install LiveSwapping dependencies with proper PyTorch index URLs that are compatible with `uv`.

## Available Scripts

### Main Installation Scripts
- `install_cuda.bat` / `install_cuda.sh` - Install dependencies for CUDA GPU acceleration (with version selection)
- `install_cpu.bat` / `install_cpu.sh` - Install dependencies for CPU-only usage

### Specific CUDA Version Scripts
- `install_cuda121.bat` / `install_cuda121.sh` - Install for CUDA 12.1 specifically
- `install_cuda128.bat` / `install_cuda128.sh` - Install for CUDA 12.8 specifically

### Platform-Specific Scripts
- `install_dml.bat` - DirectML installation for AMD GPUs (Windows only)
- `install_openvino.bat` / `install_openvino.sh` - OpenVINO installation for Intel optimization

### Optional Components
- `install_tensorrt_optional.bat` - Optional TensorRT installation for additional GPU optimizations

## Usage

### Windows (PowerShell/CMD)

**Using system Python:**
```cmd
cd installers
install_cuda.bat
```
*The script will prompt you to choose between CUDA 12.1 or 12.8*

**Using specific Python installation:**
```cmd
cd installers
install_cuda.bat "C:\path\to\your\python.exe"
```

**Example with your Python path:**
```cmd
cd installers
install_cuda.bat "..\python311\python.exe"
```

**Direct installation for specific CUDA version:**
```cmd
cd installers
install_cuda121.bat "..\python311\python.exe"  # For CUDA 12.1
install_cuda128.bat "..\python311\python.exe"  # For CUDA 12.8
```

### Linux/MacOS (Bash)

**Using system Python:**
```bash
cd installers
./install_cuda.sh
```
*The script will prompt you to choose between CUDA 12.1 or 12.8*

**Using specific Python installation:**
```bash
cd installers
./install_cuda.sh "/path/to/your/python"
```

**Example:**
```bash
cd installers
./install_cuda.sh "/usr/bin/python3.11"
```

**Direct installation for specific CUDA version:**
```bash
cd installers
./install_cuda121.sh "/usr/bin/python3.11"  # For CUDA 12.1
./install_cuda128.sh "/usr/bin/python3.11"  # For CUDA 12.8
```

### Optional TensorRT Installation

TensorRT provides additional GPU optimizations but is **not required** for basic functionality:

```cmd
cd installers
install_tensorrt_optional.bat "..\python311\python.exe"
```

**Note:** TensorRT requires manual installation of NVIDIA TensorRT SDK. Download from [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)

## Features

- ✅ **CUDA version selection** - Choose between CUDA 12.1 and 12.8 interactively
- ✅ **Automatic path detection** - Scripts automatically find project root
- ✅ **Automatic uv installation** - Installs uv package manager if not found
- ✅ **Error handling** - Installation stops on any error
- ✅ **Flexible Python paths** - Use system Python or specify custom path
- ✅ **Cross-platform** - Works on Windows, Linux, and MacOS
- ✅ **UV compatible** - Fixes PyTorch index URL issues with uv package manager
- ✅ **Unified dependencies** - Single requirements file for all CUDA versions

## Requirements

- Python 3.8+ with pip support
- `uv` package manager (will be automatically installed if not found)

## Troubleshooting

If you get permission errors on Linux/MacOS, make sure scripts are executable:
```bash
chmod +x install_cuda.sh install_cpu.sh
``` 
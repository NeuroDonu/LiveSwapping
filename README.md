# LiveSwapping - Real-time Face Swap

**[🇷🇺 Русская версия](README_RU.md)**

LiveSwapping is an advanced real-time face swapping system supporting multiple models and cutting-edge optimization technologies.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1%20%7C%2012.8-orange.svg)
![TensorRT](https://img.shields.io/badge/TensorRT-10.x-red.svg)

## ✨ Key Features

- 🚀 **Real-time face swapping** with webcam support
- 🎬 **Video processing** with high-quality output
- 🧠 **Multiple AI models**: StyleTransfer, DFM, InSwapper
- ⚡ **GPU acceleration** with torch-tensorrt optimization (3x speedup)
- 🎯 **Advanced face detection** using InsightFace
- 🔧 **Smart upscaling** with GFPGAN and RealESRGAN
- 🎨 **Color correction** and adaptive blending
- 🖥️ **User-friendly GUI** with progress indicators
- 🐍 **Python API** for integration

## 🤖 Supported Models

| Model | Type | Resolution | Description |
|-------|------|------------|-------------|
| **reswapper128** | StyleTransfer | 128x128 | Fast, good quality face swapping |
| **reswapper256** | StyleTransfer | 256x256 | High quality, slower processing |
| **inswapper128** | InsightFace | 128x128 | Industry-standard face swapping |
| **DFM models** | Custom | Variable | Deep Face Model format support |

## 🚀 Installation

### Quick Start (Recommended)

The automated installer detects your GPU and configures optimal settings:

**Windows:**
```cmd
# System Python
install.bat

# Custom Python path
install.bat "C:\Python311\python.exe"
```

**Linux/macOS:**
```bash
# System Python
./install.sh

# Custom Python path
./install.sh "/usr/bin/python3.11"
```

### Manual Installation

1. **Clone the repository:**
```bash
git clone https://github.com/NeuroDonu/LiveSwapping.git
cd LiveSwapping
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

3. **Install dependencies:**
```bash
# CUDA GPU (recommended)
cd installers
./install_cuda.sh  # Linux/macOS
install_cuda.bat   # Windows

# CPU only
./install_cpu.sh   # Linux/macOS
install_cpu.bat    # Windows
```

### Requirements

- **Python 3.8+**
- **CUDA 12.1/12.8** (for GPU acceleration)
- **8GB+ RAM** (16GB+ recommended)
- **GPU**: NVIDIA (CUDA), AMD (DirectML), Intel (OpenVINO)

## 🎯 Usage

### GUI Interface

**Real-time processing:**
```bash
python run.py  # Select option 1
```

**Video processing:**
```bash
python run.py  # Select option 2
```

### Command Line Interface

**Real-time face swap:**
```bash
python run.py realtime \
    --source_image path/to/source.jpg \
    --model_path models/reswapper-1019500.pth \
    --camera_id 0
```

**Image processing:**
```bash
python run.py image \
    --source path/to/source.jpg \
    --target path/to/target.jpg \
    --modelPath models/reswapper-1019500.pth \
    --output result.jpg
```

**Video processing:**
```bash
python run.py video \
    --source path/to/source.jpg \
    --target_video path/to/video.mp4 \
    --modelPath models/reswapper-1019500.pth \
    --output output.mp4
```

### Python API

```python
from liveswapping.ai_models.models import load_model
from liveswapping.core import realtime, video

# Load model with TensorRT optimization
model = load_model("reswapper128", use_tensorrt=True)

# Process single image
result = process_image(source_img, target_img, model)

# Real-time processing
realtime.start_processing(model, camera_id=0)
```

## ⚡ Performance Benchmarks

*Tested on RTX 4090*

| Component | Without Optimization | With Optimization | Speedup | Technology |
|-----------|---------------------|-------------------|---------|------------|
| **reswapper128** | ~15 FPS | ~45 FPS | **3.0x** | torch-tensorrt |
| **reswapper256** | ~8 FPS | ~25 FPS | **3.1x** | torch-tensorrt |
| **GFPGAN** | ~2.5 FPS | ~7 FPS | **2.8x** | torch-tensorrt |
| **RealESRGAN** | ~1.8 FPS | ~5.2 FPS | **2.9x** | torch-tensorrt |
| **Face alignment** | ~5 ms | ~2 ms | **2.5x** | CuPy |

### Optimization Technologies

- **🔥 torch-tensorrt**: 3x speedup for PyTorch models
- **⚡ CuPy acceleration**: GPU-accelerated NumPy operations
- **🎨 Adaptive processing**: Smart GPU/CPU selection based on image size
- **🧠 Memory optimization**: Efficient VRAM usage

## 🛠️ Advanced Configuration

### TensorRT Optimization

```python
# Enable TensorRT for maximum performance
model = load_model("reswapper128", use_tensorrt=True)

# Disable for debugging
model = load_model("reswapper128", use_tensorrt=False)
```

### Provider Selection

```python
# Force specific provider
model = load_model("reswapper128", provider_type="cuda")     # NVIDIA GPU
model = load_model("reswapper128", provider_type="directml") # AMD GPU
model = load_model("reswapper128", provider_type="openvino") # Intel
model = load_model("reswapper128", provider_type="cpu")      # CPU only
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce resolution or use CPU
   python run.py image --model_provider cpu
   ```

2. **torch-tensorrt compilation failed**
   ```python
   # Disable TensorRT optimization
   model = load_model("reswapper128", use_tensorrt=False)
   ```

3. **basicsr import error** (video processing only)
   ```bash
   # Automatic fix
   python liveswapping/utils/fix_basicsr.py
   ```

4. **Model download issues**
   - Check internet connection
   - Ensure sufficient disk space (2GB+)
   - Manually download to `models/` folder

### Performance Tips

- **Use CUDA** for 3-5x speedup over CPU
- **Enable TensorRT** for additional 3x speedup
- **Increase resolution** gradually to find optimal balance
- **Use SSD storage** for faster model loading

## 📁 Project Structure

```
LiveSwapping/
├── liveswapping/           # Main package
│   ├── ai_models/          # Model implementations
│   │   ├── models.py       # Unified model loading
│   │   ├── download_models.py # Automatic model download
│   │   └── style_transfer_model_128.py
│   ├── core/               # Core processing modules
│   │   ├── Image.py        # Image processing
│   │   ├── video.py        # Video processing
│   │   ├── realtime.py     # Real-time processing
│   │   └── face_align.py   # Face alignment utilities
│   ├── gui/                # GUI interfaces
│   │   ├── realtime_gui.py # Real-time GUI
│   │   └── video_gui.py    # Video processing GUI
│   └── utils/              # Utility modules
│       ├── upscalers.py    # GFPGAN/RealESRGAN
│       └── gpu_utils.py    # GPU acceleration
├── models/                 # Model storage
├── installers/             # Installation scripts
└── requirements/           # Dependency files
```

## 🧪 Development

### Running Tests

```bash
# Test image processing
python run.py image --source test/source.jpg --target test/target.jpg

# Benchmark performance
python -c "from liveswapping.utils.gpu_utils import benchmark_performance; benchmark_performance()"
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10+, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8+
- **RAM**: 8GB
- **Storage**: 5GB free space

### Recommended Requirements
- **GPU**: NVIDIA RTX 30/40 series, AMD RX 6000+ series
- **RAM**: 16GB+
- **Storage**: 10GB+ SSD space
- **CUDA**: 12.1 or 12.8

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for face detection
- [GFPGAN](https://github.com/TencentARC/GFPGAN) for face restoration
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for upscaling
- [torch-tensorrt](https://github.com/pytorch/TensorRT) for optimization

## 🔗 Links

- **🇷🇺 Russian Version**: [README_RU.md](README_RU.md)
- **📧 Support**: Create an issue for bug reports and feature requests
- **💬 Community**: Discussions and help available in Issues section
- **📖 Documentation**: See this README and inline code comments

---

**⭐ Star this repository if LiveSwapping helped you!** 
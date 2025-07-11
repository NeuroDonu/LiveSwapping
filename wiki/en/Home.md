# 🎭 LiveSwapping Wiki

**Welcome to the LiveSwapping Wiki** - an advanced real-time face swapping system with multiple model support and cutting-edge optimization technologies.

> 🌍 **English version** | 🇷🇺 **[Русская версия](../Home)**

![LiveSwapping Banner](https://img.shields.io/badge/LiveSwapping-Real--time%20Face%20Swap-blue?style=for-the-badge)

## 🚀 Quick Navigation

### 📖 Main Sections
- **[📥 Installation](Installation)** - Setup and configuration
- **[🎯 Quick Start](Quick-Start)** - Get started in 5 minutes
- **[👤 User Guide](User-Guide)** - Complete usage guide
- **[💻 GUI Interface](GUI-Guide)** - Graphical interface guide

### 🔧 For Developers
- **[📋 API Reference](API-Reference)** - API documentation
- **[💡 Code Examples](Code-Examples)** - Practical examples
- **[🏗️ Developer Guide](Developer-Guide)** - Architecture and development
- **[⚡ Performance Optimization](Performance-Optimization)** - Speed optimization

### 🤖 Models and Technologies
- **[🧠 AI Models](AI-Models)** - Available models and usage
- **[🔄 Providers](Providers)** - CUDA, DirectML, OpenVINO, CPU
- **[📈 Upscalers](Upscalers)** - GFPGAN, RealESRGAN, enhancement
- **[🚀 CUDA Setup](CUDA-Installation-Guide)** - Complete CUDA+cuDNN+TensorRT guide

### 🛠️ Troubleshooting
- **[❓ FAQ](FAQ)** - Frequently asked questions
- **[🔧 Troubleshooting](Troubleshooting)** - Problem solving
- **[🐛 Known Issues](Known-Issues)** - Current limitations

---

## ✨ Core Features

| Feature | Description | Status |
|---------|-------------|--------|
| 🎥 **Real-time Processing** | Real-time face swapping from webcam | ✅ Ready |
| 🎬 **Video Processing** | High-quality video file processing | ✅ Ready |
| 🖼️ **Image Processing** | Face swapping on static images | ✅ Ready |
| ⚡ **TensorRT Optimization** | 3x acceleration for PyTorch models | ✅ Ready |
| 🎯 **Multi-providers** | CUDA, DirectML, OpenVINO, CPU | ✅ Ready |
| 🔄 **Auto-optimization** | Smart system adaptation | ✅ Ready |
| 📺 **OBS Integration** | Direct streaming to OBS | ✅ Ready |
| 🎨 **Quality Enhancement** | GFPGAN, RealESRGAN upscaling | ✅ Ready |

---

## 🚦 Platform Support Status

| Platform | Status | Features |
|----------|--------|----------|
| ![Windows](https://img.shields.io/badge/Windows-0078D6?style=flat&logo=windows&logoColor=white) | ✅ Full Support | CUDA, DirectML, CPU |
| ![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat&logo=linux&logoColor=black) | ✅ Full Support | CUDA, OpenVINO, CPU |
| ![macOS](https://img.shields.io/badge/macOS-000000?style=flat&logo=apple&logoColor=white) | ⚠️ Basic Support | CPU only |

---

## 🎯 Quick Start

### 1️⃣ Installation
```bash
# Automatic installation (recommended)
./install.sh  # Linux/macOS
install.bat   # Windows
```

### 2️⃣ Launch GUI
```bash
python run.py
```

### 3️⃣ Real-time Processing
```bash
python -m liveswapping.run realtime \
    --source my_face.jpg \
    --modelPath models/reswapper128.pth
```

---

## 📊 Performance Benchmarks

### RTX 4090 Benchmarks

| Component | Without Optimization | With Optimization | Speedup |
|-----------|---------------------|-------------------|---------|
| **reswapper128** | ~15 FPS | ~45 FPS | **3.0x** |
| **reswapper256** | ~8 FPS | ~25 FPS | **3.1x** |
| **GFPGAN** | ~2.5 FPS | ~7 FPS | **2.8x** |
| **RealESRGAN** | ~1.8 FPS | ~5.2 FPS | **2.9x** |

---

## 🌟 Supported Models

| Model | Type | Resolution | Description | Optimization |
|-------|------|------------|-------------|--------------|
| **reswapper128** | StyleTransfer | 128x128 | Fast, good quality | TensorRT |
| **reswapper256** | StyleTransfer | 256x256 | High quality | TensorRT |
| **inswapper128** | InsightFace | 128x128 | Industry standard | ONNX Runtime |

---

## 🆘 Need Help?

- 📖 **Start with**: [Quick Start Guide](Quick-Start)
- 🔍 **Search problems**: [Troubleshooting](Troubleshooting)
- ❓ **Common questions**: [FAQ](FAQ)
- 💬 **Discussions**: [GitHub Issues](https://github.com/your-repo/issues)

---

## 🤝 Contributing

We welcome contributions to the project! Check out the [Developer Guide](Developer-Guide) for information on how to get started with development.

---

## 📄 License

This project is distributed under the MIT License. See [LICENSE](https://github.com/your-repo/blob/main/LICENSE) for details.

---

## 🌍 Language Selection

- 🌍 **English** (current)
- 🇷🇺 **[Русский](../Home)**

---

*Last updated: December 2024*
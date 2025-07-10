# 🎯 Quick Start Guide

**Get started with LiveSwapping in 5 minutes!**

> 🌍 **English version** | 🇷🇺 **[Русская версия](../Quick-Start)**

---

## 🚀 What You'll Learn

In this guide you will:
- ✅ Install LiveSwapping in 5 minutes
- ✅ Run your first face swap
- ✅ Set up real-time processing
- ✅ Configure optimal performance

---

## 📋 Prerequisites

Before starting, make sure you have:
- 🖥️ **Windows 10+**, **Linux**, or **macOS**
- 🐍 **Python 3.8+** installed
- 🎮 **NVIDIA GPU** (optional but recommended)
- 📷 **Webcam** (for real-time processing)

---

## ⚡ Quick Installation

### Option 1: Automatic Installation (Recommended)

```bash
# Clone repository
git clone https://github.com/NeuroDonu/LiveSwapping.git
cd LiveSwapping

# Run automatic installer
./install.sh  # Linux/macOS
install.bat   # Windows

# The installer will:
# ✅ Create conda environment
# ✅ Install all dependencies
# ✅ Download required models
# ✅ Test GPU compatibility
```

### Option 2: Manual Installation

```bash
# Create environment
conda create -n liveswapping python=3.10 -y
conda activate liveswapping

# Install dependencies
pip install -r requirements.txt

# For CUDA support (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu
```

---

## 🎭 Your First Face Swap

### 1️⃣ Launch GUI

```bash
conda activate liveswapping
python run.py
```

### 2️⃣ Load Models

1. **Open GUI interface**
2. **Click "Load Model"**
3. **Select** `reswapper128` (fastest)
4. **Wait for loading** (first time may take 30 seconds)

### 3️⃣ Set Source Face

1. **Click "Select Source"**
2. **Choose your face image** (JPG/PNG)
3. **Wait for processing**

### 4️⃣ Start Processing

**For images:**
1. **Click "Select Target"**
2. **Choose target image**
3. **Click "Process"**

**For webcam:**
1. **Select webcam** in dropdown
2. **Click "Start Real-time"**
3. **See live face swap!**

---

## 🖼️ Image Processing Example

```bash
# Command line processing
python -m liveswapping.run image \
    --source your_face.jpg \
    --target group_photo.jpg \
    --output result.jpg \
    --modelPath models/reswapper128.pth
```

**Expected processing time:**
- **reswapper128**: ~0.1s per face
- **reswapper256**: ~0.3s per face  
- **With CUDA**: 3-5x faster

---

## 🎥 Real-time Processing

```bash
# Real-time webcam processing
python -m liveswapping.run realtime \
    --source your_face.jpg \
    --modelPath models/reswapper128.pth \
    --webcam_id 0
```

**Expected performance:**
- **RTX 4090**: 45+ FPS
- **RTX 3080**: 30+ FPS
- **RTX 2070**: 20+ FPS
- **CPU only**: 2-5 FPS

---

## 🎬 Video Processing

```bash
# Process video file
python -m liveswapping.run video \
    --source your_face.jpg \
    --target input_video.mp4 \
    --output output_video.mp4 \
    --modelPath models/reswapper256.pth
```

**Processing parameters:**
- **Quality**: High with reswapper256
- **Speed**: Fast with reswapper128
- **Enhancement**: Add `--enhancer gfpgan`

---

## ⚡ Performance Optimization

### Enable TensorRT (3x speedup)

```bash
# Enable TensorRT optimization
python -m liveswapping.run realtime \
    --source your_face.jpg \
    --modelPath models/reswapper128.pth \
    --use_tensorrt \
    --fp16
```

### GPU Selection

```bash
# Check available GPUs
nvidia-smi

# Use specific GPU
export CUDA_VISIBLE_DEVICES=0
python run.py
```

### Memory Optimization

```bash
# For low VRAM GPUs
python run.py --low_mem
```

---

## 🔧 Common Settings

### Quality Settings

| Setting | reswapper128 | reswapper256 | Description |
|---------|-------------|-------------|-------------|
| **Fast** | ✅ Default | ⚠️ Slower | Real-time capable |
| **Quality** | ✅ Good | ✅ Excellent | Best for videos |
| **Memory** | ~2GB VRAM | ~4GB VRAM | GPU requirements |

### Enhancement Options

```bash
# Add face enhancement
--enhancer gfpgan

# Add upscaling
--upscaler realesrgan

# Color correction
--color_correct
```

---

## 🆘 Quick Troubleshooting

### Problem: "No CUDA devices found"
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Problem: "Model loading failed"
```bash
# Download models manually
python scripts/download_models.py
```

### Problem: Low FPS
```bash
# Check GPU usage
nvidia-smi

# Use lighter model
--modelPath models/reswapper128.pth

# Enable optimizations
--use_tensorrt --fp16
```

### Problem: Poor quality
```bash
# Use higher quality model
--modelPath models/reswapper256.pth

# Add enhancement
--enhancer gfpgan
```

---

## 🎯 Next Steps

### ✅ You've completed Quick Start!

**Now you can:**

1. **🔧 Optimize Performance** - [Performance Guide](Performance-Optimization)
2. **📖 Learn Advanced Features** - [User Guide](User-Guide)
3. **🛠️ Install CUDA Stack** - [CUDA Installation](CUDA-Installation-Guide)
4. **💻 Explore API** - [API Reference](API-Reference)

### 🎥 Real-world Usage

**For content creators:**
- **Streaming**: OBS integration
- **Videos**: Batch processing
- **Photos**: Batch face replacement

**For developers:**
- **API integration**: REST API
- **Custom models**: Model training
- **Performance tuning**: Optimization

---

## 📊 Performance Reference

### Typical Performance on Different Hardware

| GPU | reswapper128 | reswapper256 | Memory Usage |
|-----|-------------|-------------|--------------|
| **RTX 4090** | 45 FPS | 25 FPS | 6GB |
| **RTX 4080** | 40 FPS | 22 FPS | 5GB |
| **RTX 3080** | 35 FPS | 18 FPS | 4GB |
| **RTX 3070** | 30 FPS | 15 FPS | 3GB |
| **RTX 2070** | 20 FPS | 10 FPS | 2GB |

---

## 🌍 Language Selection

- 🌍 **English** (current)
- 🇷🇺 **[Русский](../Quick-Start)**

---

*[⬅️ Home](Home) | [Installation ➡️](Installation)*
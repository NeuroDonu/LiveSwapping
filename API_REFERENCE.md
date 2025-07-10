# LiveSwapping API Reference

Quick reference for LiveSwapping public APIs and functions.

## Entry Points

```python
# Main application
from liveswapping.run import run, main, start_gui
run()                    # Start application
main()                   # Alias for run()
start_gui()             # Start GUI mode

# Command line modes
python -m liveswapping.run image --source src.jpg --target tgt.jpg --modelPath model.pth
python -m liveswapping.run video --source src.jpg --target_video vid.mp4 --modelPath model.pth  
python -m liveswapping.run realtime --source src.jpg --modelPath model.pth
```

## Core Processing

### Real-time Processing
```python
from liveswapping.core.realtime import main, parse_arguments, cli

# Parse arguments
args = parse_arguments(['--source', 'face.jpg', '--modelPath', 'model.pth'])

# Run processing
main(args)              # With parsed args
main()                  # Parse sys.argv
cli(['--source', 'face.jpg', '--modelPath', 'model.pth'])  # Direct CLI
```

### Video Processing
```python
from liveswapping.core.video import main, parse_arguments, cli

args = parse_arguments([
    '--source', 'face.jpg',
    '--target_video', 'video.mp4',
    '--modelPath', 'model.pth',
    '--upscale', '2'
])
main(args)
```

### Image Utilities
```python
from liveswapping.core.image_utils import *

# Face processing
face_tensor = postprocess_face(tensor)
blob = getBlob(image, (128, 128))
latent = getLatent(face_object)

# Image blending
result = blend_swapped_image(swapped_face, target_image, transform_matrix)
result = blend_swapped_image_gpu(swapped_face, target_image, transform_matrix)  # GPU version
```

## AI Models

### Model Loading
```python
from liveswapping.ai_models.models import *

# Load models
model = load_model("reswapper128", use_tensorrt=True)
model = load_model("inswapper128", provider_type="cuda") 
model = load_model("/path/to/model.pth", use_tensorrt=False)

# Model registry
models = list_available_models()
model_type = get_model_type(Path("model.pth"))
provider = get_optimal_provider()

# ONNX sessions
session = create_session("model.onnx", provider="cuda")
```

### Available Models
- `reswapper128`: StyleTransfer 128x128, TensorRT support
- `reswapper256`: StyleTransfer 256x256, TensorRT support  
- `inswapper128`: InSwapper 128x128, ONNX Runtime

### Providers
- `cuda`: NVIDIA GPU (with TensorRT)
- `directml`: AMD GPU
- `openvino`: Intel GPU/CPU
- `cpu`: CPU only

## GUI Components

```python
# Real-time GUI
from liveswapping.gui.realtime_gui import main
main()

# Video processing GUI  
from liveswapping.gui.video_gui import main
main()
```

## Utilities

### Upscalers
```python
from liveswapping.utils.upscalers import *

# GFPGAN upscaler
upscaler = GFPGANUpscaler(use_tensorrt=True)
enhanced = upscaler.upscale(image)
cropped_faces, restored_img, restored_faces = upscaler.enhance(image)

# Factory function
upscaler = create_optimized_gfpgan(use_tensorrt=True)

# RealESRGAN upscaler
upscaler = RealESRGANUpscaler(scale=2, tile=400)
upscaled = upscaler.upscale(image)
```

### GPU Acceleration
```python
from liveswapping.utils.gpu_utils import *

# Array management
manager = GPUArrayManager(use_cupy=True)
gpu_array = manager.to_gpu(numpy_array)
result = manager.to_cpu(gpu_result)
manager.synchronize()

# Histogram matching
matched = accelerated_histogram_matching(source_img, target_img, alpha=0.5)

# System info
print_gpu_info()
config = get_optimal_config()
providers = get_provider_info()
```

### Adaptive CuPy
```python
from liveswapping.utils.adaptive_cupy import *

# Create processor for image size
processor = create_adaptive_processor(1080)  # For 1080p

# Color transfer and blending
color_transfer = AdaptiveColorTransfer(processor)
blending = AdaptiveBlending(processor)

result = color_transfer.apply_color_transfer_adaptive(source_path, target, face_analysis)
result = blending.blend_swapped_image_adaptive(swapped_face, target, transform_matrix)
```

## Common Patterns

### Basic Face Swap
```python
from liveswapping.ai_models.models import load_model
import cv2

model = load_model("reswapper128", use_tensorrt=True)
source = cv2.imread("source.jpg")
target = cv2.imread("target.jpg")
result = model.swap_face(source, target)  # Simplified example
cv2.imwrite("result.jpg", result)
```

### Optimized Processing
```python
# Load with best provider
provider = get_optimal_provider()
model = load_model("reswapper128", provider_type=provider, use_tensorrt=True)

# GPU acceleration
gpu_manager = GPUArrayManager(use_cupy=True)
upscaler = create_optimized_gfpgan(use_tensorrt=True)

# Process
gpu_array = gpu_manager.to_gpu(image)
# ... GPU processing ...
result = gpu_manager.to_cpu(gpu_result)
enhanced = upscaler.upscale(result)
```

### Error Handling
```python
try:
    model = load_model("reswapper256", provider_type="cuda")
except RuntimeError as e:
    if "out of memory" in str(e):
        model = load_model("reswapper256", provider_type="cpu")

# Automatic basicsr fix
from liveswapping.run import check_and_fix_basicsr
check_and_fix_basicsr()
```

## Performance Tips

- **Enable TensorRT**: `use_tensorrt=True` for 3x speedup on NVIDIA GPUs
- **Choose optimal provider**: Use `get_optimal_provider()` for automatic detection
- **Use CuPy acceleration**: `GPUArrayManager(use_cupy=True)` for numpy operations
- **Optimize batch size**: Check `get_optimal_config()['recommended_batch_size']`
- **Memory management**: Use CPU provider for large models on limited VRAM

## Command Line Arguments

### Realtime
- `--source`: Source face image (required)
- `--modelPath`: Model file path (required)  
- `--resolution`: Face crop resolution (default: 128)
- `--obs`: Send to OBS virtual camera
- `--mouth_mask`: Retain target mouth
- `--delay`: Delay in milliseconds
- `--enhance_res`: Use 1920x1080 webcam resolution

### Video
- `--source`: Source face image (required)
- `--target_video`: Target video (required)
- `--modelPath`: Model file path (required)
- `--upscale`: Upscaling factor (default: 2)
- `--bg_upsampler`: Background upsampler (default: "realesrgan")
- `--weight`: Blending weight (default: 0.5)
- `--mouth_mask`: Retain target mouth

### Image
- `--source`: Source image path (required)
- `--target`: Target image path (required)
- `--modelPath`: Model file path (required)
- `--output`: Output path (required)

## Performance Benchmarks (RTX 4090)

| Component | Standard | Optimized | Speedup |
|-----------|----------|-----------|---------|
| reswapper128 | 15 FPS | 45 FPS | 3.0x |
| reswapper256 | 8 FPS | 25 FPS | 3.1x |
| GFPGAN | 2.5 FPS | 7 FPS | 2.8x |
| RealESRGAN | 1.8 FPS | 5.2 FPS | 2.9x |
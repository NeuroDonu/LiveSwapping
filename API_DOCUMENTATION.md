# LiveSwapping API Documentation

This document provides comprehensive documentation for all public APIs, functions, and components in the LiveSwapping real-time face swapping system.

## Table of Contents

1. [Main Entry Points](#main-entry-points)
2. [Core Processing APIs](#core-processing-apis)
3. [AI Models API](#ai-models-api)
4. [GUI Components](#gui-components)
5. [Utility APIs](#utility-apis)
6. [Usage Examples](#usage-examples)
7. [Performance Optimization](#performance-optimization)

---

## Main Entry Points

### `liveswapping.run`

Main application entry point with command-line interface support.

#### Functions

##### `run()`
Main entry point for the LiveSwapping application.

```python
from liveswapping.run import run

# Start GUI mode
run()

# Command line usage:
# python -m liveswapping.run image --source source.jpg --target target.jpg --modelPath model.pth --output result.jpg
# python -m liveswapping.run video --source source.jpg --target_video video.mp4 --modelPath model.pth --output output.mp4
# python -m liveswapping.run realtime --source source.jpg --modelPath model.pth --camera_id 0
```

##### `main()`
Alias for `run()` function.

##### `start_gui()`
Entry point specifically for GUI mode.

```python
from liveswapping.run import start_gui

start_gui()
```

---

## Core Processing APIs

### `liveswapping.core.realtime`

Real-time face swapping with webcam support.

#### Functions

##### `main(parsed_args=None)`
Main real-time processing function.

**Parameters:**
- `parsed_args`: Optional pre-parsed arguments from `parse_arguments()`

**Example:**
```python
from liveswapping.core.realtime import main, parse_arguments

# Using command line arguments
args = parse_arguments(['--source', 'source.jpg', '--modelPath', 'model.pth'])
main(args)

# Direct function call
main()  # Will parse sys.argv
```

##### `parse_arguments(argv=None)`
Parses command-line arguments for real-time processing.

**Parameters:**
- `argv`: Optional argument list. If None, uses `sys.argv[1:]`

**Returns:**
- `argparse.Namespace`: Parsed arguments

**Available Arguments:**
- `--source`: Path to source face image (required)
- `--modelPath`: Path to the trained face swap model (required)
- `--resolution`: Resolution of face crop (default: 128)
- `--face_attribute_direction`: Path to face attribute direction.npy
- `--face_attribute_steps`: Amount to move in attribute direction (default: 0.0)
- `--obs`: Send frames to OBS virtual camera
- `--mouth_mask`: Retain target mouth
- `--delay`: Delay time in milliseconds (default: 0)
- `--fps_delay`: Show FPS and delay time on screen
- `--enhance_res`: Increase webcam resolution to 1920x1080

##### `cli(argv=None)`
Command-line interface wrapper.

**Example:**
```python
from liveswapping.core.realtime import cli

# Run with custom arguments
cli(['--source', 'face.jpg', '--modelPath', 'model.pth', '--obs'])
```

#### Helper Functions

##### `load_model(model_path)`
Loads a face swap model with TensorRT optimization.

**Parameters:**
- `model_path`: Path to model file

**Returns:**
- Loaded model object with TensorRT optimization if available

##### `create_source_latent(source_image, direction_path=None, steps=0.0)`
Creates latent representation of source face.

**Parameters:**
- `source_image`: Source face image
- `direction_path`: Optional path to attribute direction file
- `steps`: Steps to move in attribute direction

**Returns:**
- Latent representation array or None if no face detected

### `liveswapping.core.video`

Video processing for offline face swapping.

#### Functions

##### `main(parsed_args=None)`
Main video processing function with enhancement and upscaling.

**Parameters:**
- `parsed_args`: Optional pre-parsed arguments

**Example:**
```python
from liveswapping.core.video import main, parse_arguments

args = parse_arguments([
    '--source', 'source.jpg',
    '--target_video', 'input.mp4', 
    '--modelPath', 'model.pth',
    '--upscale', '2'
])
main(args)
```

##### `parse_arguments(argv=None)`
Parses video processing arguments.

**Available Arguments:**
- `--source`: Path to source face image (required)
- `--target_video`: Path to target video (required)
- `--modelPath`: Path to face swap model (required)
- `--resolution`: Face crop resolution (default: 128)
- `--mouth_mask`: Retain target mouth
- `--upscale`: Final upsampling scale (default: 2)
- `--bg_upsampler`: Background upsampler type (default: "realesrgan")
- `--bg_tile`: Tile size for background sampler (default: 400)
- `--weight`: Adjustable weights (default: 0.5)
- `--std`: Standard deviation for noise (default: 1)
- `--blur`: Blur amount (default: 1)

### `liveswapping.core.image_utils`

Core image processing utilities.

#### Functions

##### `postprocess_face(face_tensor)`
Converts face tensor to image format.

**Parameters:**
- `face_tensor`: PyTorch tensor of face

**Returns:**
- `np.ndarray`: BGR image array

##### `getBlob(aimg, input_size=(128, 128))`
Creates DNN blob from image.

**Parameters:**
- `aimg`: Input image
- `input_size`: Target size tuple

**Returns:**
- DNN blob for model input

##### `getLatent(source_face)`
Extracts latent representation from face.

**Parameters:**
- `source_face`: Face object with embedding

**Returns:**
- Normalized latent array

##### `blend_swapped_image(swapped_face, target_image, M)`
Blends swapped face with target image.

**Parameters:**
- `swapped_face`: Swapped face image
- `target_image`: Target image
- `M`: Affine transformation matrix

**Returns:**
- Blended result image

##### `blend_swapped_image_gpu(swapped_face, target_image, M)`
GPU-accelerated version of image blending.

**Parameters:**
- Same as `blend_swapped_image`

**Returns:**
- GPU-accelerated blended result

---

## AI Models API

### `liveswapping.ai_models.models`

Unified model loading and management system.

#### Functions

##### `load_model(name, use_tensorrt=True, provider_type=None, **kwargs)`
Unified model loader with optimization support.

**Parameters:**
- `name`: Model name from registry or file path
- `use_tensorrt`: Enable torch-tensorrt optimization for PyTorch models
- `provider_type`: ONNX provider type ('cuda', 'directml', 'openvino', 'cpu')
- `**kwargs`: Additional model parameters

**Returns:**
- Loaded and optimized model object

**Example:**
```python
from liveswapping.ai_models.models import load_model

# Load from registry with TensorRT optimization
model = load_model("reswapper128", use_tensorrt=True)

# Load specific provider
model = load_model("inswapper128", provider_type="cuda")

# Load from file path
model = load_model("path/to/model.pth", use_tensorrt=False)
```

##### `list_available_models()`
Returns information about all available models.

**Returns:**
- `Dict[str, Dict[str, Any]]`: Model registry with metadata

**Example:**
```python
models = list_available_models()
for name, info in models.items():
    print(f"{name}: {info['description']}")
```

##### `get_model_type(model_path)`
Determines model type from file extension.

**Parameters:**
- `model_path`: Path to model file

**Returns:**
- `str`: Model type ('dfm', 'inswapper', 'styletransfer')

##### `create_session(model_path, provider=None)`
Creates ONNX Runtime session with optimal providers.

**Parameters:**
- `model_path`: Path to ONNX model
- `provider`: Specific provider to use

**Returns:**
- Configured ONNX Runtime session

#### Provider Management

##### `get_optimal_provider()`
Returns the best available provider for current system.

**Returns:**
- `str`: Optimal provider name

**Example:**
```python
from liveswapping.ai_models.models import get_optimal_provider

provider = get_optimal_provider()
print(f"Using provider: {provider}")  # e.g., "cuda", "directml", "cpu"
```

### Model Registry

The following models are available in the registry:

#### `reswapper128`
- **Type**: StyleTransfer
- **Resolution**: 128x128
- **Description**: Fast, good quality face swapping
- **Optimization**: TensorRT support

#### `reswapper256`
- **Type**: StyleTransfer  
- **Resolution**: 256x256
- **Description**: High quality, slower processing
- **Optimization**: TensorRT support

#### `inswapper128`
- **Type**: InSwapper
- **Resolution**: 128x128
- **Description**: Industry-standard face swapping
- **Optimization**: ONNX Runtime providers

---

## GUI Components

### `liveswapping.gui.realtime_gui`

Real-time processing GUI application.

#### Classes

##### `RealtimeGUI`
Main real-time GUI application class.

**Methods:**
- `__init__()`: Initialize GUI
- `_browse_source()`: Browse source image
- `_browse_model()`: Browse model file
- `_start()`: Start real-time processing
- `_stop()`: Stop processing
- `_change_language()`: Change interface language

**Example:**
```python
from liveswapping.gui.realtime_gui import main

# Start real-time GUI
main()
```

### `liveswapping.gui.video_gui`

Video processing GUI application.

#### Classes

##### `VideoGUI`
Main video processing GUI class.

**Methods:**
- `__init__()`: Initialize GUI
- `_browse_source()`: Browse source image
- `_browse_target()`: Browse target video
- `_browse_model()`: Browse model file
- `_start()`: Start video processing
- `_stop()`: Stop processing
- `_upscaler_toggled(checked)`: Toggle upscaler

**Example:**
```python
from liveswapping.gui.video_gui import main

# Start video processing GUI
main()
```

---

## Utility APIs

### `liveswapping.utils.upscalers`

Image upscaling and enhancement utilities.

#### Classes

##### `GFPGANUpscaler`
GFPGAN-based face restoration with TensorRT optimization.

**Constructor:**
```python
GFPGANUpscaler(model_path=None, use_tensorrt=True, bg_upsampler=None)
```

**Parameters:**
- `model_path`: Path to GFPGAN model (auto-downloads if None)
- `use_tensorrt`: Enable TensorRT optimization
- `bg_upsampler`: Background upsampler instance

**Methods:**

###### `upscale(image)`
Upscales and enhances image.

**Parameters:**
- `image`: Input image array

**Returns:**
- Enhanced image array

###### `enhance(image, **kwargs)`
Advanced enhancement with additional options.

**Parameters:**
- `image`: Input image
- `**kwargs`: Additional enhancement parameters

**Returns:**
- Tuple of (cropped_faces, restored_img, restored_faces)

**Example:**
```python
from liveswapping.utils.upscalers import GFPGANUpscaler

# Create optimized upscaler
upscaler = GFPGANUpscaler(use_tensorrt=True)

# Enhance image
enhanced = upscaler.upscale(image)

# Advanced enhancement
cropped_faces, restored_img, restored_faces = upscaler.enhance(
    image, 
    has_aligned=False,
    only_center_face=True,
    paste_back=True,
    weight=0.5
)
```

##### `RealESRGANUpscaler`
RealESRGAN-based upscaling with TensorRT optimization.

**Constructor:**
```python
RealESRGANUpscaler(model_path=None, use_tensorrt=True, scale=2, tile=400)
```

#### Factory Functions

##### `create_optimized_gfpgan(model_path=None, use_tensorrt=True, bg_upsampler=None)`
Factory function for creating optimized GFPGAN.

**Returns:**
- Configured GFPGANUpscaler instance

##### `ensure_gfpgan_model()`
Ensures GFPGAN model is downloaded and available.

**Returns:**
- Path to GFPGAN model file

### `liveswapping.utils.gpu_utils`

GPU acceleration utilities for numpy operations.

#### Classes

##### `GPUArrayManager`
Manages numpy/CuPy arrays for optimal GPU performance.

**Constructor:**
```python
GPUArrayManager(use_cupy=True, verbose=False)
```

**Methods:**

###### `to_gpu(array)`
Converts numpy array to CuPy array if available.

###### `to_cpu(array)`
Converts CuPy array back to numpy array.

###### `synchronize()`
Synchronizes GPU operations.

**Example:**
```python
from liveswapping.utils.gpu_utils import GPUArrayManager

manager = GPUArrayManager(use_cupy=True)

# Move array to GPU
gpu_array = manager.to_gpu(numpy_array)

# Process on GPU
result_gpu = gpu_operation(gpu_array)

# Move back to CPU
result = manager.to_cpu(result_gpu)
manager.synchronize()
```

#### Functions

##### `accelerated_histogram_matching(source_image, target_image, alpha=0.5, use_gpu=True)`
GPU-accelerated histogram matching using CuPy.

**Parameters:**
- `source_image`: Source image array
- `target_image`: Target image array  
- `alpha`: Blending factor (0.0-1.0)
- `use_gpu`: Whether to use GPU acceleration

**Returns:**
- Histogram matched image

##### `get_optimal_config()`
Gets optimal configuration for current system.

**Returns:**
- Dictionary with optimal settings

##### `print_gpu_info()`
Prints detailed GPU acceleration status.

##### `get_provider_info()`
Returns information about available providers.

**Returns:**
- List of provider dictionaries with availability info

**Example:**
```python
from liveswapping.utils.gpu_utils import (
    accelerated_histogram_matching,
    print_gpu_info,
    get_optimal_config
)

# Print system info
print_gpu_info()

# Get optimal settings
config = get_optimal_config()
print(f"Recommended batch size: {config['recommended_batch_size']}")

# GPU-accelerated histogram matching
matched = accelerated_histogram_matching(
    source_image, 
    target_image, 
    alpha=0.7,
    use_gpu=True
)
```

### `liveswapping.utils.adaptive_cupy`

Adaptive CuPy acceleration based on image size.

#### Classes

##### `AdaptiveCuPyProcessor`
Automatically chooses GPU/CPU based on optimal performance.

##### `AdaptiveColorTransfer`
GPU-accelerated color transfer with automatic CPU fallback.

##### `AdaptiveBlending`
GPU-accelerated image blending with size-based optimization.

#### Functions

##### `create_adaptive_processor(image_height)`
Creates adaptive processor based on image size.

**Parameters:**
- `image_height`: Height of images to be processed

**Returns:**
- Configured AdaptiveCuPyProcessor

---

## Usage Examples

### Basic Face Swap

```python
from liveswapping.ai_models.models import load_model
from liveswapping.core import image_utils as Image
import cv2

# Load model with TensorRT optimization
model = load_model("reswapper128", use_tensorrt=True)

# Load images
source_img = cv2.imread("source.jpg")
target_img = cv2.imread("target.jpg") 

# Process faces (simplified example)
# In practice, you'd use face detection and alignment
result = model.swap_face(source_img, target_img)

cv2.imwrite("result.jpg", result)
```

### Real-time Processing

```python
from liveswapping.core.realtime import main, parse_arguments

# Set up arguments
args = parse_arguments([
    '--source', 'my_face.jpg',
    '--modelPath', 'models/reswapper128.pth',
    '--obs',  # Send to OBS virtual camera
    '--enhance_res'  # Use high resolution
])

# Start real-time processing
main(args)
```

### Video Processing with Upscaling

```python
from liveswapping.core.video import main, parse_arguments

args = parse_arguments([
    '--source', 'face.jpg',
    '--target_video', 'input_video.mp4',
    '--modelPath', 'models/reswapper256.pth',
    '--upscale', '2',
    '--bg_upsampler', 'realesrgan',
    '--weight', '0.8'
])

main(args)
```

### Custom Model Loading

```python
from liveswapping.ai_models.models import load_model, get_optimal_provider

# Detect best provider
provider = get_optimal_provider()
print(f"Using provider: {provider}")

# Load model with specific provider
model = load_model(
    "inswapper128", 
    provider_type=provider,
    use_tensorrt=True
)

# Use model for inference
# ... processing code ...
```

### GPU-Accelerated Processing

```python
from liveswapping.utils.gpu_utils import (
    GPUArrayManager, 
    accelerated_histogram_matching,
    print_gpu_info
)

# Check GPU status
print_gpu_info()

# Set up GPU acceleration
gpu_manager = GPUArrayManager(use_cupy=True)

# Process with GPU acceleration
source_gpu = gpu_manager.to_gpu(source_image)
result_gpu = gpu_process(source_gpu)
result = gpu_manager.to_cpu(result_gpu)

# Color matching with GPU
matched = accelerated_histogram_matching(
    source_image, 
    target_image,
    alpha=0.6,
    use_gpu=True
)
```

### Enhanced Upscaling

```python
from liveswapping.utils.upscalers import create_optimized_gfpgan

# Create optimized upscaler
upscaler = create_optimized_gfpgan(
    use_tensorrt=True,
    bg_upsampler=None  # Can add RealESRGAN here
)

# Enhance face
enhanced_faces, full_result, face_results = upscaler.enhance(
    image,
    has_aligned=False,
    only_center_face=False,
    paste_back=True,
    weight=0.5
)
```

---

## Performance Optimization

### TensorRT Optimization

TensorRT provides 3x speedup for PyTorch models:

```python
# Enable TensorRT for maximum performance
model = load_model("reswapper128", use_tensorrt=True)

# Disable for debugging
model = load_model("reswapper128", use_tensorrt=False)
```

### Provider Selection

Choose optimal provider for your hardware:

```python
# Automatic detection (recommended)
model = load_model("inswapper128")

# Force specific provider
model = load_model("inswapper128", provider_type="cuda")     # NVIDIA GPU
model = load_model("inswapper128", provider_type="directml") # AMD GPU  
model = load_model("inswapper128", provider_type="openvino") # Intel
model = load_model("inswapper128", provider_type="cpu")      # CPU only
```

### CuPy Acceleration

Use CuPy for numpy operations acceleration:

```python
from liveswapping.utils.adaptive_cupy import create_adaptive_processor

# Create processor for your image size
processor = create_adaptive_processor(1080)  # For 1080p images

# Use adaptive color transfer and blending
color_transfer = AdaptiveColorTransfer(processor)
blending = AdaptiveBlending(processor)
```

### Memory Optimization

For limited GPU memory:

```python
# Use CPU provider for large models
model = load_model("reswapper256", provider_type="cpu")

# Reduce batch size
config = get_optimal_config()
batch_size = config['recommended_batch_size']

# Use mixed precision if supported
if config['use_mixed_precision']:
    # Enable FP16 processing
    pass
```

### Performance Benchmarks

Expected performance on RTX 4090:

| Component | Without Optimization | With Optimization | Speedup |
|-----------|---------------------|-------------------|---------|
| reswapper128 | ~15 FPS | ~45 FPS | **3.0x** |
| reswapper256 | ~8 FPS | ~25 FPS | **3.1x** |
| GFPGAN | ~2.5 FPS | ~7 FPS | **2.8x** |
| RealESRGAN | ~1.8 FPS | ~5.2 FPS | **2.9x** |

---

## Error Handling

### Common Issues

#### CUDA Out of Memory
```python
try:
    model = load_model("reswapper256", provider_type="cuda")
except RuntimeError as e:
    if "out of memory" in str(e):
        # Fallback to CPU
        model = load_model("reswapper256", provider_type="cpu")
```

#### TensorRT Compilation Failed
```python
# Automatic fallback
model = load_model("reswapper128", use_tensorrt=True)
# Will automatically disable TensorRT if compilation fails
```

#### basicsr Import Error
```python
# Automatic fix available
from liveswapping.run import check_and_fix_basicsr

if check_and_fix_basicsr():
    print("basicsr fixed successfully")
```

### Debugging

Enable verbose logging:

```python
# For model loading
model = load_model("reswapper128", use_tensorrt=True, verbose=True)

# For GPU operations
gpu_manager = GPUArrayManager(use_cupy=True, verbose=True)
```

---

This documentation covers all major public APIs and components in the LiveSwapping system. For additional examples and advanced usage, refer to the source code and example scripts.
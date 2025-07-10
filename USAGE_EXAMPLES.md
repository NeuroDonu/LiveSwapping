# LiveSwapping Usage Examples & Tutorials

This document provides step-by-step examples and tutorials for using LiveSwapping effectively.

## Table of Contents

1. [Quick Start Examples](#quick-start-examples)
2. [Real-time Processing](#real-time-processing)
3. [Video Processing](#video-processing)
4. [Programmatic Usage](#programmatic-usage)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting Guide](#troubleshooting-guide)

---

## Quick Start Examples

### 1. Basic Image Face Swap (Command Line)

```bash
# Simple image face swap
python -m liveswapping.run image \
    --source my_face.jpg \
    --target celebrity.jpg \
    --modelPath models/reswapper128.pth \
    --output result.jpg
```

### 2. Real-time Webcam Face Swap

```bash
# Real-time face swap with webcam
python -m liveswapping.run realtime \
    --source my_face.jpg \
    --modelPath models/reswapper128.pth \
    --obs  # Send to OBS virtual camera
```

### 3. Video Face Swap with Upscaling

```bash
# Process video with enhancement
python -m liveswapping.run video \
    --source actor_face.jpg \
    --target_video input_video.mp4 \
    --modelPath models/reswapper256.pth \
    --upscale 2 \
    --bg_upsampler realesrgan
```

---

## Real-time Processing

### Basic Real-time Setup

```python
from liveswapping.core.realtime import main, parse_arguments

# Configure real-time processing
args = parse_arguments([
    '--source', 'reference_face.jpg',
    '--modelPath', 'models/reswapper128.pth',
    '--resolution', '128',
    '--delay', '0'
])

# Start processing
main(args)
```

### Real-time with OBS Integration

```python
from liveswapping.core.realtime import parse_arguments, main

# Set up for streaming
args = parse_arguments([
    '--source', 'streamer_face.jpg',
    '--modelPath', 'models/reswapper128.pth',
    '--obs',  # Send to OBS virtual camera
    '--enhance_res',  # Use 1920x1080 resolution
    '--fps_delay'  # Show performance info
])

main(args)
```

### Real-time with Face Attributes

```python
# Using face attribute modification
args = parse_arguments([
    '--source', 'base_face.jpg',
    '--modelPath', 'models/reswapper128.pth',
    '--face_attribute_direction', 'smile_direction.npy',
    '--face_attribute_steps', '2.0',  # Increase smile
    '--mouth_mask'  # Preserve original mouth
])

main(args)
```

### Real-time Performance Optimization

```python
from liveswapping.ai_models.models import get_optimal_provider
from liveswapping.core.realtime import parse_arguments, main

# Detect optimal provider
provider = get_optimal_provider()
print(f"Using optimal provider: {provider}")

# Configure for maximum performance
args = parse_arguments([
    '--source', 'face.jpg',
    '--modelPath', 'models/reswapper128.pth',  # Use 128 for speed
    '--resolution', '128',
    '--delay', '0'
])

main(args)
```

---

## Video Processing

### Basic Video Processing

```python
from liveswapping.core.video import main, parse_arguments

args = parse_arguments([
    '--source', 'actor_face.jpg',
    '--target_video', 'scene.mp4',
    '--modelPath', 'models/reswapper256.pth',
    '--resolution', '256'
])

main(args)
```

### High-Quality Video with Enhancement

```python
# Maximum quality video processing
args = parse_arguments([
    '--source', 'reference.jpg',
    '--target_video', 'input.mp4',
    '--modelPath', 'models/reswapper256.pth',
    '--upscale', '2',
    '--bg_upsampler', 'realesrgan',
    '--bg_tile', '400',
    '--weight', '0.8',  # Higher weight for better blending
    '--std', '0.5',     # Lower noise
    '--blur', '1'       # Minimal blur
])

main(args)
```

### Batch Video Processing

```python
import os
from liveswapping.core.video import main, parse_arguments

# Process multiple videos
video_files = ['video1.mp4', 'video2.mp4', 'video3.mp4']
source_face = 'celebrity_face.jpg'
model_path = 'models/reswapper256.pth'

for video_file in video_files:
    output_file = f"swapped_{os.path.basename(video_file)}"
    
    args = parse_arguments([
        '--source', source_face,
        '--target_video', video_file,
        '--modelPath', model_path,
        '--upscale', '2'
    ])
    
    print(f"Processing {video_file}...")
    main(args)
    
    # Rename output (video processing saves as output.mp4)
    if os.path.exists('output.mp4'):
        os.rename('output.mp4', output_file)
        print(f"Saved as {output_file}")
```

---

## Programmatic Usage

### Custom Face Swap Pipeline

```python
import cv2
import numpy as np
from liveswapping.ai_models.models import load_model
from liveswapping.core.image_utils import *
from insightface.app import FaceAnalysis

# Initialize face analysis
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(512, 512))

# Load optimized model
model = load_model("reswapper128", use_tensorrt=True)

def swap_faces_in_image(source_path, target_path, output_path):
    """Custom face swap function."""
    
    # Load images
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)
    
    # Detect faces
    source_faces = face_app.get(source_img)
    target_faces = face_app.get(target_img)
    
    if len(source_faces) == 0:
        raise ValueError("No face found in source image")
    if len(target_faces) == 0:
        raise ValueError("No face found in target image")
    
    # Get source latent
    source_latent = getLatent(source_faces[0])
    
    # Process target face
    target_face = target_faces[0]
    
    # Align and process face
    from liveswapping.core.face_align import norm_crop2
    aligned_face, M = norm_crop2(target_img, target_face.kps, 128)
    face_blob = getBlob(aligned_face, (128, 128))
    
    # Swap face
    if hasattr(model, 'swap_face'):
        swapped_face = model.swap_face(aligned_face, source_latent)
    else:
        # For StyleTransfer models
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target_tensor = torch.from_numpy(face_blob).to(device)
        source_tensor = torch.from_numpy(source_latent).to(device)
        
        with torch.no_grad():
            swapped_tensor = model(target_tensor, source_tensor)
        
        swapped_face = postprocess_face(swapped_tensor)
    
    # Blend result
    result = blend_swapped_image_gpu(swapped_face, target_img, M)
    
    # Save result
    cv2.imwrite(output_path, result)
    print(f"Face swap completed: {output_path}")

# Usage
swap_faces_in_image("source.jpg", "target.jpg", "result.jpg")
```

### Multi-Face Processing

```python
from liveswapping.ai_models.models import load_model
from insightface.app import FaceAnalysis
import cv2

class MultiFaceSwapper:
    def __init__(self, model_name="reswapper128"):
        self.model = load_model(model_name, use_tensorrt=True)
        self.face_app = FaceAnalysis(name="buffalo_l")
        self.face_app.prepare(ctx_id=0, det_size=(512, 512))
    
    def process_image_multi_face(self, source_paths, target_path, output_path):
        """Swap multiple faces in a single image."""
        
        target_img = cv2.imread(target_path)
        target_faces = self.face_app.get(target_img)
        
        result_img = target_img.copy()
        
        # Process each target face with corresponding source
        for i, (source_path, target_face) in enumerate(zip(source_paths, target_faces)):
            if i >= len(source_paths):
                break
                
            source_img = cv2.imread(source_path)
            source_faces = self.face_app.get(source_img)
            
            if len(source_faces) == 0:
                print(f"No face found in {source_path}, skipping")
                continue
            
            # Swap this face
            source_latent = getLatent(source_faces[0])
            # ... face swapping logic ...
            # result_img = swap_single_face(result_img, target_face, source_latent)
        
        cv2.imwrite(output_path, result_img)

# Usage
swapper = MultiFaceSwapper()
swapper.process_image_multi_face(
    ["person1.jpg", "person2.jpg"], 
    "group_photo.jpg", 
    "swapped_group.jpg"
)
```

### Batch Processing with Progress

```python
from liveswapping.ai_models.models import load_model
import os
from tqdm import tqdm

class BatchProcessor:
    def __init__(self, model_name="reswapper128"):
        self.model = load_model(model_name, use_tensorrt=True)
    
    def process_directory(self, source_path, input_dir, output_dir):
        """Process all images in a directory."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(image_extensions)]
        
        # Process with progress bar
        for filename in tqdm(image_files, desc="Processing images"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"swapped_{filename}")
            
            try:
                # Use your custom swap function
                swap_faces_in_image(source_path, input_path, output_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Usage
processor = BatchProcessor()
processor.process_directory("celebrity.jpg", "input_photos/", "output_photos/")
```

---

## Performance Optimization

### GPU Memory Optimization

```python
from liveswapping.ai_models.models import load_model, get_optimal_config
from liveswapping.utils.gpu_utils import print_gpu_info, get_optimal_config

# Check system capabilities
print_gpu_info()
config = get_optimal_config()

# Optimize based on available memory
if config['memory_gb'] >= 8:
    # High-end GPU
    model = load_model("reswapper256", use_tensorrt=True, provider_type="cuda")
    batch_size = 4
elif config['memory_gb'] >= 4:
    # Mid-range GPU
    model = load_model("reswapper128", use_tensorrt=True, provider_type="cuda")
    batch_size = 2
else:
    # Low memory or CPU
    model = load_model("reswapper128", use_tensorrt=False, provider_type="cpu")
    batch_size = 1

print(f"Using batch size: {batch_size}")
```

### CuPy Acceleration Setup

```python
from liveswapping.utils.gpu_utils import GPUArrayManager
from liveswapping.utils.adaptive_cupy import create_adaptive_processor
import numpy as np

# Set up GPU acceleration
gpu_manager = GPUArrayManager(use_cupy=True, verbose=True)

# Create adaptive processor for your image size
processor = create_adaptive_processor(1080)  # For 1080p images

def optimized_image_processing(image):
    """Example of optimized image processing."""
    
    # Move to GPU
    gpu_image = gpu_manager.to_gpu(image)
    
    # Perform GPU operations
    # ... your processing code ...
    
    # Move back to CPU
    result = gpu_manager.to_cpu(gpu_image)
    gpu_manager.synchronize()
    
    return result
```

### Upscaler Optimization

```python
from liveswapping.utils.upscalers import create_optimized_gfpgan

# Create optimized upscaler
upscaler = create_optimized_gfpgan(
    use_tensorrt=True,  # Enable TensorRT for 3x speedup
    bg_upsampler=None   # Can add RealESRGAN background upsampler
)

def enhance_image_optimized(image_path, output_path):
    """Optimized image enhancement."""
    
    image = cv2.imread(image_path)
    
    # Enhance with optimized GFPGAN
    _, restored_img, _ = upscaler.enhance(
        image,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=0.5
    )
    
    cv2.imwrite(output_path, restored_img)

# Usage
enhance_image_optimized("low_quality.jpg", "enhanced.jpg")
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. CUDA Out of Memory

```python
import torch

def handle_cuda_memory():
    """Handle CUDA memory issues."""
    
    try:
        model = load_model("reswapper256", provider_type="cuda")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("CUDA out of memory, switching to CPU")
            model = load_model("reswapper256", provider_type="cpu")
        else:
            raise e
    
    return model

# Also clear cache between operations
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

#### 2. TensorRT Compilation Issues

```python
def load_model_with_fallback(model_name):
    """Load model with TensorRT fallback."""
    
    try:
        # Try with TensorRT first
        model = load_model(model_name, use_tensorrt=True)
        print("TensorRT optimization successful")
    except Exception as e:
        print(f"TensorRT compilation failed: {e}")
        print("Falling back to standard PyTorch model")
        model = load_model(model_name, use_tensorrt=False)
    
    return model
```

#### 3. basicsr Import Errors

```python
def fix_basicsr_automatically():
    """Automatically fix basicsr import issues."""
    
    from liveswapping.run import check_and_fix_basicsr
    
    if check_and_fix_basicsr():
        print("basicsr fixed successfully")
        return True
    else:
        print("Could not fix basicsr automatically")
        print("Try: python liveswapping/utils/fix_basicsr.py")
        return False

# Call before video processing
fix_basicsr_automatically()
```

#### 4. Model Download Issues

```python
def ensure_model_downloaded(model_name):
    """Ensure model is downloaded before use."""
    
    try:
        from liveswapping.ai_models.download_models import ensure_model
        model_path = ensure_model(model_name)
        print(f"Model {model_name} available at: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        print("Check internet connection and disk space")
        return None

# Usage
model_path = ensure_model_downloaded("reswapper128")
if model_path:
    model = load_model(model_path)
```

### Performance Debugging

```python
import time
from liveswapping.utils.gpu_utils import analyze_cupy_performance

def benchmark_processing():
    """Benchmark your processing pipeline."""
    
    # Analyze CuPy performance
    analyze_cupy_performance()
    
    # Benchmark model loading
    start_time = time.time()
    model = load_model("reswapper128", use_tensorrt=True)
    load_time = time.time() - start_time
    print(f"Model loading time: {load_time:.2f}s")
    
    # Benchmark inference
    import numpy as np
    dummy_input = np.random.rand(1, 3, 128, 128).astype(np.float32)
    dummy_latent = np.random.rand(1, 512).astype(np.float32)
    
    start_time = time.time()
    for _ in range(10):
        # Simulate model inference
        pass
    inference_time = (time.time() - start_time) / 10
    print(f"Average inference time: {inference_time:.4f}s")

benchmark_processing()
```

### Error Logging Setup

```python
import logging

def setup_detailed_logging():
    """Set up detailed logging for debugging."""
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('liveswapping.log'),
            logging.StreamHandler()
        ]
    )
    
    # Enable verbose model loading
    model = load_model("reswapper128", use_tensorrt=True, verbose=True)
    
    return model

# Usage for debugging
model = setup_detailed_logging()
```

---

This comprehensive guide covers most common usage patterns and troubleshooting scenarios for LiveSwapping. For additional help, refer to the main API documentation and source code examples.
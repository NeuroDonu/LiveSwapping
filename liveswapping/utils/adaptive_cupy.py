#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adaptive CuPy acceleration for real-time face swapping."""

import numpy as np
import cv2
from typing import Tuple, Optional, Union

# Import CuPy utilities
try:
    from liveswapping.utils.gpu_utils import (
        GPUArrayManager, 
        accelerated_histogram_matching, 
        CUPY_AVAILABLE
    )
    ACCELERATION_AVAILABLE = True
except ImportError:
    ACCELERATION_AVAILABLE = False

class AdaptiveCuPyProcessor:
    """Adaptive processor that decides when to use CuPy based on frame size."""
    
    def __init__(self, 
                 min_resolution_threshold: int = 720,
                 enable_color_correction: bool = True,
                 enable_histogram_matching: bool = True):
        """
        Args:
            min_resolution_threshold: Minimum height for CuPy acceleration
            enable_color_correction: Enable CuPy for color correction
            enable_histogram_matching: Enable CuPy for histogram matching
        """
        self.min_threshold = min_resolution_threshold
        self.enable_color_correction = enable_color_correction
        self.enable_histogram_matching = enable_histogram_matching
        self.gpu_manager = None
        
        if ACCELERATION_AVAILABLE:
            self.gpu_manager = GPUArrayManager(use_cupy=True, verbose=False)
        self._logged_acceleration = False
    
    def should_use_gpu(self, frame: np.ndarray) -> bool:
        """Decide whether to use GPU acceleration based on frame size."""
        if not ACCELERATION_AVAILABLE or self.gpu_manager is None:
            return False
        
        height = frame.shape[0]
        use_gpu = height >= self.min_threshold
        
        # Log acceleration once when first used
        if use_gpu and not self._logged_acceleration:
            #print(f"[GPU] Using CuPy acceleration for {frame.shape[1]}x{frame.shape[0]} frames")
            self._logged_acceleration = True
        
        return use_gpu
    
    def get_frame_info(self, frame: np.ndarray) -> dict:
        """Get frame information for debugging."""
        height, width = frame.shape[:2]
        use_gpu = self.should_use_gpu(frame)
        
        return {
            "resolution": f"{width}x{height}",
            "height": height,
            "pixels": width * height,
            "use_gpu": use_gpu,
            "reason": "GPU" if use_gpu else f"CPU (height < {self.min_threshold})"
        }

class AdaptiveColorTransfer:
    """Adaptive color transfer with CuPy acceleration for large frames."""
    
    def __init__(self, processor: AdaptiveCuPyProcessor):
        self.processor = processor
        
    def apply_color_transfer_adaptive(self, 
                                    source_path: str, 
                                    target_frame: np.ndarray,
                                    face_analysis) -> np.ndarray:
        """Apply color transfer with adaptive CuPy acceleration."""
        
        use_gpu = (self.processor.should_use_gpu(target_frame) and 
                  self.processor.enable_color_correction)
        
        if use_gpu:
            return self._apply_color_transfer_gpu(source_path, target_frame, face_analysis)
        else:
            return self._apply_color_transfer_cpu(source_path, target_frame, face_analysis)
    
    def _apply_color_transfer_cpu(self, 
                                source_path: str, 
                                target_frame: np.ndarray,
                                face_analysis) -> np.ndarray:
        """CPU version of color transfer."""
        source = cv2.imread(source_path)
        target_faces = face_analysis.get(target_frame)
        
        if len(target_faces) == 0:
            return source
            
        x1, y1, x2, y2 = target_faces[0]["bbox"]
        target_crop = target_frame[int(y1):int(y2), int(x1):int(x2)]
        
        # Standard CPU LAB color transfer
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
        target_lab = cv2.cvtColor(target_crop, cv2.COLOR_BGR2LAB).astype("float32")
        
        source_mean, source_std = cv2.meanStdDev(source_lab)
        target_mean, target_std = cv2.meanStdDev(target_lab)
        
        source_mean = source_mean.reshape(1, 1, 3)
        source_std = source_std.reshape(1, 1, 3)
        target_mean = target_mean.reshape(1, 1, 3)
        target_std = target_std.reshape(1, 1, 3)
        
        result = (source_lab - source_mean) * (target_std / source_std) + target_mean
        return cv2.cvtColor(np.clip(result, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)
    
    def _apply_color_transfer_gpu(self, 
                                source_path: str, 
                                target_frame: np.ndarray,
                                face_analysis) -> np.ndarray:
        """GPU-accelerated version using histogram matching."""
        
        # First get CPU result as base
        cpu_result = self._apply_color_transfer_cpu(source_path, target_frame, face_analysis)
        
        # Then apply GPU histogram matching for fine-tuning
        if self.processor.enable_histogram_matching:
            try:
                # Use target face region for more accurate matching
                target_faces = face_analysis.get(target_frame)
                if len(target_faces) > 0:
                    x1, y1, x2, y2 = target_faces[0]["bbox"]
                    target_crop = target_frame[int(y1):int(y2), int(x1):int(x2)]
                    
                    # Apply histogram matching between source and target crop
                    enhanced_result = accelerated_histogram_matching(
                        cpu_result, target_crop, alpha=0.3, use_gpu=True
                    )
                    return enhanced_result
            except Exception as e:
                print(f"[WARNING] GPU histogram matching failed: {e}")
        
        return cpu_result

class AdaptiveBlending:
    """Adaptive blending with CuPy acceleration for large frames."""
    
    def __init__(self, processor: AdaptiveCuPyProcessor):
        self.processor = processor
    
    def blend_swapped_image_adaptive(self, 
                                   swapped_face: np.ndarray,
                                   target_image: np.ndarray, 
                                   M: np.ndarray) -> np.ndarray:
        """Blend with adaptive acceleration."""
        
        use_gpu = self.processor.should_use_gpu(target_image)
        
        if use_gpu:
            return self._blend_gpu_enhanced(swapped_face, target_image, M)
        else:
            return self._blend_cpu(swapped_face, target_image, M)
    
    def _blend_cpu(self, swapped_face: np.ndarray, target_image: np.ndarray, M: np.ndarray) -> np.ndarray:
        """CPU blending (use existing implementation)."""
        from liveswapping.core.image_utils import blend_swapped_image
        return blend_swapped_image(swapped_face, target_image, M)
    
    def _blend_gpu_enhanced(self, swapped_face: np.ndarray, target_image: np.ndarray, M: np.ndarray) -> np.ndarray:
        """GPU-enhanced blending with better color matching."""
        
        # First do standard GPU blend
        from liveswapping.core.image_utils import blend_swapped_image_gpu
        result = blend_swapped_image_gpu(swapped_face, target_image, M)
        
        # Apply subtle histogram matching for better color integration
        if self.processor.enable_histogram_matching:
            try:
                # Only on larger images where it's beneficial
                if target_image.shape[0] >= 720:
                    enhanced_result = accelerated_histogram_matching(
                        result, target_image, alpha=0.1, use_gpu=True
                    )
                    return enhanced_result
            except Exception as e:
                print(f"[WARNING] GPU blend enhancement failed: {e}")
        
        return result

def create_adaptive_processor(frame_height: Optional[int] = None) -> AdaptiveCuPyProcessor:
    """Create adaptive processor with smart threshold selection."""
    
    if frame_height is None:
        # Default threshold
        threshold = 720
    elif frame_height >= 1080:
        # High resolution - definitely use CuPy
        threshold = 720
    elif frame_height >= 720:
        # Medium resolution - use CuPy
        threshold = 720  
    else:
        # Low resolution - higher threshold to avoid overhead
        threshold = 1080
    
    return AdaptiveCuPyProcessor(
        min_resolution_threshold=threshold,
        enable_color_correction=True,
        enable_histogram_matching=True
    )

def benchmark_frame_processing(frame: np.ndarray, 
                             iterations: int = 5) -> dict:
    """Benchmark processing times for a frame."""
    import time
    
    processor = create_adaptive_processor()
    frame_info = processor.get_frame_info(frame)
    
    results = {
        "frame_info": frame_info,
        "cpu_time": 0.0,
        "gpu_time": 0.0,
        "speedup": 0.0
    }
    
    if not ACCELERATION_AVAILABLE:
        print(" CuPy not available for benchmarking")
        return results
    
    # Create test data
    test_source = np.random.randint(0, 256, frame.shape, dtype=np.uint8)
    
    # CPU benchmark
    start_time = time.time()
    for _ in range(iterations):
        from liveswapping.utils.gpu_utils import _cpu_histogram_matching
        _ = _cpu_histogram_matching(test_source, frame, 0.5)
    cpu_time = (time.time() - start_time) / iterations
    
    # GPU benchmark  
    start_time = time.time()
    for _ in range(iterations):
        _ = accelerated_histogram_matching(test_source, frame, 0.5, use_gpu=True)
    gpu_time = (time.time() - start_time) / iterations
    
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    
    results.update({
        "cpu_time": cpu_time,
        "gpu_time": gpu_time, 
        "speedup": speedup
    })
    
    return results

def analyze_performance():
    """Анализирует производительность CuPy vs NumPy для типичных операций."""
    if not CUPY_AVAILABLE:
        print("[INFO] CuPy not available for performance analysis")
        return
    
    import time
    
    #print("[PERF] Analyzing CuPy vs NumPy performance...")
    #print("=" * 50)
    
    # Тестовые данные
    test_sizes = [
        (512, 512, 3),   # Типичный размер лица
        (1024, 1024, 3), # Высокое разрешение
        (1920, 1080, 3)  # Full HD кадр
    ]
    
    for size in test_sizes:
        import cupy as cp
        print(f"\nTesting {size[0]}x{size[1]} arrays:")
        
        # Создание тестовых данных
        a = np.random.random(size).astype(np.float32)
        b = np.random.random(size).astype(np.float32)
        
        # CPU тест
        start = time.time()
        for _ in range(10):
            cpu_result = np.multiply(a, b)
        cpu_time = (time.time() - start) / 10
        
        # GPU тест
        a_gpu = cp.asarray(a)
        b_gpu = cp.asarray(b)
        cp.cuda.Stream.null.synchronize()
        
        start = time.time()
        for _ in range(10):
            gpu_result = cp.multiply(a_gpu, b_gpu)
            cp.cuda.Stream.null.synchronize()
        gpu_time = (time.time() - start) / 10
        
        # Результаты
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        print(f"  [CPU] CPU time: {cpu_time:.4f}s")
        print(f"  [GPU] GPU time: {gpu_time:.4f}s") 
        print(f"  [SPEEDUP] Speedup: {speedup:.2f}x")
        
        if speedup > 2.0:
            print(f"  [SUCCESS] Significant benefit!")
        elif speedup > 1.2:
            print(f"  [GOOD] Moderate benefit")
        else:
            print(f"  [LIMITED] Limited benefit")

if __name__ == "__main__":
    # Test with different frame sizes
    test_sizes = [
        (480, 640, 3),    # SD
        (720, 1280, 3),   # HD
        (1080, 1920, 3),  # Full HD
    ]
    
    #print("🔬 Adaptive CuPy Performance Analysis")
    #print("=" * 50)
    
    for size in test_sizes:
        #print(f"\n📺 Testing {size[1]}x{size[0]} frames:")
        test_frame = np.random.randint(0, 256, size, dtype=np.uint8)
        
        processor = create_adaptive_processor(size[0])
        frame_info = processor.get_frame_info(test_frame)
        
        #print(f"  Resolution: {frame_info['resolution']}")
        #print(f"  Processing: {frame_info['reason']}")
        
        if ACCELERATION_AVAILABLE and frame_info['use_gpu']:
            benchmark = benchmark_frame_processing(test_frame)
            #print(f"  CPU time: {benchmark['cpu_time']:.4f}s")
            #print(f"  GPU time: {benchmark['gpu_time']:.4f}s")
            #print(f"  Speedup: {benchmark['speedup']:.2f}x")
            
            if benchmark['speedup'] > 2.0:
                print("  Significant benefit!")
            elif benchmark['speedup'] > 1.2:
                print("  Moderate benefit")
            else:
                print("  Limited benefit")
        else:
            print("  CPU processing") 
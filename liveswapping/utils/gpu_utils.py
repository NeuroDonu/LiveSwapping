#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU utilities for accelerating numpy operations with CuPy."""

import numpy as np
import torch
from typing import Union, Optional, Any

# Try to import CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("[GPU] CuPy available - GPU acceleration for numpy operations enabled")
except ImportError:
    cp = None
    CUPY_AVAILABLE = False
    print("[CPU] CuPy not available - using CPU numpy operations")

ArrayLike = Union[np.ndarray, torch.Tensor, Any]

class GPUArrayManager:
    """Manages numpy/CuPy arrays for optimal GPU performance."""
    
    def __init__(self, use_cupy: bool = True, verbose: bool = False):
        self.use_cupy = use_cupy and CUPY_AVAILABLE
        self.device_id = 0
        self.verbose = verbose
        
        if self.use_cupy:
            cp.cuda.Device(self.device_id).use()
            if self.verbose:
                print(f"[GPU] Using CuPy on GPU {self.device_id}")
        elif self.verbose:
            print("[CPU] Using CPU numpy operations")
    
    def to_gpu(self, array: np.ndarray) -> Union[np.ndarray, Any]:
        """Convert numpy array to CuPy array if available."""
        if self.use_cupy and isinstance(array, np.ndarray):
            return cp.asarray(array)
        return array
    
    def to_cpu(self, array: Union[np.ndarray, Any]) -> np.ndarray:
        """Convert CuPy array back to numpy array."""
        if self.use_cupy and hasattr(array, 'get'):
            return array.get()
        return np.asarray(array)
    
    def synchronize(self):
        """Synchronize GPU operations."""
        if self.use_cupy:
            cp.cuda.Stream.null.synchronize()

def accelerated_histogram_matching(source_image: np.ndarray, 
                                   target_image: np.ndarray, 
                                   alpha: float = 0.5,
                                   use_gpu: bool = True) -> np.ndarray:
    """
    GPU-accelerated histogram matching using CuPy.
    
    Args:
        source_image: Source image (H, W, C) in range [0, 255]
        target_image: Target image (H, W, C) in range [0, 255] 
        alpha: Blending factor (0.0 = target only, 1.0 = matched only)
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Histogram matched image
    """
    gpu_manager = GPUArrayManager(use_gpu, verbose=False)
    
    if not gpu_manager.use_cupy:
        # Fallback to CPU numpy implementation
        return _cpu_histogram_matching(source_image, target_image, alpha)
    
    # Convert to GPU
    source_gpu = gpu_manager.to_gpu(source_image.astype(np.float32) / 255.0)
    target_gpu = gpu_manager.to_gpu(target_image.astype(np.float32) / 255.0)
    matched_gpu = cp.copy(target_gpu)
    
    # Process each channel
    for channel in range(3):
        source_channel = source_gpu[:, :, channel].flatten()
        target_channel = target_gpu[:, :, channel].flatten()
        
        # Compute histograms on GPU
        source_hist = cp.histogram(source_channel, bins=256, range=(0.0, 1.0))[0]
        target_hist = cp.histogram(target_channel, bins=256, range=(0.0, 1.0))[0]
        
        # Compute CDFs
        source_cdf = cp.cumsum(source_hist).astype(cp.float32)
        target_cdf = cp.cumsum(target_hist).astype(cp.float32)
        
        # Normalize CDFs
        source_cdf = source_cdf / source_cdf[-1] if source_cdf[-1] > 0 else source_cdf
        target_cdf = target_cdf / target_cdf[-1] if target_cdf[-1] > 0 else target_cdf
        
        # Perform histogram matching
        bin_centers = cp.linspace(0.0, 1.0, 256)
        matched_values = cp.interp(target_cdf, source_cdf, bin_centers)
        
        # Map target values to matched values
        target_indices = cp.clip((target_channel * 255).astype(cp.int32), 0, 255)
        matched_channel = matched_values[target_indices].reshape(target_gpu.shape[:2])
        
        # Update matched image
        matched_gpu[:, :, channel] = matched_channel
    
    # Blend images
    if alpha < 1.0:
        matched_gpu = alpha * matched_gpu + (1.0 - alpha) * target_gpu
    
    # Convert back to CPU and uint8
    result = gpu_manager.to_cpu(matched_gpu * 255.0).astype(np.uint8)
    gpu_manager.synchronize()
    
    return result

def _cpu_histogram_matching(source_image: np.ndarray, 
                           target_image: np.ndarray, 
                           alpha: float) -> np.ndarray:
    """CPU fallback for histogram matching."""
    source = source_image.astype(np.float32) / 255.0
    target = target_image.astype(np.float32) / 255.0
    matched = np.copy(target)
    
    for channel in range(3):
        source_channel = source[:, :, channel].flatten()
        target_channel = target[:, :, channel].flatten()
        
        # Compute histograms
        source_hist, bins = np.histogram(source_channel, bins=256, range=(0.0, 1.0))
        target_hist, _ = np.histogram(target_channel, bins=256, range=(0.0, 1.0))
        
        # Compute CDFs
        source_cdf = np.cumsum(source_hist).astype(np.float32)
        target_cdf = np.cumsum(target_hist).astype(np.float32)
        
        # Normalize CDFs
        if source_cdf[-1] > 0:
            source_cdf = source_cdf / source_cdf[-1]
        if target_cdf[-1] > 0:
            target_cdf = target_cdf / target_cdf[-1]
        
        # Perform histogram matching
        bin_centers = (bins[:-1] + bins[1:]) / 2
        matched_values = np.interp(target_cdf, source_cdf, bin_centers)
        
        # Map target values to matched values
        target_indices = np.clip((target_channel * 255).astype(np.int32), 0, 255)
        matched_channel = matched_values[target_indices].reshape(target.shape[:2])
        
        matched[:, :, channel] = matched_channel
    
    # Blend images
    if alpha < 1.0:
        matched = alpha * matched + (1.0 - alpha) * target
    
    return (matched * 255.0).astype(np.uint8)

def accelerated_face_alignment(landmarks: np.ndarray, 
                              reference_landmarks: np.ndarray,
                              use_gpu: bool = True) -> np.ndarray:
    """
    GPU-accelerated face alignment using CuPy.
    
    Args:
        landmarks: Detected face landmarks (N, 2)
        reference_landmarks: Reference landmarks (N, 2) 
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Transformation matrix
    """
    gpu_manager = GPUArrayManager(use_gpu, verbose=False)
    
    if not gpu_manager.use_cupy:
        # Fallback to existing CPU implementation
        from skimage import transform as trans
        tform = trans.SimilarityTransform()
        tform.estimate(landmarks, reference_landmarks)
        return tform.params[0:2, :]
    
    # GPU implementation using CuPy
    landmarks_gpu = gpu_manager.to_gpu(landmarks.astype(np.float64))
    reference_gpu = gpu_manager.to_gpu(reference_landmarks.astype(np.float64))
    
    # Compute centroids
    landmarks_mean = cp.mean(landmarks_gpu, axis=0)
    reference_mean = cp.mean(reference_gpu, axis=0)
    
    # Center the points
    landmarks_centered = landmarks_gpu - landmarks_mean
    reference_centered = reference_gpu - reference_mean
    
    # Compute optimal transformation
    H = cp.dot(landmarks_centered.T, reference_centered)
    U, S, Vt = cp.linalg.svd(H)
    R = cp.dot(Vt.T, U.T)
    
    # Ensure proper rotation (det = 1)
    if cp.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = cp.dot(Vt.T, U.T)
    
    # Compute scale
    scale = cp.sum(S) / cp.sum(landmarks_centered ** 2) if cp.sum(landmarks_centered ** 2) > 0 else 1.0
    
    # Compute translation
    translation = reference_mean - scale * cp.dot(landmarks_mean, R.T)
    
    # Construct transformation matrix
    M = cp.zeros((2, 3))
    M[:2, :2] = scale * R
    M[:, 2] = translation
    
    return gpu_manager.to_cpu(M.astype(np.float64))

def analyze_cupy_performance():
    """Анализирует производительность CuPy vs NumPy."""
    if not CUPY_AVAILABLE:
        print("[CPU] CuPy not available - skipping performance analysis")
        return
    
    print("[PERF] CuPy Performance Analysis")
    print("=" * 40)
    
    # Test array sizes
    sizes = [1000, 5000, 10000]
    
    for size in sizes:
        print(f"\nTesting with {size}x{size} arrays:")
        
        # Generate test data
        a = np.random.random((size, size)).astype(np.float32)
        b = np.random.random((size, size)).astype(np.float32)
        
        # CPU timing
        import time
        start_time = time.time()
        cpu_result = np.dot(a, b)
        cpu_time = time.time() - start_time
        
        # GPU timing
        a_gpu = cp.asarray(a)
        b_gpu = cp.asarray(b)
        cp.cuda.Stream.null.synchronize()  # Ensure arrays are on GPU
        
        start_time = time.time()
        gpu_result = cp.dot(a_gpu, b_gpu)
        cp.cuda.Stream.null.synchronize()  # Ensure computation is complete
        gpu_time = time.time() - start_time
        
        # Results
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        
        print(f"  [CPU] CPU time: {cpu_time:.4f}s")
        print(f"  [GPU] GPU time: {gpu_time:.4f}s")
        print(f"  [SPEEDUP] Speedup: {speedup:.2f}x")
        
        if speedup > 5:
            print(f"  [EXCELLENT] Excellent speedup")
        elif speedup > 2:
            print(f"  [GOOD] Moderate speedup")
        else:
            print(f"  [LIMITED] Limited speedup")
        
        # Verify correctness
        gpu_result_cpu = gpu_result.get()
        if np.allclose(cpu_result, gpu_result_cpu, rtol=1e-5):
            print(f"  [VERIFIED] Results match")
        else:
            print(f"  [ERROR] Results don't match!")

def get_optimal_config() -> dict:
    """Получает оптимальную конфигурацию для данной системы."""
    config = {
        'use_cupy': CUPY_AVAILABLE,
        'device_count': 0,
        'memory_gb': 0,
        'compute_capability': 'unknown',
        'recommended_batch_size': 1,
        'use_mixed_precision': False
    }
    
    if CUPY_AVAILABLE:
        try:
            # Get device info
            device = cp.cuda.Device()
            config['device_count'] = cp.cuda.runtime.getDeviceCount()
            config['memory_gb'] = device.mem_info[1] / (1024**3)  # Total memory in GB
            
            # Get compute capability
            major = device.compute_capability[0]
            minor = device.compute_capability[1]
            config['compute_capability'] = f"{major}.{minor}"
            
            # Recommended settings based on memory
            if config['memory_gb'] >= 8:
                config['recommended_batch_size'] = 4
                config['use_mixed_precision'] = True
            elif config['memory_gb'] >= 4:
                config['recommended_batch_size'] = 2
                config['use_mixed_precision'] = major >= 7  # Tensor cores available
            else:
                config['recommended_batch_size'] = 1
                config['use_mixed_precision'] = False
                
        except Exception as e:
            print(f"[ERROR] Error getting GPU info: {e}")
    
    return config

def print_gpu_info():
    """Выводит информацию о доступности GPU."""
    print("[INFO] GPU Acceleration Status:")
    print("=" * 30)
    
    config = get_optimal_config()
    
    if config['use_cupy']:
        print(f"[GPU] CuPy: Available")
        print(f"[INFO] Devices: {config['device_count']}")
        print(f"[INFO] Memory: {config['memory_gb']:.1f} GB")
        print(f"[INFO] Compute: {config['compute_capability']}")
        print(f"[INFO] Recommended batch size: {config['recommended_batch_size']}")
        print(f"[INFO] Mixed precision: {config['use_mixed_precision']}")
        
        # Performance recommendation
        cc_major = int(config['compute_capability'].split('.')[0])
        if cc_major >= 7:
            print("[GPU] Excellent GPU for CuPy acceleration!")
        elif cc_major >= 6:
            print("[GOOD] Good GPU for CuPy acceleration")
        else:
            print("[LIMITED] Older GPU - limited CuPy benefits")
    else:
        print("[CPU] CuPy: Not available")
        print("[INFO] Using CPU-only operations")

def get_provider_info():
    """Возвращает информацию о доступных провайдерах."""
    providers = []
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            providers.append({
                'name': 'cuda',
                'available': True,
                'device_count': torch.cuda.device_count(),
                'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            })
        else:
            providers.append({'name': 'cuda', 'available': False})
    except ImportError:
        providers.append({'name': 'cuda', 'available': False, 'error': 'torch not installed'})
    
    # Check DirectML  
    try:
        import torch_directml
        providers.append({
            'name': 'dml',
            'available': True,
            'device_count': torch_directml.device_count(),
            'devices': [f"DirectML Device {i}" for i in range(torch_directml.device_count())]
        })
    except ImportError:
        providers.append({'name': 'dml', 'available': False, 'error': 'torch-directml not installed'})
    
    # Check OpenVINO
    try:
        import openvino
        providers.append({
            'name': 'openvino', 
            'available': True,
            'version': openvino.__version__
        })
    except ImportError:
        providers.append({'name': 'openvino', 'available': False, 'error': 'openvino not installed'})
    
    # CPU always available
    import os
    providers.append({
        'name': 'cpu',
        'available': True,
        'cores': os.cpu_count()
    })
    
    return providers

def print_provider_info():
    """Выводит информацию о доступных провайдерах."""
    providers = get_provider_info()
    
    print("[INFO] Available Providers:")
    print("=" * 25)
    
    for provider in providers:
        name = provider['name'].upper()
        if provider['available']:
            print(f"[OK] {name}: Available")
            if 'device_count' in provider:
                print(f"     Devices: {provider['device_count']}")
            if 'devices' in provider:
                for device in provider['devices']:
                    print(f"     - {device}")
            if 'version' in provider:
                print(f"     Version: {provider['version']}")
            if 'cores' in provider:
                print(f"     CPU Cores: {provider['cores']}")
        else:
            error = provider.get('error', 'Unknown error')
            print(f"[NO] {name}: Not available ({error})")

def create_model_with_provider(model_name: str, provider: str = None):
    """Создает модель с указанным провайдером."""
    if provider is None:
        # Auto-detect best provider
        providers = get_provider_info()
        for p in providers:
            if p['available'] and p['name'] in ['cuda', 'dml', 'openvino']:
                provider = p['name']
                break
        else:
            provider = 'cpu'
    
    print(f"[MODEL] Creating {model_name} with {provider.upper()} provider")
    
    # Mock model creation - в реальном коде здесь будет загрузка модели
    model_config = {
        'name': model_name,
        'provider': provider,
        'created': True
    }
    
    if provider == 'cuda':
        model_config['device'] = 'cuda:0'
        model_config['precision'] = 'fp16' if get_optimal_config()['use_mixed_precision'] else 'fp32'
    elif provider == 'dml':
        model_config['device'] = 'dml'
        model_config['precision'] = 'fp32'
    elif provider == 'openvino':
        model_config['device'] = 'CPU'
        model_config['precision'] = 'fp32'
    else:  # cpu
        model_config['device'] = 'cpu'
        model_config['precision'] = 'fp32'
    
    return model_config

def demo_provider_usage():
    """Демонстрирует использование новой системы провайдеров."""
    print("[DEMO] LiveSwapping Provider System Demo")
    print("=" * 50)
    
    # Show available providers
    print_provider_info()
    print()
    
    # Show GPU info
    print_gpu_info()
    print()
    
    # Create example models
    models = ['face_detection', 'face_swap', 'upscaler']
    
    for model in models:
        config = create_model_with_provider(model)
        print(f"[CONFIG] {config}")
        print()
    
    # Performance analysis
    if CUPY_AVAILABLE:
        analyze_cupy_performance()

if __name__ == "__main__":
    # Run demo
    demo_provider_usage()

def get_installation_command():
    """Возвращает команду установки для текущей системы."""
    import platform
    
    system = platform.system().lower()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    commands = {
        'windows': {
            'cuda': f'pip install cupy-cuda12x',
            'dml': f'pip install torch-directml',
            'openvino': f'pip install openvino',
        },
        'linux': {
            'cuda': f'pip install cupy-cuda12x',
            'openvino': f'pip install openvino',
        },
        'darwin': {  # macOS
            'cpu': f'pip install torch torchvision',
        }
    }
    
    if system in commands:
        return commands[system]
    else:
        return {'cpu': 'pip install torch torchvision'}

# Auto-initialize on import
try:
    _config = get_optimal_config()
    if _config['use_cupy']:
        print(f"[INIT] GPU acceleration ready: {_config['memory_gb']:.1f}GB GPU")
    else:
        print("[INIT] Using CPU operations")
except Exception:
    print("[INIT] Using CPU operations (GPU detection failed)") 
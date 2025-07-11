# -*- coding: utf-8 -*-
"""Модуль для загрузки и управления моделями в LiveSwapping.

Поддерживает различные типы моделей:
- StyleTransfer (.pth) - PyTorch модели с torch-tensorrt оптимизацией
- DFM (.dfm) - Deep Face Model
- InSwapper (.onnx) - InsightFace модели
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch  # type: ignore

__all__ = [
    "MODEL_REGISTRY", 
    "load_model", 
    "list_available_models", 
    "get_model_type", 
    "_create_providers"
]

# Импортируем реальный реестр моделей из download_models
from .download_models import MODELS as DOWNLOAD_MODELS

# Расширяем модели информацией о типах для загрузчика
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "inswapper128": {
        **DOWNLOAD_MODELS["inswapper128"],
        "type": "inswapper",
        "description": "InsightFace inswapper model 128x128"
    },
    "reswapper128": {
        **DOWNLOAD_MODELS["reswapper128"],
        "type": "styletransfer", 
        "description": "StyleTransfer reswapper model 128x128"
    },
    "reswapper256": {
        **DOWNLOAD_MODELS["reswapper256"],
        "type": "styletransfer",
        "description": "StyleTransfer reswapper model 256x256"
    }
}


# Проверяем какие провайдеры доступны
def _detect_available_providers() -> Dict[str, bool]:
    """Определяет доступные провайдеры ONNX Runtime."""
    providers = {
        "tensorrt": False,
        "cuda": False,
        "directml": False,
        "openvino": False,
        "cpu": True  # CPU всегда доступен
    }
    
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        
        providers["tensorrt"] = "TensorrtExecutionProvider" in available
        providers["cuda"] = "CUDAExecutionProvider" in available
        providers["directml"] = "DmlExecutionProvider" in available
        providers["openvino"] = "OpenVINOExecutionProvider" in available
        
    except ImportError:
        pass
    
    return providers


def _create_providers(force_provider: Optional[str] = None) -> List:
    """Создает список провайдеров ONNX Runtime с автоматическим определением.
    
    Parameters
    ----------
    force_provider : str, optional
        Принудительно использовать конкретный провайдер:
        'cuda', 'directml', 'openvino', 'cpu'
    
    Returns
    -------
    List
        Список провайдеров в порядке приоритета
    """
    try:
        import onnxruntime as ort
        available_providers = _detect_available_providers()
        providers = []
        
        # Если указан конкретный провайдер
        if force_provider:
            if force_provider == "cuda" and available_providers["cuda"]:
                if available_providers["tensorrt"]:
                    providers.append(("TensorrtExecutionProvider", {
                        "trt_fp16_enable": "0",
                        "trt_engine_cache_enable": "1"
                    }))
                providers.append("CUDAExecutionProvider")
                
            elif force_provider == "directml" and available_providers["directml"]:
                providers.append("DmlExecutionProvider")
                
            elif force_provider == "openvino" and available_providers["openvino"]:
                providers.append(("OpenVINOExecutionProvider", {
                    "device_type": "GPU_FP32",  # Используем GPU если доступно
                    "precision": "FP32"
                }))
                
            providers.append("CPUExecutionProvider")
            return providers
        
        # Автоматическое определение лучшего провайдера
        # Приоритет: TensorRT+CUDA > DirectML > OpenVINO > CUDA > CPU
        
        # NVIDIA GPU (TensorRT + CUDA)
        if available_providers["tensorrt"] and available_providers["cuda"]:
            providers.append(("TensorrtExecutionProvider", {
                "trt_fp16_enable": "0",
                "trt_engine_cache_enable": "1",
                "trt_max_workspace_size": "2147483648",  # 2GB
                "trt_max_partition_iterations": "1000"
            }))
            providers.append("CUDAExecutionProvider")
            
        # AMD GPU (DirectML - только Windows)
        elif available_providers["directml"]:
            providers.append(("DmlExecutionProvider", {
                "device_id": 0,
                "disable_metacommands": False
            }))
            
        # Intel GPU/CPU (OpenVINO)
        elif available_providers["openvino"]:
            # Пытаемся использовать GPU, если не получается - CPU
            providers.append(("OpenVINOExecutionProvider", {
                "device_type": "GPU_FP32",
                "precision": "FP32",
                "enable_opencl_throttling": False
            }))
            providers.append(("OpenVINOExecutionProvider", {
                "device_type": "CPU",
                "precision": "FP32"
            }))
            
        # Только CUDA (без TensorRT)
        elif available_providers["cuda"]:
            providers.append("CUDAExecutionProvider")
            
        # CPU fallback (всегда добавляем)
        providers.append("CPUExecutionProvider")
        
        return providers
        
    except ImportError:
        return ["CPUExecutionProvider"]


def get_optimal_provider() -> str:
    """Возвращает название оптимального провайдера для текущей системы."""
    providers = _detect_available_providers()
    
    if providers["tensorrt"] and providers["cuda"]:
        return "cuda"  # TensorRT + CUDA
    elif providers["directml"]:
        return "directml"
    elif providers["openvino"]:
        return "openvino" 
    elif providers["cuda"]:
        return "cuda"
    else:
        return "cpu"


def create_session(model_path: str, provider: Optional[str] = None) -> Any:
    """Создает ONNX Runtime сессию с оптимальными провайдерами.
    
    Parameters
    ----------
    model_path : str
        Путь к ONNX модели
    provider : str, optional
        Конкретный провайдер для использования
        
    Returns
    -------
    onnxruntime.InferenceSession
        Настроенная сессия
    """
    import onnxruntime as ort
    
    providers = _create_providers(force_provider=provider)
    
    # Глобальные настройки для производительности
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_mem_pattern = True
    session_options.enable_cpu_mem_arena = True
    
    return ort.InferenceSession(
        model_path, 
        providers=providers,
        sess_options=session_options
    )


def _create_tensorrt_model(model: torch.nn.Module, input_shape: tuple = (1, 3, 128, 128), 
                      source_shape: tuple = (1, 512)) -> torch.nn.Module:
    """Оптимизирует PyTorch модель с помощью torch-tensorrt.
    
    Parameters
    ----------
    model : torch.nn.Module
        Исходная PyTorch модель
    input_shape : tuple
        Форма входного тензора для target изображения  
    source_shape : tuple
        Форма входного тензора для source эмбеддинга
        
    Returns
    -------
    torch.nn.Module
        Оптимизированная модель или исходная при ошибке
    """
    try:
        import torch_tensorrt  # type: ignore
        
        # Создаем примеры входных данных
        target_input = torch_tensorrt.Input(
            min_shape=input_shape,
            opt_shape=input_shape, 
            max_shape=input_shape,
            dtype=torch.float32
        )
        source_input = torch_tensorrt.Input(
            min_shape=source_shape,
            opt_shape=source_shape,
            max_shape=source_shape, 
            dtype=torch.float32
        )
        
        # Компилируем модель с torch-tensorrt
        compiled_model = torch_tensorrt.compile(
            model,
            inputs=[target_input, source_input],
            enabled_precisions={torch.float32},  # Используем FP32 для стабильности
            ir="torch_compile",  # Используем torch.compile backend
            min_block_size=3,  # Минимальный размер блока для TensorRT
        )
        
        #print("[SUCCESS] Модель успешно оптимизирована с torch-tensorrt")
        return compiled_model
        
    except ImportError:
        print("[WARNING] torch-tensorrt не установлен, используется стандартная модель")
        return model
    except Exception as e:
        print(f"[WARNING] Ошибка оптимизации с torch-tensorrt: {e}")
        print("Используется стандартная модель")
        return model


def load_model(name: str, use_tensorrt: bool = True, provider_type: Optional[str] = None, **kwargs) -> Any:
    """Загружает модель по имени из реестра или по пути.

    Parameters
    ----------
    name : str
        Имя модели из реестра или путь к файлу модели
    use_tensorrt : bool
        Использовать torch-tensorrt оптимизацию для PyTorch моделей
    provider_type : str, optional
        Тип ONNX Runtime провайдера: 'cuda', 'directml', 'openvino', 'cpu'
        Если None, автоматически определяется оптимальный провайдер
    **kwargs
        Дополнительные параметры для модели
        
    Returns
    -------
    Any
        Загруженная модель (DFMModel, InSwapperModel или StyleTransferModel)
        
    Examples
    --------
    >>> # Загрузка из реестра с TensorRT оптимизацией
    >>> model = load_model("reswapper128", use_tensorrt=True)
    >>> 
    >>> # Загрузка по пути без оптимизации
    >>> model = load_model("path/to/model.onnx", use_tensorrt=False)
    """
    # Автоматическое определение провайдера если не указан
    if provider_type is None:
        provider_type = get_optimal_provider()
    
    # Проверяем, есть ли модель в реестре
    if name in MODEL_REGISTRY:
        return _load_registry_model(name, use_tensorrt=use_tensorrt, provider_type=provider_type, **kwargs)
    
    # Если не в реестре, пробуем загрузить как путь к файлу
    model_path = Path(name)
    if model_path.exists():
        return _load_model_by_path(model_path, use_tensorrt=use_tensorrt, provider_type=provider_type, **kwargs)
    
    raise ValueError(f"Model '{name}' not found in registry and path doesn't exist")


def _load_registry_model(name: str, use_tensorrt: bool = True, provider_type: Optional[str] = None, **kwargs) -> Any:
    """Загружает модель из реестра."""
    if name == "inswapper128":
        from .download_models import ensure_model
        from .inswapper_model import InSwapperModel
        
        model_path = ensure_model(name)
        return InSwapperModel(str(model_path), provider_type=provider_type)
    
    elif name in ("reswapper128", "reswapper256"):
        from .download_models import ensure_model
        from .style_transfer_model_128 import StyleTransferModel
        import torch  # type: ignore
        
        model_path = ensure_model(name)
        
        # Получаем оптимальное устройство для провайдера
        device_str = get_torch_device(provider_type)
        device = torch.device(device_str)
        
        model = StyleTransferModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False), strict=False)
        model.eval()
        
        # Применяем TensorRT оптимизацию для CUDA PyTorch моделей
        if use_tensorrt and device.type == "cuda" and provider_type == "cuda":
            try:
                # Определяем входные формы для reswapper модели
                if name == "reswapper128":
                    input_shapes = {
                        "target": (1, 3, 128, 128),
                        "source": (1, 512)  # Эмбеддинг
                    }
                else:  # reswapper256
                    input_shapes = {
                        "target": (1, 3, 256, 256), 
                        "source": (1, 512)
                    }
                
                model = compile_with_tensorrt(model, input_shapes, precision="fp32")
                
            except Exception as e:
                print(f"[WARNING] TensorRT compilation failed: {e}")
                print("Using standard PyTorch model")
            
        return model
    
    else:
        raise ValueError(f"Registry model '{name}' not implemented yet")


def _load_model_by_path(model_path: Path, use_tensorrt: bool = True, provider_type: Optional[str] = None, **kwargs) -> Any:
    """Загружает модель по пути к файлу."""
    model_type = get_model_type(model_path)
    
    if model_type == "dfm":
        from .dfm_model import DFMModel
        import torch  # type: ignore
        
        # Получаем оптимальное устройство
        device_str = get_torch_device(provider_type)
        return DFMModel(str(model_path), device=device_str, provider_type=provider_type)
    
    elif model_type == "inswapper":
        from .inswapper_model import InSwapperModel
        
        return InSwapperModel(str(model_path), provider_type=provider_type)
    
    elif model_type == "styletransfer":
        from .style_transfer_model_128 import StyleTransferModel
        import torch  # type: ignore
        
        # Получаем оптимальное устройство для провайдера
        device_str = get_torch_device(provider_type)
        device = torch.device(device_str)
        
        model = StyleTransferModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False), strict=False)
        model.eval()
        
        # Применяем TensorRT оптимизацию для CUDA PyTorch моделей
        if use_tensorrt and device.type == "cuda" and provider_type == "cuda":
            try:
                # Стандартные входные формы для StyleTransfer
                input_shapes = {
                    "target": (1, 3, 128, 128),
                    "source": (1, 512)
                }
                
                model = compile_with_tensorrt(model, input_shapes, precision="fp32")
                
            except Exception as e:
                print(f"[WARNING] TensorRT compilation failed: {e}")
                print("Using standard PyTorch model")
            
        return model
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_model_type(model_path: Path) -> str:
    """Определяет тип модели по расширению файла.
    
    Parameters
    ----------
    model_path : Path
        Путь к файлу модели
        
    Returns
    -------
    str
        Тип модели: 'dfm', 'inswapper', или 'styletransfer'
    """
    suffix = model_path.suffix.lower()
    
    if suffix == ".dfm":
        return "dfm"
    elif suffix == ".onnx":
        # Различаем inswapper и другие ONNX модели по имени
        if "inswapper" in model_path.name.lower():
            return "inswapper"
        else:
            return "dfm"  # По умолчанию считаем DFM
    elif suffix in (".pth", ".pt"):
        return "styletransfer"
    else:
        raise ValueError(f"Unsupported model format: {suffix}")


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """Возвращает список всех доступных моделей из реестра.
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Словарь с информацией о моделях
    """
    return MODEL_REGISTRY.copy() 


def _configure_torch_backend(provider_type: Optional[str] = None) -> str:
    """Настраивает PyTorch backend в зависимости от провайдера.
    
    Parameters
    ----------
    provider_type : str, optional
        Тип провайдера
        
    Returns
    -------
    str
        Рекомендуемое PyTorch устройство
    """
    import torch
    
    # CUDA + TensorRT
    if provider_type == "cuda":
        try:
            import torch_tensorrt
            #print(f"[SUCCESS] TensorRT available for PyTorch: {torch_tensorrt.__version__}")
        except ImportError:
            print("[WARNING] torch-tensorrt not available, using standard CUDA")
        
        device_str = "cuda:0"
        
    # DirectML для AMD GPU
    elif provider_type == "directml":
        try:
            import torch_directml
            device = torch_directml.device()
            #print(f"[SUCCESS] DirectML device: {device}")
            device_str = str(device)
        except ImportError:
            print("[WARNING] torch-directml not available, falling back to CPU")
            device_str = "cpu"
            
    # DirectML для Intel
    elif provider_type == "openvino":
        try:
            import torch_directml
            device = torch_directml.device()
            #print(f"[SUCCESS] Using DirectML for PyTorch on Intel: {device}")
            device_str = str(device)
        except ImportError:
            print("[INFO] DirectML not available, using CPU for PyTorch")
            device_str = "cpu"
    
    # CPU fallback
    else:  # provider_type == "cpu" or None
        device_str = "cpu"
    
    return device_str


def compile_with_tensorrt(model, input_shapes: Dict[str, tuple], precision: str = "fp32") -> Any:
    """Компилирует PyTorch модель с TensorRT оптимизациями.
    
    Parameters
    ----------
    model : torch.nn.Module
        PyTorch модель для оптимизации
    input_shapes : Dict[str, tuple]
        Словарь с именами входов и их формами
    precision : str
        Точность вычислений: "fp32", "fp16", "int8"
        
    Returns
    -------
    torch.nn.Module
        Оптимизированная модель
    """
    try:
        import torch
        import torch_tensorrt
        
        if not torch.cuda.is_available():
            print("[WARNING] CUDA not available, skipping TensorRT compilation")
            return model
        
        #print(f"[COMPILE] Compiling model with TensorRT ({precision})...")
        
        # Настройка точности
        if precision == "fp16":
            enabled_precisions = {torch.float16}
            print("Using FP16 precision")
        elif precision == "int8":
            enabled_precisions = {torch.int8}
            print("Using INT8 precision")
        else:  # fp32
            enabled_precisions = {torch.float32}
            print("Using FP32 precision")
        
        # Создание входных тензоров
        inputs = []
        for name, shape in input_shapes.items():
            tensor_input = torch_tensorrt.Input(
                min_shape=shape,
                opt_shape=shape,
                max_shape=shape,
                dtype=list(enabled_precisions)[0]
            )
            inputs.append(tensor_input)
            #print(f"Input '{name}': {shape}")
        
        # Компиляция модели
        trt_model = torch_tensorrt.compile(
            model,
            inputs=inputs,
            enabled_precisions=enabled_precisions,
            ir="torch_compile",
            min_block_size=3
        )
        
        #print("[SUCCESS] TensorRT compilation successful!")
        return trt_model
        
    except Exception as e:
        print(f"[WARNING] TensorRT compilation failed: {e}")
        print("Falling back to standard PyTorch model")
        return model


def get_torch_device(provider_type: Optional[str] = None) -> str:
    """Возвращает оптимальное PyTorch устройство для провайдера.
    
    Parameters
    ---------- 
    provider_type : str, optional
        Тип ONNX Runtime провайдера
        
    Returns
    -------
    str
        PyTorch device string
    """
    if provider_type is None:
        provider_type = get_optimal_provider()
        
    return _configure_torch_backend(provider_type) 
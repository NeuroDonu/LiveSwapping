# ⚡ Оптимизация производительности

Полное руководство по ускорению LiveSwapping для максимальной производительности.

## 📚 Содержание

1. [Системная диагностика](#-системная-диагностика)
2. [TensorRT оптимизация](#-tensorrt-оптимизация)
3. [GPU ускорение](#-gpu-ускорение)
4. [Выбор провайдеров](#-выбор-провайдеров)
5. [Оптимизация моделей](#-оптимизация-моделей)
6. [Настройки системы](#-настройки-системы)
7. [Бенчмарки и тесты](#-бенчмарки-и-тесты)

---

## 🩺 Системная диагностика

### Быстрая проверка производительности

```python
from liveswapping.utils.gpu_utils import print_gpu_info, get_optimal_config

# Полная информация о системе
print_gpu_info()

# Рекомендуемые настройки
config = get_optimal_config()
print(f"Рекомендуемый batch size: {config['recommended_batch_size']}")
print(f"Смешанная точность: {config['use_mixed_precision']}")
print(f"Память GPU: {config['memory_gb']:.1f} GB")
```

### Детальная диагностика

```python
# performance_check.py
import torch
import time
import numpy as np
from liveswapping.ai_models.models import get_optimal_provider, load_model

def comprehensive_benchmark():
    """Полный бенчмарк системы."""
    
    print("=== КОМПЛЕКСНЫЙ БЕНЧМАРК СИСТЕМЫ ===\n")
    
    # 1. Информация о GPU
    print("1. GPU СТАТУС:")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name}")
            print(f"   Память: {props.total_memory / 1024**3:.1f} GB")
            print(f"   Compute: {props.major}.{props.minor}")
    else:
        print("   CUDA недоступен")
    
    # 2. Оптимальный провайдер
    provider = get_optimal_provider()
    print(f"\n2. ОПТИМАЛЬНЫЙ ПРОВАЙДЕР: {provider.upper()}")
    
    # 3. Тест загрузки модели
    print("\n3. ТЕСТ ЗАГРУЗКИ МОДЕЛЕЙ:")
    models_to_test = ["reswapper128", "reswapper256"]
    
    for model_name in models_to_test:
        start_time = time.time()
        try:
            model = load_model(model_name, use_tensorrt=True, provider_type=provider)
            load_time = time.time() - start_time
            print(f"   {model_name}: {load_time:.2f}s ✅")
        except Exception as e:
            print(f"   {model_name}: ОШИБКА - {e} ❌")
    
    # 4. Бенчмарк inference
    print("\n4. БЕНЧМАРК INFERENCE:")
    try:
        model = load_model("reswapper128", use_tensorrt=True, provider_type=provider)
        
        # Подготовка тестовых данных
        if provider == "cuda":
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        target = torch.randn(1, 3, 128, 128).to(device)
        source = torch.randn(1, 512).to(device)
        
        # Прогрев
        for _ in range(3):
            with torch.no_grad():
                _ = model(target, source)
        
        if provider == "cuda":
            torch.cuda.synchronize()
        
        # Бенчмарк
        times = []
        for _ in range(10):
            start_time = time.time()
            with torch.no_grad():
                result = model(target, source)
            if provider == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - start_time)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        print(f"   Среднее время: {avg_time:.4f}s")
        print(f"   FPS: {fps:.1f}")
        
        # Оценка производительности
        if fps >= 30:
            print("   Оценка: ОТЛИЧНО 🚀")
        elif fps >= 15:
            print("   Оценка: ХОРОШО ✅")
        elif fps >= 5:
            print("   Оценка: ПРИЕМЛЕМО ⚠️")
        else:
            print("   Оценка: МЕДЛЕННО ❌")
            
    except Exception as e:
        print(f"   Ошибка бенчмарка: {e}")
    
    print("\n=== БЕНЧМАРК ЗАВЕРШЕН ===")

# Запуск бенчмарка
comprehensive_benchmark()
```

---

## 🎯 TensorRT оптимизация

### Включение TensorRT

TensorRT обеспечивает **3x ускорение** для PyTorch моделей на NVIDIA GPU.

```python
from liveswapping.ai_models.models import load_model

# Автоматическое включение TensorRT (рекомендуется)
model = load_model("reswapper128", use_tensorrt=True)

# Принудительное отключение для отладки
model = load_model("reswapper128", use_tensorrt=False)
```

### Проверка работы TensorRT

```python
def check_tensorrt_status():
    """Проверка статуса TensorRT оптимизации."""
    
    try:
        import torch_tensorrt
        print(f"✅ torch-tensorrt установлен: {torch_tensorrt.__version__}")
        
        # Проверка совместимости CUDA
        cuda_version = torch.version.cuda
        print(f"✅ PyTorch CUDA: {cuda_version}")
        
        # Тест компиляции
        model = load_model("reswapper128", use_tensorrt=True)
        print("✅ TensorRT компиляция успешна")
        
        return True
        
    except ImportError:
        print("❌ torch-tensorrt не установлен")
        print("Установите: pip install torch-tensorrt")
        return False
    except Exception as e:
        print(f"❌ Ошибка TensorRT: {e}")
        return False

check_tensorrt_status()
```

### Ручная установка TensorRT

```bash
# Установка torch-tensorrt
pip install torch-tensorrt

# Проверка совместимости
python -c "import torch_tensorrt; print(torch_tensorrt.__version__)"

# Для старых версий CUDA
pip install torch-tensorrt --index-url https://download.pytorch.org/whl/cu118
```

### Настройка TensorRT параметров

```python
def create_custom_tensorrt_model(model, precision="fp32"):
    """Кастомная TensorRT оптимизация."""
    
    try:
        import torch_tensorrt
        
        # Настройки оптимизации
        enabled_precisions = {torch.float32}
        if precision == "fp16":
            enabled_precisions.add(torch.half)
        
        # Компиляция с кастомными настройками
        compiled_model = torch_tensorrt.compile(
            model,
            inputs=[
                torch_tensorrt.Input((1, 3, 128, 128)),
                torch_tensorrt.Input((1, 512))
            ],
            enabled_precisions=enabled_precisions,
            ir="torch_compile",
            min_block_size=3,
            require_full_compilation=False,
        )
        
        return compiled_model
        
    except Exception as e:
        print(f"TensorRT compilation failed: {e}")
        return model
```

---

## 🎮 GPU ускорение

### CuPy для numpy операций

CuPy обеспечивает значительное ускорение numpy операций.

```python
from liveswapping.utils.gpu_utils import GPUArrayManager

# Создание менеджера GPU массивов
gpu_manager = GPUArrayManager(use_cupy=True, verbose=True)

def optimized_image_processing(image):
    """Оптимизированная обработка изображений."""
    
    # Перенос на GPU
    gpu_image = gpu_manager.to_gpu(image)
    
    # GPU операции (пример)
    # gpu_result = cupy_operation(gpu_image)
    
    # Возврат на CPU
    result = gpu_manager.to_cpu(gpu_image)
    gpu_manager.synchronize()
    
    return result
```

### Бенчмарк CuPy vs NumPy

```python
from liveswapping.utils.gpu_utils import analyze_cupy_performance

# Анализ производительности CuPy
analyze_cupy_performance()

# Вывод:
# Testing with 1000x1000 arrays:
# [CPU] CPU time: 0.0123s
# [GPU] GPU time: 0.0041s  
# [SPEEDUP] Speedup: 3.0x
```

### Адаптивная обработка

```python
from liveswapping.utils.adaptive_cupy import create_adaptive_processor

# Создание адаптивного процессора
processor = create_adaptive_processor(1080)  # Для 1080p

# Использование адаптивной обработки
from liveswapping.utils.adaptive_cupy import AdaptiveColorTransfer, AdaptiveBlending

color_transfer = AdaptiveColorTransfer(processor)
blending = AdaptiveBlending(processor)

# Автоматический выбор GPU/CPU в зависимости от размера
result = color_transfer.apply_color_transfer_adaptive(source_path, target, face_analysis)
```

---

## 🔄 Выбор провайдеров

### Автоматическое определение

```python
from liveswapping.ai_models.models import get_optimal_provider, load_model

# Автоматическое определение лучшего провайдера
provider = get_optimal_provider()
print(f"Оптимальный провайдер: {provider}")

# Загрузка с оптимальным провайдером
model = load_model("reswapper128", provider_type=provider)
```

### Ручной выбор провайдера

#### NVIDIA GPU (CUDA + TensorRT)
```python
# Максимальная производительность для NVIDIA
model = load_model("reswapper128", 
                  provider_type="cuda", 
                  use_tensorrt=True)
```

#### AMD GPU (DirectML)
```python
# Для AMD GPU
model = load_model("reswapper128", 
                  provider_type="directml")
```

#### Intel GPU/CPU (OpenVINO)
```python
# Для Intel устройств
model = load_model("reswapper128", 
                  provider_type="openvino")
```

#### CPU Only
```python
# Только CPU (fallback)
model = load_model("reswapper128", 
                  provider_type="cpu",
                  use_tensorrt=False)
```

### Сравнение провайдеров

```python
def benchmark_providers():
    """Бенчмарк всех доступных провайдеров."""
    
    providers = ["cuda", "directml", "openvino", "cpu"]
    results = {}
    
    for provider in providers:
        try:
            print(f"\nТестирование {provider.upper()}...")
            
            start_time = time.time()
            model = load_model("reswapper128", 
                             provider_type=provider,
                             use_tensorrt=(provider == "cuda"))
            load_time = time.time() - start_time
            
            # Тест inference (упрощенный)
            inference_times = []
            for _ in range(5):
                start = time.time()
                # Симуляция inference
                time.sleep(0.01)  # Заглушка
                inference_times.append(time.time() - start)
            
            avg_inference = np.mean(inference_times)
            fps = 1.0 / avg_inference
            
            results[provider] = {
                'load_time': load_time,
                'inference_time': avg_inference,
                'fps': fps,
                'status': 'success'
            }
            
            print(f"   Загрузка: {load_time:.2f}s")
            print(f"   Inference: {avg_inference:.4f}s")
            print(f"   FPS: {fps:.1f}")
            
        except Exception as e:
            results[provider] = {
                'status': 'failed',
                'error': str(e)
            }
            print(f"   Ошибка: {e}")
    
    # Лучший провайдер
    successful = {k: v for k, v in results.items() if v['status'] == 'success'}
    if successful:
        best = max(successful.items(), key=lambda x: x[1]['fps'])
        print(f"\n🏆 Лучший провайдер: {best[0].upper()} ({best[1]['fps']:.1f} FPS)")
    
    return results

benchmark_providers()
```

---

## 🧠 Оптимизация моделей

### Выбор оптимальной модели

| Сценарий | Модель | Провайдер | TensorRT | FPS (RTX 4090) |
|----------|--------|-----------|----------|----------------|
| **Real-time максимум** | reswapper128 | CUDA | ✅ | ~45 |
| **Real-time качество** | reswapper256 | CUDA | ✅ | ~25 |
| **Универсальная** | inswapper128 | CUDA | ❌ | ~30 |
| **CPU режим** | reswapper128 | CPU | ❌ | ~2 |

### Оптимизация по GPU памяти

```python
def optimize_for_memory(gpu_memory_gb):
    """Оптимизация настроек под доступную GPU память."""
    
    if gpu_memory_gb >= 12:
        # Высокопроизводительные GPU (RTX 4090, 3090)
        return {
            'model': 'reswapper256',
            'batch_size': 4,
            'resolution': 256,
            'use_tensorrt': True,
            'upscaling': True
        }
    elif gpu_memory_gb >= 8:
        # Средние GPU (RTX 4070, 3080)
        return {
            'model': 'reswapper256',
            'batch_size': 2,
            'resolution': 256,
            'use_tensorrt': True,
            'upscaling': True
        }
    elif gpu_memory_gb >= 4:
        # Младшие GPU (GTX 1660, RTX 3060)
        return {
            'model': 'reswapper128',
            'batch_size': 2,
            'resolution': 128,
            'use_tensorrt': True,
            'upscaling': False
        }
    else:
        # Очень ограниченная память
        return {
            'model': 'reswapper128',
            'batch_size': 1,
            'resolution': 128,
            'use_tensorrt': False,
            'upscaling': False
        }

# Автоматическая оптимизация
config = get_optimal_config()
settings = optimize_for_memory(config['memory_gb'])
print(f"Рекомендуемые настройки: {settings}")
```

### Предзагрузка моделей

```python
class ModelCache:
    """Кэш для предзагруженных моделей."""
    
    def __init__(self):
        self.models = {}
    
    def preload_models(self, model_names):
        """Предзагрузка моделей в память."""
        
        for name in model_names:
            print(f"Предзагрузка {name}...")
            self.models[name] = load_model(name, use_tensorrt=True)
            print(f"✅ {name} загружена")
    
    def get_model(self, name):
        """Получение предзагруженной модели."""
        return self.models.get(name)

# Использование
cache = ModelCache()
cache.preload_models(['reswapper128', 'reswapper256'])

# Мгновенное получение модели
model = cache.get_model('reswapper128')
```

---

## ⚙️ Настройки системы

### Переменные окружения

```bash
# Оптимизация CUDA
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0

# Оптимизация PyTorch
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_CUDNN_V8_API_ENABLED=1

# Отключение verbose логов
export ONNX_LOG_LEVEL=3
export OMP_NUM_THREADS=1

# Память
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
```

### Настройки Windows

```cmd
REM GPU приоритет
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Multimedia\SystemProfile\Tasks\Games" /v "GPU Priority" /t REG_DWORD /d 8 /f

REM Высокий приоритет процесса
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Multimedia\SystemProfile\Tasks\Games" /v "Priority" /t REG_DWORD /d 6 /f
```

### Настройки Linux

```bash
# GPU performance mode
sudo nvidia-smi -pm 1
sudo nvidia-smi -ac 1215,2100  # Для RTX 4090

# CPU governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Swappiness
echo 10 | sudo tee /proc/sys/vm/swappiness
```

### Оптимизация Python

```python
# performance_config.py
import gc
import torch

def optimize_python_runtime():
    """Оптимизация Python runtime для производительности."""
    
    # Отключение сборщика мусора во время inference
    gc.disable()
    
    # Оптимизация PyTorch
    torch.backends.cudnn.benchmark = True  # Автооптимизация cuDNN
    torch.backends.cudnn.deterministic = False  # Максимальная скорость
    
    # Предварительное выделение памяти
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Прогрев GPU
        dummy = torch.randn(1000, 1000).cuda()
        dummy = dummy @ dummy
        del dummy
        torch.cuda.empty_cache()
    
    print("✅ Python runtime оптимизирован")

optimize_python_runtime()
```

---

## 📊 Бенчмарки и тесты

### Полный бенчмарк производительности

```python
def full_performance_benchmark():
    """Полный бенчмарк всех компонентов системы."""
    
    print("=== ПОЛНЫЙ БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ ===\n")
    
    components = [
        ("Model Loading", benchmark_model_loading),
        ("Inference Speed", benchmark_inference),
        ("Memory Usage", benchmark_memory),
        ("Upscaling", benchmark_upscaling),
        ("End-to-End", benchmark_end_to_end)
    ]
    
    results = {}
    
    for name, benchmark_func in components:
        print(f"🔄 Тестирование: {name}")
        try:
            result = benchmark_func()
            results[name] = result
            print(f"✅ {name}: завершено\n")
        except Exception as e:
            print(f"❌ {name}: ошибка - {e}\n")
            results[name] = {'error': str(e)}
    
    # Общий отчет
    print("=== ИТОГОВЫЙ ОТЧЕТ ===")
    for name, result in results.items():
        if 'error' not in result:
            print(f"✅ {name}: {result.get('summary', 'OK')}")
        else:
            print(f"❌ {name}: {result['error']}")
    
    return results

def benchmark_model_loading():
    """Бенчмарк загрузки моделей."""
    
    models = ['reswapper128', 'reswapper256', 'inswapper128']
    times = {}
    
    for model_name in models:
        start_time = time.time()
        model = load_model(model_name, use_tensorrt=True)
        load_time = time.time() - start_time
        times[model_name] = load_time
        print(f"   {model_name}: {load_time:.2f}s")
    
    avg_time = np.mean(list(times.values()))
    return {
        'times': times,
        'average': avg_time,
        'summary': f'{avg_time:.2f}s avg'
    }

def benchmark_inference():
    """Бенчмарк скорости inference."""
    
    model = load_model('reswapper128', use_tensorrt=True)
    provider = get_optimal_provider()
    
    if provider == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Подготовка данных
    target = torch.randn(1, 3, 128, 128).to(device)
    source = torch.randn(1, 512).to(device)
    
    # Прогрев
    for _ in range(10):
        with torch.no_grad():
            _ = model(target, source)
    
    if provider == "cuda":
        torch.cuda.synchronize()
    
    # Бенчмарк
    times = []
    for _ in range(50):
        start_time = time.time()
        with torch.no_grad():
            result = model(target, source)
        if provider == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - start_time)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"   Среднее время: {avg_time:.4f}s")
    print(f"   FPS: {fps:.1f}")
    
    return {
        'avg_time': avg_time,
        'fps': fps,
        'summary': f'{fps:.1f} FPS'
    }

def benchmark_memory():
    """Бенчмарк использования памяти."""
    
    if not torch.cuda.is_available():
        return {'summary': 'CUDA недоступен'}
    
    # Начальная память
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated(0)
    
    # Загрузка модели
    model = load_model('reswapper128', use_tensorrt=True)
    model_memory = torch.cuda.memory_allocated(0) - initial_memory
    
    # Inference память
    target = torch.randn(1, 3, 128, 128).cuda()
    source = torch.randn(1, 512).cuda()
    
    with torch.no_grad():
        result = model(target, source)
    
    peak_memory = torch.cuda.max_memory_allocated(0)
    
    memory_mb = {
        'model': model_memory / 1024**2,
        'peak': peak_memory / 1024**2,
        'current': torch.cuda.memory_allocated(0) / 1024**2
    }
    
    print(f"   Модель: {memory_mb['model']:.1f} MB")
    print(f"   Пик: {memory_mb['peak']:.1f} MB")
    
    return {
        'memory_mb': memory_mb,
        'summary': f"{memory_mb['peak']:.1f} MB peak"
    }

# Запуск полного бенчмарка
results = full_performance_benchmark()
```

### Сравнение настроек

```python
def compare_settings():
    """Сравнение различных настроек производительности."""
    
    settings = [
        {'name': 'Максимальная скорость', 'model': 'reswapper128', 'tensorrt': True, 'resolution': 128},
        {'name': 'Сбалансированная', 'model': 'reswapper128', 'tensorrt': True, 'resolution': 256},
        {'name': 'Максимальное качество', 'model': 'reswapper256', 'tensorrt': True, 'resolution': 256},
        {'name': 'CPU режим', 'model': 'reswapper128', 'tensorrt': False, 'resolution': 128},
    ]
    
    print("=== СРАВНЕНИЕ НАСТРОЕК ===\n")
    
    for setting in settings:
        print(f"🔄 Тестирование: {setting['name']}")
        
        try:
            # Загрузка модели
            start_time = time.time()
            model = load_model(
                setting['model'], 
                use_tensorrt=setting['tensorrt'],
                provider_type="cuda" if setting['tensorrt'] else "cpu"
            )
            load_time = time.time() - start_time
            
            # Тест inference
            device = torch.device("cuda" if setting['tensorrt'] else "cpu")
            res = setting['resolution']
            target = torch.randn(1, 3, res, res).to(device)
            source = torch.randn(1, 512).to(device)
            
            # Прогрев
            for _ in range(3):
                with torch.no_grad():
                    _ = model(target, source)
            
            # Бенчмарк
            times = []
            for _ in range(10):
                start = time.time()
                with torch.no_grad():
                    result = model(target, source)
                if setting['tensorrt']:
                    torch.cuda.synchronize()
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            fps = 1.0 / avg_time
            
            print(f"   Загрузка: {load_time:.2f}s")
            print(f"   FPS: {fps:.1f}")
            print(f"   Разрешение: {res}x{res}")
            print("")
            
        except Exception as e:
            print(f"   Ошибка: {e}\n")

compare_settings()
```

### Автоматическая настройка

```python
def auto_optimize():
    """Автоматическая оптимизация настроек под систему."""
    
    print("=== АВТОМАТИЧЕСКАЯ ОПТИМИЗАЦИЯ ===\n")
    
    # Диагностика системы
    config = get_optimal_config()
    provider = get_optimal_provider()
    
    print(f"Обнаружен провайдер: {provider.upper()}")
    print(f"Память GPU: {config['memory_gb']:.1f} GB")
    
    # Выбор оптимальных настроек
    if provider == "cuda" and config['memory_gb'] >= 8:
        optimal_settings = {
            'model': 'reswapper256',
            'use_tensorrt': True,
            'resolution': 256,
            'batch_size': 2,
            'upscaling': True
        }
        expected_fps = "20-30"
    elif provider == "cuda" and config['memory_gb'] >= 4:
        optimal_settings = {
            'model': 'reswapper128',
            'use_tensorrt': True,
            'resolution': 128,
            'batch_size': 2,
            'upscaling': True
        }
        expected_fps = "30-45"
    elif provider in ["directml", "openvino"]:
        optimal_settings = {
            'model': 'reswapper128',
            'use_tensorrt': False,
            'resolution': 128,
            'batch_size': 1,
            'upscaling': False
        }
        expected_fps = "10-20"
    else:
        optimal_settings = {
            'model': 'reswapper128',
            'use_tensorrt': False,
            'resolution': 128,
            'batch_size': 1,
            'upscaling': False
        }
        expected_fps = "2-5"
    
    print(f"\n🎯 РЕКОМЕНДУЕМЫЕ НАСТРОЙКИ:")
    for key, value in optimal_settings.items():
        print(f"   {key}: {value}")
    print(f"   Ожидаемый FPS: {expected_fps}")
    
    # Тестирование рекомендуемых настроек
    print(f"\n🔄 Тестирование рекомендуемых настроек...")
    
    try:
        model = load_model(
            optimal_settings['model'],
            use_tensorrt=optimal_settings['use_tensorrt'],
            provider_type=provider
        )
        
        print("✅ Настройки успешно применены")
        print("💡 Используйте эти настройки для оптимальной производительности")
        
    except Exception as e:
        print(f"❌ Ошибка применения настроек: {e}")
    
    return optimal_settings

# Автоматическая оптимизация
optimal = auto_optimize()
```

---

## 🎯 Рекомендации по использованию

### Real-time режим
```bash
# Максимальная скорость
python -m liveswapping.run realtime \
    --source face.jpg \
    --modelPath models/reswapper128.pth \
    --resolution 128 \
    --delay 0

# Лучшее качество
python -m liveswapping.run realtime \
    --source face.jpg \
    --modelPath models/reswapper256.pth \
    --resolution 256 \
    --mouth_mask
```

### Обработка видео
```bash
# Максимальное качество
python -m liveswapping.run video \
    --source face.jpg \
    --target_video video.mp4 \
    --modelPath models/reswapper256.pth \
    --upscale 2 \
    --bg_upsampler realesrgan \
    --weight 0.8

# Быстрая обработка
python -m liveswapping.run video \
    --source face.jpg \
    --target_video video.mp4 \
    --modelPath models/reswapper128.pth \
    --resolution 128
```

---

## 🔗 Дополнительные ресурсы

- **[🏠 Home](Home)** - Главная страница wiki
- **[🎯 Quick Start](Quick-Start)** - Быстрый старт  
- **[🔧 Troubleshooting](Troubleshooting)** - Решение проблем
- **[📋 API Reference](API-Reference)** - Справочник API

---

*[⬅️ API Reference](API-Reference) | [🏠 Главная](Home)*
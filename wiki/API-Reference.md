# 📋 API Reference

Полный справочник по API LiveSwapping для разработчиков.

## 📚 Содержание

1. [Основные модули](#-основные-модули)
2. [Core API](#-core-api)
3. [AI Models API](#-ai-models-api)
4. [Utility APIs](#-utility-apis)
5. [GUI Components](#-gui-components)
6. [Конфигурация](#-конфигурация)

---

## 🏗️ Основные модули

### Entry Points

#### `liveswapping.run`
```python
from liveswapping.run import run, main, start_gui

run()                    # Главная точка входа
main()                   # Алиас для run()
start_gui()              # Только GUI режим
```

#### Command Line Interface
```bash
# Основные режимы
python -m liveswapping.run image --source src.jpg --target tgt.jpg --modelPath model.pth --output result.jpg
python -m liveswapping.run video --source src.jpg --target_video vid.mp4 --modelPath model.pth
python -m liveswapping.run realtime --source src.jpg --modelPath model.pth
```

---

## 🔄 Core API

### `liveswapping.core.realtime`

Real-time обработка с веб-камеры.

#### Functions

##### `main(parsed_args=None)`
Основная функция real-time обработки.

**Parameters:**
- `parsed_args`: Optional[argparse.Namespace] - Предварительно разобранные аргументы

**Returns:**
- `int`: Exit code (0 - успех, 1 - ошибка)

**Example:**
```python
from liveswapping.core.realtime import main, parse_arguments

args = parse_arguments(['--source', 'face.jpg', '--modelPath', 'model.pth'])
exit_code = main(args)
```

##### `parse_arguments(argv=None)`
Парсинг аргументов командной строки.

**Parameters:**
- `argv`: Optional[List[str]] - Список аргументов. Если None, использует sys.argv[1:]

**Returns:**
- `argparse.Namespace`: Объект с разобранными аргументами

**Available Arguments:**
- `--source`: str (required) - Путь к исходному изображению лица
- `--modelPath`: str (required) - Путь к модели AI
- `--resolution`: int (default: 128) - Разрешение обработки лица
- `--face_attribute_direction`: str - Путь к файлу направления атрибутов лица
- `--face_attribute_steps`: float (default: 0.0) - Шаги в направлении атрибута
- `--obs`: bool - Отправка в OBS Virtual Camera
- `--mouth_mask`: bool - Сохранение естественных движений губ
- `--delay`: int (default: 0) - Задержка в миллисекундах
- `--fps_delay`: bool - Показ FPS на экране
- `--enhance_res`: bool - Использование высокого разрешения камеры (1920x1080)

##### `cli(argv=None)`
CLI обертка для удобного вызова.

**Parameters:**
- `argv`: Optional[List[str]] - Аргументы командной строки

**Example:**
```python
from liveswapping.core.realtime import cli

cli(['--source', 'face.jpg', '--modelPath', 'model.pth', '--obs'])
```

#### Helper Functions

##### `load_model(model_path)`
Загрузка модели с TensorRT оптимизацией.

**Parameters:**
- `model_path`: str - Путь к файлу модели

**Returns:**
- Model object - Загруженная и оптимизированная модель

##### `create_source_latent(source_image, direction_path=None, steps=0.0)`
Создание латентного представления исходного лица.

**Parameters:**
- `source_image`: np.ndarray - Изображение исходного лица
- `direction_path`: Optional[str] - Путь к файлу направления атрибутов
- `steps`: float - Количество шагов в направлении атрибута

**Returns:**
- Optional[np.ndarray] - Латентное представление или None если лицо не найдено

---

### `liveswapping.core.video`

Обработка видео файлов.

#### Functions

##### `main(parsed_args=None)`
Основная функция обработки видео.

**Parameters:**
- `parsed_args`: Optional[argparse.Namespace] - Предварительно разобранные аргументы

**Returns:**
- `int`: Exit code

**Example:**
```python
from liveswapping.core.video import main, parse_arguments

args = parse_arguments([
    '--source', 'actor.jpg',
    '--target_video', 'movie.mp4',
    '--modelPath', 'model.pth',
    '--upscale', '2'
])
main(args)
```

##### `parse_arguments(argv=None)`
Парсинг аргументов для обработки видео.

**Available Arguments:**
- `--source`: str (required) - Исходное изображение лица
- `--target_video`: str (required) - Видео для обработки
- `--modelPath`: str (required) - Путь к модели
- `--resolution`: int (default: 128) - Разрешение обработки лица
- `--mouth_mask`: bool - Сохранение рта
- `--upscale`: int (default: 2) - Коэффициент увеличения
- `--bg_upsampler`: str (default: "realesrgan") - Тип фонового апскейлера
- `--bg_tile`: int (default: 400) - Размер тайла для обработки
- `--weight`: float (default: 0.5) - Вес смешивания
- `--std`: int (default: 1) - Стандартное отклонение шума
- `--blur`: int (default: 1) - Размытие

---

### `liveswapping.core.image_utils`

Утилиты обработки изображений.

#### Functions

##### `postprocess_face(face_tensor)`
Постобработка тензора лица в изображение.

**Parameters:**
- `face_tensor`: torch.Tensor - Тензор лица

**Returns:**
- `np.ndarray`: BGR изображение

**Example:**
```python
from liveswapping.core.image_utils import postprocess_face
import torch

face_tensor = model(target, source)  # Получаем тензор от модели
face_image = postprocess_face(face_tensor)
```

##### `getBlob(aimg, input_size=(128, 128))`
Создание DNN blob из изображения.

**Parameters:**
- `aimg`: np.ndarray - Входное изображение
- `input_size`: Tuple[int, int] - Целевой размер

**Returns:**
- `np.ndarray`: DNN blob

##### `getLatent(source_face)`
Извлечение латентного представления из лица.

**Parameters:**
- `source_face`: Face object - Объект лица с эмбеддингом

**Returns:**
- `np.ndarray`: Нормализованный латентный вектор

##### `blend_swapped_image(swapped_face, target_image, M)`
Смешивание замененного лица с целевым изображением.

**Parameters:**
- `swapped_face`: np.ndarray - Замененное лицо
- `target_image`: np.ndarray - Целевое изображение
- `M`: np.ndarray - Матрица аффинного преобразования

**Returns:**
- `np.ndarray`: Результат смешивания

##### `blend_swapped_image_gpu(swapped_face, target_image, M)`
GPU-ускоренная версия смешивания изображений.

**Parameters:**
- Same as `blend_swapped_image`

**Returns:**
- `np.ndarray`: GPU-ускоренный результат

---

## 🧠 AI Models API

### `liveswapping.ai_models.models`

Унифицированная система загрузки моделей.

#### Core Functions

##### `load_model(name, use_tensorrt=True, provider_type=None, **kwargs)`
Основная функция загрузки моделей.

**Parameters:**
- `name`: str - Имя модели из реестра или путь к файлу
- `use_tensorrt`: bool (default: True) - Включить torch-tensorrt оптимизацию
- `provider_type`: Optional[str] - Тип провайдера ('cuda', 'directml', 'openvino', 'cpu')
- `**kwargs`: Additional model parameters

**Returns:**
- Model object - Загруженная и оптимизированная модель

**Examples:**
```python
from liveswapping.ai_models.models import load_model

# Загрузка из реестра с TensorRT
model = load_model("reswapper128", use_tensorrt=True)

# Загрузка с конкретным провайдером
model = load_model("inswapper128", provider_type="cuda")

# Загрузка из файла
model = load_model("/path/to/model.pth", use_tensorrt=False)
```

##### `list_available_models()`
Список всех доступных моделей в реестре.

**Returns:**
- `Dict[str, Dict[str, Any]]`: Словарь с информацией о моделях

**Example:**
```python
models = list_available_models()
for name, info in models.items():
    print(f"{name}: {info['description']}")
    print(f"  Type: {info['type']}")
    print(f"  Size: {info.get('size', 'Unknown')}")
```

##### `get_model_type(model_path)`
Определение типа модели по пути к файлу.

**Parameters:**
- `model_path`: Path - Путь к файлу модели

**Returns:**
- `str`: Тип модели ('dfm', 'inswapper', 'styletransfer')

##### `create_session(model_path, provider=None)`
Создание ONNX Runtime сессии с оптимальными провайдерами.

**Parameters:**
- `model_path`: str - Путь к ONNX модели
- `provider`: Optional[str] - Конкретный провайдер

**Returns:**
- `onnxruntime.InferenceSession`: Настроенная сессия

#### Provider Management

##### `get_optimal_provider()`
Определение лучшего доступного провайдера для текущей системы.

**Returns:**
- `str`: Имя оптимального провайдера

**Example:**
```python
provider = get_optimal_provider()
print(f"Optimal provider: {provider}")
# Выводит: "cuda", "directml", "openvino", или "cpu"
```

##### `_create_providers(force_provider=None)`
Создание списка провайдеров ONNX Runtime.

**Parameters:**
- `force_provider`: Optional[str] - Принудительное использование провайдера

**Returns:**
- `List`: Список провайдеров в порядке приоритета

#### Model Registry

Доступные модели в реестре:

| Модель | Тип | Разрешение | Описание | Оптимизация |
|--------|-----|------------|----------|-------------|
| `reswapper128` | StyleTransfer | 128x128 | Быстрая, хорошее качество | TensorRT |
| `reswapper256` | StyleTransfer | 256x256 | Высокое качество, медленнее | TensorRT |
| `inswapper128` | InSwapper | 128x128 | Промышленный стандарт | ONNX Runtime |

---

## 🛠️ Utility APIs

### `liveswapping.utils.upscalers`

Апскейлинг и улучшение изображений.

#### Classes

##### `GFPGANUpscaler`
GFPGAN-based face restoration с TensorRT оптимизацией.

**Constructor:**
```python
GFPGANUpscaler(model_path=None, use_tensorrt=True, bg_upsampler=None)
```

**Parameters:**
- `model_path`: Optional[str] - Путь к модели GFPGAN (автозагрузка если None)
- `use_tensorrt`: bool - Включить TensorRT оптимизацию
- `bg_upsampler`: Optional[object] - Экземпляр background upsampler

**Methods:**

###### `upscale(image)`
Апскейлинг и улучшение изображения.

**Parameters:**
- `image`: np.ndarray - Входное изображение

**Returns:**
- `np.ndarray`: Улучшенное изображение

###### `enhance(image, **kwargs)`
Расширенное улучшение с дополнительными опциями.

**Parameters:**
- `image`: np.ndarray - Входное изображение
- `**kwargs`: Дополнительные параметры для enhance

**Returns:**
- `Tuple[List, np.ndarray, List]`: (cropped_faces, restored_img, restored_faces)

**Example:**
```python
from liveswapping.utils.upscalers import GFPGANUpscaler

upscaler = GFPGANUpscaler(use_tensorrt=True)

# Простое улучшение
enhanced = upscaler.upscale(image)

# Расширенное улучшение
cropped_faces, restored_img, restored_faces = upscaler.enhance(
    image,
    has_aligned=False,
    only_center_face=True,
    paste_back=True,
    weight=0.5
)
```

##### `RealESRGANUpscaler`
RealESRGAN-based upscaling с TensorRT оптимизацией.

**Constructor:**
```python
RealESRGANUpscaler(model_path=None, use_tensorrt=True, scale=2, tile=400)
```

#### Factory Functions

##### `create_optimized_gfpgan(model_path=None, use_tensorrt=True, bg_upsampler=None)`
Фабричная функция для создания оптимизированного GFPGAN.

**Returns:**
- `GFPGANUpscaler`: Настроенный экземпляр

##### `ensure_gfpgan_model()`
Обеспечивает загрузку модели GFPGAN.

**Returns:**
- `str`: Путь к модели GFPGAN

---

### `liveswapping.utils.gpu_utils`

GPU ускорение для numpy операций.

#### Classes

##### `GPUArrayManager`
Управление numpy/CuPy массивами для оптимальной производительности GPU.

**Constructor:**
```python
GPUArrayManager(use_cupy=True, verbose=False)
```

**Methods:**

###### `to_gpu(array)`
Конвертация numpy массива в CuPy массив.

**Parameters:**
- `array`: np.ndarray - Входной массив

**Returns:**
- `Union[np.ndarray, cupy.ndarray]`: GPU массив если доступен

###### `to_cpu(array)`
Конвертация CuPy массива обратно в numpy.

**Parameters:**
- `array`: Union[np.ndarray, cupy.ndarray] - Входной массив

**Returns:**
- `np.ndarray`: CPU массив

###### `synchronize()`
Синхронизация GPU операций.

**Example:**
```python
from liveswapping.utils.gpu_utils import GPUArrayManager

manager = GPUArrayManager(use_cupy=True)

# Перенос на GPU
gpu_array = manager.to_gpu(numpy_array)

# Обработка на GPU
result_gpu = gpu_operation(gpu_array)

# Перенос обратно на CPU
result = manager.to_cpu(result_gpu)
manager.synchronize()
```

#### Functions

##### `accelerated_histogram_matching(source_image, target_image, alpha=0.5, use_gpu=True)`
GPU-ускоренное сопоставление гистограмм.

**Parameters:**
- `source_image`: np.ndarray - Исходное изображение
- `target_image`: np.ndarray - Целевое изображение
- `alpha`: float - Коэффициент смешивания (0.0-1.0)
- `use_gpu`: bool - Использовать GPU ускорение

**Returns:**
- `np.ndarray`: Изображение с сопоставленной гистограммой

##### `get_optimal_config()`
Получение оптимальной конфигурации для текущей системы.

**Returns:**
- `Dict[str, Any]`: Словарь с оптимальными настройками
  - `use_cupy`: bool
  - `device_count`: int
  - `memory_gb`: float
  - `compute_capability`: str
  - `recommended_batch_size`: int
  - `use_mixed_precision`: bool

##### `print_gpu_info()`
Вывод подробной информации о GPU ускорении.

##### `get_provider_info()`
Информация о доступных провайдерах.

**Returns:**
- `List[Dict[str, Any]]`: Список словарей с информацией о провайдерах

**Example:**
```python
from liveswapping.utils.gpu_utils import (
    print_gpu_info,
    get_optimal_config,
    accelerated_histogram_matching
)

# Информация о системе
print_gpu_info()

# Оптимальные настройки
config = get_optimal_config()
print(f"Recommended batch size: {config['recommended_batch_size']}")

# GPU-ускоренная обработка
matched = accelerated_histogram_matching(
    source_image,
    target_image,
    alpha=0.7,
    use_gpu=True
)
```

---

### `liveswapping.utils.adaptive_cupy`

Адаптивное ускорение CuPy на основе размера изображения.

#### Classes

##### `AdaptiveCuPyProcessor`
Автоматический выбор GPU/CPU на основе оптимальной производительности.

##### `AdaptiveColorTransfer`
GPU-ускоренный цветовой перенос с автоматическим fallback на CPU.

##### `AdaptiveBlending`
GPU-ускоренное смешивание изображений с оптимизацией по размеру.

#### Functions

##### `create_adaptive_processor(image_height)`
Создание адаптивного процессора на основе размера изображения.

**Parameters:**
- `image_height`: int - Высота обрабатываемых изображений

**Returns:**
- `AdaptiveCuPyProcessor`: Настроенный процессор

**Example:**
```python
from liveswapping.utils.adaptive_cupy import create_adaptive_processor

# Создание процессора для 1080p изображений
processor = create_adaptive_processor(1080)

# Использование с color transfer и blending
color_transfer = AdaptiveColorTransfer(processor)
blending = AdaptiveBlending(processor)
```

---

## 🖥️ GUI Components

### `liveswapping.gui.realtime_gui`

GUI для real-time обработки.

#### Functions

##### `main()`
Запуск real-time GUI приложения.

**Example:**
```python
from liveswapping.gui.realtime_gui import main

# Запуск GUI
main()
```

### `liveswapping.gui.video_gui`

GUI для обработки видео.

#### Functions

##### `main()`
Запуск GUI приложения для обработки видео.

**Example:**
```python
from liveswapping.gui.video_gui import main

# Запуск GUI
main()
```

---

## ⚙️ Конфигурация

### Environment Variables

Переменные окружения для настройки системы:

```bash
# Отключение verbose логов
export ONNX_LOG_LEVEL=3
export OMP_NUM_THREADS=1

# CUDA настройки
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda

# Пути к моделям
export LIVESWAPPING_MODELS_DIR=/path/to/models
```

### Configuration Files

Поддерживается конфигурационный файл `config.json`:

```json
{
    "default_model": "models/reswapper128.pth",
    "default_resolution": 128,
    "use_tensorrt": true,
    "provider": "cuda",
    "upscale_factor": 2,
    "enable_mouth_mask": false,
    "model_download_url": "https://custom-server.com/models/"
}
```

### Provider Configuration

Настройка провайдеров:

```python
from liveswapping.ai_models.models import _create_providers

# Автоматические провайдеры
providers = _create_providers()

# Принудительный провайдер
cuda_providers = _create_providers(force_provider="cuda")
directml_providers = _create_providers(force_provider="directml")
```

---

## 📊 Performance Monitoring

### Benchmarking

```python
from liveswapping.utils.gpu_utils import analyze_cupy_performance

# Анализ производительности CuPy
analyze_cupy_performance()
```

### Resource Monitoring

```python
import psutil
import torch

def get_system_stats():
    stats = {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent
    }
    
    if torch.cuda.is_available():
        stats.update({
            'gpu_memory_allocated': torch.cuda.memory_allocated(0) / 1024**3,
            'gpu_memory_cached': torch.cuda.memory_reserved(0) / 1024**3,
            'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3
        })
    
    return stats
```

---

## 🔍 Error Handling

### Exception Types

Основные типы исключений:

```python
# Model loading errors
class ModelLoadError(Exception):
    pass

# GPU memory errors  
class GPUMemoryError(Exception):
    pass

# Face detection errors
class FaceDetectionError(Exception):
    pass
```

### Error Handling Patterns

```python
from liveswapping.ai_models.models import load_model

try:
    model = load_model("reswapper256", provider_type="cuda")
except RuntimeError as e:
    if "out of memory" in str(e):
        # Fallback to CPU
        model = load_model("reswapper256", provider_type="cpu")
    else:
        raise e
```

---

## 📚 Type Hints

Основные типы:

```python
from typing import Union, Optional, List, Dict, Any, Tuple
import numpy as np
import torch

# Common types
ImageArray = np.ndarray  # Shape: (H, W, C), dtype: uint8
TensorImage = torch.Tensor  # Shape: (B, C, H, W), dtype: float32
BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)
Landmarks = np.ndarray  # Shape: (N, 2)

# Model types
ModelPath = Union[str, Path]
ProviderType = Literal["cuda", "directml", "openvino", "cpu"]
```

---

## 🔗 See Also

- **[🏠 Home](Home)** - Главная страница wiki
- **[🎯 Quick Start](Quick-Start)** - Быстрый старт
- **[👤 User Guide](User-Guide)** - Руководство пользователя  
- **[🔧 Troubleshooting](Troubleshooting)** - Решение проблем

---

*[⬅️ Troubleshooting](Troubleshooting) | [🏠 Главная](Home) | [➡️ Performance Optimization](Performance-Optimization)*
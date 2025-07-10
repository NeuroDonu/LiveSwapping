# LiveSwapping - Реалтайм Face Swap

LiveSwapping - это продвинутая система замены лиц в реальном времени с поддержкой множества моделей и технологий.

## Поддерживаемые модели

- **StyleTransfer** - собственные модели на базе PyTorch (.pth) с torch-tensorrt оптимизацией
- **DFM** - модели Deep Face Model в формате ONNX (.onnx, .dfm)
- **inswapper128** - модель InsightFace для замены лиц (128x128)
- **reswapper128/256** - альтернативные модели замены лиц

## Установка

### 1. Клонирование репозитория
```bash
git clone https://github.com/NeuroDonu/LiveSwapping.git
cd LiveSwapping
```

### 2. Автоматическая установка (рекомендуется)

#### Системный Python
```bash
# Windows
install.bat

# Linux/macOS
./install.sh
```

#### Кастомный Python
```bash
# Windows - с относительным путем
install.bat "..\python311\python.exe"

# Windows - с абсолютным путем
install.bat "C:\Python311\python.exe"

# Linux/macOS
./install.sh "/usr/bin/python3.11"
./install.sh "../python311/bin/python"
```

**Особенности автоматической установки:**
- ✅ Автоматическая детекция GPU (NVIDIA/AMD/Intel)
- ✅ Выбор оптимальной конфигурации
- ✅ Установка `uv` для быстрой загрузки пакетов
- ✅ Поддержка CUDA 12.1 и 12.8 с выбором версии
- ✅ Обработка относительных путей к Python

### 3. Ручная установка

#### Создание окружения Python
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# или
venv\Scripts\activate  # Windows
```

#### Установка зависимостей
```bash
# Используйте специализированные установщики
cd installers

# CUDA GPU
install_cuda.bat "path\to\python.exe"    # Windows
./install_cuda.sh "/path/to/python"      # Linux/macOS

# CPU only
install_cpu.bat "path\to\python.exe"     # Windows  
./install_cpu.sh "/path/to/python"       # Linux/macOS

# Подробности в installers/README.md
```

### 4. Загрузка моделей

#### Автоматическая загрузка
```python
from liveswapping.download_models import ensure_model

# Загрузка inswapper128
ensure_model("inswapper128")

# Загрузка reswapper моделей
ensure_model("reswapper128")
ensure_model("reswapper256")
```

#### Ручная загрузка
1. Создайте папку `models/`
2. Поместите .pth файлы для StyleTransfer моделей
3. Поместите .onnx файлы для DFM/inswapper моделей

## Использование

### 1. Реалтайм GUI
```bash
python liveswapping/realtime_gui.py
```

### 2. Видео GUI
```bash
python liveswapping/video_gui.py
```

### 3. CLI для реалтайм
```bash
python -m liveswapping.realtime \
    --source_image path/to/source.jpg \
    --model_path models/your_model.pth \
    --camera_id 0
```

### 4. CLI для видео
```bash
python -m liveswapping.video \
    --source_image path/to/source.jpg \
    --target_video path/to/video.mp4 \
    --model_path models/your_model.onnx \
    --output_path output.mp4
```

### 5. Программный API
```python
from liveswapping.models import load_model
from liveswapping import realtime, video

# Загрузка модели с torch-tensorrt оптимизацией
model = load_model("reswapper128", use_tensorrt=True)

# Загрузка без оптимизации
model = load_model("inswapper128", use_tensorrt=False)

# Использование в коде
result = realtime.process_frame(frame, model, source_embedding)
```

## Оптимизация производительности

### Torch-TensorRT (рекомендуется)

Для максимальной производительности PyTorch моделей используется **torch-tensorrt**:

```python
# Автоматическая оптимизация при загрузке
model = load_model("reswapper128", use_tensorrt=True)

# Ручная оптимизация
import torch_tensorrt
compiled_model = torch_tensorrt.compile(
    model,
    inputs=[target_input, source_input],
    enabled_precisions={torch.float32},
    ir="torch_compile"
)
```

#### Преимущества torch-tensorrt:
- **До 3-5x ускорение** для PyTorch моделей
- **Автоматическая оптимизация** графа вычислений
- **Гибридное выполнение** (TensorRT + PyTorch)
- **JIT компиляция** с автоматической рекомпиляцией

#### Поддерживаемые модели:
- ✅ **StyleTransfer** (.pth) - автоматическая оптимизация
- ✅ **GFPGAN** - оптимизация внутренней PyTorch модели
- ✅ **RealESRGAN** - оптимизация backbone модели
- ⚪ **DFM/InSwapper** (.onnx) - используют TensorRT Provider

#### Системные требования:
- NVIDIA GPU с Compute Capability ≥ 7.0
- CUDA ≥ 11.8
- TensorRT ≥ 8.6  
- PyTorch ≥ 2.0
- NumPy ≥ 1.24.0, < 2.0 (совместимость)

### Установка torch-tensorrt

```bash
# Автоматически устанавливается с requirements.txt
pip install torch-tensorrt>=2.0.0

# Или отдельно
pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu128
```

### GPU ускорение NumPy с CuPy (опционально)

Для дополнительного ускорения NumPy операций можно установить **CuPy**:

```bash
# Установка CuPy для CUDA 12.x
pip install cupy-cuda12x>=12.0.0

# Проверка производительности
python liveswapping/gpu_utils.py
```

#### Когда CuPy полезен:
- ✅ **Histogram matching** - до 2-4x ускорение
- ✅ **Face alignment** - GPU вычисления трансформаций
- ✅ **Color correction** - массовые операции с пикселями
- ⚪ **Малые изображения** (128x128) - ограниченная польза
- ⚪ **Старые GPU** (compute capability < 6.0) - минимальная польза

### Оптимизированные апскейлеры

```python
from liveswapping.upscalers import create_optimized_gfpgan, RealESRGANUpscaler

# GFPGAN с torch-tensorrt оптимизацией
gfpgan = create_optimized_gfpgan(use_tensorrt=True)

# RealESRGAN с torch-tensorrt оптимизацией  
realesrgan = RealESRGANUpscaler(use_tensorrt=True)

# Использование
enhanced_image = gfpgan.upscale(image)
```

## Поддерживаемые провайдеры

### ONNX Runtime провайдеры
- **CPUExecutionProvider** - базовый CPU провайдер
- **CUDAExecutionProvider** - NVIDIA GPU ускорение
- **TensorrtExecutionProvider** - оптимизация через TensorRT (FP16 отключен)

### TensorRT конфигурация
Для максимальной производительности используются следующие опции:
- `trt_fp16_enable: "0"` - отключение FP16 для стабильности
- `trt_engine_cache_enable: "1"` - кеширование движков

## Типы моделей

### StyleTransfer (.pth) - с torch-tensorrt
```python
model = load_model("reswapper128", use_tensorrt=True)
# Автоматически оптимизируется с torch-tensorrt
```

### DFM (.onnx, .dfm) - с TensorRT Provider
```python
model = load_model("dfm_model")
# Использует TensorrtExecutionProvider в ONNX Runtime
```

### inswapper128 - с TensorRT Provider
```python
model = load_model("inswapper128")
# Использует TensorrtExecutionProvider в ONNX Runtime
```

## Примеры использования

### Замена лица на веб-камере
```python
import cv2
from liveswapping.models import load_model
from liveswapping.realtime import process_frame

# Загрузка с torch-tensorrt оптимизацией
model = load_model("reswapper128", use_tensorrt=True)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = process_frame(frame, model, source_embedding)
    cv2.imshow('LiveSwapping', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Обработка видео файла
```python
from liveswapping.video import main, parse_arguments

args = parse_arguments()
args.source_image = "source.jpg"
args.target_video = "input.mp4"
args.model_path = "models/reswapper128.pth"  # Автоматически оптимизируется
args.output_path = "output.mp4"

main(args)
```

## Конфигурация

### GPU ускорение
Для использования GPU убедитесь, что установлен `onnxruntime-gpu`:
```bash
pip install onnxruntime-gpu
```

### TensorRT (опционально)
Для максимальной производительности установите TensorRT:
```bash
pip install tensorrt
```

### Отключение torch-tensorrt оптимизации
```python
# При проблемах с torch-tensorrt
model = load_model("reswapper128", use_tensorrt=False)
```

## Структура проекта

```
LiveSwapping/
├── liveswapping/          # Основной пакет
│   ├── models.py          # Унифицированная загрузка моделей + torch-tensorrt
│   ├── realtime.py        # Реалтайм обработка
│   ├── video.py           # Видео обработка
│   ├── realtime_gui.py    # GUI для реалтайм
│   ├── video_gui.py       # GUI для видео
│   └── download_models.py # Автозагрузка моделей
├── dfm/                   # DFM модели
├── models/                # Папка для моделей
├── requirements.txt       # Зависимости + torch-tensorrt
└── README.md             # Документация
```

## Производительность

### Бенчмарки (на RTX 4090):

| Компонент | Без оптимизации | С оптимизацией | Ускорение | Технология |
|-----------|----------------|----------------|-----------|------------|
| reswapper128 | ~15 FPS | ~45 FPS | **3.0x** | torch-tensorrt |
| reswapper256 | ~8 FPS | ~25 FPS | **3.1x** | torch-tensorrt |
| StyleTransfer | ~12 FPS | ~40 FPS | **3.3x** | torch-tensorrt |
| GFPGAN | ~2.5 FPS | ~7 FPS | **2.8x** | torch-tensorrt |
| RealESRGAN | ~1.8 FPS | ~5.2 FPS | **2.9x** | torch-tensorrt |
| Histogram matching | ~25 ms | ~8 ms | **3.1x** | CuPy |
| Face alignment | ~5 ms | ~2 ms | **2.5x** | CuPy |

### Рекомендации по NumPy/CuPy:

**NumPy 2.0 совместимость**: Пока используем NumPy 1.x из-за проблем совместимости с некоторыми библиотеками (InsightFace, GFPGAN). NumPy 2.0 будет поддержан после обновления зависимостей.

**CuPy ускорение наиболее эффективно для**:
- 📐 Крупных изображений (≥512x512)  
- 🎨 Множественной цветокоррекции
- 📊 Операций с гистограммами
- 🔄 Batch обработки

*Результаты могут варьироваться в зависимости от конфигурации системы*

### Настройка оптимизации

```python
# Отключение torch-tensorrt для отладки
model = load_model("reswapper128", use_tensorrt=False)
gfpgan = create_optimized_gfpgan(use_tensorrt=False)

# Включение только для конкретных моделей
face_model = load_model("reswapper128", use_tensorrt=True)
upscaler = create_optimized_gfpgan(use_tensorrt=False)  # Если проблемы с GFPGAN
```

## Известные проблемы и решения

### 1. Проблемы с basicsr (только для видео обработки)

**Проблема**: Ошибка импорта `from torchvision.transforms.functional_tensor import rgb_to_grayscale`

**Причина**: В новых версиях torchvision функция `rgb_to_grayscale` была перемещена из `functional_tensor` в `functional`.

**Автоматическое решение**:
```bash
python liveswapping/fix_basicsr.py
```

**Ручное решение**:
1. Найдите файл `degradations.py` в вашем basicsr:
   - Conda: `<env_path>/Lib/site-packages/basicsr/data/degradations.py`
   - venv: `.\liveswapping\venv\Lib\site-packages\basicsr\data\degradations.py`

2. Измените строку 8:
   ```python
   # Было:
   from torchvision.transforms.functional_tensor import rgb_to_grayscale
   
   # Должно быть:
   from torchvision.transforms.functional import rgb_to_grayscale
   ```

**Примечание**: Этот фикс нужен **только для видео обработки**. Реалтайм face swap работает без этого исправления.

### 2. Другие проблемы

1. **CUDA Out of Memory** - уменьшите разрешение или используйте CPU
2. **torch-tensorrt compilation failed** - используйте `use_tensorrt=False`
3. **TensorRT version mismatch** - обновите CUDA/TensorRT драйверы

### 3. Отладка проблем с basicsr

Если автоматический фикс не сработал:

```bash
# Проверить установку basicsr
pip show basicsr

# Найти файл вручную
python -c "import basicsr; print(basicsr.__file__)"

# Проверить содержимое degradations.py
python -c "
import basicsr.data.degradations
import inspect
print(inspect.getfile(basicsr.data.degradations))
"
```

## Лицензия

См. файл LICENSE для деталей.

## Участие в разработке

1. Форкните репозиторий
2. Создайте ветку для фичи
3. Внесите изменения
4. Создайте Pull Request

## ⭐ История звёзд

<a href="https://star-history.com/#NeuroDonu/LiveSwapping&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=NeuroDonu/LiveSwapping&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=NeuroDonu/LiveSwapping&type=Date" />
   <img alt="График истории звёзд" src="https://api.star-history.com/svg?repos=NeuroDonu/LiveSwapping&type=Date" />
 </picture>
</a>

[![Количество звёзд во времени](https://starchart.cc/NeuroDonu/LiveSwapping.svg?variant=adaptive)](https://starchart.cc/NeuroDonu/LiveSwapping)

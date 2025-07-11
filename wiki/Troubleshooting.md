# 🔧 Решение проблем (Troubleshooting)

Подробное руководство по диагностике и решению технических проблем LiveSwapping.

## 📚 Содержание

1. [Диагностика системы](#-диагностика-системы)
2. [Проблемы установки](#-проблемы-установки)
3. [Ошибки GPU и CUDA](#-ошибки-gpu-и-cuda)
4. [Проблемы с моделями](#-проблемы-с-моделями)
5. [Ошибки времени выполнения](#-ошибки-времени-выполнения)
6. [Проблемы производительности](#-проблемы-производительности)
7. [Специфичные ошибки](#-специфичные-ошибки)

---

## 🩺 Диагностика системы

### Общая проверка работоспособности

Запустите полную диагностику системы:

```python
# diagnostic_full.py
import sys
print(f"Python версия: {sys.version}")

# 1. Проверка импорта основных модулей
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"✅ GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError as e:
    print(f"❌ PyTorch import error: {e}")

try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV import error: {e}")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy import error: {e}")

# 2. Проверка LiveSwapping модулей
try:
    from liveswapping.utils.gpu_utils import print_gpu_info, get_provider_info
    print("\n=== GPU ИНФОРМАЦИЯ ===")
    print_gpu_info()
    
    print("\n=== ПРОВАЙДЕРЫ ===")
    providers = get_provider_info()
    for provider in providers:
        status = "✅" if provider['available'] else "❌"
        print(f"{status} {provider['name'].upper()}")
        
except ImportError as e:
    print(f"❌ LiveSwapping import error: {e}")

# 3. Проверка моделей
try:
    from liveswapping.ai_models.models import list_available_models
    models = list_available_models()
    print(f"\n✅ Доступные модели: {len(models)}")
    for name in models:
        print(f"  - {name}")
except Exception as e:
    print(f"❌ Models error: {e}")

print("\n=== ДИАГНОСТИКА ЗАВЕРШЕНА ===")
```

### Базовые проверки

#### Проверка Python и виртуального окружения
```bash
# Проверка версии Python
python --version

# Проверка активации venv
which python  # Linux/macOS
where python  # Windows

# Проверка установленных пакетов
pip list | grep torch
pip list | grep opencv
pip list | grep onnx
```

#### Проверка PATH и переменных окружения
```bash
# Проверка CUDA
echo $CUDA_HOME
echo $PATH | grep cuda

# Проверка NVIDIA
nvidia-smi
nvcc --version
```

---

## 📦 Проблемы установки

### "No module named 'liveswapping'"

**Причина:** Не активировано виртуальное окружение или неправильная структура папок.

**Решение:**
```bash
# 1. Проверьте что находитесь в правильной папке
pwd
ls -la  # Должны видеть run.py и папку liveswapping/

# 2. Активируйте виртуальное окружение
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# 3. Проверьте установку
python -c "import liveswapping; print('OK')"

# 4. Если не помогает - переустановка
pip install -e .
```

### Ошибки компиляции при установке

**Ошибка:** `Microsoft Visual C++ 14.0 is required`

**Решение (Windows):**
```bash
# 1. Установите Microsoft Visual C++ Build Tools
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# 2. Или установите Visual Studio Community
# https://visualstudio.microsoft.com/downloads/

# 3. Обновите pip и setuptools
pip install --upgrade pip setuptools wheel

# 4. Переустановите проблемные пакеты
pip install --force-reinstall --no-cache-dir torch torchvision
```

### Конфликты зависимостей

**Ошибка:** `pip dependency resolver conflicts`

**Решение:**
```bash
# 1. Очистите pip cache
pip cache purge

# 2. Создайте новое виртуальное окружение
rm -rf venv
python -m venv venv
source venv/bin/activate

# 3. Обновите pip
pip install --upgrade pip

# 4. Установите зависимости поэтапно
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/requirements_cuda.txt
```

### Проблемы с сетью при установке

**Ошибка:** `Connection timeout` или `SSL certificate verify failed`

**Решение:**
```bash
# 1. Обновите certificates
pip install --upgrade certifi

# 2. Используйте --trusted-host
pip install --trusted-host pypi.org --trusted-host pypi.python.org torch

# 3. Настройте proxy (если нужно)
pip install --proxy http://user:password@proxy.server:port torch

# 4. Локальная установка из файлов
# Скачайте .whl файлы вручную и установите:
pip install torch-1.13.0-cp39-cp39-win_amd64.whl
```

---

## 🎮 Ошибки GPU и CUDA

### "CUDA out of memory"

**Причина:** Недостаточно видеопамяти для модели.

**Диагностика:**
```python
import torch
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU memory: {gpu_memory:.1f} GB")
    
    # Проверка текущего использования
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
    print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")
```

**Решение:**
```bash
# 1. Очистите GPU память
python -c "import torch; torch.cuda.empty_cache()"

# 2. Используйте меньшую модель
python -m liveswapping.run realtime \
    --source face.jpg \
    --modelPath models/reswapper128.pth \
    --resolution 128

# 3. Переключитесь на CPU
python -m liveswapping.run realtime \
    --source face.jpg \
    --modelPath models/reswapper128.pth \
    --provider cpu

# 4. Закройте другие GPU приложения
nvidia-smi  # Проверьте что занимает память
```

### "CUDA driver version is insufficient"

**Причина:** Устаревший драйвер NVIDIA.

**Решение:**
```bash
# 1. Проверьте версию драйвера
nvidia-smi

# 2. Требуемые версии:
# CUDA 12.1 требует driver >= 530.30.02
# CUDA 12.8 требует driver >= 550.54.15

# 3. Обновите драйвер:
# https://www.nvidia.com/drivers/

# 4. Перезагрузите систему

# 5. Проверьте совместимость
python -c "import torch; print(torch.cuda.is_available())"
```

### "No CUDA-capable device is detected"

**Диагностика и решение:**
```bash
# 1. Проверьте обнаружение GPU
lspci | grep -i nvidia  # Linux
nvidia-smi

# 2. Проверьте загрузку драйвера
lsmod | grep nvidia  # Linux

# 3. Переустановите драйвер
sudo apt purge nvidia*  # Linux
sudo apt install nvidia-driver-525

# 4. Проверьте CUDA установку
nvcc --version
/usr/local/cuda/bin/nvcc --version

# 5. Установите переменные окружения
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
```

### DirectML ошибки (AMD GPU)

**Ошибка:** `torch-directml not found`

**Решение:**
```bash
# 1. Установите torch-directml
pip install torch-directml

# 2. Проверьте поддержку
python -c "import torch_directml; print(torch_directml.device_count())"

# 3. Используйте DirectML provider
python -m liveswapping.run realtime \
    --source face.jpg \
    --modelPath models/reswapper128.pth \
    --provider directml
```

---

## 🧠 Проблемы с моделями

### "Model file not found"

**Диагностика:**
```python
import os
from pathlib import Path

# Проверьте структуру папок
models_dir = Path("models")
print(f"Models directory exists: {models_dir.exists()}")
if models_dir.exists():
    print("Files in models/:")
    for file in models_dir.iterdir():
        print(f"  {file.name} ({file.stat().st_size / 1024**2:.1f} MB)")
```

**Решение:**
```python
# 1. Автоматическая загрузка
from liveswapping.ai_models.download_models import ensure_model

try:
    model_path = ensure_model("reswapper128")
    print(f"Model downloaded to: {model_path}")
except Exception as e:
    print(f"Download failed: {e}")

# 2. Ручная загрузка
# Скачайте модели с GitHub Releases и поместите в models/
```

### "Model loading failed"

**Ошибка:** Corrupted model file или несовместимость версий.

**Решение:**
```bash
# 1. Проверьте целостность файла
ls -la models/  # Размер должен быть > 100MB

# 2. Удалите и перекачайте модель
rm models/reswapper128.pth
python -c "from liveswapping.ai_models.download_models import ensure_model; ensure_model('reswapper128')"

# 3. Проверьте версию PyTorch
pip install torch==2.0.1  # Совместимая версия
```

### "torch-tensorrt compilation failed"

**Это не критичная ошибка.** TensorRT отключается автоматически.

**Для исправления:**
```python
# 1. Обновите torch-tensorrt
pip install --upgrade torch-tensorrt

# 2. Проверьте совместимость CUDA
import torch
print(f"PyTorch CUDA: {torch.version.cuda}")
print(f"System CUDA: $(nvcc --version)")

# 3. Отключите TensorRT если проблемы
model = load_model("reswapper128", use_tensorrt=False)
```

---

## ⚠️ Ошибки времени выполнения

### "No face detected in source image"

**Диагностика:**
```python
import cv2
from insightface.app import FaceAnalysis

# Проверьте изображение
img = cv2.imread("source.jpg")
print(f"Image shape: {img.shape if img is not None else 'Failed to load'}")

# Проверьте детекцию лиц
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(512, 512))

faces = face_app.get(img)
print(f"Faces detected: {len(faces)}")

if len(faces) > 0:
    face = faces[0]
    print(f"Face bbox: {face.bbox}")
    print(f"Face landmarks: {face.kps.shape if face.kps is not None else 'None'}")
```

**Решение:**
```bash
# 1. Используйте качественное фото
# - Лицо четко видно (фронтальный ракурс)
# - Хорошее освещение
# - Без очков, масок, шляп
# - Разрешение 512x512+ пикселей

# 2. Проверьте формат файла
file source.jpg  # Должен быть JPEG/PNG

# 3. Проверьте размер файла
ls -la source.jpg  # Не должен быть 0 байт
```

### "Webcam not found" / "Camera initialization failed"

**Диагностика:**
```python
import cv2

# Проверьте доступные камеры
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i}: Available")
        cap.release()
    else:
        print(f"Camera {i}: Not available")
```

**Решение:**
```bash
# 1. Проверьте разрешения (Linux)
sudo usermod -a -G video $USER  # Добавить пользователя в группу video
# Перелогиньтесь

# 2. Проверьте устройства (Linux)
ls /dev/video*

# 3. Попробуйте другой ID камеры
python -m liveswapping.run realtime \
    --source face.jpg \
    --modelPath models/reswapper128.pth \
    --camera_id 1

# 4. Закройте другие приложения, использующие камеру
```

### "basicsr import error"

**Ошибка:** `ModuleNotFoundError: No module named 'basicsr'` или проблемы с degradations.

**Автоматическое решение:**
```bash
python liveswapping/utils/fix_basicsr.py
```

**Ручное решение:**
```python
# 1. Переустановите basicsr
pip uninstall basicsr
pip install basicsr

# 2. Исправьте файл degradations.py
# Найдите файл: site-packages/basicsr/data/degradations.py
# Замените строку:
# from torchvision.transforms.functional_tensor import rgb_to_grayscale
# на:
# from torchvision.transforms.functional import rgb_to_grayscale
```

---

## 🐌 Проблемы производительности

### Низкий FPS в real-time режиме

**Диагностика:**
```python
from liveswapping.utils.gpu_utils import analyze_cupy_performance, get_optimal_config

# Проверьте производительность системы
analyze_cupy_performance()

# Получите рекомендуемые настройки
config = get_optimal_config()
print(f"Recommended batch size: {config['recommended_batch_size']}")
print(f"Use mixed precision: {config['use_mixed_precision']}")
```

**Оптимизация:**
```bash
# 1. Используйте быструю модель
python -m liveswapping.run realtime \
    --source face.jpg \
    --modelPath models/reswapper128.pth \
    --resolution 128

# 2. Оптимизируйте настройки
python -m liveswapping.run realtime \
    --source face.jpg \
    --modelPath models/reswapper128.pth \
    --delay 0 \
    --resolution 128

# 3. Закройте ненужные приложения
# 4. Используйте SSD для хранения моделей
```

### Медленная обработка видео

**Оптимизация:**
```bash
# 1. Отключите upscaling для тестов
python -m liveswapping.run video \
    --source face.jpg \
    --target_video video.mp4 \
    --modelPath models/reswapper128.pth
    # Без --upscale

# 2. Используйте меньшее разрешение
python -m liveswapping.run video \
    --source face.jpg \
    --target_video video.mp4 \
    --modelPath models/reswapper128.pth \
    --resolution 128

# 3. Обрабатывайте короткие сегменты
ffmpeg -i long_video.mp4 -t 60 -c copy short_test.mp4
```

---

## 🔍 Специфичные ошибки

### OBS Virtual Camera не работает

**Проблема:** OBS не видит виртуальную камеру.

**Решение:**
```bash
# 1. Установите OBS Virtual Camera plugin
# https://obsproject.com/forum/resources/obs-virtualcam.949/

# 2. Запустите OBS перед LiveSwapping

# 3. Проверьте права доступа (Linux)
sudo modprobe v4l2loopback

# 4. Добавьте источник в OBS:
# Sources -> Add -> Video Capture Device -> OBS Virtual Camera
```

### Проблемы с audio в видео

**Проблема:** Звук не сохраняется или искажается.

**Диагностика:**
```bash
# Проверьте аудио в исходном видео
ffmpeg -i input_video.mp4 -hide_banner

# Проверьте кодеки
ffprobe -v quiet -print_format json -show_format -show_streams input_video.mp4
```

**Решение:**
```bash
# 1. Убедитесь что moviepy установлен с аудио поддержкой
pip install moviepy[optional]

# 2. Установите ffmpeg
# Ubuntu: sudo apt install ffmpeg
# Windows: https://ffmpeg.org/download.html

# 3. Конвертируйте аудио формат при необходимости
ffmpeg -i input.mp4 -c:v copy -c:a aac output.mp4
```

### Проблемы с памятью при длинных видео

**Ошибка:** `MemoryError` или system freeze.

**Решение:**
```bash
# 1. Увеличьте virtual memory (Windows)
# Settings -> System -> About -> Advanced system settings -> Performance Settings -> Virtual memory

# 2. Обрабатывайте видео частями
ffmpeg -i long_video.mp4 -t 300 -c copy part1.mp4  # Первые 5 минут
ffmpeg -i long_video.mp4 -ss 300 -t 300 -c copy part2.mp4  # Следующие 5 минут

# 3. Используйте меньшее разрешение
# 4. Очищайте временные файлы
rm -rf temp_results*
```

---

## 🛠️ Продвинутая диагностика

### Логирование для отладки

```python
import logging

# Настройте детальное логирование
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('liveswapping_debug.log'),
        logging.StreamHandler()
    ]
)

# Запустите с логированием
logger = logging.getLogger('liveswapping')
logger.info("Starting diagnostics...")
```

### Мониторинг ресурсов

```python
import psutil
import time

def monitor_resources(duration=60):
    """Мониторинг использования ресурсов."""
    for i in range(duration):
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
            print(f"CPU: {cpu}% | RAM: {memory}% | GPU RAM: {gpu_memory:.1f}GB")
        else:
            print(f"CPU: {cpu}% | RAM: {memory}%")
        
        time.sleep(1)

# Запустите мониторинг
monitor_resources(30)
```

### Тест производительности

```python
import time
import numpy as np
from liveswapping.ai_models.models import load_model

def benchmark_model(model_name="reswapper128", iterations=10):
    """Бенчмарк производительности модели."""
    
    model = load_model(model_name, use_tensorrt=True)
    
    # Создайте тестовые данные
    if "128" in model_name:
        target_size = (1, 3, 128, 128)
    else:
        target_size = (1, 3, 256, 256)
    
    target_tensor = torch.randn(target_size).cuda()
    source_latent = torch.randn(1, 512).cuda()
    
    # Прогрев
    for _ in range(3):
        with torch.no_grad():
            _ = model(target_tensor, source_latent)
    
    # Бенчмарк
    times = []
    for i in range(iterations):
        start_time = time.time()
        with torch.no_grad():
            result = model(target_tensor, source_latent)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Iteration {i+1}: {times[-1]:.4f}s")
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    print(f"\nAverage time: {avg_time:.4f}s")
    print(f"Average FPS: {fps:.1f}")

# Запустите бенчмарк
benchmark_model("reswapper128")
```

---

## 🆘 Получение помощи

Если проблема не решена:

1. **Соберите информацию:**
   - Запустите полную диагностику (см. начало страницы)
   - Скопируйте полный текст ошибки
   - Укажите ОС и версию Python
   - Приложите логи

2. **Создайте issue:**
   - **[GitHub Issues](https://github.com/your-repo/issues)**
   - Используйте шаблон для баг-репортов
   - Приложите диагностическую информацию

3. **Дополнительные ресурсы:**
   - **[❓ FAQ](FAQ)** - частые вопросы  
   - **[🐛 Известные проблемы](Known-Issues)** - текущие ограничения
   - **[📋 API Reference](API-Reference)** - техническая документация

---

*[⬅️ FAQ](FAQ) | [🏠 Главная](Home) | [➡️ API Reference](API-Reference)*
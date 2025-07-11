# 🐛 Известные проблемы

Список текущих ограничений, известных багов и планируемых исправлений LiveSwapping.

## 📚 Содержание

1. [Критические проблемы](#-критические-проблемы)
2. [Ограничения производительности](#-ограничения-производительности)
3. [Проблемы совместимости](#-проблемы-совместимости)
4. [GUI проблемы](#-gui-проблемы)
5. [Планируемые исправления](#-планируемые-исправления)
6. [Обходные пути](#-обходные-пути)

---

## 🚨 Критические проблемы

### basicsr ImportError (ВЫСОКИЙ ПРИОРИТЕТ)

**Проблема:** Ошибка импорта `basicsr.data.degradations`
```
ImportError: cannot import name 'rgb_to_grayscale' from 'torchvision.transforms.functional_tensor'
```

**Затронутые компоненты:**
- GFPGAN upscaler
- RealESRGAN background enhancement

**Статус:** ⚠️ Временное решение доступно

**Обходной путь:**
```bash
# Автоматическое исправление
python liveswapping/utils/fix_basicsr.py

# Или ручное
pip uninstall basicsr
pip install basicsr --no-deps
```

**Планируемое исправление:** v1.2.0

### TensorRT Compilation Warnings

**Проблема:** Предупреждения при компиляции TensorRT
```
torch-tensorrt compilation failed, falling back to PyTorch
```

**Затронутые системы:**
- CUDA 12.8 + PyTorch 2.1+
- Некоторые RTX 40-series GPU

**Влияние:** ⚠️ Снижение производительности (fallback на стандартный PyTorch)

**Статус:** 🔄 В работе

**Обходной путь:**
```python
# Принудительное отключение TensorRT
model = load_model("reswapper128", use_tensorrt=False)
```

### Memory Leak в длительных сессиях

**Проблема:** Постепенное увеличение потребления памяти при длительной работе real-time режима

**Симптомы:**
- FPS постепенно снижается
- `CUDA out of memory` после 30+ минут работы

**Затронутые режимы:**
- Real-time processing
- Длительная обработка видео

**Статус:** 🔄 Исследуется

**Обходной путь:**
```python
# Периодическая очистка памяти
import torch
torch.cuda.empty_cache()

# Перезапуск приложения каждые 30 минут
```

---

## ⚡ Ограничения производительности

### CPU Performance

**Проблема:** Очень низкая производительность на CPU (1-3 FPS)

**Причина:** Модели оптимизированы для GPU ускорения

**Статус:** 📋 Запланировано

**Планируемое решение:**
- ONNX Runtime CPU оптимизации
- Квантизированные модели для CPU
- OpenVINO интеграция для Intel CPU

### macOS ограничения

**Проблема:** Ограниченная поддержка GPU ускорения на macOS

**Текущий статус:**
- ❌ CUDA не поддерживается
- ❌ DirectML недоступен  
- ⚠️ Только CPU режим

**Планируемые улучшения:**
- Metal Performance Shaders (MPS) поддержка
- Core ML интеграция

### Real-time ограничения разрешения

**Проблема:** Снижение FPS при высоких разрешениях

| Разрешение | RTX 4090 | RTX 3080 | GTX 1660 |
|------------|----------|----------|----------|
| 128x128    | 45 FPS   | 35 FPS   | 20 FPS   |
| 256x256    | 25 FPS   | 18 FPS   | 8 FPS    |
| 512x512    | 8 FPS    | 5 FPS    | 2 FPS    |

**Планируемые оптимизации:**
- Dynamic resolution scaling
- Temporal upsampling
- Selective processing

---

## 🔧 Проблемы совместимости

### Windows DirectML

**Проблема:** Нестабильная работа с некоторыми драйверами AMD

**Затронутые GPU:**
- AMD RX 5000 series (старые драйверы)
- Некоторые интегрированные AMD GPU

**Симптомы:**
- Зависание при загрузке модели
- Неожиданные ошибки DirectML

**Обходной путь:**
```bash
# Принудительное использование CPU
python -m liveswapping.run realtime --provider cpu
```

### Linux NVIDIA Driver Issues

**Проблема:** Конфликты с некоторыми версиями драйверов NVIDIA на Linux

**Проблемные версии:**
- nvidia-driver-470 на Ubuntu 22.04
- Nouveau драйвер (не поддерживается)

**Решение:**
```bash
# Установка рекомендуемых драйверов
sudo apt purge nvidia*
sudo apt install nvidia-driver-525
sudo reboot
```

### Python Version Compatibility

**Проблема:** Ограниченная поддержка Python 3.12+

**Причина:** Некоторые зависимости еще не поддерживают Python 3.12

**Рекомендуемые версии:**
- ✅ Python 3.9
- ✅ Python 3.10  
- ✅ Python 3.11
- ⚠️ Python 3.12 (экспериментальная)

### CUDA Version Conflicts

**Проблема:** Конфликты между PyTorch CUDA и системной CUDA

**Частые ошибки:**
```
RuntimeError: CUDA version mismatch
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
```

**Решение:**
1. Используйте PyTorch с предустановленной CUDA
2. Или убедитесь в совпадении версий:
   - PyTorch CUDA 12.1 = System CUDA 12.1+
   - PyTorch CUDA 11.8 = System CUDA 11.8+

---

## 🖥️ GUI проблемы

### Tkinter высокое DPI

**Проблема:** Размытые элементы интерфейса на высоких DPI экранах (Windows)

**Затронутые системы:**
- Windows 10/11 с scaling > 100%
- 4K мониторы

**Статус:** 📋 Запланировано

**Временное решение:**
```python
# Принудительное отключение DPI scaling для приложения
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(1)
```

### GUI Responsiveness

**Проблема:** Интерфейс может зависать при длительной обработке

**Причина:** Блокирующие операции в главном потоке

**Планируемое исправление:**
- Асинхронная обработка
- Progress bars
- Отмена операций

### File Dialog Issues

**Проблема:** Медленные диалоги выбора файлов на некоторых системах

**Обходной путь:**
- Использование командной строки
- Предварительное размещение файлов в известных папках

---

## 🎯 Планируемые исправления

### Версия 1.2.0 (Ближайшая)

**Критические исправления:**
- ✅ Исправление basicsr import error
- ✅ Улучшение обработки ошибок
- ✅ Оптимизация использования памяти

**Новые возможности:**
- 🔄 Батчевая обработка изображений
- 🔄 Настройки по умолчанию
- 🔄 Улучшенное логирование

### Версия 1.3.0 (Средний срок)

**Производительность:**
- 📋 CPU оптимизации
- 📋 Квантизированные модели
- 📋 Dynamic resolution scaling

**Новые платформы:**
- 📋 macOS Metal support
- 📋 OpenVINO для Intel
- 📋 ARM64 поддержка

### Версия 2.0.0 (Долгосрочный план)

**Архитектурные изменения:**
- 📋 Полная переработка GUI (Qt/PyQt)
- 📋 REST API для удаленного доступа
- 📋 Плагиновая архитектура
- 📋 Мультипоточная обработка

**Новые модели:**
- 📋 Face restoration модели
- 📋 Style transfer модели
- 📋 Expression transfer

---

## 🛠️ Обходные пути

### Для низкой производительности

```bash
# Минимальные настройки для слабых систем
python -m liveswapping.run realtime \
    --source face.jpg \
    --modelPath models/reswapper128.pth \
    --resolution 128 \
    --delay 100 \
    --provider cpu
```

### Для проблем с памятью

```python
# Скрипт для периодической очистки памяти
import time
import torch
import gc

def clear_memory_periodically():
    while True:
        time.sleep(300)  # Каждые 5 минут
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Memory cleared")

# Запустить в отдельном потоке
import threading
threading.Thread(target=clear_memory_periodically, daemon=True).start()
```

### Для проблем с моделями

```python
# Fallback загрузка моделей
def safe_load_model(model_name):
    try:
        # Сначала пробуем с TensorRT
        return load_model(model_name, use_tensorrt=True)
    except Exception as e1:
        try:
            # Fallback без TensorRT
            return load_model(model_name, use_tensorrt=False)
        except Exception as e2:
            # CPU fallback
            return load_model(model_name, provider_type="cpu", use_tensorrt=False)
```

### Для нестабильных драйверов

```bash
# Принудительное использование конкретного провайдера
export ONNX_PROVIDERS="CUDAExecutionProvider"  # Только CUDA
export ONNX_PROVIDERS="CPUExecutionProvider"   # Только CPU
```

---

## 📊 Отчет о проблемах

### Как сообщить о проблеме

1. **Проверьте существующие issue:** [GitHub Issues](https://github.com/your-repo/issues)

2. **Соберите информацию:**
   ```python
   # Запустите диагностику
   from liveswapping.utils.gpu_utils import print_gpu_info
   print_gpu_info()
   ```

3. **Создайте детальный отчет:**
   - Версия LiveSwapping
   - Операционная система
   - GPU и драйверы
   - Полный текст ошибки
   - Шаги для воспроизведения

### Template для bug report

```markdown
## Bug Report

### Environment
- OS: [Windows 11 / Ubuntu 22.04 / macOS 13]
- Python: [3.10.8]
- LiveSwapping: [1.1.0]
- GPU: [NVIDIA RTX 4090]
- Driver: [545.23.06]

### Expected Behavior
[Что должно было произойти]

### Actual Behavior  
[Что произошло на самом деле]

### Error Message
```
[Полный текст ошибки]
```

### Steps to Reproduce
1. [Шаг 1]
2. [Шаг 2] 
3. [Шаг 3]

### Additional Context
[Дополнительная информация]
```

---

## 🔄 Статус обновлений

### Текущие приоритеты

| Проблема | Приоритет | Статус | ETA |
|----------|-----------|--------|-----|
| basicsr import error | 🔴 Высокий | 🔄 В работе | v1.2.0 |
| Memory leak | 🟡 Средний | 🔍 Исследуется | v1.3.0 |
| TensorRT warnings | 🟡 Средний | 🔄 В работе | v1.2.0 |
| macOS support | 🟢 Низкий | 📋 Запланировано | v2.0.0 |
| GUI improvements | 🟡 Средний | 📋 Запланировано | v1.3.0 |

### История исправлений

#### v1.1.0
- ✅ Исправлена проблема с CUDA memory allocation
- ✅ Улучшена стабильность real-time режима
- ✅ Добавлена поддержка DirectML

#### v1.0.0
- ✅ Начальный релиз
- ⚠️ Известные проблемы с basicsr

---

## 🆘 Получение помощи

Если ваша проблема не указана здесь:

1. **[❓ FAQ](FAQ)** - Частые вопросы
2. **[🔧 Troubleshooting](Troubleshooting)** - Решение технических проблем  
3. **[GitHub Issues](https://github.com/your-repo/issues)** - Создание bug report
4. **[GitHub Discussions](https://github.com/your-repo/discussions)** - Общие вопросы

---

*Последнее обновление: Декабрь 2024*

---

*[⬅️ Performance Optimization](Performance-Optimization) | [🏠 Главная](Home)*
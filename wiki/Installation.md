# 📥 Установка LiveSwapping

Полное руководство по установке LiveSwapping на всех поддерживаемых платформах.

## 🚀 Быстрая установка (рекомендуется)

Автоматический установщик определяет вашу систему и настраивает оптимальные параметры:

### Windows
```cmd
# Системный Python
install.bat

# Кастомный путь к Python
install.bat "C:\Python311\python.exe"
```

### Linux/macOS
```bash
# Системный Python
./install.sh

# Кастомный путь к Python
./install.sh "/usr/bin/python3.11"
```

---

## 📋 Требования к системе

### Минимальные требования
- **ОС**: Windows 10+, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8+
- **RAM**: 8GB
- **Свободное место**: 5GB

### Рекомендуемые требования
- **GPU**: NVIDIA RTX 30/40 серии, AMD RX 6000+ серии
- **RAM**: 16GB+
- **Свободное место**: 10GB+ на SSD
- **CUDA**: 12.1 или 12.8 (для NVIDIA GPU)

---

## 🔧 Ручная установка

### 1. Клонирование репозитория
```bash
git clone https://github.com/NeuroDonu/LiveSwapping.git
cd LiveSwapping
```

### 2. Создание виртуального окружения
```bash
python -m venv venv

# Активация
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Установка зависимостей

#### CUDA GPU (рекомендуется)
```bash
cd installers

# Linux/macOS
./install_cuda.sh

# Windows
install_cuda.bat
```

#### CPU only
```bash
cd installers

# Linux/macOS
./install_cpu.sh

# Windows
install_cpu.bat
```

---

## 🎯 Установка по провайдерам

### NVIDIA GPU (CUDA)
```bash
# Основные пакеты
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# TensorRT для максимальной производительности
pip install torch-tensorrt

# CuPy для ускорения numpy операций
pip install cupy-cuda12x

# Основные зависимости
pip install -r requirements/requirements_cuda.txt
```

### AMD GPU (DirectML)
```bash
# PyTorch с DirectML
pip install torch-directml

# Основные зависимости
pip install -r requirements/requirements_dml.txt
```

### Intel GPU/CPU (OpenVINO)
```bash
# OpenVINO
pip install openvino

# PyTorch CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Основные зависимости
pip install -r requirements/requirements_openvino.txt
```

### CPU Only
```bash
# PyTorch CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Основные зависимости
pip install -r requirements/requirements_cpu.txt
```

---

## 📦 Структура установки

После установки ваша структура папок будет выглядеть так:

```
LiveSwapping/
├── liveswapping/           # Основной пакет
├── models/                 # Папка для AI моделей (создается автоматически)
├── installers/             # Скрипты установки
├── requirements/           # Файлы зависимостей
├── run.py                  # Главный скрипт запуска
├── install.sh              # Автоустановщик Linux/macOS
└── install.bat             # Автоустановщик Windows
```

---

## 🔍 Проверка установки

### 1. Базовая проверка
```bash
python run.py --help
```

### 2. Проверка GPU ускорения
```python
from liveswapping.utils.gpu_utils import print_gpu_info
print_gpu_info()
```

### 3. Проверка моделей
```python
from liveswapping.ai_models.models import list_available_models
models = list_available_models()
print(models.keys())
```

### 4. Тест производительности
```python
from liveswapping.utils.gpu_utils import analyze_cupy_performance
analyze_cupy_performance()
```

---

## 🛠️ Решение проблем установки

### Проблема: CUDA не найдена
```bash
# Проверить установку CUDA
nvidia-smi
nvcc --version

# Переустановить CUDA Toolkit
# https://developer.nvidia.com/cuda-downloads
```

### Проблема: Ошибки компиляции
```bash
# Обновить pip и setuptools
pip install --upgrade pip setuptools wheel

# Установить Microsoft Visual C++ Build Tools (Windows)
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### Проблема: Недостаточно памяти
```bash
# Увеличить виртуальную память (Windows)
# Закрыть другие приложения
# Использовать CPU версию для тестирования
```

### Проблема: basicsr ошибки
```bash
# Автоматическое исправление
python liveswapping/utils/fix_basicsr.py
```

---

## 🔄 Обновление

### Обновление кода
```bash
git pull origin main
```

### Обновление зависимостей
```bash
pip install -r requirements/requirements_cuda.txt --upgrade
```

### Переустановка с нуля
```bash
# Удалить виртуальное окружение
rm -rf venv

# Создать заново
python -m venv venv
source venv/bin/activate
./install.sh
```

---

## 🚦 Проверка статуса установки

Запустите диагностический скрипт для проверки всех компонентов:

```python
# diagnostic.py
from liveswapping.utils.gpu_utils import print_gpu_info, get_provider_info
from liveswapping.ai_models.models import list_available_models, get_optimal_provider

print("=== ДИАГНОСТИКА УСТАНОВКИ ===")

# GPU информация
print("\n1. GPU статус:")
print_gpu_info()

# Провайдеры
print("\n2. Доступные провайдеры:")
providers = get_provider_info()
for provider in providers:
    status = "✅" if provider['available'] else "❌"
    print(f"{status} {provider['name'].upper()}")

# Оптимальный провайдер
print(f"\n3. Рекомендуемый провайдер: {get_optimal_provider()}")

# Модели
print(f"\n4. Доступные модели: {len(list_available_models())}")

print("\n=== ДИАГНОСТИКА ЗАВЕРШЕНА ===")
```

---

## 📱 Следующие шаги

После успешной установки:

1. **[🎯 Быстрый старт](Quick-Start)** - Запустите первый face swap
2. **[👤 Руководство пользователя](User-Guide)** - Изучите все возможности
3. **[⚡ Оптимизация производительности](Performance-Optimization)** - Ускорьте работу

---

## 🆘 Помощь

Если у вас возникли проблемы:

- **[🔧 Troubleshooting](Troubleshooting)** - Решение распространенных проблем
- **[❓ FAQ](FAQ)** - Частые вопросы
- **[🐛 Известные проблемы](Known-Issues)** - Баги и ограничения

---

*[⬅️ Назад к главной](Home) | [➡️ Быстрый старт](Quick-Start)*
# 🚀 Полное руководство по установке CUDA+cuDNN+TensorRT

**Безопасная и простая установка полного CUDA stack для LiveSwapping через conda/miniconda**

> 🌍 **[English version](en/CUDA-Installation-Guide)** | 🇷🇺 **Русская версия**

## 📚 Содержание

1. [Зачем нужна эта установка](#-зачем-нужна-эта-установка)
2. [Подготовка системы](#-подготовка-системы)
3. [Установка через Miniconda (рекомендуется)](#-установка-через-miniconda-рекомендуется)
4. [Проверка установки](#-проверка-установки)
5. [Решение проблем](#-решение-проблем)
6. [Альтернативные методы](#-альтернативные-методы)

---

## 🎯 Зачем нужна эта установка

### Что дает правильная установка CUDA stack:
- **3x ускорение** LiveSwapping с TensorRT
- **Стабильная работа** без конфликтов версий
- **GPU ускорение** для всех операций
- **Совместимость** с PyTorch и ONNX Runtime

### Компоненты stack:
- **CUDA Toolkit** - основная платформа GPU вычислений
- **cuDNN** - библиотека нейронных сетей для CUDA
- **TensorRT** - оптимизация inference для NVIDIA GPU
- **PyTorch** - с CUDA поддержкой
- **ONNX Runtime** - с GPU провайдерами

---

## 🛠️ Подготовка системы

### Проверка совместимости GPU

```bash
# Проверка наличия NVIDIA GPU
nvidia-smi

# Должен показать информацию о GPU:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.54.03    Driver Version: 535.54.03    CUDA Version: 12.2   |
# +-------------------------------+----------------------+----------------------+
```

### Поддерживаемые GPU:
- ✅ **RTX 40 series** (4090, 4080, 4070, 4060)
- ✅ **RTX 30 series** (3090, 3080, 3070, 3060)
- ✅ **RTX 20 series** (2080, 2070, 2060)
- ✅ **GTX 16 series** (1660, 1650)
- ✅ **Tesla, Quadro** серии
- ❌ **GTX 10xx и старше** (ограниченная поддержка TensorRT)

### Требования к драйверам:

| CUDA Version | Минимальная версия драйвера |
|--------------|----------------------------|
| CUDA 12.1 | 530.30.02+ (Linux), 531.14+ (Windows) |
| CUDA 12.4 | 550.54.15+ (Linux), 551.61+ (Windows) |
| CUDA 11.8 | 520.61.05+ (Linux), 522.06+ (Windows) |

---

## 🐍 Установка через Miniconda (рекомендуется)

### Почему conda лучше всего:
- ✅ **Изолированные окружения** - никаких конфликтов
- ✅ **Автоматическое управление** зависимостями
- ✅ **Предкомпилированные** пакеты
- ✅ **Простая переустановка** при проблемах
- ✅ **Работает везде** одинаково

### Шаг 1: Установка Miniconda

#### Windows:
```cmd
# Скачайте Miniconda с официального сайта
# https://docs.conda.io/en/latest/miniconda.html

# Или через PowerShell:
Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile "miniconda.exe"
.\miniconda.exe /S /D=C:\miniconda3

# Перезапустите терминал или выполните:
C:\miniconda3\Scripts\activate.bat
```

#### Linux:
```bash
# Скачивание и установка
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Добавление в PATH
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Шаг 2: Создание изолированного окружения

```bash
# Создание окружения с Python 3.10
conda create -n liveswapping python=3.10 -y

# Активация окружения
conda activate liveswapping

# Проверка
python --version  # Должно показать Python 3.10.x
which python      # Должно показать путь в miniconda
```

### Шаг 3: Установка CUDA Toolkit через conda

```bash
# Установка CUDA 12.1 (рекомендуется для PyTorch 2.1+)
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -y

# Или CUDA 11.8 (для более старых систем)
# conda install nvidia/label/cuda-11.8.0::cuda-toolkit -y

# Проверка установки
nvcc --version
```

### Шаг 4: Установка cuDNN

```bash
# cuDNN через conda-forge
conda install conda-forge::cudnn -y

# Или конкретную версию для CUDA 12.1
conda install nvidia::cudnn=8.9.7.29 -y
```

### Шаг 5: Установка PyTorch с CUDA

```bash
# PyTorch с CUDA 12.1 support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Или для CUDA 11.8
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### Шаг 6: Установка TensorRT

```bash
# TensorRT через pip (так как conda версия может быть устаревшей)
pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu121

# Или для CUDA 11.8
# pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu118
```

### Шаг 7: Установка дополнительных зависимостей

```bash
# ONNX Runtime с GPU поддержкой
pip install onnxruntime-gpu

# CuPy для ускорения numpy операций
conda install cupy -c conda-forge -y

# Дополнительные библиотеки
pip install opencv-python pillow numpy scipy
```

### Шаг 8: Установка LiveSwapping

```bash
# Клонирование репозитория
git clone https://github.com/NeuroDonu/LiveSwapping.git
cd LiveSwapping

# Установка в режиме разработки
pip install -e .

# Или установка зависимостей
pip install -r requirements/requirements_cuda.txt
```

---

## ✅ Проверка установки

### Комплексная диагностика

Создайте файл `test_cuda_stack.py`:

```python
#!/usr/bin/env python3
"""
Комплексная проверка CUDA stack для LiveSwapping
"""

import sys
import subprocess

def check_component(name, check_func):
    """Проверка компонента с цветным выводом."""
    try:
        result = check_func()
        if result:
            print(f"✅ {name}: OK")
            if isinstance(result, str):
                print(f"   {result}")
        else:
            print(f"❌ {name}: FAILED")
        return bool(result)
    except Exception as e:
        print(f"❌ {name}: ERROR - {e}")
        return False

def check_python():
    """Проверка версии Python."""
    version = sys.version.split()[0]
    major, minor = map(int, version.split('.')[:2])
    if major == 3 and minor >= 8:
        return f"Python {version}"
    return None

def check_conda():
    """Проверка conda окружения."""
    try:
        result = subprocess.run(['conda', '--version'], 
                              capture_output=True, text=True, check=True)
        env_name = subprocess.run(['conda', 'info', '--envs'], 
                                capture_output=True, text=True, check=True)
        if '*' in env_name.stdout:
            active_env = [line for line in env_name.stdout.split('\n') 
                         if '*' in line][0].split()[0]
            return f"{result.stdout.strip()}, active env: {active_env}"
        return result.stdout.strip()
    except:
        return None

def check_cuda_toolkit():
    """Проверка CUDA Toolkit."""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, check=True)
        # Извлечение версии из вывода nvcc
        lines = result.stdout.split('\n')
        version_line = [line for line in lines if 'release' in line][0]
        version = version_line.split('release ')[1].split(',')[0]
        return f"CUDA {version}"
    except:
        return None

def check_nvidia_driver():
    """Проверка драйвера NVIDIA."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        driver_version = result.stdout.strip()
        return f"Driver {driver_version}"
    except:
        return None

def check_pytorch():
    """Проверка PyTorch с CUDA."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            return f"PyTorch {torch.__version__}, CUDA {cuda_version}, {device_count} GPU(s), Current: {device_name}"
        else:
            return f"PyTorch {torch.__version__} (CUDA NOT AVAILABLE)"
    except ImportError:
        return None

def check_tensorrt():
    """Проверка TensorRT."""
    try:
        import torch_tensorrt
        return f"TensorRT {torch_tensorrt.__version__}"
    except ImportError:
        return None

def check_onnxruntime():
    """Проверка ONNX Runtime."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        gpu_providers = [p for p in providers if 'CUDA' in p or 'DirectML' in p or 'OpenVINO' in p]
        return f"ONNX Runtime {ort.__version__}, GPU providers: {gpu_providers}"
    except ImportError:
        return None

def check_cupy():
    """Проверка CuPy."""
    try:
        import cupy as cp
        # Простой тест
        a = cp.array([1, 2, 3])
        b = cp.array([4, 5, 6])
        c = a + b
        return f"CuPy {cp.__version__}"
    except ImportError:
        return None
    except Exception as e:
        return f"CuPy installed but error: {e}"

def check_liveswapping():
    """Проверка LiveSwapping."""
    try:
        from liveswapping.utils.gpu_utils import print_gpu_info, get_optimal_provider
        provider = get_optimal_provider()
        return f"LiveSwapping OK, optimal provider: {provider}"
    except ImportError:
        return None

def performance_test():
    """Быстрый тест производительности."""
    try:
        import torch
        import time
        
        if not torch.cuda.is_available():
            return "CUDA не доступен для тестирования"
        
        device = torch.device('cuda')
        
        # Создание тестовых данных
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        
        # Прогрев
        for _ in range(10):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
        
        # Тест
        start_time = time.time()
        for _ in range(100):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        ops_per_sec = 1.0 / avg_time
        
        return f"Matrix multiplication: {avg_time:.4f}s per op, {ops_per_sec:.1f} ops/sec"
        
    except Exception as e:
        return f"Performance test failed: {e}"

def main():
    """Основная функция диагностики."""
    print("🔍 ДИАГНОСТИКА CUDA STACK ДЛЯ LIVESWAPPING")
    print("=" * 60)
    
    checks = [
        ("Python", check_python),
        ("Conda", check_conda),
        ("NVIDIA Driver", check_nvidia_driver),
        ("CUDA Toolkit", check_cuda_toolkit),
        ("PyTorch", check_pytorch),
        ("TensorRT", check_tensorrt),
        ("ONNX Runtime", check_onnxruntime),
        ("CuPy", check_cupy),
        ("LiveSwapping", check_liveswapping),
    ]
    
    results = []
    for name, check_func in checks:
        result = check_component(name, check_func)
        results.append(result)
    
    print("\n🚀 ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("-" * 30)
    check_component("GPU Performance", performance_test)
    
    print(f"\n📊 ОБЩИЙ РЕЗУЛЬТАТ")
    print("-" * 20)
    success_count = sum(results)
    total_count = len(results)
    
    if success_count == total_count:
        print("🎉 ВСЕ КОМПОНЕНТЫ РАБОТАЮТ ОТЛИЧНО!")
        print("   LiveSwapping готов к максимальной производительности")
    elif success_count >= 6:
        print("✅ СИСТЕМА ГОТОВА К РАБОТЕ")
        print("   Некоторые компоненты могут быть недоступны, но основная функциональность работает")
    else:
        print("⚠️  ТРЕБУЕТСЯ ДОПОЛНИТЕЛЬНАЯ НАСТРОЙКА")
        print("   Проверьте неудачные компоненты и повторите установку")
    
    print(f"\nСтатус: {success_count}/{total_count} компонентов работают")

if __name__ == "__main__":
    main()
```

Запустите диагностику:

```bash
python test_cuda_stack.py
```

### Ожидаемый вывод (успешная установка):

```
🔍 ДИАГНОСТИКА CUDA STACK ДЛЯ LIVESWAPPING
============================================================
✅ Python: OK
   Python 3.10.12
✅ Conda: OK
   conda 23.7.4, active env: liveswapping
✅ NVIDIA Driver: OK
   Driver 535.54.03
✅ CUDA Toolkit: OK
   CUDA 12.1
✅ PyTorch: OK
   PyTorch 2.1.0, CUDA 12.1, 1 GPU(s), Current: NVIDIA GeForce RTX 4090
✅ TensorRT: OK
   TensorRT 2.1.0
✅ ONNX Runtime: OK
   ONNX Runtime 1.16.1, GPU providers: ['CUDAExecutionProvider']
✅ CuPy: OK
   CuPy 12.2.0
✅ LiveSwapping: OK
   optimal provider: cuda

🚀 ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ
------------------------------
✅ GPU Performance: OK
   Matrix multiplication: 0.0001s per op, 8542.3 ops/sec

📊 ОБЩИЙ РЕЗУЛЬТАТ
--------------------
🎉 ВСЕ КОМПОНЕНТЫ РАБОТАЮТ ОТЛИЧНО!
   LiveSwapping готов к максимальной производительности

Статус: 9/9 компонентов работают
```

---

## 🔧 Решение проблем

### Проблема 1: "CUDA driver version is insufficient"

**Симптомы:**
```
RuntimeError: CUDA driver version is insufficient for CUDA runtime version
```

**Решение:**
```bash
# Проверьте версию драйвера
nvidia-smi

# Обновите драйвер NVIDIA:
# Windows: https://www.nvidia.com/drivers/
# Ubuntu: sudo apt update && sudo apt install nvidia-driver-535

# Перезагрузите систему
sudo reboot
```

### Проблема 2: "torch-tensorrt compilation failed"

**Симптомы:**
```
WARNING: torch-tensorrt compilation failed, falling back to PyTorch
```

**Решение:**
```bash
# Переустановите torch-tensorrt
pip uninstall torch-tensorrt
pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu121

# Проверьте совместимость версий
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

### Проблема 3: Конфликт версий CUDA

**Симптомы:**
```
RuntimeError: CUDA version mismatch
```

**Решение - полная переустановка:**
```bash
# Деактивация окружения
conda deactivate

# Удаление старого окружения
conda env remove -n liveswapping

# Создание нового окружения с нуля
conda create -n liveswapping python=3.10 -y
conda activate liveswapping

# Чистая установка всего stack
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -y
conda install conda-forge::cudnn -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu121
```

### Проблема 4: "No module named 'cupy'"

**Решение:**
```bash
# Установка CuPy для CUDA 12.1
conda install cupy -c conda-forge

# Или через pip с конкретной версией CUDA
pip install cupy-cuda12x
```

### Проблема 5: Медленная работа несмотря на GPU

**Диагностика:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name()}")

# Проверка что модели используют GPU
from liveswapping.ai_models.models import get_optimal_provider
print(f"Optimal provider: {get_optimal_provider()}")
```

**Решение:**
```bash
# Убедитесь что переменные окружения настроены
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Проверьте загрузку GPU в реальном времени
nvidia-smi -l 1
```

---

## 🔄 Альтернативные методы

### Метод 2: Docker (для опытных пользователей)

```dockerfile
# Dockerfile для LiveSwapping с CUDA
FROM nvidia/cuda:12.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install torch-tensorrt onnxruntime-gpu cupy-cuda12x

WORKDIR /app
COPY . .
RUN pip3 install -e .

CMD ["python3", "run.py"]
```

### Метод 3: Системная установка (не рекомендуется)

<details>
<summary>⚠️ Только для экспертов (может сломать систему)</summary>

```bash
# ВНИМАНИЕ: Может конфликтовать с другими приложениями!

# Ubuntu - установка CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Настройка PATH и LD_LIBRARY_PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

</details>

---

## 🎯 Финальная проверка LiveSwapping

После установки проверьте работу LiveSwapping:

```bash
# Запуск диагностики
python test_cuda_stack.py

# Проверка загрузки моделей
python -c "
from liveswapping.ai_models.models import load_model, get_optimal_provider
print(f'Optimal provider: {get_optimal_provider()}')
model = load_model('reswapper128', use_tensorrt=True)
print('✅ Model loaded successfully with TensorRT!')
"

# Тест GUI
python run.py
```

## 📚 Дополнительные ресурсы

- **[NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)**
- **[PyTorch Installation Guide](https://pytorch.org/get-started/locally/)**
- **[TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/)**
- **[Conda User Guide](https://docs.conda.io/en/latest/)**

---

## 🆘 Получение помощи

Если у вас остались проблемы:

1. **Запустите диагностику:** `python test_cuda_stack.py`
2. **Проверьте логи:** сохраните вывод команд
3. **Создайте issue:** [GitHub Issues](https://github.com/NeuroDonu/LiveSwapping/issues)
4. **Приложите информацию:**
   - Вывод `nvidia-smi`
   - Вывод диагностического скрипта
   - Операционная система и версия
   - Модель GPU

---

> **💡 Совет:** Всегда используйте conda окружения для изоляции! Это предотвращает конфликты и позволяет легко переустанавливать компоненты.

---

*[⬅️ Installation](Installation) | [🏠 Главная](Home) | [🌍 English version](en/CUDA-Installation-Guide)*
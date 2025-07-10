# 🎯 Быстрый старт

Начните работу с LiveSwapping за 5 минут! Это руководство поможет вам сделать первую замену лица.

## ⚡ Экспресс-запуск

### 1️⃣ Проверьте установку
```bash
python run.py --help
```
Если команда работает - у вас всё готово! Если нет - перейдите к [Установке](Installation).

### 2️⃣ Запустите GUI
```bash
python run.py
```

### 3️⃣ Выберите режим
- **Вариант 1**: Real-time обработка (веб-камера)
- **Вариант 2**: Обработка видео

---

## 🎥 Real-time замена лиц

### Шаг 1: Подготовьте изображение
Вам понадобится **качественное фото вашего лица**:
- ✅ Лицо четко видно
- ✅ Хорошее освещение  
- ✅ Лицо смотрит в камеру
- ✅ Формат: JPG, PNG
- ❌ Избегайте: очки, шляпы, маски

### Шаг 2: Запустите real-time режим
```bash
python -m liveswapping.run realtime \
    --source your_face.jpg \
    --modelPath models/reswapper128.pth
```

### Шаг 3: Управление
- **Q** - выход
- **+/-** - изменить задержку
- **Камера должна показать** ваше лицо на видео с веб-камеры

### 🎮 Расширенные опции
```bash
# С OBS интеграцией для стрима
python -m liveswapping.run realtime \
    --source streamer_face.jpg \
    --modelPath models/reswapper128.pth \
    --obs \
    --enhance_res

# С сохранением рта (более реалистично)
python -m liveswapping.run realtime \
    --source your_face.jpg \
    --modelPath models/reswapper128.pth \
    --mouth_mask
```

---

## 🎬 Обработка видео

### Шаг 1: Подготовьте файлы
- **Исходное лицо**: фото того, кого хотите "вставить"
- **Видео**: файл MP4 с лицом, которое заменяем

### Шаг 2: Запустите обработку
```bash
python -m liveswapping.run video \
    --source actor_face.jpg \
    --target_video input_video.mp4 \
    --modelPath models/reswapper256.pth
```

### Шаг 3: Дождитесь результата
- Прогресс показывается в консоли
- Результат сохранится как `output.mp4`

### 🎨 С улучшением качества
```bash
python -m liveswapping.run video \
    --source celebrity.jpg \
    --target_video movie_scene.mp4 \
    --modelPath models/reswapper256.pth \
    --upscale 2 \
    --bg_upsampler realesrgan \
    --weight 0.8
```

---

## 🖼️ Обработка изображений

### Простая замена лица
```bash
python -m liveswapping.run image \
    --source source_face.jpg \
    --target target_photo.jpg \
    --modelPath models/reswapper128.pth \
    --output result.jpg
```

---

## 🧠 Выбор модели

| Модель | Скорость | Качество | Рекомендации |
|--------|----------|----------|-------------|
| **reswapper128** | ⚡⚡⚡ Высокая | ⭐⭐ Хорошее | Real-time, быстрые тесты |
| **reswapper256** | ⚡⚡ Средняя | ⭐⭐⭐ Отличное | Видео, финальный результат |
| **inswapper128** | ⚡⚡ Средняя | ⭐⭐⭐ Отличное | Универсальная |

### Автоматическая загрузка моделей
При первом запуске модели загружаются автоматически:
```
Downloading reswapper128... ████████████ 100%
Model saved to: models/reswapper128.pth
```

---

## 🎛️ GUI интерфейс

### Real-time GUI
1. Запустите: `python run.py` → выберите **"1"**
2. **Выберите файлы**:
   - Source Image: ваше фото
   - Model: модель (загрузится автоматически)
3. **Настройте параметры**:
   - Resolution: 128 для скорости, 256 для качества
   - Delay: задержка в мс
4. **Старт**: нажмите "Start"

### Video GUI  
1. Запустите: `python run.py` → выберите **"2"**
2. **Выберите файлы**:
   - Source Image: лицо для вставки
   - Target Video: видео для обработки
   - Model: AI модель
3. **Настройте качество**:
   - Enable Upscaler: улучшение качества
   - Upscale: коэффициент увеличения
4. **Старт**: нажмите "Start Processing"

---

## 📊 Мониторинг производительности

### Проверка GPU
```python
from liveswapping.utils.gpu_utils import print_gpu_info
print_gpu_info()
```

Вывод:
```
[GPU] CuPy: Available
[INFO] Devices: 1
[INFO] Memory: 24.0 GB
[INFO] Compute: 8.9
[GPU] Excellent GPU for CuPy acceleration!
```

### Бенчмарк системы
```python
from liveswapping.utils.gpu_utils import analyze_cupy_performance
analyze_cupy_performance()
```

---

## 🚨 Первые проблемы и решения

### 🔴 "No module named 'liveswapping'"
```bash
# Убедитесь что вы в правильной папке
cd LiveSwapping
python run.py
```

### 🔴 "CUDA out of memory"
```bash
# Используйте меньшую модель
python -m liveswapping.run realtime \
    --source face.jpg \
    --modelPath models/reswapper128.pth \
    --resolution 128
```

### 🔴 "No face detected"
- Используйте фото с четко видимым лицом
- Хорошее освещение
- Лицо смотрит в камеру
- Без сильных теней

### 🔴 Медленная работа
```bash
# Проверьте что используется GPU
from liveswapping.ai_models.models import get_optimal_provider
print(get_optimal_provider())  # Должно быть "cuda" для NVIDIA

# Используйте TensorRT оптимизацию  
model = load_model("reswapper128", use_tensorrt=True)
```

---

## 🎯 Примеры использования

### Для стримеров
```bash
# Замена лица с OBS интеграцией
python -m liveswapping.run realtime \
    --source streamer_avatar.jpg \
    --modelPath models/reswapper128.pth \
    --obs \
    --enhance_res \
    --delay 100
```

### Для создателей контента
```bash
# Высококачественная обработка видео
python -m liveswapping.run video \
    --source actor.jpg \
    --target_video scene.mp4 \
    --modelPath models/reswapper256.pth \
    --upscale 2 \
    --bg_upsampler realesrgan \
    --mouth_mask
```

### Для разработчиков
```python
from liveswapping.ai_models.models import load_model
from liveswapping.core.image_utils import *
import cv2

# Загрузка модели с оптимизацией
model = load_model("reswapper128", use_tensorrt=True)

# Простая обработка
source = cv2.imread("source.jpg")
target = cv2.imread("target.jpg")
# ... ваш код обработки
```

---

## 📈 Следующие шаги

Поздравляем! Вы освоили основы LiveSwapping. Теперь изучите:

1. **[👤 Руководство пользователя](User-Guide)** - все возможности системы
2. **[⚡ Оптимизация производительности](Performance-Optimization)** - ускорение работы
3. **[🧠 AI Модели](AI-Models)** - выбор подходящей модели
4. **[💻 GUI интерфейс](GUI-Guide)** - работа с графическим интерфейсом

---

## 🎁 Бонус: готовые команды

Скопируйте и используйте готовые команды:

```bash
# Real-time с максимальной скоростью
python -m liveswapping.run realtime --source face.jpg --modelPath models/reswapper128.pth --resolution 128

# Real-time с максимальным качеством  
python -m liveswapping.run realtime --source face.jpg --modelPath models/reswapper256.pth --resolution 256 --mouth_mask

# Видео для YouTube
python -m liveswapping.run video --source actor.jpg --target_video input.mp4 --modelPath models/reswapper256.pth --upscale 2

# Тест изображения
python -m liveswapping.run image --source src.jpg --target tgt.jpg --modelPath models/reswapper128.pth --output result.jpg
```

---

## 🆘 Нужна помощь?

- **[❓ FAQ](FAQ)** - частые вопросы
- **[🔧 Troubleshooting](Troubleshooting)** - решение проблем  
- **[📋 API Reference](API-Reference)** - для разработчиков

---

*[⬅️ Установка](Installation) | [🏠 Главная](Home) | [➡️ Руководство пользователя](User-Guide)*
# 👤 Руководство пользователя

Полное руководство по использованию всех функций LiveSwapping для пользователей.

## 📚 Содержание

1. [Интерфейсы системы](#-интерфейсы-системы)
2. [Real-time обработка](#-real-time-обработка)
3. [Обработка видео](#-обработка-видео)
4. [Обработка изображений](#️-обработка-изображений)
5. [Настройки качества](#-настройки-качества)
6. [Работа с моделями](#-работа-с-моделями)
7. [Оптимизация производительности](#-оптимизация-производительности)

---

## 🎛️ Интерфейсы системы

### GUI интерфейс (рекомендуется)
```bash
python run.py
```
- Удобный графический интерфейс
- Визуальный выбор файлов
- Предварительный просмотр настроек
- Мониторинг прогресса

### Командная строка
```bash
python -m liveswapping.run [режим] [параметры]
```
- Автоматизация
- Скрипты
- Удаленное использование

### Python API
```python
from liveswapping.core import realtime, video
```
- Интеграция в свои проекты
- Программное управление

---

## 🎥 Real-time обработка

### Базовое использование

#### Через GUI
1. `python run.py` → выберите **"1"**
2. Настройте параметры:
   - **Source Image**: фото вашего лица
   - **Model Path**: путь к модели AI
   - **Resolution**: качество обработки
3. Нажмите **"Start"**

#### Через командную строку
```bash
python -m liveswapping.run realtime \
    --source your_face.jpg \
    --modelPath models/reswapper128.pth
```

### Расширенные функции

#### OBS интеграция для стриминга
```bash
python -m liveswapping.run realtime \
    --source streamer_face.jpg \
    --modelPath models/reswapper128.pth \
    --obs \
    --enhance_res
```

**Настройка OBS:**
1. Добавьте источник "Video Capture Device"
2. Выберите "OBS Virtual Camera"
3. Настройте сцену

#### Управление в реальном времени
- **Q** - выход из программы
- **+** - увеличить задержку на 50мс
- **-** - уменьшить задержку на 50мс
- **ESC** - пауза/возобновление

#### Сохранение рта (mouth_mask)
```bash
python -m liveswapping.run realtime \
    --source face.jpg \
    --modelPath models/reswapper128.pth \
    --mouth_mask
```
- Сохраняет естественные движения губ
- Более реалистичный результат
- Подходит для речи и пения

#### Настройка задержки
```bash
python -m liveswapping.run realtime \
    --source face.jpg \
    --modelPath models/reswapper128.pth \
    --delay 200 \
    --fps_delay
```
- `--delay`: задержка в миллисекундах
- `--fps_delay`: показывать FPS на экране

#### Атрибуты лица
```bash
python -m liveswapping.run realtime \
    --source base_face.jpg \
    --modelPath models/reswapper128.pth \
    --face_attribute_direction smile.npy \
    --face_attribute_steps 2.0
```
- Изменение выражения лица
- Файлы направлений (.npy) - опционально

---

## 🎬 Обработка видео

### Базовая обработка

#### Через GUI
1. `python run.py` → выберите **"2"**
2. Настройте файлы:
   - **Source Image**: лицо для замены
   - **Target Video**: видео для обработки
   - **Model**: AI модель
3. Выберите качество:
   - **Enable Upscaler**: улучшение GFPGAN
   - **Upscale Factor**: коэффициент увеличения
4. **"Start Processing"**

#### Через командную строку
```bash
python -m liveswapping.run video \
    --source actor_face.jpg \
    --target_video input_video.mp4 \
    --modelPath models/reswapper256.pth
```

### Настройки качества

#### Максимальное качество
```bash
python -m liveswapping.run video \
    --source celebrity.jpg \
    --target_video movie.mp4 \
    --modelPath models/reswapper256.pth \
    --upscale 2 \
    --bg_upsampler realesrgan \
    --bg_tile 400 \
    --weight 0.8 \
    --std 0.5 \
    --blur 1
```

#### Параметры качества
- `--upscale`: коэффициент увеличения (1, 2, 4)
- `--bg_upsampler`: фоновый апскейлер ("realesrgan", None)
- `--bg_tile`: размер тайла для обработки
- `--weight`: вес смешивания (0.0-1.0)
- `--std`: стандартное отклонение шума
- `--blur`: размытие для сглаживания

#### Сохранение естественности
```bash
python -m liveswapping.run video \
    --source source.jpg \
    --target_video video.mp4 \
    --modelPath models/reswapper256.pth \
    --mouth_mask \
    --weight 0.7
```

### Форматы и кодирование

#### Поддерживаемые входные форматы
- **Видео**: MP4, AVI, MOV, MKV
- **Изображения**: JPG, PNG, BMP

#### Выходной формат
- Всегда сохраняется как `output.mp4`
- Кодек: H.264 (libx264)
- Аудио: AAC (сохраняется из оригинала)

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

### Батчевая обработка
```python
import os
from liveswapping.core import image_utils

# Обработка всех изображений в папке
source_face = "celebrity.jpg"
input_dir = "photos/"
output_dir = "results/"

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.png')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"swapped_{filename}")
        
        # Ваш код обработки
        process_image(source_face, input_path, output_path)
```

---

## ⚙️ Настройки качества

### Выбор разрешения

| Разрешение | Качество | Скорость | Применение |
|------------|----------|----------|------------|
| **128x128** | ⭐⭐ Хорошее | ⚡⚡⚡ Быстро | Real-time, тесты |
| **256x256** | ⭐⭐⭐ Отличное | ⚡⚡ Средне | Видео, фото |
| **512x512** | ⭐⭐⭐⭐ Превосходное | ⚡ Медленно | Финальный результат |

### Настройки смешивания

#### Weight (вес смешивания)
```bash
--weight 0.5  # 50% исходного, 50% замененного
--weight 0.8  # 80% замененного, 20% исходного
--weight 0.3  # 30% замененного, 70% исходного
```

#### Шум и размытие
```bash
--std 1    # Стандартный шум
--std 0.5  # Меньше шума (чище)
--std 2    # Больше шума (текстура)

--blur 1   # Минимальное размытие
--blur 3   # Сильное размытие (мягче)
```

### Upscaling опции

#### GFPGAN (улучшение лиц)
- Автоматически улучшает качество лиц
- Убирает артефакты
- Повышает четкость

#### RealESRGAN (фон)
- Улучшает фон и детали
- Увеличивает разрешение
- Убирает размытие

---

## 🧠 Работа с моделями

### Доступные модели

#### StyleTransfer модели
- **reswapper128.pth**: быстрая, хорошее качество
- **reswapper256.pth**: медленная, отличное качество

#### InSwapper модели  
- **inswapper128.onnx**: универсальная, стабильная

#### DFM модели
- ***.dfm**: кастомные модели

### Автоматическая загрузка
```bash
# При первом использовании:
python -m liveswapping.run realtime \
    --source face.jpg \
    --modelPath models/reswapper128.pth

# Вывод:
# Downloading reswapper128... ████████████ 100%
# Model saved to: models/reswapper128.pth
```

### Ручная загрузка моделей
```python
from liveswapping.ai_models.download_models import ensure_model

# Загрузить конкретную модель
model_path = ensure_model("reswapper256")
print(f"Model saved to: {model_path}")
```

### Выбор оптимальной модели

#### Для real-time
- **reswapper128**: максимальная скорость
- TensorRT оптимизация включена
- Разрешение 128x128

#### Для видео высокого качества
- **reswapper256**: лучшее качество
- Разрешение 256x256
- Включить upscaling

#### Для экспериментов
- **inswapper128**: стабильные результаты
- Хорошая совместимость
- ONNX Runtime оптимизация

---

## ⚡ Оптимизация производительности

### Проверка системы
```python
from liveswapping.utils.gpu_utils import print_gpu_info, get_optimal_config

# Информация о GPU
print_gpu_info()

# Оптимальные настройки
config = get_optimal_config()
print(f"Рекомендуемый размер батча: {config['recommended_batch_size']}")
print(f"Смешанная точность: {config['use_mixed_precision']}")
```

### Настройки для разных GPU

#### High-end GPU (RTX 4090, 3090)
```bash
python -m liveswapping.run video \
    --source face.jpg \
    --target_video video.mp4 \
    --modelPath models/reswapper256.pth \
    --upscale 2 \
    --bg_upsampler realesrgan
```

#### Mid-range GPU (RTX 3070, 4070)
```bash
python -m liveswapping.run video \
    --source face.jpg \
    --target_video video.mp4 \
    --modelPath models/reswapper128.pth \
    --upscale 2
```

#### Low-end GPU или CPU
```bash
python -m liveswapping.run video \
    --source face.jpg \
    --target_video video.mp4 \
    --modelPath models/reswapper128.pth
    # Без upscaling
```

### Мониторинг производительности

#### Real-time FPS
```bash
python -m liveswapping.run realtime \
    --source face.jpg \
    --modelPath models/reswapper128.pth \
    --fps_delay  # Показывать FPS на экране
```

#### Бенчмарк системы
```python
from liveswapping.utils.gpu_utils import analyze_cupy_performance
analyze_cupy_performance()
```

---

## 🔧 Настройки и конфигурация

### Конфигурационный файл
Создайте `config.json` для сохранения настроек:
```json
{
    "default_model": "models/reswapper128.pth",
    "default_resolution": 128,
    "use_tensorrt": true,
    "provider": "cuda",
    "upscale_factor": 2,
    "enable_mouth_mask": false
}
```

### Переменные окружения
```bash
# Отключить verbose логи
export ONNX_LOG_LEVEL=3
export OMP_NUM_THREADS=1

# Настройки CUDA
export CUDA_VISIBLE_DEVICES=0
```

### Папки по умолчанию
```
LiveSwapping/
├── models/          # AI модели
├── inputs/          # Входные файлы  
├── outputs/         # Результаты
└── temp/           # Временные файлы
```

---

## 🎯 Практические советы

### Для лучшего качества

#### Подготовка исходного изображения
- **Разрешение**: минимум 512x512px
- **Освещение**: равномерное, без теней
- **Поза**: лицо анфас, взгляд в камеру
- **Качество**: четкое, без размытия

#### Настройки обработки
- Используйте **reswapper256** для финального результата
- Включите **mouth_mask** для естественности
- Установите **weight 0.7-0.8** для лучшего смешивания
- Включите **upscaling** для повышения качества

### Для максимальной скорости

#### Real-time оптимизация
- Используйте **reswapper128**
- Разрешение **128x128**
- Отключите дополнительные эффекты
- Минимальная задержка **--delay 0**

#### Batch обработка
- Обрабатывайте файлы пакетами
- Используйте SSD для хранения
- Закройте другие приложения

---

## 📝 Шпаргалка команд

### Real-time
```bash
# Базовый
python -m liveswapping.run realtime --source face.jpg --modelPath models/reswapper128.pth

# Стриминг
python -m liveswapping.run realtime --source face.jpg --modelPath models/reswapper128.pth --obs --enhance_res

# Качество
python -m liveswapping.run realtime --source face.jpg --modelPath models/reswapper256.pth --mouth_mask --resolution 256
```

### Видео
```bash
# Базовый
python -m liveswapping.run video --source face.jpg --target_video video.mp4 --modelPath models/reswapper256.pth

# Качество
python -m liveswapping.run video --source face.jpg --target_video video.mp4 --modelPath models/reswapper256.pth --upscale 2 --bg_upsampler realesrgan

# Быстрый
python -m liveswapping.run video --source face.jpg --target_video video.mp4 --modelPath models/reswapper128.pth
```

### Изображения
```bash
# Простой
python -m liveswapping.run image --source src.jpg --target tgt.jpg --modelPath models/reswapper128.pth --output result.jpg
```

---

## 🆘 Получение помощи

- **[❓ FAQ](FAQ)** - часто задаваемые вопросы
- **[🔧 Troubleshooting](Troubleshooting)** - решение проблем
- **[🐛 Известные проблемы](Known-Issues)** - текущие ограничения
- **[📋 API Reference](API-Reference)** - техническая документация

---

*[⬅️ Быстрый старт](Quick-Start) | [🏠 Главная](Home) | [➡️ GUI интерфейс](GUI-Guide)*
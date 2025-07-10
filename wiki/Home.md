# 🎭 LiveSwapping Wiki

**Добро пожаловать в Wiki системы LiveSwapping** - продвинутой системы реал-тайм замены лиц с поддержкой множественных моделей и передовых технологий оптимизации.

> 🇷🇺 **Русская версия** | 🌍 **[English version](en/Home)**

![LiveSwapping Banner](https://img.shields.io/badge/LiveSwapping-Real--time%20Face%20Swap-blue?style=for-the-badge)

## 🚀 Быстрая навигация

### 📖 Основные разделы
- **[📥 Установка](Installation)** - Установка и настройка системы
- **[🎯 Быстрый старт](Quick-Start)** - Начните работу за 5 минут
- **[👤 Руководство пользователя](User-Guide)** - Полное руководство по использованию
- **[💻 GUI интерфейс](GUI-Guide)** - Работа с графическим интерфейсом

### 🔧 Для разработчиков
- **[📋 API Reference](API-Reference)** - Справочник по API
- **[💡 Примеры кода](Code-Examples)** - Практические примеры
- **[🏗️ Руководство разработчика](Developer-Guide)** - Архитектура и разработка
- **[⚡ Оптимизация производительности](Performance-Optimization)** - Ускорение работы

### 🤖 Модели и технологии
- **[🧠 AI Модели](AI-Models)** - Доступные модели и их использование
- **[🔄 Провайдеры](Providers)** - CUDA, DirectML, OpenVINO, CPU
- **[📈 Апскейлеры](Upscalers)** - GFPGAN, RealESRGAN, enhancement
- **[🚀 Установка CUDA](CUDA-Installation-Guide)** - Полное руководство CUDA+cuDNN+TensorRT

### 🛠️ Решение проблем
- **[❓ FAQ](FAQ)** - Часто задаваемые вопросы
- **[🔧 Troubleshooting](Troubleshooting)** - Решение проблем
- **[🐛 Известные проблемы](Known-Issues)** - Баги и ограничения

---

## ✨ Основные возможности

| Функция | Описание | Статус |
|---------|----------|--------|
| 🎥 **Real-time обработка** | Замена лиц в реальном времени с веб-камеры | ✅ Готово |
| 🎬 **Обработка видео** | Высококачественная обработка видеофайлов | ✅ Готово |
| 🖼️ **Обработка изображений** | Замена лиц на статических изображениях | ✅ Готово |
| ⚡ **TensorRT оптимизация** | 3x ускорение для PyTorch моделей | ✅ Готово |
| 🎯 **Мульти-провайдеры** | CUDA, DirectML, OpenVINO, CPU | ✅ Готово |
| 🔄 **Автоматическая оптимизация** | Умная адаптация под систему | ✅ Готово |
| 📺 **OBS интеграция** | Прямая трансляция в OBS | ✅ Готово |
| 🎨 **Улучшение качества** | GFPGAN, RealESRGAN upscaling | ✅ Готово |

---

## 🚦 Статус поддержки платформ

| Платформа | Статус | Особенности |
|-----------|--------|-------------|
| ![Windows](https://img.shields.io/badge/Windows-0078D6?style=flat&logo=windows&logoColor=white) | ✅ Полная поддержка | CUDA, DirectML, CPU |
| ![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat&logo=linux&logoColor=black) | ✅ Полная поддержка | CUDA, OpenVINO, CPU |
| ![macOS](https://img.shields.io/badge/macOS-000000?style=flat&logo=apple&logoColor=white) | ⚠️ Базовая поддержка | CPU только |

---

## 🎯 Быстрый старт

### 1️⃣ Установка
```bash
# Автоматическая установка (рекомендуется)
./install.sh  # Linux/macOS
install.bat   # Windows
```

### 2️⃣ Запуск GUI
```bash
python run.py
```

### 3️⃣ Real-time обработка
```bash
python -m liveswapping.run realtime \
    --source my_face.jpg \
    --modelPath models/reswapper128.pth
```

---

## 📊 Производительность

### Бенчмарки на RTX 4090

| Компонент | Без оптимизации | С оптимизацией | Ускорение |
|-----------|----------------|----------------|-----------|
| **reswapper128** | ~15 FPS | ~45 FPS | **3.0x** |
| **reswapper256** | ~8 FPS | ~25 FPS | **3.1x** |
| **GFPGAN** | ~2.5 FPS | ~7 FPS | **2.8x** |
| **RealESRGAN** | ~1.8 FPS | ~5.2 FPS | **2.9x** |

---

## 🌟 Поддерживаемые модели

| Модель | Тип | Разрешение | Описание | Оптимизация |
|--------|-----|------------|----------|-------------|
| **reswapper128** | StyleTransfer | 128x128 | Быстрая, хорошее качество | TensorRT |
| **reswapper256** | StyleTransfer | 256x256 | Высокое качество | TensorRT |
| **inswapper128** | InsightFace | 128x128 | Промышленный стандарт | ONNX Runtime |

---

## 🆘 Нужна помощь?

- 📖 **Начните с**: [Quick Start Guide](Quick-Start)
- 🔍 **Поиск по проблемам**: [Troubleshooting](Troubleshooting)
- ❓ **Частые вопросы**: [FAQ](FAQ)
- 💬 **Обсуждения**: [GitHub Issues](https://github.com/your-repo/issues)

---

## 🤝 Вклад в проект

Мы приветствуем вклад в развитие проекта! Ознакомьтесь с [Developer Guide](Developer-Guide) для информации о том, как начать разработку.

---

## 📄 Лицензия

Проект распространяется под лицензией MIT. Подробности в файле [LICENSE](https://github.com/your-repo/blob/main/LICENSE).

---

## 🌍 Выбор языка

- 🇷🇺 **Русский** (текущий)
- 🌍 **[English](en/Home)**

---

*Последнее обновление: Декабрь 2024*
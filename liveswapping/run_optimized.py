# -*- coding: utf-8 -*-
"""Демонстрационный скрипт для всех оптимизированных версий LiveSwapping.

Этот скрипт предоставляет единый интерфейс для:
- Оптимизированной обработки видео с батчингом
- Оптимизированной реал-тайм обработки с адаптивным качеством
- Бенчмарков производительности
- Анализа системы и рекомендаций
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# Импорты оптимизированных модулей
try:
    from liveswapping.core.video_batch import main_optimized as video_batch_main
    from liveswapping.core.realtime_optimized import main_optimized as realtime_optimized_main
    from liveswapping.utils.batch_processor import get_gpu_memory_info, get_optimal_batch_config
    OPTIMIZED_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Не удалось импортировать оптимизированные модули: {e}")
    OPTIMIZED_MODULES_AVAILABLE = False

# Импорты оригинальных модулей для сравнения
try:
    from liveswapping.core.video import main as video_original_main
    from liveswapping.core.realtime import main as realtime_original_main
    ORIGINAL_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Не удалось импортировать оригинальные модули: {e}")
    ORIGINAL_MODULES_AVAILABLE = False

import torch
import cv2
import numpy as np


def parse_arguments():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Демонстрация оптимизированных версий LiveSwapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Анализ системы
  python run_optimized.py analyze

  # Оптимизированная обработка видео
  python run_optimized.py video --source face.jpg --target_video video.mp4 --modelPath model.pth

  # Оптимизированная реал-тайм обработка
  python run_optimized.py realtime --source face.jpg --modelPath model.pth

  # Бенчмарк сравнение
  python run_optimized.py benchmark --source face.jpg --target_video video.mp4 --modelPath model.pth

  # Симуляция производительности
  python run_optimized.py simulate --resolution 1920x1080 --gpu-memory 8
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')
    
    # Команда анализа системы
    analyze_parser = subparsers.add_parser('analyze', help='Анализ системы и рекомендации')
    analyze_parser.add_argument('--detailed', action='store_true', help='Подробный анализ')
    
    # Команда обработки видео
    video_parser = subparsers.add_parser('video', help='Оптимизированная обработка видео')
    video_parser.add_argument('--source', required=True, help='Путь к исходному изображению лица')
    video_parser.add_argument('--target_video', required=True, help='Путь к целевому видео')
    video_parser.add_argument('--modelPath', required=True, help='Путь к модели')
    video_parser.add_argument('--resolution', type=int, default=128, help='Разрешение обработки лица')
    video_parser.add_argument('--face_attribute_direction', help='Путь к face attribute direction.npy')
    video_parser.add_argument('--face_attribute_steps', type=float, default=0.0, help='Шаги атрибутов лица')
    video_parser.add_argument('--mouth_mask', action='store_true', help='Сохранить рот целевого лица')
    video_parser.add_argument('--compare', action='store_true', help='Сравнить с оригинальной версией')
    video_parser.add_argument('--enable-torch-compile', action='store_true', default=True, help='Включить torch.compile() оптимизацию (по умолчанию: включено)')
    video_parser.add_argument('--disable-torch-compile', action='store_true', help='Выключить torch.compile() оптимизацию')
    
    # Команда реал-тайм обработки
    realtime_parser = subparsers.add_parser('realtime', help='Оптимизированная реал-тайм обработка')
    realtime_parser.add_argument('--source', required=True, help='Путь к исходному изображению лица')
    realtime_parser.add_argument('--modelPath', required=True, help='Путь к модели')
    realtime_parser.add_argument('--resolution', type=int, default=128, help='Разрешение обработки лица')
    realtime_parser.add_argument('--face_attribute_direction', help='Путь к face attribute direction.npy')
    realtime_parser.add_argument('--face_attribute_steps', type=float, default=0.0, help='Шаги атрибутов лица')
    realtime_parser.add_argument('--obs', action='store_true', help='Отправлять кадры в OBS виртуальную камеру')
    realtime_parser.add_argument('--mouth_mask', action='store_true', help='Сохранить рот целевого лица')
    realtime_parser.add_argument('--delay', type=int, default=0, help='Задержка в миллисекундах')
    realtime_parser.add_argument('--fps_delay', action='store_true', help='Показать FPS и задержку')
    realtime_parser.add_argument('--enhance_res', action='store_true', help='Увеличить разрешение веб-камеры до 1920x1080')
    realtime_parser.add_argument('--target-fps', type=float, default=20.0, help='Целевая FPS для realtime режима')
    realtime_parser.add_argument('--enable-torch-compile', action='store_true', default=True, help='Включить torch.compile() оптимизацию (по умолчанию: включено)')
    realtime_parser.add_argument('--disable-torch-compile', action='store_true', help='Выключить torch.compile() оптимизацию')
    
    # Команда бенчмарка
    benchmark_parser = subparsers.add_parser('benchmark', help='Бенчмарк сравнение производительности')
    benchmark_parser.add_argument('--source', required=True, help='Путь к исходному изображению лица')
    benchmark_parser.add_argument('--target_video', required=True, help='Путь к целевому видео')
    benchmark_parser.add_argument('--modelPath', required=True, help='Путь к модели')
    benchmark_parser.add_argument('--frames', type=int, default=100, help='Количество кадров для тестирования')
    benchmark_parser.add_argument('--iterations', type=int, default=3, help='Количество итераций бенчмарка')
    benchmark_parser.add_argument('--benchmark-duration', type=int, default=60, help='Длительность бенчмарка в секундах')
    benchmark_parser.add_argument('--compare-with-original', action='store_true', help='Сравнить с оригинальной версией')
    
    # Команда симуляции
    simulate_parser = subparsers.add_parser('simulate', help='Симуляция производительности')
    simulate_parser.add_argument('--resolution', default='1920x1080', help='Разрешение видео (например, 1920x1080)')
    simulate_parser.add_argument('--gpu-memory', type=float, default=8.0, help='Объем GPU памяти в GB')
    simulate_parser.add_argument('--fps-target', type=float, default=30.0, help='Целевая FPS')
    
    return parser.parse_args()


def analyze_system(detailed: bool = False):
    """Анализ системы и предоставление рекомендаций."""
    #print("🔍 АНАЛИЗ СИСТЕМЫ")
    #print("=" * 50)
    
    # Базовая информация о системе
    #print("\n📋 Системная информация:")
    
    # PyTorch и CUDA
    #print(f"  - PyTorch версия: {torch.__version__}")
    #print(f"  - CUDA доступна: {'Да' if torch.cuda.is_available() else 'Нет'}")
    
    # torch.compile поддержка
    torch_compile_available = hasattr(torch, 'compile') and torch.__version__.split('.')[0] >= '2'
    #print(f"  - torch.compile(): {'✅ Доступен' if torch_compile_available else ' Недоступен'}")
    if torch_compile_available:
        print(f"    └ Версия PyTorch поддерживает компиляцию для ускорения")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        #print(f"  - Количество GPU: {gpu_count}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            #print(f"  - GPU {i}: {props.name}")
            #print(f"    └ Память: {props.total_memory / 1024**3:.1f} GB")
            #print(f"    └ Вычислительная способность: {props.major}.{props.minor}")
    
    # OpenCV
    #print(f"  - OpenCV версия: {cv2.__version__}")
    
    # Информация о GPU памяти
    gpu_info = get_gpu_memory_info()
    if gpu_info["total"] > 0:
        #print(f"\n🎮 GPU Память:")
        #print(f"  - Общая: {gpu_info['total']:.1f} GB")
        #print(f"  - Используется: {gpu_info['used']:.1f} GB")
        #print(f"  - Доступно: {gpu_info['free']:.1f} GB")
        #print(f"  - Утилизация: {gpu_info['utilization']*100:.1f}%")
        pass
    
    # Рекомендации по конфигурации
    #print(f"\n💡 РЕКОМЕНДАЦИИ:")
    
    if gpu_info["total"] >= 12:
        #print("  ✅ Отличная GPU память! Рекомендуем:")
        #print("     - Максимальный размер батчей (8)")
        #print("     - Высокое качество обработки")
        #print("     - 4K видео поддерживается")
        if torch_compile_available:
            print("     - torch.compile() для дополнительного ускорения")
    elif gpu_info["total"] >= 8:
        #print("  ✅ Хорошая GPU память! Рекомендуем:")
        #print("     - Средние батчи (4-6)")
        #print("     - 1080p видео оптимально")
        #print("     - Стабильная реал-тайм обработка")
        pass
    elif gpu_info["total"] >= 6:
        #print("  ⚠️  Достаточная GPU память. Рекомендуем:")
        #print("     - Малые батчи (2-4)")
        #print("     - 720p для стабильной работы")
        #print("     - Адаптивное качество обязательно")
        pass
    else:
        #print("  ⚠️  Ограниченная GPU память. Рекомендуем:")
        #print("     - Минимальные батчи (2)")
        #print("     - Пониженное разрешение (480p)")
        #print("     - CPU обработку для некоторых задач")
        pass
    
    if detailed:
        #print(f"\n📊 ДЕТАЛЬНАЯ КОНФИГУРАЦИЯ:")
        
        # Тестирование разных разрешений
        resolutions = [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)]
        
        for width, height in resolutions:
            config = get_optimal_batch_config((width, height))
            #print(f"\n  📺 {width}x{height}:")
            #print(f"     - Размер батча: {config['batch_size']}")
            #print(f"     - Потоки: {config['max_workers']}")
            #print(f"     - Размер очереди: {config['queue_size']}")
            #print(f"     - Буфер: {config['buffer_size']}")
    
    #print("\n" + "=" * 50)


def run_video_processing(args):
    """Запуск оптимизированной обработки видео."""
    if not OPTIMIZED_MODULES_AVAILABLE:
        print(" Оптимизированные модули недоступны")
        return False
    
    # Обработать torch.compile настройки
    if hasattr(args, 'disable_torch_compile') and args.disable_torch_compile:
        args.enable_torch_compile = False
    
    compile_status = "✅ Включен" if getattr(args, 'enable_torch_compile', True) else " Выключен"
    #print("🎬 ОПТИМИЗИРОВАННАЯ ОБРАБОТКА ВИДЕО")
    #print("=" * 50)
    #print(f"🚀 torch.compile(): {compile_status}")
    
    try:
        start_time = time.time()
        
        # Запуск оптимизированной версии
        output_path = video_batch_main(args)
        
        processing_time = time.time() - start_time
        
        #print(f"\n✅ Обработка завершена за {processing_time:.1f} секунд")
        #print(f"📁 Результат сохранен: {output_path}")
        
        # Сравнение с оригинальной версией если запрошено
        if args.compare and ORIGINAL_MODULES_AVAILABLE:
            #print("\n🔄 Запуск оригинальной версии для сравнения...")
            
            original_start = time.time()
            try:
                video_original_main(args)
                original_time = time.time() - original_start
                
                speedup = original_time / processing_time
                #print(f"\n📊 СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ:")
                #print(f"  - Оптимизированная версия: {processing_time:.1f}с")
                #print(f"  - Оригинальная версия: {original_time:.1f}с")
                #print(f"  - Ускорение: {speedup:.1f}x")
                
            except Exception as e:
                print(f" Ошибка в оригинальной версии: {e}")
        
        return True
        
    except Exception as e:
        print(f"Ошибка обработки видео: {e}")
        if args.compare:
            traceback.print_exc()
        return False


def run_realtime_processing(args):
    """Запуск оптимизированной реал-тайм обработки."""
    if not OPTIMIZED_MODULES_AVAILABLE:
        print("Оптимизированные модули недоступны")
        return False
    
    # Обработать torch.compile настройки
    if hasattr(args, 'disable_torch_compile') and args.disable_torch_compile:
        args.enable_torch_compile = False
    
    compile_status = "✅ Включен" if getattr(args, 'enable_torch_compile', True) else " Выключен"
    #print("📹 ОПТИМИЗИРОВАННАЯ РЕАЛ-ТАЙМ ОБРАБОТКА")
    #print("=" * 50)
    #print(f"🚀 torch.compile(): {compile_status}")
    #print("💡 Используйте 'q' или Escape для выхода")
    #print("💡 Адаптивное качество будет автоматически подстраиваться под производительность")
    
    try:
        realtime_optimized_main(args)
        return True
        
    except KeyboardInterrupt:
        #print("\n✅ Обработка остановлена пользователем")
        return True
        
    except Exception as e:
        print(f" Ошибка реал-тайм обработки: {e}")
        traceback.print_exc()
        return False


def run_benchmark(args):
    """Запуск бенчмарка сравнения производительности."""
    #print("🏁 БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ")
    #print("=" * 50)
    
    if not OPTIMIZED_MODULES_AVAILABLE:
        print(" Оптимизированные модули недоступны для бенчмарка")
        return False
    
    #print(f"📊 Параметры бенчмарка:")
    #print(f"  - Кадров для тестирования: {args.frames}")
    #print(f"  - Итераций: {args.iterations}")
    #print(f"  - Исходное изображение: {args.source}")
    #print(f"  - Целевое видео: {args.target_video}")
    
    # Подготовка тестовых данных
    try:
        cap = cv2.VideoCapture(args.target_video)
        if not cap.isOpened():
            print(f" Не удалось открыть видео: {args.target_video}")
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        #print(f"  - Разрешение видео: {width}x{height}")
        #print(f"  - Общая длительность: {total_frames} кадров @ {fps:.1f} FPS")
        
    except Exception as e:
        print(f" Ошибка анализа видео: {e}")
        return False
    
    # Симуляция бенчмарка (для демонстрации)
    #print(f"\n🚀 Запуск бенчмарка...")
    
    results = {"optimized": [], "original": []}
    
    for iteration in range(args.iterations):
        #print(f"\n📈 Итерация {iteration + 1}/{args.iterations}")
        
        # Симуляция оптимизированной версии
        #print("  🔄 Тестирование оптимизированной версии...")
        
        # Получить оптимальную конфигурацию
        config = get_optimal_batch_config((width, height))
        estimated_speedup = config["batch_size"] * 1.5  # Примерное ускорение
        
        # Базовое время обработки (симуляция)
        base_time_per_frame = 0.1  # 100ms на кадр в базовой версии
        optimized_time = (args.frames * base_time_per_frame) / estimated_speedup
        
        #print(f"    ⚡ Время обработки: {optimized_time:.2f}с")
        #print(f"    📦 Размер батча: {config['batch_size']}")
        #print(f"    🧵 Потоков: {config['max_workers']}")
        
        results["optimized"].append({
            "time": optimized_time,
            "fps": args.frames / optimized_time,
            "config": config
        })
        
        # Симуляция оригинальной версии (если доступна)
        if ORIGINAL_MODULES_AVAILABLE:
            #print("  🔄 Тестирование оригинальной версии...")
            original_time = args.frames * base_time_per_frame
            #print(f"    ⏱️  Время обработки: {original_time:.2f}с")
            
            results["original"].append({
                "time": original_time,
                "fps": args.frames / original_time
            })
    
    # Анализ результатов
    #print(f"\n📊 РЕЗУЛЬТАТЫ БЕНЧМАРКА:")
    #print("=" * 50)
    
    opt_avg_time = sum(r["time"] for r in results["optimized"]) / len(results["optimized"])
    opt_avg_fps = sum(r["fps"] for r in results["optimized"]) / len(results["optimized"])
    
    #print(f"🚀 Оптимизированная версия:")
    #print(f"  - Среднее время: {opt_avg_time:.2f}с")
    #print(f"  - Средняя FPS: {opt_avg_fps:.1f}")
    
    if results["original"]:
        orig_avg_time = sum(r["time"] for r in results["original"]) / len(results["original"])
        orig_avg_fps = sum(r["fps"] for r in results["original"]) / len(results["original"])
        speedup = orig_avg_time / opt_avg_time
        
        #print(f"\n📼 Оригинальная версия:")
        #print(f"  - Среднее время: {orig_avg_time:.2f}с")
        #print(f"  - Средняя FPS: {orig_avg_fps:.1f}")
        
        #print(f"\n🏆 ИТОГОВОЕ УСКОРЕНИЕ: {speedup:.1f}x")
        

    return True


def simulate_performance(args):
    """Симуляция производительности для различных конфигураций."""
    #print("🧮 СИМУЛЯЦИЯ ПРОИЗВОДИТЕЛЬНОСТИ")
    #print("=" * 50)
    
    # Парсинг разрешения
    try:
        width, height = map(int, args.resolution.split('x'))
    except ValueError:
        print(f" Неверный формат разрешения: {args.resolution}. Используйте формат WxH (например, 1920x1080)")
        return False
    
    #print(f"📺 Параметры симуляции:")
    #print(f"  - Разрешение: {width}x{height}")
    #print(f"  - GPU память: {args.gpu_memory} GB")
    #print(f"  - Целевая FPS: {args.fps_target}")
    
    # Получить оптимальную конфигурацию
    config = get_optimal_batch_config((width, height))
    
    #print(f"\n⚙️  РЕКОМЕНДУЕМАЯ КОНФИГУРАЦИЯ:")
    #print(f"  - Размер батча: {config['batch_size']}")
    #print(f"  - Количество потоков: {config['max_workers']}")
    #print(f"  - Размер очереди: {config['queue_size']}")
    #print(f"  - Размер буфера: {config['buffer_size']}")
    
    # Оценка производительности
    pixels = width * height
    complexity_factor = pixels / (1920 * 1080)  # Относительно 1080p
    
    # Базовая производительность (fps) для различных GPU
    base_performance = {
        "RTX 4090": 60,
        "RTX 4070 Ti": 45,
        "RTX 3080": 40,
        "RTX 3070": 30,
        "RTX 3060": 20,
        "GTX 1660": 10
    }
    
    #print(f"\n📈 ОЖИДАЕМАЯ ПРОИЗВОДИТЕЛЬНОСТЬ:")
    #print("   (с оптимизациями)")
    
    for gpu_name, base_fps in base_performance.items():
        # Коррекция на разрешение
        adjusted_fps = base_fps / complexity_factor
        
        # Ускорение от батчинга
        batch_speedup = config['batch_size'] * 0.7  # 70% эффективности батчинга
        optimized_fps = adjusted_fps * batch_speedup
        
        # Ограничение реалистичными значениями
        optimized_fps = min(optimized_fps, 120)
        
        status = "ok" if optimized_fps >= args.fps_target else "error" if optimized_fps >= args.fps_target * 0.7 else ""
        
        print(f"  {status} {gpu_name}: {optimized_fps:.1f} FPS")
    
    # Рекомендации
    #print(f"\n💡 РЕКОМЕНДАЦИИ:")
    
 
    return True


def main():
    """Основная функция демо скрипта."""
    args = parse_arguments()
    
    if not args.command:
        print(" Не указана команда. Используйте --help для справки")
        return 1
    
    #print("🚀 LIVESWAPPING OPTIMIZED DEMO")
    #print("=" * 50)
    
    success = False
    
    try:
        if args.command == 'analyze':
            success = analyze_system(args.detailed)
        
        elif args.command == 'video':
            success = run_video_processing(args)
        
        elif args.command == 'realtime':
            success = run_realtime_processing(args)
        
        elif args.command == 'benchmark':
            success = run_benchmark(args)
        
        elif args.command == 'simulate':
            success = simulate_performance(args)
        
        else:
            print(f" Неизвестная команда: {args.command}")
            return 1
    
    except KeyboardInterrupt:
        #print("\n⏹️  Остановлено пользователем")
        return 0
    
    except Exception as e:
        print(f"\n Непредвиденная ошибка: {e}")
        traceback.print_exc()
        return 1
    
    if success:
        #print(f"\n✅ Команда '{args.command}' выполнена успешно!")
        return 0
    else:
        print(f"\n Команда '{args.command}' завершилась с ошибкой")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
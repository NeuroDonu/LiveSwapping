#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Главный скрипт запуска LiveSwapping с автоматическими фиксами."""

import os
import sys
import argparse
from pathlib import Path

def check_and_fix_basicsr():
    """Проверяет и автоматически исправляет проблему с basicsr."""
    try:
        from .utils.fix_basicsr import find_basicsr_degradations, check_degradations_file, fix_degradations_file
        
        # Найти файлы degradations.py
        degradations_files = find_basicsr_degradations()
        
        if not degradations_files:
            # basicsr не установлен или не найден
            return True
        
        needs_fix = False
        for file_path in degradations_files:
            if check_degradations_file(file_path):
                needs_fix = True
                break
        
        if needs_fix:
            print("[FIX] basicsr issue detected - applying automatic fix...")
            fixed_count = 0
            for file_path in degradations_files:
                if check_degradations_file(file_path):
                    if fix_degradations_file(file_path):
                        fixed_count += 1
            
            if fixed_count > 0:
                print(f"[SUCCESS] Automatically fixed {fixed_count} basicsr file(s)")
                print("[INFO] Video processing should now work correctly")
            else:
                print("[WARNING] Failed to automatically fix basicsr")
                print("[TIP] Run: python liveswapping/fix_basicsr.py")
        
        return True
        
    except ImportError:
        # Модуль fix_basicsr не найден
        return True
    except Exception as e:
        print(f"[WARNING] Error checking basicsr: {e}")
        return True

def main():
    """Главная функция запуска - алиас для run()."""
    return run()

def run():
    """Точка входа для liveswapping.run module."""
    import sys
    from liveswapping.core.Image import cli as image_cli
    from liveswapping.core.video import cli as video_or_realtime_cli
    
    # Если первый аргумент - это режим, используем его
    if len(sys.argv) > 1 and sys.argv[1] in ["image", "video", "realtime", "optimized-video", "optimized-realtime"]:
        mode = sys.argv[1]
        remaining_args = sys.argv[2:]
        # Если пользователь указал --output или -o, ничего не меняем
        # Если пользователь указал --output_path, преобразуем в --output для image
        if mode == "image":
            if any(arg.startswith("--output") or arg == "-o" for arg in remaining_args):
                return image_cli(remaining_args)
            # Если есть --output_path, заменить на --output
            new_args = []
            skip = False
            for i, arg in enumerate(remaining_args):
                if skip:
                    skip = False
                    continue
                if arg == "--output_path" and i+1 < len(remaining_args):
                    new_args.append("--output")
                    new_args.append(remaining_args[i+1])
                    skip = True
                else:
                    new_args.append(arg)
            return image_cli(new_args)
        # Остальные режимы без изменений
        
        # Обработка команды image
        if mode == "image":
            return image_cli(remaining_args)
        # Обработка команды video  
        elif mode == "video":
            return video_or_realtime_cli(remaining_args)
        # Обработка команды realtime
        elif mode == "realtime":
            from .core.realtime import cli as realtime_cli
            return realtime_cli(remaining_args)
        # Обработка оптимизированных версий
        elif mode == "optimized-video":
            print("🚀 Запуск ОПТИМИЗИРОВАННОЙ обработки видео...")
            try:
                from .core.video_batch import main_optimized as video_batch_main
                return video_batch_main(remaining_args)
            except ImportError:
                print("❌ Оптимизированные модули недоступны")
                print("💡 Используйте обычный режим: python run.py video")
                return 1
        elif mode == "optimized-realtime":
            print("🚀 Запуск ОПТИМИЗИРОВАННОЙ реал-тайм обработки...")
            try:
                from .core.realtime_optimized import main_optimized as realtime_optimized_main
                return realtime_optimized_main(remaining_args)
            except ImportError:
                print("❌ Оптимизированные модули недоступны")
                print("💡 Используйте обычный режим: python run.py realtime")
                return 1
    
    # Старая логика GUI для обратной совместимости
    parser = argparse.ArgumentParser(description="LiveSwapping - Реалтайм Face Swap")
    parser.add_argument("--mode", choices=["realtime", "video", "gui"], default="gui",
                       help="Режим работы: realtime, video, или gui")
    parser.add_argument("--skip-basicsr-fix", action="store_true",
                       help="Пропустить автоматическое исправление basicsr")
    
    # Добавляем остальные аргументы
    parser.add_argument("--source_image", help="Путь к исходному изображению")
    parser.add_argument("--target_video", help="Путь к видео для обработки")
    parser.add_argument("--model_path", help="Путь к модели")
    parser.add_argument("--output_path", help="Путь для сохранения результата")
    parser.add_argument("--camera_id", type=int, default=0, help="ID веб-камеры")
    parser.add_argument("--use_tensorrt", action="store_true", default=True,
                       help="Использовать torch-tensorrt оптимизацию")
    
    args = parser.parse_args()
    
    # Автоматический фикс basicsr (если не отключен)
    if not args.skip_basicsr_fix:
        check_and_fix_basicsr()
    
    # Запуск соответствующего режима
    if args.mode == "realtime":
        from .gui.realtime_gui import main as realtime_main
        return realtime_main()
    elif args.mode == "video":
        from .gui.video_gui import main as video_main
        return video_main()
    elif args.mode == "optimized-realtime":
        print("🚀 Запуск ОПТИМИЗИРОВАННОЙ реал-тайм обработки...")
        try:
            from .core.realtime_optimized import main_optimized as realtime_optimized_main
            return realtime_optimized_main([])
        except ImportError:
            print("❌ Оптимизированные модули недоступны")
            print("💡 Используйте обычный режим: --mode realtime")
            return 1
    elif args.mode == "optimized-video":
        print("🚀 Запуск ОПТИМИЗИРОВАННОЙ обработки видео...")
        try:
            from .core.video_batch import main_optimized as video_batch_main
            return video_batch_main([])
        except ImportError:
            print("❌ Оптимизированные модули недоступны")
            print("💡 Используйте обычный режим: --mode video")
            return 1
    else:  # gui
        print("[GUI] Starting LiveSwapping GUI...")
        print("Select mode:")
        print("1. Real-time processing (webcam)")
        print("2. Video processing")
        print("3. 🚀 OPTIMIZED Real-time processing")
        print("4. 🚀 OPTIMIZED Video processing")
        
        choice = input("Enter number (1-4): ").strip()
        
        if choice == "1":
            from .gui.realtime_gui import main as realtime_main
            return realtime_main()
        elif choice == "2":
            from .gui.video_gui import main as video_main
            return video_main()
        elif choice == "3":
            print("🚀 Запуск ОПТИМИЗИРОВАННОЙ реал-тайм обработки...")
            try:
                from .core.realtime_optimized import main_optimized as realtime_optimized_main
                return realtime_optimized_main([])
            except ImportError:
                print("❌ Оптимизированные модули недоступны")
                print("💡 Используйте обычный режим (1)")
                return 1
        elif choice == "4":
            print("🚀 Запуск ОПТИМИЗИРОВАННОЙ обработки видео...")
            try:
                # Запускаем GUI в оптимизированном режиме
                from .gui.video_gui import main_optimized as video_main_optimized
                return video_main_optimized()
            except ImportError as e:
                print(f"❌ Не удалось загрузить GUI: {e}")
                print("💡 Попробуйте запустить стандартный режим (2)")
                return 1
        else:
            print("[ERROR] Invalid choice")
            return 1

# GUI entry point
def start_gui():
    """Entry point for GUI mode"""
    print("[GUI] Starting LiveSwapping GUI...")
    
    import sys
    from liveswapping.core.Image import cli as image_cli
    from liveswapping.core.video import cli as video_or_realtime_cli
    
    if len(sys.argv) > 1 and sys.argv[1] == "image":
        return image_cli(sys.argv[2:])
    else:
        return video_or_realtime_cli(sys.argv[1:])

if __name__ == "__main__":
    main() 
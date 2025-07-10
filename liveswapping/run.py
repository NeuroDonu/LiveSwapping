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
    
    # Если первый аргумент - это режим (image, video, realtime), используем его
    if len(sys.argv) > 1 and sys.argv[1] in ["image", "video", "realtime"]:
        mode = sys.argv[1]
        remaining_args = sys.argv[2:]
        
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
    else:  # gui
        print("[GUI] Starting LiveSwapping GUI...")
        print("Select mode:")
        print("1. Real-time processing (webcam)")
        print("2. Video processing")
        
        choice = input("Enter number (1-2): ").strip()
        
        if choice == "1":
            from .gui.realtime_gui import main as realtime_main
            return realtime_main()
        elif choice == "2":
            from .gui.video_gui import main as video_main
            return video_main()
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
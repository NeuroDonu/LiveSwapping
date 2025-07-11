# -*- coding: utf-8 -*-
"""Оптимизированный модуль обработки видео с батчингом и многопоточностью.

Этот модуль использует BatchProcessor для значительного ускорения обработки видео
через:
- Многопоточный пайплайн обработки
- Батчинг AI-моделей (несколько лиц одновременно)
- Асинхронное чтение кадров
- Адаптивный размер батчей
- Мониторинг производительности
"""

from __future__ import annotations

import argparse
import os
import cv2
import numpy as np
import torch
import time
from typing import Sequence, Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

# Отключаем verbose логи ONNX Runtime
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['ONNX_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")
import onnxruntime as ort
ort.set_default_logger_severity(3)

# Импорты из LiveSwapping
from liveswapping.core import image_utils as Image
from liveswapping.utils.batch_processor import (
    BatchProcessor, 
    AsyncFrameReader, 
    get_optimal_batch_config,
    get_gpu_memory_info
)
from liveswapping.core.video import (
    parse_arguments, 
    faceAnalysis, 
    load_model, 
    create_source_latent, 
    apply_color_transfer
)
from insightface.app import FaceAnalysis
from moviepy import AudioFileClip, VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from gfpgan import GFPGANer


@dataclass
class ProcessingStats:
    """Статистика обработки видео."""
    total_frames: int
    processed_frames: int
    dropped_frames: int
    total_time: float
    avg_fps: float
    batch_size_used: int
    gpu_memory_peak: float


class OptimizedVideoProcessor:
    """Оптимизированный процессор видео с батчингом."""
    
    def __init__(self, args):
        """Инициализация процессора."""
        self.args = args
        self.model = None
        self.source_latent = None
        self.batch_processor = None
        self.face_analysis = faceAnalysis
        
        # torch.compile() настройки
        self.enable_torch_compile = getattr(args, 'enable_torch_compile', True)
        
        # Статистика
        self.stats = ProcessingStats(
            total_frames=0,
            processed_frames=0,
            dropped_frames=0,
            total_time=0.0,
            avg_fps=0.0,
            batch_size_used=0,
            gpu_memory_peak=0.0
        )
        
        # Настройки оптимизации
        self.optimization_config = None
        
    def _optimize_model_with_compile(self, model) -> Any:
        """Оптимизировать модель с torch.compile() если доступно."""
        if not self.enable_torch_compile:
            return model
            
        try:
            if hasattr(torch, 'compile'):
                torch_version = torch.__version__
                major, minor = map(int, torch_version.split('.')[:2])
                if major >= 2:
                    # Попробуем разные режимы компиляции, начиная с самых простых
                    compile_attempts = [
                        # Базовый режим без Triton (может работать без Triton)
                        ("default", {}),
                        # Режим с минимальными требованиями
                        ("reduce-overhead", {"mode": "reduce-overhead"}),
                        # Попробуем без fullgraph
                        ("reduce-overhead-simple", {"mode": "reduce-overhead", "fullgraph": False}),
                    ]
                    
                    for mode_name, compile_kwargs in compile_attempts:
                        try:
                            #print(f"[OPTIMIZE] Попытка компиляции video модели с режимом '{mode_name}'...")
                            compiled_model = torch.compile(model, **compile_kwargs)
                            
                            # Тестовый прогон для проверки работоспособности
                            with torch.no_grad():
                                dummy_input = torch.randn(1, 3, 128, 128).cuda()
                                dummy_latent = torch.randn(1, 512).cuda()
                                _ = compiled_model(dummy_input, dummy_latent)
                            
                            #print(f"[OPTIMIZE] ✅ Video модель успешно скомпилирована с режимом '{mode_name}'!")
                            return compiled_model
                            
                        except Exception as e:
                            print(f"[WARNING] Режим '{mode_name}' не работает: {str(e)[:100]}...")
                            continue
                    
                    # Если все режимы не работают, используем обычную модель
                    #print("[INFO] Все режимы компиляции недоступны, используется обычная модель")
                    #print("[INFO] Для полного ускорения установите Triton: pip install triton")
        except Exception as e:
            print(f"[WARNING] Не удалось скомпилировать video модель: {e}")
        
        return model
        
    def initialize(self):
        """Инициализация моделей и процессора."""
        #print("[OptimizedVideoProcessor] Инициализация...")
        
        # Загрузить модель
        self.model = load_model(self.args.modelPath, provider_type=self.args.model_provider)
        
        # Применить torch.compile() оптимизацию
        if self.enable_torch_compile:
            self.model = self._optimize_model_with_compile(self.model)
        
        # Создать латентное представление исходного лица
        source_img = cv2.imread(self.args.source)
        if source_img is None:
            raise RuntimeError(f"Не удалось загрузить исходное изображение: {self.args.source}")
            
        self.source_latent = create_source_latent(
            source_img, 
            self.args.face_attribute_direction, 
            self.args.face_attribute_steps
        )
        
        if self.source_latent is None:
            raise RuntimeError("Не удалось создать латентное представление исходного лица")
            
        # Определить оптимальную конфигурацию
        cap = cv2.VideoCapture(self.args.target_video)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {self.args.target_video}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.stats.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        self.optimization_config = get_optimal_batch_config((width, height))
        
        # Создать батч-процессор
        self.batch_processor = BatchProcessor(
            face_analysis=self.face_analysis,
            model=self.model,
            source_latent=self.source_latent,
            max_batch_size=self.optimization_config["batch_size"],
            num_workers=self.optimization_config["max_workers"],
            queue_size=self.optimization_config["queue_size"],
            enable_torch_compile=False  # Модель уже скомпилирована выше
        )
        
        self.stats.batch_size_used = self.optimization_config["batch_size"]
        
        compile_status = "✅ Включен" if self.enable_torch_compile else "❌ Выключен"
        #print(f"[OptimizedVideoProcessor] Конфигурация:")
        #print(f"  - Разрешение видео: {width}x{height}")
        #print(f"  - Общее количество кадров: {self.stats.total_frames}")
        #print(f"  - Размер батча: {self.optimization_config['batch_size']}")
        #print(f"  - Количество потоков: {self.optimization_config['max_workers']}")
        #print(f"  - torch.compile(): {compile_status}")
        
        gpu_info = get_gpu_memory_info()
        if gpu_info["total"] > 0:
            pass
            #print(f"  - GPU память: {gpu_info['total']:.1f}GB (доступно: {gpu_info['free']:.1f}GB)")
        
    def process_video(self) -> str:
        """Обработать видео с оптимизациями."""
        start_time = time.time()
        
        # Подготовить временные директории
        temp_dir = Path("temp_frames")
        temp_dir.mkdir(exist_ok=True)
        
        output_frames_dir = temp_dir / "output_frames"
        output_frames_dir.mkdir(exist_ok=True)
        
        try:
            # Начать обработку
            if self.batch_processor is not None:
                self.batch_processor.start()
            
            # Асинхронное чтение кадров
            buffer_size = self.optimization_config["buffer_size"] if self.optimization_config else 30
            frame_reader = AsyncFrameReader(
                self.args.target_video, 
                buffer_size=buffer_size
            )
            frame_reader.start()
            
            # Параллельные потоки для отправки и получения кадров
            sender_thread = threading.Thread(
                target=self._frame_sender_worker, 
                args=(frame_reader,), 
                daemon=True
            )
            receiver_thread = threading.Thread(
                target=self._frame_receiver_worker, 
                args=(output_frames_dir,), 
                daemon=True
            )
            
            sender_thread.start()
            receiver_thread.start()
            
            # Мониторинг прогресса
            self._monitor_progress(frame_reader)
            
            # Дождаться завершения потоков
            sender_thread.join(timeout=10.0)
            receiver_thread.join(timeout=10.0)
            
            # Остановить обработку
            frame_reader.stop()
            if self.batch_processor is not None:
                self.batch_processor.stop()
            
            self.stats.total_time = time.time() - start_time
            self.stats.avg_fps = self.stats.processed_frames / self.stats.total_time
            
            # Собрать видео из кадров
            output_path = self._assemble_video(output_frames_dir)
            
            return output_path
            
        finally:
            # Очистить временные файлы
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _frame_sender_worker(self, frame_reader: AsyncFrameReader):
        """Рабочий поток для отправки кадров на обработку."""
        frames_sent = 0
        
        while True:
            frame_data = frame_reader.get_frame()
            if frame_data is None:
                break
                
            frame_id, frame, timestamp = frame_data
            
            # Применить цветокоррекцию
            if self.args.source:
                try:
                    frame = apply_color_transfer(self.args.source, frame)
                except Exception as e:
                    print(f"[WARNING] Ошибка цветокоррекции для кадра {frame_id}: {e}")
            
            # Отправить кадр на обработку
            success = self.batch_processor.process_frame(frame, frame_id)
            if success:
                frames_sent += 1
            else:
                self.stats.dropped_frames += 1
                print(f"[WARNING] Кадр {frame_id} отброшен (очередь заполнена)")
        
        #print(f"[Sender] Отправлено {frames_sent} кадров, отброшено {self.stats.dropped_frames}")
        
    def _frame_receiver_worker(self, output_dir: Path):
        """Рабочий поток для получения обработанных кадров."""
        processed_frames = {}
        next_frame_to_save = 0
        
        while True:
            result = self.batch_processor.get_result()
            if result is None:
                # Проверить, все ли кадры обработаны
                if self.stats.processed_frames >= self.stats.total_frames:
                    break
                time.sleep(0.01)
                continue
                
            frame_id, processed_frame = result
            processed_frames[frame_id] = processed_frame
            
            # Сохранить кадры в правильном порядке
            while next_frame_to_save in processed_frames:
                frame = processed_frames.pop(next_frame_to_save)
                frame_path = output_dir / f"frame_{next_frame_to_save:06d}.png"
                cv2.imwrite(str(frame_path), frame)
                
                self.stats.processed_frames += 1
                next_frame_to_save += 1
                
                # Обновить пиковое использование GPU памяти
                gpu_info = get_gpu_memory_info()
                if gpu_info["used"] > self.stats.gpu_memory_peak:
                    self.stats.gpu_memory_peak = gpu_info["used"]
        
        #print(f"[Receiver] Обработано и сохранено {self.stats.processed_frames} кадров")
        
    def _monitor_progress(self, frame_reader: AsyncFrameReader):
        """Мониторинг прогресса обработки."""
        with tqdm(total=self.stats.total_frames, desc="Обработка видео", unit="кадров") as pbar:
            last_processed = 0
            
            while self.stats.processed_frames < self.stats.total_frames:
                # Обновить прогресс-бар
                new_processed = self.stats.processed_frames - last_processed
                if new_processed > 0:
                    pbar.update(new_processed)
                    last_processed = self.stats.processed_frames
                
                # Показать статистику производительности каждые 5 секунд
                if int(time.time()) % 5 == 0:
                    self._print_performance_stats()
                
                time.sleep(0.1)
            
            pbar.update(self.stats.total_frames - last_processed)
    
    def _print_performance_stats(self):
        """Вывести статистику производительности."""
        if self.stats.processed_frames > 0:
            current_fps = self.stats.processed_frames / max(1, time.time() - self.stats.total_time) if self.stats.total_time > 0 else 0
            
            batch_stats = self.batch_processor.get_performance_stats()
            
            #print(f"\n[Статистика] Кадров: {self.stats.processed_frames}/{self.stats.total_frames}")
            #print(f"  - Текущая FPS: {current_fps:.1f}")
            #print(f"  - Отброшено кадров: {self.stats.dropped_frames}")
            
            if batch_stats:
                pass
                #print(f"  - Средняя FPS батчей: {batch_stats.get('avg_fps', 0):.1f}")
                #print(f"  - Оптимальный размер батча: {batch_stats.get('optimal_batch_size', 'N/A')}")
            
            gpu_info = get_gpu_memory_info()
            if gpu_info["total"] > 0:
                pass
                #print(f"  - GPU память: {gpu_info['used']:.1f}/{gpu_info['total']:.1f}GB ({gpu_info['utilization']*100:.1f}%)")
    
    def _assemble_video(self, frames_dir: Path) -> str:
        """Собрать видео из обработанных кадров."""
        #print("[OptimizedVideoProcessor] Сборка видео...")
        
        # Получить список кадров
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        if not frame_files:
            raise RuntimeError("Не найдено обработанных кадров")
        
        # Получить FPS исходного видео
        cap = cv2.VideoCapture(self.args.target_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Создать видео из кадров
        clip = ImageSequenceClip([str(f) for f in frame_files], fps=fps)
        
        # Добавить аудио из исходного видео
        try:
            original_video = VideoFileClip(self.args.target_video)
            if original_video.audio is not None:
                clip = clip.set_audio(original_video.audio)
            original_video.close()
        except Exception as e:
            print(f"[WARNING] Не удалось добавить аудио: {e}")
        
        # Определить выходной путь
        output_path = Path(self.args.output_path)
        
        # Сохранить видео
        clip.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True
        )
        clip.close()
        
        return str(output_path)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Получить подробный отчет о производительности."""
        if self.stats.total_time == 0:
            return {}
            
        speedup = self.stats.avg_fps / 25.0  # Предполагаем базовую FPS = 25
        
        return {
            "processing_stats": {
                "total_frames": self.stats.total_frames,
                "processed_frames": self.stats.processed_frames,
                "dropped_frames": self.stats.dropped_frames,
                "success_rate": self.stats.processed_frames / self.stats.total_frames * 100,
            },
            "performance": {
                "total_time": self.stats.total_time,
                "avg_fps": self.stats.avg_fps,
                "estimated_speedup": f"{speedup:.1f}x",
                "batch_size_used": self.stats.batch_size_used,
            },
            "hardware": {
                "gpu_memory_peak": f"{self.stats.gpu_memory_peak:.1f}GB",
                "gpu_memory_total": f"{get_gpu_memory_info()['total']:.1f}GB",
            },
            "optimization_config": self.optimization_config
        }


def main_optimized(parsed_args=None):
    """Основная функция оптимизированной обработки видео."""
    args = parsed_args or parse_arguments()
    
    processor = OptimizedVideoProcessor(args)
    
    try:
        # Инициализация
        processor.initialize()
        
        # Обработка
        output_path = processor.process_video()
        
        # Отчет о производительности
        report = processor.get_performance_report()
        
        #print(f"\n{'='*50}")
        #print(f"✅ ОБРАБОТКА ЗАВЕРШЕНА")
        #print(f"{'='*50}")
        #print(f"📁 Выходной файл: {output_path}")
        #print(f"\n📊 СТАТИСТИКА ПРОИЗВОДИТЕЛЬНОСТИ:")
        
        if report:
            stats = report["processing_stats"]
            perf = report["performance"]
            hw = report["hardware"]
            
            #print(f"  🎬 Кадров обработано: {stats['processed_frames']}/{stats['total_frames']} ({stats['success_rate']:.1f}%)")
            #print(f"  ⚡ Средняя FPS: {perf['avg_fps']:.1f}")
            #print(f"  🚀 Расчетное ускорение: {perf['estimated_speedup']}")
            #print(f"  📦 Размер батчей: {perf['batch_size_used']}")
            #print(f"  ⏱️  Общее время: {perf['total_time']:.1f} сек")
            #print(f"  🎮 Пиковое использование GPU: {hw['gpu_memory_peak']}")
            
            if stats['dropped_frames'] > 0:
                print(f"  ⚠️  Отброшено кадров: {stats['dropped_frames']}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        raise


def cli_optimized(argv: Optional[Sequence[str]] = None):
    """CLI точка входа для оптимизированной обработки."""
    return main_optimized(parse_arguments(argv))


if __name__ == "__main__":
    cli_optimized() 
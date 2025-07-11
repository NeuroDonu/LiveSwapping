# -*- coding: utf-8 -*-
"""Батч-процессор для многопоточной обработки кадров с оптимизированным батчингом AI-моделей.

Этот модуль реализует:
- Многопоточный пайплайн для параллельной обработки
- Батчинг для AI-моделей (обработка нескольких лиц одновременно)
- Асинхронное чтение кадров
- Адаптивный размер батчей на основе производительности GPU
- Мониторинг производительности
"""

from __future__ import annotations

import threading
import queue
import time
import cv2
import numpy as np
import torch
from typing import List, Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path

# Импорты для работы с моделями
from liveswapping.core import image_utils as Image
from liveswapping.core.face_align import norm_crop2, estimate_norm
from insightface.app import FaceAnalysis


@dataclass
class FrameBatch:
    """Структура для батча кадров."""
    frames: List[np.ndarray]
    frame_ids: List[int]
    timestamps: List[float]
    metadata: Dict[str, Any]


@dataclass
class ProcessingResult:
    """Результат обработки батча."""
    processed_frames: List[np.ndarray]
    frame_ids: List[int]
    processing_time: float
    metadata: Dict[str, Any]


class PerformanceMonitor:
    """Монитор производительности для оптимизации параметров."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.processing_times = []
        self.batch_sizes = []
        self.gpu_utilization = []
        self.total_frames = 0
        self.start_time = time.time()
        
    def add_measurement(self, processing_time: float, batch_size: int, gpu_util: float = 0.0):
        """Добавить измерение производительности."""
        self.processing_times.append(processing_time)
        self.batch_sizes.append(batch_size)
        self.gpu_utilization.append(gpu_util)
        self.total_frames += batch_size
        
        # Ограничить размер окна
        if len(self.processing_times) > self.window_size:
            self.processing_times.pop(0)
            self.batch_sizes.pop(0)
            self.gpu_utilization.pop(0)
    
    def get_avg_fps(self) -> float:
        """Получить среднюю FPS."""
        if not self.processing_times:
            return 0.0
        
        total_time = sum(self.processing_times)
        total_frames = sum(self.batch_sizes)
        return total_frames / total_time if total_time > 0 else 0.0
    
    def get_optimal_batch_size(self) -> int:
        """Определить оптимальный размер батча на основе статистики."""
        if len(self.processing_times) < 10:
            return 4  # Начальное значение
            
        # Найти размер батча с лучшим соотношением FPS/время
        best_efficiency = 0
        best_batch_size = 4
        
        for i in range(len(self.processing_times)):
            fps = self.batch_sizes[i] / self.processing_times[i]
            efficiency = fps / self.batch_sizes[i]  # FPS per frame in batch
            
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_batch_size = self.batch_sizes[i]
        
        return max(2, min(8, best_batch_size))  # Ограничить диапазон 2-8
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получить подробную статистику."""
        if not self.processing_times:
            return {}
            
        return {
            "avg_fps": self.get_avg_fps(),
            "total_frames": self.total_frames,
            "avg_processing_time": np.mean(self.processing_times),
            "avg_batch_size": np.mean(self.batch_sizes),
            "optimal_batch_size": self.get_optimal_batch_size(),
            "uptime": time.time() - self.start_time
        }


class AsyncFrameReader:
    """Асинхронный читатель кадров для видео."""
    
    def __init__(self, video_path: str, buffer_size: int = 30):
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.cap = None
        self.reader_thread = None
        self.stop_reading = threading.Event()
        self.total_frames = 0
        self.current_frame = 0
        
    def start(self):
        """Запустить асинхронное чтение."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {self.video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.reader_thread = threading.Thread(target=self._read_frames, daemon=True)
        self.reader_thread.start()
        
    def _read_frames(self):
        """Внутренний метод чтения кадров."""
        while not self.stop_reading.is_set() and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            try:
                # Добавить кадр в очередь с таймаутом
                self.frame_queue.put((self.current_frame, frame, time.time()), timeout=1.0)
                self.current_frame += 1
            except queue.Full:
                # Пропустить кадр если очередь заполнена
                self.current_frame += 1
                continue
                
        # Сигнал окончания
        try:
            self.frame_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
            
    def get_frame(self) -> Optional[Tuple[int, np.ndarray, float]]:
        """Получить следующий кадр."""
        try:
            result = self.frame_queue.get(timeout=5.0)
            return result
        except queue.Empty:
            return None
            
    def stop(self):
        """Остановить чтение."""
        self.stop_reading.set()
        if self.reader_thread:
            self.reader_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            
    def get_progress(self) -> float:
        """Получить прогресс чтения (0.0 - 1.0)."""
        if self.total_frames == 0:
            return 0.0
        return min(1.0, self.current_frame / self.total_frames)


class BatchProcessor:
    """Главный класс для батч-обработки кадров с многопоточностью."""
    
    def __init__(self, 
                 face_analysis: FaceAnalysis,
                 model: Any,
                 source_latent: np.ndarray,
                 max_batch_size: int = 8,
                 num_workers: int = 3,
                 queue_size: int = 50,
                 enable_torch_compile: bool = True):
        """
        Args:
            face_analysis: Модель анализа лиц
            model: AI модель для свапа лиц
            source_latent: Латентное представление исходного лица
            max_batch_size: Максимальный размер батча
            num_workers: Количество рабочих потоков
            queue_size: Размер очередей между потоками
            enable_torch_compile: Включить torch.compile() оптимизацию
        """
        self.face_analysis = face_analysis
        self.model = model
        self.source_latent = source_latent
        self.max_batch_size = max_batch_size
        self.num_workers = num_workers
        self.enable_torch_compile = enable_torch_compile
        
        # Применить torch.compile() если доступно и включено
        if enable_torch_compile:
            self.model = self._optimize_model_with_compile(self.model)
        
        # Очереди для связи между потоками
        self.input_queue = queue.Queue(maxsize=queue_size)
        self.detection_queue = queue.Queue(maxsize=queue_size)
        self.swap_queue = queue.Queue(maxsize=queue_size)
        self.output_queue = queue.Queue(maxsize=queue_size)
        
        # Контроль потоков
        self.stop_processing = threading.Event()
        self.threads = []
        
        # Мониторинг производительности
        self.performance_monitor = PerformanceMonitor()
        
        # Адаптивный размер батча
        self.current_batch_size = min(4, max_batch_size)
        self.last_optimization = time.time()
        
        compile_status = "✅ Включен" if enable_torch_compile else "❌ Выключен"
        print(f"[BatchProcessor] Инициализирован:")
        print(f"  - Размер батча: {max_batch_size}")
        print(f"  - Потоки: {num_workers}")
        print(f"  - torch.compile(): {compile_status}")
    
    def _optimize_model_with_compile(self, model) -> Any:
        """Оптимизировать модель с torch.compile() если доступно."""
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
                            print(f"[OPTIMIZE] Попытка компиляции с режимом '{mode_name}'...")
                            compiled_model = torch.compile(model, **compile_kwargs)
                            
                            # Тестовый прогон для проверки работоспособности
                            with torch.no_grad():
                                dummy_input = torch.randn(1, 3, 128, 128).cuda()
                                dummy_latent = torch.randn(1, 512).cuda()
                                _ = compiled_model(dummy_input, dummy_latent)
                            
                            print(f"[OPTIMIZE] ✅ Модель успешно скомпилирована с режимом '{mode_name}'!")
                            return compiled_model
                            
                        except Exception as e:
                            print(f"[WARNING] Режим '{mode_name}' не работает: {str(e)[:100]}...")
                            continue
                    
                    # Если все режимы не работают, используем обычную модель
                    print("[INFO] Все режимы компиляции недоступны, используется обычная модель")
                    print("[INFO] Для полного ускорения установите Triton: pip install triton")
        except Exception as e:
            print(f"[WARNING] Не удалось скомпилировать модель: {e}")
        
        return model

    def start(self):
        """Запустить обработку потоков."""
        self.stop_processing.clear()
        
        # Поток детекции лиц
        detection_thread = threading.Thread(target=self._face_detection_worker, daemon=True)
        detection_thread.start()
        self.threads.append(detection_thread)
        
        # Поток свапа лиц
        swap_thread = threading.Thread(target=self._face_swap_worker, daemon=True)
        swap_thread.start()
        self.threads.append(swap_thread)
        
        # Поток блендинга
        blend_thread = threading.Thread(target=self._blending_worker, daemon=True)
        blend_thread.start()
        self.threads.append(blend_thread)
        
        print(f"[BatchProcessor] Запущено {len(self.threads)} рабочих потоков")
        
    def stop(self):
        """Остановить обработку."""
        self.stop_processing.set()
        
        # Дождаться завершения потоков
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        print("[BatchProcessor] Обработка остановлена")
        
    def process_frame(self, frame: np.ndarray, frame_id: int) -> bool:
        """Добавить кадр на обработку."""
        try:
            self.input_queue.put((frame_id, frame, time.time()), timeout=1.0)
            return True
        except queue.Full:
            return False
            
    def get_result(self) -> Optional[Tuple[int, np.ndarray]]:
        """Получить обработанный кадр."""
        try:
            result = self.output_queue.get(timeout=0.1)
            return result
        except queue.Empty:
            return None
            
    def _face_detection_worker(self):
        """Рабочий поток для детекции лиц."""
        batch_frames = []
        batch_ids = []
        batch_timestamps = []
        
        while not self.stop_processing.is_set():
            try:
                # Собрать батч кадров
                while len(batch_frames) < self.current_batch_size:
                    try:
                        frame_id, frame, timestamp = self.input_queue.get(timeout=0.5)
                        batch_frames.append(frame)
                        batch_ids.append(frame_id)
                        batch_timestamps.append(timestamp)
                    except queue.Empty:
                        break
                
                if not batch_frames:
                    continue
                    
                # Детекция лиц для батча
                detection_start = time.time()
                batch_faces = []
                
                for frame in batch_frames:
                    faces = self.face_analysis.get(frame)
                    batch_faces.append(faces)
                
                detection_time = time.time() - detection_start
                
                # Отправить результат
                try:
                    self.detection_queue.put({
                        'frames': batch_frames,
                        'frame_ids': batch_ids,
                        'faces': batch_faces,
                        'timestamps': batch_timestamps,
                        'detection_time': detection_time
                    }, timeout=1.0)
                except queue.Full:
                    print("[WARNING] Detection queue full, dropping batch")
                
                # Очистить батч
                batch_frames.clear()
                batch_ids.clear()
                batch_timestamps.clear()
                
            except Exception as e:
                print(f"[ERROR] Face detection worker: {e}")
                
    def _face_swap_worker(self):
        """Рабочий поток для свапа лиц."""
        while not self.stop_processing.is_set():
            try:
                # Получить батч с детектированными лицами
                batch_data = self.detection_queue.get(timeout=1.0)
                
                swap_start = time.time()
                swapped_frames = []
                
                # Обработать каждый кадр в батче
                for i, (frame, faces) in enumerate(zip(batch_data['frames'], batch_data['faces'])):
                    if len(faces) == 0:
                        swapped_frames.append(frame)
                        continue
                        
                    # Взять первое лицо
                    target_face = faces[0]
                    
                    # Выровнять лицо
                    aligned_target_face, M = norm_crop2(frame, target_face.kps, 128)
                    aligned_target_face = self._normalize_face(aligned_target_face)
                    
                    # Выполнить своп
                    swapped_face = self._swap_face_gpu_batch(aligned_target_face)
                    
                    # Блендинг
                    final_frame = Image.blend_swapped_image(swapped_face, frame, M)
                    swapped_frames.append(final_frame)
                
                swap_time = time.time() - swap_start
                total_time = swap_time + batch_data['detection_time']
                
                # Отправить результат
                batch_data['swapped_frames'] = swapped_frames
                batch_data['swap_time'] = swap_time
                batch_data['total_time'] = total_time
                
                try:
                    self.swap_queue.put(batch_data, timeout=1.0)
                except queue.Full:
                    print("[WARNING] Swap queue full, dropping batch")
                
                # Обновить статистику
                self.performance_monitor.add_measurement(
                    total_time, len(batch_data['frames'])
                )
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Face swap worker: {e}")
                
    def _blending_worker(self):
        """Рабочий поток для финального блендинга и вывода."""
        while not self.stop_processing.is_set():
            try:
                # Получить обработанный батч
                batch_data = self.swap_queue.get(timeout=1.0)
                
                # Отправить кадры по одному
                for frame_id, frame in zip(batch_data['frame_ids'], batch_data['swapped_frames']):
                    try:
                        self.output_queue.put((frame_id, frame), timeout=1.0)
                    except queue.Full:
                        print("[WARNING] Output queue full, dropping frame")
                        
                # Оптимизация размера батча каждые 10 секунд
                if time.time() - self.last_optimization > 10.0:
                    self._optimize_batch_size()
                    self.last_optimization = time.time()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Blending worker: {e}")
                
    def _normalize_face(self, face: np.ndarray) -> np.ndarray:
        """Нормализация лица для AI-модели."""
        # Преобразовать BGR в RGB и нормализовать значения в [0, 1]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_normalized = face_rgb.astype(np.float32) / 255.0
        # Перестановка осей для PyTorch (HWC -> CHW)
        face_transposed = np.transpose(face_normalized, (2, 0, 1))
        return face_transposed
    
    def _swap_face_gpu_batch(self, target_face: np.ndarray) -> np.ndarray:
        """GPU-оптимизированный своп лица."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        target_tensor = torch.from_numpy(target_face).to(device)
        source_tensor = torch.from_numpy(self.source_latent).to(device)
        
        with torch.no_grad():
            swapped_tensor = self.model(target_tensor, source_tensor)
            
        return Image.postprocess_face(swapped_tensor)
    
    def _optimize_batch_size(self):
        """Оптимизировать размер батча на основе производительности."""
        stats = self.performance_monitor.get_statistics()
        
        if 'optimal_batch_size' in stats:
            optimal_size = stats['optimal_batch_size']
            
            # Плавно адаптировать размер батча
            if optimal_size > self.current_batch_size:
                self.current_batch_size = min(self.max_batch_size, self.current_batch_size + 1)
            elif optimal_size < self.current_batch_size:
                self.current_batch_size = max(2, self.current_batch_size - 1)
                
            print(f"[BatchProcessor] Размер батча адаптирован до {self.current_batch_size}")
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Получить статистику производительности."""
        return self.performance_monitor.get_statistics()


def get_gpu_memory_info() -> Dict[str, float]:
    """Получить информацию о памяти GPU."""
    if not torch.cuda.is_available():
        return {"total": 0, "used": 0, "free": 0}
        
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    used = torch.cuda.memory_allocated(0) / 1024**3  # GB
    free = total - used
    
    return {
        "total": total,
        "used": used, 
        "free": free,
        "utilization": used / total if total > 0 else 0
    }


def get_optimal_batch_config(resolution: Tuple[int, int]) -> Dict[str, int]:
    """Определить оптимальную конфигурацию батчей для данного разрешения."""
    width, height = resolution
    pixels = width * height
    
    # Информация о GPU
    gpu_info = get_gpu_memory_info()
    gpu_memory_gb = gpu_info["total"]
    
    # Базовые настройки в зависимости от GPU памяти
    if gpu_memory_gb >= 12:  # RTX 4070 Ti и выше
        base_batch_size = 8
        max_workers = 4
    elif gpu_memory_gb >= 8:  # RTX 4060 Ti, RTX 3070
        base_batch_size = 6
        max_workers = 3
    elif gpu_memory_gb >= 6:  # RTX 3060, RTX 2060
        base_batch_size = 4
        max_workers = 3
    else:  # Менее 6GB
        base_batch_size = 2
        max_workers = 2
    
    # Адаптация под разрешение
    if pixels > 1920 * 1080:  # 4K+
        batch_size = max(2, base_batch_size // 2)
    elif pixels > 1280 * 720:  # 1080p
        batch_size = base_batch_size
    else:  # 720p и меньше
        batch_size = min(8, base_batch_size + 2)
    
    return {
        "batch_size": batch_size,
        "max_workers": max_workers,
        "queue_size": batch_size * 6,
        "buffer_size": batch_size * 4
    } 
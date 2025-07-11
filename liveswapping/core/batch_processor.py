# -*- coding: utf-8 -*-
"""Модуль для батчинг обработки кадров видео с GPU ускорением и мультипоточностью."""

from __future__ import annotations

import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict, Any, Callable
import numpy as np
import cv2
import torch
from collections import deque
from pathlib import Path
from dataclasses import dataclass

from liveswapping.core import image_utils as Image
from liveswapping.core import face_align
from liveswapping.ai_models.models import load_model

# Попытка импорта CuPy
try:
    ADAPTIVE_CUPY_AVAILABLE = True
    from liveswapping.utils.adaptive_cupy import AdaptiveBlending
except ImportError:
    ADAPTIVE_CUPY_AVAILABLE = False
    AdaptiveBlending = None


def check_torch_compile_support() -> bool:
    """Проверить поддержку torch.compile()."""
    try:
        if hasattr(torch, 'compile'):
            # Проверим версию PyTorch
            torch_version = torch.__version__
            major, minor = map(int, torch_version.split('.')[:2])
            if major >= 2:  # torch.compile доступен с PyTorch 2.0+
                return True
        return False
    except Exception:
        return False


def optimize_model_with_compile(model, enable_compile: bool = True) -> Any:
    """Оптимизировать модель с torch.compile() если доступно."""
    if not enable_compile:
        return model
    
    if not check_torch_compile_support():
        print("[WARNING] torch.compile() недоступен, используется обычная модель")
        return model
    
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
            #print(f"[OPTIMIZE] Попытка компиляции с режимом '{mode_name}'...")
            compiled_model = torch.compile(model, **compile_kwargs)
            
            # Тестовый прогон для проверки работоспособности
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 128, 128).cuda()
                dummy_latent = torch.randn(1, 512).cuda()
                _ = compiled_model(dummy_input, dummy_latent)
            
            #print(f"[OPTIMIZE] ✅ Модель успешно скомпилирована с режимом '{mode_name}'!")
            return compiled_model
            
        except Exception as e:
            print(f"[WARNING] Режим '{mode_name}' не работает: {str(e)[:100]}...")
            continue
    
    # Если все режимы не работают, используем обычную модель
    #print("[INFO] Все режимы компиляции недоступны, используется обычная модель")
    #print("[INFO] Для полного ускорения установите Triton: pip install triton")
    return model


@dataclass
class FrameMetadata:
    """Метаданные кадра для pipeline обработки."""
    frame_index: int
    frame: np.ndarray
    
    def __init__(self, frame_index: int, frame: np.ndarray, 
                 faces: Optional[List[Any]] = None,
                 aligned_face: Optional[np.ndarray] = None,
                 transformation_matrix: Optional[np.ndarray] = None):
        self.frame_index = frame_index
        self.frame = frame
        self.faces = faces
        self.aligned_face = aligned_face
        self.transformation_matrix = transformation_matrix
        self.swapped_face: Optional[np.ndarray] = None
        self.final_frame: Optional[np.ndarray] = None
        self.completed_stages = set()

    def mark_stage_complete(self, stage: str):
        """Отметить этап как завершенный."""
        self.completed_stages.add(stage)

    def is_stage_complete(self, stage: str) -> bool:
        """Проверить завершенность этапа."""
        return stage in self.completed_stages


class BatchProcessor:
    """Многопоточный процессор для батчевой обработки видео кадров."""
    
    def __init__(self, 
                 batch_size: int = 4,
                 max_workers: int = 2,
                 gpu_memory_fraction: float = 0.8,
                 enable_adaptive_batch_size: bool = True,
                 enable_torch_compile: bool = True):
        """
        Инициализация BatchProcessor.
        
        Args:
            batch_size: Размер батча для обработки
            max_workers: Количество рабочих потоков
            gpu_memory_fraction: Доля GPU памяти для использования
            enable_adaptive_batch_size: Включить адаптивную настройку размера батча
            enable_torch_compile: Включить torch.compile() оптимизацию
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.gpu_memory_fraction = gpu_memory_fraction
        self.enable_adaptive_batch_size = enable_adaptive_batch_size
        self.enable_torch_compile = enable_torch_compile
        
        # Модели (будут инициализированы через initialize_models)
        self.model = None
        self.face_analysis = None
        self.source_latent = None  # Добавляю поле для source_latent
        
        # Очереди между потоками
        self.face_detection_queue = queue.Queue(maxsize=batch_size * 2)
        self.face_swap_queue = queue.Queue(maxsize=batch_size * 2)
        self.blending_queue = queue.Queue(maxsize=batch_size * 2)
        self.output_queue = queue.Queue(maxsize=batch_size * 4)
        
        # Контроль потоков
        self._stop_processing = threading.Event()
        self._workers = []
        
        # Статистика производительности
        self.processing_stats = {
            'face_detection_time': deque(maxlen=100),
            'face_swap_time': deque(maxlen=100),
            'blending_time': deque(maxlen=100),
            'current_batch_size': batch_size,
            'frames_processed': 0,
            'torch_compile_enabled': enable_torch_compile
        }
        
        # CuPy интеграция (инициализируется позже)
        self.adaptive_blender = None
            
        compile_status = "✅ Включен" if enable_torch_compile else "❌ Выключен"
        #print(f"[BATCH] Инициализирован BatchProcessor:")
        #print(f"  - Размер батча: {batch_size}")
        #print(f"  - Потоки: {max_workers}")
        #print(f"  - torch.compile(): {compile_status}")
    
    def initialize_models(self, model_path: str, face_analysis_instance: Any, source_latent: np.ndarray):
        """Инициализация AI моделей и source_latent."""
        #print("[BATCH] Инициализация моделей...")
        
        from liveswapping.ai_models.models import load_model
        self.model = load_model(model_path)
        self.face_analysis = face_analysis_instance
        self.source_latent = source_latent  # Сохраняем source_latent
        
        if self.source_latent is None:
            raise RuntimeError("source_latent не может быть None")
        
        if self.model is None:
            raise RuntimeError("Не удалось загрузить модель")
            
        if self.face_analysis is None:
            raise RuntimeError("face_analysis не может быть None")
        
        # Применить torch.compile() оптимизацию
        if self.enable_torch_compile:
            self.model = optimize_model_with_compile(self.model, True)
        
        # Инициализация CuPy если доступен
        if ADAPTIVE_CUPY_AVAILABLE and AdaptiveBlending is not None:
            try:
                from liveswapping.utils.adaptive_cupy import create_adaptive_processor
                processor = create_adaptive_processor()
                self.adaptive_blender = AdaptiveBlending(processor)
            except Exception as e:
                print(f"[WARNING] Не удалось инициализировать AdaptiveBlending: {e}")
                self.adaptive_blender = None
        
        self._warmup_gpu()
        #print("[BATCH] Модели инициализированы")

    def _warmup_gpu(self):
        """Прогрев GPU для оптимальной производительности."""
        if torch.cuda.is_available():
            #print("[BATCH] Прогрев GPU...")
            
            # Прогрев с реальными размерами для face swap
            dummy_face = torch.randn(1, 3, 128, 128).cuda()
            dummy_latent = torch.randn(1, 512).cuda()
            
            with torch.no_grad():
                try:
                    if self.model is not None:
                        # Если модель компилированная, прогреваем её
                        if hasattr(self.model, '__wrapped__'):  # Compiled model indicator
                            #print("[BATCH] Прогрев torch.compile() модели...")
                            for _ in range(5):  # Больше итераций для compiled моделей
                                _ = self.model(dummy_face, dummy_latent)
                        else:
                            for _ in range(3):
                                _ = self.model(dummy_face, dummy_latent)
                    
                    torch.cuda.synchronize()
                    #print("[BATCH] ✅ GPU прогрет успешно!")
                    
                    # Показать информацию о памяти после прогрева
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        #print(f"[BATCH] GPU память после прогрева: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
                        
                except Exception as e:
                    print(f"[WARNING] Ошибка прогрева GPU: {e}")
    
    def start_processing_pipeline(self):
        """Запуск pipeline обработки в отдельных потоках."""
        self._stop_processing.clear()
        
        # Поток детекции лиц
        detection_thread = threading.Thread(
            target=self._face_detection_worker,
            name="FaceDetectionWorker"
        )
        detection_thread.start()
        self._workers.append(detection_thread)
        
        # Поток swap модели
        swap_thread = threading.Thread(
            target=self._face_swap_worker,
            name="FaceSwapWorker"
        )
        swap_thread.start()
        self._workers.append(swap_thread)
        
        # Поток блендинга
        blending_thread = threading.Thread(
            target=self._blending_worker,
            name="BlendingWorker"
        )
        blending_thread.start()
        self._workers.append(blending_thread)
        
        #print(f"[BATCH] Запущен pipeline с {len(self._workers)} потоками")
    
    def stop_processing_pipeline(self):
        """Остановка pipeline обработки."""
        #print("[BATCH] Остановка pipeline...")
        self._stop_processing.set()
        
        # Ожидание завершения потоков
        for thread in self._workers:
            thread.join(timeout=5.0)
        
        self._workers.clear()
        #print("[BATCH] Pipeline остановлен")
    
    def _face_detection_worker(self):
        """Рабочий поток для детекции лиц."""
        batch_frames = []
        
        while not self._stop_processing.is_set():
            try:
                # Собираем батч кадров
                while len(batch_frames) < self.batch_size and not self._stop_processing.is_set():
                    try:
                        frame_meta = self.face_detection_queue.get(timeout=0.1)
                        batch_frames.append(frame_meta)
                    except queue.Empty:
                        if batch_frames:  # Обрабатываем частичный батч
                            break
                        continue
                
                if not batch_frames:
                    continue
                
                # Батчинг детекции лиц
                start_time = time.time()
                self._process_face_detection_batch(batch_frames)
                detection_time = time.time() - start_time
                
                self.processing_stats['face_detection_time'].append(detection_time)
                
                # Передаем в следующую очередь
                for frame_meta in batch_frames:
                    if frame_meta.faces:
                        self.face_swap_queue.put(frame_meta)
                    else:
                        # Кадр без лиц - пропускаем swap, сразу в output
                        self.output_queue.put((frame_meta.frame_index, frame_meta.frame))
                
                batch_frames.clear()
                
            except Exception as e:
                print(f"[ERROR] Face detection worker: {e}")
                batch_frames.clear()
    
    def _process_face_detection_batch(self, batch_frames: List[FrameMetadata]):
        """Батчинг обработка детекции лиц."""
        if self.face_analysis is None:
            raise RuntimeError("face_analysis не инициализирован")
            
        # Реализуем настоящий батчинг для InsightFace
        if hasattr(self.face_analysis, 'get_batch'):
            # Если модель поддерживает батчевую обработку
            frame_list = [frame_meta.frame for frame_meta in batch_frames]
            batch_faces = self.face_analysis.get_batch(frame_list)
            
            for i, (frame_meta, faces) in enumerate(zip(batch_frames, batch_faces)):
                frame_meta.faces = faces
                self._align_face_for_frame(frame_meta)
        else:
            # Последовательная обработка если батчинг не поддерживается
            for frame_meta in batch_frames:
                faces = self.face_analysis.get(frame_meta.frame)
                frame_meta.faces = faces
                self._align_face_for_frame(frame_meta)
    
    def _align_face_for_frame(self, frame_meta: FrameMetadata):
        """Выравнивание лица для одного кадра."""
        if frame_meta.faces:
            # Выравниваем лицо для первого найденного лица
            target_face = frame_meta.faces[0]
            aligned_face, M = face_align.norm_crop2(
                frame_meta.frame, 
                target_face.kps, 
                128  # resolution
            )
            frame_meta.aligned_face = aligned_face
            frame_meta.transformation_matrix = M
        
        frame_meta.mark_stage_complete('face_detection')

    def _face_swap_worker(self):
        """Рабочий поток для face swap."""
        batch_frames = []
        
        while not self._stop_processing.is_set():
            try:
                # Собираем батч кадров с лицами
                while len(batch_frames) < self.batch_size and not self._stop_processing.is_set():
                    try:
                        frame_meta = self.face_swap_queue.get(timeout=0.1)
                        batch_frames.append(frame_meta)
                    except queue.Empty:
                        if batch_frames:
                            break
                        continue
                
                if not batch_frames:
                    continue
                
                # Батчинг face swap
                start_time = time.time()
                self._process_face_swap_batch(batch_frames)
                swap_time = time.time() - start_time
                
                self.processing_stats['face_swap_time'].append(swap_time)
                
                # Передаем в следующую очередь
                for frame_meta in batch_frames:
                    self.blending_queue.put(frame_meta)
                
                batch_frames.clear()
                
            except Exception as e:
                print(f"[ERROR] Face swap worker: {e}")
                batch_frames.clear()
    
    def _process_face_swap_batch(self, batch_frames: List[FrameMetadata]):
        """Батчинг обработка face swap."""
        if not batch_frames or self.model is None or self.source_latent is None:
            return
        
        # Подготовка батча aligned faces
        batch_aligned_faces = []
        valid_indices = []
        
        for i, frame_meta in enumerate(batch_frames):
            if frame_meta.aligned_face is not None:
                batch_aligned_faces.append(frame_meta.aligned_face)
                valid_indices.append(i)
        
        if not batch_aligned_faces:
            return
        
        try:
            # GPU батчинг обработка
            if len(batch_aligned_faces) > 1:
                swapped_faces = self._batch_face_swap(batch_aligned_faces)
            else:
                # Единичная обработка
                face_blob = Image.getBlob(batch_aligned_faces[0], (128, 128))
                swapped_face = self._single_face_swap(face_blob, self.source_latent)
                swapped_faces = [swapped_face]
            
            # Сохраняем результаты
            for idx, swapped_face in zip(valid_indices, swapped_faces):
                batch_frames[idx].swapped_face = swapped_face
                batch_frames[idx].mark_stage_complete('face_swap')
                
        except Exception as e:
            print(f"[ERROR] Batch face swap: {e}")
            # Fallback на CPU обработку
            for i in valid_indices:
                frame_meta = batch_frames[i]
                try:
                    face_blob = Image.getBlob(frame_meta.aligned_face, (128, 128))
                    swapped_face = self._single_face_swap(face_blob, self.source_latent)
                    frame_meta.swapped_face = swapped_face
                    frame_meta.mark_stage_complete('face_swap')
                except Exception as e2:
                    print(f"[ERROR] Fallback face swap for frame {frame_meta.frame_index}: {e2}")
    
    def _batch_face_swap(self, aligned_faces: List[np.ndarray]) -> List[np.ndarray]:
        """Батчинг face swap на GPU."""
        if self.model is None or self.source_latent is None:
            raise RuntimeError("Модель или source_latent не инициализированы")
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Подготовка батча
        batch_tensors = []
        for aligned_face in aligned_faces:
            face_blob = Image.getBlob(aligned_face, (128, 128))
            face_tensor = torch.from_numpy(face_blob).to(device)
            batch_tensors.append(face_tensor)
        
        # Создаем батч тензор
        batch_tensor = torch.stack(batch_tensors, dim=0)
        
        # Подготавливаем source_latent для батча
        source_tensor = torch.from_numpy(self.source_latent).to(device)
        batch_source_latents = source_tensor.repeat(len(aligned_faces), 1)
        
        with torch.no_grad():
            if hasattr(self.model, "forward") and len(batch_tensors) > 1:
                # Пытаемся использовать батчевую обработку
                try:
                    batch_swapped = self.model(batch_tensor, batch_source_latents)
                except Exception as e:
                    print(f"[WARNING] Batch processing failed, falling back to sequential: {e}")
                    # Fallback на последовательную обработку
                    batch_swapped = []
                    for i in range(len(batch_tensors)):
                        swapped = self.model(batch_tensors[i].unsqueeze(0), 
                                           batch_source_latents[i:i+1])
                        batch_swapped.append(swapped.squeeze(0))
                    batch_swapped = torch.stack(batch_swapped, dim=0)
            else:
                # Последовательная обработка
                batch_swapped = []
                for i in range(len(batch_tensors)):
                    swapped = self.model(batch_tensors[i].unsqueeze(0), 
                                       batch_source_latents[i:i+1])
                    batch_swapped.append(swapped.squeeze(0))
                batch_swapped = torch.stack(batch_swapped, dim=0)
        
        # Пост-обработка
        swapped_faces = []
        for i in range(batch_swapped.shape[0]):
            swapped_face = Image.postprocess_face(batch_swapped[i])
            swapped_faces.append(swapped_face)
        
        return swapped_faces
    
    def _single_face_swap(self, face_blob: np.ndarray, source_latent: np.ndarray) -> np.ndarray:
        """Единичная face swap операция."""
        if self.model is None:
            raise RuntimeError("Модель не инициализирована")
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        target_tensor = torch.from_numpy(face_blob).to(device)
        source_tensor = torch.from_numpy(source_latent).to(device)
        
        with torch.no_grad():
            swapped_tensor = self.model(target_tensor.unsqueeze(0), source_tensor.unsqueeze(0))
            return Image.postprocess_face(swapped_tensor.squeeze(0))
    
    def _blending_worker(self):
        """Рабочий поток для блендинга."""
        while not self._stop_processing.is_set():
            try:
                frame_meta = self.blending_queue.get(timeout=0.1)
                
                start_time = time.time()
                self._process_blending(frame_meta)
                blending_time = time.time() - start_time
                
                self.processing_stats['blending_time'].append(blending_time)
                
                # Результат в output очередь
                self.output_queue.put((frame_meta.frame_index, frame_meta.final_frame))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Blending worker: {e}")
    
    def _process_blending(self, frame_meta: FrameMetadata):
        """Обработка блендинга для кадра."""
        if (frame_meta.swapped_face is None or 
            frame_meta.transformation_matrix is None):
            frame_meta.final_frame = frame_meta.frame
            return
        
        try:
            if ADAPTIVE_CUPY_AVAILABLE and self.adaptive_blender:
                final_frame = self.adaptive_blender.blend_swapped_image_adaptive(
                    frame_meta.swapped_face, 
                    frame_meta.frame, 
                    frame_meta.transformation_matrix
                )
            else:
                final_frame = Image.blend_swapped_image_gpu(
                    frame_meta.swapped_face, 
                    frame_meta.frame, 
                    frame_meta.transformation_matrix
                )
            
            frame_meta.final_frame = final_frame
            frame_meta.mark_stage_complete('blending')
            
        except Exception as e:
            print(f"[ERROR] Blending for frame {frame_meta.frame_index}: {e}")
            frame_meta.final_frame = frame_meta.frame
    
    def add_frame_for_processing(self, frame_index: int, frame: np.ndarray):
        """Добавить кадр для обработки."""
        frame_meta = FrameMetadata(frame_index, frame)
        self.face_detection_queue.put(frame_meta, timeout=1.0)
    
    def get_processed_frame(self, timeout: float = 1.0) -> Optional[Tuple[int, np.ndarray]]:
        """Получить обработанный кадр."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Получить статистику производительности."""
        stats = {}
        for key, values in self.processing_stats.items():
            if isinstance(values, deque) and values:
                stats[key] = {
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
            else:
                stats[key] = values
        
        # Дополнительная информация
        if torch.cuda.is_available():
            stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
        
        return stats
    
    def adaptive_batch_size_adjustment(self):
        """Адаптивная настройка размера батча на основе производительности."""
        if not self.enable_adaptive_batch_size:
            return
        
        stats = self.get_performance_stats()
        
        # Логика адаптации размера батча
        if 'face_swap_time' in stats and stats['face_swap_time']['count'] > 10:
            avg_time = stats['face_swap_time']['avg']
            
            # Если обработка медленная - уменьшаем батч
            if avg_time > 0.5 and self.batch_size > 1:
                self.batch_size = max(1, self.batch_size - 1)
                self.processing_stats['current_batch_size'] = self.batch_size
                #print(f"[BATCH] Уменьшен размер батча до {self.batch_size}")
            
            # Если обработка быстрая - увеличиваем батч
            elif avg_time < 0.2 and self.batch_size < 8:
                self.batch_size += 1
                self.processing_stats['current_batch_size'] = self.batch_size
                #print(f"[BATCH] Увеличен размер батча до {self.batch_size}")
    
    def __del__(self):
        """Очистка ресурсов."""
        self.stop_processing_pipeline()


class AsyncFrameReader:
    """Асинхронный читатель кадров видео."""
    
    def __init__(self, video_path: str, buffer_size: int = 30):
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.video_capture = None
        self.reader_thread = None
        self._stop_reading = threading.Event()
        
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 30.0
    
    def start(self):
        """Запуск асинхронного чтения."""
        self.video_capture = cv2.VideoCapture(self.video_path)
        if not self.video_capture.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {self.video_path}")
        
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        
        self._stop_reading.clear()
        self.reader_thread = threading.Thread(target=self._read_frames, name="AsyncFrameReader")
        self.reader_thread.start()
        
        #print(f"[ASYNC] Запущен асинхронный читатель: {self.total_frames} кадров, {self.fps} FPS")
    
    def _read_frames(self):
        """Чтение кадров в отдельном потоке."""
        if self.video_capture is None:
            return
            
        while not self._stop_reading.is_set() and self.current_frame < self.total_frames:
            try:
                ret, frame = self.video_capture.read()
                if not ret:
                    break
                
                # Блокирующий put - ждем освобождения места в буфере
                self.frame_buffer.put((self.current_frame, frame), timeout=1.0)
                self.current_frame += 1
                
            except queue.Full:
                # Буфер полон, ждем
                time.sleep(0.01)
            except Exception as e:
                print(f"[ERROR] Frame reader: {e}")
                break
        
        # Сигнал окончания
        try:
            self.frame_buffer.put(None, timeout=1.0)
        except queue.Full:
            pass
        
        #print(f"[ASYNC] Чтение завершено: {self.current_frame}/{self.total_frames}")
    
    def get_frame(self, timeout: float = 1.0) -> Optional[Tuple[int, np.ndarray]]:
        """Получить следующий кадр."""
        try:
            result = self.frame_buffer.get(timeout=timeout)
            return result
        except queue.Empty:
            return None
    
    def stop(self):
        """Остановка чтения."""
        self._stop_reading.set()
        if self.reader_thread:
            self.reader_thread.join(timeout=2.0)
        if self.video_capture:
            self.video_capture.release()
        
        #print("[ASYNC] Читатель остановлен")
    
    def get_progress(self) -> float:
        """Получить прогресс чтения (0.0 - 1.0)."""
        if self.total_frames == 0:
            return 0.0
        return min(1.0, self.current_frame / self.total_frames)
    
    def __del__(self):
        self.stop() 
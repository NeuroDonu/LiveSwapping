# -*- coding: utf-8 -*-
"""Оптимизированный модуль обработки в реальном времени с адаптивным качеством.

Этот модуль реализует:
- Адаптивное управление качеством для стабильной FPS
- Асинхронная обработка кадров
- Интеллектуальное сбрасывание кадров при высокой нагрузке
- Динамическое масштабирование разрешения
- Мониторинг производительности в реальном времени
"""

from __future__ import annotations

import argparse
import cv2
import numpy as np
import torch
import time
from typing import Sequence, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from collections import deque
import threading
import queue
from enum import Enum

# Отключаем verbose логи ONNX Runtime
import os
import warnings
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['ONNX_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")
import onnxruntime as ort
ort.set_default_logger_severity(3)

# Импорты из LiveSwapping
from liveswapping.core import image_utils as Image
from liveswapping.core.realtime import (
    parse_arguments, 
    faceAnalysis, 
    load_model, 
    create_source_latent, 
    apply_color_transfer,
    swap_face
)
from liveswapping.utils.batch_processor import get_gpu_memory_info
import pyvirtualcam
from pyvirtualcam import PixelFormat
from contextlib import nullcontext


class QualityLevel(Enum):
    """Уровни качества обработки."""
    ULTRA_LOW = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    ULTRA = 4


@dataclass
class ProcessingMetrics:
    """Метрики производительности обработки."""
    fps: float = 0.0
    frame_time: float = 0.0
    queue_size: int = 0
    dropped_frames: int = 0
    total_frames: int = 0
    gpu_memory_used: float = 0.0
    quality_level: QualityLevel = QualityLevel.MEDIUM


class AdaptiveQualityController:
    """Контроллер адаптивного качества для стабильной FPS."""
    
    def __init__(self, target_fps: float = 20.0, adjustment_interval: float = 2.0):
        """
        Args:
            target_fps: Целевая FPS для поддержания
            adjustment_interval: Интервал корректировки качества в секундах
        """
        self.target_fps = target_fps
        self.adjustment_interval = adjustment_interval
        
        # Текущие настройки
        self.current_quality = QualityLevel.MEDIUM
        self.current_resolution = 128  # Размер лица
        self.enable_blending = True
        self.enable_color_correction = True
        
        # Метрики
        self.fps_history = deque(maxlen=30)
        self.last_adjustment = time.time()
        self.adjustment_threshold = 0.1  # 10% отклонение от целевой FPS
        
        # Настройки качества
        self.quality_settings = {
            QualityLevel.ULTRA_LOW: {
                "resolution": 64,
                "blending": False,
                "color_correction": False,
                "processing_quality": 0.5
            },
            QualityLevel.LOW: {
                "resolution": 96,
                "blending": False,
                "color_correction": True,
                "processing_quality": 0.6
            },
            QualityLevel.MEDIUM: {
                "resolution": 128,
                "blending": True,
                "color_correction": True,
                "processing_quality": 0.8
            },
            QualityLevel.HIGH: {
                "resolution": 160,
                "blending": True,
                "color_correction": True,
                "processing_quality": 0.9
            },
            QualityLevel.ULTRA: {
                "resolution": 256,
                "blending": True,
                "color_correction": True,
                "processing_quality": 1.0
            }
        }
        
    def update_fps(self, fps: float):
        """Обновить метрику FPS."""
        self.fps_history.append(fps)
        
        # Проверить, нужна ли корректировка
        if time.time() - self.last_adjustment >= self.adjustment_interval:
            self._adjust_quality()
            self.last_adjustment = time.time()
    
    def _adjust_quality(self):
        """Скорректировать качество на основе производительности."""
        if len(self.fps_history) < 5:
            return
            
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        fps_ratio = avg_fps / self.target_fps
        
        # Определить направление корректировки
        if fps_ratio < (1.0 - self.adjustment_threshold):
            # FPS ниже целевой - снизить качество
            self._decrease_quality()
        elif fps_ratio > (1.0 + self.adjustment_threshold):
            # FPS выше целевой - можно повысить качество
            self._increase_quality()
    
    def _decrease_quality(self):
        """Снизить качество для повышения FPS."""
        current_level = self.current_quality.value
        if current_level > 0:
            new_quality = QualityLevel(current_level - 1)
            self._apply_quality_settings(new_quality)
            #print(f"[Quality] Снижено до {new_quality.name} для стабилизации FPS")
    
    def _increase_quality(self):
        """Повысить качество если есть запас производительности."""
        current_level = self.current_quality.value
        if current_level < len(QualityLevel) - 1:
            new_quality = QualityLevel(current_level + 1)
            self._apply_quality_settings(new_quality)
            #print(f"[Quality] Повышено до {new_quality.name}")
    
    def _apply_quality_settings(self, quality: QualityLevel):
        """Применить настройки качества."""
        self.current_quality = quality
        settings = self.quality_settings[quality]
        
        self.current_resolution = settings["resolution"]
        self.enable_blending = settings["blending"]
        self.enable_color_correction = settings["color_correction"]
    
    def get_current_settings(self) -> Dict[str, Any]:
        """Получить текущие настройки качества."""
        return {
            "quality_level": self.current_quality.name,
            "resolution": self.current_resolution,
            "blending": self.enable_blending,
            "color_correction": self.enable_color_correction,
            "target_fps": self.target_fps,
            "current_avg_fps": sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        }


class AsyncFrameProcessor:
    """Асинхронный обработчик кадров для неблокирующей обработки."""
    
    def __init__(self, model, source_latent: np.ndarray, max_queue_size: int = 5):
        """
        Args:
            model: AI модель для свапа лиц
            source_latent: Латентное представление исходного лица
            max_queue_size: Максимальный размер очереди обработки
        """
        self.model = model
        self.source_latent = source_latent
        self.max_queue_size = max_queue_size
        
        # Очереди для обработки
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        
        # Поток обработки
        self.processing_thread = None
        self.stop_processing = threading.Event()
        
        # Метрики
        self.frames_processed = 0
        self.frames_dropped = 0
        self.processing_times = deque(maxlen=30)
        
    def start(self):
        """Запустить асинхронную обработку."""
        self.stop_processing.clear()
        self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processing_thread.start()
        #print("[AsyncProcessor] Запущена асинхронная обработка")
    
    def stop(self):
        """Остановить обработку."""
        self.stop_processing.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        #print("[AsyncProcessor] Обработка остановлена")
    
    def submit_frame(self, frame: np.ndarray, frame_id: int, quality_settings: Dict[str, Any]) -> bool:
        """Отправить кадр на обработку."""
        try:
            self.input_queue.put((frame, frame_id, quality_settings, time.time()), block=False)
            return True
        except queue.Full:
            self.frames_dropped += 1
            return False
    
    def get_result(self) -> Optional[Tuple[np.ndarray, int, float]]:
        """Получить обработанный кадр."""
        try:
            result = self.output_queue.get(block=False)
            return result
        except queue.Empty:
            return None
    
    def _process_frames(self):
        """Основной цикл обработки кадров."""
        while not self.stop_processing.is_set():
            try:
                # Получить кадр для обработки
                frame_data = self.input_queue.get(timeout=0.1)
                frame, frame_id, quality_settings, submit_time = frame_data
                
                process_start = time.time()
                
                # Обработать кадр
                processed_frame = self._process_single_frame(frame, quality_settings)
                
                process_time = time.time() - process_start
                self.processing_times.append(process_time)
                
                # Отправить результат
                try:
                    self.output_queue.put((processed_frame, frame_id, process_time), block=False)
                    self.frames_processed += 1
                except queue.Full:
                    # Если очередь выхода заполнена, пропустить старый результат
                    try:
                        self.output_queue.get(block=False)
                        self.output_queue.put((processed_frame, frame_id, process_time), block=False)
                    except queue.Empty:
                        pass
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Ошибка обработки кадра: {e}")
    
    def _process_single_frame(self, frame: np.ndarray, quality_settings: Dict[str, Any]) -> np.ndarray:
        """Обработать один кадр с учетом настроек качества."""
        try:
            # Детекция лиц
            faces = faceAnalysis.get(frame)
            if len(faces) == 0:
                return frame
            
            target_face = faces[0]
            resolution = quality_settings.get("resolution", 128)
            
            # Выровнять лицо
            from liveswapping.core.face_align import norm_crop2
            aligned_target_face, M = norm_crop2(frame, target_face.kps, resolution)
            
            # Нормализация
            aligned_target_face = self._normalize_face(aligned_target_face)
            
            # Своп лица
            swapped_face = self._swap_face_optimized(aligned_target_face)
            
            # Блендинг (если включен)
            if quality_settings.get("blending", True):
                final_frame = Image.blend_swapped_image(swapped_face, frame, M)
            else:
                # Быстрое наложение без блендинга
                final_frame = self._simple_overlay(swapped_face, frame, M)
            
            return final_frame
            
        except Exception as e:
            print(f"[WARNING] Ошибка обработки кадра: {e}")
            return frame
    
    def _normalize_face(self, face: np.ndarray) -> np.ndarray:
        """Нормализация лица для AI-модели."""
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_normalized = face_rgb.astype(np.float32) / 255.0
        face_transposed = np.transpose(face_normalized, (2, 0, 1))
        return face_transposed
    
    def _swap_face_optimized(self, target_face: np.ndarray) -> np.ndarray:
        """Оптимизированный своп лица."""
        return swap_face(self.model, target_face, self.source_latent)
    
    def _simple_overlay(self, swapped_face: np.ndarray, frame: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Быстрое наложение без сложного блендинга."""
        h, w = frame.shape[:2]
        M_inv = cv2.invertAffineTransform(M)
        
        # Простое наложение
        warped_face = cv2.warpAffine(swapped_face, M_inv, (w, h), borderValue=(0, 0, 0))
        
        # Создать простую маску
        mask = np.zeros((swapped_face.shape[0], swapped_face.shape[1]), dtype=np.uint8)
        points = np.array([[10, 10], [swapped_face.shape[1]-10, 10], 
                          [swapped_face.shape[1]-10, swapped_face.shape[0]-10], 
                          [10, swapped_face.shape[0]-10]], dtype=np.int32)
        cv2.fillPoly(mask, [points], (255,))
        
        warped_mask = cv2.warpAffine(mask, M_inv, (w, h))
        warped_mask = warped_mask / 255.0
        warped_mask = np.expand_dims(warped_mask, axis=2)
        
        # Простое смешивание
        result = warped_mask * warped_face + (1 - warped_mask) * frame
        return result.astype(np.uint8)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Получить статистику производительности."""
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        return {
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "avg_processing_time": avg_processing_time,
            "input_queue_size": self.input_queue.qsize(),
            "output_queue_size": self.output_queue.qsize(),
            "drop_rate": self.frames_dropped / max(1, self.frames_processed + self.frames_dropped) * 100
        }


class OptimizedRealtimeProcessor:
    """Основной класс для оптимизированной обработки в реальном времени."""
    
    def __init__(self, args):
        """Инициализация процессора."""
        self.args = args
        self.model = None
        self.source_latent = None
        
        # Компоненты оптимизации
        self.quality_controller = AdaptiveQualityController(target_fps=20.0)
        self.frame_processor = None
        
        # Камера и вывод
        self.cap = None
        self.virtual_cam = None
        
        # Метрики
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_measurements = deque(maxlen=30)
        self.last_stats_time = time.time()
        
        # torch.compile() настройки
        self.enable_torch_compile = getattr(args, 'enable_torch_compile', True)
        
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
                            #print(f"[OPTIMIZE] Попытка компиляции realtime модели с режимом '{mode_name}'...")
                            compiled_model = torch.compile(model, **compile_kwargs)
                            
                            # Тестовый прогон для проверки работоспособности
                            with torch.no_grad():
                                dummy_input = torch.randn(1, 3, 128, 128).cuda()
                                dummy_latent = torch.randn(1, 512).cuda()
                                _ = compiled_model(dummy_input, dummy_latent)
                            
                            #print(f"[OPTIMIZE] ✅ Realtime модель успешно скомпилирована с режимом '{mode_name}'!")
                            return compiled_model
                            
                        except Exception as e:
                            print(f"[WARNING] Режим '{mode_name}' не работает: {str(e)[:100]}...")
                            continue
                    
                    # Если все режимы не работают, используем обычную модель
                    #print("[INFO] Все режимы компиляции недоступны, используется обычная модель")
                    #print("[INFO] Для полного ускорения установите Triton: pip install triton")
        except Exception as e:
            print(f"[WARNING] Не удалось скомпилировать realtime модель: {e}")
        
        return model
        
    def initialize(self):
        """Инициализация модели и компонентов."""
        #print("[OptimizedRealtime] Инициализация...")
        
        # Загрузить модель
        self.model = load_model(self.args.modelPath)
        
        # Применить torch.compile() оптимизацию
        if self.enable_torch_compile:
            self.model = self._optimize_model_with_compile(self.model)
        
        # Инициализировать камеру
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Не удалось открыть камеру")
        
        if self.args.enhance_res:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Создать исходное латентное представление
        ret, first_frame = self.cap.read()
        if not ret:
            raise RuntimeError("Не удалось получить кадр с камеры")
        
        source_img = apply_color_transfer(self.args.source, first_frame)
        self.source_latent = create_source_latent(
            source_img, 
            self.args.face_attribute_direction, 
            self.args.face_attribute_steps
        )
        
        if self.source_latent is None:
            raise RuntimeError("Не удалось создать латентное представление исходного лица")
        
        # Инициализировать асинхронный процессор
        self.frame_processor = AsyncFrameProcessor(
            self.model, 
            self.source_latent, 
            max_queue_size=3
        )
        self.frame_processor.start()
        
        compile_status = "✅ Включен" if self.enable_torch_compile else "❌ Выключен"
        #print(f"[OptimizedRealtime] Инициализация завершена")
        #print(f"  - torch.compile(): {compile_status}")

    def run(self):
        """Запустить обработку в реальном времени."""
        #print("[OptimizedRealtime] Запуск обработки...")
        
        cv2.namedWindow("Optimized Live Face Swap", cv2.WINDOW_NORMAL)
        
        # Инициализировать виртуальную камеру если требуется
        cam_context = nullcontext()
        if self.args.obs:
            cam_context = pyvirtualcam.Camera(width=960, height=540, fps=20, fmt=PixelFormat.BGR)
        
        try:
            with cam_context as cam:
                self.virtual_cam = cam
                self._main_processing_loop()
        
        except KeyboardInterrupt:
            print("\n[OptimizedRealtime] Остановлено пользователем")
        
        finally:
            self._cleanup()
    
    def _main_processing_loop(self):
        """Основной цикл обработки."""
        last_result_frame = None
        
        while True:
            loop_start = time.time()
            
            # Получить кадр с камеры
            if self.cap is not None:
                ret, frame = self.cap.read()
                if not ret:
                    print("[WARNING] Не удалось получить кадр с камеры")
                    continue
            else:
                break
            
            self.frame_count += 1
            
            # Получить настройки качества
            quality_settings = self.quality_controller.get_current_settings()
            
            # Отправить кадр на асинхронную обработку
            submitted = False
            if self.frame_processor is not None:
                submitted = self.frame_processor.submit_frame(frame, self.frame_count, quality_settings)
            
            # Получить результат обработки (если доступен)
            result = None
            if self.frame_processor is not None:
                result = self.frame_processor.get_result()
            if result is not None:
                processed_frame, frame_id, process_time = result
                last_result_frame = processed_frame
            
            # Показать кадр (обработанный или исходный)
            display_frame = last_result_frame if last_result_frame is not None else frame
            
            # Добавить информацию о производительности
            if self.args.fps_delay:
                display_frame = self._add_performance_overlay(display_frame)
            
            # Показать кадр
            cv2.imshow("Optimized Live Face Swap", display_frame)
            
            # Отправить в виртуальную камеру
            if self.virtual_cam and last_result_frame is not None:
                resized_frame = cv2.resize(last_result_frame, (960, 540))
                self.virtual_cam.send(resized_frame)
            
            # Обновить метрики FPS
            loop_time = time.time() - loop_start
            fps = 1.0 / loop_time if loop_time > 0 else 0
            self.fps_measurements.append(fps)
            self.quality_controller.update_fps(fps)
            
            # Показать статистику каждые 5 секунд
            if time.time() - self.last_stats_time >= 5.0:
                self._print_performance_stats()
                self.last_stats_time = time.time()
            
            # Применить задержку если указана
            if self.args.delay > 0:
                time.sleep(self.args.delay / 1000.0)
            
            # Проверить выход
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' или Escape
                break
    
    def _add_performance_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Добавить оверлей с информацией о производительности."""
        overlay_frame = frame.copy()
        
        # Вычислить метрики
        avg_fps = sum(self.fps_measurements) / len(self.fps_measurements) if self.fps_measurements else 0
        quality_settings = self.quality_controller.get_current_settings()
        processor_stats = self.frame_processor.get_performance_stats() if self.frame_processor else {}
        gpu_info = get_gpu_memory_info()
        
        # Информация для отображения
        info_lines = [
            f"FPS: {avg_fps:.1f}",
            f"Quality: {quality_settings['quality_level']}",
            f"Resolution: {quality_settings['resolution']}px",
            f"Blending: {'ON' if quality_settings['blending'] else 'OFF'}",
            f"Dropped: {processor_stats['frames_dropped']} ({processor_stats['drop_rate']:.1f}%)",
            f"Queue: {processor_stats['input_queue_size']}/{processor_stats['output_queue_size']}",
        ]
        
        if gpu_info["total"] > 0:
            info_lines.append(f"GPU: {gpu_info['used']:.1f}/{gpu_info['total']:.1f}GB")
        
        # Отрисовка текста
        y_offset = 30
        for line in info_lines:
            cv2.putText(overlay_frame, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        return overlay_frame
    
    def _print_performance_stats(self):
        """Вывести статистику производительности."""
        avg_fps = sum(self.fps_measurements) / len(self.fps_measurements) if self.fps_measurements else 0
        quality_settings = self.quality_controller.get_current_settings()
        processor_stats = self.frame_processor.get_performance_stats() if self.frame_processor else {}
        
        #print(f"\n[Статистика] FPS: {avg_fps:.1f}, Качество: {quality_settings['quality_level']}")
        #print(f"  - Обработано кадров: {processor_stats['frames_processed']}")
        #print(f"  - Отброшено: {processor_stats['frames_dropped']} ({processor_stats['drop_rate']:.1f}%)")
        #print(f"  - Среднее время обработки: {processor_stats['avg_processing_time']*1000:.1f}ms")
    
    def _cleanup(self):
        """Очистка ресурсов."""
        #print("[OptimizedRealtime] Очистка ресурсов...")
        
        if self.frame_processor:
            self.frame_processor.stop()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Финальная статистика
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time if total_time > 0 else 0
        
        #print(f"\n[Итого] Обработано {self.frame_count} кадров за {total_time:.1f}с (Средняя FPS: {avg_fps:.1f})")


def main_optimized(parsed_args=None):
    """Основная функция оптимизированной обработки в реальном времени."""
    args = parsed_args or parse_arguments()
    
    processor = OptimizedRealtimeProcessor(args)
    
    try:
        processor.initialize()
        processor.run()
    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        raise


def cli_optimized(argv: Optional[Sequence[str]] = None):
    """CLI точка входа для оптимизированной обработки в реальном времени."""
    return main_optimized(parse_arguments(argv))


if __name__ == "__main__":
    cli_optimized() 
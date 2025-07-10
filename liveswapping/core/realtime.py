# -*- coding: utf-8 -*-
"""Модуль `liveswapping.realtime` – реал-тайм свап лиц с веб-камеры.

Основан на legacy-скрипте `swap_live_video.py`, перемещён в пакет `liveswapping`.
"""

from __future__ import annotations

import argparse
import cv2  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
# Отключаем verbose логи ONNX Runtime
import os
import warnings
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['ONNX_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")
import onnxruntime as ort
ort.set_default_logger_severity(3)  # ERROR level only

from liveswapping.core import image_utils as Image  # type: ignore  # noqa
from insightface.app import FaceAnalysis  # type: ignore
from liveswapping.core import face_align  # type: ignore
from liveswapping.ai_models.style_transfer_model_128 import StyleTransferModel  # type: ignore
import time
from collections import deque
import pyvirtualcam  # type: ignore
from pyvirtualcam import PixelFormat  # type: ignore
from contextlib import nullcontext
import traceback
from typing import Sequence, Optional
from liveswapping.ai_models.dfm_model import DFMModel  # type: ignore

# Adaptive CuPy acceleration
try:
    from liveswapping.utils.adaptive_cupy import (
        create_adaptive_processor,
        AdaptiveColorTransfer,
        AdaptiveBlending
    )
    ADAPTIVE_CUPY_AVAILABLE = True
except ImportError:
    ADAPTIVE_CUPY_AVAILABLE = False

__all__ = ["cli", "main", "parse_arguments"]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_arguments(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Live face swap via webcam")
    parser.add_argument("--source", required=True, help="Path to source face image")
    parser.add_argument("--modelPath", required=True, help="Path to the trained face swap model")
    parser.add_argument("--resolution", type=int, default=128, help="Resolution of the face crop")
    parser.add_argument("--face_attribute_direction", default=None, help="Path to face attribute direction.npy")
    parser.add_argument("--face_attribute_steps", type=float, default=0.0, help="Amount to move in attribute direction")
    parser.add_argument("--obs", action="store_true", help="Send frames to obs virtual cam")
    parser.add_argument("--mouth_mask", action="store_true", help="Retain target mouth")
    parser.add_argument("--delay", type=int, default=0, help="Delay time in milliseconds")
    parser.add_argument("--fps_delay", action="store_true", help="Show fps and delay time on top corner")
    parser.add_argument("--enhance_res", action="store_true", help="Increase webcam resolution to 1920x1080")
    return parser.parse_args(argv)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from pathlib import Path
import logging

# Подавляем логи insightface и onnxruntime
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
logging.getLogger("onnxruntime.capi").setLevel(logging.ERROR)

# insightface автоматически добавляет "models" к root пути, поэтому используем parent
models_root = Path(__file__).parent.parent.parent

# Временно подавляем stdout для инициализации
import sys
from io import StringIO

old_stdout = sys.stdout
sys.stdout = StringIO()

try:
    faceAnalysis = FaceAnalysis(name="buffalo_l", root=str(models_root), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    faceAnalysis.prepare(ctx_id=0, det_size=(512, 512))
finally:
    sys.stdout = old_stdout

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path):
    """Загружает модель с поддержкой torch-tensorrt оптимизации.
    
    Использует новый унифицированный загрузчик из liveswapping.models
    с автоматической torch-tensorrt оптимизацией для PyTorch моделей.
    """
    from liveswapping.ai_models.models import load_model as unified_load_model
    
    # Используем унифицированный загрузчик с torch-tensorrt оптимизацией
    return unified_load_model(str(model_path), use_tensorrt=True)


def swap_face(model, target_face, source_face_latent):
    device = get_device()
    target_tensor = torch.from_numpy(target_face).to(device)
    source_tensor = torch.from_numpy(source_face_latent).to(device)
    with torch.no_grad():
        swapped_tensor = model(target_tensor, source_tensor)
    swapped_face = Image.postprocess_face(swapped_tensor)
    return swapped_face


def create_source_latent(source_image, direction_path=None, steps=0.0):
    faces = faceAnalysis.get(source_image)
    if len(faces) == 0:
        print("No face detected in source image.")
        return None
    source_latent = Image.getLatent(faces[0])
    if direction_path:
        direction = np.load(direction_path)
        direction = direction / np.linalg.norm(direction)
        source_latent += direction * steps
    return source_latent


def apply_color_transfer(source_path, target):
    """Apply color transfer with adaptive CuPy acceleration."""
    if ADAPTIVE_CUPY_AVAILABLE:
        # Use adaptive processor for large frames
        global adaptive_color_transfer
        if 'adaptive_color_transfer' not in globals():
            processor = create_adaptive_processor(target.shape[0])
            adaptive_color_transfer = AdaptiveColorTransfer(processor)
        
        return adaptive_color_transfer.apply_color_transfer_adaptive(
            source_path, target, faceAnalysis
        )
    else:
        # Fallback to CPU implementation
        return apply_color_transfer_cpu(source_path, target)

def apply_color_transfer_cpu(source_path, target):
    """Original CPU-only color transfer."""
    source = cv2.imread(source_path)
    target_faces = faceAnalysis.get(target)
    x1, y1, x2, y2 = target_faces[0]["bbox"]
    target_crop = target[int(y1): int(y2), int(x1):int(x2)]
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target_crop, cv2.COLOR_BGR2LAB).astype("float32")
    source_mean, source_std = cv2.meanStdDev(source_lab)
    target_mean, target_std = cv2.meanStdDev(target_lab)
    source_mean = source_mean.reshape(1, 1, 3)
    source_std = source_std.reshape(1, 1, 3)
    target_mean = target_mean.reshape(1, 1, 3)
    target_std = target_std.reshape(1, 1, 3)
    result = (source_lab - source_mean) * (target_std / source_std) + target_mean
    return cv2.cvtColor(np.clip(result, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(parsed_args=None):
    args = parsed_args or parse_arguments()
    model = load_model(args.modelPath)
    cap = cv2.VideoCapture(0)
    if args.enhance_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    
    # Initialize adaptive processing globals
    global adaptive_processor_main
    adaptive_processor_main = None

    fps_update_interval = 0.5    
    frame_count = 0
    prev_time = time.time()
    cv2.namedWindow("Live Face Swap", cv2.WINDOW_NORMAL)

    create_latent_flag = True
    buffer: deque[tuple[np.ndarray, float]] = deque()

    with pyvirtualcam.Camera(width=960, height=540, fps=20, fmt=PixelFormat.BGR) if args.obs else nullcontext() as cam:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if create_latent_flag:
                try:
                    source_img = apply_color_transfer(args.source, frame)
                    source_latent = create_source_latent(source_img, args.face_attribute_direction, args.face_attribute_steps)
                    if source_latent is None:
                        raise RuntimeError("Face not found in source image")
                    create_latent_flag = False
                except Exception:
                    cv2.imshow("Live Face Swap", frame)

            # Initialize adaptive processor once
            if adaptive_processor_main is None and ADAPTIVE_CUPY_AVAILABLE:
                adaptive_processor_main = create_adaptive_processor(frame.shape[0])

            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= fps_update_interval:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time
                print(f"FPS: {fps:.2f}")

            faces = faceAnalysis.get(frame)
            if len(faces) == 0:
                cv2.imshow("Live Face Swap", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            target_face = faces[0]
            aligned_face, M = face_align.norm_crop2(frame, target_face.kps, args.resolution)
            face_blob = Image.getBlob(aligned_face, (args.resolution, args.resolution))

            try:
                if hasattr(model, "convert"):
                    aligned_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                    aligned_tensor = torch.from_numpy(aligned_rgb).permute(2, 0, 1).to(get_device())
                    out_face, _, _ = model.convert(aligned_tensor)
                    swapped_face = cv2.cvtColor(out_face.cpu().numpy(), cv2.COLOR_RGB2BGR)
                elif hasattr(model, "swap_face"):
                    # InSwapper модель
                    swapped_face = model.swap_face(aligned_face, source_latent)
                else:
                    swapped_face = swap_face(model, face_blob, source_latent)
                
                # Adaptive blending based on frame size
                if ADAPTIVE_CUPY_AVAILABLE:
                    if 'adaptive_blender' not in globals():
                        processor = create_adaptive_processor(frame.shape[0])
                        adaptive_blender = AdaptiveBlending(processor)
                    final_frame = adaptive_blender.blend_swapped_image_adaptive(swapped_face, frame, M)
                else:
                    final_frame = Image.blend_swapped_image_gpu(swapped_face, frame, M)
                if args.mouth_mask:
                    face_mask = Image.create_face_mask(target_face, frame)  # type: ignore
                    _, mouth_cutout, mouth_box, lower_lip_polygon = Image.create_lower_mouth_mask(target_face, frame)  # type: ignore
                    final_frame = Image.apply_mouth_area(final_frame, mouth_cutout, mouth_box, face_mask, lower_lip_polygon)  # type: ignore

                if args.fps_delay:
                    cv2.putText(final_frame, f"Delay: {args.delay} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                buffer_end = time.time()
                buffer.append((final_frame, buffer_end))

                if (buffer_end - buffer[0][1]) * 1000 >= args.delay:
                    if cam:
                        display_frame = cv2.resize(buffer[0][0], (960, 540))
                        cam.send(display_frame)
                        cam.sleep_until_next_frame()
                        buffer.popleft()
                    else:
                        cv2.imshow("Live Face Swap", cv2.resize(buffer[0][0], None, fx=2.0, fy=2.0))
                        buffer.popleft()
            except Exception as e:
                print(f"Swap error: {e}")
                traceback.print_exc()
                cv2.imshow("Live Face Swap", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in (ord("+"), ord("=")):
                args.delay += 50
                print(f"Delay increased to {args.delay} ms")
            elif key == ord("-"):
                args.delay = max(0, args.delay - 50)
                while (buffer_end - buffer[0][1]) * 1000 > args.delay:
                    buffer.popleft()
                print(f"Delay decreased to {args.delay} ms")

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

def cli(argv: Optional[Sequence[str]] = None):
    args = parse_arguments(argv)
    main(args)


if __name__ == "__main__":
    cli() 
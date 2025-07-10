# -*- coding: utf-8 -*-
"""Модуль `liveswapping.video` – CLI и API для офлайн-свапа лиц в видео.

Код основан на legacy-скрипте `swap_video.py`, но теперь инкапсулирован в пакет
`liveswapping`. Логика оставлена без изменений, за исключением небольших
правок для удобства импорта и возможности вызывать из кода (через `cli()` /
`main()`).
"""

from __future__ import annotations

import argparse
import os
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
# import line_profiler  # опционально
import pyvirtualcam  # type: ignore
from pyvirtualcam import PixelFormat  # type: ignore
from contextlib import nullcontext
from moviepy import AudioFileClip, VideoFileClip  # type: ignore
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip  # type: ignore
from tqdm import tqdm  # type: ignore
import shutil
from gfpgan import GFPGANer  # type: ignore
import glob
from basicsr.utils import imwrite  # type: ignore
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
# Helpers & argument parsing
# ---------------------------------------------------------------------------

def parse_arguments(argv: Optional[Sequence[str]] = None):
    """Создаёт `argparse.ArgumentParser` и разбирает `argv`.

    Если `argv is None`, используется `sys.argv[1:]` по умолчанию.
    """
    parser = argparse.ArgumentParser(description="Live face swap via webcam")
    parser.add_argument("--source", required=True, help="Path to source face image")
    parser.add_argument("--target_video", required=True, help="Path to target video")
    parser.add_argument("--modelPath", required=True, help="Path to the trained face swap model")
    parser.add_argument("--resolution", type=int, default=128, help="Resolution of the face crop")
    parser.add_argument("--face_attribute_direction", default=None, help="Path to face attribute direction.npy")
    parser.add_argument("--face_attribute_steps", type=float, default=0.0, help="Amount to move in attribute direction")
    parser.add_argument("--mouth_mask", action="store_true", help="Retain target mouth")
    parser.add_argument("-s", "--upscale", type=int, default=2, help="The final upsampling scale of the image. Default: 2")
    parser.add_argument("--bg_upsampler", type=str, default="realesrgan", help="background upsampler. Default: realesrgan")
    parser.add_argument("--bg_tile", type=int, default=400, help="Tile size for background sampler, 0 for no tile during testing. Default: 400")
    parser.add_argument("--suffix", type=str, default=None, help="Suffix of the restored faces")
    parser.add_argument("--only_center_face", action="store_true", help="Only restore the center face")
    parser.add_argument("--aligned", action="store_true", help="Input are aligned faces")
    parser.add_argument("--ext", type=str, default="auto", help="Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto")
    parser.add_argument("-w", "--weight", type=float, default=0.5, help="Adjustable weights.")
    parser.add_argument("-std", "--std", type=int, default=1, help="standard deviation for noise")
    parser.add_argument("-blur", "--blur", type=int, default=1, help="blur")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Core helpers (взяты без изменений из legacy-скрипта)
# ---------------------------------------------------------------------------

# Setup face detector (глобальный, чтобы инициализировать лишь раз)
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
        global adaptive_color_transfer_video
        if 'adaptive_color_transfer_video' not in globals():
            processor = create_adaptive_processor(target.shape[0])
            adaptive_color_transfer_video = AdaptiveColorTransfer(processor)
        
        return adaptive_color_transfer_video.apply_color_transfer_adaptive(
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
    target = target[int(y1): int(y2), int(x1):int(x2)]
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    source_mean, source_std = cv2.meanStdDev(source)
    target_mean, target_std = cv2.meanStdDev(target)

    # Reshape mean and std to be broadcastable
    source_mean = source_mean.reshape(1, 1, 3)
    source_std = source_std.reshape(1, 1, 3)
    target_mean = target_mean.reshape(1, 1, 3)
    target_std = target_std.reshape(1, 1, 3)

    # Perform the color transfer
    source = (source - source_mean) * (target_std / source_std) + target_mean

    return cv2.cvtColor(np.clip(source, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)

# ---------------------------------------------------------------------------
# Main processing routine (практически без изменений)
# ---------------------------------------------------------------------------

def main(parsed_args=None):
    """Выполняет свапинг лиц по переданным аргументам.

    При вызове из кода можно передать уже распарсенный `argparse.Namespace`.
    Если `parsed_args is None`, аргументы парсятся из `sys.argv`.
    """
    args = parsed_args or parse_arguments()

    # Настройка апскейлера / реставратора
    if args.bg_upsampler == "realesrgan":
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn(
                "The unoptimized RealESRGAN is slow on CPU. We do not use it. "
                "If you really want to use it, please modify the corresponding codes.")
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
            from realesrgan import RealESRGANer  # type: ignore
            _model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                model=_model,
                tile=args.bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True,  # need to set False in CPU mode
            )
    else:
        bg_upsampler = None

    # Используем оптимизированный GFPGAN с torch-tensorrt и автозагрузкой
    from liveswapping.utils.upscalers import create_optimized_gfpgan
    
    restorer = create_optimized_gfpgan(
        model_path=None,  # Автоматическая загрузка GFPGANv1.3.pth
        use_tensorrt=True,  # Включаем torch-tensorrt оптимизацию
        bg_upsampler=bg_upsampler
    )

    video_path = args.target_video
    model = load_model(args.modelPath)

    # Проверяем наличие аудио
    video_forcheck = VideoFileClip(video_path)
    no_audio = video_forcheck.audio is None
    del video_forcheck
    video_audio_clip = None
    if not no_audio:
        video_audio_clip = AudioFileClip(video_path)

    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    temp_results_dir = "./temp_results"
    os.makedirs(temp_results_dir, exist_ok=True)

    create_latent_flag = True
    
    # Initialize adaptive processing globals
    global adaptive_blending_video
    adaptive_blending_video = None

    for frame_index in tqdm(range(frame_count)):
        ret, frame = video.read()
        if not ret:
            break

        if create_latent_flag:
            source = apply_color_transfer(source_path=args.source, target=frame)
            source_latent = create_source_latent(source, args.face_attribute_direction, args.face_attribute_steps)
            if source_latent is None:
                raise RuntimeError("Face not found in source image")
            create_latent_flag = False

        faces = faceAnalysis.get(frame)
        if len(faces) == 0:
            continue  # пропускаем кадр без лиц

        target_face = faces[0]
        aligned_face, M = face_align.norm_crop2(frame, target_face.kps, args.resolution)
        face_blob = Image.getBlob(aligned_face, (args.resolution, args.resolution))

        try:
            if hasattr(model, "convert"):
                # DFM модель
                aligned_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                aligned_tensor = torch.from_numpy(aligned_rgb).permute(2, 0, 1).to(get_device())
                out_face, _, _ = model.convert(aligned_tensor)
                swapped_face = cv2.cvtColor(out_face.cpu().numpy(), cv2.COLOR_RGB2BGR)
            elif hasattr(model, "swap_face"):
                # InSwapper модель
                swapped_face = model.swap_face(aligned_face, source_latent)
            else:
                swapped_face = swap_face(model, face_blob, source_latent)
            
            # Use adaptive blending with CuPy acceleration
            if ADAPTIVE_CUPY_AVAILABLE:
                if adaptive_blending_video is None:
                    processor = create_adaptive_processor(frame.shape[0])
                    adaptive_blending_video = AdaptiveBlending(processor)
                final_frame = adaptive_blending_video.blend_swapped_image_adaptive(swapped_face, frame, M)
            else:
                final_frame = Image.blend_swapped_image_gpu(swapped_face, frame, M)
            if args.mouth_mask:
                face_mask = Image.create_face_mask(target_face, frame)  # type: ignore
                _, mouth_cutout, mouth_box, lower_lip_polygon = Image.create_lower_mouth_mask(target_face, frame)  # type: ignore
                final_frame = Image.apply_mouth_area(final_frame, mouth_cutout, mouth_box, face_mask, lower_lip_polygon)  # type: ignore
            cv2.imwrite(os.path.join(temp_results_dir, f"frame_{frame_index:07d}.jpg"), final_frame)
        except Exception as e:
            print(f"Swap error on frame {frame_index}: {e}")

    # --- Пост-обработка и сбор видео -------------------------------------

    img_list = sorted(glob.glob(os.path.join(temp_results_dir, "*")))
    temp_restored_dir = "./temp_results2/restored_imgs"
    os.makedirs(temp_restored_dir, exist_ok=True)

    for img_path in img_list:
        img_name = os.path.basename(img_path)
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        _, _, restored_img = restorer.enhance(
            input_img,
            has_aligned=args.aligned,
            only_center_face=args.only_center_face,
            paste_back=True,
            weight=args.weight,
        )

        if restored_img is None:
            continue

        extension = ext[1:] if args.ext == "auto" else args.ext
        if args.suffix is not None:
            save_restore_path = os.path.join(temp_restored_dir, f"{basename}_{args.suffix}.{extension}")
        else:
            save_restore_path = os.path.join(temp_restored_dir, f"{basename}.{extension}")

        # Добавляем текстуру/шум и блюр
        noise = np.random.normal(0, args.std, restored_img.shape).astype(np.float32)
        textured_img = np.clip(restored_img + noise, 0, 255).astype(np.uint8)
        textured_blurred_img = cv2.GaussianBlur(textured_img, (args.blur, args.blur), 0)
        imwrite(textured_blurred_img, save_restore_path)

    image_filenames = sorted(glob.glob(os.path.join(temp_restored_dir, "*.jpg")))
    clips = ImageSequenceClip(image_filenames, fps=fps)
    if not no_audio and video_audio_clip is not None:
        clips = clips.with_audio(video_audio_clip)
    clips.write_videofile("output.mp4", codec="libx264", audio_codec="aac")

    # cleanup
    shutil.rmtree("./temp_results2", ignore_errors=True)
    shutil.rmtree(temp_results_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Public CLI entry -----------------------------------------------------------
# ---------------------------------------------------------------------------

def cli(argv: Optional[Sequence[str]] = None):
    """CLI-обёртка: парсит `argv` и запускает `main`."""
    args = parse_arguments(argv)
    main(args)


if __name__ == "__main__":
    cli() 
# -*- coding: utf-8 -*-
"""Модуль для офлайн-замены лиц в изображениях.

Основан на модуле video.py, но адаптирован для работы с изображениями.
"""

from __future__ import annotations

import argparse
from typing import Sequence, Optional
from pathlib import Path

__all__ = ["cli", "main", "parse_arguments"]

# ---------------------------------------------------------------------------
# Helpers & argument parsing
# ---------------------------------------------------------------------------

def parse_arguments(argv: Optional[Sequence[str]] = None):
    """Создаёт `argparse.ArgumentParser` и разбирает `argv` для обработки изображений."""
    parser = argparse.ArgumentParser(description="Face swap for images")
    parser.add_argument("--source", required=True, help="Path to source face image")
    parser.add_argument("--target", required=True, help="Path to target image")
    parser.add_argument("--modelPath", required=True, help="Path to the trained face swap model")
    parser.add_argument("--resolution", type=int, default=128, help="Resolution of the face crop")
    parser.add_argument("--face_attribute_direction", default=None, help="Path to face attribute direction.npy")
    parser.add_argument("--face_attribute_steps", type=float, default=0.0, help="Amount to move in attribute direction")
    parser.add_argument("--output", default="output.jpg", help="Output image path")
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
    parser.add_argument("--model_provider", type=str, default="cpu", help="Model provider (cpu, cuda)")
    parser.add_argument("--upscaler_provider", type=str, default="cpu", help="Upscaler provider (cpu, cuda)")

    return parser.parse_args(argv)

# ---------------------------------------------------------------------------
# Main processing routine
# ---------------------------------------------------------------------------

def main(parsed_args=None):
    """Выполняет замену лиц в изображении.""" 
    args = parsed_args or parse_arguments()
    
    print(f"[IMAGE] Processing image: {args.target}")
    print(f"[SOURCE] Source: {args.source}")
    print(f"[MODEL] Model: {args.modelPath}")
    print("[INFO] Processing single image...")
    
    # Импорты для обработки изображений
    import cv2
    import numpy as np
    from pathlib import Path
    from liveswapping.ai_models.models import load_model
    from insightface.app import FaceAnalysis
    from liveswapping.core import face_align
    from liveswapping.core import image_utils as Image
    
    # Инициализация face detector
    models_root = Path(__file__).parent.parent.parent
    faceAnalysis = FaceAnalysis(name="buffalo_l", root=str(models_root), 
                               providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    faceAnalysis.prepare(ctx_id=0, det_size=(512, 512))
    
    # Загрузка модели
    print(f"[MODEL] Loading model: {args.modelPath}")
    model = load_model(args.modelPath, provider_type=args.model_provider)
    
    # Загрузка изображений
    print("[INPUT] Loading images...")
    source_image = cv2.imread(args.source)
    target_image = cv2.imread(args.target)
    
    if source_image is None:
        raise ValueError(f"Could not load source image: {args.source}")
    if target_image is None:
        raise ValueError(f"Could not load target image: {args.target}")
    
    print(f"[INPUT] Source image shape: {source_image.shape}")
    print(f"[INPUT] Target image shape: {target_image.shape}")
    
    # Поиск лиц в source изображении
    print("[FACE] Detecting faces in source image...")
    source_faces = faceAnalysis.get(source_image)
    if len(source_faces) == 0:
        raise ValueError("No faces found in source image")
    source_face = source_faces[0]
    print(f"[FACE] Found {len(source_faces)} face(s) in source image")
    
    # Поиск лиц в target изображении  
    print("[FACE] Detecting faces in target image...")
    target_faces = faceAnalysis.get(target_image)
    if len(target_faces) == 0:
        raise ValueError("No faces found in target image")
    target_face = target_faces[0]
    print(f"[FACE] Found {len(target_faces)} face(s) in target image")
    
    # Выполнение face swap
    print("[SWAP] Performing face swap...")
    
    # Выравнивание source лица
    source_face_landmark = source_face.kps
    source_aligned = face_align.norm_crop(source_image, source_face_landmark, args.resolution)
    
    # Выравнивание target лица
    target_face_landmark = target_face.kps  
    target_aligned = face_align.norm_crop(target_image, target_face_landmark, args.resolution)
    M = face_align.estimate_norm(target_face_landmark, args.resolution)
    
    # Применение модели
    if hasattr(model, 'get'):  # ONNX model
        source_embedding = model.get(source_aligned, target_aligned)
        swapped_face = source_embedding
    else:  # PyTorch model (StyleTransfer)
        import torch
        device = torch.device(args.model_provider if args.model_provider != 'cpu' else 'cpu')
        
        # Получаем эмбеддинг source лица через InsightFace
        source_embedding = source_face.normed_embedding.reshape(1, -1)
        source_embedding_tensor = torch.from_numpy(source_embedding).float().to(device)
        
        # Подготавливаем target изображение
        target_tensor = torch.from_numpy(target_aligned).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        
        with torch.no_grad():
            swapped_tensor = model(target_tensor, source_embedding_tensor)
            swapped_face = (swapped_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    
    # Смешивание результата с оригинальным изображением
    print("[BLEND] Blending result...")
    result_image = Image.blend_swapped_image(swapped_face, target_image, M)
    
    # Upscaling если требуется
    if args.upscale > 1:
        print(f"[UPSCALE] Applying {args.upscale}x upscaling...")
        try:
            from liveswapping.utils.upscalers import create_optimized_gfpgan
            
            # Настройка background upsampler если нужен
            bg_upsampler = None
            if args.bg_upsampler == "realesrgan":
                try:
                    from basicsr.archs.rrdbnet_arch import RRDBNet
                    from realesrgan import RealESRGANer
                    _model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                    bg_upsampler = RealESRGANer(
                        scale=2,
                        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                        model=_model,
                        tile=args.bg_tile,
                        tile_pad=10,
                        pre_pad=0,
                        half=True
                    )
                except Exception as e:
                    print(f"[WARNING] Could not initialize RealESRGAN: {e}")
            
            # Создание GFPGAN upscaler
            restorer = create_optimized_gfpgan(
                model_path=None,
                use_tensorrt=(args.upscaler_provider == 'cuda'),
                bg_upsampler=bg_upsampler
            )
            
            # Применение upscaling
            _, result_image, _ = restorer.upscale(result_image)
            print("[SUCCESS] Upscaling completed")
            
        except Exception as e:
            print(f"[WARNING] Upscaling failed: {e}")
            print("[INFO] Saving result without upscaling")
    
    # Сохранение результата
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    success = cv2.imwrite(str(output_path), result_image)
    if success:
        print(f"[SAVED] Result saved as: {output_path}")
        print(f"[SUCCESS] Image processing completed!")
    else:
        print(f"[ERROR] Failed to save result to: {output_path}")
        return 1
    
    return 0

# ---------------------------------------------------------------------------
# Public CLI entry
# ---------------------------------------------------------------------------

def cli(argv: Optional[Sequence[str]] = None):
    """CLI-обёртка: парсит `argv` и запускает `main`."""
    args = parse_arguments(argv)
    main(args)

if __name__ == "__main__":
    cli() 
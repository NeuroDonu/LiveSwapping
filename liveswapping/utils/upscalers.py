# -*- coding: utf-8 -*-
"""Оптимизированные апскейлеры с поддержкой torch-tensorrt для LiveSwapping."""

from __future__ import annotations

import torch  # type: ignore
import numpy as np  # type: ignore
from typing import Optional, Union

__all__ = ["GFPGANUpscaler", "RealESRGANUpscaler", "DEFAULT_UPSCALER", "create_optimized_gfpgan", "ensure_gfpgan_model"]


def _create_tensorrt_model(model: torch.nn.Module, input_shape: tuple = (1, 3, 512, 512)) -> torch.nn.Module:
    """Оптимизирует PyTorch модель с помощью torch-tensorrt для GFPGAN/RealESRGAN.
    
    Parameters
    ----------
    model : torch.nn.Module
        Исходная PyTorch модель
    input_shape : tuple
        Форма входного тензора (B, C, H, W)
        
    Returns
    -------
    torch.nn.Module
        Оптимизированная модель или исходная при ошибке
    """
    try:
        import torch_tensorrt  # type: ignore
        
        # Создаем пример входных данных для GFPGAN/RealESRGAN
        input_spec = torch_tensorrt.Input(
            min_shape=input_shape,
            opt_shape=input_shape,
            max_shape=input_shape,
            dtype=torch.float32
        )
        
        # Компилируем модель с torch-tensorrt
        compiled_model = torch_tensorrt.compile(
            model,
            inputs=[input_spec],
            enabled_precisions={torch.float32},  # FP32 для стабильности
            ir="torch_compile",  # Используем torch.compile backend
            min_block_size=5,  # Больший блок для апскейлеров
            require_full_compilation=False,  # Разрешаем гибридное выполнение
        )
        
        print(f"[SUCCESS] GFPGAN/RealESRGAN model optimized with torch-tensorrt")
        return compiled_model
        
    except ImportError:
        print("[WARNING] torch-tensorrt not installed, using default model")
        return model
    except Exception as e:
        print(f"[WARNING] Error optimizing with torch-tensorrt: {e}")
        print("Используется стандартная модель")
        return model


def ensure_gfpgan_model() -> str:
    """Убеждается что модель GFPGAN загружена и возвращает путь к ней.
    
    Returns
    -------
    str
        Путь к загруженной модели GFPGAN
    """
    try:
        from liveswapping.ai_models.download_models import ensure_model
        model_path = ensure_model("gfpgan")
        return str(model_path)
    except Exception as e:
        print(f"⚠️ Не удалось загрузить GFPGAN модель: {e}")
        # Fallback to default path
        return "./models/GFPGANv1.3.pth"


def create_optimized_gfpgan(model_path: str | None = None, use_tensorrt: bool = True, 
                           bg_upsampler: Optional[object] = None) -> "GFPGANUpscaler":
    """Создает оптимизированный GFPGAN с torch-tensorrt поддержкой.
    
    Parameters
    ----------
    model_path : str, optional
        Путь к модели GFPGAN. Если None, автоматически загружает модель
    use_tensorrt : bool
        Использовать torch-tensorrt оптимизацию
    bg_upsampler : optional
        Background upsampler (RealESRGAN)
        
    Returns
    -------
    GFPGANUpscaler
        Оптимизированный апскейлер
    """
    # Автоматически загружаем модель если путь не указан
    if model_path is None:
        model_path = ensure_gfpgan_model()
    
    return GFPGANUpscaler(
        model_path=model_path, 
        use_tensorrt=use_tensorrt,
        bg_upsampler=bg_upsampler
    )


class GFPGANUpscaler:
    """Оптимизированный GFPGAN апскейлер с torch-tensorrt поддержкой."""
    
    def __init__(self, model_path: str | None = None, use_tensorrt: bool = True, 
                 bg_upsampler: Optional[object] = None):
        """Инициализация GFPGAN с возможной torch-tensorrt оптимизацией.
        
        Parameters
        ----------
        model_path : str, optional
            Путь к модели GFPGAN
        use_tensorrt : bool
            Использовать torch-tensorrt оптимизацию
        bg_upsampler : optional
            Background upsampler
        """
        try:
            from gfpgan import GFPGANer  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Для использования GFPGANUpscaler необходимо установить пакет gfpgan"
            ) from exc
            
        self.use_tensorrt = use_tensorrt and torch.cuda.is_available()
        
        # Автоматически загружаем модель если путь не указан
        if model_path is None:
            model_path = ensure_gfpgan_model()
        
        # Создаем стандартный GFPGANer
        self._restorer = GFPGANer(
            model_path=model_path,
            upscale=2,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=bg_upsampler,
        )
        
        # Оптимизируем модель с torch-tensorrt если доступно
        if self.use_tensorrt:
            self._optimize_model()

    def _optimize_model(self):
        """Оптимизирует внутреннюю модель GFPGAN с torch-tensorrt."""
        try:
            # Получаем доступ к внутренней PyTorch модели GFPGAN
            if hasattr(self._restorer, 'gfpgan') and hasattr(self._restorer.gfpgan, 'net_g'):
                original_model = self._restorer.gfpgan.net_g
                
                # Переводим в eval режим
                original_model.eval()
                
                # Оптимизируем с torch-tensorrt
                optimized_model = _create_tensorrt_model(
                    original_model, 
                    input_shape=(1, 3, 512, 512)  # Стандартный размер для GFPGAN
                )
                
                # Заменяем модель на оптимизированную
                self._restorer.gfpgan.net_g = optimized_model
                
        except Exception as e:
            print(f"⚠️ Не удалось оптимизировать GFPGAN: {e}")
            self.use_tensorrt = False

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """Выполняет апскейлинг изображения.
        
        Parameters
        ----------
        image : np.ndarray
            Входное изображение
            
        Returns
        -------
        np.ndarray
            Улучшенное изображение
        """
        _, res, *rest = self._restorer.enhance(
            image, 
            has_aligned=False, 
            only_center_face=False
        )
        return res
        
    def enhance(self, image: np.ndarray, **kwargs) -> tuple:
        """Расширенная функция улучшения с дополнительными параметрами.
        
        Parameters
        ----------
        image : np.ndarray
            Входное изображение
        **kwargs
            Дополнительные параметры для enhance
            
        Returns
        -------
        tuple
            Результат enhance (cropped_faces, restored_img, restored_faces)
        """
        return self._restorer.enhance(image, **kwargs)


class RealESRGANUpscaler:
    """Оптимизированный RealESRGAN апскейлер с torch-tensorrt поддержкой."""
    
    def __init__(self, model_path: str | None = None, use_tensorrt: bool = True, 
                 scale: int = 2, tile: int = 400):
        """Инициализация RealESRGAN с torch-tensorrt оптимизацией.
        
        Parameters
        ----------
        model_path : str, optional
            Путь к модели RealESRGAN
        use_tensorrt : bool
            Использовать torch-tensorrt оптимизацию
        scale : int
            Масштаб увеличения
        tile : int
            Размер тайла для обработки
        """
        self.use_tensorrt = use_tensorrt and torch.cuda.is_available()
        
        if not torch.cuda.is_available():
            import warnings
            warnings.warn(
                "RealESRGAN медленно работает на CPU. Рекомендуется использовать GPU."
            )
            self._upsampler = None
            return
            
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
            from realesrgan import RealESRGANer  # type: ignore
            
            # Создаем модель
            model = RRDBNet(
                num_in_ch=3, 
                num_out_ch=3, 
                num_feat=64, 
                num_block=23, 
                num_grow_ch=32, 
                scale=scale
            )
            
            # Оптимизируем с torch-tensorrt если нужно
            if self.use_tensorrt:
                model = _create_tensorrt_model(
                    model, 
                    input_shape=(1, 3, tile, tile)
                )
            
            # Создаем RealESRGANer
            self._upsampler = RealESRGANer(
                scale=scale,
                model_path=model_path or "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                model=model,
                tile=tile,
                tile_pad=10,
                pre_pad=0,
                half=True,  # FP16 для экономии памяти
            )
            
        except ImportError as exc:
            raise RuntimeError(
                "Для использования RealESRGAN необходимо установить пакеты basicsr и realesrgan"
            ) from exc

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """Выполняет апскейлинг изображения.
        
        Parameters
        ----------
        image : np.ndarray
            Входное изображение
            
        Returns
        -------
        np.ndarray
            Увеличенное изображение
        """
        if self._upsampler is None:
            return image
            
        output, _ = self._upsampler.enhance(image, outscale=2)
        return output


# Создаем глобальный экземпляр с оптимизацией
DEFAULT_UPSCALER = None

def get_default_upscaler(use_tensorrt: bool = True) -> GFPGANUpscaler:
    """Получает глобальный экземпляр оптимизированного апскейлера.
    
    Parameters
    ----------
    use_tensorrt : bool
        Использовать torch-tensorrt оптимизацию
        
    Returns
    -------
    GFPGANUpscaler
        Глобальный апскейлер
    """
    global DEFAULT_UPSCALER
    if DEFAULT_UPSCALER is None:
        DEFAULT_UPSCALER = GFPGANUpscaler(use_tensorrt=use_tensorrt)
    return DEFAULT_UPSCALER 
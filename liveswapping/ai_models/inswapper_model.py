# -*- coding: utf-8 -*-
"""Wrapper для inswapper128 модели InsightFace.

Модель inswapper128 - это ONNX модель от InsightFace для замены лиц,
которая работает с эмбеддингами от buffalo_l ArcFace модели.
"""

from __future__ import annotations

import cv2  # type: ignore
import numpy as np  # type: ignore
import onnxruntime as ort  # type: ignore
from typing import List, Tuple, Optional

__all__ = ["InSwapperModel"]


class InSwapperModel:
    """Wrapper для inswapper128 ONNX модели."""
    
    def __init__(self, model_path: str, providers: Optional[List] = None, provider_type: Optional[str] = None):
        """Инициализация модели inswapper128.
        
        Parameters
        ----------
        model_path : str
            Путь к .onnx файлу модели inswapper128
        providers : List, optional
            Список ONNX Runtime провайдеров
        provider_type : str, optional
            Тип провайдера: 'cuda', 'directml', 'openvino', 'cpu'
        """
        if providers is None:
            from liveswapping.ai_models.models import _create_providers
            providers = _create_providers(force_provider=provider_type)
            
        # Создаем сессию с оптимизированными настройками
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True
        
        # Дополнительные оптимизации для разных провайдеров
        if any("Dml" in str(p) for p in providers):
            # DirectML оптимизации
            session_options.enable_mem_reuse = True
        elif any("OpenVINO" in str(p) for p in providers):
            # OpenVINO оптимизации  
            session_options.inter_op_num_threads = 0  # Автоматическое определение
            session_options.intra_op_num_threads = 0
            
        self.session = ort.InferenceSession(
            model_path, 
            providers=providers,
            sess_options=session_options
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
    def get(self, aligned_face: np.ndarray, source_embedding: np.ndarray) -> np.ndarray:
        """Выполняет свап лица используя inswapper128.
        
        Parameters
        ----------
        aligned_face : np.ndarray
            Выровненное лицо (112x112, BGR)
        source_embedding : np.ndarray
            Эмбеддинг исходного лица от buffalo_l модели
            
        Returns
        -------
        np.ndarray
            Результат свапа лица (112x112, BGR)
        """
        # Подготовка входных данных
        face_input = aligned_face.astype(np.float32) / 127.5 - 1.0
        face_input = np.expand_dims(face_input.transpose(2, 0, 1), axis=0)
        
        embedding_input = source_embedding.reshape(1, -1)
        
        # Inference
        inputs = {
            "target": face_input,
            "source": embedding_input
        }
        
        result = self.session.run([self.output_name], inputs)[0]
        
        # Постобработка
        result = result[0].transpose(1, 2, 0)
        result = np.clip((result + 1.0) * 127.5, 0, 255).astype(np.uint8)
        
        return result
        
    def swap_face(self, aligned_face: np.ndarray, source_embedding: np.ndarray) -> np.ndarray:
        """Alias для get() для совместимости."""
        return self.get(aligned_face, source_embedding) 
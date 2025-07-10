# -*- coding: utf-8 -*-
"""Модуль утилит для обработки изображений из старого Image.py."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Загружаем emap из правильного места
_emap_path = Path(__file__).parent.parent / "ai_models" / "emap.npy"
emap = np.load(_emap_path)

input_std = 255.0
input_mean = 0.0

def postprocess_face(face_tensor):
    """Постобработка тензора лица в изображение."""
    face_tensor = face_tensor.squeeze().cpu().detach()
    face_np = (face_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
    return face_np

def getBlob(aimg, input_size=(128, 128)):
    """Создает blob для DNN из изображения."""
    blob = cv2.dnn.blobFromImage(aimg, 1.0 / input_std, input_size,
                            (input_mean, input_mean, input_mean), swapRB=True)
    return blob

def getLatent(source_face):
    """Получает латентное представление лица."""
    latent = source_face.normed_embedding.reshape((1,-1))
    latent = np.dot(latent, emap)
    latent /= np.linalg.norm(latent)
    return latent

def blend_swapped_image(swapped_face, target_image, M):
    """Смешивает заменённое лицо с целевым изображением."""
    # get image size
    h, w = target_image.shape[:2]
    
    # create inverse affine transform
    M_inv = cv2.invertAffineTransform(M)
    
    # warp swapped face back to target space
    warped_face = cv2.warpAffine(
        swapped_face,
        M_inv,
        (w, h),
        borderValue=(0, 0, 0)
    )
    
    # create initial white mask
    img_white = np.full(
        (swapped_face.shape[0], swapped_face.shape[1]),
        255,
        dtype=np.float32
    )
    
    # warp white mask to target space
    img_mask = cv2.warpAffine(
        img_white,
        M_inv,
        (w, h),
        borderValue=(0,)
    )
    
    # threshold and refine mask
    img_mask[img_mask > 20] = 255
    
    # calculate mask size for kernel scaling
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    if len(mask_h_inds) > 0 and len(mask_w_inds) > 0:  # safety check
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h * mask_w))
        
        # erode mask
        k = max(mask_size // 10, 10)
        kernel = np.ones((k, k), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
        
        # blur mask
        k = max(mask_size // 20, 5)
        kernel_size = (k, k)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
    
    # normalize mask
    img_mask = img_mask / 255.0
    img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
    
    # blend images using mask
    result = img_mask * warped_face + (1 - img_mask) * target_image.astype(np.float32)
    result = result.astype(np.uint8)
    
    return result

def drawKeypoints(image, keypoints, colorBGR, keypointsRadius=2):
    """Рисует ключевые точки на изображении."""
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(image, (x, y), radius=keypointsRadius, color=colorBGR, thickness=-1) # BGR format, -1 means filled circle

def blend_swapped_image_gpu(swapped_face, target_image, M):
    """GPU-ускоренное смешивание заменённого лица с целевым изображением."""
    h, w = target_image.shape[:2]
    M_inv = cv2.invertAffineTransform(M)

    # Warp swapped face
    warped_face = cv2.warpAffine(
        swapped_face,
        M_inv,
        (w, h),
        borderValue=(0, 0, 0)
    )
    # Create white mask
    img_white = np.full(swapped_face.shape[:2], 255, dtype=np.uint8)
    img_mask = cv2.warpAffine(img_white, M_inv, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0,))
    # Threshold and refine mask

    _, _, w_box, h_box = cv2.boundingRect(img_mask)
    mask_size = int(np.sqrt(w_box * h_box))

    k = max(mask_size // 10, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    img_mask = cv2.erode(img_mask, kernel, iterations=1)

    k_blur = max(mask_size // 20, 5)
    blur_size = (2 * k_blur + 1, 2 * k_blur + 1)
    img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)

    # Move to GPU for blending
    img_mask = torch.from_numpy(img_mask).to('cuda').unsqueeze(2)/255  # HWC, single channel
    warped_face = torch.from_numpy(warped_face).to(device='cuda')
    target_image = torch.from_numpy(target_image).to(device='cuda')
    # Blend
    result = img_mask * warped_face + (1 - img_mask) * target_image
    result = result.clamp(0, 255).byte().cpu().numpy()  # Back to CPU and uint8

    return result

def create_face_mask(face, frame) -> np.ndarray:
    """Создает маску лица на основе ключевых точек."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        # Convert landmarks to int32
        landmarks = landmarks.astype(np.int32)

        # Extract facial features
        right_side_face = landmarks[0:16]
        left_side_face = landmarks[17:32]
        right_eye = landmarks[33:42]
        right_eye_brow = landmarks[43:51]
        left_eye = landmarks[87:96]
        left_eye_brow = landmarks[97:105]

        # Calculate forehead extension
        right_eyebrow_top = np.min(right_eye_brow[:, 1])
        left_eyebrow_top = np.min(left_eye_brow[:, 1])
        eyebrow_top = min(right_eyebrow_top, left_eyebrow_top)

        face_top = np.min([right_side_face[0, 1], left_side_face[-1, 1]])
        forehead_height = face_top - eyebrow_top
        extended_forehead_height = int(forehead_height * 5.0)  # Extend by 50%

        # Create forehead points
        forehead_left = right_side_face[0].copy()
        forehead_right = left_side_face[-1].copy()
        forehead_left[1] -= extended_forehead_height
        forehead_right[1] -= extended_forehead_height

        # Combine all points to create the face outline
        face_outline = np.vstack(
            [
                [forehead_left],
                right_side_face,
                left_side_face[
                    ::-1
                ],  # Reverse left side to create a continuous outline
                [forehead_right],
            ]
        )

        # Calculate padding
        padding = int(
            np.linalg.norm(right_side_face[0] - left_side_face[-1]) * 0.05
        )  # 5% of face width

        # Create a slightly larger convex hull for padding
        hull = cv2.convexHull(face_outline)
        hull_padded = []
        for point in hull:
            x, y = point[0]
            center = np.mean(face_outline, axis=0)
            direction = np.array([x, y]) - center
            direction = direction / np.linalg.norm(direction)
            padded_point = np.array([x, y]) + direction * padding
            hull_padded.append(padded_point)

        hull_padded = np.array(hull_padded, dtype=np.int32)

        # Create mask
        cv2.fillPoly(mask, [hull_padded], (255,))

        # Blur mask for softer edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)

    return mask 
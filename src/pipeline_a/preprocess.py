"""OpenCV 影像前處理：deskew、去噪、二值化。三步都可由 config 開關，
供 Phase 4 做「有無前處理」消融實驗（這正是本管線要展示 CV 深度的地方）。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    deskew: bool = True
    denoise: bool = True
    binarize: bool = True


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def estimate_skew_angle(gray: np.ndarray) -> float:
    """回傳文件相對水平的旋轉角度（正值＝逆時針偏轉，對應 cv2 旋轉矩陣慣例）。

    用 Otsu 反相二值化取出前景（文字）像素，取其 minAreaRect 的角度。
    OpenCV 的 minAreaRect 角度落在 [-90, 0)，且哪條邊被當作「寬」是不固定的，
    因此用 `< -45 則 +90` 這個業界慣用轉換把角度映射回 (-45, 45] 的偏轉量；
    已用已知旋轉角度的合成測試圖（scripts/make_sample_image.py 的 sample_002，
    旋轉 12°）驗證過此轉換方向正確（估出 ≈ 12°）。
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if coords.shape[0] < 10:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    return float(angle)


def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """回傳 (校正後影像, 估計偏轉角度)。旋轉以白色補邊，避免黑角污染後續二值化。"""
    angle = estimate_skew_angle(gray)
    if abs(angle) < 0.3:  # 偏轉太小時不旋轉，避免無謂的重新取樣模糊
        return gray, angle
    h, w = gray.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(
        gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return rotated, angle


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=31, C=15,
    )


def preprocess(image: np.ndarray, config: PreprocessConfig | None = None) -> np.ndarray:
    """完整前處理管線；config 為 None 時等同全部開啟。"""
    config = config or PreprocessConfig()
    gray = to_grayscale(image)

    if config.deskew:
        gray, _angle = deskew(gray)
    if config.denoise:
        gray = denoise(gray)
    if config.binarize:
        gray = binarize(gray)

    return gray

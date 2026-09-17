"""
Извлечение признаков (embedding) из вырезанного растения.

Задача — построить компактный числовой "портрет" растения, по которому
можно сравнивать похожесть между эталонными фото сорняков и объектами,
найденными на фото полей: цвет + текстура листа + форма контура
(инвариантные к повороту и масштабу моменты Ху). Это классический,
объяснимый и не требующий скачивания тяжёлых нейросетевых весов подход —
он полностью укладывается в офлайн-демо хакатона и уже "из коробки"
доступен через opencv/scikit-image.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

FEATURE_SIZE = 128  # сторона квадрата, к которому приводится каждый патч


def _color_histogram(bgr: np.ndarray, mask: np.ndarray, bins: int = 16) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = []
    for ch, n in zip((0, 1, 2), (bins, bins, bins)):
        h = cv2.calcHist([hsv], [ch], mask, [n], [0, 256])
        h = cv2.normalize(h, None, 0, 1, cv2.NORM_MINMAX).flatten()
        hist.append(h)
    return np.concatenate(hist)


def _texture_histogram(gray: np.ndarray, mask: np.ndarray, p: int = 8, r: int = 1) -> np.ndarray:
    lbp = local_binary_pattern(gray, p, r, method="uniform")
    n_bins = p + 2
    values = lbp[mask > 0]
    if values.size == 0:
        values = lbp.flatten()
    hist, _ = np.histogram(values, bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def _shape_descriptor(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(9, dtype=np.float32)

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull) + 1e-6

    aspect_ratio = w / (h + 1e-6)
    extent = area / (w * h + 1e-6)
    solidity = area / hull_area
    compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)

    hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    basic = np.array([aspect_ratio, extent, solidity, compactness], dtype=np.float32)
    return np.concatenate([basic, hu_log[:5].astype(np.float32)])


def extract_features(bgr_patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Считает единый вектор признаков для патча растения + его маски.

    bgr_patch и mask должны быть одного размера (mask — 0/255, uint8).
    """
    patch = cv2.resize(bgr_patch, (FEATURE_SIZE, FEATURE_SIZE))
    mask_r = cv2.resize(mask, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    color_feat = _color_histogram(patch, mask_r)
    texture_feat = _texture_histogram(gray, mask_r)
    shape_feat = _shape_descriptor(mask_r)

    feat = np.concatenate([color_feat, texture_feat, shape_feat]).astype(np.float32)
    norm = np.linalg.norm(feat)
    if norm > 1e-6:
        feat = feat / norm
    return feat

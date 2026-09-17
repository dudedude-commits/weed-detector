"""
Сегментация растительности на фото поля с дрона.

Идея: на кадрах с дрона почва тёмная и слабо насыщена зелёным, а листья
растений (и культуры, и сорняков) заметно зеленее. Поэтому вместо тяжёлой
нейросети для первого прохода используется классический вегетационный
индекс Excess Green (ExG) + адаптивная бинаризация (Otsu) — это быстро,
не требует обучения и работает прямо "из коробки" на любых новых снимках.

После бинаризации ищутся связные компоненты (отдельные кустики/розетки),
которые дальше передаются в классификатор (pipeline/classifier.py) для
определения вида и стадии.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Blob:
    """Один найденный на фото растительный объект-кандидат."""

    bbox: tuple[int, int, int, int]  # x, y, w, h в пикселях исходного фото
    mask: np.ndarray  # бинарная маска объекта размером с bbox (uint8, 0/255)
    area_px: int
    contour: np.ndarray


def excess_green_mask(bgr: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
    """Строит бинарную маску растительности по индексу Excess Green (ExG).

    ExG = 2*G - R - B, нормализованный по каналам. Работает устойчиво на
    открытой почве, т.к. у голой земли каналы R/G/B близки друг к другу,
    а у листвы канал G заметно доминирует.
    """
    img = bgr.astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    total = r + g + b + 1e-6
    r_n, g_n, b_n = r / total, g / total, b / total

    exg = 2 * g_n - r_n - b_n
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if blur_ksize > 1:
        exg_norm = cv2.GaussianBlur(exg_norm, (blur_ksize, blur_ksize), 0)

    _, mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Убираем мелкий шум и склеиваем разорванные листья одного растения.
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    return mask


def find_plant_blobs(
    bgr: np.ndarray,
    min_area_frac: float = 0.0008,
    max_area_frac: float = 0.5,
    pad_frac: float = 0.15,
) -> list[Blob]:
    """Находит на фото поля отдельные кустики растительности.

    min_area_frac / max_area_frac — доля площади кадра, отсекающая шум
    (пылинки, солому) и слишком крупные объекты (если в кадр случайно
    попал большой сорняк почти на весь кадр — такое тоже оставляем, но
    предохранитель от объединения "всё поле в один blob" нужен).
    """
    h, w = bgr.shape[:2]
    frame_area = h * w
    mask = excess_green_mask(bgr)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs: list[Blob] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_frac * frame_area or area > max_area_frac * frame_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Небольшой отступ вокруг найденного объекта, чтобы в вырезанный
        # патч попали кончики листьев целиком (важно для классификации).
        pad = int(max(bw, bh) * pad_frac)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)

        local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        shifted = cnt - [x0, y0]
        cv2.drawContours(local_mask, [shifted], -1, 255, thickness=cv2.FILLED)

        blobs.append(
            Blob(
                bbox=(x0, y0, x1 - x0, y1 - y0),
                mask=local_mask,
                area_px=int(area),
                contour=cnt,
            )
        )

    # Крупные объекты обычно интереснее для демонстрации — но сортируем
    # по площади, чтобы в UI/CSV сначала шли самые заметные экземпляры.
    blobs.sort(key=lambda b: b.area_px, reverse=True)
    return blobs


def segment_reference_plant(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Выделяет главное растение на эталонном крупноплановом фото.

    Эталонные снимки почти всегда сняты так, что нужное растение занимает
    центр кадра и является самым крупным зелёным объектом на снимке.
    Возвращает (bbox-кроп исходного изображения, маска в его координатах).
    Если ничего не найдено (например, кадр совсем без зелени) — возвращает
    весь кадр целиком с полностью белой маской, чтобы признаки всё равно
    можно было посчитать.
    """
    h, w = bgr.shape[:2]
    mask_full = excess_green_mask(bgr)
    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return bgr, np.full((h, w), 255, dtype=np.uint8)

    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    pad = int(max(bw, bh) * 0.05)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)

    crop = bgr[y0:y1, x0:x1]
    local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    shifted = largest - [x0, y0]
    cv2.drawContours(local_mask, [shifted], -1, 255, thickness=cv2.FILLED)

    return crop, local_mask

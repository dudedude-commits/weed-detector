"""
Юникод-безопасные обёртки над чтением/записью изображений.

Настоящая причина бага "Не удалось построить индекс эталонов" на Windows:
cv2.imread()/cv2.imwrite() внутри используют функции, которые на Windows не
умеют работать с путями, содержащими не-ASCII символы (кириллицу). А у нас
ВСЕ папки эталонов называются кириллицей (`Бодяк полевой/Розетка/...`), да и
сам путь к проекту на компьютере пользователя часто содержит кириллицу
(имя пользователя Windows). cv2.imread в такой ситуации не бросает
исключение, а молча возвращает None — что выглядит так, будто "все фото
битые", хотя на самом деле дело только в пути.

Решение — читать файл в байтах через обычный Python (Path.read_bytes,
он с юникодными путями работает нормально в любой ОС) и декодировать байты
в картинку через cv2.imdecode/cv2.imencode, которые от пути уже не зависят.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: str | Path) -> np.ndarray | None:
    """Аналог cv2.imread, но корректно работает с путями на кириллице."""
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def imwrite_unicode(path: str | Path, img: np.ndarray) -> bool:
    """Аналог cv2.imwrite, но корректно работает с путями на кириллице."""
    path = Path(path)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())
    return True

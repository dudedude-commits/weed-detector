"""
Оркестрация полного цикла обработки одного фото поля:

    сегментация растительности -> кандидаты (blobs)
        -> классификация каждого кандидата по базе эталонов (вид + стадия)
        -> отсев низкой уверенности
        -> подсчёт по видам/стадиям
        -> отрисовка рамок на изображении
        -> сохранение результатов в JSON/CSV
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .classifier import ReferenceDatabase
from .io_utils import imwrite_unicode
from .segmentation import find_plant_blobs

# OpenCV umeet risovat' tol'ko ASCII cherez svoi vstroennye Hershey-shrifty,
# kirillica prevrashchaetsya v "?????". Poetomu podpisi risuem cherez PIL
# s TTF-shriftom, podderzhivayushchim russkie bukvy.
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _FONT_CACHE:
        font = None
        for path in _FONT_CANDIDATES:
            if Path(path).exists():
                font = ImageFont.truetype(path, size)
                break
        if font is None:
            font = ImageFont.load_default()
        _FONT_CACHE[size] = font
    return _FONT_CACHE[size]

# Валидированная категориальная палитра (фиксированный порядок слотов —
# не переставляется и не зацикливается на 9-м виде: см. dataviz skill,
# references/palette.md). Один и тот же порядок используется и для рамок
# на фото, и для графика в интерфейсе, поэтому цвет вида везде одинаковый.
SPECIES_PALETTE_HEX = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


SPECIES_COLORS = [_hex_to_bgr(h) for h in SPECIES_PALETTE_HEX]


def _color_for(species: str, species_order: list[str]) -> tuple[int, int, int]:
    if species not in species_order:
        species_order.append(species)
    idx = species_order.index(species) % len(SPECIES_COLORS)
    return SPECIES_COLORS[idx]


@dataclass
class Detection:
    id: int
    bbox: list[int]  # x, y, w, h
    species: str
    stage: str | None
    confidence: float
    area_px: int
    crop_file: str | None = None  # относительный путь к вырезанному фото объекта (crops/…)


@dataclass
class ImageResult:
    image_name: str
    width: int
    height: int
    detections: list[Detection]
    counts: dict[str, int]  # "Вид (Стадия)" -> количество


def _make_cutout(patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Вырезает найденный сорняк из фото поля: пиксели растения остаются как
    на исходном фото, а всё, что вне маски (почва, соседние объекты),
    становится прозрачным — получается "отсечённый" объект на PNG с альфа-
    каналом, который можно рассматривать отдельно от фона поля."""
    alpha = np.where(mask > 0, 255, 0).astype(np.uint8)
    bgra = cv2.cvtColor(patch, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra


def process_image(
    bgr: np.ndarray,
    image_name: str,
    ref_db: ReferenceDatabase,
    min_confidence: float = 0.35,
    species_order: list[str] | None = None,
) -> tuple[ImageResult, np.ndarray, dict[str, np.ndarray]]:
    """Обрабатывает одно фото поля.

    Возвращает (результат, аннотированную копию фото, словарь "отсечённых"
    вырезок по каждому найденному объекту — ключ вида ``crops/<имя>.png``,
    значение — BGRA-изображение с прозрачным фоном вне маски растения).
    """

    species_order = species_order if species_order is not None else []
    h, w = bgr.shape[:2]
    blobs = find_plant_blobs(bgr)
    stem = Path(image_name).stem

    detections: list[Detection] = []
    counts: dict[str, int] = {}
    boxes_to_draw: list[tuple[tuple[int, int, int, int], str, tuple[int, int, int], str]] = []
    crops: dict[str, np.ndarray] = {}

    for i, blob in enumerate(blobs, start=1):
        x, y, bw, bh = blob.bbox
        patch = bgr[y : y + bh, x : x + bw]
        result = ref_db.classify(patch, blob.mask)

        if result is None or result.confidence < min_confidence:
            continue

        crop_file = f"crops/{stem}_{i:03d}.png"
        det = Detection(
            id=i,
            bbox=[x, y, bw, bh],
            species=result.species,
            stage=result.stage,
            confidence=round(result.confidence, 3),
            area_px=blob.area_px,
            crop_file=crop_file,
        )
        detections.append(det)
        crops[crop_file] = _make_cutout(patch, blob.mask)

        label_key = f"{result.species} ({result.stage})" if result.stage else result.species
        counts[label_key] = counts.get(label_key, 0) + 1

        color = _color_for(result.species, species_order)
        label = f"{result.species}"
        if result.stage:
            label += f" / {result.stage}"
        label += f" {result.confidence:.2f}"
        boxes_to_draw.append(((x, y, bw, bh), label, color, result.species))

    annotated = _draw_annotations(bgr, boxes_to_draw)

    return (
        ImageResult(image_name=image_name, width=w, height=h, detections=detections, counts=counts),
        annotated,
        crops,
    )


def _draw_annotations(
    bgr: np.ndarray,
    boxes: list[tuple[tuple[int, int, int, int], str, tuple[int, int, int], str]],
) -> np.ndarray:
    """Обводит найденные объекты кружками (цвет = вид) и подписывает их
    (кириллица через PIL), затем добавляет в угол фото легенду "цвет → вид".
    Возвращает BGR-массив."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    font_size = max(12, min(22, bgr.shape[1] // 90))
    font = _get_font(font_size)
    legend_font = _get_font(max(20, font_size + 6))
    line_width = max(2, bgr.shape[1] // 700)

    legend_entries: list[tuple[str, tuple[int, int, int]]] = []
    seen_species: set[str] = set()

    for (x, y, bw, bh), label, color_bgr, species in boxes:
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        draw.ellipse([x, y, x + bw, y + bh], outline=color_rgb, width=line_width)

        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        pad = 3
        y_text_top = max(0, y - th - 2 * pad)
        draw.rectangle([x, y_text_top, x + tw + 2 * pad, y_text_top + th + 2 * pad], fill=color_rgb)
        draw.text((x + pad, y_text_top + pad - text_bbox[1]), label, font=font, fill=(255, 255, 255))

        if species not in seen_species:
            seen_species.add(species)
            legend_entries.append((species, color_rgb))

    _draw_legend(draw, legend_entries, bgr.shape[1], bgr.shape[0], legend_font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _draw_legend(
    draw: "ImageDraw.ImageDraw",
    entries: list[tuple[str, tuple[int, int, int]]],
    img_w: int,
    img_h: int,
    font: "ImageFont.FreeTypeFont",
) -> None:
    """Рисует в левом нижнем углу подпись "какой цвет — какой вид", чтобы
    цветные кружки на фото было легко расшифровать без обращения к CSV."""
    if not entries:
        return

    pad_outer = 24
    pad_inner = 18
    swatch_r = max(9, font.size // 2)
    row_gap = 10
    row_h = max(font.size + 10, swatch_r * 2 + 6)

    max_tw = 0
    for species, _color in entries:
        b = draw.textbbox((0, 0), species, font=font)
        max_tw = max(max_tw, b[2] - b[0])

    box_w = pad_inner * 2 + swatch_r * 2 + 14 + max_tw
    box_h = pad_inner * 2 + len(entries) * row_h + (len(entries) - 1) * row_gap

    x0 = pad_outer
    y0 = max(0, img_h - pad_outer - box_h)
    x1 = min(img_w - 1, x0 + box_w)
    y1 = y0 + box_h

    draw.rectangle([x0, y0, x1, y1], fill=(17, 20, 17))

    ty = y0 + pad_inner
    for species, color in entries:
        cy = ty + row_h / 2
        cx = x0 + pad_inner + swatch_r
        draw.ellipse([cx - swatch_r, cy - swatch_r, cx + swatch_r, cy + swatch_r], fill=color)
        text_bbox = draw.textbbox((0, 0), species, font=font)
        th = text_bbox[3] - text_bbox[1]
        draw.text(
            (cx + swatch_r + 14, cy - th / 2 - text_bbox[1]),
            species,
            font=font,
            fill=(240, 240, 235),
        )
        ty += row_h + row_gap


def results_to_json(results: list[ImageResult]) -> str:
    payload = []
    for r in results:
        payload.append(
            {
                "image": r.image_name,
                "width": r.width,
                "height": r.height,
                "detections": [asdict(d) for d in r.detections],
                "counts": r.counts,
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def results_to_dataframe(results: list[ImageResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for d in r.detections:
            rows.append(
                {
                    "image": r.image_name,
                    "id": d.id,
                    "species": d.species,
                    "stage": d.stage or "",
                    "confidence": d.confidence,
                    "x": d.bbox[0],
                    "y": d.bbox[1],
                    "w": d.bbox[2],
                    "h": d.bbox[3],
                    "area_px": d.area_px,
                    "crop_file": d.crop_file or "",
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["image", "id", "species", "stage", "confidence", "x", "y", "w", "h", "area_px", "crop_file"]
        )
    return pd.DataFrame(rows)


def summary_dataframe(results: list[ImageResult]) -> pd.DataFrame:
    """Сводная таблица: сколько экземпляров каждого вида/стадии на каждом фото."""
    rows = []
    for r in results:
        for label, n in r.counts.items():
            rows.append({"image": r.image_name, "вид_стадия": label, "количество": n})
    if not rows:
        return pd.DataFrame(columns=["image", "вид_стадия", "количество"])
    return pd.DataFrame(rows)


def aggregate_species_stage_counts(results: list[ImageResult], species_order: list[str] | None = None) -> pd.DataFrame:
    """Суммарные счётчики по (вид, стадия) для всех обработанных фото сразу —
    используется для графика и сводного метрик-блока в интерфейсе."""
    totals: dict[tuple[str, str], int] = {}
    for r in results:
        for d in r.detections:
            key = (d.species, d.stage or "—")
            totals[key] = totals.get(key, 0) + 1

    order = list(species_order) if species_order else []
    for species, _stage in totals:
        if species not in order:
            order.append(species)

    rows = [
        {"вид": species, "стадия": stage, "метка": f"{species} · {stage}" if stage != "—" else species, "количество": n}
        for (species, stage), n in totals.items()
    ]
    df = pd.DataFrame(rows, columns=["вид", "стадия", "метка", "количество"])
    if df.empty:
        return df
    df["вид"] = pd.Categorical(df["вид"], categories=order, ordered=True)
    return df.sort_values(["вид", "стадия"]).reset_index(drop=True)


def save_results(
    results: list[ImageResult],
    annotated_images: dict[str, np.ndarray],
    crops: dict[str, np.ndarray] | None = None,
    out_dir: str | Path = "results",
) -> dict[str, str]:
    """Сохраняет JSON, CSV (детально и сводно), аннотированные фото и
    отдельные "отсечённые" вырезки каждого найденного сорняка на диск.

    ``crops`` — словарь из process_image(): ключ "crops/<имя>.png" (уже тот
    же путь, что записан в detections.csv/results.json как crop_file),
    значение — BGRA-картинка объекта с прозрачным фоном.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "results.json"
    json_path.write_text(results_to_json(results), encoding="utf-8")

    csv_path = out_dir / "detections.csv"
    results_to_dataframe(results).to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary_path = out_dir / "summary.csv"
    summary_dataframe(results).to_csv(summary_path, index=False, encoding="utf-8-sig")

    annotated_dir = out_dir / "annotated"
    annotated_dir.mkdir(exist_ok=True)
    for name, img in annotated_images.items():
        imwrite_unicode(annotated_dir / name, img)

    crops_dir = out_dir / "crops"
    if crops:
        crops_dir.mkdir(exist_ok=True)
        for rel_name, img in crops.items():
            # rel_name уже вида "crops/<файл>.png" — сохраняем относительно out_dir
            imwrite_unicode(out_dir / rel_name, img)

    return {
        "json": str(json_path),
        "detections_csv": str(csv_path),
        "summary_csv": str(summary_path),
        "annotated_dir": str(annotated_dir),
        "crops_dir": str(crops_dir),
    }

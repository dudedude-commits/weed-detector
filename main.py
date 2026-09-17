"""
CLI-запуск обработки без веб-интерфейса — удобно для пакетной обработки
или для демонстрации/проверки организаторами.

Пример:
    python main.py \\
        --ref-dir reference_data \\
        --field-dir uploads/fields \\
        --out-dir results \\
        --min-confidence 0.35

Структура --ref-dir ожидается такой же, как в исходном датасете:
    reference_dir/<Вид сорняка>/<Стадия>/*.jpg
(для видов без стадии, напр. злаков — можно положить фото прямо в
подпапку "_" или в любую единственную подпапку с фото).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.classifier import ReferenceDatabase
from pipeline.io_utils import imread_unicode
from pipeline.pipeline import process_image, save_results

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Поиск и классификация сорняков на фото полей с дрона")
    parser.add_argument("--ref-dir", default="reference_data", help="Папка с эталонными фото сорняков (Вид/Стадия/*.jpg)")
    parser.add_argument("--field-dir", required=True, help="Папка с фото полей для обработки")
    parser.add_argument("--out-dir", default="results", help="Куда сохранить JSON/CSV и размеченные фото")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="Порог уверенности классификации (0..1)")
    args = parser.parse_args()

    ref_db = ReferenceDatabase(args.ref_dir)
    if ref_db.is_empty():
        raise SystemExit(
            f"В {args.ref_dir} нет эталонных фото. Ожидается структура "
            f"<ref-dir>/<Вид>/<Стадия>/*.jpg — добавьте эталоны и повторите запуск."
        )

    print("Строим индекс эталонов…")
    ref_db.build_index()

    field_dir = Path(args.field_dir)
    image_paths = sorted(p for p in field_dir.iterdir() if p.suffix in IMAGE_EXTS)
    if not image_paths:
        raise SystemExit(f"В {args.field_dir} не найдено изображений.")

    results = []
    annotated_images = {}
    all_crops = {}
    species_order: list[str] = []

    for path in image_paths:
        bgr = imread_unicode(path)
        if bgr is None:
            print(f"  ! не удалось прочитать {path.name}, пропускаю")
            continue
        print(f"  обрабатываю {path.name}…")
        result, annotated, crops = process_image(
            bgr, path.name, ref_db, min_confidence=args.min_confidence, species_order=species_order
        )
        results.append(result)
        annotated_images[path.name] = annotated
        all_crops.update(crops)
        print(f"    найдено объектов: {len(result.detections)}  ({result.counts})")

    paths = save_results(results, annotated_images, crops=all_crops, out_dir=args.out_dir)
    print("\nГотово. Сохранено:")
    for key, val in paths.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()

"""
База эталонов сорняков и классификатор "вид + стадия вегетации" на её основе.

Никакого обучения "с нуля" не требуется: пользователь загружает эталонные
фото (как в исходном датасете хакатона — по папкам вид/стадия), для каждого
эталона считается embedding (pipeline/features.py), а новый объект,
найденный на фото поля, классифицируется методом k ближайших соседей
(k-NN) по косинусному расстоянию до эталонов. Это стандартный приём для
few-shot классификации, когда размеченных данных мало, а полноценно
обучать нейросеть негде и не на чем.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .features import extract_features
from .io_utils import imread_unicode
from .segmentation import segment_reference_plant

SUPPORTED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


@dataclass
class ClassificationResult:
    species: str
    stage: str | None
    confidence: float
    neighbors: list[tuple[str, str | None, float]] = field(default_factory=list)


class ReferenceDatabase:
    """Хранит эталонные фото сорняков на диске в структуре

        root/<Вид>/<Стадия>/картинка.jpg

    и строит по ним индекс для k-NN классификации. Стадию можно не
    указывать (например, для злаков без розетки) — тогда фото кладётся в
    root/<Вид>/_/картинка.jpg и стадия не предсказывается для этого вида.
    """

    NO_STAGE = "_"

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._features: np.ndarray | None = None
        self._labels: list[tuple[str, str | None]] = []
        self._paths: list[str] = []
        self._nn: NearestNeighbors | None = None

    # ------------------------------------------------------------------ #
    # Наполнение базы
    # ------------------------------------------------------------------ #
    def add_image(self, species: str, stage: str | None, filename: str, data: bytes) -> Path:
        species = species.strip()
        stage_dir = (stage or self.NO_STAGE).strip() or self.NO_STAGE
        target_dir = self.root_dir / species / stage_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename
        # избегаем перезаписи одноимённых файлов из разных загрузок
        stem, suffix = target_path.stem, target_path.suffix
        counter = 1
        while target_path.exists():
            target_path = target_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        target_path.write_bytes(data)
        return target_path

    def list_summary(self) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for species_dir in sorted(p for p in self.root_dir.iterdir() if p.is_dir()):
            stages: dict[str, int] = {}
            for stage_dir in sorted(p for p in species_dir.iterdir() if p.is_dir()):
                n = sum(1 for f in stage_dir.iterdir() if f.suffix in SUPPORTED_IMAGE_EXT)
                if n:
                    stages[stage_dir.name] = n
            if stages:
                summary[species_dir.name] = stages
        return summary

    def is_empty(self) -> bool:
        return not self.list_summary()

    # ------------------------------------------------------------------ #
    # Построение индекса
    # ------------------------------------------------------------------ #
    def build_index(self) -> None:
        feats: list[np.ndarray] = []
        labels: list[tuple[str, str | None]] = []
        paths: list[str] = []

        for species_dir in sorted(p for p in self.root_dir.iterdir() if p.is_dir()):
            for stage_dir in sorted(p for p in species_dir.iterdir() if p.is_dir()):
                stage_name = None if stage_dir.name == self.NO_STAGE else stage_dir.name
                for img_path in sorted(stage_dir.iterdir()):
                    if img_path.suffix not in SUPPORTED_IMAGE_EXT:
                        continue
                    bgr = imread_unicode(img_path)
                    if bgr is None:
                        continue
                    crop, mask = segment_reference_plant(bgr)
                    feat = extract_features(crop, mask)
                    feats.append(feat)
                    labels.append((species_dir.name, stage_name))
                    paths.append(str(img_path))

        if not feats:
            self._features = None
            self._labels = []
            self._paths = []
            self._nn = None
            return

        self._features = np.stack(feats)
        self._labels = labels
        self._paths = paths
        n_neighbors = min(7, len(feats))
        self._nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        self._nn.fit(self._features)

    @property
    def ready(self) -> bool:
        return self._nn is not None

    # ------------------------------------------------------------------ #
    # Классификация
    # ------------------------------------------------------------------ #
    def classify(self, bgr_patch: np.ndarray, mask: np.ndarray, k: int = 5) -> ClassificationResult | None:
        if not self.ready:
            return None

        feat = extract_features(bgr_patch, mask).reshape(1, -1)
        k = min(k, self._features.shape[0])
        distances, indices = self._nn.kneighbors(feat, n_neighbors=k)
        distances, indices = distances[0], indices[0]
        similarities = 1.0 - distances  # cosine distance -> similarity

        neighbor_labels = [self._labels[i] for i in indices]
        neighbors = [
            (lab[0], lab[1], float(sim)) for lab, sim in zip(neighbor_labels, similarities)
        ]

        # Голосуем по виду, взвешивая голоса сходством с эталоном.
        species_votes: dict[str, float] = {}
        for (species, _stage), sim in zip(neighbor_labels, similarities):
            species_votes[species] = species_votes.get(species, 0.0) + max(sim, 0.0)
        best_species = max(species_votes, key=species_votes.get)

        # Стадию определяем только по соседям того же вида.
        stage_votes: dict[str, float] = {}
        for (species, stage), sim in zip(neighbor_labels, similarities):
            if species == best_species and stage is not None:
                stage_votes[stage] = stage_votes.get(stage, 0.0) + max(sim, 0.0)
        best_stage = max(stage_votes, key=stage_votes.get) if stage_votes else None

        same_species_sims = [sim for (species, _s), sim in zip(neighbor_labels, similarities) if species == best_species]
        confidence = float(np.mean(same_species_sims)) if same_species_sims else float(np.mean(similarities))
        confidence = max(0.0, min(1.0, confidence))

        return ClassificationResult(
            species=best_species,
            stage=best_stage,
            confidence=confidence,
            neighbors=neighbors,
        )

    def to_json_summary(self) -> str:
        return json.dumps(self.list_summary(), ensure_ascii=False, indent=2)

"""Deterministic local fixtures for the data-cleaning pipeline tests."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np


def inventory_for_selection(now: datetime | None = None) -> list[dict[str, Any]]:
    reference = now or datetime(2026, 9, 16, tzinfo=UTC)
    inventory: list[dict[str, Any]] = []
    for label in ("fries", "burger"):
        for index in range(3):
            inventory.append(
                {
                    "id": f"wc-{label}-{index}",
                    "key": f"incoming/wc_{label}_{0.90 + index * 0.01:.2f}.jpg",
                    "last_modified": (reference - timedelta(days=index)).isoformat(),
                }
            )
    inventory.extend(
        [
            {"id": "other-brand", "key": "bb_fries_0.9.jpg", "last_modified": reference.isoformat()},
            {
                "id": "old", "key": "wc_fries_0.9.jpg",
                "last_modified": (reference - timedelta(days=400)).isoformat(),
            },
            {"id": "underrepresented", "key": "wc_salad_0.9.jpg", "last_modified": reference.isoformat()},
        ]
    )
    return inventory


def embedding_fixture(
    *, classes: int = 3, samples_per_class: int = 10, dimensions: int = 16, seed: int = 2026
) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(classes, dimensions)).astype(np.float32) * 3
    features = []
    labels = []
    for class_id in range(classes):
        features.append(centers[class_id] + rng.normal(scale=0.25, size=(samples_per_class, dimensions)))
        labels.extend([f"class-{class_id}"] * samples_per_class)
    return np.vstack(features).astype(np.float32), labels


def mislabeled_probabilities(
    *, classes: int = 3, samples_per_class: int = 20, injected_mislabels: int = 10
) -> tuple[np.ndarray, list[dict[str, str]], list[str], set[str]]:
    total = classes * samples_per_class
    true_labels = np.repeat(np.arange(classes), samples_per_class)
    stored_labels = true_labels.copy()
    injected_indices = np.linspace(0, total - 1, injected_mislabels, dtype=int)
    for index in injected_indices:
        stored_labels[index] = (true_labels[index] + 1) % classes
    probabilities = np.full((total, classes), 0.02 / max(classes - 1, 1), dtype=np.float64)
    probabilities[np.arange(total), true_labels] = 0.98
    class_names = [f"class-{index}" for index in range(classes)]
    mapping = [{"id": f"image-{index}", "label": class_names[label]} for index, label in enumerate(stored_labels)]
    injected_ids = {f"image-{index}" for index in injected_indices}
    return probabilities, mapping, class_names, injected_ids

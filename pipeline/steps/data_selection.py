"""Pipeline step: SelectRecentBrandData.

Container entrypoint referenced by the SageMaker Pipeline DAG. Scans the raw
image channel, keeps files matching the brand naming pattern, and emits a
selection ``metrics.json`` consumed by the DatasetBoundsGate:

    { "dataset_size": <int>, "minimum_class_count": <int> }

It also writes a curated manifest that the embedding step consumes. Label and
confidence are parsed from the ``<brand>_<label>_<confidence>.<ext>`` filename
scheme; the confidence field is retained for downstream reporting.

Python 3.10 compatible (SageMaker DLC containers run 3.10).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def _parse_name(path: Path) -> tuple[str, str] | None:
    """Return (label, confidence) from ``brand_label_confidence.ext`` or None."""
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    # brand = parts[0]; confidence = parts[-1]; label = everything between.
    label = "".join(parts[1:-1])
    confidence = parts[-1]
    return label, confidence


def select(
    image_root: Path,
    naming_prefix: str,
    *,
    min_images_per_class: int,
    target_dataset_min: int,
    target_dataset_max: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    for path in sorted(image_root.rglob("*")):
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        if naming_prefix and not path.name.startswith(naming_prefix):
            continue
        parsed = _parse_name(path)
        if parsed is None:
            continue
        label, confidence = parsed
        label_counts[label] += 1
        manifest.append(
            {
                "id": path.name,
                "key": path.name,
                "image_path": path.name,
                "label": label,
                "confidence": confidence,
                "last_modified": datetime.now(UTC).isoformat(),
            }
        )
    dataset_size = len(manifest)
    minimum_class_count = min(label_counts.values()) if label_counts else 0
    metrics = {
        "dataset_size": dataset_size,
        "minimum_class_count": minimum_class_count,
        "class_distribution": dict(sorted(label_counts.items())),
        "target_dataset_min": target_dataset_min,
        "target_dataset_max": target_dataset_max,
        "min_images_per_class": min_images_per_class,
    }
    return manifest, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="SelectRecentBrandData step")
    parser.add_argument("--images", default="/opt/ml/processing/input/images")
    parser.add_argument("--dataset-out", default="/opt/ml/processing/output/dataset")
    parser.add_argument("--metrics-out", default="/opt/ml/processing/output/metrics")
    args = parser.parse_args()

    naming_prefix = os.environ.get("NAMING_PATTERN", "wc_*").replace("*", "")
    min_images_per_class = int(os.environ.get("MIN_IMAGES_PER_CLASS", "20"))
    target_dataset_min = int(os.environ.get("TARGET_DATASET_MIN", "40"))
    target_dataset_max = int(os.environ.get("TARGET_DATASET_MAX", "50000"))

    manifest, metrics = select(
        Path(args.images),
        naming_prefix,
        min_images_per_class=min_images_per_class,
        target_dataset_min=target_dataset_min,
        target_dataset_max=target_dataset_max,
    )

    dataset_out = Path(args.dataset_out)
    dataset_out.mkdir(parents=True, exist_ok=True)
    (dataset_out / "curated_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    metrics_out = Path(args.metrics_out)
    metrics_out.mkdir(parents=True, exist_ok=True)
    (metrics_out / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[data_selection] dataset_size={metrics['dataset_size']} "
        f"minimum_class_count={metrics['minimum_class_count']} "
        f"distribution={metrics['class_distribution']}"
    )


if __name__ == "__main__":
    main()

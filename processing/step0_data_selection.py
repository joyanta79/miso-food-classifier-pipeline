"""Step 0: select recent, per-brand image records into a curated manifest.

The module is deliberately S3-client agnostic so it can be exercised locally from a
JSON inventory and reused by a SageMaker Processing job after inventory export.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any

_METADATA_PATTERN = re.compile(
    r"^(?P<brand>[^_]+)_(?P<label>.+)_(?P<confidence>\d+(?:\.\d+)?)$"
)


def parse_filename_metadata(key: str) -> tuple[str, str, float]:
    """Parse ``brand_label_confidence.ext`` metadata encoded in an image key."""
    stem = Path(key).stem
    match = _METADATA_PATTERN.match(stem)
    if not match:
        raise ValueError(f"Image key does not match brand_label_confidence format: {key}")
    return match.group("brand"), match.group("label"), float(match.group("confidence"))


def _parse_timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        timestamp = value
    else:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return timestamp.replace(tzinfo=UTC) if timestamp.tzinfo is None else timestamp.astimezone(UTC)


def select_data(
    inventory: Iterable[dict[str, Any]],
    *,
    naming_pattern: str,
    recency_months: int,
    min_images_per_class: int,
    target_dataset_min: int,
    target_dataset_max: int,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return a deterministic, balanced manifest from an S3 inventory.

    Records require ``key`` and ``last_modified``. Labels and confidence are parsed
    from the filename rather than trusted from an external annotation field.
    """
    if min_images_per_class < 1 or target_dataset_min < 1 or target_dataset_max < target_dataset_min:
        raise ValueError("Invalid dataset-size bounds")
    reference_time = (now or datetime.now(UTC)).astimezone(UTC)
    # A fixed 30-day month keeps local and Processing-job selection deterministic.
    cutoff = reference_time - timedelta(days=recency_months * 30)
    selected: list[dict[str, Any]] = []
    for record in inventory:
        key = str(record["key"])
        if not fnmatch.fnmatch(Path(key).name, naming_pattern):
            continue
        modified = _parse_timestamp(record["last_modified"])
        if modified < cutoff:
            continue
        brand, label, confidence = parse_filename_metadata(key)
        selected.append(
            {
                "id": str(record.get("id", key)),
                "key": key,
                "label": label,
                "source_brand": brand,
                "source_confidence": confidence,
                "last_modified": modified.isoformat(),
            }
        )

    selected.sort(key=lambda item: (item["label"], item["key"]))
    counts = Counter(item["label"] for item in selected)
    eligible_labels = {label for label, count in counts.items() if count >= min_images_per_class}
    curated = [item for item in selected if item["label"] in eligible_labels]
    if len(curated) < target_dataset_min:
        raise ValueError(
            f"Curated dataset has {len(curated)} images, below required minimum {target_dataset_min}"
        )
    return curated[:target_dataset_max]


def write_manifest(records: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a curated brand manifest from a JSON inventory")
    parser.add_argument("--input", default="/opt/ml/processing/input/inventory.json")
    parser.add_argument("--output", default="/opt/ml/processing/output/curated_manifest.json")
    parser.add_argument("--naming-pattern", required=True)
    parser.add_argument("--recency-months", type=int, default=6)
    parser.add_argument("--min-images-per-class", type=int, default=500)
    parser.add_argument("--target-min", type=int, default=6000)
    parser.add_argument("--target-max", type=int, default=50000)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Relax class/dataset floors to 1 for a small real-image end-to-end run.",
    )
    args = parser.parse_args()
    if args.smoke:
        args.min_images_per_class = min(args.min_images_per_class, 1)
        args.target_min = min(args.target_min, 1)
    inventory = json.loads(Path(args.input).read_text(encoding="utf-8"))
    manifest = select_data(
        inventory,
        naming_pattern=args.naming_pattern,
        recency_months=args.recency_months,
        min_images_per_class=args.min_images_per_class,
        target_dataset_min=args.target_min,
        target_dataset_max=args.target_max,
    )
    write_manifest(manifest, args.output)


if __name__ == "__main__":
    main()

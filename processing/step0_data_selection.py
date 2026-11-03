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

_METADATA_PATTERN = re.compile(r"^(?P<brand>[^_]+)_(?P<label>.+)_(?P<confidence>\d+(?:\.\d+)?)$")


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
    if (
        min_images_per_class < 1
        or target_dataset_min < 1
        or target_dataset_max < target_dataset_min
    ):
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
            f"Curated dataset has {len(curated)} images, "
            f"below required minimum {target_dataset_min}"
        )

    # Apply the size cap as a stratified per-class quota rather than a prefix
    # cut, so that a single attacker-chosen label cannot evict every other
    # class from the curated manifest. See CWE-841.
    if not eligible_labels:
        truncated: list[dict[str, Any]] = []
    elif len(curated) <= target_dataset_max:
        truncated = curated
    else:
        num_labels = len(eligible_labels)
        per_class_quota = target_dataset_max // num_labels
        if per_class_quota < min_images_per_class:
            # The cap cannot admit min_images_per_class records for every
            # eligible class simultaneously; refuse rather than silently
            # dropping a class.
            raise ValueError(
                f"target_dataset_max={target_dataset_max} too small to admit "
                f"min_images_per_class={min_images_per_class} across "
                f"{num_labels} eligible classes"
            )
        by_label: dict[str, list[dict[str, Any]]] = {}
        for item in curated:
            by_label.setdefault(item["label"], []).append(item)
        # Deterministic label order (sorted) keeps selection reproducible
        # across runs while preventing any one label from monopolising the cap.
        label_order = sorted(by_label)
        quotas = {label: min(per_class_quota, len(by_label[label])) for label in label_order}
        # Distribute any remaining capacity round-robin across labels that
        # still have records, so we fill up to target_dataset_max when possible.
        remaining = target_dataset_max - sum(quotas.values())
        while remaining > 0:
            progressed = False
            for label in label_order:
                if remaining <= 0:
                    break
                if quotas[label] < len(by_label[label]):
                    quotas[label] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break
        truncated = [
            item
            for label in label_order
            for item in by_label[label][: quotas[label]]
        ]

    # Re-validate the balance invariant on the actually-returned manifest.
    final_counts = Counter(item["label"] for item in truncated)
    expected_label_count = len(eligible_labels)
    if len(final_counts) < expected_label_count:
        missing = sorted(eligible_labels - set(final_counts))
        raise ValueError(
            f"Curated dataset lost classes after size-cap truncation: {missing}"
        )
    deficient = {
        label: count
        for label, count in final_counts.items()
        if count < min_images_per_class
    }
    if deficient:
        raise ValueError(
            f"Curated dataset classes below min_images_per_class after truncation: {deficient}"
        )
    if len(truncated) < target_dataset_min:
        raise ValueError(
            f"Curated dataset has {len(truncated)} images after per-class quota, "
            f"below required minimum {target_dataset_min}"
        )
    return truncated


def write_manifest(records: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a curated brand manifest from a JSON inventory"
    )
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

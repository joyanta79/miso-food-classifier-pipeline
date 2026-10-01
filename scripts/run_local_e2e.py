"""Local end-to-end driver for the data-cleaning path (Steps 0->1->2->3).

Runs the real processing modules on real on-disk images with the EfficientNetV2
backbone (ImageNet weights) when TensorFlow is available, proving the pipeline
end to end without any AWS resources. Injects two deliberate mislabels so Step 3
CleanLab detection can be demonstrated.

Usage:
    python scripts/run_local_e2e.py --image-dir tests/real_data/french_fries
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from processing.step0_data_selection import select_data  # noqa: E402
from processing.step1_extract_embeddings import extract_embeddings  # noqa: E402
from processing.step2_kfold_predict import kfold_predict  # noqa: E402
from processing.step3_cleanlab_filter import filter_label_issues  # noqa: E402


def build_inventory(image_dir: Path) -> list[dict[str, str]]:
    """Turn on-disk images into a Step 0 inventory (recent last_modified)."""
    now = datetime.now(UTC).isoformat()
    inventory = []
    for path in sorted(image_dir.glob("*.jpg")):
        inventory.append({"key": path.name, "id": path.name, "last_modified": now})
    return inventory


def inject_mislabels(manifest: list[dict], count: int) -> list[str]:
    """Swap labels on a few records to seed CleanLab detection; return their ids."""
    labels = sorted({r["label"] for r in manifest})
    if len(labels) < 2:
        return []
    flip = {labels[0]: labels[1], labels[1]: labels[0]}
    corrupted: list[str] = []
    for record in manifest:
        if record["label"] in flip and len(corrupted) < count:
            record["true_label"] = record["label"]
            record["label"] = flip[record["label"]]
            corrupted.append(record["id"])
    return corrupted


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Steps 0-3 locally on real images")
    parser.add_argument("--image-dir", default="tests/real_data/french_fries")
    parser.add_argument("--mislabels", type=int, default=3)
    parser.add_argument("--no-real-backbone", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    image_dir = (PROJECT_ROOT / args.image_dir).resolve()
    inventory = build_inventory(image_dir)
    print(f"[step0] inventory: {len(inventory)} images from {image_dir}")

    manifest = select_data(
        inventory,
        naming_pattern="wc_*",
        recency_months=6,
        min_images_per_class=1,   # smoke bounds for a small real dataset
        target_dataset_min=1,
        target_dataset_max=50000,
    )
    print(f"[step0] curated manifest: {len(manifest)} images")
    label_counts: dict[str, int] = {}
    for record in manifest:
        label_counts[record["label"]] = label_counts.get(record["label"], 0) + 1
    print(f"[step0] class counts: {label_counts}")

    corrupted = inject_mislabels(manifest, args.mislabels)
    print(f"[step0] injected {len(corrupted)} deliberate mislabels: {corrupted}")

    embeddings, mapping = extract_embeddings(
        manifest,
        image_root=image_dir,
        use_real_backbone=not args.no_real_backbone,
        backbone_weights="imagenet",
    )
    backend = mapping[0]["embedding_backend"] if mapping else "n/a"
    print(f"[step1] embeddings: shape={embeddings.shape} backend={backend} kept={len(mapping)}")

    probabilities, classes = kfold_predict(
        embeddings, [r["label"] for r in mapping], n_folds=5, classifier="xgboost"
    )
    print(f"[step2] OOS probabilities: shape={probabilities.shape} classes={classes}")

    flagged, cleaned, report = filter_label_issues(
        probabilities, mapping, classes, flag_threshold_fraction=0.05
    )
    print(f"[step3] flagged={report['flagged_images']} / {report['total_images']} "
          f"(fraction={report['flagged_fraction']:.3f}); cleaned={len(cleaned)}")

    flagged_ids = {f["id"] for f in flagged}
    caught = [c for c in corrupted if c in flagged_ids]
    recall = len(caught) / len(corrupted) if corrupted else 1.0
    print(f"[step3] CleanLab caught {len(caught)}/{len(corrupted)} injected mislabels "
          f"(recall={recall:.0%})")
    for f in flagged:
        print(f"        - {f['id']}: labeled={f['label']} predicted={f['predicted_label']} "
              f"reason={f['reason']} conf={f['confidence']:.2f}")

    result = {
        "images": len(manifest),
        "class_counts": label_counts,
        "embedding_backend": backend,
        "embedding_shape": list(embeddings.shape),
        "flagged_fraction": report["flagged_fraction"],
        "injected_mislabels": corrupted,
        "caught_mislabels": caught,
        "cleanlab_recall": recall,
        "classes": classes,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"[done] wrote {args.output}")
    print("[done] local end-to-end (Steps 0-3) complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""In-container runner for the Miso data-cleaning path (Steps 0->3).

Executed BY a SageMaker Processing job. Reads real images from
/opt/ml/processing/input/images, builds the Step 0 inventory, runs the real
EfficientNetV2 embedding path (Step 1), K-fold OOS prediction (Step 2), and
CleanLab filtering (Step 3), and writes results to /opt/ml/processing/output.

Injects a few deliberate mislabels so Step 3 detection can be demonstrated on
the smoke dataset, exactly as the local end-to-end driver does.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path

IN = Path("/opt/ml/processing/input")
CODE = Path("/opt/ml/processing/input/code")
OUT = Path("/opt/ml/processing/output")


def _pip(*pkgs: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-input", *pkgs],
        check=False,
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # Runtime deps not guaranteed in the base DLC image.
    _pip("scikit-learn", "xgboost", "cleanlab", "pillow")

    sys.path.insert(0, str(CODE))
    from processing.step0_data_selection import select_data
    from processing.step1_extract_embeddings import extract_embeddings
    from processing.step2_kfold_predict import kfold_predict
    from processing.step3_cleanlab_filter import filter_label_issues

    image_dir = IN / "images"
    now = datetime.now(UTC).isoformat()
    inventory = [
        {"key": p.name, "id": p.name, "last_modified": now}
        for p in sorted(image_dir.glob("*.jpg"))
    ]
    print(f"[step0] inventory: {len(inventory)} images", flush=True)

    manifest = select_data(
        inventory,
        naming_pattern="wc_*",
        recency_months=6,
        min_images_per_class=1,
        target_dataset_min=1,
        target_dataset_max=50000,
    )
    counts: dict[str, int] = {}
    for r in manifest:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print(f"[step0] curated {len(manifest)} images; classes={counts}", flush=True)

    # Inject deliberate mislabels for the CleanLab proof.
    labels = sorted(counts)
    corrupted: list[str] = []
    if len(labels) >= 2:
        flip = {labels[0]: labels[1], labels[1]: labels[0]}
        for r in manifest:
            if r["label"] in flip and len(corrupted) < 3:
                r["true_label"] = r["label"]
                r["label"] = flip[r["label"]]
                corrupted.append(r["id"])
    print(f"[step0] injected mislabels: {corrupted}", flush=True)

    embeddings, mapping = extract_embeddings(
        manifest, image_root=image_dir, use_real_backbone=True, backbone_weights="imagenet"
    )
    backend = mapping[0]["embedding_backend"] if mapping else "n/a"
    print(f"[step1] embeddings shape={embeddings.shape} backend={backend}", flush=True)

    probs, classes = kfold_predict(
        embeddings, [r["label"] for r in mapping], n_folds=5, classifier="xgboost"
    )
    print(f"[step2] OOS probs shape={probs.shape} classes={classes}", flush=True)

    flagged, cleaned, report = filter_label_issues(
        probs, mapping, classes, flag_threshold_fraction=0.05
    )
    caught = [c for c in corrupted if c in {f["id"] for f in flagged}]
    print(
        f"[step3] flagged={report['flagged_images']}/{report['total_images']} "
        f"cleaned={len(cleaned)} caught={len(caught)}/{len(corrupted)}",
        flush=True,
    )

    (OUT / "curated_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "cleaned_dataset.json").write_text(json.dumps(cleaned, indent=2) + "\n")
    (OUT / "flagged_images.json").write_text(json.dumps(flagged, indent=2) + "\n")
    (OUT / "cleanlab_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "smoke_result.json").write_text(
        json.dumps(
            {
                "images": len(manifest),
                "class_counts": counts,
                "embedding_backend": backend,
                "embedding_shape": list(embeddings.shape),
                "classes": classes,
                "flagged_fraction": report["flagged_fraction"],
                "injected_mislabels": corrupted,
                "caught_mislabels": caught,
                "cleanlab_recall": (len(caught) / len(corrupted)) if corrupted else 1.0,
            },
            indent=2,
        )
        + "\n"
    )
    print("[done] data-cleaning path complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

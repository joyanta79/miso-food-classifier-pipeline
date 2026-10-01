"""Training-step entry point: TrainBrandClassifier.

The SageMaker Training job runs this. It receives:
  * channel ``images``   -> the raw brand images
  * channel ``dataset``  -> cleaned_dataset.json from the quality step
  * channel ``code``     -> the training package (this file + train.py + config)

It arranges the cleaned images into ``train/<class>/`` and ``validation/<class>/``
directories (80/20 split, deterministic), then calls the proven train_model().
The model.keras + class_names.json land in SM_MODEL_DIR, which SageMaker packages
into model.tar.gz.

Python 3.10 compatible.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/opt/ml/input/data/pkgcode")

from training.train import train_model  # noqa: E402


def _prep_class_dirs(
    images_root: Path, cleaned: list[dict], work: Path, val_fraction: float = 0.2
) -> None:
    by_class: dict[str, list[dict]] = {}
    for record in cleaned:
        by_class.setdefault(record["label"], []).append(record)
    train_root = work / "train"
    val_root = work / "validation"
    for label, records in by_class.items():
        records = sorted(records, key=lambda r: r["image_path"])
        n_val = max(1, int(len(records) * val_fraction))
        val = records[:n_val]
        train = records[n_val:] or records  # never leave train empty
        for split_root, split in ((train_root, train), (val_root, val)):
            dest = split_root / label
            dest.mkdir(parents=True, exist_ok=True)
            for record in split:
                src = images_root / record["image_path"]
                if src.exists():
                    shutil.copy(src, dest / record["image_path"])
    counts = {label: len(records) for label, records in by_class.items()}
    print(f"[train_step] prepared class dirs: {counts}")


def main() -> None:
    images_root = Path(os.getenv("SM_CHANNEL_IMAGES", "/opt/ml/input/data/images"))
    dataset_root = Path(os.getenv("SM_CHANNEL_DATASET", "/opt/ml/input/data/dataset"))
    model_dir = Path(os.getenv("SM_MODEL_DIR", "/opt/ml/model"))
    checkpoint_dir = Path(os.getenv("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints"))
    output_dir = Path(os.getenv("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    brand_id = os.environ.get("BRAND_ID", "white-castle")

    cleaned = json.loads((dataset_root / "cleaned_dataset.json").read_text(encoding="utf-8"))

    work = Path("/opt/ml/input/prepared")
    _prep_class_dirs(images_root, cleaned, work)

    # Build the config inline so the runner does not depend on config.yaml being
    # staged; force the real ImageNet backbone and a small epoch count for smoke.
    epochs = int(os.environ.get("EPOCHS", "3"))
    batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    config = {
        "training": {
            "image_size": 224,
            "batch_size": batch_size,
            "epochs": epochs,
            "checkpoint_every_epochs": 5,
            "learning_rate": 0.001,
        },
        "embedding": {"backbone_source": "imagenet"},
        "local_mode": {"enabled": False, "use_imagenet_weights": True, "seed": 2026},
    }

    result = train_model(
        config=config,
        training_dir=work / "train",
        validation_dir=work / "validation",
        model_dir=model_dir,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        brand_id=brand_id,
        epochs=epochs,
    )
    print(f"[train_step] done: {json.dumps(result.get('metrics', {}))}")


if __name__ == "__main__":
    main()

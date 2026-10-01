"""Pipeline step: EvaluateModel.

Evaluates the trained Keras model (from the training step's model.tar.gz) on the
cleaned image set and emits an evaluation ``metrics.json`` consumed by the
EvaluateModelQualityGate:

    {
      "accuracy": <float>,
      "emergency_false_positive_rate": <float>,
      "latency_ms": <float>
    }

The model is a fine-tuned EfficientNetV2 classifier saved by train.py. This step
arranges the cleaned images by class, runs inference, and computes accuracy, an
emergency false-positive rate (treating the first class alphabetically as the
"safe/negative" reference), and mean per-image latency.

Python 3.10 compatible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, "/opt/ml/processing/input/pkgcode/pipeline/steps")
import _bootstrap  # noqa: E402

_bootstrap.ensure({"PIL": "pillow", "numpy": "numpy"})

import numpy as np


def _load_model(model_dir: Path):
    import tensorflow as tf

    # train.py saves either a SavedModel dir or model.keras; find it.
    for candidate in ("model.keras", "1", "model"):
        p = model_dir / candidate
        if p.exists():
            return tf.keras.models.load_model(p)
    # Fall back: first loadable entry.
    for p in sorted(model_dir.rglob("*")):
        if p.name.endswith(".keras") or (p.is_dir() and (p / "saved_model.pb").exists()):
            return tf.keras.models.load_model(p)
    raise FileNotFoundError(f"No loadable model found under {model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EvaluateModel step")
    parser.add_argument("--model", default="/opt/ml/processing/input/model")
    parser.add_argument("--images", default="/opt/ml/processing/input/images")
    parser.add_argument(
        "--cleaned", default="/opt/ml/processing/input/dataset/cleaned_dataset.json"
    )
    parser.add_argument("--metrics-out", default="/opt/ml/processing/output/metrics")
    args = parser.parse_args()

    import tensorflow as tf
    from PIL import Image

    image_size = int(os.environ.get("IMAGE_SIZE", "224"))
    model_dir = Path(args.model)

    # Unpack model.tar.gz if present.
    tarball = model_dir / "model.tar.gz"
    if tarball.exists():
        with tarfile.open(tarball) as handle:
            handle.extractall(model_dir)

    model = _load_model(model_dir)

    cleaned = json.loads(Path(args.cleaned).read_text(encoding="utf-8"))
    labels_sorted = sorted({record["label"] for record in cleaned})
    class_index = {label: i for i, label in enumerate(labels_sorted)}
    safe_label = labels_sorted[0]  # alphabetical first = reference "negative" class

    images_root = Path(args.images)
    correct = 0
    total = 0
    false_positives = 0
    negatives = 0
    latencies = []
    for record in cleaned:
        path = images_root / record["image_path"]
        if not path.exists():
            continue
        with Image.open(path) as handle:
            arr = np.asarray(
                handle.convert("RGB").resize((image_size, image_size)), dtype=np.float32
            )
        x = tf.keras.applications.efficientnet_v2.preprocess_input(arr[np.newaxis, ...])
        started = time.perf_counter()
        probs = model.predict(x, verbose=0)[0]
        latencies.append((time.perf_counter() - started) * 1000)
        pred_idx = int(np.argmax(probs))
        true_idx = class_index[record["label"]]
        total += 1
        if pred_idx == true_idx:
            correct += 1
        # Emergency FP: a truly-safe image predicted as a non-safe class.
        if record["label"] == safe_label:
            negatives += 1
            if pred_idx != class_index[safe_label]:
                false_positives += 1

    accuracy = correct / total if total else 0.0
    emergency_fp_rate = false_positives / negatives if negatives else 0.0
    latency_ms = float(np.mean(latencies)) if latencies else 0.0

    metrics = {
        "accuracy": accuracy,
        "emergency_false_positive_rate": emergency_fp_rate,
        "latency_ms": latency_ms,
        "sample_count": total,
        "classes": labels_sorted,
    }
    metrics_out = Path(args.metrics_out)
    metrics_out.mkdir(parents=True, exist_ok=True)
    (metrics_out / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[evaluation] accuracy={accuracy:.3f} "
        f"emergency_fp_rate={emergency_fp_rate:.3f} latency_ms={latency_ms:.1f} n={total}"
    )


if __name__ == "__main__":
    main()

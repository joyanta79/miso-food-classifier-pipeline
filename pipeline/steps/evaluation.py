"""Pipeline step: EvaluateModel.

Evaluates the trained Keras model (from the training step's model.tar.gz) on the
cleaned image set and emits an evaluation ``metrics.json`` consumed by the
EvaluateModelQualityGate.

The emergency false-positive rate is the hazardous direction defined by
processing/step6_evaluate.py and processing/step7_register_and_baseline.py:
truly-``not_done`` images predicted as ``done``. The reference class names are
taken from CLI arguments (``--done-label`` / ``--not-done-label``) or the
``DONE_LABEL`` / ``NOT_DONE_LABEL`` environment variables, defaulting to
``done`` / ``not_done`` -- never from alphabetical order.

The step fails closed: if any configured image file is missing, if the evaluated
sample count or the per-class counts fall below the configured floors, or if the
``not_done`` reference class contributes no evaluated samples, the step writes a
``metrics.json`` containing values the EvaluateModelQualityGate must reject
(``accuracy=0.0``, ``emergency_false_positive_rate=1.0``) and exits non-zero so
the SageMaker pipeline run fails before the gate is consulted.

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


# Sentinel metric values guaranteed to fail the EvaluateModelQualityGate
# (accuracy >= AccuracyThreshold and emergency_false_positive_rate
# <= FpRateEmergencyThreshold). Written whenever the step aborts fail-closed.
_FAIL_ACCURACY = 0.0
_FAIL_FP_RATE = 1.0


def _write_metrics(metrics_out: Path, metrics: dict) -> None:
    metrics_out.mkdir(parents=True, exist_ok=True)
    (metrics_out / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


def _abort(metrics_out: Path, reason: str, extra: dict) -> None:
    """Write a gate-failing metrics.json and exit non-zero."""
    metrics = {
        "accuracy": _FAIL_ACCURACY,
        "emergency_false_positive_rate": _FAIL_FP_RATE,
        "latency_ms": 0.0,
        "sample_count": int(extra.get("sample_count", 0)),
        "not_done_sample_count": int(extra.get("not_done_sample_count", 0)),
        "done_sample_count": int(extra.get("done_sample_count", 0)),
        "missing_images": int(extra.get("missing_images", 0)),
        "classes": list(extra.get("classes", [])),
        "abort_reason": reason,
    }
    _write_metrics(metrics_out, metrics)
    print(f"[evaluation] ABORT: {reason}", file=sys.stderr)
    sys.exit(1)


def _positive_int(value, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
    parser.add_argument(
        "--done-label",
        default=os.environ.get("DONE_LABEL", "done"),
        help="Name of the done/safe/positive class.",
    )
    parser.add_argument(
        "--not-done-label",
        default=os.environ.get("NOT_DONE_LABEL", "not_done"),
        help="Name of the not_done/unsafe class (hazardous prediction direction).",
    )
    parser.add_argument(
        "--min-sample-count",
        type=int,
        default=_positive_int(os.environ.get("MIN_SAMPLE_COUNT"), 1),
        help="Minimum evaluated samples; below this the step fails closed.",
    )
    parser.add_argument(
        "--min-per-class-count",
        type=int,
        default=_positive_int(os.environ.get("MIN_PER_CLASS_COUNT"), 1),
        help="Minimum evaluated samples per labelled class; below this the step fails closed.",
    )
    args = parser.parse_args()

    metrics_out = Path(args.metrics_out)
    done_label = args.done_label
    not_done_label = args.not_done_label

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

    # Explicit label semantics: refuse to proceed if the hazardous reference
    # class or the positive class is missing from the dataset. Alphabetical
    # ordering is NOT used to pick the safety direction.
    if not_done_label not in class_index:
        _abort(
            metrics_out,
            (
                f"not_done label {not_done_label!r} is not present in dataset classes "
                f"{labels_sorted!r}; cannot measure the hazardous false-positive direction."
            ),
            {"classes": labels_sorted},
        )
    if done_label not in class_index:
        _abort(
            metrics_out,
            (
                f"done label {done_label!r} is not present in dataset classes "
                f"{labels_sorted!r}."
            ),
            {"classes": labels_sorted},
        )

    images_root = Path(args.images)
    correct = 0
    total = 0
    false_positives = 0  # truly not_done predicted as done (hazardous direction)
    not_done_evaluated = 0
    done_evaluated = 0
    missing_images = 0
    latencies: list[float] = []
    for record in cleaned:
        path = images_root / record["image_path"]
        if not path.exists():
            # Count missing files instead of silently dropping records; the step
            # aborts below when missing_images > 0.
            missing_images += 1
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
        true_label = record["label"]
        if true_label not in class_index:
            _abort(
                metrics_out,
                f"record label {true_label!r} not in known classes {labels_sorted!r}",
                {
                    "classes": labels_sorted,
                    "sample_count": total,
                    "not_done_sample_count": not_done_evaluated,
                    "done_sample_count": done_evaluated,
                    "missing_images": missing_images,
                },
            )
        true_idx = class_index[true_label]
        total += 1
        if pred_idx == true_idx:
            correct += 1
        # Emergency FP: truly-not_done image predicted as done (hazardous direction).
        if true_label == not_done_label:
            not_done_evaluated += 1
            if pred_idx == class_index[done_label]:
                false_positives += 1
        elif true_label == done_label:
            done_evaluated += 1

    extras = {
        "classes": labels_sorted,
        "sample_count": total,
        "not_done_sample_count": not_done_evaluated,
        "done_sample_count": done_evaluated,
        "missing_images": missing_images,
    }

    # Fail closed on dataset-shrinking manipulation or vacuous evaluation.
    if missing_images > 0:
        _abort(
            metrics_out,
            (
                f"{missing_images} cleaned_dataset records had no matching image under the "
                f"images channel; refusing to publish metrics computed over a filtered subset."
            ),
            extras,
        )
    if total < args.min_sample_count:
        _abort(
            metrics_out,
            (
                f"evaluated sample_count={total} is below the configured floor "
                f"min_sample_count={args.min_sample_count}."
            ),
            extras,
        )
    if not_done_evaluated < args.min_per_class_count:
        _abort(
            metrics_out,
            (
                f"not_done evaluated samples={not_done_evaluated} is below "
                f"min_per_class_count={args.min_per_class_count}; cannot produce an "
                f"emergency_false_positive_rate without hazardous-class samples."
            ),
            extras,
        )
    if done_evaluated < args.min_per_class_count:
        _abort(
            metrics_out,
            (
                f"done evaluated samples={done_evaluated} is below "
                f"min_per_class_count={args.min_per_class_count}."
            ),
            extras,
        )

    accuracy = correct / total
    emergency_fp_rate = false_positives / not_done_evaluated
    latency_ms = float(np.mean(latencies)) if latencies else 0.0

    metrics = {
        "accuracy": accuracy,
        "emergency_false_positive_rate": emergency_fp_rate,
        "latency_ms": latency_ms,
        "sample_count": total,
        "not_done_sample_count": not_done_evaluated,
        "done_sample_count": done_evaluated,
        "missing_images": missing_images,
        "classes": labels_sorted,
    }
    _write_metrics(metrics_out, metrics)
    print(
        f"[evaluation] accuracy={accuracy:.3f} "
        f"emergency_fp_rate={emergency_fp_rate:.3f} latency_ms={latency_ms:.1f} "
        f"n={total} not_done_n={not_done_evaluated} done_n={done_evaluated} "
        f"missing_images={missing_images}"
    )


if __name__ == "__main__":
    main()

"""Step 6: evaluate candidate predictions in a SageMaker Processing job.

The metric contract published here matches ``pipeline/steps/evaluation.py`` so
the EvaluateModelQualityGate (see ``pipeline/pipeline.py`` and
``pipeline/deploy_pipeline.py``) can read the same ``metrics.json`` regardless
of which evaluation implementation produced it. The hazardous false-positive
direction is published as ``emergency_false_positive_rate`` and the step fails
closed when the hazardous reference class is missing or has fewer evaluated
samples than ``evaluation.min_per_class_count`` -- in that case we emit the
gate-failing sentinel (``accuracy=0.0``, ``emergency_false_positive_rate=1.0``)
with an ``abort_reason`` and raise so callers / CLI runs exit non-zero.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

DEFAULT_INPUT_DIR = Path("/opt/ml/processing/input")
DEFAULT_OUTPUT_DIR = Path("/opt/ml/processing/evaluation")

# Sentinel metric values guaranteed to fail the EvaluateModelQualityGate
# (accuracy >= AccuracyThreshold and emergency_false_positive_rate
# <= FpRateEmergencyThreshold). Written whenever evaluation aborts fail-closed.
_FAIL_ACCURACY = 0.0
_FAIL_FP_RATE = 1.0


class EvaluationAbort(ValueError):
    """Raised when evaluation must fail closed (hazardous class missing, etc.)."""


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_predictions(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not {"label", "prediction"} <= record.keys():
                raise ValueError(
                    f"Prediction line {line_number} requires label and prediction fields."
                )
            records.append(record)
    if not records:
        raise ValueError("Evaluation requires at least one prediction.")
    return records


def _abort_metrics(
    reason: str,
    *,
    sample_count: int,
    not_done_sample_count: int,
    done_sample_count: int,
    missing_images: int,
    classes: list[str],
    label_distribution: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Metrics dict guaranteed to fail the EvaluateModelQualityGate."""
    return {
        "accuracy": _FAIL_ACCURACY,
        "emergency_false_positive_rate": _FAIL_FP_RATE,
        "latency_ms": 0.0,
        "sample_count": int(sample_count),
        "not_done_sample_count": int(not_done_sample_count),
        "done_sample_count": int(done_sample_count),
        "missing_images": int(missing_images),
        "classes": list(classes),
        "abort_reason": reason,
        # Legacy aliases retained for backward compatibility.
        "done_when_not_done_fp_rate": _FAIL_FP_RATE,
        "fp_rate_done_when_not_done": _FAIL_FP_RATE,
        "done_when_not_done_false_positives": 0,
        "label_distribution": dict(label_distribution or {}),
        "confidence_summary": {"count": 0, "mean": None, "min": None, "max": None},
    }


def calculate_metrics(
    predictions: Iterable[dict[str, Any]],
    done_label: str = "done",
    not_done_label: str = "not_done",
    *,
    min_per_class_count: int = 1,
) -> dict[str, Any]:
    records = list(predictions)
    if not records:
        raise EvaluationAbort("Evaluation requires at least one prediction.")

    label_distribution = dict(
        sorted(Counter(str(record["label"]) for record in records).items())
    )
    classes = sorted(label_distribution.keys())

    actual_not_done = [record for record in records if record["label"] == not_done_label]
    actual_done = [record for record in records if record["label"] == done_label]
    not_done_sample_count = len(actual_not_done)
    done_sample_count = len(actual_done)

    # Fail closed: the hazardous reference class must appear in the labelled
    # samples. Without it we cannot measure the hazardous false-positive
    # direction and must not publish a null measurement that downstream
    # consumers (step7 baseline, Model Monitor) would persist as a real value.
    if not_done_sample_count == 0:
        raise EvaluationAbort(
            f"not_done label {not_done_label!r} is not present in evaluated labels "
            f"{classes!r}; cannot measure the hazardous false-positive direction."
        )
    if not_done_sample_count < int(min_per_class_count):
        raise EvaluationAbort(
            f"not_done evaluated samples={not_done_sample_count} is below "
            f"min_per_class_count={int(min_per_class_count)}; cannot produce an "
            f"emergency_false_positive_rate without enough hazardous-class samples."
        )
    if done_sample_count < int(min_per_class_count):
        raise EvaluationAbort(
            f"done evaluated samples={done_sample_count} is below "
            f"min_per_class_count={int(min_per_class_count)}."
        )

    false_positives = [record for record in actual_not_done if record["prediction"] == done_label]
    confidences = [
        float(record["confidence"]) for record in records if record.get("confidence") is not None
    ]
    fp_rate = len(false_positives) / not_done_sample_count
    accuracy = sum(record["label"] == record["prediction"] for record in records) / len(records)
    return {
        # Canonical names consumed by EvaluateModelQualityGate.
        "accuracy": accuracy,
        "emergency_false_positive_rate": fp_rate,
        "latency_ms": 0.0,
        "sample_count": len(records),
        "not_done_sample_count": not_done_sample_count,
        "done_sample_count": done_sample_count,
        "missing_images": 0,
        "classes": classes,
        # Legacy aliases retained for backward compatibility.
        "done_when_not_done_fp_rate": fp_rate,
        "fp_rate_done_when_not_done": fp_rate,
        "done_when_not_done_false_positives": len(false_positives),
        "label_distribution": label_distribution,
        "confidence_summary": {
            "count": len(confidences),
            "mean": sum(confidences) / len(confidences) if confidences else None,
            "min": min(confidences) if confidences else None,
            "max": max(confidences) if confidences else None,
        },
    }


def _build_report(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    policy = config.get("evaluation", {}) if config else {}
    accuracy_threshold = float(policy.get("accuracy_threshold", 0.97))
    fp_threshold = float(policy.get("fp_rate_emergency_threshold", 0.05))
    fp_rate = metrics.get("emergency_false_positive_rate")
    accuracy = metrics.get("accuracy")
    quality_gate = (
        "passed"
        if accuracy is not None
        and fp_rate is not None
        and "abort_reason" not in metrics
        and accuracy >= accuracy_threshold
        and fp_rate <= fp_threshold
        else "failed"
    )
    return {
        "schema_version": "1.0",
        "quality_gate": quality_gate,
        "thresholds": {
            "accuracy": accuracy_threshold,
            "emergency_false_positive_rate": fp_threshold,
            # Legacy alias.
            "done_when_not_done_fp_rate": fp_threshold,
        },
        "metrics": metrics,
        "model_metrics": {
            "classification_metrics": {
                "accuracy": {"value": accuracy, "standard_deviation": None},
                "emergency_false_positive_rate": {
                    "value": fp_rate,
                    "standard_deviation": None,
                },
                # Legacy alias.
                "done_when_not_done_fp_rate": {"value": fp_rate, "standard_deviation": None},
            }
        },
    }


def evaluate(
    predictions: Iterable[dict[str, Any]],
    config: dict[str, Any],
    *,
    done_label: str = "done",
    not_done_label: str = "not_done",
) -> dict[str, Any]:
    policy = (config or {}).get("evaluation", {})
    min_per_class_count = int(policy.get("min_per_class_count", 1))
    records = list(predictions)
    try:
        metrics = calculate_metrics(
            records,
            done_label,
            not_done_label,
            min_per_class_count=min_per_class_count,
        )
    except EvaluationAbort as exc:
        label_distribution = dict(
            sorted(Counter(str(record["label"]) for record in records).items())
        )
        classes = sorted(label_distribution.keys())
        not_done_sample_count = sum(1 for r in records if r["label"] == not_done_label)
        done_sample_count = sum(1 for r in records if r["label"] == done_label)
        metrics = _abort_metrics(
            str(exc),
            sample_count=len(records),
            not_done_sample_count=not_done_sample_count,
            done_sample_count=done_sample_count,
            missing_images=0,
            classes=classes,
            label_distribution=label_distribution,
        )
    return _build_report(metrics, config or {"evaluation": {}})


def run(
    labels_path: str | Path,
    probabilities_path: str | Path,
    classes_path: str | Path,
    output_path: str | Path,
    done_label: str = "done",
    config: dict[str, Any] | None = None,
    not_done_label: str = "not_done",
) -> dict[str, Any]:
    """Compatibility entry point for pipeline channels with NumPy probabilities."""
    import numpy as np

    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    classes = json.loads(Path(classes_path).read_text(encoding="utf-8"))
    probabilities = np.load(probabilities_path)
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probability rows must have the same length.")

    class_strs = [str(c) for c in classes]
    directory = Path(output_path)
    directory.mkdir(parents=True, exist_ok=True)

    def _emit_abort(reason: str) -> None:
        label_distribution = dict(
            sorted(Counter(str(label) for label in labels).items())
        )
        label_classes = sorted(set(class_strs) | set(label_distribution.keys()))
        metrics = _abort_metrics(
            reason,
            sample_count=len(labels),
            not_done_sample_count=sum(1 for label in labels if str(label) == not_done_label),
            done_sample_count=sum(1 for label in labels if str(label) == done_label),
            missing_images=0,
            classes=label_classes,
            label_distribution=label_distribution,
        )
        aborted = _build_report(metrics, config or {"evaluation": {}})
        (directory / "evaluation.json").write_text(
            json.dumps(aborted, indent=2), encoding="utf-8"
        )
        (directory / "metrics.json").write_text(
            json.dumps(aborted["metrics"], indent=2), encoding="utf-8"
        )
        raise EvaluationAbort(reason)

    # Fail closed up front if the configured hazardous / safe reference
    # classes are not present in the model's class vocabulary. Without this
    # check an operator-supplied --not-done-label would be silently dropped
    # when the dataset uses a different spelling (e.g. 'raw' vs 'not_done').
    if not_done_label not in class_strs:
        _emit_abort(
            f"not_done label {not_done_label!r} is not present in model classes "
            f"{class_strs!r}; cannot measure the hazardous false-positive direction."
        )
    if done_label not in class_strs:
        _emit_abort(
            f"done label {done_label!r} is not present in model classes {class_strs!r}."
        )

    started = time.perf_counter()
    predictions = [
        {
            "label": str(label),
            "prediction": str(classes[int(row.argmax())]),
            "confidence": float(row.max()),
        }
        for label, row in zip(labels, probabilities)
    ]
    report = evaluate(
        predictions,
        config or {"evaluation": {}},
        done_label=done_label,
        not_done_label=not_done_label,
    )
    report["metrics"]["latency_ms"] = (time.perf_counter() - started) * 1000
    aborted = "abort_reason" in report["metrics"]
    (directory / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (directory / "metrics.json").write_text(
        json.dumps(report["metrics"], indent=2), encoding="utf-8"
    )
    if aborted:
        raise EvaluationAbort(str(report["metrics"].get("abort_reason")))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a candidate food-classification model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--predictions")
    parser.add_argument("--labels")
    parser.add_argument("--probabilities")
    parser.add_argument("--classes")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--done-label", default="done")
    parser.add_argument("--not-done-label", default="not_done")
    return parser.parse_args()


def main(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    if args.predictions:
        report = evaluate(
            read_predictions(args.predictions),
            config,
            done_label=args.done_label,
            not_done_label=args.not_done_label,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (output_dir / "metrics.json").write_text(
            json.dumps(report["metrics"], indent=2), encoding="utf-8"
        )
        # Fail closed on aborts so the CLI exits non-zero and downstream
        # baselining never consumes a sentinel report as a real measurement.
        if "abort_reason" in report["metrics"]:
            raise EvaluationAbort(str(report["metrics"]["abort_reason"]))
        return report
    if not all((args.labels, args.probabilities, args.classes)):
        raise ValueError(
            "Provide --predictions or all of --labels, --probabilities, and --classes."
        )
    return run(
        args.labels,
        args.probabilities,
        args.classes,
        args.output_dir,
        args.done_label,
        config,
        not_done_label=args.not_done_label,
    )


if __name__ == "__main__":
    print(json.dumps(main(parse_args()), indent=2))

"""Step 6: evaluate candidate predictions in a SageMaker Processing job."""

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


def calculate_metrics(
    predictions: Iterable[dict[str, Any]],
    done_label: str = "done",
    not_done_label: str = "not_done",
) -> dict[str, Any]:
    records = list(predictions)
    if not records:
        raise ValueError("Evaluation requires at least one prediction.")
    actual_not_done = [record for record in records if record["label"] == not_done_label]
    false_positives = [record for record in actual_not_done if record["prediction"] == done_label]
    confidences = [
        float(record["confidence"]) for record in records if record.get("confidence") is not None
    ]
    fp_rate = len(false_positives) / len(actual_not_done) if actual_not_done else None
    accuracy = sum(record["label"] == record["prediction"] for record in records) / len(records)
    return {
        "sample_count": len(records),
        "accuracy": accuracy,
        # Both spellings are retained: the shorter legacy pipeline property and
        # the explicit safety metric used in quality reports.
        "done_when_not_done_fp_rate": fp_rate,
        "fp_rate_done_when_not_done": fp_rate,
        "done_when_not_done_false_positives": len(false_positives),
        "not_done_sample_count": len(actual_not_done),
        "label_distribution": dict(
            sorted(Counter(str(record["label"]) for record in records).items())
        ),
        "confidence_summary": {
            "count": len(confidences),
            "mean": sum(confidences) / len(confidences) if confidences else None,
            "min": min(confidences) if confidences else None,
            "max": max(confidences) if confidences else None,
        },
    }


def evaluate(
    predictions: Iterable[dict[str, Any]],
    config: dict[str, Any],
    *,
    done_label: str = "done",
    not_done_label: str = "not_done",
) -> dict[str, Any]:
    metrics = calculate_metrics(predictions, done_label, not_done_label)
    policy = config.get("evaluation", {})
    accuracy_threshold = float(policy.get("accuracy_threshold", 0.97))
    fp_threshold = float(policy.get("fp_rate_emergency_threshold", 0.05))
    fp_rate = metrics["done_when_not_done_fp_rate"]
    quality_gate = (
        "passed"
        if metrics["accuracy"] >= accuracy_threshold
        and fp_rate is not None
        and fp_rate <= fp_threshold
        else "failed"
    )
    return {
        "schema_version": "1.0",
        "quality_gate": quality_gate,
        "thresholds": {"accuracy": accuracy_threshold, "done_when_not_done_fp_rate": fp_threshold},
        "metrics": metrics,
        "model_metrics": {
            "classification_metrics": {
                "accuracy": {"value": metrics["accuracy"], "standard_deviation": None},
                "done_when_not_done_fp_rate": {"value": fp_rate, "standard_deviation": None},
            }
        },
    }


def run(
    labels_path: str | Path,
    probabilities_path: str | Path,
    classes_path: str | Path,
    output_path: str | Path,
    done_label: str = "done",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility entry point for pipeline channels with NumPy probabilities."""
    import numpy as np

    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    classes = json.loads(Path(classes_path).read_text(encoding="utf-8"))
    probabilities = np.load(probabilities_path)
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probability rows must have the same length.")
    started = time.perf_counter()
    predictions = [
        {
            "label": str(label),
            "prediction": str(classes[int(row.argmax())]),
            "confidence": float(row.max()),
        }
        for label, row in zip(labels, probabilities)
    ]
    report = evaluate(predictions, config or {"evaluation": {}}, done_label=done_label)
    report["metrics"]["latency_ms"] = (time.perf_counter() - started) * 1000
    directory = Path(output_path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (directory / "metrics.json").write_text(
        json.dumps(report["metrics"], indent=2), encoding="utf-8"
    )
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
        return report
    if not all((args.labels, args.probabilities, args.classes)):
        raise ValueError(
            "Provide --predictions or all of --labels, --probabilities, and --classes."
        )
    return run(
        args.labels, args.probabilities, args.classes, args.output_dir, args.done_label, config
    )


if __name__ == "__main__":
    print(json.dumps(main(parse_args()), indent=2))

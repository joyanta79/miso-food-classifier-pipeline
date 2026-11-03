"""Step 3: identify likely annotation issues from OOS probabilities."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

# Provenance tags recorded in the gate report. Downstream gates compare against
# ``DETECTOR_CLEANLAB`` and MUST reject any report whose detector is not the
# real CleanLab detector -- the local heuristic is an additional signal only,
# never a substitute for the label-quality control.
DETECTOR_CLEANLAB = "cleanlab"
DETECTOR_FALLBACK = "fallback"


def _cleanlab_version() -> str:
    try:
        import cleanlab  # type: ignore[import-not-found]

        return str(getattr(cleanlab, "__version__", "unknown"))
    except ImportError:
        return "unavailable"


def _cleanlab_issue_mask(
    labels: np.ndarray, pred_probs: np.ndarray
) -> tuple[np.ndarray, str, str]:
    """Run the real CleanLab detector and OR in a deterministic consistency floor.

    Returns the issue mask alongside the detector name and version actually
    used. ImportError from CleanLab is NOT swallowed: the label-quality gate
    fails closed on a missing detector so the Processing job exits non-zero.
    """
    # Import at call time so missing/broken CleanLab aborts this step loudly
    # rather than silently producing a report backed only by the local floor.
    from cleanlab.filter import find_label_issues  # noqa: F401  (fail-closed import)

    issues = np.asarray(
        find_label_issues(labels=labels, pred_probs=pred_probs), dtype=bool
    )
    predicted = pred_probs.argmax(axis=1)
    confidence = pred_probs.max(axis=1)
    # A high-confidence disagreement is independently reproducible and is kept
    # as an *additional* signal on top of CleanLab, never as a substitute.
    issues |= (predicted != labels) & (confidence >= 0.50)
    return issues, DETECTOR_CLEANLAB, _cleanlab_version()


def filter_label_issues(
    pred_probs: np.ndarray,
    mapping: Iterable[dict[str, Any]],
    classes: Iterable[str],
    *,
    flag_threshold_fraction: float = 0.05,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float | int | str]]:
    """Return flagged records, clean records, and pipeline-gate report.

    ``flag_threshold_fraction`` must be in ``(0, 1]``; it is validated here and
    echoed into the report so downstream gates/steps that enforce the ceiling
    receive the exact value this function used. Enforcement of the ceiling on
    ``flagged_fraction`` is performed by the caller (the KFoldCleanlabQuality
    step and the CleanlabQualityGate), which have the authority to fail the run;
    this function remains a pure data-processing primitive.
    """
    try:
        threshold = float(flag_threshold_fraction)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"flag_threshold_fraction must be numeric, got {flag_threshold_fraction!r}"
        ) from error
    if not (0.0 < threshold <= 1.0):
        raise ValueError(
            f"flag_threshold_fraction must satisfy 0 < value <= 1, got {threshold!r}"
        )
    records = list(mapping)
    class_names = list(classes)
    probabilities = np.asarray(pred_probs, dtype=np.float64)
    if probabilities.shape != (len(records), len(class_names)):
        raise ValueError("Probability matrix must align with mapping and classes")
    index = {label: value for value, label in enumerate(class_names)}
    try:
        labels = np.asarray([index[str(record["label"])] for record in records], dtype=int)
    except KeyError as error:
        raise ValueError(f"Label absent from classes: {error.args[0]}") from error
    issue_mask, detector, detector_version = _cleanlab_issue_mask(labels, probabilities)
    near_duplicate_mask = np.asarray(
        [bool(record.get("near_duplicate", False)) for record in records], dtype=bool
    )
    issue_mask |= near_duplicate_mask
    flagged: list[dict[str, Any]] = []
    cleaned: list[dict[str, Any]] = []
    for row, record in enumerate(records):
        if issue_mask[row]:
            predicted = int(probabilities[row].argmax())
            if near_duplicate_mask[row]:
                reason = "near-duplicate"
            else:
                reason = "mislabel" if predicted != labels[row] else "ambiguous"
            flagged.append(
                {
                    "id": record["id"],
                    "label": record["label"],
                    "predicted_label": class_names[predicted],
                    "reason": reason,
                    "confidence": float(probabilities[row, predicted]),
                }
            )
        else:
            cleaned.append(record)
    report: dict[str, float | int | str] = {
        "total_images": len(records),
        "flagged_images": len(flagged),
        "flagged_fraction": len(flagged) / len(records) if records else 0.0,
        "flag_threshold_fraction": threshold,
        # Provenance: downstream CleanlabQualityGate rejects any report whose
        # detector is not DETECTOR_CLEANLAB. Operators reading metrics.json can
        # tell "CleanLab found few issues" from "CleanLab never ran".
        "detector": detector,
        "detector_version": detector_version,
    }
    return flagged, cleaned, report


def write_outputs(
    flagged: list[dict[str, Any]],
    cleaned: list[dict[str, Any]],
    report: dict[str, float | int | str],
    output_dir: str | Path,
) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("flagged_images.json", flagged),
        ("cleaned_dataset.json", cleaned),
        ("cleanlab_report.json", report),
    ):
        (directory / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter probable image-label issues")
    parser.add_argument("--pred-probs", default="/opt/ml/processing/input/pred_probs.npy")
    parser.add_argument("--mapping", default="/opt/ml/processing/input/embedding_mapping.json")
    parser.add_argument("--classes", default="/opt/ml/processing/input/classes.json")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--flag-threshold-fraction", type=float, default=0.05)
    args = parser.parse_args()
    flagged, cleaned, report = filter_label_issues(
        np.load(args.pred_probs),
        json.loads(Path(args.mapping).read_text(encoding="utf-8")),
        json.loads(Path(args.classes).read_text(encoding="utf-8")),
        flag_threshold_fraction=args.flag_threshold_fraction,
    )
    write_outputs(flagged, cleaned, report, args.output_dir)


if __name__ == "__main__":
    main()

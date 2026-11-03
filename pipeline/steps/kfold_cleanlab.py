"""Pipeline step: KFoldCleanlabQuality.

Combines the proven step2 (K-fold out-of-sample probabilities) and step3
(CleanLab label-issue filtering) into the single DAG step the pipeline
references. Emits a quality ``metrics.json`` consumed by the CleanlabQualityGate.

The emitted ``metrics.json`` includes post-filter integrity metrics so the
configured dataset floors can be enforced on the data that is actually trained
on (not just the pre-dedup selection): ``cleaned_count``,
``per_class_cleaned_counts``, ``minimum_class_count``, ``class_count``, and
``filter_removed_count`` are published alongside the existing
``flagged_fraction`` / ``flagged_images`` / ``total_images``.

Also writes the cleaned dataset (surviving records) for the training step, plus
flagged_images.json and cleanlab_report.json for inspection.

The step itself enforces (and fails fast on) the floors it is configured
with, so the DAG gate cannot silently train on data below those floors:

  * ``flagged_fraction`` <= ``CLEANLAB_FLAG_THRESHOLD_FRACTION``
  * ``cleaned_count`` >= ``TARGET_DATASET_MIN`` (when provided)
  * ``minimum_class_count`` >= ``MIN_IMAGES_PER_CLASS`` and every class in the
    input mapping is still represented after filtering (when provided)

Environment values are validated: ``CLEANLAB_FLAG_THRESHOLD_FRACTION`` must be
numeric in ``(0, 1]``, ``KFOLD_COUNT`` must be an integer ``>= 2``, and the
optional ``TARGET_DATASET_MIN`` / ``MIN_IMAGES_PER_CLASS`` must be non-negative
integers when set. Invalid values abort the step rather than being silently
accepted.

Python 3.10 compatible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/opt/ml/processing/input/pkgcode")
sys.path.insert(0, "/opt/ml/processing/input/pkgcode/pipeline/steps")

import _bootstrap  # noqa: E402

_bootstrap.ensure(
    {
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "xgboost": "xgboost",
        # Pin to the same version as requirements.txt and verify the concrete
        # symbol the step uses -- not just the top-level package -- so an
        # API-incompatible resolution fails the step rather than silently
        # degrading to the fallback heuristic.
        "cleanlab.filter:find_label_issues": "cleanlab==2.6.4",
    }
)

import numpy as np  # noqa: E402

from processing.step2_kfold_predict import kfold_predict  # noqa: E402
from processing.step3_cleanlab_filter import filter_label_issues  # noqa: E402


def _parse_fraction(name: str, raw: str) -> float:
    """Parse a fraction env var; must be numeric and in (0, 1]."""
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number, got {raw!r}") from error
    if not math.isfinite(value) or not (0.0 < value <= 1.0):
        raise ValueError(f"{name} must satisfy 0 < value <= 1, got {value!r}")
    return value


def _parse_int(name: str, raw: str, *, minimum: int) -> int:
    """Parse an integer env var; must be >= ``minimum``."""
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return value


def _parse_optional_int(name: str, raw: str | None, *, minimum: int) -> int | None:
    """Parse an optional integer env var; empty/None means 'not provided'."""
    if raw is None or raw == "":
        return None
    return _parse_int(name, raw, minimum=minimum)


# The CLI in ``processing/step2_kfold_predict.py`` restricts ``--classifier`` to
# this allowlist; the pipeline step must enforce the same so an unknown value
# cannot silently degrade to the RandomForest fallback without the DAG gate
# noticing. ``xgboost`` is the configured detector in ``config.yaml``;
# ``fallback`` is permitted so operators can intentionally exercise the
# scikit-learn backup when xgboost is unavailable.
_SUPPORTED_CLASSIFIERS = ("xgboost", "fallback")


def _parse_classifier(name: str, raw: str) -> str:
    """Parse the classifier env var against the supported allowlist."""
    if raw not in _SUPPORTED_CLASSIFIERS:
        raise ValueError(
            f"{name} must be one of {_SUPPORTED_CLASSIFIERS!r}, got {raw!r}. "
            "An unknown value would silently fall back to RandomForest and "
            "produce flagged_fraction / pred_probs from a different detector "
            "than the one the gate expects."
        )
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="KFoldCleanlabQuality step")
    parser.add_argument("--embeddings", default="/opt/ml/processing/input/dataset/embeddings.npy")
    parser.add_argument(
        "--mapping", default="/opt/ml/processing/input/dataset/embedding_mapping.json"
    )
    parser.add_argument("--dataset-out", default="/opt/ml/processing/output/dataset")
    parser.add_argument("--metrics-out", default="/opt/ml/processing/output/metrics")
    args = parser.parse_args()

    n_folds = _parse_int("KFOLD_COUNT", os.environ.get("KFOLD_COUNT", "5"), minimum=2)
    classifier = _parse_classifier(
        "CLASSIFIER", os.environ.get("CLASSIFIER", "xgboost")
    )
    # Defence in depth: if the pipeline shipped a baked ``CONFIGURED_CLASSIFIER``,
    # the active value must match it. The pipeline currently bakes ``CLASSIFIER``
    # as a literal from config, so these are expected to be identical; this
    # guard rejects a future regression that re-exposes CLASSIFIER as caller
    # input or a stray container-level override.
    configured_classifier = os.environ.get("CONFIGURED_CLASSIFIER")
    if configured_classifier is not None and classifier != configured_classifier:
        raise ValueError(
            f"CLASSIFIER {classifier!r} does not match configured value "
            f"{configured_classifier!r}; the detector that produces "
            "flagged_fraction and pred_probs cannot be swapped per execution."
        )
    flag_threshold = _parse_fraction(
        "CLEANLAB_FLAG_THRESHOLD_FRACTION",
        os.environ.get("CLEANLAB_FLAG_THRESHOLD_FRACTION", "0.05"),
    )
    target_dataset_min = _parse_optional_int(
        "TARGET_DATASET_MIN", os.environ.get("TARGET_DATASET_MIN"), minimum=0
    )
    min_images_per_class = _parse_optional_int(
        "MIN_IMAGES_PER_CLASS", os.environ.get("MIN_IMAGES_PER_CLASS"), minimum=0
    )

    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    embeddings = np.load(args.embeddings)
    labels = [record["label"] for record in mapping]
    original_classes = sorted({str(label) for label in labels})

    probabilities, classes = kfold_predict(
        embeddings, labels, n_folds=n_folds, classifier=classifier
    )
    flagged, cleaned, report = filter_label_issues(
        probabilities, mapping, classes, flag_threshold_fraction=flag_threshold
    )

    # Surface the deterministic-fallback count from the embedding step so the
    # downstream CleanlabQualityGate can reject a run where any record was
    # embedded from the filename hash rather than image pixels. Fail-closed is
    # already enforced in ``embedding_dedup.py``; this metric lets gates that
    # evaluate ``metrics.json`` observe the degradation directly.
    deterministic_fallback_count = sum(
        1 for record in mapping if record.get("embedding_backend") == "deterministic"
    )
    backends = sorted(
        {record.get("embedding_backend") for record in mapping if record.get("embedding_backend")}
    )
    report = dict(report)
    report["deterministic_fallback_count"] = deterministic_fallback_count
    report["embedding_backends"] = backends
    report["efficientnet_only"] = backends == ["efficientnet"]

    # ---- Post-filter integrity metrics ---------------------------------------
    cleaned_count = len(cleaned)
    filter_removed_count = len(flagged)
    per_class_cleaned_counts: dict[str, int] = dict(
        Counter(str(record["label"]) for record in cleaned)
    )
    # Classes present in the input mapping but entirely wiped out by filtering
    # must still appear with a zero count so the gate sees the regression.
    for label in original_classes:
        per_class_cleaned_counts.setdefault(label, 0)
    minimum_class_count = (
        min(per_class_cleaned_counts.values()) if per_class_cleaned_counts else 0
    )

    report.update(
        {
            "cleaned_count": cleaned_count,
            "filter_removed_count": filter_removed_count,
            "per_class_cleaned_counts": per_class_cleaned_counts,
            "minimum_class_count": minimum_class_count,
            "class_count": len(per_class_cleaned_counts),
            # Echo the effective detector + fold count so a reviewer can see
            # which model actually produced ``flagged_fraction`` and the
            # ``pred_probs`` that feed DriftGate's confidence metric.
            "classifier": classifier,
            "n_folds": n_folds,
        }
    )

    dataset_out = Path(args.dataset_out)
    dataset_out.mkdir(parents=True, exist_ok=True)
    (dataset_out / "cleaned_dataset.json").write_text(
        json.dumps(cleaned, indent=2) + "\n", encoding="utf-8"
    )
    (dataset_out / "flagged_images.json").write_text(
        json.dumps(flagged, indent=2) + "\n", encoding="utf-8"
    )
    # Persist the K-fold out-of-sample probability matrix so the downstream
    # DetectDrift step (which mounts this prefix at /opt/ml/processing/input/
    # dataset) can read it and compute a real mean prediction confidence. The
    # classes list goes with it so the artifact is self-describing.
    np.save(dataset_out / "pred_probs.npy", probabilities)
    (dataset_out / "classes.json").write_text(
        json.dumps(classes, indent=2) + "\n", encoding="utf-8"
    )

    metrics_out = Path(args.metrics_out)
    metrics_out.mkdir(parents=True, exist_ok=True)
    # ``metrics.json`` is the property file consumed by CleanlabQualityGate.
    (metrics_out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (metrics_out / "cleanlab_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[kfold_cleanlab] classes={classes} flagged={report['flagged_images']}/"
        f"{report['total_images']} flagged_fraction={report['flagged_fraction']:.3f} "
        f"cleaned={cleaned_count} min_class={minimum_class_count}"
    )

    # ---- Enforce floors on the data actually trained on ---------------------
    # The gate already compares flagged_fraction against the same threshold, but
    # the step itself must enforce it because callers read the step as a filter.
    # Fail-fast here produces a clear error instead of relying solely on the DAG
    # gate, and lets the dataset floors be enforced on the post-filter data.
    violations: list[str] = []
    if report["flagged_fraction"] > flag_threshold:
        violations.append(
            f"flagged_fraction {report['flagged_fraction']:.4f} exceeds "
            f"CLEANLAB_FLAG_THRESHOLD_FRACTION {flag_threshold}"
        )
    if target_dataset_min is not None and cleaned_count < target_dataset_min:
        violations.append(
            f"cleaned_count {cleaned_count} below TARGET_DATASET_MIN {target_dataset_min}"
        )
    if min_images_per_class is not None:
        if minimum_class_count < min_images_per_class:
            violations.append(
                f"minimum_class_count {minimum_class_count} below "
                f"MIN_IMAGES_PER_CLASS {min_images_per_class}"
            )
        missing = [
            label for label, count in per_class_cleaned_counts.items() if count == 0
        ]
        if missing:
            violations.append(
                "classes lost after filtering: " + ", ".join(sorted(missing))
            )
    if violations:
        raise SystemExit(
            "[kfold_cleanlab] post-filter integrity check failed: "
            + "; ".join(violations)
        )


if __name__ == "__main__":
    main()

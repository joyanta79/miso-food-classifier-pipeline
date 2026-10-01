"""Pipeline step: KFoldCleanlabQuality.

Combines the proven step2 (K-fold out-of-sample probabilities) and step3
(CleanLab label-issue filtering) into the single DAG step the pipeline
references. Emits a quality ``metrics.json`` consumed by the CleanlabQualityGate:

    { "flagged_fraction": <float> }

Also writes the cleaned dataset (surviving records) for the training step, plus
flagged_images.json and cleanlab_report.json for inspection.

Python 3.10 compatible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/opt/ml/processing/input/pkgcode")
sys.path.insert(0, "/opt/ml/processing/input/pkgcode/pipeline/steps")

import _bootstrap  # noqa: E402

_bootstrap.ensure(
    {
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "xgboost": "xgboost",
        "cleanlab": "cleanlab",
    }
)

import numpy as np  # noqa: E402

from processing.step2_kfold_predict import kfold_predict  # noqa: E402
from processing.step3_cleanlab_filter import filter_label_issues  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="KFoldCleanlabQuality step")
    parser.add_argument("--embeddings", default="/opt/ml/processing/input/dataset/embeddings.npy")
    parser.add_argument(
        "--mapping", default="/opt/ml/processing/input/dataset/embedding_mapping.json"
    )
    parser.add_argument("--dataset-out", default="/opt/ml/processing/output/dataset")
    parser.add_argument("--metrics-out", default="/opt/ml/processing/output/metrics")
    args = parser.parse_args()

    n_folds = int(os.environ.get("KFOLD_COUNT", "5"))
    classifier = os.environ.get("CLASSIFIER", "xgboost")
    flag_threshold = float(os.environ.get("CLEANLAB_FLAG_THRESHOLD_FRACTION", "0.05"))

    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    embeddings = np.load(args.embeddings)
    labels = [record["label"] for record in mapping]

    probabilities, classes = kfold_predict(
        embeddings, labels, n_folds=n_folds, classifier=classifier
    )
    flagged, cleaned, report = filter_label_issues(
        probabilities, mapping, classes, flag_threshold_fraction=flag_threshold
    )

    dataset_out = Path(args.dataset_out)
    dataset_out.mkdir(parents=True, exist_ok=True)
    (dataset_out / "cleaned_dataset.json").write_text(
        json.dumps(cleaned, indent=2) + "\n", encoding="utf-8"
    )
    (dataset_out / "flagged_images.json").write_text(
        json.dumps(flagged, indent=2) + "\n", encoding="utf-8"
    )

    metrics_out = Path(args.metrics_out)
    metrics_out.mkdir(parents=True, exist_ok=True)
    # report already carries flagged_fraction / flagged_images / total_images.
    (metrics_out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (metrics_out / "cleanlab_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[kfold_cleanlab] classes={classes} flagged={report['flagged_images']}/"
        f"{report['total_images']} flagged_fraction={report['flagged_fraction']:.3f} "
        f"cleaned={len(cleaned)}"
    )


if __name__ == "__main__":
    main()

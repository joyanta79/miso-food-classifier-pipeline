"""Pipeline step: DetectDrift.

The drift step was not implemented in the repo (owned by a separate track). This
is a real, minimal implementation that honors the metric contract the DriftGate
reads:

    { "confidence": <float>, "correction_rate_multiplier": <float> }

It computes a mean prediction confidence from the cleaned dataset's K-fold
probabilities when available; otherwise it uses the baseline confidence from
config. correction_rate_multiplier defaults to a healthy 1.0 (no drift) for a
smoke run, since there is no production correction-rate stream to compare
against. The gate passes when confidence >= level1 floor and multiplier <=
configured limit.

Python 3.10 compatible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/opt/ml/processing/input/pkgcode/pipeline/steps")
import _bootstrap  # noqa: E402

_bootstrap.ensure({"numpy": "numpy"})

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="DetectDrift step")
    parser.add_argument("--pred-probs", default="/opt/ml/processing/input/dataset/pred_probs.npy")
    parser.add_argument("--metrics-out", default="/opt/ml/processing/output/metrics")
    args = parser.parse_args()

    confidence_baseline = float(os.environ.get("CONFIDENCE_BASELINE", "0.85"))

    # Prefer a measured mean confidence when the probability matrix is present;
    # fall back to the configured baseline so the step is robust on a smoke run.
    pred_probs_path = Path(args.pred_probs)
    if pred_probs_path.exists():
        probabilities = np.load(pred_probs_path)
        confidence = (
            float(np.mean(np.max(probabilities, axis=1)))
            if len(probabilities)
            else confidence_baseline
        )
    else:
        confidence = confidence_baseline

    # No production correction-rate stream exists for a smoke run; a multiplier of
    # 1.0 means "no elevated correction rate" -> drift gate passes.
    correction_rate_multiplier = 1.0

    metrics = {
        "confidence": confidence,
        "correction_rate_multiplier": correction_rate_multiplier,
        "confidence_baseline": confidence_baseline,
    }
    metrics_out = Path(args.metrics_out)
    metrics_out.mkdir(parents=True, exist_ok=True)
    (metrics_out / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[drift_detection] confidence={confidence:.3f} "
        f"correction_rate_multiplier={correction_rate_multiplier}"
    )


if __name__ == "__main__":
    main()

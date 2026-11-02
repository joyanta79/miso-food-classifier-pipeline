"""Pipeline step: DetectDrift.

Computes the two metrics the DriftGate enforces and records whether each one
was measured from a real signal or missing. The step is fail-closed:

    { "confidence": <float>,
      "correction_rate_multiplier": <float>,
      "confidence_baseline": <float>,
      "level1_confidence_floor": <float>,
      "level2_confidence_threshold": <float>,
      "correction_rate_multiplier_limit": <float>,
      "level": <int, 0|1|2>,
      "confidence_measured": <bool>,
      "correction_rate_measured": <bool>,
      "status": "ok" | "fail",
      "reasons": [ ... ]
    }

Contract:
* ``--pred-probs`` must point at a non-empty NumPy probability matrix produced
  upstream (``KFoldCleanlabQuality`` persists ``pred_probs.npy`` into the
  DetectDrift input prefix). If the file is missing or empty, the step exits
  non-zero AND writes a metrics file with ``confidence = 0.0`` so the gate
  fails even if the step's exit code is swallowed.
* ``correction_rate_multiplier`` is computed from a real correction-rate
  source: a JSON document with ``observed_correction_rate`` and
  ``baseline_correction_rate`` numeric fields (located via ``--correction-rate``
  or ``CORRECTION_RATE_SOURCE``). When the source is unavailable or malformed,
  the step fails closed and emits a sentinel multiplier (``inf``) so the gate
  rejects the artifact.
* The injected environment variables ``LEVEL1_CONFIDENCE_FLOOR``,
  ``LEVEL2_CONFIDENCE_THRESHOLD`` and ``CORRECTION_RATE_MULTIPLIER`` are
  consumed to derive an escalation ``level`` (0: healthy, 1: watch, 2: block).
* ``CONFIDENCE_BASELINE`` and the related numeric parameters are validated to
  be in ``[0, 1]`` (correction-rate limit is a positive float) and
  ``baseline >= level1_floor >= 0``; invalid configuration fails the step.
* ``metrics.json`` records whether each metric was measured or defaulted so
  the gate / human reviewer can reject defaulted values.

Python 3.10 compatible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, "/opt/ml/processing/input/pkgcode/pipeline/steps")
import _bootstrap  # noqa: E402

_bootstrap.ensure({"numpy": "numpy"})

import numpy as np  # noqa: E402


# A sentinel multiplier the DriftGate will always reject. DriftGate asserts
# correction_rate_multiplier <= CorrectionRateMultiplier (a small positive
# number such as 3). Using a very large finite float keeps the metrics file
# RFC-8259-strict JSON so JsonGet-style readers accept it.
_FAIL_CORRECTION_RATE_MULTIPLIER = 1e308


class DriftConfigError(ValueError):
    """Raised when the drift step's injected configuration is invalid."""


def _parse_probability(name: str, raw: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise DriftConfigError(f"{name} must be a float, got {raw!r}") from error
    if math.isnan(value) or math.isinf(value):
        raise DriftConfigError(f"{name} must be finite, got {value!r}")
    if not 0.0 <= value <= 1.0:
        raise DriftConfigError(f"{name} must be in [0, 1], got {value!r}")
    return value


def _parse_positive(name: str, raw: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise DriftConfigError(f"{name} must be a float, got {raw!r}") from error
    if math.isnan(value) or math.isinf(value) or value <= 0.0:
        raise DriftConfigError(f"{name} must be a positive finite float, got {value!r}")
    return value


def _load_configuration() -> dict[str, float]:
    """Read and validate every pipeline parameter the gate relies on.

    Missing values are an error: the gate's thresholds are deliberately
    injected into the container, so silently defaulting any of them would
    reintroduce the fail-open behavior.
    """
    try:
        confidence_baseline = _parse_probability(
            "CONFIDENCE_BASELINE", os.environ["CONFIDENCE_BASELINE"]
        )
        level1_floor = _parse_probability(
            "LEVEL1_CONFIDENCE_FLOOR", os.environ["LEVEL1_CONFIDENCE_FLOOR"]
        )
        level2_threshold = _parse_probability(
            "LEVEL2_CONFIDENCE_THRESHOLD", os.environ["LEVEL2_CONFIDENCE_THRESHOLD"]
        )
        correction_rate_limit = _parse_positive(
            "CORRECTION_RATE_MULTIPLIER", os.environ["CORRECTION_RATE_MULTIPLIER"]
        )
    except KeyError as error:
        raise DriftConfigError(
            f"Required drift parameter {error.args[0]} is not set"
        ) from error

    # Baseline must be at least as strong as the hard floor: a baseline below
    # the floor would mean "healthy production is already failing the gate",
    # which is nonsensical and almost certainly a config/param-override error.
    if confidence_baseline < level1_floor:
        raise DriftConfigError(
            f"CONFIDENCE_BASELINE ({confidence_baseline}) must be >= "
            f"LEVEL1_CONFIDENCE_FLOOR ({level1_floor})"
        )
    # Level 2 is a stricter escalation point than level 1; keep that ordering.
    if level2_threshold < level1_floor:
        raise DriftConfigError(
            f"LEVEL2_CONFIDENCE_THRESHOLD ({level2_threshold}) must be >= "
            f"LEVEL1_CONFIDENCE_FLOOR ({level1_floor})"
        )
    return {
        "confidence_baseline": confidence_baseline,
        "level1_confidence_floor": level1_floor,
        "level2_confidence_threshold": level2_threshold,
        "correction_rate_multiplier_limit": correction_rate_limit,
    }


def _measure_confidence(pred_probs_path: Path) -> tuple[float, bool, str | None]:
    """Return (confidence, measured, reason).

    ``measured`` is False when the probability matrix is missing or empty; the
    caller treats that as a fail-closed condition rather than defaulting to
    the configured baseline.
    """
    if not pred_probs_path.exists():
        return 0.0, False, f"pred_probs file not found at {pred_probs_path}"
    try:
        probabilities = np.load(pred_probs_path)
    except (OSError, ValueError) as error:
        return 0.0, False, f"pred_probs could not be loaded: {error}"
    if probabilities.ndim != 2 or probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
        return 0.0, False, "pred_probs matrix is empty"
    confidence = float(np.mean(np.max(probabilities, axis=1)))
    if math.isnan(confidence):
        return 0.0, False, "pred_probs produced a NaN mean confidence"
    return confidence, True, None


def _measure_correction_rate(
    source_path: Path | None,
) -> tuple[float, bool, str | None]:
    """Return (multiplier, measured, reason).

    The multiplier is observed_correction_rate / baseline_correction_rate. A
    multiplier > 1.0 means the operator correction rate is elevated vs. the
    historical baseline, so the gate (which asserts multiplier <= configured
    limit) correctly fails on sustained drift. When the source is unavailable
    we return an ``inf`` multiplier so the gate rejects the artifact.
    """
    if source_path is None:
        return (
            _FAIL_CORRECTION_RATE_MULTIPLIER,
            False,
            "correction-rate source not configured",
        )
    if not source_path.exists():
        return (
            _FAIL_CORRECTION_RATE_MULTIPLIER,
            False,
            f"correction-rate source not found at {source_path}",
        )
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return (
            _FAIL_CORRECTION_RATE_MULTIPLIER,
            False,
            f"correction-rate source could not be parsed: {error}",
        )
    try:
        observed = float(payload["observed_correction_rate"])
        baseline = float(payload["baseline_correction_rate"])
    except (KeyError, TypeError, ValueError) as error:
        return (
            _FAIL_CORRECTION_RATE_MULTIPLIER,
            False,
            f"correction-rate source missing numeric fields: {error}",
        )
    if math.isnan(observed) or math.isnan(baseline) or baseline <= 0.0 or observed < 0.0:
        return (
            _FAIL_CORRECTION_RATE_MULTIPLIER,
            False,
            "correction-rate source has non-positive baseline or invalid values",
        )
    return observed / baseline, True, None


def _escalation_level(
    confidence: float,
    multiplier: float,
    config: dict[str, float],
) -> int:
    """Map metrics to the documented escalation levels.

    * Level 0 (healthy): confidence >= level2_threshold AND multiplier <= 1.
    * Level 1 (watch): confidence >= level1_floor AND multiplier <= limit.
    * Level 2 (block): either condition fails -> gate rejects.
    """
    if confidence < config["level1_confidence_floor"]:
        return 2
    if multiplier > config["correction_rate_multiplier_limit"]:
        return 2
    if (
        confidence < config["level2_confidence_threshold"]
        or multiplier > 1.0
    ):
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="DetectDrift step")
    parser.add_argument(
        "--pred-probs",
        default="/opt/ml/processing/input/dataset/pred_probs.npy",
        help="K-fold out-of-sample probability matrix persisted by the upstream step.",
    )
    parser.add_argument(
        "--correction-rate",
        default=os.environ.get("CORRECTION_RATE_SOURCE", ""),
        help=(
            "Path to a JSON document with observed_correction_rate / "
            "baseline_correction_rate. Required for a measured multiplier; "
            "when unavailable the step fails closed."
        ),
    )
    parser.add_argument("--metrics-out", default="/opt/ml/processing/output/metrics")
    args = parser.parse_args()

    metrics_out = Path(args.metrics_out)
    metrics_out.mkdir(parents=True, exist_ok=True)

    try:
        config = _load_configuration()
    except DriftConfigError as error:
        # Fail closed: write a sentinel artifact AND exit non-zero so the gate
        # rejects even if the step's exit code is swallowed.
        failure_metrics = {
            "confidence": 0.0,
            "correction_rate_multiplier": _FAIL_CORRECTION_RATE_MULTIPLIER,
            "confidence_baseline": None,
            "level1_confidence_floor": None,
            "level2_confidence_threshold": None,
            "correction_rate_multiplier_limit": None,
            "level": 2,
            "confidence_measured": False,
            "correction_rate_measured": False,
            "status": "fail",
            "reasons": [f"configuration error: {error}"],
        }
        (metrics_out / "metrics.json").write_text(
            json.dumps(failure_metrics, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"[drift_detection] configuration error: {error}", file=sys.stderr)
        sys.exit(2)

    reasons: list[str] = []

    confidence, confidence_measured, confidence_reason = _measure_confidence(
        Path(args.pred_probs)
    )
    if confidence_reason:
        reasons.append(confidence_reason)

    correction_rate_source = Path(args.correction_rate) if args.correction_rate else None
    multiplier, correction_measured, correction_reason = _measure_correction_rate(
        correction_rate_source
    )
    if correction_reason:
        reasons.append(correction_reason)

    level = _escalation_level(confidence, multiplier, config)
    status = "ok" if (confidence_measured and correction_measured and level < 2) else "fail"

    metrics = {
        "confidence": confidence,
        "correction_rate_multiplier": multiplier,
        "confidence_baseline": config["confidence_baseline"],
        "level1_confidence_floor": config["level1_confidence_floor"],
        "level2_confidence_threshold": config["level2_confidence_threshold"],
        "correction_rate_multiplier_limit": config["correction_rate_multiplier_limit"],
        "level": level,
        "confidence_measured": confidence_measured,
        "correction_rate_measured": correction_measured,
        "status": status,
        "reasons": reasons,
    }
    (metrics_out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"[drift_detection] confidence={confidence:.3f} "
        f"correction_rate_multiplier={multiplier} level={level} status={status}"
    )

    # Exit non-zero when any required signal is missing or the metrics
    # themselves indicate a block-level drift event. The written metrics still
    # record the sentinel values so the DriftGate also independently rejects.
    if status != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()

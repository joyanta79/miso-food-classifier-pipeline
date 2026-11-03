"""Pipeline step: SelectRecentBrandData.

Container entrypoint referenced by the SageMaker Pipeline DAG. Scans the raw
image channel, keeps files matching the brand naming pattern, and emits a
selection ``metrics.json`` consumed by the DatasetBoundsGate:

    { "dataset_size": <int>, "minimum_class_count": <int> }

It also writes a curated manifest that the embedding step consumes. Label and
confidence are parsed from the ``<brand>_<label>_<confidence>.<ext>`` filename
scheme; the confidence field is retained for downstream reporting.

Brand separation (defence in depth against CWE-639)
---------------------------------------------------
The input channel for this step is expected to be a per-brand S3 prefix
(mounted at ``/opt/ml/processing/input/images``). As a second layer this step:

* refuses to run with an empty/absent ``NAMING_PATTERN`` (fail-closed);
* matches ``NAMING_PATTERN`` with ``fnmatch`` -- the same real-glob semantics
  used by ``processing/step0_data_selection.py`` and ``lambdas/handlers.py`` --
  rather than reducing the pattern to a literal prefix with ``replace("*","")``
  (``*`` -> ``""`` previously disabled the filter entirely); and
* derives an expected brand token from the literal prefix of the naming
  pattern (the characters before the first wildcard) and asserts that the
  ``<brand>`` token parsed from each filename matches it. A pattern with no
  literal brand prefix (e.g. ``*``) is refused outright, so an operator
  override of ``BrandNamingPattern=*`` can no longer mix every brand's images
  into this brand's dataset.

Python 3.10 compatible (SageMaker DLC containers run 3.10).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
_WILDCARD_CHARS = frozenset("*?[")
# Default class allow-list mirrors ``evaluation.done_label`` and
# ``evaluation.not_done_label`` in ``config.yaml``. Overridable via the
# ``DONE_LABEL`` / ``NOT_DONE_LABEL`` env vars so brand pipelines that rename
# the classes continue to work without a code change.
_DEFAULT_DONE_LABEL = "done"
_DEFAULT_NOT_DONE_LABEL = "not_done"


def _parse_name(path: Path) -> tuple[str, str, str] | None:
    """Return ``(brand, label, confidence)`` from ``brand_label_confidence.ext``.

    The label is allowed to contain embedded underscores (e.g.
    ``wc_not_done_0.9.jpg`` -> label ``not_done``) so the admitted class names
    stay round-trip identical to the configured ``evaluation.done_label`` /
    ``evaluation.not_done_label`` values. Returns ``None`` when the filename
    does not match the expected scheme.
    """
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    # brand = parts[0]; confidence = parts[-1]; label = everything between,
    # re-joined with ``_`` so the parsed label matches the on-disk convention
    # (e.g. ``not_done``) and the configured class names (CWE-841).
    brand = parts[0]
    label = "_".join(parts[1:-1])
    confidence = parts[-1]
    return brand, label, confidence


def _expected_brand_token(naming_pattern: str) -> str:
    """Return the literal brand-token prefix implied by ``naming_pattern``.

    The brand token is defined as the characters that precede both the first
    glob metacharacter and the first ``_``. Patterns without any literal
    prefix (e.g. ``*``, ``*_bb_*``) return the empty string and are rejected
    by callers -- they cannot be attributed to a single brand.
    """
    literal_prefix_end = len(naming_pattern)
    for index, char in enumerate(naming_pattern):
        if char in _WILDCARD_CHARS:
            literal_prefix_end = index
            break
    literal = naming_pattern[:literal_prefix_end]
    underscore = literal.find("_")
    if underscore == -1:
        return ""
    return literal[:underscore]




def select(
    image_root: Path,
    naming_pattern: str,
    *,
    brand_id: str = "",
    min_images_per_class: int,
    target_dataset_min: int,
    target_dataset_max: int,
    allowed_labels: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the curated manifest and selection metrics for one brand.

    ``naming_pattern`` is a glob (e.g. ``wc_*``) and is required. The literal
    prefix before the first wildcard must contain a non-empty ``<brand>``
    token (``wc`` for ``wc_*``); patterns without one -- including ``*`` --
    are refused so the step cannot silently select data for every brand.
    ``brand_id`` is accepted for logging/metadata but is not used as the
    attribution key, because the configured ``brand.id`` (e.g. ``white-castle``)
    does not have to equal the filename brand token (``wc``).

    ``allowed_labels`` restricts which filename-derived labels are admitted
    into the manifest. Mirrors ``processing/step0_data_selection.py``:
    attacker-controlled filename tokens (e.g. ``wc_zzz_0.1.jpg``) cannot
    introduce a novel class that drives ``minimum_class_count`` down to 1 and
    permanently fails the ``DatasetBoundsGate`` (CWE-841). Classes below
    ``min_images_per_class`` are then dropped from the manifest and from
    ``minimum_class_count`` so one under-populated class cannot block
    retraining either. ``target_dataset_max`` is enforced as a deterministic
    stratified per-class quota so one legitimate class cannot evict the
    others from the curated manifest.
    """
    if not naming_pattern:
        raise ValueError(
            "NAMING_PATTERN is required for brand data selection; refusing "
            "to run with an empty pattern (would otherwise select every file)."
        )
    expected_brand = _expected_brand_token(naming_pattern)
    if not expected_brand:
        raise ValueError(
            "NAMING_PATTERN does not pin a brand: expected a literal "
            "'<brand>_...' prefix (e.g. 'wc_*'), got "
            f"{naming_pattern!r}. Refusing to run so cross-brand data cannot "
            "be mixed into this brand's dataset."
        )
    if allowed_labels is None:
        allowed_label_set: set[str] = {_DEFAULT_DONE_LABEL, _DEFAULT_NOT_DONE_LABEL}
    else:
        allowed_label_set = {str(label) for label in allowed_labels if str(label)}
    if len(allowed_label_set) < 2:
        raise ValueError(
            "At least two distinct allowed labels are required for brand data "
            f"selection, got {sorted(allowed_label_set)!r}. Configure "
            "evaluation.done_label and evaluation.not_done_label."
        )

    # Phase 1: enumerate all candidate records and tally raw label counts.
    # Records whose label is not in ``allowed_label_set`` are reported in
    # ``rejected_labels`` but never admitted to the manifest; this closes the
    # attacker-controlled-filename vector where a novel label token (e.g.
    # ``wc_zzz_0.1.jpg``) would otherwise drive ``minimum_class_count`` to
    # the attacker's chosen value.
    candidates: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    rejected_label_counts: Counter[str] = Counter()
    for path in sorted(image_root.rglob("*")):
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        if not fnmatch.fnmatch(path.name, naming_pattern):
            continue
        parsed = _parse_name(path)
        if parsed is None:
            continue
        brand, label, confidence = parsed
        # The parsed brand token comes from an attacker-controlled filename.
        # Only accept it when it matches the brand token pinned by the
        # configured naming pattern -- otherwise an uploader who guesses
        # another brand's convention can poison this brand's dataset.
        if brand != expected_brand:
            continue
        # Record the path relative to ``image_root`` as a POSIX string so that
        # discovery (``rglob``) and resolution (``images_root / image_path``)
        # use the same namespace. ``id`` and ``key`` are set to the same
        # relative key so records are uniquely identified even when two files
        # under different sub-prefixes share a basename.
        try:
            rel_path = path.relative_to(image_root).as_posix()
        except ValueError:
            rel_path = Path(path.name).as_posix()
        if label not in allowed_label_set:
            rejected_label_counts[label] += 1
            continue
        label_counts[label] += 1
        candidates.append(
            {
                "id": rel_path,
                "key": rel_path,
                "image_path": rel_path,
                "label": label,
                "confidence": confidence,
                "last_modified": datetime.now(UTC).isoformat(),
            }
        )

    # Phase 2: drop classes below ``min_images_per_class`` so an
    # under-populated class (one or two uploads with a legitimate label) does
    # not permanently fail the DatasetBoundsGate. These classes are reported
    # under ``ineligible_labels`` so operators can see why retraining is
    # waiting on more data for a specific class.
    eligible_labels = {
        label for label, count in label_counts.items() if count >= min_images_per_class
    }
    ineligible_label_counts = {
        label: count
        for label, count in label_counts.items()
        if label not in eligible_labels
    }
    eligible_candidates = [
        item for item in candidates if item["label"] in eligible_labels
    ]
    eligible_counts: Counter[str] = Counter(
        item["label"] for item in eligible_candidates
    )

    # Phase 3: enforce ``target_dataset_max`` as a stratified per-class quota
    # inside the step (mirrors ``processing/step0_data_selection.py``) so the
    # DatasetBoundsGate's ``dataset_size <= target_dataset_max`` condition
    # does not hard-fail on a bulk upload. See CWE-841.
    if target_dataset_max < 0:
        raise ValueError(
            f"target_dataset_max must be non-negative, got {target_dataset_max!r}"
        )
    manifest = _apply_per_class_quota(
        eligible_candidates,
        eligible_counts,
        target_dataset_max=target_dataset_max,
    )

    dataset_size = len(manifest)
    final_counts: Counter[str] = Counter(item["label"] for item in manifest)
    # ``minimum_class_count`` is computed over the eligible classes only, and
    # only when every configured class is represented. Otherwise it is 0 so
    # the gate fails closed with a clear reason in metrics.json rather than
    # being poisoned by an attacker-chosen label.
    missing_required = sorted(allowed_label_set - set(final_counts))
    if missing_required or not final_counts:
        minimum_class_count = 0
    else:
        minimum_class_count = min(final_counts.values())
    metrics = {
        "dataset_size": dataset_size,
        "minimum_class_count": minimum_class_count,
        "class_distribution": dict(sorted(final_counts.items())),
        "raw_class_distribution": dict(sorted(label_counts.items())),
        "ineligible_labels": dict(sorted(ineligible_label_counts.items())),
        "rejected_labels": dict(sorted(rejected_label_counts.items())),
        "missing_required_labels": missing_required,
        "allowed_labels": sorted(allowed_label_set),
        "target_dataset_min": target_dataset_min,
        "target_dataset_max": target_dataset_max,
        "min_images_per_class": min_images_per_class,
    }
    return manifest, metrics


def _apply_per_class_quota(
    candidates: list[dict[str, Any]],
    counts: Counter[str],
    *,
    target_dataset_max: int,
) -> list[dict[str, Any]]:
    """Deterministically cap ``candidates`` at ``target_dataset_max`` per class.

    The cap is split evenly across every present class so that one legitimate
    (but much larger) class cannot evict the others. Any remaining capacity
    is filled round-robin in sorted-label order to keep the result
    reproducible across runs. If ``target_dataset_max`` is zero the manifest
    is empty; if it is at least ``len(candidates)`` the candidates are
    returned unchanged (preserving existing ordering).
    """
    total = len(candidates)
    if total == 0 or target_dataset_max == 0:
        return [] if target_dataset_max == 0 else list(candidates)
    if total <= target_dataset_max:
        return list(candidates)
    present_labels = sorted(counts)
    num_labels = len(present_labels)
    per_class_quota = target_dataset_max // num_labels
    quotas: dict[str, int] = {
        label: min(per_class_quota, counts[label]) for label in present_labels
    }
    remaining = target_dataset_max - sum(quotas.values())
    while remaining > 0:
        progressed = False
        for label in present_labels:
            if remaining <= 0:
                break
            if quotas[label] < counts[label]:
                quotas[label] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    taken: Counter[str] = Counter()
    truncated: list[dict[str, Any]] = []
    for item in candidates:
        label = item["label"]
        if taken[label] < quotas.get(label, 0):
            truncated.append(item)
            taken[label] += 1
    return truncated


def main() -> None:
    parser = argparse.ArgumentParser(description="SelectRecentBrandData step")
    parser.add_argument("--images", default="/opt/ml/processing/input/images")
    parser.add_argument("--dataset-out", default="/opt/ml/processing/output/dataset")
    parser.add_argument("--metrics-out", default="/opt/ml/processing/output/metrics")
    args = parser.parse_args()

    # Fail-closed: refuse to run without an explicit, non-empty, brand-pinning
    # pattern. Do not strip wildcards -- fnmatch handles them natively and
    # matches the behaviour of the sibling implementations in
    # processing/step0_data_selection.py and lambdas/handlers.py.
    naming_pattern = os.environ.get("NAMING_PATTERN", "").strip()
    brand_id = os.environ.get("BRAND_ID", "").strip()
    if not naming_pattern:
        print(
            "[data_selection] FATAL: NAMING_PATTERN env var is empty or unset; "
            "refusing to select every file in the mounted channel.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not _expected_brand_token(naming_pattern):
        print(
            "[data_selection] FATAL: NAMING_PATTERN "
            f"{naming_pattern!r} has no literal '<brand>_' prefix; "
            "refusing to run so cross-brand data cannot be selected.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    min_images_per_class = int(os.environ.get("MIN_IMAGES_PER_CLASS", "20"))
    target_dataset_min = int(os.environ.get("TARGET_DATASET_MIN", "40"))
    target_dataset_max = int(os.environ.get("TARGET_DATASET_MAX", "50000"))
    # Mirror the sibling implementation (processing/step0_data_selection.py)
    # by pinning the admitted label set to the configured class names. The
    # env vars default to the ``evaluation.done_label`` /
    # ``evaluation.not_done_label`` values in ``config.yaml`` so brand
    # pipelines that have not yet been updated to pass these variables keep
    # working.
    done_label = os.environ.get("DONE_LABEL", _DEFAULT_DONE_LABEL).strip()
    not_done_label = os.environ.get("NOT_DONE_LABEL", _DEFAULT_NOT_DONE_LABEL).strip()
    allowed_labels = {done_label, not_done_label}

    manifest, metrics = select(
        Path(args.images),
        naming_pattern,
        brand_id=brand_id,
        min_images_per_class=min_images_per_class,
        target_dataset_min=target_dataset_min,
        target_dataset_max=target_dataset_max,
        allowed_labels=allowed_labels,
    )

    dataset_out = Path(args.dataset_out)
    dataset_out.mkdir(parents=True, exist_ok=True)
    (dataset_out / "curated_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    metrics_out = Path(args.metrics_out)
    metrics_out.mkdir(parents=True, exist_ok=True)
    (metrics_out / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[data_selection] dataset_size={metrics['dataset_size']} "
        f"minimum_class_count={metrics['minimum_class_count']} "
        f"distribution={metrics['class_distribution']} "
        f"ineligible={metrics['ineligible_labels']} "
        f"rejected={metrics['rejected_labels']} "
        f"missing_required={metrics['missing_required_labels']}"
    )


if __name__ == "__main__":
    main()

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
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
_WILDCARD_CHARS = frozenset("*?[")

# Strict ``<brand>_<label>_<confidence>`` schema shared with
# ``processing/step0_data_selection.py``. The brand token forbids ``_`` so the
# parse is unambiguous, the label preserves underscores (e.g. ``not_done``),
# and the confidence must be a non-negative decimal number. Filenames that do
# not match -- for example ``wc_done_0.9_old`` (non-numeric confidence) or
# ``wc_junk_abc`` -- are rejected rather than counted into
# ``minimum_class_count``.
_METADATA_PATTERN = re.compile(
    r"^(?P<brand>[^_]+)_(?P<label>.+)_(?P<confidence>\d+(?:\.\d+)?)$"
)


def _parse_name(path: Path) -> tuple[str, str, str] | None:
    """Return ``(brand, label, confidence)`` from ``brand_label_confidence.ext``.

    Returns ``None`` when the filename does not match the strict schema. In
    particular, the confidence suffix MUST be a numeric literal; names such as
    ``wc_done_0.9_old`` are refused so that an uploader cannot invent a new
    label by adding an extra underscore-separated suffix.
    """
    stem = path.stem
    match = _METADATA_PATTERN.match(stem)
    if not match:
        return None
    return match.group("brand"), match.group("label"), match.group("confidence")


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
    allowed_labels: tuple[str, ...] | list[str] | set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the curated manifest and selection metrics for one brand.

    ``naming_pattern`` is a glob (e.g. ``wc_*``) and is required. The literal
    prefix before the first wildcard must contain a non-empty ``<brand>``
    token (``wc`` for ``wc_*``); patterns without one -- including ``*`` --
    are refused so the step cannot silently select data for every brand.
    ``brand_id`` is accepted for logging/metadata but is not used as the
    attribution key, because the configured ``brand.id`` (e.g. ``white-castle``)
    does not have to equal the filename brand token (``wc``).

    ``allowed_labels`` is the operator-approved label policy (typically the
    ``done_label`` / ``not_done_label`` from ``config.yaml``). Records whose
    filename-derived label is not in this allowlist are rejected before being
    counted. When ``allowed_labels`` is ``None`` the allowlist check is
    skipped (used by unit tests only; production callers always pass one).
    Rejected filenames do not contribute to ``minimum_class_count``; their
    count is reported under ``metrics['rejected_filenames']`` so operators
    can see when uploads are being discarded.
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

    allowed_label_set: set[str] | None
    if allowed_labels is None:
        allowed_label_set = None
    else:
        allowed_label_set = {label for label in allowed_labels if label}
        if not allowed_label_set:
            raise ValueError(
                "allowed_labels was provided but is empty after filtering; "
                "refusing to run so an attacker-chosen label cannot define "
                "the training label space."
            )

    manifest: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    # Counters of filenames rejected *after* the brand glob matched. These
    # feed the operator-visible ``metrics['rejected_filenames']`` so a sudden
    # jump signals bad uploads.
    rejected_malformed = 0  # filename does not match <brand>_<label>_<conf>
    rejected_brand_mismatch = 0  # parsed brand token != pinned brand
    rejected_label_not_allowed = 0  # label not in operator allowlist
    for path in sorted(image_root.rglob("*")):
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        if not fnmatch.fnmatch(path.name, naming_pattern):
            continue
        parsed = _parse_name(path)
        if parsed is None:
            # Does not match <brand>_<label>_<numeric-confidence>: e.g.
            # ``wc_done_0.9_old`` (extra underscore / non-numeric confidence)
            # or ``wc_junk_abc``. Reject rather than invent a label.
            rejected_malformed += 1
            continue
        brand, label, confidence = parsed
        # The parsed brand token comes from an attacker-controlled filename.
        # Only accept it when it matches the brand token pinned by the
        # configured naming pattern -- otherwise an uploader who guesses
        # another brand's convention can poison this brand's dataset.
        if brand != expected_brand:
            rejected_brand_mismatch += 1
            continue
        # Reject labels outside the operator-configured allowlist so an
        # uploader cannot introduce new classes (``junk``, etc.) into the
        # binary food-safety classifier.
        if allowed_label_set is not None and label not in allowed_label_set:
            rejected_label_not_allowed += 1
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
        label_counts[label] += 1
        manifest.append(
            {
                "id": rel_path,
                "key": rel_path,
                "image_path": rel_path,
                "label": label,
                "confidence": confidence,
                "last_modified": datetime.now(UTC).isoformat(),
            }
        )

    # Per-class eligibility filter (mirrors ``processing/step0_data_selection``):
    # labels below ``min_images_per_class`` are dropped from both the manifest
    # and the published ``minimum_class_count`` so a single stray upload
    # cannot collapse the gate metric and wedge the brand's pipeline.
    eligible_labels = {
        label
        for label, count in label_counts.items()
        if count >= min_images_per_class
    }
    rejected_below_min_class = sum(
        count for label, count in label_counts.items() if label not in eligible_labels
    )
    manifest = [item for item in manifest if item["label"] in eligible_labels]
    eligible_counts = Counter(
        {label: count for label, count in label_counts.items() if label in eligible_labels}
    )

    dataset_size = len(manifest)
    minimum_class_count = (
        min(eligible_counts.values()) if eligible_counts else 0
    )
    rejected_total = (
        rejected_malformed
        + rejected_brand_mismatch
        + rejected_label_not_allowed
        + rejected_below_min_class
    )
    metrics = {
        "dataset_size": dataset_size,
        "minimum_class_count": minimum_class_count,
        "class_distribution": dict(sorted(eligible_counts.items())),
        "target_dataset_min": target_dataset_min,
        "target_dataset_max": target_dataset_max,
        "min_images_per_class": min_images_per_class,
        "rejected_filenames": {
            "total": rejected_total,
            "malformed": rejected_malformed,
            "brand_mismatch": rejected_brand_mismatch,
            "label_not_allowed": rejected_label_not_allowed,
            "below_min_images_per_class": rejected_below_min_class,
        },
    }
    return manifest, metrics


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

    # Operator-configured label allowlist (``evaluation.done_label`` /
    # ``evaluation.not_done_label`` from config.yaml). Pipeline wiring injects
    # these as literal env vars so a per-execution PipelineParameters override
    # cannot broaden the label space.
    done_label = os.environ.get("DONE_LABEL", "done").strip()
    not_done_label = os.environ.get("NOT_DONE_LABEL", "not_done").strip()
    allowed_labels = tuple(
        label for label in (done_label, not_done_label) if label
    )
    if not allowed_labels:
        print(
            "[data_selection] FATAL: DONE_LABEL / NOT_DONE_LABEL env vars are "
            "empty; refusing to run without a label allowlist.",
            file=sys.stderr,
        )
        raise SystemExit(2)

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
        f"rejected={metrics['rejected_filenames']}"
    )


if __name__ == "__main__":
    main()

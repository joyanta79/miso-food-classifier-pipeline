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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Match the sys.path layout the other pipeline steps use so this step can
# share ``processing.step0_data_selection.parse_filename_metadata`` -- the
# single, strict grammar for ``<brand>_<label>_<confidence>.<ext>``.
sys.path.insert(0, "/opt/ml/processing/input/pkgcode")
sys.path.insert(0, "/opt/ml/processing/input/pkgcode/pipeline/steps")

try:  # pragma: no cover - import side effect depends on container layout
    from processing.step0_data_selection import (  # noqa: E402
        parse_filename_metadata as _parse_filename_metadata,
    )
except Exception:  # pragma: no cover - fall back for local test contexts
    import re as _re

    _METADATA_PATTERN = _re.compile(
        r"^(?P<brand>[^_]+)_(?P<label>.+)_(?P<confidence>\d+(?:\.\d+)?)$"
    )

    def _parse_filename_metadata(key: str) -> tuple[str, str, float]:
        stem = Path(key).stem
        match = _METADATA_PATTERN.match(stem)
        if not match:
            raise ValueError(
                f"Image key does not match brand_label_confidence format: {key}"
            )
        return (
            match.group("brand"),
            match.group("label"),
            float(match.group("confidence")),
        )


UTC = timezone.utc
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
_WILDCARD_CHARS = frozenset("*?[")


def _parse_name(path: Path) -> tuple[str, str, float] | None:
    """Return ``(brand, label, confidence)`` from ``brand_label_confidence.ext``.

    Uses the shared strict grammar from
    ``processing.step0_data_selection.parse_filename_metadata`` so the label
    keeps its underscores (``wc_not_done_0.93.jpg`` -> label ``not_done``) and
    the confidence must be numeric. Returns ``None`` when the filename does
    not match the expected scheme instead of silently reconstructing fields.
    """
    try:
        brand, label, confidence = _parse_filename_metadata(path.name)
    except ValueError:
        return None
    # Enforce the documented [0, 1] confidence range so out-of-band values
    # (e.g. ``wc_label_1234567.jpg`` -> 1234567.0) are rejected at the edge.
    if not (0.0 <= confidence <= 1.0):
        return None
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




def _normalise_label_allowlist(
    labels: Iterable[str] | None,
) -> frozenset[str] | None:
    """Return the de-duplicated, non-empty label allowlist (or ``None``).

    ``None`` disables allowlist enforcement; an empty iterable is treated as
    "no labels are permitted" so operator misconfiguration is fail-closed.
    """
    if labels is None:
        return None
    cleaned = {label.strip() for label in labels if label and label.strip()}
    return frozenset(cleaned)


def select(
    image_root: Path,
    naming_pattern: str,
    *,
    brand_id: str = "",
    min_images_per_class: int,
    target_dataset_min: int,
    target_dataset_max: int,
    label_allowlist: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the curated manifest and selection metrics for one brand.

    ``naming_pattern`` is a glob (e.g. ``wc_*``) and is required. The literal
    prefix before the first wildcard must contain a non-empty ``<brand>``
    token (``wc`` for ``wc_*``); patterns without one -- including ``*`` --
    are refused so the step cannot silently select data for every brand.
    ``brand_id`` is accepted for logging/metadata but is not used as the
    attribution key, because the configured ``brand.id`` (e.g. ``white-castle``)
    does not have to equal the filename brand token (``wc``).
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

    allowlist = _normalise_label_allowlist(label_allowlist)

    manifest: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
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
        # Enforce the operator-configured label allowlist (e.g. the
        # ``done`` / ``not_done`` labels referenced by the evaluation gate).
        # Filenames whose label is not in the allowlist are dropped so an
        # uploader cannot invent new training classes via the object name.
        if allowlist is not None and label not in allowlist:
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
    dataset_size = len(manifest)
    minimum_class_count = min(label_counts.values()) if label_counts else 0
    metrics = {
        "dataset_size": dataset_size,
        "minimum_class_count": minimum_class_count,
        "class_distribution": dict(sorted(label_counts.items())),
        "target_dataset_min": target_dataset_min,
        "target_dataset_max": target_dataset_max,
        "min_images_per_class": min_images_per_class,
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
    # Build the label allowlist from the configured done/not_done labels.
    # ``LABEL_ALLOWLIST`` (comma-separated) may extend the pair if an
    # operator provisions additional classes. Fail-closed when neither is set.
    allowlist_tokens: list[str] = []
    done_label = os.environ.get("DONE_LABEL", "").strip()
    not_done_label = os.environ.get("NOT_DONE_LABEL", "").strip()
    if done_label:
        allowlist_tokens.append(done_label)
    if not_done_label:
        allowlist_tokens.append(not_done_label)
    extra = os.environ.get("LABEL_ALLOWLIST", "").strip()
    if extra:
        allowlist_tokens.extend(token.strip() for token in extra.split(","))
    label_allowlist: list[str] | None = (
        [token for token in allowlist_tokens if token] if allowlist_tokens else None
    )
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

    if label_allowlist is None:
        print(
            "[data_selection] FATAL: no label allowlist configured; set "
            "DONE_LABEL / NOT_DONE_LABEL (and optionally LABEL_ALLOWLIST) "
            "so an uploader cannot invent training classes via filename.",
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
        label_allowlist=label_allowlist,
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
        f"distribution={metrics['class_distribution']}"
    )


if __name__ == "__main__":
    main()

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
# A label is used downstream as a filesystem directory component (see
# pipeline/steps/train_step.py: ``dest = split_root / label``). Constrain it
# to a strict, single safe path component so attacker-controlled filenames
# (e.g. ``wc_.._0.1.jpg`` -> label ``".."``, ``wc__0.1.jpg`` -> label ``""``)
# cannot escape the prepared class-directory tree or silently alias to the
# split root. The pattern mirrors the mitigation guidance on the finding.
_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,63}$")


def _is_safe_label(label: str) -> bool:
    """Return True when ``label`` is safe to use as a class directory name.

    Guarantees enforced:
    * non-empty, bounded length;
    * no path separators, ``.`` / ``..`` traversal tokens, NULs or other
      control characters;
    * matches the strict allow-list pattern ``^[A-Za-z0-9][A-Za-z0-9-]{0,63}$``.
    """
    if not isinstance(label, str):
        return False
    return bool(_LABEL_PATTERN.fullmatch(label))


def _parse_name(path: Path) -> tuple[str, str, str] | None:
    """Return ``(brand, label, confidence)`` from ``brand_label_confidence.ext``.

    Returns ``None`` when the filename does not match the expected scheme.
    """
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    # brand = parts[0]; confidence = parts[-1]; label = everything between.
    brand = parts[0]
    label = "".join(parts[1:-1])
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
        # The label is attacker-controlled and is later used verbatim as a
        # filesystem directory component in the training container
        # (pipeline/steps/train_step.py: ``dest = split_root / label``).
        # Reject anything that is not a safe single path component so that
        # ``.``, ``..``, the empty string and control characters cannot
        # become class names and escape the prepared class-directory tree.
        if not _is_safe_label(label):
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

    manifest, metrics = select(
        Path(args.images),
        naming_pattern,
        brand_id=brand_id,
        min_images_per_class=min_images_per_class,
        target_dataset_min=target_dataset_min,
        target_dataset_max=target_dataset_max,
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

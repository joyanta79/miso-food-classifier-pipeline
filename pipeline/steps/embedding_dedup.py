"""Pipeline step: EmbedAndDeduplicate.

Wraps the proven step1 embedding logic: loads the curated manifest + the real
images from the images channel, runs the frozen EfficientNetV2-B0 backbone
(ImageNet) to produce (N, 1280) embeddings, applies cosine de-duplication, and
writes embeddings.npy + embedding_mapping.json for the quality step.

Python 3.10 compatible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# The processing package ships alongside this script in the code channel.
sys.path.insert(0, "/opt/ml/processing/input/pkgcode")
sys.path.insert(0, "/opt/ml/processing/input/pkgcode/pipeline/steps")

import _bootstrap  # noqa: E402

_bootstrap.ensure({"PIL": "pillow", "numpy": "numpy"})

import numpy as np  # noqa: E402

from processing.step1_extract_embeddings import extract_embeddings, write_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="EmbedAndDeduplicate step")
    parser.add_argument(
        "--manifest", default="/opt/ml/processing/input/dataset/curated_manifest.json"
    )
    parser.add_argument("--images", default="/opt/ml/processing/input/images")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output/dataset")
    parser.add_argument(
        "--metrics-out",
        default=None,
        help=(
            "Directory to write ``metrics.json`` with the deterministic-fallback "
            "count. Defaults to ``--output-dir`` so the existing dedup output "
            "channel surfaces it alongside the embeddings."
        ),
    )
    parser.add_argument(
        "--allow-deterministic",
        action="store_true",
        default=os.environ.get("ALLOW_DETERMINISTIC_EMBEDDINGS", "").lower()
        in {"1", "true", "yes"},
        help=(
            "Permit records produced by the deterministic (hash-based) backend. "
            "Intended for CI only; production runs must fail closed when any "
            "record did not use the EfficientNetV2 backbone."
        ),
    )
    args = parser.parse_args()

    embedding_dim = int(os.environ.get("EMBEDDING_DIM", "1280"))
    cosine = float(os.environ.get("COSINE_DEDUP_THRESHOLD", "0.99"))
    backbone_source = os.environ.get("BACKBONE_SOURCE", "imagenet")

    # The pipeline bakes the configured values as literals into the step
    # Environment, so a well-formed execution ships identical ``CONFIGURED_*``
    # entries. We validate against them as a defence-in-depth guard: even if a
    # future change (or a stray container-level override) re-introduced caller
    # control over the active value, the step must refuse to run with a knob
    # that is more permissive than the configured baseline. The previous
    # ``(0, 1]`` interval alone let a caller push ``COSINE_DEDUP_THRESHOLD``
    # toward 1.0 and retain every non-identical near-duplicate.
    configured_cosine_raw = os.environ.get("CONFIGURED_COSINE_DEDUP_THRESHOLD")
    if configured_cosine_raw is not None:
        try:
            configured_cosine = float(configured_cosine_raw)
        except ValueError as error:
            raise SystemExit(
                f"[embedding_dedup] CONFIGURED_COSINE_DEDUP_THRESHOLD must be numeric; "
                f"got {configured_cosine_raw!r}"
            ) from error
        if not 0.0 < configured_cosine <= 1.0:
            raise SystemExit(
                "[embedding_dedup] CONFIGURED_COSINE_DEDUP_THRESHOLD must satisfy "
                f"0 < value <= 1, got {configured_cosine!r}"
            )
        # Lower threshold = more aggressive dedup, so ``cosine`` must not exceed
        # the configured baseline. Raising it would retain more duplicates and
        # weaken the downstream CleanlabQualityGate floors.
        if cosine > configured_cosine:
            raise SystemExit(
                f"[embedding_dedup] COSINE_DEDUP_THRESHOLD {cosine!r} exceeds configured "
                f"value {configured_cosine!r}; a per-execution override cannot relax "
                "near-duplicate removal."
            )

    configured_dim_raw = os.environ.get("CONFIGURED_EMBEDDING_DIM")
    if configured_dim_raw is not None:
        try:
            configured_dim = int(configured_dim_raw)
        except ValueError as error:
            raise SystemExit(
                f"[embedding_dedup] CONFIGURED_EMBEDDING_DIM must be an integer; "
                f"got {configured_dim_raw!r}"
            ) from error
        if embedding_dim != configured_dim:
            raise SystemExit(
                f"[embedding_dedup] EMBEDDING_DIM {embedding_dim!r} does not match "
                f"configured value {configured_dim!r}; the backbone cannot be swapped "
                "per execution."
            )

    configured_backbone = os.environ.get("CONFIGURED_BACKBONE_SOURCE")
    if configured_backbone is not None and backbone_source != configured_backbone:
        raise SystemExit(
            f"[embedding_dedup] BACKBONE_SOURCE {backbone_source!r} does not match "
            f"configured value {configured_backbone!r}; the backbone provenance "
            "cannot be swapped per execution."
        )

    # A smoke run uses ImageNet weights; production would point at the prior
    # fine-tuned backbone path. 'imagenet' -> pretrained weights, else deterministic.
    weights = "imagenet" if backbone_source == "imagenet" else None

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    embeddings, mapping = extract_embeddings(
        manifest,
        embedding_dim=embedding_dim,
        cosine_dedup_threshold=cosine,
        image_root=args.images,
        use_real_backbone=True,
        backbone_weights=weights,
    )
    write_outputs(embeddings, mapping, args.output_dir)
    backends = {record.get("embedding_backend") for record in mapping}
    deterministic_fallback_count = sum(
        1 for record in mapping if record.get("embedding_backend") == "deterministic"
    )
    efficientnet_only = backends == {"efficientnet"}

    # Emit the deterministic-fallback count so the downstream quality gates can
    # reject a run where any record was embedded without image pixels.
    metrics_out = Path(args.metrics_out) if args.metrics_out else Path(args.output_dir)
    metrics_out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "total_records": len(mapping),
        "deterministic_fallback_count": deterministic_fallback_count,
        "backends": sorted(backend for backend in backends if backend is not None),
        "efficientnet_only": efficientnet_only,
        "allow_deterministic": bool(args.allow_deterministic),
        # Echo the effective knobs so a reviewer (and any downstream gate) can
        # see which values actually ran. The pipeline bakes these as literals,
        # so they should always match the configured baseline.
        "cosine_dedup_threshold": cosine,
        "embedding_dim": embedding_dim,
        "backbone_source": backbone_source,
    }
    (metrics_out / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"[embedding_dedup] embeddings={np.asarray(embeddings).shape} "
        f"kept={len(mapping)} backends={backends} "
        f"deterministic_fallback_count={deterministic_fallback_count}"
    )

    # Fail closed in production: a run that did not use the EfficientNetV2
    # backbone for every record cannot be trusted for de-duplication or the
    # downstream label-quality / drift gates, which all compute metrics from
    # ``embeddings.npy``. CI explicitly opts in via ``--allow-deterministic``.
    if mapping and not efficientnet_only and not args.allow_deterministic:
        raise SystemExit(
            "[embedding_dedup] refusing to proceed: "
            f"{deterministic_fallback_count}/{len(mapping)} record(s) used a "
            f"non-EfficientNetV2 backend (backends={sorted(b for b in backends if b)}). "
            "Pass --allow-deterministic (or set ALLOW_DETERMINISTIC_EMBEDDINGS=1) "
            "for CI runs only."
        )


if __name__ == "__main__":
    main()

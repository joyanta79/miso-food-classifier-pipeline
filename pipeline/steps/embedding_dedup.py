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
    parser.add_argument("--manifest", default="/opt/ml/processing/input/dataset/curated_manifest.json")
    parser.add_argument("--images", default="/opt/ml/processing/input/images")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output/dataset")
    args = parser.parse_args()

    embedding_dim = int(os.environ.get("EMBEDDING_DIM", "1280"))
    cosine = float(os.environ.get("COSINE_DEDUP_THRESHOLD", "0.99"))
    backbone_source = os.environ.get("BACKBONE_SOURCE", "imagenet")
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
    print(f"[embedding_dedup] embeddings={np.asarray(embeddings).shape} "
          f"kept={len(mapping)} backends={backends}")


if __name__ == "__main__":
    main()

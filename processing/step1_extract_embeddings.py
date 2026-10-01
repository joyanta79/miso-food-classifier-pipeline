"""Step 1: embedding extraction and cosine-similarity de-duplication.

Two embedding backends share one contract (a 1280-dimension unit vector per image):

* ``efficientnet`` -- loads real image pixels, resizes to the configured size,
  applies EfficientNetV2 preprocessing, and runs a frozen EfficientNetV2-B0
  backbone (ImageNet weights, ``pooling="avg"`` -> 1280 features). Used when
  TensorFlow and Pillow are available and a real image file exists on disk. This
  is the path exercised by an end-to-end run on real images.
* ``deterministic`` -- a stable SHA-256-derived unit vector, used in local/CI
  mode so the module imports and runs without a TensorFlow download or GPU. It
  never opens image pixels.

The active backend is chosen automatically per record (real backbone when the
image is loadable and TF/Pillow import; deterministic otherwise), and reported in
``embedding_backend`` on each output record so downstream steps and tests can
assert which path ran.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_EMBEDDING_DIM = 1280
DEFAULT_IMAGE_SIZE = 224
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def deterministic_embedding(
    payload: bytes | np.ndarray, dimension: int = DEFAULT_EMBEDDING_DIM
) -> np.ndarray:
    """Derive a stable unit-length embedding from local bytes without ML dependencies."""
    raw = payload.tobytes() if isinstance(payload, np.ndarray) else payload
    values = np.empty(dimension, dtype=np.float32)
    block = b""
    offset = 0
    counter = 0
    while offset < dimension:
        block += hashlib.sha256(counter.to_bytes(4, "big") + raw).digest()
        available = min(dimension - offset, len(block) // 4)
        if available:
            integers = np.frombuffer(block[: available * 4], dtype=">u4").astype(np.float32)
            values[offset : offset + available] = integers / np.float32(2**32 - 1) * 2.0 - 1.0
            offset += available
            block = block[available * 4 :]
        counter += 1
    norm = np.linalg.norm(values)
    return values if norm == 0 else values / norm


@lru_cache(maxsize=1)
def _load_backbone(image_size: int, weights: str | None):
    """Build and cache a frozen EfficientNetV2-B0 feature extractor.

    Returns ``None`` when TensorFlow is unavailable so callers fall back to the
    deterministic embedding. Cached so a batch of images builds the graph once.
    """
    try:
        import tensorflow as tf
    except ImportError:
        return None
    base = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=weights,
        input_shape=(image_size, image_size, 3),
        pooling="avg",
    )
    base.trainable = False
    return base


def _resolve_image_path(record: dict[str, Any], image_root: Path | None) -> Path | None:
    value = record.get("image_path", record.get("key", record.get("id")))
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_absolute() and image_root is not None:
        path = image_root / path
    if path.exists() and path.suffix.lower() in _IMAGE_SUFFIXES:
        return path
    return None


def _real_embedding(path: Path, image_size: int, weights: str | None) -> np.ndarray | None:
    """Run the real EfficientNetV2 backbone on one image; None if deps missing."""
    backbone = _load_backbone(image_size, weights)
    if backbone is None:
        return None
    try:
        import tensorflow as tf
        from PIL import Image
    except ImportError:
        return None
    with Image.open(path) as handle:
        image = handle.convert("RGB").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32)
    preprocessed = tf.keras.applications.efficientnet_v2.preprocess_input(array[np.newaxis, ...])
    features = backbone(preprocessed, training=False)
    vector = np.asarray(features, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else (vector / norm)


def _read_local_payload(record: dict[str, Any], image_root: Path | None) -> bytes:
    value = record.get("image_path", record.get("key", record["id"]))
    path = Path(str(value))
    if not path.is_absolute() and image_root is not None:
        path = image_root / path
    if path.exists():
        return path.read_bytes()
    return str(value).encode("utf-8")


def _embed_record(
    record: dict[str, Any],
    *,
    embedding_dim: int,
    image_size: int,
    image_root: Path | None,
    weights: str | None,
    use_real_backbone: bool,
) -> tuple[np.ndarray, str]:
    """Return (embedding, backend_name) for one record, preferring the real backbone."""
    if use_real_backbone and embedding_dim == DEFAULT_EMBEDDING_DIM:
        path = _resolve_image_path(record, image_root)
        if path is not None:
            vector = _real_embedding(path, image_size, weights)
            if vector is not None and vector.shape[0] == embedding_dim:
                return vector.astype(np.float32), "efficientnet"
    return deterministic_embedding(
        _read_local_payload(record, image_root), embedding_dim
    ), "deterministic"


def deduplicate_cosine(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """Return kept row indices, retaining the first record in each duplicate cluster."""
    if not 0 < threshold <= 1:
        raise ValueError("cosine threshold must be in (0, 1]")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a matrix")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, np.finfo(np.float32).eps)
    kept: list[int] = []
    for index, vector in enumerate(normalized):
        if not kept or np.all(normalized[kept] @ vector < threshold):
            kept.append(index)
    return np.asarray(kept, dtype=np.int64)


def extract_embeddings(
    manifest: Iterable[dict[str, Any]],
    *,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    cosine_dedup_threshold: float = 0.99,
    image_root: str | Path | None = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
    use_real_backbone: bool = True,
    backbone_weights: str | None = "imagenet",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Extract embeddings and return records surviving cosine de-duplication.

    ``use_real_backbone`` enables the EfficientNetV2 path when the image is
    loadable and TensorFlow/Pillow import; otherwise each record falls back to the
    deterministic embedding. Set it False to force the deterministic path (CI).
    """
    records = list(manifest)
    root = Path(image_root) if image_root is not None else None
    if records:
        pairs = [
            _embed_record(
                record,
                embedding_dim=embedding_dim,
                image_size=image_size,
                image_root=root,
                weights=backbone_weights,
                use_real_backbone=use_real_backbone,
            )
            for record in records
        ]
        embeddings = np.vstack([vector for vector, _ in pairs]).astype(np.float32)
        backends = [backend for _, backend in pairs]
    else:
        embeddings = np.empty((0, embedding_dim), dtype=np.float32)
        backends = []
    kept = (
        deduplicate_cosine(embeddings, cosine_dedup_threshold)
        if len(records)
        else np.array([], dtype=int)
    )
    mapping = []
    for index in kept:
        record = dict(records[int(index)])
        record["is_duplicate"] = False
        record["embedding_backend"] = backends[int(index)]
        mapping.append(record)
    return embeddings[kept], mapping


def write_outputs(
    embeddings: np.ndarray, mapping: list[dict[str, Any]], output_dir: str | Path
) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "embeddings.npy", embeddings)
    (directory / "embedding_mapping.json").write_text(
        json.dumps(mapping, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract image embeddings (EfficientNetV2 or deterministic)"
    )
    parser.add_argument("--manifest", default="/opt/ml/processing/input/curated_manifest.json")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--image-root")
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--cosine-dedup-threshold", type=float, default=0.99)
    parser.add_argument(
        "--no-real-backbone",
        action="store_true",
        help="Force the deterministic embedding path even when TensorFlow is available.",
    )
    parser.add_argument(
        "--backbone-weights",
        default="imagenet",
        help=(
            "Weights for EfficientNetV2 (imagenet or a local path); "
            "'none' disables pretrained weights."
        ),
    )
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    weights = None if args.backbone_weights.lower() == "none" else args.backbone_weights
    embeddings, mapping = extract_embeddings(
        manifest,
        embedding_dim=args.embedding_dim,
        cosine_dedup_threshold=args.cosine_dedup_threshold,
        image_root=args.image_root,
        image_size=args.image_size,
        use_real_backbone=not args.no_real_backbone,
        backbone_weights=weights,
    )
    write_outputs(embeddings, mapping, args.output_dir)


if __name__ == "__main__":
    main()

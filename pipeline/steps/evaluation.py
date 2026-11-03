"""Pipeline step: EvaluateModel.

Evaluates the trained Keras model (from the training step's model.tar.gz) on the
cleaned image set and emits an evaluation ``metrics.json`` consumed by the
EvaluateModelQualityGate:

    {
      "accuracy": <float>,
      "emergency_false_positive_rate": <float>,
      "latency_ms": <float>
    }

The model is a fine-tuned EfficientNetV2 classifier saved by train.py. This step
arranges the cleaned images by class, runs inference, and computes accuracy, an
emergency false-positive rate (treating the first class alphabetically as the
"safe/negative" reference), and mean per-image latency.

Python 3.10 compatible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, "/opt/ml/processing/input/pkgcode/pipeline/steps")
import _bootstrap  # noqa: E402

_bootstrap.ensure({"PIL": "pillow", "numpy": "numpy"})

import numpy as np

# Safety caps for untrusted tarball extraction. The model.tar.gz is mounted from
# an S3 prefix that any principal with s3:PutObject on the model-artifacts
# bucket (or a pipeline-parameter override) can write to, so it must be treated
# as untrusted input. These caps bound a potential decompression bomb; the
# per-member validation below bounds path-traversal / link attacks.
_MAX_MEMBERS = 10_000
_MAX_TOTAL_UNCOMPRESSED_BYTES = 10 * 1024 * 1024 * 1024  # 10 GiB
_MAX_MEMBER_BYTES = 5 * 1024 * 1024 * 1024  # 5 GiB per member


def _safe_extract_tar(handle: tarfile.TarFile, dest: Path) -> None:
    """Extract ``handle`` into ``dest`` with the equivalent of ``filter='data'``.

    The step runs on CPython < 3.12 where ``tarfile.extractall`` defaults to
    the fully-trusted legacy filter, so we validate every member explicitly:

    * reject absolute member names and names containing ``..`` components;
    * reject non-regular-file members (symlinks, hardlinks, devices, FIFOs)
      because they can redirect writes outside ``dest`` or escape later;
    * require the resolved destination of each member to stay inside the
      resolved ``dest`` directory;
    * enforce a member-count cap and a cumulative uncompressed-size cap to
      stop decompression bombs.
    """
    dest_root = Path(os.path.realpath(dest))
    total_bytes = 0
    member_count = 0
    safe_members: list[tarfile.TarInfo] = []
    for member in handle:
        member_count += 1
        if member_count > _MAX_MEMBERS:
            raise ValueError(
                f"tarball contains more than {_MAX_MEMBERS} members; refusing to extract"
            )
        name = member.name
        # Reject absolute paths and parent-directory traversal.
        if name.startswith("/") or os.path.isabs(name):
            raise ValueError(f"tarball member has absolute path: {name!r}")
        if os.path.normpath(name).startswith("..") or any(
            part == ".." for part in name.replace("\\", "/").split("/")
        ):
            raise ValueError(f"tarball member escapes extraction root: {name!r}")
        # Reject anything that isn't a plain file or directory. Symlinks and
        # hardlinks could redirect subsequent writes outside dest_root; device
        # and FIFO members have no legitimate use in a model artifact.
        if not (member.isreg() or member.isdir()):
            raise ValueError(
                f"tarball member {name!r} is not a regular file or directory "
                f"(type={member.type!r})"
            )
        # Containment check against the real resolved paths.
        target = os.path.realpath(os.path.join(dest_root, name))
        if target != str(dest_root) and not target.startswith(str(dest_root) + os.sep):
            raise ValueError(f"tarball member {name!r} resolves outside extraction root")
        if member.isreg():
            if member.size < 0 or member.size > _MAX_MEMBER_BYTES:
                raise ValueError(
                    f"tarball member {name!r} declared size {member.size} exceeds cap"
                )
            total_bytes += member.size
            if total_bytes > _MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise ValueError(
                    "tarball cumulative uncompressed size exceeds "
                    f"{_MAX_TOTAL_UNCOMPRESSED_BYTES} bytes; refusing to extract"
                )
        safe_members.append(member)
    # Drop any setuid/setgid/world-writable bits before writing to disk.
    for member in safe_members:
        member.mode = member.mode & 0o755
    handle.extractall(dest_root, members=safe_members)


def _load_model(model_dir: Path):
    import tensorflow as tf

    # train.py saves either a SavedModel dir or model.keras; find it.
    for candidate in ("model.keras", "1", "model"):
        p = model_dir / candidate
        if p.exists():
            return tf.keras.models.load_model(p)
    # Fall back: first loadable entry.
    for p in sorted(model_dir.rglob("*")):
        if p.name.endswith(".keras") or (p.is_dir() and (p / "saved_model.pb").exists()):
            return tf.keras.models.load_model(p)
    raise FileNotFoundError(f"No loadable model found under {model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EvaluateModel step")
    parser.add_argument("--model", default="/opt/ml/processing/input/model")
    parser.add_argument("--images", default="/opt/ml/processing/input/images")
    parser.add_argument(
        "--cleaned", default="/opt/ml/processing/input/dataset/cleaned_dataset.json"
    )
    parser.add_argument("--metrics-out", default="/opt/ml/processing/output/metrics")
    args = parser.parse_args()

    import tensorflow as tf
    from PIL import Image

    image_size = int(os.environ.get("IMAGE_SIZE", "224"))
    model_dir = Path(args.model)

    # Unpack model.tar.gz if present. The archive comes from an S3 prefix that
    # is not trusted end-to-end (see the model-artifacts bucket wiring in
    # pipeline/pipeline.py), so validate every member before writing.
    tarball = model_dir / "model.tar.gz"
    if tarball.exists():
        with tarfile.open(tarball) as handle:
            _safe_extract_tar(handle, model_dir)

    model = _load_model(model_dir)

    cleaned = json.loads(Path(args.cleaned).read_text(encoding="utf-8"))
    labels_sorted = sorted({record["label"] for record in cleaned})
    class_index = {label: i for i, label in enumerate(labels_sorted)}
    safe_label = labels_sorted[0]  # alphabetical first = reference "negative" class

    images_root = Path(args.images)
    correct = 0
    total = 0
    false_positives = 0
    negatives = 0
    latencies = []
    for record in cleaned:
        path = images_root / record["image_path"]
        if not path.exists():
            continue
        with Image.open(path) as handle:
            arr = np.asarray(
                handle.convert("RGB").resize((image_size, image_size)), dtype=np.float32
            )
        x = tf.keras.applications.efficientnet_v2.preprocess_input(arr[np.newaxis, ...])
        started = time.perf_counter()
        probs = model.predict(x, verbose=0)[0]
        latencies.append((time.perf_counter() - started) * 1000)
        pred_idx = int(np.argmax(probs))
        true_idx = class_index[record["label"]]
        total += 1
        if pred_idx == true_idx:
            correct += 1
        # Emergency FP: a truly-safe image predicted as a non-safe class.
        if record["label"] == safe_label:
            negatives += 1
            if pred_idx != class_index[safe_label]:
                false_positives += 1

    accuracy = correct / total if total else 0.0
    emergency_fp_rate = false_positives / negatives if negatives else 0.0
    latency_ms = float(np.mean(latencies)) if latencies else 0.0

    metrics = {
        "accuracy": accuracy,
        "emergency_false_positive_rate": emergency_fp_rate,
        "latency_ms": latency_ms,
        "sample_count": total,
        "classes": labels_sorted,
    }
    metrics_out = Path(args.metrics_out)
    metrics_out.mkdir(parents=True, exist_ok=True)
    (metrics_out / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[evaluation] accuracy={accuracy:.3f} "
        f"emergency_fp_rate={emergency_fp_rate:.3f} latency_ms={latency_ms:.1f} n={total}"
    )


if __name__ == "__main__":
    main()

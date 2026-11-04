"""Pipeline step: EvaluateModel.

Evaluates the trained Keras model (from the training step's model.tar.gz) on the
cleaned image set and emits an evaluation ``metrics.json`` consumed by the
EvaluateModelQualityGate.

The emergency false-positive rate is the hazardous direction defined by
processing/step6_evaluate.py and processing/step7_register_and_baseline.py:
truly-``not_done`` images predicted as ``done``. The reference class names are
taken from CLI arguments (``--done-label`` / ``--not-done-label``) or the
``DONE_LABEL`` / ``NOT_DONE_LABEL`` environment variables, defaulting to
``done`` / ``not_done`` -- never from alphabetical order.

The step fails closed: if any configured image file is missing, if the evaluated
sample count or the per-class counts fall below the configured floors, or if the
``not_done`` reference class contributes no evaluated samples, the step writes a
``metrics.json`` containing values the EvaluateModelQualityGate must reject
(``accuracy=0.0``, ``emergency_false_positive_rate=1.0``) and exits non-zero so
the SageMaker pipeline run fails before the gate is consulted.

Python 3.10 compatible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
import zipfile
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


# Sentinel metric values guaranteed to fail the EvaluateModelQualityGate
# (accuracy >= AccuracyThreshold and emergency_false_positive_rate
# <= FpRateEmergencyThreshold). Written whenever the step aborts fail-closed.
_FAIL_ACCURACY = 0.0
_FAIL_FP_RATE = 1.0


def _write_metrics(metrics_out: Path, metrics: dict) -> None:
    metrics_out.mkdir(parents=True, exist_ok=True)
    (metrics_out / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


def _abort(metrics_out: Path, reason: str, extra: dict) -> None:
    """Write a gate-failing metrics.json and exit non-zero."""
    metrics = {
        "accuracy": _FAIL_ACCURACY,
        "emergency_false_positive_rate": _FAIL_FP_RATE,
        "latency_ms": 0.0,
        "sample_count": int(extra.get("sample_count", 0)),
        "not_done_sample_count": int(extra.get("not_done_sample_count", 0)),
        "done_sample_count": int(extra.get("done_sample_count", 0)),
        "missing_images": int(extra.get("missing_images", 0)),
        "classes": list(extra.get("classes", [])),
        "abort_reason": reason,
    }
    _write_metrics(metrics_out, metrics)
    print(f"[evaluation] ABORT: {reason}", file=sys.stderr)
    sys.exit(1)


def _positive_int(value, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# The Keras loader is code-bearing for legacy HDF5 and SavedModel inputs
# (Lambda layers are restored via ``marshal.loads``), and even the Keras v3
# ``.keras`` archive can embed layer classes whose ``from_config`` executes
# arbitrary code. The ``model-artifacts`` input is sourced from an S3 location
# that is writable by any principal holding ``s3:PutObject`` on that bucket,
# so the artifact must be treated as untrusted:
#
# * Load exactly one expected relative path (``model.keras``). Do not sniff
#   candidate names, do not rglob the mount, do not fall back to HDF5 /
#   SavedModel. train.py (training/train.py) only ever writes ``model.keras``.
# * Validate the archive is a Keras v3 ZIP (``.keras``) with a well-formed
#   ``config.json`` whose referenced layer classes are all on an allowlist
#   of built-in Keras layers used by the brand classifier. Reject anything
#   that mentions ``Lambda``/``TFOpLambda``/custom modules.
# * Load with ``safe_mode=True`` and ``compile=False`` and without any
#   ``custom_objects`` so the Keras-side guards against arbitrary callables
#   remain armed.
_EXPECTED_MODEL_FILENAME = "model.keras"
_MAX_KERAS_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB cap on the archive
_MAX_KERAS_CONFIG_BYTES = 16 * 1024 * 1024  # 16 MiB cap on config.json

# Allowlist of Keras layer ``class_name`` values permitted in the archive's
# ``config.json``. The brand classifier (training/train.py) uses an
# EfficientNetV2 backbone with a small classification head; every layer class
# the saved model can legitimately reference must appear here. Notably
# ``Lambda`` and ``TFOpLambda`` are **not** allowed: those classes can embed
# Python callables that execute at load time.
_ALLOWED_LAYER_CLASS_NAMES = frozenset(
    {
        # Container / wrapper classes used in the saved config.
        "Functional",
        "Sequential",
        "Model",
        "InputLayer",
        # Core / head layers.
        "Dense",
        "Dropout",
        "Flatten",
        "GlobalAveragePooling2D",
        "GlobalMaxPooling2D",
        "Activation",
        "Rescaling",
        "Normalization",
        # Convolution / pooling layers used by EfficientNetV2.
        "Conv2D",
        "DepthwiseConv2D",
        "SeparableConv2D",
        "MaxPooling2D",
        "AveragePooling2D",
        "ZeroPadding2D",
        "BatchNormalization",
        "ReLU",
        "Add",
        "Multiply",
        "Reshape",
        "Concatenate",
    }
)

# Layer class names that are explicitly forbidden regardless of allowlist
# matching. Keeping these out of the allowlist is sufficient, but naming them
# here makes the intent explicit and the rejection error precise.
_FORBIDDEN_LAYER_CLASS_NAMES = frozenset({"Lambda", "TFOpLambda"})


def _walk_class_names(node) -> "list[str]":
    """Collect every ``class_name`` string found anywhere in a Keras config."""
    found: list[str] = []
    if isinstance(node, dict):
        value = node.get("class_name")
        if isinstance(value, str):
            found.append(value)
        for v in node.values():
            found.extend(_walk_class_names(v))
    elif isinstance(node, list):
        for item in node:
            found.extend(_walk_class_names(item))
    return found


def _validate_keras_archive(path: Path) -> None:
    """Validate that ``path`` is a Keras v3 archive with an allowlisted config.

    Raises ``ValueError`` with a precise reason if the archive is not a Keras
    v3 ZIP, if it embeds unexpected members, if ``config.json`` is missing or
    unparseable, or if it references a layer class outside the allowlist.
    The archive is not opened by ``tf.keras.models.load_model`` until this
    function returns successfully.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"cannot stat model artifact {path}: {exc}") from exc
    if size <= 0 or size > _MAX_KERAS_ARCHIVE_BYTES:
        raise ValueError(
            f"model artifact {path} size {size} is outside the accepted range"
        )
    if not zipfile.is_zipfile(path):
        raise ValueError(
            f"model artifact {path} is not a Keras v3 archive (expected a .keras ZIP);"
            " HDF5 and SavedModel inputs are rejected"
        )
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if "config.json" not in names:
            raise ValueError(
                f"model artifact {path} is missing config.json; not a Keras v3 archive"
            )
        # Every member must be a regular, contained entry. ZipFile.extract would
        # normalise paths, but we do not extract: we only read config.json for
        # validation. Still, reject obviously malformed entries defensively.
        for name in names:
            if name.startswith("/") or ".." in name.replace("\\", "/").split("/"):
                raise ValueError(
                    f"model artifact {path} contains unsafe member name {name!r}"
                )
        info = archive.getinfo("config.json")
        if info.file_size < 0 or info.file_size > _MAX_KERAS_CONFIG_BYTES:
            raise ValueError(
                f"model artifact {path} config.json size {info.file_size} exceeds cap"
            )
        with archive.open("config.json") as handle:
            raw = handle.read(_MAX_KERAS_CONFIG_BYTES + 1)
        if len(raw) > _MAX_KERAS_CONFIG_BYTES:
            raise ValueError(
                f"model artifact {path} config.json exceeds cap when decompressed"
            )
        try:
            config = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"model artifact {path} config.json is not valid UTF-8 JSON: {exc}"
            ) from exc
    class_names = _walk_class_names(config)
    if not class_names:
        raise ValueError(
            f"model artifact {path} config.json contained no class_name entries"
        )
    forbidden = sorted({c for c in class_names if c in _FORBIDDEN_LAYER_CLASS_NAMES})
    if forbidden:
        raise ValueError(
            f"model artifact {path} references forbidden layer classes {forbidden!r};"
            " refusing to load (code-bearing layer types are not permitted)"
        )
    unexpected = sorted({c for c in class_names if c not in _ALLOWED_LAYER_CLASS_NAMES})
    if unexpected:
        raise ValueError(
            f"model artifact {path} references layer classes {unexpected!r} that are"
            f" not on the allowlist; refusing to load"
        )


def _load_model(model_dir: Path):
    """Load the training step's Keras v3 model artifact.

    Only ``<model_dir>/model.keras`` is accepted (that is what
    ``training/train.py`` writes). HDF5 (`.h5`), legacy SavedModel directories
    and arbitrary ``rglob`` matches are rejected: those loaders are
    code-bearing in the pinned ``tensorflow==2.14.1`` stack. Loading goes
    through ``safe_mode=True`` with ``compile=False`` and no
    ``custom_objects`` after the archive's ``config.json`` has been validated.
    """
    import tensorflow as tf

    model_path = model_dir / _EXPECTED_MODEL_FILENAME
    if not model_path.is_file():
        raise FileNotFoundError(
            f"expected single Keras v3 artifact at {model_path}; "
            f"HDF5 and SavedModel inputs are not accepted"
        )
    _validate_keras_archive(model_path)
    return tf.keras.models.load_model(
        model_path, compile=False, safe_mode=True, custom_objects=None
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="EvaluateModel step")
    parser.add_argument("--model", default="/opt/ml/processing/input/model")
    parser.add_argument("--images", default="/opt/ml/processing/input/images")
    parser.add_argument(
        "--cleaned", default="/opt/ml/processing/input/dataset/cleaned_dataset.json"
    )
    parser.add_argument("--metrics-out", default="/opt/ml/processing/output/metrics")
    parser.add_argument(
        "--done-label",
        default=os.environ.get("DONE_LABEL", "done"),
        help="Name of the done/safe/positive class.",
    )
    parser.add_argument(
        "--not-done-label",
        default=os.environ.get("NOT_DONE_LABEL", "not_done"),
        help="Name of the not_done/unsafe class (hazardous prediction direction).",
    )
    parser.add_argument(
        "--min-sample-count",
        type=int,
        default=_positive_int(os.environ.get("MIN_SAMPLE_COUNT"), 1),
        help="Minimum evaluated samples; below this the step fails closed.",
    )
    parser.add_argument(
        "--min-per-class-count",
        type=int,
        default=_positive_int(os.environ.get("MIN_PER_CLASS_COUNT"), 1),
        help="Minimum evaluated samples per labelled class; below this the step fails closed.",
    )
    args = parser.parse_args()

    metrics_out = Path(args.metrics_out)
    done_label = args.done_label
    not_done_label = args.not_done_label

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

    # Explicit label semantics: refuse to proceed if the hazardous reference
    # class or the positive class is missing from the dataset. Alphabetical
    # ordering is NOT used to pick the safety direction.
    if not_done_label not in class_index:
        _abort(
            metrics_out,
            (
                f"not_done label {not_done_label!r} is not present in dataset classes "
                f"{labels_sorted!r}; cannot measure the hazardous false-positive direction."
            ),
            {"classes": labels_sorted},
        )
    if done_label not in class_index:
        _abort(
            metrics_out,
            (
                f"done label {done_label!r} is not present in dataset classes "
                f"{labels_sorted!r}."
            ),
            {"classes": labels_sorted},
        )

    images_root = Path(args.images)
    correct = 0
    total = 0
    false_positives = 0  # truly not_done predicted as done (hazardous direction)
    not_done_evaluated = 0
    done_evaluated = 0
    missing_images = 0
    latencies: list[float] = []
    for record in cleaned:
        path = images_root / record["image_path"]
        if not path.exists():
            # Count missing files instead of silently dropping records; the step
            # aborts below when missing_images > 0.
            missing_images += 1
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
        true_label = record["label"]
        if true_label not in class_index:
            _abort(
                metrics_out,
                f"record label {true_label!r} not in known classes {labels_sorted!r}",
                {
                    "classes": labels_sorted,
                    "sample_count": total,
                    "not_done_sample_count": not_done_evaluated,
                    "done_sample_count": done_evaluated,
                    "missing_images": missing_images,
                },
            )
        true_idx = class_index[true_label]
        total += 1
        if pred_idx == true_idx:
            correct += 1
        # Emergency FP: truly-not_done image predicted as done (hazardous direction).
        if true_label == not_done_label:
            not_done_evaluated += 1
            if pred_idx == class_index[done_label]:
                false_positives += 1
        elif true_label == done_label:
            done_evaluated += 1

    extras = {
        "classes": labels_sorted,
        "sample_count": total,
        "not_done_sample_count": not_done_evaluated,
        "done_sample_count": done_evaluated,
        "missing_images": missing_images,
    }

    # Fail closed on dataset-shrinking manipulation or vacuous evaluation.
    if missing_images > 0:
        _abort(
            metrics_out,
            (
                f"{missing_images} cleaned_dataset records had no matching image under the "
                f"images channel; refusing to publish metrics computed over a filtered subset."
            ),
            extras,
        )
    if total < args.min_sample_count:
        _abort(
            metrics_out,
            (
                f"evaluated sample_count={total} is below the configured floor "
                f"min_sample_count={args.min_sample_count}."
            ),
            extras,
        )
    if not_done_evaluated < args.min_per_class_count:
        _abort(
            metrics_out,
            (
                f"not_done evaluated samples={not_done_evaluated} is below "
                f"min_per_class_count={args.min_per_class_count}; cannot produce an "
                f"emergency_false_positive_rate without hazardous-class samples."
            ),
            extras,
        )
    if done_evaluated < args.min_per_class_count:
        _abort(
            metrics_out,
            (
                f"done evaluated samples={done_evaluated} is below "
                f"min_per_class_count={args.min_per_class_count}."
            ),
            extras,
        )

    accuracy = correct / total
    emergency_fp_rate = false_positives / not_done_evaluated
    latency_ms = float(np.mean(latencies)) if latencies else 0.0

    metrics = {
        "accuracy": accuracy,
        "emergency_false_positive_rate": emergency_fp_rate,
        "latency_ms": latency_ms,
        "sample_count": total,
        "not_done_sample_count": not_done_evaluated,
        "done_sample_count": done_evaluated,
        "missing_images": missing_images,
        "classes": labels_sorted,
    }
    _write_metrics(metrics_out, metrics)
    print(
        f"[evaluation] accuracy={accuracy:.3f} "
        f"emergency_fp_rate={emergency_fp_rate:.3f} latency_ms={latency_ms:.1f} "
        f"n={total} not_done_n={not_done_evaluated} done_n={done_evaluated} "
        f"missing_images={missing_images}"
    )


if __name__ == "__main__":
    main()

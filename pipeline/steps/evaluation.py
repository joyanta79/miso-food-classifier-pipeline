"""Pipeline step: EvaluateModel.

Evaluates the trained Keras model (from the training step's model.tar.gz) on the
cleaned image set and emits an evaluation ``metrics.json`` consumed by the
EvaluateModelQualityGate.

The emergency false-positive rate is the hazardous direction defined by
processing/step6_evaluate.py and processing/step7_register_and_baseline.py:
truly-``not_done`` images predicted as ``done``. The reference class names are
taken from CLI arguments (``--done-label`` / ``--not-done-label``) or the
``DONE_LABEL`` / ``NOT_DONE_LABEL`` environment variables, defaulting to
``done`` / ``not_done`` -- never from alphabetical order. The softmax
index-to-class mapping comes from the ``class_names.json`` file written next
to ``model.keras`` by ``training/train.py`` (never from sorted dataset
labels), so an uploader-controlled filename token cannot shift the model's
hazardous output onto the wrong index.

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
    dataset_labels = sorted({record["label"] for record in cleaned})

    # Allowlist of safety labels. Any dataset label outside this set is
    # uploader-controlled noise (filename tokens are parsed verbatim by
    # data_selection.py) and could shift the model's class ordering relative to
    # the dataset-derived view; fail closed before we touch the model.
    allowed_labels = {done_label, not_done_label}
    unexpected = [label for label in dataset_labels if label not in allowed_labels]
    if unexpected:
        _abort(
            metrics_out,
            (
                f"dataset labels {unexpected!r} are not in the configured safety "
                f"allowlist {{{done_label!r}, {not_done_label!r}}}; refusing to "
                "evaluate against uploader-controlled label values."
            ),
            {"classes": dataset_labels},
        )
    if not_done_label not in dataset_labels:
        _abort(
            metrics_out,
            (
                f"not_done label {not_done_label!r} is not present in dataset classes "
                f"{dataset_labels!r}; cannot measure the hazardous false-positive direction."
            ),
            {"classes": dataset_labels},
        )
    if done_label not in dataset_labels:
        _abort(
            metrics_out,
            (
                f"done label {done_label!r} is not present in dataset classes "
                f"{dataset_labels!r}."
            ),
            {"classes": dataset_labels},
        )

    # Index map comes from the MODEL, not from the dataset. train.py writes
    # class_names.json alongside model.keras, mirroring the subdirectory order
    # that Keras assigned to the softmax output. Deriving the index from this
    # file (and asserting it matches the model's output width) closes the
    # inversion where an uploader-reachable filename token (empty label,
    # dot-prefixed label, ...) shifts the dataset-derived mapping by one.
    class_names_path = model_dir / "class_names.json"
    if not class_names_path.exists():
        for candidate in model_dir.rglob("class_names.json"):
            class_names_path = candidate
            break
    if not class_names_path.exists():
        _abort(
            metrics_out,
            (
                "model artifact is missing class_names.json; cannot verify the "
                "softmax-index -> class mapping, so refusing to compute the "
                "hazardous false-positive rate."
            ),
            {"classes": dataset_labels},
        )
    try:
        model_class_names = json.loads(class_names_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        _abort(
            metrics_out,
            f"model class_names.json is not parseable JSON: {error}",
            {"classes": dataset_labels},
        )
    if not isinstance(model_class_names, list) or not all(
        isinstance(name, str) for name in model_class_names
    ):
        _abort(
            metrics_out,
            (
                f"model class_names.json has unexpected shape {type(model_class_names).__name__}; "
                "expected a JSON list of strings matching the model's softmax output ordering."
            ),
            {"classes": dataset_labels},
        )
    if done_label not in model_class_names:
        _abort(
            metrics_out,
            (
                f"done label {done_label!r} is not present in model class_names "
                f"{model_class_names!r}; the trained model cannot represent the "
                "configured safety direction."
            ),
            {"classes": model_class_names},
        )
    if not_done_label not in model_class_names:
        _abort(
            metrics_out,
            (
                f"not_done label {not_done_label!r} is not present in model class_names "
                f"{model_class_names!r}; cannot measure the hazardous false-positive direction."
            ),
            {"classes": model_class_names},
        )
    class_index = {name: i for i, name in enumerate(model_class_names)}
    done_class_index = class_index[done_label]
    not_done_class_index = class_index[not_done_label]

    images_root = Path(args.images)
    correct = 0
    total = 0
    false_positives = 0  # truly not_done predicted as done (hazardous direction)
    not_done_evaluated = 0
    done_evaluated = 0
    missing_images = 0
    model_output_width: int | None = None
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
        # The model's softmax output width must match the class_names.json
        # taken from the artifact. If it does not, the index map we built
        # cannot be trusted.
        if model_output_width is None:
            model_output_width = int(np.shape(probs)[-1])
            if model_output_width != len(model_class_names):
                _abort(
                    metrics_out,
                    (
                        f"model output width {model_output_width} does not match "
                        f"class_names.json length {len(model_class_names)}; the artifact "
                        "is inconsistent with its declared class ordering."
                    ),
                    {
                        "classes": model_class_names,
                        "sample_count": total,
                        "not_done_sample_count": not_done_evaluated,
                        "done_sample_count": done_evaluated,
                        "missing_images": missing_images,
                    },
                )
        pred_idx = int(np.argmax(probs))
        true_label = record["label"]
        # Dataset labels are allowlisted above, but defend in depth in case
        # cleaned_dataset.json is edited between the allowlist check and the
        # evaluation loop.
        if true_label not in allowed_labels:
            _abort(
                metrics_out,
                (
                    f"record label {true_label!r} is not in the configured safety "
                    f"allowlist {{{done_label!r}, {not_done_label!r}}}"
                ),
                {
                    "classes": model_class_names,
                    "sample_count": total,
                    "not_done_sample_count": not_done_evaluated,
                    "done_sample_count": done_evaluated,
                    "missing_images": missing_images,
                },
            )
        true_idx = class_index[true_label]
        # Accuracy is counted only over the two safety classes so a diluting
        # phantom class cannot pad the numerator (defence in depth; the
        # allowlist above already rejects such records).
        total += 1
        if pred_idx == true_idx:
            correct += 1
        # Emergency FP: truly-not_done image predicted as done (hazardous direction).
        if true_label == not_done_label:
            not_done_evaluated += 1
            if pred_idx == done_class_index:
                false_positives += 1
        elif true_label == done_label:
            done_evaluated += 1

    extras = {
        "classes": model_class_names,
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
        "classes": model_class_names,
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

"""SageMaker-compatible EfficientNetV2 training entry point.

TensorFlow is optional locally: when unavailable the script produces a
self-describing local artifact rather than downloading weights or requiring a
GPU image.  SageMaker training images install ``training/requirements.txt`` and
perform actual model fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Name of the manifest file that is written next to each weights file.  The
# manifest records the authoritative epoch number, a SHA-256 digest of the
# weights file, and the SageMaker training-job name that produced the
# checkpoint, so that a later run can verify the checkpoint was produced by
# the same job and has not been swapped out underneath it.
_MANIFEST_SUFFIX = ".manifest.json"


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def efficientnet_weights(config: dict[str, Any]) -> str | None:
    """Never request ImageNet downloads in the configured local test path."""
    local = config.get("local_mode", {})
    if local.get("enabled", False) and not local.get("use_imagenet_weights", False):
        return None
    return "imagenet" if config.get("embedding", {}).get("backbone_source") == "imagenet" else None


def checkpoint_epoch(path: Path | None) -> int:
    if path is None:
        return 0
    match = re.fullmatch(r"epoch-(\d+)\.weights\.h5", path.name)
    return int(match.group(1)) if match else 0


def latest_checkpoint(directory: str | Path) -> Path | None:
    return max(Path(directory).glob("epoch-*.weights.h5"), key=checkpoint_epoch, default=None)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path_for(weights_path: Path) -> Path:
    # epoch-NNN.weights.h5 -> epoch-NNN.weights.h5.manifest.json
    return weights_path.with_name(weights_path.name + _MANIFEST_SUFFIX)


def _current_training_job_name() -> str | None:
    """Return the SageMaker-provided training job name, if any.

    SageMaker Training containers expose this via ``TRAINING_JOB_NAME``.
    When the variable is absent (local development, unit tests) binding is
    opportunistic: digest verification still runs, but the stricter
    producing-job check is skipped so local resume continues to work.
    """
    value = os.environ.get("TRAINING_JOB_NAME")
    return value.strip() if value and value.strip() else None


def write_checkpoint_manifest(weights_path: Path, epoch: int) -> Path:
    """Write an authenticated manifest alongside a saved weights file.

    The manifest records the epoch number, a SHA-256 digest of the weights
    file and the current training-job name (when known). Writing is atomic
    so a partially-written manifest can never be observed on resume.
    """
    weights_path = Path(weights_path)
    manifest = {
        "epoch": int(epoch),
        "sha256": _sha256_file(weights_path),
        "filename": weights_path.name,
        "training_job_name": _current_training_job_name(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = _manifest_path_for(weights_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: temp file in the same directory + os.replace.
    fd, tmp_name = tempfile.mkstemp(
        prefix=weights_path.name + ".", suffix=".tmp", dir=str(manifest_path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, manifest_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return manifest_path


def _verify_checkpoint(weights_path: Path, *, current_job: str | None) -> tuple[bool, str]:
    """Return (ok, reason) describing whether a checkpoint can be trusted.

    A checkpoint is only trusted when:
      * its manifest exists and parses as JSON;
      * the manifest's filename and epoch agree with the weights filename;
      * the SHA-256 of the weights file matches the manifest digest;
      * when a SageMaker training-job name is available, the manifest's
        training_job_name matches it (defence against another job writing
        into this prefix).
    """
    manifest_path = _manifest_path_for(weights_path)
    if not manifest_path.is_file():
        return False, "missing manifest"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"unreadable manifest: {exc}"
    if not isinstance(manifest, dict):
        return False, "manifest is not an object"
    name_match = re.fullmatch(r"epoch-(\d+)\.weights\.h5", weights_path.name)
    if not name_match:
        return False, "unexpected checkpoint filename"
    filename_epoch = int(name_match.group(1))
    try:
        manifest_epoch = int(manifest.get("epoch"))
    except (TypeError, ValueError):
        return False, "manifest epoch is not an integer"
    if manifest_epoch != filename_epoch:
        return False, "manifest epoch does not match filename"
    expected_digest = manifest.get("sha256")
    if not isinstance(expected_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        return False, "manifest sha256 missing or malformed"
    actual_digest = _sha256_file(weights_path)
    if actual_digest != expected_digest:
        return False, "weights digest does not match manifest"
    manifest_job = manifest.get("training_job_name")
    if current_job is not None:
        if not isinstance(manifest_job, str) or manifest_job != current_job:
            return False, "checkpoint not produced by current training job"
    return True, "ok"


def trusted_checkpoint(directory: str | Path) -> tuple[Path | None, int]:
    """Return the highest-epoch checkpoint whose manifest validates.

    Iterates checkpoints in descending epoch order and picks the first one
    that passes :func:`_verify_checkpoint`.  Returns ``(None, 0)`` when no
    trusted checkpoint is available.  Rejection reasons are printed so an
    operator can tell poisoning attempts apart from a cold start.
    """
    directory = Path(directory)
    candidates = sorted(
        directory.glob("epoch-*.weights.h5"), key=checkpoint_epoch, reverse=True
    )
    current_job = _current_training_job_name()
    for candidate in candidates:
        ok, reason = _verify_checkpoint(candidate, current_job=current_job)
        if ok:
            return candidate, checkpoint_epoch(candidate)
        print(f"[train] ignoring untrusted checkpoint {candidate.name}: {reason}", flush=True)
    return None, 0


def _local_artifact(
    model_dir: Path,
    checkpoint_dir: Path,
    training_dir: Path,
    *,
    epochs: int,
    batch_size: int,
    weights: str | None,
) -> dict[str, Any]:
    """Provide a deterministic no-TensorFlow path for local tests and CI."""
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": "local-smoke",
        "training_dir": str(training_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "efficientnet_weights": weights,
        "resume_supported": True,
    }
    (model_dir / "local_model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (checkpoint_dir / "checkpoint.json").write_text(
        json.dumps({"checkpoint_every_epochs": 5, "resume_supported": True}, indent=2),
        encoding="utf-8",
    )
    return payload


def train_model(
    *,
    config: dict[str, Any],
    training_dir: str | Path,
    validation_dir: str | Path,
    model_dir: str | Path,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    brand_id: str,
    epochs: int | None = None,
) -> dict[str, Any]:
    """Train, checkpoint every configured interval, and resume when possible."""
    training = config.get("training", {})
    image_size = int(training.get("image_size", 224))
    batch_size = int(training.get("batch_size", 32))
    total_epochs = int(epochs or training.get("epochs", 50))
    checkpoint_every = int(training.get("checkpoint_every_epochs", 5))
    learning_rate = float(training.get("learning_rate", 0.001))
    train_path, validation_path = Path(training_dir), Path(validation_dir)
    model_path, checkpoint_path, output_path = (
        Path(model_dir),
        Path(checkpoint_dir),
        Path(output_dir),
    )
    weights = efficientnet_weights(config)

    try:
        import tensorflow as tf
    except ImportError:
        result = _local_artifact(
            model_path,
            checkpoint_path,
            train_path,
            epochs=total_epochs,
            batch_size=batch_size,
            weights=weights,
        )
        result.update({"brand_id": brand_id, "resumed_from_epoch": 0, "final_epoch": total_epochs})
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "training_metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result

    if not train_path.is_dir() or not validation_path.is_dir():
        raise ValueError(
            "SageMaker train and validation channels must be class-labelled image directories."
        )
    seed = int(config.get("local_mode", {}).get("seed", 2026))
    tf.keras.utils.set_random_seed(seed)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        validation_path, image_size=(image_size, image_size), batch_size=batch_size, shuffle=False
    )
    if train_ds.class_names != validation_ds.class_names:
        raise ValueError("Training and validation datasets must have identical class names.")

    base = tf.keras.applications.EfficientNetV2B0(
        include_top=False, weights=weights, input_shape=(image_size, image_size, 3), pooling="avg"
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    values = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
    outputs = tf.keras.layers.Dense(len(train_ds.class_names), activation="softmax")(
        base(values, training=False)
    )
    model = tf.keras.Model(inputs, outputs, name="food_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint, initial_epoch = trusted_checkpoint(checkpoint_path)
    # Reject out-of-range resume state instead of silently iterating zero
    # epochs (which would otherwise save the loaded-but-untrained weights
    # as the released artifact).
    if initial_epoch < 0 or initial_epoch >= total_epochs:
        raise ValueError(
            f"Refusing to resume: trusted checkpoint reports initial_epoch={initial_epoch}, "
            f"which is not in the valid range [0, {total_epochs}). Delete the stale "
            "checkpoint prefix or raise training.epochs before retrying."
        )
    if checkpoint:
        model.load_weights(checkpoint)

    class PeriodicCheckpoint(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
            if (epoch + 1) % checkpoint_every == 0:
                weights_file = checkpoint_path / f"epoch-{epoch + 1:03d}.weights.h5"
                self.model.save_weights(weights_file)
                # Persist an authenticated manifest alongside the weights so
                # that a later resume can verify the file was produced by
                # this job and has not been tampered with.
                write_checkpoint_manifest(weights_file, epoch + 1)

    history = model.fit(
        train_ds.prefetch(tf.data.AUTOTUNE),
        validation_data=validation_ds.prefetch(tf.data.AUTOTUNE),
        initial_epoch=initial_epoch,
        epochs=total_epochs,
        callbacks=[PeriodicCheckpoint(), tf.keras.callbacks.TerminateOnNaN()],
        verbose=2,
    )
    # Guard against the "resume loaded weights, then fit() iterated an empty
    # epoch range" case: without this, model.save() would publish whatever
    # weights were loaded from the checkpoint directory as the released
    # artifact with no training evidence recorded.
    if not getattr(history, "history", None):
        raise RuntimeError(
            "model.fit produced no history entries; refusing to save a model that "
            f"was not trained (initial_epoch={initial_epoch}, epochs={total_epochs})."
        )
    model_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save(model_path / "model.keras")
    (model_path / "class_names.json").write_text(json.dumps(train_ds.class_names), encoding="utf-8")
    result = {
        "brand_id": brand_id,
        "resumed_from_epoch": initial_epoch,
        "final_epoch": total_epochs,
        "efficientnet_weights": weights,
        "metrics": {metric: values[-1] for metric, values in history.history.items() if values},
    }
    (output_path / "training_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a per-brand food classifier.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--brand-id", required=True)
    parser.add_argument(
        "--train-dir", default=os.getenv("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    )
    parser.add_argument(
        "--validation-dir",
        default=os.getenv("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"),
    )
    parser.add_argument("--model-dir", default=os.getenv("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument(
        "--checkpoint-dir", default=os.getenv("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")
    )
    parser.add_argument(
        "--output-dir", default=os.getenv("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    )
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        json.dumps(
            train_model(
                config=load_config(arguments.config),
                training_dir=arguments.train_dir,
                validation_dir=arguments.validation_dir,
                model_dir=arguments.model_dir,
                checkpoint_dir=arguments.checkpoint_dir,
                output_dir=arguments.output_dir,
                brand_id=arguments.brand_id,
                epochs=arguments.epochs,
            ),
            indent=2,
        )
    )

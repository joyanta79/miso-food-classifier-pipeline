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
from pathlib import Path
from typing import Any

import yaml


CHECKPOINT_MANIFEST_NAME = "checkpoint_manifest.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_training_job_name() -> str:
    """Return the identity of the currently executing training job.

    SageMaker exposes the job name via ``TRAINING_JOB_NAME``. Locally we fall
    back to an explicit override so tests remain deterministic.
    """
    return (
        os.environ.get("TRAINING_JOB_NAME")
        or os.environ.get("SM_TRAINING_JOB_NAME")
        or os.environ.get("CONTINUUM_TRAINING_JOB_NAME")
        or ""
    )


def _load_checkpoint_manifest(directory: Path) -> dict[str, Any]:
    manifest_path = directory / CHECKPOINT_MANIFEST_NAME
    if not manifest_path.is_file():
        return {"entries": {}}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8")) or {}
    except (OSError, json.JSONDecodeError):
        return {"entries": {}}
    if not isinstance(data, dict):
        return {"entries": {}}
    data.setdefault("entries", {})
    if not isinstance(data["entries"], dict):
        data["entries"] = {}
    return data


def _write_checkpoint_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    manifest_path = directory / CHECKPOINT_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _verified_resume_checkpoint(
    directory: Path,
    *,
    training_job_name: str,
    total_epochs: int,
) -> Path | None:
    """Select a resume checkpoint only if it is in the manifest written by *this* job.

    Returns ``None`` when no legitimate resume candidate exists. Raises
    ``RuntimeError`` when an unverified or out-of-range checkpoint is present,
    which indicates tampering (e.g. an attacker-planted ``epoch-999.weights.h5``
    synced into ``SM_CHECKPOINT_DIR``).
    """
    candidates = sorted(
        Path(directory).glob("epoch-*.weights.h5"), key=checkpoint_epoch
    )
    if not candidates:
        return None
    manifest = _load_checkpoint_manifest(Path(directory))
    entries: dict[str, Any] = manifest.get("entries", {})
    best: Path | None = None
    for candidate in candidates:
        epoch = checkpoint_epoch(candidate)
        if epoch <= 0 or epoch >= total_epochs:
            raise RuntimeError(
                f"Refusing to resume: checkpoint {candidate.name!r} has epoch {epoch} "
                f"outside the valid range (1..{total_epochs - 1})."
            )
        entry = entries.get(candidate.name)
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"Refusing to resume: checkpoint {candidate.name!r} is not in the "
                "manifest produced by this training job."
            )
        expected_job = entry.get("training_job_name", "")
        if not training_job_name or expected_job != training_job_name:
            raise RuntimeError(
                f"Refusing to resume: checkpoint {candidate.name!r} was written by "
                f"job {expected_job!r}, not the current job {training_job_name!r}."
            )
        expected_sha = entry.get("sha256", "")
        actual_sha = _file_sha256(candidate)
        if not expected_sha or expected_sha != actual_sha:
            raise RuntimeError(
                f"Refusing to resume: checkpoint {candidate.name!r} digest mismatch "
                f"(expected {expected_sha!r}, got {actual_sha!r})."
            )
        best = candidate
    return best


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
    training_job_name = _current_training_job_name()
    checkpoint = _verified_resume_checkpoint(
        checkpoint_path,
        training_job_name=training_job_name,
        total_epochs=total_epochs,
    )
    initial_epoch = checkpoint_epoch(checkpoint)
    if initial_epoch >= total_epochs:
        raise RuntimeError(
            f"Refusing to resume: initial_epoch={initial_epoch} is not less than "
            f"total_epochs={total_epochs}; no training would occur."
        )
    if checkpoint:
        model.load_weights(checkpoint)

    class PeriodicCheckpoint(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
            if (epoch + 1) % checkpoint_every == 0:
                weight_path = checkpoint_path / f"epoch-{epoch + 1:03d}.weights.h5"
                self.model.save_weights(weight_path)
                manifest = _load_checkpoint_manifest(checkpoint_path)
                manifest["entries"][weight_path.name] = {
                    "sha256": _file_sha256(weight_path),
                    "training_job_name": training_job_name,
                    "epoch": epoch + 1,
                }
                _write_checkpoint_manifest(checkpoint_path, manifest)

    history = model.fit(
        train_ds.prefetch(tf.data.AUTOTUNE),
        validation_data=validation_ds.prefetch(tf.data.AUTOTUNE),
        initial_epoch=initial_epoch,
        epochs=total_epochs,
        callbacks=[PeriodicCheckpoint(), tf.keras.callbacks.TerminateOnNaN()],
        verbose=2,
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

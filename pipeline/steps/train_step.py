"""Training-step entry point: TrainBrandClassifier.

The SageMaker Training job runs this. It receives:
  * channel ``images``   -> the raw brand images
  * channel ``dataset``  -> cleaned_dataset.json from the quality step
  * channel ``code``     -> the training package (this file + train.py + config)

It arranges the cleaned images into ``train/<class>/`` and ``validation/<class>/``
directories (80/20 split, deterministic), then calls the proven train_model().
The model.keras + class_names.json land in SM_MODEL_DIR, which SageMaker packages
into model.tar.gz.

Python 3.10 compatible.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, "/opt/ml/input/data/pkgcode")

from training.train import train_model  # noqa: E402

# Strict allowlist for manifest ``label`` values that will become a filesystem
# component. Keep this in sync with the pipeline-step validator in
# pipeline/steps/evaluation.py and with the ``done``/``not_done`` labels
# configured in config.yaml. The manifest is read from an S3 prefix that any
# principal with s3:PutObject on the training bucket can write to, so labels
# are untrusted input (CWE-22).
_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _allowed_labels() -> set[str]:
    """Return the configured class-name allowlist for manifest labels.

    deploy_pipeline.py propagates ``DONE_LABEL`` / ``NOT_DONE_LABEL`` from
    config.yaml to the step's environment; the defaults match the shipped
    config (``done`` / ``not_done``).
    """
    return {
        os.environ.get("DONE_LABEL", "done"),
        os.environ.get("NOT_DONE_LABEL", "not_done"),
    }


def _validate_label(label: object) -> str:
    """Validate a manifest ``label`` before using it as a path component.

    Rejects anything that is not a non-empty string, that contains a path
    separator or ``..`` traversal segment, that is absolute, or that does
    not match ``_LABEL_PATTERN``. Also rejects labels outside the configured
    class-name allowlist so a forged manifest cannot introduce a new class
    directory even if the string survives the shape check.
    """
    if not isinstance(label, str) or not label:
        raise ValueError(f"[train_step] manifest label must be a non-empty string, got {label!r}")
    # Reject absolute values and anything containing a path separator or a
    # parent-directory segment. pathlib.Path discards the left operand when
    # the right side is absolute, so this is the primary traversal gate.
    if os.path.isabs(label) or label.startswith(("/", "\\")):
        raise ValueError(f"[train_step] manifest label is absolute: {label!r}")
    if "/" in label or "\\" in label or "\x00" in label:
        raise ValueError(f"[train_step] manifest label contains a path separator: {label!r}")
    normalised = label.replace("\\", "/")
    if normalised in (".", "..") or any(
        part in ("", ".", "..") for part in normalised.split("/")
    ):
        raise ValueError(f"[train_step] manifest label contains a traversal segment: {label!r}")
    if not _LABEL_PATTERN.match(label):
        raise ValueError(
            f"[train_step] manifest label {label!r} does not match the required pattern "
            f"{_LABEL_PATTERN.pattern}"
        )
    allowed = _allowed_labels()
    if label not in allowed:
        raise ValueError(
            f"[train_step] manifest label {label!r} is not in the configured class-name "
            f"allowlist {sorted(allowed)!r}"
        )
    return label


def _validate_contained(path: Path, root: Path, what: str) -> Path:
    """Return ``path`` after verifying it stays inside ``root`` on the real FS.

    Mirrors the containment check used for tar members in
    pipeline/steps/evaluation.py:_safe_extract_tar: resolve both sides with
    ``os.path.realpath`` so symlinks cannot be used to escape the sandbox,
    then require the resolved child to equal or sit under the resolved root.
    """
    resolved_root = os.path.realpath(root)
    resolved = os.path.realpath(path)
    if resolved != resolved_root and not resolved.startswith(resolved_root + os.sep):
        raise ValueError(
            f"[train_step] {what} {str(path)!r} resolves to {resolved!r} outside "
            f"{resolved_root!r}"
        )
    return Path(resolved)


def _dest_filename(record: dict) -> str:
    """Build a collision-free destination filename for ``record``.

    The manifest ``id`` is unique per record (relative key); we hash it to a
    short digest and keep the original suffix so Keras' file-type sniffing
    still works. Falls back to the image_path basename only when ``id`` is
    missing (legacy manifests).
    """
    rel = record.get("image_path", "")
    suffix = PurePosixPath(str(rel)).suffix or Path(record.get("id", "")).suffix
    identity = str(record.get("id") or rel)
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
    return f"{digest}{suffix}"


def _prep_class_dirs(
    images_root: Path, cleaned: list[dict], work: Path, val_fraction: float = 0.2
) -> None:
    # Validate every label before using it as a filesystem component (CWE-22).
    # A forged cleaned_dataset.json can set ``label`` to an absolute path or a
    # ``..``-containing string; pathlib's absolute-join behaviour would then
    # drop ``split_root`` and let the subsequent mkdir/shutil.copy write to an
    # attacker-chosen directory inside the training container.
    by_class: dict[str, list[dict]] = {}
    for record in cleaned:
        label = _validate_label(record.get("label"))
        by_class.setdefault(label, []).append(record)
    train_root = work / "train"
    val_root = work / "validation"
    # Create the work roots up front so we can resolve them for the
    # containment check below.
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)
    images_root_resolved = Path(os.path.realpath(images_root))
    copied_counts: dict[str, int] = {}
    for label, records in by_class.items():
        records = sorted(records, key=lambda r: r["image_path"])
        n_val = max(1, int(len(records) * val_fraction))
        val = records[:n_val]
        train = records[n_val:] or records  # never leave train empty
        label_copied = 0
        for split_root, split in ((train_root, train), (val_root, val)):
            dest = split_root / label
            # Belt-and-braces: even with the label validator above, verify
            # that the joined destination resolves inside the split root
            # before any mkdir / copy touches the filesystem.
            _validate_contained(dest, split_root, "prepared class directory")
            dest.mkdir(parents=True, exist_ok=True)
            for record in split:
                src = images_root / record["image_path"]
                if not src.exists():
                    # Fail closed: the gate metrics are computed over the
                    # manifest, so a missing source means the materialised
                    # dataset would diverge from what the gates approved.
                    raise FileNotFoundError(
                        f"[train_step] manifest record {record.get('id', record['image_path'])!r} "
                        f"resolves to missing file {src}; refusing to train on a dataset "
                        "that disagrees with the selection gate metrics."
                    )
                # Reject manifest image_path values that escape the images
                # channel (e.g. absolute paths or ``..`` segments). This
                # closes the companion read-side CWE-22 noted in the finding.
                _validate_contained(src, images_root_resolved, "manifest source file")
                # Derive the destination filename from a unique record id so
                # two records with the same basename (e.g. a/x.jpg and b/x.jpg)
                # do not overwrite each other in the prepared class dir.
                shutil.copy(src, dest / _dest_filename(record))
                label_copied += 1
        copied_counts[label] = label_copied
    manifest_counts = {label: len(records) for label, records in by_class.items()}
    if copied_counts != manifest_counts:
        raise RuntimeError(
            "[train_step] copied-file counts do not match manifest counts: "
            f"manifest={manifest_counts} copied={copied_counts}"
        )
    print(f"[train_step] prepared class dirs: {copied_counts}")


def main() -> None:
    images_root = Path(os.getenv("SM_CHANNEL_IMAGES", "/opt/ml/input/data/images"))
    dataset_root = Path(os.getenv("SM_CHANNEL_DATASET", "/opt/ml/input/data/dataset"))
    model_dir = Path(os.getenv("SM_MODEL_DIR", "/opt/ml/model"))
    checkpoint_dir = Path(os.getenv("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints"))
    output_dir = Path(os.getenv("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    brand_id = os.environ.get("BRAND_ID", "white-castle")

    cleaned = json.loads((dataset_root / "cleaned_dataset.json").read_text(encoding="utf-8"))

    work = Path("/opt/ml/input/prepared")
    _prep_class_dirs(images_root, cleaned, work)

    # Build the config inline so the runner does not depend on config.yaml being
    # staged; force the real ImageNet backbone and a small epoch count for smoke.
    epochs = int(os.environ.get("EPOCHS", "3"))
    batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    config = {
        "training": {
            "image_size": 224,
            "batch_size": batch_size,
            "epochs": epochs,
            "checkpoint_every_epochs": 5,
            "learning_rate": 0.001,
        },
        "embedding": {"backbone_source": "imagenet"},
        "local_mode": {"enabled": False, "use_imagenet_weights": True, "seed": 2026},
    }

    result = train_model(
        config=config,
        training_dir=work / "train",
        validation_dir=work / "validation",
        model_dir=model_dir,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        brand_id=brand_id,
        epochs=epochs,
    )
    print(f"[train_step] done: {json.dumps(result.get('metrics', {}))}")


if __name__ == "__main__":
    main()

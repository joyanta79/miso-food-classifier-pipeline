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

# Strict allowlist pattern for any manifest ``label`` that is used as a
# filesystem directory component. Must start with an alphanumeric, be at most
# 64 chars long, and contain only ``A-Za-z0-9._-``. ``..`` (and anything whose
# first byte is ``.``) is excluded by the leading ``[A-Za-z0-9]`` anchor.
_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _configured_label_allowlist() -> set[str]:
    """Return the configured ``done``/``not_done`` label names, if provided.

    When ``DONE_LABEL`` / ``NOT_DONE_LABEL`` are set in the environment they
    constitute an explicit allowlist. If both are unset we fall back to the
    regex check only, since the deployed definition does not forward them to
    this step today.
    """
    allow: set[str] = set()
    for key in ("DONE_LABEL", "NOT_DONE_LABEL"):
        value = os.environ.get(key, "").strip()
        if value:
            allow.add(value)
    return allow


def _is_safe_label(label: str, allowlist: set[str]) -> bool:
    """True iff ``label`` is safe to use as a single path component.

    A non-empty configured ``allowlist`` is authoritative: only members pass.
    Otherwise accept labels that match the strict allowlist pattern. ``..``
    and any value containing a path separator or NUL is always rejected.
    """
    if not isinstance(label, str) or not label:
        return False
    if label in (".", "..") or "/" in label or "\\" in label or "\x00" in label:
        return False
    if allowlist:
        return label in allowlist
    return _LABEL_PATTERN.fullmatch(label) is not None


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
    by_class: dict[str, list[dict]] = {}
    allowlist = _configured_label_allowlist()
    for record in cleaned:
        label = record.get("label")
        # Validate the label *before* it is ever used as a path component.
        # This protects against a filename-derived ``..`` / separator / NUL
        # label that would otherwise escape the per-class split directory
        # (CWE-22). The manifest should already be sanitised upstream, but we
        # fail closed here as a defence-in-depth check.
        if not _is_safe_label(label, allowlist):
            raise ValueError(
                f"[train_step] refusing to prepare class directory for unsafe "
                f"label {label!r}: labels must be in the configured allowlist "
                "or match ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ (and must not be "
                "'.' or '..')."
            )
        by_class.setdefault(label, []).append(record)
    train_root = work / "train"
    val_root = work / "validation"
    # Resolve the containment root once; every per-class ``dest`` must stay
    # inside it after symlink/`..` resolution.
    work_real = os.path.realpath(work)
    for label, records in by_class.items():
        records = sorted(records, key=lambda r: r["image_path"])
        n_val = max(1, int(len(records) * val_fraction))
        val = records[:n_val]
        train = records[n_val:] or records  # never leave train empty
        for split_root, split in ((train_root, train), (val_root, val)):
            dest = split_root / label
            # Containment check: ensure the resolved destination is still
            # inside the prepared work root. Reject (do not create) if the
            # label somehow escaped despite the allowlist above.
            dest_real = os.path.realpath(dest)
            if (
                dest_real != work_real
                and not dest_real.startswith(work_real + os.sep)
            ):
                raise ValueError(
                    f"[train_step] refusing to create class directory "
                    f"{dest!r} for label {label!r}: resolved path "
                    f"{dest_real!r} escapes the prepared root {work_real!r}."
                )
            # Also reject the trivial case where the label collapses to the
            # split root itself (e.g. label='.'), which would mix every
            # record of this class into the split root rather than a class
            # subdirectory.
            if dest_real == os.path.realpath(split_root):
                raise ValueError(
                    f"[train_step] refusing to use split root as class dir "
                    f"for label {label!r} (resolved to {dest_real!r})."
                )
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
                # Derive the destination filename from a unique record id so
                # two records with the same basename (e.g. a/x.jpg and b/x.jpg)
                # do not overwrite each other in the prepared class dir.
                shutil.copy(src, dest / _dest_filename(record))
    manifest_counts = {label: len(records) for label, records in by_class.items()}
    # Count files actually present under each ``train/<class>/`` and
    # ``validation/<class>/`` directory rather than counting copy operations,
    # so that a label whose directory escaped the split root cannot still
    # pass the integrity check (CWE-22 defence-in-depth).
    materialised_counts: dict[str, int] = {}
    for label in by_class:
        train_files = (
            sum(1 for p in (train_root / label).iterdir() if p.is_file())
            if (train_root / label).is_dir()
            else 0
        )
        val_files = (
            sum(1 for p in (val_root / label).iterdir() if p.is_file())
            if (val_root / label).is_dir()
            else 0
        )
        materialised_counts[label] = train_files + val_files
    if materialised_counts != manifest_counts:
        raise RuntimeError(
            "[train_step] materialised class-directory counts do not match "
            f"manifest counts: manifest={manifest_counts} "
            f"materialised={materialised_counts}"
        )
    print(f"[train_step] prepared class dirs: {materialised_counts}")


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

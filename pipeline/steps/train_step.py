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

# Defence-in-depth: labels reach this step via cleaned_dataset.json after
# flowing through the selection -> embedding -> cleanlab pipeline. The upstream
# selection step (pipeline/steps/data_selection.py) is the primary validator,
# but any label that is used as a filesystem directory component here must
# itself be a safe single path component, so this step rejects anything else.
_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,63}$")


def _is_safe_label(label: object) -> bool:
    """Return True when ``label`` is safe to use as a class directory name."""
    return isinstance(label, str) and bool(_LABEL_PATTERN.fullmatch(label))


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
    for record in cleaned:
        label = record["label"]
        # Defence-in-depth against CWE-22: refuse any label that is not a
        # strict single safe path component. Upstream data_selection.select()
        # already enforces this allow-list, but revalidating here means a
        # cleaned_dataset.json produced by any other path (manual edit,
        # legacy manifest, future refactor) cannot smuggle ``.``, ``..``,
        # empty strings, separators or control characters into a filesystem
        # path and escape the prepared class-directory tree.
        if not _is_safe_label(label):
            raise ValueError(
                f"[train_step] refusing to use unsafe label {label!r} as a "
                "class directory component; expected a single path segment "
                "matching ^[A-Za-z0-9][A-Za-z0-9-]{0,63}$."
            )
        by_class.setdefault(label, []).append(record)
    train_root = work / "train"
    val_root = work / "validation"
    work_resolved = work.resolve()
    copied_counts: dict[str, int] = {}
    for label, records in by_class.items():
        records = sorted(records, key=lambda r: r["image_path"])
        n_val = max(1, int(len(records) * val_fraction))
        val = records[:n_val]
        train = records[n_val:] or records  # never leave train empty
        label_copied = 0
        for split_root, split in ((train_root, train), (val_root, val)):
            dest = split_root / label
            # Containment check: even with a validated label, assert that the
            # resolved destination stays under ``work``. This guards against
            # symlink or future-code surprises in the staging tree.
            try:
                dest.resolve().relative_to(work_resolved)
            except ValueError as exc:
                raise ValueError(
                    f"[train_step] destination {dest} for label {label!r} "
                    f"escapes the staging root {work_resolved}; refusing to "
                    "materialise the training set."
                ) from exc
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

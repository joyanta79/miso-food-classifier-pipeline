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

# Labels become filesystem directory components, so they must be strict,
# single-segment names. The regex requires an alphanumeric leading character
# and allows only [A-Za-z0-9._-] afterwards; '.'/'..' are rejected explicitly
# below because they would collapse to the parent directory. Anything with a
# path separator or that is absolute is also rejected before the regex check.
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_label(label: object) -> str:
    """Return ``label`` if it is a safe single-segment directory component.

    Raises ``ValueError`` otherwise. The manifest field is attacker-
    controllable (see cleaned_dataset.json rewrite), so it must not be used
    as a path component without this check.
    """
    if not isinstance(label, str) or not label:
        raise ValueError(f"[train_step] manifest label must be a non-empty string: {label!r}")
    if label in (".", ".."):
        raise ValueError(f"[train_step] manifest label {label!r} is a reserved path component")
    if os.path.isabs(label) or "/" in label or "\\" in label or "\x00" in label:
        raise ValueError(
            f"[train_step] manifest label {label!r} contains a path separator or is absolute"
        )
    if not _LABEL_RE.match(label):
        raise ValueError(
            f"[train_step] manifest label {label!r} does not match ^[A-Za-z0-9][A-Za-z0-9._-]*$"
        )
    return label


def _assert_within(child: Path, root: Path) -> str:
    """Resolve ``child`` and ``root`` and require ``child`` to stay inside ``root``.

    Returns the resolved real path of ``child`` on success. Raises
    ``ValueError`` when the resolution escapes ``root`` (via '..', symlinks,
    or an absolute component swallowing the join).
    """
    root_real = os.path.realpath(root)
    child_real = os.path.realpath(child)
    if child_real != root_real and not child_real.startswith(root_real + os.sep):
        raise ValueError(
            f"[train_step] path {child!r} resolves to {child_real!r}, outside {root_real!r}"
        )
    return child_real

sys.path.insert(0, "/opt/ml/input/data/pkgcode")

from training.train import train_model  # noqa: E402


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
    # Group records by their validated label. Rejecting the whole run on an
    # invalid label (rather than skipping) matches the fail-closed posture of
    # the surrounding code: the selection gate was computed over the full
    # manifest, so silently dropping records would diverge from it.
    by_class: dict[str, list[dict]] = {}
    for record in cleaned:
        label = _validate_label(record.get("label"))
        by_class.setdefault(label, []).append(record)
    work.mkdir(parents=True, exist_ok=True)
    work_real = os.path.realpath(work)
    train_root = work / "train"
    val_root = work / "validation"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)
    # Ensure the split roots themselves are inside work (defence in depth; they
    # are constructed from a literal join above, but a future refactor could
    # change that).
    _assert_within(train_root, work_real)
    _assert_within(val_root, work_real)

    expected_dests: set[str] = set()
    copied_dests: set[str] = set()
    for label, records in by_class.items():
        records = sorted(records, key=lambda r: r["image_path"])
        n_val = max(1, int(len(records) * val_fraction))
        val = records[:n_val]
        train = records[n_val:] or records  # never leave train empty
        for split_root, split in ((train_root, train), (val_root, val)):
            dest = split_root / label
            # Re-check containment against ``work`` after the join: validated
            # labels cannot traverse, but this is the invariant we rely on.
            dest_real = _assert_within(dest, work_real)
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
                target = dest / _dest_filename(record)
                # Containment check for the final file path too: the filename
                # is a sha1 digest + suffix today, but keep the guarantee
                # independent of _dest_filename's implementation details.
                target_real = os.path.realpath(target)
                if not target_real.startswith(dest_real + os.sep):
                    raise ValueError(
                        f"[train_step] record {record.get('id', record['image_path'])!r} "
                        f"resolves to {target_real!r}, outside class dir {dest_real!r}"
                    )
                expected_dests.add(target_real)
                shutil.copy(src, target)
                copied_dests.add(target_real)
    if copied_dests != expected_dests:
        raise RuntimeError(
            "[train_step] copied destinations do not match expected destinations: "
            f"missing={sorted(expected_dests - copied_dests)} "
            f"extra={sorted(copied_dests - expected_dests)}"
        )
    copied_counts = {label: 0 for label in by_class}
    for path in copied_dests:
        # Attribute each copied destination to its class dir for the summary
        # log; the invariant above has already asserted set equality.
        parent = os.path.basename(os.path.dirname(path))
        if parent in copied_counts:
            copied_counts[parent] += 1
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

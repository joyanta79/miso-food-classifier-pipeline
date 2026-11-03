"""In-container training runner for the Miso smoke test.

Executed BY a SageMaker Training job (TensorFlow 2.14 CPU DLC). The `train`
channel delivers the flat `wc_<label>_<conf>.jpg` images; this runner arranges
them into the class-labelled `train/<class>` + `validation/<class>` directory
layout that training.train.train_model expects (80/20 split, deterministic),
then trains EfficientNetV2B0 for a few epochs and writes the model to
/opt/ml/model.

Smoke scope: 25 images/class cannot reach the spec's 0.97 accuracy gate. The
goal is to prove the real training path executes on SageMaker and produces a
saved model + metrics, not to hit production accuracy.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

MOUNT = Path("/opt/ml/input/data/code")
RAW = Path("/opt/ml/input/data/train")  # flat images delivered here
MODEL = Path(os.getenv("SM_MODEL_DIR", "/opt/ml/model"))
OUTPUT = Path(os.getenv("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
WORK = Path("/tmp/miso_train")
EPOCHS = int(os.getenv("MISO_EPOCHS", "3"))


def _verify_and_extract_code(mount: Path) -> Path:
    """Verify the uploaded code tarball and extract it to a trusted path.

    The submitter (``scripts/sm_submit_train.py``) uploads the packaged source
    tree to a per-digest S3 prefix and passes the expected SHA-256 via the
    ``MISO_CODE_SHA256`` environment variable. We refuse to touch ``sys.path``
    with the raw mount; see ``scripts/sm_clean_runner.py`` for the full
    rationale (CWE-829 / CWE-1214).
    """
    expected = os.environ.get("MISO_CODE_SHA256")
    if not expected:
        raise RuntimeError(
            "MISO_CODE_SHA256 is not set; refusing to execute unverified code from "
            f"{mount} (CWE-829)."
        )
    tarball = mount / "payload.tar.gz"
    if not tarball.is_file():
        raise RuntimeError(
            f"Expected code tarball at {tarball} not found; refusing to execute "
            "unverified code (CWE-829)."
        )
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(
            f"Code tarball sha256 mismatch: got {digest}, expected {expected}. "
            "Refusing to execute (CWE-1214)."
        )
    extracted = Path("/tmp/miso_code_verified")
    if extracted.exists():
        shutil.rmtree(extracted)
    extracted.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball) as tar:
        for member in tar.getmembers():
            mname = member.name
            if mname.startswith("/") or ".." in Path(mname).parts:
                raise RuntimeError(f"Unsafe tar member rejected: {mname}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Link members not allowed in code tarball: {mname}")
        tar.extractall(extracted)
    print(f"[verify] code sha256 OK ({digest}); extracted to {extracted}", flush=True)
    return extracted


CODE = _verify_and_extract_code(MOUNT)

# The shared bootstrap enforces --require-hashes / --no-deps and a pinned
# private index, and refuses to resolve bare names from the public PyPI.
sys.path.insert(0, str(CODE / "pipeline" / "steps"))
import _bootstrap  # noqa: E402


def _pip(packages: dict[str, str]) -> None:
    """Hash-pinned, index-restricted install via ``_bootstrap.ensure``."""
    _bootstrap.ensure(packages)


def _label_from_name(name: str) -> str | None:
    # wc_<label>_<conf>.jpg
    parts = Path(name).stem.split("_")
    if len(parts) >= 3:
        return parts[1]
    return None


def arrange(raw: Path, work: Path) -> dict[str, int]:
    """Split flat images into train/<class> + validation/<class> (80/20)."""
    by_class: dict[str, list[Path]] = {}
    for p in sorted(raw.glob("*.jpg")):
        label = _label_from_name(p.name)
        if label:
            by_class.setdefault(label, []).append(p)
    counts: dict[str, int] = {}
    for label, files in by_class.items():
        n_val = max(1, len(files) // 5)
        val, train = files[:n_val], files[n_val:]
        for split, group in (("train", train), ("validation", val)):
            dest = work / split / label
            dest.mkdir(parents=True, exist_ok=True)
            for f in group:
                shutil.copy(f, dest / f.name)
        counts[label] = len(files)
    return counts


def main() -> int:
    _pip({"yaml": "pyyaml", "PIL": "pillow"})
    sys.path.insert(0, str(CODE))
    from training.train import train_model

    counts = arrange(RAW, WORK)
    print(f"[prep] class counts: {counts}", flush=True)

    # Build config inline (smoke scope) so the run does not depend on a
    # config.yaml being present in the code channel. Force the real
    # EfficientNetV2 (ImageNet) path, overriding local_mode.
    config = {
        "local_mode": {"enabled": False, "seed": 2026},
        "embedding": {"backbone_source": "imagenet"},
        "training": {
            "image_size": 224,
            "batch_size": 8,
            "epochs": EPOCHS,
            "learning_rate": 0.001,
            "checkpoint_every_epochs": 5,
        },
    }

    result = train_model(
        config=config,
        training_dir=WORK / "train",
        validation_dir=WORK / "validation",
        model_dir=MODEL,
        checkpoint_dir=Path(os.getenv("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")),
        output_dir=OUTPUT,
        brand_id="white-castle",
        epochs=EPOCHS,
    )
    print("[train] result:", json.dumps(result), flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "training_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print("[done] training complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

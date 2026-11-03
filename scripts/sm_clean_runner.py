"""In-container runner for the Miso data-cleaning path (Steps 0->3).

Executed BY a SageMaker Processing job. Reads real images from
/opt/ml/processing/input/images, builds the Step 0 inventory, runs the real
EfficientNetV2 embedding path (Step 1), K-fold OOS prediction (Step 2), and
CleanLab filtering (Step 3), and writes results to /opt/ml/processing/output.

Injects a few deliberate mislabels so Step 3 detection can be demonstrated on
the smoke dataset, exactly as the local end-to-end driver does.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path

IN = Path("/opt/ml/processing/input")
MOUNT = Path("/opt/ml/processing/input/code")
OUT = Path("/opt/ml/processing/output")


def _verify_and_extract_code(mount: Path) -> Path:
    """Verify the uploaded code tarball and extract it to a trusted path.

    The submitter (``scripts/sm_submit_clean.py``) uploads the packaged source
    tree to a per-digest S3 prefix and passes the expected SHA-256 via the
    ``MISO_CODE_SHA256`` environment variable. We refuse to touch ``sys.path``
    with the raw mount: an attacker who could add a sibling object under the
    code prefix would otherwise get arbitrary code execution. Instead we hash
    the tarball, compare against the expected digest (fail-closed if either
    the env var or the tarball is missing), and extract to a scratch
    directory that only contains the content whose hash we just verified.
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
        import shutil as _shutil

        _shutil.rmtree(extracted)
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
    """Hash-pinned, index-restricted install via ``_bootstrap.ensure``.

    Accepts an ``import_name -> pip requirement`` mapping so callers declare
    exactly which module they need; the actual resolve is driven by the
    hash-locked requirements.lock the operator mounts into the job.
    """
    _bootstrap.ensure(packages)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # Runtime deps not guaranteed in the base DLC image. The install only
    # happens when a hash-locked requirements file + pinned index are
    # configured; see pipeline/steps/_bootstrap.py.
    _pip(
        {
            "sklearn": "scikit-learn",
            "xgboost": "xgboost",
            "cleanlab.filter:find_label_issues": "cleanlab==2.6.4",
            "PIL": "pillow",
        }
    )

    sys.path.insert(0, str(CODE))
    from processing.step0_data_selection import select_data
    from processing.step1_extract_embeddings import extract_embeddings
    from processing.step2_kfold_predict import kfold_predict
    from processing.step3_cleanlab_filter import filter_label_issues

    image_dir = IN / "images"
    now = datetime.now(UTC).isoformat()
    inventory = [
        {"key": p.name, "id": p.name, "last_modified": now} for p in sorted(image_dir.glob("*.jpg"))
    ]
    print(f"[step0] inventory: {len(inventory)} images", flush=True)

    manifest = select_data(
        inventory,
        naming_pattern="wc_*",
        recency_months=6,
        min_images_per_class=1,
        target_dataset_min=1,
        target_dataset_max=50000,
    )
    counts: dict[str, int] = {}
    for r in manifest:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print(f"[step0] curated {len(manifest)} images; classes={counts}", flush=True)

    # Inject deliberate mislabels for the CleanLab proof.
    labels = sorted(counts)
    corrupted: list[str] = []
    if len(labels) >= 2:
        flip = {labels[0]: labels[1], labels[1]: labels[0]}
        for r in manifest:
            if r["label"] in flip and len(corrupted) < 3:
                r["true_label"] = r["label"]
                r["label"] = flip[r["label"]]
                corrupted.append(r["id"])
    print(f"[step0] injected mislabels: {corrupted}", flush=True)

    embeddings, mapping = extract_embeddings(
        manifest, image_root=image_dir, use_real_backbone=True, backbone_weights="imagenet"
    )
    backend = mapping[0]["embedding_backend"] if mapping else "n/a"
    print(f"[step1] embeddings shape={embeddings.shape} backend={backend}", flush=True)

    probs, classes = kfold_predict(
        embeddings, [r["label"] for r in mapping], n_folds=5, classifier="xgboost"
    )
    print(f"[step2] OOS probs shape={probs.shape} classes={classes}", flush=True)

    flagged, cleaned, report = filter_label_issues(
        probs, mapping, classes, flag_threshold_fraction=0.05
    )
    caught = [c for c in corrupted if c in {f["id"] for f in flagged}]
    print(
        f"[step3] flagged={report['flagged_images']}/{report['total_images']} "
        f"cleaned={len(cleaned)} caught={len(caught)}/{len(corrupted)}",
        flush=True,
    )

    (OUT / "curated_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "cleaned_dataset.json").write_text(json.dumps(cleaned, indent=2) + "\n")
    (OUT / "flagged_images.json").write_text(json.dumps(flagged, indent=2) + "\n")
    (OUT / "cleanlab_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "smoke_result.json").write_text(
        json.dumps(
            {
                "images": len(manifest),
                "class_counts": counts,
                "embedding_backend": backend,
                "embedding_shape": list(embeddings.shape),
                "classes": classes,
                "flagged_fraction": report["flagged_fraction"],
                "injected_mislabels": corrupted,
                "caught_mislabels": caught,
                "cleanlab_recall": (len(caught) / len(corrupted)) if corrupted else 1.0,
            },
            indent=2,
        )
        + "\n"
    )
    print("[done] data-cleaning path complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

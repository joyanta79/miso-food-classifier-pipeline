"""Submit the Miso data-cleaning Processing job (Steps 0->3) to SageMaker.

Real cloud run on real images using the TensorFlow 2.14 CPU DLC (real
EfficientNetV2 backbone). Uploads the runner + code to a version-pinned,
digest-named S3 prefix in the project pipeline-artifacts bucket, then
launches the job referencing that exact digest so the container executes
only the tree whose SHA-256 the submitter signed off on.

Security notes (CWE-829, CWE-1214):

* ``MISO_BUCKET`` MUST be set explicitly to a project-owned, write-restricted
  pipeline-artifacts bucket. The historical default of
  ``sagemaker-<region>-<account>`` is REJECTED because that bucket is shared
  by every SageMaker identity in the account (the AWS-managed SageMaker
  policies grant access to ``sagemaker-*``), so it is not a trustworthy code
  source.
* The code tree is packaged, hashed, and uploaded here. The job references a
  unique per-digest S3 prefix and the in-container runner verifies the
  SHA-256 of the tarball against the ``MISO_CODE_SHA256`` environment
  variable before inserting any mounted path into ``sys.path``.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import tarfile
import time
from pathlib import Path

import boto3

REGION = os.environ.get("AWS_REGION", "us-west-2")
ACCOUNT = os.environ.get("MISO_ACCOUNT_ID", "<replace-with-account-id>")

_SAGEMAKER_DEFAULT_RE = re.compile(r"^sagemaker-[a-z0-9-]+-\d{12}$")
_BUCKET_ALLOWLIST_RE = re.compile(
    os.environ.get("MISO_BUCKET_ALLOWLIST_REGEX", r"^miso-[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
)


def _resolve_bucket() -> str:
    bucket = os.environ.get("MISO_BUCKET")
    if not bucket:
        raise RuntimeError(
            "MISO_BUCKET is not set. Set it to a project-owned, write-restricted "
            "pipeline-artifacts bucket (not the account-shared sagemaker-* default)."
        )
    if _SAGEMAKER_DEFAULT_RE.match(bucket):
        raise RuntimeError(
            f"MISO_BUCKET={bucket!r} looks like the account-shared SageMaker default "
            "bucket. Refusing to use it as a code source (CWE-829)."
        )
    if not _BUCKET_ALLOWLIST_RE.match(bucket):
        raise RuntimeError(
            f"MISO_BUCKET={bucket!r} is not on the project bucket allowlist "
            f"(pattern {_BUCKET_ALLOWLIST_RE.pattern!r}). Set MISO_BUCKET_ALLOWLIST_REGEX "
            "to extend the allowlist deliberately."
        )
    return bucket


def _validate_account(account: str) -> str:
    if not re.fullmatch(r"\d{12}", account):
        raise RuntimeError(
            f"MISO_ACCOUNT_ID={account!r} is not a 12-digit AWS account id. "
            "Set MISO_ACCOUNT_ID explicitly before submitting."
        )
    return account


BUCKET = _resolve_bucket()
ACCOUNT = _validate_account(ACCOUNT)
ROLE = f"arn:aws:iam::{ACCOUNT}:role/miso-food-clf-sagemaker-exec"
IMAGE = "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.14.1-cpu-py310"
PREFIX = "miso-smoke"

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGED_DIRS = ("scripts", "pipeline", "processing", "training")
_SKIP_NAMES = {"__pycache__"}
_SKIP_SUFFIXES = {".pyc", ".pyo"}


def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    name = Path(info.name).name
    if name in _SKIP_NAMES:
        return None
    if any(info.name.endswith(sfx) for sfx in _SKIP_SUFFIXES):
        return None
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _build_payload(repo_root: Path, dirs: tuple[str, ...]) -> bytes:
    missing = [d for d in dirs if not (repo_root / d).is_dir()]
    if missing:
        raise RuntimeError(f"Packaged source dirs missing from repo: {missing}")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=6) as tar:
        for d in dirs:
            tar.add(repo_root / d, arcname=d, filter=_tar_filter)
    return buf.getvalue()


payload = _build_payload(REPO_ROOT, PACKAGED_DIRS)
code_sha256 = hashlib.sha256(payload).hexdigest()
digest_prefix = f"{PREFIX}/code/{code_sha256}"
code_tarball_key = f"{digest_prefix}/payload.tar.gz"
code_runner_key = f"{digest_prefix}/scripts/sm_clean_runner.py"
code_prefix_uri = f"s3://{BUCKET}/{digest_prefix}/"

runner_src = (REPO_ROOT / "scripts" / "sm_clean_runner.py").read_bytes()

s3 = boto3.client("s3", region_name=REGION)
s3.put_object(
    Bucket=BUCKET,
    Key=code_tarball_key,
    Body=payload,
    ChecksumAlgorithm="SHA256",
)
runner_resp = s3.put_object(
    Bucket=BUCKET,
    Key=code_runner_key,
    Body=runner_src,
    ChecksumAlgorithm="SHA256",
)
code_version_id = runner_resp.get("VersionId", "")
print(
    f"UPLOADED s3://{BUCKET}/{code_tarball_key} sha256={code_sha256} "
    f"version_id={code_version_id!r}"
)

sm = boto3.client("sagemaker", region_name=REGION)

job_name = f"miso-clean-{int(time.time())}"

resp = sm.create_processing_job(
    ProcessingJobName=job_name,
    RoleArn=ROLE,
    AppSpecification={
        "ImageUri": IMAGE,
        "ContainerEntrypoint": [
            "python",
            "/opt/ml/processing/input/code/scripts/sm_clean_runner.py",
        ],
    },
    Environment={
        "MISO_CODE_SHA256": code_sha256,
        "MISO_CODE_VERSION_ID": code_version_id,
        "MISO_CODE_BUCKET": BUCKET,
        "MISO_CODE_KEY": code_tarball_key,
    },
    ProcessingResources={
        "ClusterConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.m5.xlarge",
            "VolumeSizeInGB": 20,
        }
    },
    StoppingCondition={"MaxRuntimeInSeconds": 3600},
    ProcessingInputs=[
        {
            "InputName": "images",
            "S3Input": {
                "S3Uri": f"s3://{BUCKET}/{PREFIX}/input/images/",
                "LocalPath": "/opt/ml/processing/input/images",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
            },
        },
        {
            "InputName": "code",
            "S3Input": {
                "S3Uri": code_prefix_uri,
                "LocalPath": "/opt/ml/processing/input/code",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
            },
        },
    ],
    ProcessingOutputConfig={
        "Outputs": [
            {
                "OutputName": "cleaned",
                "S3Output": {
                    "S3Uri": f"s3://{BUCKET}/{PREFIX}/output/clean/",
                    "LocalPath": "/opt/ml/processing/output",
                    "S3UploadMode": "EndOfJob",
                },
            }
        ]
    },
)
print("SUBMITTED", job_name)
print("ARN", resp["ProcessingJobArn"])
print("CODE_SHA256", code_sha256)

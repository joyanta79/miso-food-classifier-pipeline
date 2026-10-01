"""Submit the Miso data-cleaning Processing job (Steps 0->3) to SageMaker.

Real cloud run on real images using the TensorFlow 2.14 CPU DLC (real
EfficientNetV2 backbone). Uploads the runner + code, launches the job, and
prints the job name so a poller can follow it.
"""

from __future__ import annotations

import os
import time

import boto3

REGION = os.environ.get("AWS_REGION", "us-west-2")
ACCOUNT = os.environ.get("MISO_ACCOUNT_ID", "<replace-with-account-id>")
BUCKET = os.environ.get("MISO_BUCKET", f"sagemaker-{REGION}-{ACCOUNT}")
ROLE = f"arn:aws:iam::{ACCOUNT}:role/miso-food-clf-sagemaker-exec"
IMAGE = "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.14.1-cpu-py310"
PREFIX = "miso-smoke"

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
                "S3Uri": f"s3://{BUCKET}/{PREFIX}/code/",
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

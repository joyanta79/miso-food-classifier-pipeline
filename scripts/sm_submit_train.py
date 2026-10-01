"""Submit the Miso EfficientNetV2 training job to SageMaker (smoke scope).

Uses the TensorFlow 2.14 CPU DLC with a custom entrypoint. Delivers the flat
images (train channel) and the code (code channel), runs a few epochs, and
saves the model to the output S3 path.
"""
from __future__ import annotations

import time

import boto3

REGION = "us-west-2"
ACCOUNT = "363491582148"
BUCKET = "sagemaker-us-west-2-363491582148"
ROLE = f"arn:aws:iam::{ACCOUNT}:role/miso-food-clf-sagemaker-exec"
IMAGE = "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.14.1-cpu-py310"
PREFIX = "miso-smoke"

sm = boto3.client("sagemaker", region_name=REGION)
job = f"miso-train-{int(time.time())}"

resp = sm.create_training_job(
    TrainingJobName=job,
    RoleArn=ROLE,
    AlgorithmSpecification={
        "TrainingImage": IMAGE,
        "TrainingInputMode": "File",
        "ContainerEntrypoint": ["python", "/opt/ml/input/data/code/scripts/sm_train_runner.py"],
    },
    ResourceConfig={"InstanceType": "ml.m5.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 20},
    StoppingCondition={"MaxRuntimeInSeconds": 3600},
    Environment={"MISO_EPOCHS": "3"},
    InputDataConfig=[
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{BUCKET}/{PREFIX}/input/images/",
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        },
        {
            "ChannelName": "code",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{BUCKET}/{PREFIX}/code/",
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        },
    ],
    OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/{PREFIX}/output/train/"},
)
print("SUBMITTED", job)
print("ARN", resp["TrainingJobArn"])

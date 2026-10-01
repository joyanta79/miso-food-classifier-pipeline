"""Configuration-driven SageMaker pipeline definition for Miso Robotics.

The module has two modes:
* ``build_dry_run_definition`` returns a complete, serialisable pipeline DAG without
  importing SageMaker or contacting AWS. This makes it suitable for CI and local
  design validation.
* ``build_sagemaker_pipeline`` creates the equivalent SageMaker SDK pipeline when
  the optional SDK is installed. Processing and training code are intentionally
  referenced as S3 contract paths; those implementations belong to their owners.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
PIPELINE_VERSION = "2020-12-01"


@dataclass(frozen=True)
class Brand:
    """A brand-specific model-training contract."""

    id: str
    naming_pattern: str
    model_package_group: str


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and minimally validate the shared pipeline configuration."""
    path = Path(config_path)
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    required_sections = {
        "aws",
        "s3",
        "brands",
        "data_selection",
        "embedding",
        "kfold",
        "cleanlab",
        "training",
        "evaluation",
        "drift",
        "notifications",
        "local_mode",
    }
    missing = required_sections.difference(config or {})
    if missing:
        raise ValueError(f"config.yaml is missing required sections: {sorted(missing)}")
    if not config["brands"]:
        raise ValueError("config.yaml must define at least one brand")
    return config


def get_brand(config: Mapping[str, Any], brand_id: str) -> Brand:
    """Return a declared brand, refusing arbitrary package-group combinations."""
    for item in config["brands"]:
        if item["id"] == brand_id:
            return Brand(
                id=item["id"],
                naming_pattern=item["naming_pattern"],
                model_package_group=item["model_package_group"],
            )
    configured = ", ".join(item["id"] for item in config["brands"])
    raise ValueError(f"Unknown brand {brand_id!r}; expected one of: {configured}")


def _parameter(name: str, value: Any, description: str) -> dict[str, Any]:
    if isinstance(value, bool):
        parameter_type = "Boolean"
    elif isinstance(value, int):
        parameter_type = "Integer"
    elif isinstance(value, float):
        parameter_type = "Float"
    else:
        parameter_type = "String"
    return {
        "Name": name,
        "Type": parameter_type,
        "DefaultValue": value,
        "Description": description,
    }


def build_parameters(config: Mapping[str, Any], brand: Brand) -> list[dict[str, Any]]:
    """Expose every configuration value used by the pipeline as an execution parameter."""
    aws = config["aws"]
    s3 = config["s3"]
    selection = config["data_selection"]
    embedding = config["embedding"]
    kfold = config["kfold"]
    cleanlab = config["cleanlab"]
    training = config["training"]
    evaluation = config["evaluation"]
    drift = config["drift"]
    notifications = config["notifications"]
    local_mode = config["local_mode"]

    values = [
        ("Region", aws["region"], "AWS Region hosting this pipeline."),
        ("AccountId", aws["account_id"], "Target AWS account identifier."),
        ("SageMakerRoleArn", aws["sagemaker_role_arn"], "Execution role for SageMaker jobs."),
        ("TrainingBucket", s3["training_bucket"], "S3 bucket for prepared training datasets."),
        (
            "ModelArtifactsBucket",
            s3["model_artifacts_bucket"],
            "S3 bucket for trained model artifacts and checkpoints.",
        ),
        (
            "PipelineArtifactsBucket",
            s3["pipeline_artifacts_bucket"],
            "S3 bucket for pipeline outputs and container code contracts.",
        ),
        ("RawArchiveBucket", s3["raw_archive_bucket"], "S3 bucket containing raw image archives."),
        ("BrandId", brand.id, "Fixed pipeline brand identifier."),
        ("BrandNamingPattern", brand.naming_pattern, "Allowed raw-data filename prefix pattern."),
        (
            "ModelPackageGroup",
            brand.model_package_group,
            "Brand-specific SageMaker Model Registry package group.",
        ),
        ("RecencyMonths", selection["recency_months"], "Maximum image age to select."),
        (
            "MinImagesPerClass",
            selection["min_images_per_class"],
            "Minimum images required in each retained class.",
        ),
        ("TargetDatasetMin", selection["target_dataset_min"], "Minimum accepted dataset size."),
        ("TargetDatasetMax", selection["target_dataset_max"], "Maximum accepted dataset size."),
        ("BackboneSource", embedding["backbone_source"], "Embedding backbone provenance."),
        (
            "BackboneModelPath",
            embedding["backbone_model_path"],
            "Prior-production backbone path used for embeddings.",
        ),
        ("EmbeddingDim", embedding["embedding_dim"], "Embedding vector dimensionality."),
        (
            "CosineDedupThreshold",
            embedding["cosine_dedup_threshold"],
            "Cosine similarity above which an image is deduplicated.",
        ),
        ("KFoldCount", kfold["n_folds"], "Number of cross-validation folds."),
        ("Classifier", kfold["classifier"], "Classifier implementation used for training."),
        (
            "CleanlabFlagThresholdFraction",
            cleanlab["flag_threshold_fraction"],
            "Maximum tolerated fraction of Cleanlab label-quality flags.",
        ),
        ("TrainingInstanceType", training["instance_type"], "SageMaker training instance type."),
        (
            "ProcessingInstanceType",
            training["processing_instance_type"],
            "SageMaker processing instance type.",
        ),
        ("UseSpot", training["use_spot"], "Whether managed spot training is enabled."),
        ("Epochs", training["epochs"], "Maximum training epochs."),
        ("BatchSize", training["batch_size"], "Training batch size."),
        ("LearningRate", training["learning_rate"], "Training learning rate."),
        (
            "CheckpointEveryEpochs",
            training["checkpoint_every_epochs"],
            "Epoch interval for persisted training checkpoints.",
        ),
        ("ImageSize", training["image_size"], "Square input image size in pixels."),
        ("AccuracyThreshold", evaluation["accuracy_threshold"], "Minimum accepted accuracy."),
        (
            "FpRateEmergencyThreshold",
            evaluation["fp_rate_emergency_threshold"],
            "Maximum accepted emergency false-positive rate.",
        ),
        ("LatencyBudgetMs", evaluation["latency_budget_ms"], "Maximum accepted inference latency."),
        (
            "ConfidenceBaseline",
            drift["confidence_baseline"],
            "Reference confidence level for drift detection.",
        ),
        (
            "Level1ConfidenceFloor",
            drift["level1_confidence_floor"],
            "Level 1 intervention confidence floor.",
        ),
        (
            "Level2ConfidenceThreshold",
            drift["level2_confidence_threshold"],
            "Level 2 escalation confidence threshold.",
        ),
        (
            "CorrectionRateMultiplier",
            drift["correction_rate_multiplier"],
            "Correction-rate multiplier that triggers drift action.",
        ),
        ("ReviewTopicArn", notifications["review_topic_arn"], "Human review notification topic."),
        ("MlLeadTopicArn", notifications["ml_lead_topic_arn"], "ML lead notification topic."),
        (
            "VpEngineeringTopicArn",
            notifications["vp_engineering_topic_arn"],
            "VP Engineering escalation notification topic.",
        ),
        ("LocalModeEnabled", local_mode["enabled"], "Whether local dry-run support is enabled."),
        ("MockAws", local_mode["mock_aws"], "Whether local mode uses mocked AWS services."),
        (
            "UseImagenetWeights",
            local_mode["use_imagenet_weights"],
            "Whether local mode loads ImageNet weights.",
        ),
        ("Seed", local_mode["seed"], "Deterministic local-mode random seed."),
    ]
    return [_parameter(*value) for value in values]


def _ref(parameter_name: str) -> dict[str, str]:
    return {"Get": f"Parameters.{parameter_name}"}


def _step_ref(step_name: str, property_file: str, path: str) -> dict[str, Any]:
    return {
        "Std:JsonGet": {
            "PropertyFile": {"Get": f"Steps.{step_name}.PropertyFiles.{property_file}"},
            "Path": path,
        }
    }


def _s3_uri(bucket_parameter: str, *parts: str) -> dict[str, Any]:
    return {
        "Std:Join": {
            "On": "/",
            "Values": ["s3:/", _ref(bucket_parameter), *parts],
        }
    }


def _environment(*pairs: tuple[str, str]) -> dict[str, Any]:
    return {key: _ref(parameter) for key, parameter in pairs}


def _processing_step(
    name: str,
    code_name: str,
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    environment: dict[str, Any],
    property_files: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "Name": name,
        "Type": "Processing",
        "Arguments": {
            "RoleArn": _ref("SageMakerRoleArn"),
            "AppSpecification": {
                "ImageUri": (
                    "683313688378.dkr.ecr.us-west-2.amazonaws.com/"
                    "sagemaker-scikit-learn:1.2-1-cpu-py3"
                ),
                "ContainerEntrypoint": ["python3", f"/opt/ml/processing/input/code/{code_name}"],
            },
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": _ref("ProcessingInstanceType"),
                    "VolumeSizeInGB": 30,
                }
            },
            "ProcessingInputs": inputs
            + [
                {
                    "InputName": "pipeline-code",
                    "S3Input": {
                        "S3Uri": _s3_uri("PipelineArtifactsBucket", "code"),
                        "LocalPath": "/opt/ml/processing/input/code",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                    },
                }
            ],
            "ProcessingOutputConfig": {"Outputs": outputs},
            "Environment": environment,
        },
    }
    if property_files:
        step["PropertyFiles"] = property_files
    return step


def build_dry_run_definition(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    brand_id: str = "white-castle",
) -> dict[str, Any]:
    """Build a complete SageMaker-compatible local DAG without AWS or SDK imports."""
    config = load_config(config_path)
    brand = get_brand(config, brand_id)

    select_data = _processing_step(
        "SelectRecentBrandData",
        "data_selection.py",
        [
            {
                "InputName": "raw-archive",
                "S3Input": {
                    "S3Uri": _s3_uri("RawArchiveBucket", ""),
                    "LocalPath": "/opt/ml/processing/input/raw",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            }
        ],
        [
            {
                "OutputName": "selected-dataset",
                "S3Output": {
                    "S3Uri": _s3_uri("TrainingBucket", brand.id, "selected"),
                    "LocalPath": "/opt/ml/processing/output/dataset",
                    "S3UploadMode": "EndOfJob",
                },
            },
            {
                "OutputName": "selection-metrics",
                "S3Output": {
                    "S3Uri": _s3_uri("PipelineArtifactsBucket", brand.id, "selection"),
                    "LocalPath": "/opt/ml/processing/output/metrics",
                    "S3UploadMode": "EndOfJob",
                },
            },
        ],
        _environment(
            ("BRAND_ID", "BrandId"),
            ("NAMING_PATTERN", "BrandNamingPattern"),
            ("RECENCY_MONTHS", "RecencyMonths"),
            ("MIN_IMAGES_PER_CLASS", "MinImagesPerClass"),
            ("TARGET_DATASET_MIN", "TargetDatasetMin"),
            ("TARGET_DATASET_MAX", "TargetDatasetMax"),
            ("SEED", "Seed"),
        ),
        [
            {
                "PropertyFileName": "SelectionMetrics",
                "OutputName": "selection-metrics",
                "Path": "metrics.json",
            }
        ],
    )

    embed_and_deduplicate = _processing_step(
        "EmbedAndDeduplicate",
        "embedding_dedup.py",
        [
            {
                "InputName": "selected-dataset",
                "S3Input": {
                    "S3Uri": _s3_uri("TrainingBucket", brand.id, "selected"),
                    "LocalPath": "/opt/ml/processing/input/dataset",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            }
        ],
        [
            {
                "OutputName": "deduplicated-dataset",
                "S3Output": {
                    "S3Uri": _s3_uri("TrainingBucket", brand.id, "deduplicated"),
                    "LocalPath": "/opt/ml/processing/output/dataset",
                    "S3UploadMode": "EndOfJob",
                },
            }
        ],
        _environment(
            ("BACKBONE_SOURCE", "BackboneSource"),
            ("BACKBONE_MODEL_PATH", "BackboneModelPath"),
            ("EMBEDDING_DIM", "EmbeddingDim"),
            ("COSINE_DEDUP_THRESHOLD", "CosineDedupThreshold"),
            ("USE_IMAGENET_WEIGHTS", "UseImagenetWeights"),
            ("SEED", "Seed"),
        ),
    )

    label_quality = _processing_step(
        "KFoldCleanlabQuality",
        "kfold_cleanlab.py",
        [
            {
                "InputName": "deduplicated-dataset",
                "S3Input": {
                    "S3Uri": _s3_uri("TrainingBucket", brand.id, "deduplicated"),
                    "LocalPath": "/opt/ml/processing/input/dataset",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            }
        ],
        [
            {
                "OutputName": "quality-dataset",
                "S3Output": {
                    "S3Uri": _s3_uri("TrainingBucket", brand.id, "quality"),
                    "LocalPath": "/opt/ml/processing/output/dataset",
                    "S3UploadMode": "EndOfJob",
                },
            },
            {
                "OutputName": "quality-metrics",
                "S3Output": {
                    "S3Uri": _s3_uri("PipelineArtifactsBucket", brand.id, "quality"),
                    "LocalPath": "/opt/ml/processing/output/metrics",
                    "S3UploadMode": "EndOfJob",
                },
            },
        ],
        _environment(
            ("KFOLD_COUNT", "KFoldCount"),
            ("CLASSIFIER", "Classifier"),
            ("CLEANLAB_FLAG_THRESHOLD_FRACTION", "CleanlabFlagThresholdFraction"),
            ("SEED", "Seed"),
        ),
        [
            {
                "PropertyFileName": "QualityMetrics",
                "OutputName": "quality-metrics",
                "Path": "metrics.json",
            }
        ],
    )

    detect_drift = _processing_step(
        "DetectDrift",
        "drift_detection.py",
        [
            {
                "InputName": "quality-dataset",
                "S3Input": {
                    "S3Uri": _s3_uri("TrainingBucket", brand.id, "quality"),
                    "LocalPath": "/opt/ml/processing/input/dataset",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            }
        ],
        [
            {
                "OutputName": "drift-metrics",
                "S3Output": {
                    "S3Uri": _s3_uri("PipelineArtifactsBucket", brand.id, "drift"),
                    "LocalPath": "/opt/ml/processing/output/metrics",
                    "S3UploadMode": "EndOfJob",
                },
            }
        ],
        _environment(
            ("CONFIDENCE_BASELINE", "ConfidenceBaseline"),
            ("LEVEL1_CONFIDENCE_FLOOR", "Level1ConfidenceFloor"),
            ("LEVEL2_CONFIDENCE_THRESHOLD", "Level2ConfidenceThreshold"),
            ("CORRECTION_RATE_MULTIPLIER", "CorrectionRateMultiplier"),
        ),
        [
            {
                "PropertyFileName": "DriftMetrics",
                "OutputName": "drift-metrics",
                "Path": "metrics.json",
            }
        ],
    )

    train_model = {
        "Name": "TrainBrandClassifier",
        "Type": "Training",
        "Arguments": {
            "RoleArn": _ref("SageMakerRoleArn"),
            "AlgorithmSpecification": {
                "TrainingImage": "433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:1.7-1",
                "TrainingInputMode": "File",
                "ContainerEntrypoint": ["python3", "/opt/ml/input/data/code/train.py"],
            },
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": _s3_uri("TrainingBucket", brand.id, "quality"),
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                },
                {
                    "ChannelName": "code",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": _s3_uri("PipelineArtifactsBucket", "code"),
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                },
            ],
            "OutputDataConfig": {"S3OutputPath": _s3_uri("ModelArtifactsBucket", brand.id)},
            "ResourceConfig": {
                "InstanceType": _ref("TrainingInstanceType"),
                "InstanceCount": 1,
                "VolumeSizeInGB": 50,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": 86_400, "MaxWaitTimeInSeconds": 172_800},
            "EnableManagedSpotTraining": _ref("UseSpot"),
            "CheckpointConfig": {"S3Uri": _s3_uri("ModelArtifactsBucket", brand.id, "checkpoints")},
            "HyperParameters": {
                "classifier": _ref("Classifier"),
                "epochs": _ref("Epochs"),
                "batch_size": _ref("BatchSize"),
                "learning_rate": _ref("LearningRate"),
                "checkpoint_every_epochs": _ref("CheckpointEveryEpochs"),
                "image_size": _ref("ImageSize"),
                "seed": _ref("Seed"),
            },
            "Tags": [{"Key": "brand", "Value": _ref("BrandId")}],
        },
    }

    evaluate_model = _processing_step(
        "EvaluateModel",
        "evaluation.py",
        [
            {
                "InputName": "model-artifacts",
                "S3Input": {
                    "S3Uri": _s3_uri("ModelArtifactsBucket", brand.id),
                    "LocalPath": "/opt/ml/processing/input/model",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            }
        ],
        [
            {
                "OutputName": "evaluation-metrics",
                "S3Output": {
                    "S3Uri": _s3_uri("PipelineArtifactsBucket", brand.id, "evaluation"),
                    "LocalPath": "/opt/ml/processing/output/metrics",
                    "S3UploadMode": "EndOfJob",
                },
            }
        ],
        _environment(
            ("ACCURACY_THRESHOLD", "AccuracyThreshold"),
            ("FP_RATE_EMERGENCY_THRESHOLD", "FpRateEmergencyThreshold"),
            ("LATENCY_BUDGET_MS", "LatencyBudgetMs"),
        ),
        [
            {
                "PropertyFileName": "EvaluationMetrics",
                "OutputName": "evaluation-metrics",
                "Path": "metrics.json",
            }
        ],
    )

    register_model = {
        "Name": "RegisterModelPendingManualApproval",
        "Type": "RegisterModel",
        "Arguments": {
            "ModelPackageGroupName": _ref("ModelPackageGroup"),
            "ModelApprovalStatus": "PendingManualApproval",
            "InferenceSpecification": {
                "Containers": [
                    {
                        "Image": "433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:1.7-1",
                        "ModelDataUrl": {
                            "Get": "Steps.TrainBrandClassifier.ModelArtifacts.S3ModelArtifacts"
                        },
                    }
                ],
                "SupportedContentTypes": ["application/json"],
                "SupportedResponseMIMETypes": ["application/json"],
                "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large"],
                "SupportedTransformInstanceTypes": ["ml.m5.large"],
            },
            "ModelMetrics": {
                "ModelQuality": {
                    "Statistics": {
                        "ContentType": "application/json",
                        "S3Uri": _s3_uri(
                            "PipelineArtifactsBucket", brand.id, "evaluation", "metrics.json"
                        ),
                    }
                }
            },
            "CustomerMetadataProperties": {
                "brand_id": _ref("BrandId"),
                "review_topic_arn": _ref("ReviewTopicArn"),
                "ml_lead_topic_arn": _ref("MlLeadTopicArn"),
                "vp_engineering_topic_arn": _ref("VpEngineeringTopicArn"),
            },
        },
    }

    fail_dataset = {
        "Name": "FailDatasetBounds",
        "Type": "Fail",
        "Arguments": {
            "ErrorMessage": "Selected dataset is outside configured size or class-coverage bounds."
        },
    }
    fail_cleanlab = {
        "Name": "FailCleanlabQuality",
        "Type": "Fail",
        "Arguments": {
            "ErrorMessage": "Cleanlab flagged label fraction exceeds the configured threshold."
        },
    }
    fail_drift = {
        "Name": "FailDriftGate",
        "Type": "Fail",
        "Arguments": {
            "ErrorMessage": (
                "Drift metrics exceed the configured confidence or correction-rate limits."
            )
        },
    }
    fail_evaluation = {
        "Name": "FailEvaluationGate",
        "Type": "Fail",
        "Arguments": {
            "ErrorMessage": "Accuracy, emergency false-positive rate, or latency gate failed."
        },
    }

    evaluation_gate = {
        "Name": "EvaluateModelQualityGate",
        "Type": "Condition",
        "Arguments": {
            "Conditions": [
                {
                    "Type": "ConditionGreaterThanOrEqualTo",
                    "LeftValue": _step_ref("EvaluateModel", "EvaluationMetrics", "accuracy"),
                    "RightValue": _ref("AccuracyThreshold"),
                },
                {
                    "Type": "ConditionLessThanOrEqualTo",
                    "LeftValue": _step_ref(
                        "EvaluateModel", "EvaluationMetrics", "emergency_false_positive_rate"
                    ),
                    "RightValue": _ref("FpRateEmergencyThreshold"),
                },
                {
                    "Type": "ConditionLessThanOrEqualTo",
                    "LeftValue": _step_ref("EvaluateModel", "EvaluationMetrics", "latency_ms"),
                    "RightValue": _ref("LatencyBudgetMs"),
                },
            ],
            "IfSteps": [register_model],
            "ElseSteps": [fail_evaluation],
        },
    }
    drift_gate = {
        "Name": "DriftGate",
        "Type": "Condition",
        "Arguments": {
            "Conditions": [
                {
                    "Type": "ConditionGreaterThanOrEqualTo",
                    "LeftValue": _step_ref("DetectDrift", "DriftMetrics", "confidence"),
                    "RightValue": _ref("Level1ConfidenceFloor"),
                },
                {
                    "Type": "ConditionLessThanOrEqualTo",
                    "LeftValue": _step_ref(
                        "DetectDrift", "DriftMetrics", "correction_rate_multiplier"
                    ),
                    "RightValue": _ref("CorrectionRateMultiplier"),
                },
            ],
            "IfSteps": [train_model, evaluate_model, evaluation_gate],
            "ElseSteps": [fail_drift],
        },
    }
    cleanlab_gate = {
        "Name": "CleanlabQualityGate",
        "Type": "Condition",
        "Arguments": {
            "Conditions": [
                {
                    "Type": "ConditionLessThanOrEqualTo",
                    "LeftValue": _step_ref(
                        "KFoldCleanlabQuality", "QualityMetrics", "flagged_fraction"
                    ),
                    "RightValue": _ref("CleanlabFlagThresholdFraction"),
                }
            ],
            "IfSteps": [detect_drift, drift_gate],
            "ElseSteps": [fail_cleanlab],
        },
    }
    dataset_gate = {
        "Name": "DatasetBoundsGate",
        "Type": "Condition",
        "Arguments": {
            "Conditions": [
                {
                    "Type": "ConditionGreaterThanOrEqualTo",
                    "LeftValue": _step_ref(
                        "SelectRecentBrandData", "SelectionMetrics", "dataset_size"
                    ),
                    "RightValue": _ref("TargetDatasetMin"),
                },
                {
                    "Type": "ConditionLessThanOrEqualTo",
                    "LeftValue": _step_ref(
                        "SelectRecentBrandData", "SelectionMetrics", "dataset_size"
                    ),
                    "RightValue": _ref("TargetDatasetMax"),
                },
                {
                    "Type": "ConditionGreaterThanOrEqualTo",
                    "LeftValue": _step_ref(
                        "SelectRecentBrandData", "SelectionMetrics", "minimum_class_count"
                    ),
                    "RightValue": _ref("MinImagesPerClass"),
                },
            ],
            "IfSteps": [embed_and_deduplicate, label_quality, cleanlab_gate],
            "ElseSteps": [fail_dataset],
        },
    }

    return _regionalize_dlc_uris(
        {
            "Version": PIPELINE_VERSION,
            "Parameters": build_parameters(config, brand),
            "Steps": [select_data, dataset_gate],
        },
        config["aws"].get("region", _DEFAULT_DLC_REGION),
    )


# AWS-managed Deep Learning Container registry accounts are region-specific and
# public. The literals throughout this module are written for us-west-2; this map
# rewrites both the account id and the region token so a deploy in another region
# resolves the correct public image instead of a wrong-region URI.
_SKLEARN_DLC_ACCOUNTS = {
    "us-west-2": "683313688378",
    "us-east-1": "683313688378",
    "us-east-2": "257758044811",
    "us-west-1": "746614075791",
    "eu-west-1": "141502667606",
    "eu-central-1": "141502667606",
    "ap-southeast-1": "245909111842",
    "ap-northeast-1": "354813040037",
}
_XGBOOST_DLC_ACCOUNTS = {
    "us-west-2": "433757028032",
    "us-east-1": "683313688378",
    "us-east-2": "257758044811",
    "us-west-1": "746614075791",
    "eu-west-1": "141502667606",
    "eu-central-1": "492215442770",
    "ap-southeast-1": "475088953585",
    "ap-northeast-1": "501404015308",
}
_DEFAULT_DLC_REGION = "us-west-2"


def _regionalize_dlc_uris(definition: dict[str, Any], region: str) -> dict[str, Any]:
    """Rewrite hardcoded us-west-2 DLC image URIs to the configured region."""
    if not region or region == _DEFAULT_DLC_REGION:
        return definition
    payload = json.dumps(definition)
    payload = payload.replace(
        f"683313688378.dkr.ecr.{_DEFAULT_DLC_REGION}.amazonaws.com/sagemaker-scikit-learn",
        f"{_SKLEARN_DLC_ACCOUNTS.get(region, '683313688378')}.dkr.ecr."
        f"{region}.amazonaws.com/sagemaker-scikit-learn",
    )
    payload = payload.replace(
        f"433757028032.dkr.ecr.{_DEFAULT_DLC_REGION}.amazonaws.com/xgboost",
        f"{_XGBOOST_DLC_ACCOUNTS.get(region, '433757028032')}.dkr.ecr."
        f"{region}.amazonaws.com/xgboost",
    )
    return json.loads(payload)


def _walk_steps(steps: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    flattened: list[Mapping[str, Any]] = []
    for step in steps:
        flattened.append(step)
        if step.get("Type") == "Condition":
            arguments = step.get("Arguments", {})
            flattened.extend(_walk_steps(arguments.get("IfSteps", [])))
            flattened.extend(_walk_steps(arguments.get("ElseSteps", [])))
    return flattened


def validate_dry_run_definition(definition: Mapping[str, Any]) -> None:
    """Validate the branch graph and non-negotiable model-release safeguards."""
    if definition.get("Version") != PIPELINE_VERSION:
        raise ValueError(f"Unsupported pipeline definition version: {definition.get('Version')!r}")

    names = [step["Name"] for step in _walk_steps(definition.get("Steps", []))]
    required = {
        "SelectRecentBrandData",
        "DatasetBoundsGate",
        "EmbedAndDeduplicate",
        "KFoldCleanlabQuality",
        "CleanlabQualityGate",
        "DetectDrift",
        "DriftGate",
        "TrainBrandClassifier",
        "EvaluateModel",
        "EvaluateModelQualityGate",
        "RegisterModelPendingManualApproval",
        "FailDatasetBounds",
        "FailCleanlabQuality",
        "FailDriftGate",
        "FailEvaluationGate",
    }
    missing = required.difference(names)
    if missing:
        raise ValueError(f"Local DAG is missing required steps: {sorted(missing)}")
    if len(names) != len(set(names)):
        raise ValueError("Local DAG contains duplicate step names")

    parameters = {item["Name"]: item for item in definition.get("Parameters", [])}
    for parameter in ("BrandId", "ModelPackageGroup", "AccuracyThreshold", "UseSpot", "Seed"):
        if parameter not in parameters:
            raise ValueError(f"Local DAG is missing required parameter: {parameter}")

    register_step = next(
        step
        for step in _walk_steps(definition["Steps"])
        if step["Name"] == "RegisterModelPendingManualApproval"
    )
    status = register_step["Arguments"].get("ModelApprovalStatus")
    if status != "PendingManualApproval":
        raise ValueError("Registered models must remain PendingManualApproval")


def render_dry_run_json(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    brand_id: str = "white-castle",
) -> str:
    """Return a canonical JSON representation of the complete local DAG."""
    definition = build_dry_run_definition(config_path, brand_id)
    validate_dry_run_definition(definition)
    return json.dumps(definition, indent=2, sort_keys=True) + "\n"


def build_sagemaker_pipeline(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    brand_id: str = "white-castle",
    pipeline_name: str | None = None,
) -> Any:
    """Build the SDK-native pipeline when SageMaker is optionally available.

    The SDK graph shares the same configuration contract as the local DAG. Its code
    S3 locations are intentionally contracts to be populated by the processing and
    training implementation tracks, rather than creating those files here.
    """
    try:
        from sagemaker.estimator import Estimator
        from sagemaker.model import Model
        from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
        from sagemaker.workflow.condition_step import ConditionStep
        from sagemaker.workflow.conditions import (
            ConditionGreaterThanOrEqualTo,
            ConditionLessThanOrEqualTo,
        )
        from sagemaker.workflow.fail_step import FailStep
        from sagemaker.workflow.functions import JsonGet
        from sagemaker.workflow.pipeline import Pipeline
        from sagemaker.workflow.pipeline_context import PipelineSession
        from sagemaker.workflow.properties import PropertyFile
        from sagemaker.workflow.step_collections import RegisterModel
        from sagemaker.workflow.steps import ProcessingStep, TrainingStep
    except ImportError as error:  # pragma: no cover - depends on optional SDK installation
        raise RuntimeError(
            "SageMaker SDK is optional for local validation. Install requirements.txt "
            "to build an SDK-native pipeline."
        ) from error

    config = load_config(config_path)
    brand = get_brand(config, brand_id)
    training = config["training"]
    evaluation = config["evaluation"]
    session = PipelineSession()
    role = config["aws"]["sagemaker_role_arn"]
    code_root = f"s3://{config['s3']['pipeline_artifacts_bucket']}/code"
    output_root = f"s3://{config['s3']['pipeline_artifacts_bucket']}/{brand.id}"

    processor = ScriptProcessor(
        image_uri="683313688378.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
        command=["python3"],
        role=role,
        instance_count=1,
        instance_type=training["processing_instance_type"],
        sagemaker_session=session,
    )

    def processing_step(
        name: str,
        script: str,
        output_name: str,
        environment: dict[str, str],
        property_file: PropertyFile | None = None,
    ) -> ProcessingStep:
        return ProcessingStep(
            name=name,
            step_args=processor.run(
                code=f"{code_root}/{script}",
                inputs=[
                    ProcessingInput(
                        source=f"s3://{config['s3']['training_bucket']}/{brand.id}",
                        destination="/opt/ml/processing/input/dataset",
                    )
                ],
                outputs=[
                    ProcessingOutput(
                        output_name=output_name,
                        source="/opt/ml/processing/output",
                        destination=f"{output_root}/{output_name}",
                    )
                ],
                environment=environment,
            ),
            property_files=[property_file] if property_file else None,
        )

    selection_report = PropertyFile(
        name="SelectionMetrics", output_name="selection", path="metrics.json"
    )
    selection_step = processing_step(
        "SelectRecentBrandData",
        "data_selection.py",
        "selection",
        {
            "BRAND_ID": brand.id,
            "NAMING_PATTERN": brand.naming_pattern,
            "RECENCY_MONTHS": str(config["data_selection"]["recency_months"]),
            "MIN_IMAGES_PER_CLASS": str(config["data_selection"]["min_images_per_class"]),
        },
        selection_report,
    )
    embeddings_step = processing_step(
        "EmbedAndDeduplicate",
        "embedding_dedup.py",
        "embeddings",
        {
            "BACKBONE_SOURCE": config["embedding"]["backbone_source"],
            "BACKBONE_MODEL_PATH": config["embedding"]["backbone_model_path"],
            "COSINE_DEDUP_THRESHOLD": str(config["embedding"]["cosine_dedup_threshold"]),
        },
    )
    quality_report = PropertyFile(name="QualityMetrics", output_name="quality", path="metrics.json")
    quality_step = processing_step(
        "KFoldCleanlabQuality",
        "kfold_cleanlab.py",
        "quality",
        {
            "KFOLD_COUNT": str(config["kfold"]["n_folds"]),
            "CLASSIFIER": config["kfold"]["classifier"],
            "CLEANLAB_FLAG_THRESHOLD_FRACTION": str(config["cleanlab"]["flag_threshold_fraction"]),
        },
        quality_report,
    )
    drift_report = PropertyFile(name="DriftMetrics", output_name="drift", path="metrics.json")
    drift_step = processing_step(
        "DetectDrift",
        "drift_detection.py",
        "drift",
        {key.upper(): str(value) for key, value in config["drift"].items()},
        drift_report,
    )

    estimator = Estimator(
        image_uri="433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:1.7-1",
        role=role,
        instance_count=1,
        instance_type=training["instance_type"],
        output_path=f"s3://{config['s3']['model_artifacts_bucket']}/{brand.id}",
        checkpoint_s3_uri=f"s3://{config['s3']['model_artifacts_bucket']}/{brand.id}/checkpoints",
        use_spot_instances=training["use_spot"],
        max_run=86_400,
        max_wait=172_800,
        sagemaker_session=session,
        hyperparameters={
            "classifier": config["kfold"]["classifier"],
            "epochs": training["epochs"],
            "batch_size": training["batch_size"],
            "learning_rate": training["learning_rate"],
            "checkpoint_every_epochs": training["checkpoint_every_epochs"],
            "image_size": training["image_size"],
            "seed": config["local_mode"]["seed"],
        },
    )
    training_step = TrainingStep(name="TrainBrandClassifier", estimator=estimator)
    evaluation_report = PropertyFile(
        name="EvaluationMetrics", output_name="evaluation", path="metrics.json"
    )
    evaluation_step = processing_step(
        "EvaluateModel",
        "evaluation.py",
        "evaluation",
        {key.upper(): str(value) for key, value in evaluation.items()},
        evaluation_report,
    )

    model = Model(
        image_uri="433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:1.7-1",
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        sagemaker_session=session,
    )
    register_step = RegisterModel(
        name="RegisterModelPendingManualApproval",
        model=model,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=brand.model_package_group,
        approval_status="PendingManualApproval",
    )
    evaluation_gate = ConditionStep(
        name="EvaluateModelQualityGate",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=evaluation_step.name,
                    property_file=evaluation_report,
                    json_path="accuracy",
                ),
                right=evaluation["accuracy_threshold"],
            ),
            ConditionLessThanOrEqualTo(
                left=JsonGet(
                    step_name=evaluation_step.name,
                    property_file=evaluation_report,
                    json_path="emergency_false_positive_rate",
                ),
                right=evaluation["fp_rate_emergency_threshold"],
            ),
            ConditionLessThanOrEqualTo(
                left=JsonGet(
                    step_name=evaluation_step.name,
                    property_file=evaluation_report,
                    json_path="latency_ms",
                ),
                right=evaluation["latency_budget_ms"],
            ),
        ],
        if_steps=[register_step],
        else_steps=[
            FailStep(name="FailEvaluationGate", error_message="Model evaluation gate failed.")
        ],
    )
    drift_gate = ConditionStep(
        name="DriftGate",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=drift_step.name, property_file=drift_report, json_path="confidence"
                ),
                right=config["drift"]["level1_confidence_floor"],
            )
        ],
        if_steps=[training_step, evaluation_step, evaluation_gate],
        else_steps=[FailStep(name="FailDriftGate", error_message="Drift gate failed.")],
    )
    quality_gate = ConditionStep(
        name="CleanlabQualityGate",
        conditions=[
            ConditionLessThanOrEqualTo(
                left=JsonGet(
                    step_name=quality_step.name,
                    property_file=quality_report,
                    json_path="flagged_fraction",
                ),
                right=config["cleanlab"]["flag_threshold_fraction"],
            )
        ],
        if_steps=[drift_step, drift_gate],
        else_steps=[
            FailStep(name="FailCleanlabQuality", error_message="Cleanlab quality gate failed.")
        ],
    )
    dataset_gate = ConditionStep(
        name="DatasetBoundsGate",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=selection_step.name,
                    property_file=selection_report,
                    json_path="dataset_size",
                ),
                right=config["data_selection"]["target_dataset_min"],
            )
        ],
        if_steps=[embeddings_step, quality_step, quality_gate],
        else_steps=[
            FailStep(name="FailDatasetBounds", error_message="Dataset bounds gate failed.")
        ],
    )
    return Pipeline(
        name=pipeline_name or f"miso-food-classifier-{brand.id}",
        parameters=[],
        steps=[selection_step, dataset_gate],
        sagemaker_session=session,
    )


if __name__ == "__main__":
    print(render_dry_run_json())

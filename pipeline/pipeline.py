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
    _validate_drift_config(config["drift"])
    return config


def _validate_drift_config(drift: Mapping[str, Any]) -> None:
    """Ensure the drift thresholds the gate relies on form a coherent band.

    The pipeline exposes ``ConfidenceBaseline`` as an operator-overridable
    pipeline parameter, so a bare ``float(...)`` at the use-site is not
    enough: a baseline outside ``[0, 1]`` or below the gate's floor would
    silently disable the gate. Reject that at definition-build time.
    """
    required_probability_keys = (
        "confidence_baseline",
        "level1_confidence_floor",
        "level2_confidence_threshold",
    )
    parsed: dict[str, float] = {}
    for key in required_probability_keys:
        if key not in drift:
            raise ValueError(f"config.yaml drift section missing {key!r}")
        try:
            value = float(drift[key])
        except (TypeError, ValueError) as error:
            raise ValueError(f"drift.{key} must be numeric, got {drift[key]!r}") from error
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"drift.{key} must be in [0, 1], got {value!r}")
        parsed[key] = value
    if "correction_rate_multiplier" not in drift:
        raise ValueError("config.yaml drift section missing 'correction_rate_multiplier'")
    try:
        multiplier = float(drift["correction_rate_multiplier"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"drift.correction_rate_multiplier must be numeric, got "
            f"{drift['correction_rate_multiplier']!r}"
        ) from error
    if not multiplier > 0.0:
        raise ValueError(
            f"drift.correction_rate_multiplier must be a positive float, got {multiplier!r}"
        )
    if parsed["confidence_baseline"] < parsed["level1_confidence_floor"]:
        raise ValueError(
            "drift.confidence_baseline must be >= drift.level1_confidence_floor"
        )
    if parsed["level2_confidence_threshold"] < parsed["level1_confidence_floor"]:
        raise ValueError(
            "drift.level2_confidence_threshold must be >= drift.level1_confidence_floor"
        )


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


# Infrastructure-identity names that MUST NOT be exposed as pipeline parameters.
# These are deployment facts (role ARN, buckets the pipeline role is scoped to) --
# not per-execution inputs. Baking them in as literals removes the
# StartPipelineExecution code-channel / role-override hijack path (CWE-494).
FORBIDDEN_PARAMETER_NAMES = frozenset(
    {
        "SageMakerRoleArn",
        "TrainingBucket",
        "ModelArtifactsBucket",
        "PipelineArtifactsBucket",
        "RawArchiveBucket",
    }
)


def build_parameters(config: Mapping[str, Any], brand: Brand) -> list[dict[str, Any]]:
    """Expose every configuration value used by the pipeline as an execution parameter.

    The SageMaker execution role ARN and the four project bucket names are
    deliberately omitted: they are baked into the definition as literals so a
    caller with ``sagemaker:StartPipelineExecution`` cannot redirect the code
    channel or the output locations by passing ``PipelineParameters``.
    """
    aws = config["aws"]
    selection = config["data_selection"]
    embedding = config["embedding"]
    kfold = config["kfold"]
    training = config["training"]
    drift = config["drift"]
    notifications = config["notifications"]
    local_mode = config["local_mode"]

    values = [
        ("Region", aws["region"], "AWS Region hosting this pipeline."),
        ("AccountId", aws["account_id"], "Target AWS account identifier."),
        ("BrandId", brand.id, "Fixed pipeline brand identifier."),
        ("BrandNamingPattern", brand.naming_pattern, "Allowed raw-data filename prefix pattern."),
        (
            "ModelPackageGroup",
            brand.model_package_group,
            "Brand-specific SageMaker Model Registry package group.",
        ),
        ("RecencyMonths", selection["recency_months"], "Maximum image age to select."),
        # NOTE: MinImagesPerClass, TargetDatasetMin, and TargetDatasetMax are deliberately
        # NOT exposed as pipeline parameters. They are safety-critical gate thresholds and
        # are baked as literal RightValues into the DatasetBoundsGate at definition-build
        # time so that StartPipelineExecution PipelineParameters cannot relax them.
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
        # NOTE: CleanlabFlagThresholdFraction is deliberately NOT exposed as a pipeline
        # parameter -- it is baked as the literal RightValue of CleanlabQualityGate so
        # callers of StartPipelineExecution cannot relax the label-quality floor.
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
        # NOTE: AccuracyThreshold, FpRateEmergencyThreshold, and LatencyBudgetMs are
        # deliberately NOT exposed as pipeline parameters. They are baked as literal
        # RightValues into EvaluateModelQualityGate so that a caller with
        # sagemaker:StartPipelineExecution cannot override and bypass the model-release
        # safety gates.
        # NOTE: DoneLabel/NotDoneLabel and the evaluation sample-count floors are
        # deliberately NOT exposed as pipeline parameters; they are baked into the
        # EvaluateModel environment and EvaluateModelQualityGate as literals.
        (
            "ConfidenceBaseline",
            drift["confidence_baseline"],
            "Reference confidence level for drift detection.",
        ),
        (
            "Level2ConfidenceThreshold",
            drift["level2_confidence_threshold"],
            "Level 2 escalation confidence threshold.",
        ),
        # NOTE: Level1ConfidenceFloor and CorrectionRateMultiplier are deliberately NOT
        # exposed as pipeline parameters -- they are baked as literal RightValues into
        # DriftGate so the drift safety gate is immutable per execution.
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
    parameters = [_parameter(*value) for value in values]
    # Defensive invariant: ensure no code path reintroduces an identity parameter.
    for parameter in parameters:
        if parameter["Name"] in FORBIDDEN_PARAMETER_NAMES:
            raise ValueError(
                f"Pipeline identity {parameter['Name']!r} must be a build-time literal, "
                "not an overridable execution parameter."
            )
    return parameters


def _ref(parameter_name: str) -> dict[str, str]:
    return {"Get": f"Parameters.{parameter_name}"}


def _step_ref(step_name: str, property_file: str, path: str) -> dict[str, Any]:
    return {
        "Std:JsonGet": {
            "PropertyFile": {"Get": f"Steps.{step_name}.PropertyFiles.{property_file}"},
            "Path": path,
        }
    }


def _s3_uri(bucket: str, *parts: str) -> str:
    """Return a concrete ``s3://`` URI literal.

    The bucket name is always baked in at build time (never a pipeline
    parameter reference) so the code channel and the output locations cannot be
    redirected by a ``StartPipelineExecution`` ``PipelineParameters`` override.
    """
    if not isinstance(bucket, str) or not bucket:
        raise ValueError("S3 bucket must be a non-empty literal string, not a reference.")
    cleaned = [part for part in parts if part != ""]
    suffix = "/" + "/".join(cleaned) if cleaned else "/"
    return f"s3://{bucket}{suffix}"


def _environment(*pairs: tuple[str, str]) -> dict[str, Any]:
    return {key: _ref(parameter) for key, parameter in pairs}


def _literal_env_value(value: Any) -> str:
    """Serialise a safety-critical config literal as the string a container will see.

    Booleans are emitted as ``"true"``/``"false"`` to match the SageMaker pipeline
    Parameter rendering they replace; everything else is passed through ``str``.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _processing_step(
    name: str,
    code_name: str,
    role_arn: str,
    code_s3_uri: str,
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    environment: dict[str, Any],
    property_files: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "Name": name,
        "Type": "Processing",
        "Arguments": {
            "RoleArn": role_arn,
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
                        "S3Uri": code_s3_uri,
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
    """Build a complete SageMaker-compatible local DAG without AWS or SDK imports.

    The SageMaker execution role ARN and all four project bucket names are
    resolved from configuration and baked into the definition as literal
    strings. They are *not* emitted as pipeline parameters, so a caller with
    ``sagemaker:StartPipelineExecution`` cannot override the code channel, the
    output locations, or the job role via ``PipelineParameters`` (CWE-494).
    """
    config = load_config(config_path)
    brand = get_brand(config, brand_id)

    role_arn = config["aws"]["sagemaker_role_arn"]
    training_bucket = config["s3"]["training_bucket"]
    model_artifacts_bucket = config["s3"]["model_artifacts_bucket"]
    pipeline_artifacts_bucket = config["s3"]["pipeline_artifacts_bucket"]
    raw_archive_bucket = config["s3"]["raw_archive_bucket"]
    code_s3_uri = _s3_uri(pipeline_artifacts_bucket, "code")

    select_data = _processing_step(
        "SelectRecentBrandData",
        "data_selection.py",
        role_arn,
        code_s3_uri,
        [
            {
                "InputName": "raw-archive",
                "S3Input": {
                    # Scope the raw-archive input to this brand's prefix so a
                    # cross-brand upload cannot be selected by this pipeline
                    # (CWE-639). Upload credentials must likewise be scoped to
                    # ``s3://<RawArchiveBucket>/<brand.id>/`` via an IAM policy
                    # with an ``s3:prefix`` condition.
                    "S3Uri": _s3_uri(raw_archive_bucket, brand.id),
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
                    "S3Uri": _s3_uri(training_bucket, brand.id, "selected"),
                    "LocalPath": "/opt/ml/processing/output/dataset",
                    "S3UploadMode": "EndOfJob",
                },
            },
            {
                "OutputName": "selection-metrics",
                "S3Output": {
                    "S3Uri": _s3_uri(pipeline_artifacts_bucket, brand.id, "selection"),
                    "LocalPath": "/opt/ml/processing/output/metrics",
                    "S3UploadMode": "EndOfJob",
                },
            },
        ],
        {
            **_environment(
                ("BRAND_ID", "BrandId"),
                ("NAMING_PATTERN", "BrandNamingPattern"),
                ("RECENCY_MONTHS", "RecencyMonths"),
                ("SEED", "Seed"),
            ),
            # Safety-critical dataset bounds are injected as literals so a per-execution
            # PipelineParameters override cannot change the in-container computation.
            "MIN_IMAGES_PER_CLASS": _literal_env_value(
                config["data_selection"]["min_images_per_class"]
            ),
            "TARGET_DATASET_MIN": _literal_env_value(
                config["data_selection"]["target_dataset_min"]
            ),
            "TARGET_DATASET_MAX": _literal_env_value(
                config["data_selection"]["target_dataset_max"]
            ),
        },
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
        role_arn,
        code_s3_uri,
        [
            {
                "InputName": "selected-dataset",
                "S3Input": {
                    "S3Uri": _s3_uri(training_bucket, brand.id, "selected"),
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
                    "S3Uri": _s3_uri(training_bucket, brand.id, "deduplicated"),
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
        role_arn,
        code_s3_uri,
        [
            {
                "InputName": "deduplicated-dataset",
                "S3Input": {
                    "S3Uri": _s3_uri(training_bucket, brand.id, "deduplicated"),
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
                    "S3Uri": _s3_uri(training_bucket, brand.id, "quality"),
                    "LocalPath": "/opt/ml/processing/output/dataset",
                    "S3UploadMode": "EndOfJob",
                },
            },
            {
                "OutputName": "quality-metrics",
                "S3Output": {
                    "S3Uri": _s3_uri(pipeline_artifacts_bucket, brand.id, "quality"),
                    "LocalPath": "/opt/ml/processing/output/metrics",
                    "S3UploadMode": "EndOfJob",
                },
            },
        ],
        {
            **_environment(
                ("KFOLD_COUNT", "KFoldCount"),
                ("CLASSIFIER", "Classifier"),
                ("SEED", "Seed"),
            ),
            # Safety-critical Cleanlab label-quality floor is injected as a literal so a
            # per-execution override cannot relax the in-container flagging budget.
            "CLEANLAB_FLAG_THRESHOLD_FRACTION": _literal_env_value(
                config["cleanlab"]["flag_threshold_fraction"]
            ),
            # Post-filter dataset floors are safety-critical too; inject as literals.
            "TARGET_DATASET_MIN": _literal_env_value(
                config["data_selection"]["target_dataset_min"]
            ),
            "MIN_IMAGES_PER_CLASS": _literal_env_value(
                config["data_selection"]["min_images_per_class"]
            ),
        },
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
        role_arn,
        code_s3_uri,
        [
            {
                "InputName": "quality-dataset",
                "S3Input": {
                    "S3Uri": _s3_uri(training_bucket, brand.id, "quality"),
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
                    "S3Uri": _s3_uri(pipeline_artifacts_bucket, brand.id, "drift"),
                    "LocalPath": "/opt/ml/processing/output/metrics",
                    "S3UploadMode": "EndOfJob",
                },
            }
        ],
        {
            **_environment(
                ("CONFIDENCE_BASELINE", "ConfidenceBaseline"),
                ("LEVEL2_CONFIDENCE_THRESHOLD", "Level2ConfidenceThreshold"),
            ),
            # Safety-critical drift thresholds are injected as literals so a per-execution
            # override cannot relax the in-container drift computation that feeds DriftGate.
            "LEVEL1_CONFIDENCE_FLOOR": _literal_env_value(
                config["drift"]["level1_confidence_floor"]
            ),
            "CORRECTION_RATE_MULTIPLIER": _literal_env_value(
                config["drift"]["correction_rate_multiplier"]
            ),
        },
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
            "RoleArn": role_arn,
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
                            "S3Uri": _s3_uri(training_bucket, brand.id, "quality"),
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                },
                {
                    "ChannelName": "code",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": code_s3_uri,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                },
            ],
            "OutputDataConfig": {
                "S3OutputPath": _s3_uri(model_artifacts_bucket, brand.id)
            },
            "ResourceConfig": {
                "InstanceType": _ref("TrainingInstanceType"),
                "InstanceCount": 1,
                "VolumeSizeInGB": 50,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": 86_400, "MaxWaitTimeInSeconds": 172_800},
            "EnableManagedSpotTraining": _ref("UseSpot"),
            "CheckpointConfig": {
                "S3Uri": _s3_uri(model_artifacts_bucket, brand.id, "checkpoints")
            },
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
        role_arn,
        code_s3_uri,
        [
            {
                # Mount only the exact training-job artifact (one model.tar.gz)
                # that RegisterModel publishes, so the measured model and the
                # registered model are guaranteed to be the same binary. Using
                # the shared brand-wide prefix here would also mount every
                # prior training job's output (and the checkpoints prefix),
                # letting any writer of that prefix decide what the release
                # gate measures. See validate_dry_run_definition's model-input
                # pinning assertion for the invariant this depends on.
                "InputName": "model-artifacts",
                "S3Input": {
                    "S3Uri": {
                        "Get": "Steps.TrainBrandClassifier.ModelArtifacts.S3ModelArtifacts"
                    },
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
                    "S3Uri": _s3_uri(pipeline_artifacts_bucket, brand.id, "evaluation"),
                    "LocalPath": "/opt/ml/processing/output/metrics",
                    "S3UploadMode": "EndOfJob",
                },
            }
        ],
        {
            # Safety-critical evaluation thresholds are injected as literals so a
            # per-execution PipelineParameters override cannot change the in-container
            # evaluation output that feeds EvaluateModelQualityGate.
            "ACCURACY_THRESHOLD": _literal_env_value(
                config["evaluation"]["accuracy_threshold"]
            ),
            "FP_RATE_EMERGENCY_THRESHOLD": _literal_env_value(
                config["evaluation"]["fp_rate_emergency_threshold"]
            ),
            "LATENCY_BUDGET_MS": _literal_env_value(
                config["evaluation"]["latency_budget_ms"]
            ),
            # Labels and sample-count floors feeding the emergency-FP metric and the
            # EvaluateModelQualityGate are safety-critical too; inject as literals.
            "DONE_LABEL": _literal_env_value(config["evaluation"].get("done_label", "done")),
            "NOT_DONE_LABEL": _literal_env_value(
                config["evaluation"].get("not_done_label", "not_done")
            ),
            "MIN_SAMPLE_COUNT": _literal_env_value(
                int(config["evaluation"].get("min_sample_count", 1))
            ),
            "MIN_PER_CLASS_COUNT": _literal_env_value(
                int(config["evaluation"].get("min_per_class_count", 1))
            ),
        },
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
                            pipeline_artifacts_bucket,
                            brand.id,
                            "evaluation",
                            "metrics.json",
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
                    "RightValue": config["evaluation"]["accuracy_threshold"],
                },
                {
                    "Type": "ConditionLessThanOrEqualTo",
                    "LeftValue": _step_ref(
                        "EvaluateModel", "EvaluationMetrics", "emergency_false_positive_rate"
                    ),
                    "RightValue": config["evaluation"]["fp_rate_emergency_threshold"],
                },
                {
                    "Type": "ConditionLessThanOrEqualTo",
                    "LeftValue": _step_ref("EvaluateModel", "EvaluationMetrics", "latency_ms"),
                    "RightValue": config["evaluation"]["latency_budget_ms"],
                },
                {
                    "Type": "ConditionGreaterThanOrEqualTo",
                    "LeftValue": _step_ref(
                        "EvaluateModel", "EvaluationMetrics", "sample_count"
                    ),
                    "RightValue": int(config["evaluation"].get("min_sample_count", 1)),
                },
                {
                    "Type": "ConditionGreaterThanOrEqualTo",
                    "LeftValue": _step_ref(
                        "EvaluateModel", "EvaluationMetrics", "not_done_sample_count"
                    ),
                    "RightValue": int(config["evaluation"].get("min_per_class_count", 1)),
                },
                {
                    "Type": "ConditionGreaterThanOrEqualTo",
                    "LeftValue": _step_ref(
                        "EvaluateModel", "EvaluationMetrics", "done_sample_count"
                    ),
                    "RightValue": int(config["evaluation"].get("min_per_class_count", 1)),
                },
                {
                    "Type": "ConditionLessThanOrEqualTo",
                    "LeftValue": _step_ref(
                        "EvaluateModel", "EvaluationMetrics", "missing_images"
                    ),
                    "RightValue": 0,
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
                    "RightValue": config["drift"]["level1_confidence_floor"],
                },
                {
                    "Type": "ConditionLessThanOrEqualTo",
                    "LeftValue": _step_ref(
                        "DetectDrift", "DriftMetrics", "correction_rate_multiplier"
                    ),
                    "RightValue": config["drift"]["correction_rate_multiplier"],
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
                    "RightValue": config["cleanlab"]["flag_threshold_fraction"],
                },
                # Reject any report whose detector is not the real CleanLab
                # detector -- fall-closed on silent degradation to the local
                # heuristic. ``detector`` is emitted by
                # processing/step3_cleanlab_filter.py alongside
                # ``flagged_fraction``.
                {
                    "Type": "ConditionEquals",
                    "LeftValue": _step_ref(
                        "KFoldCleanlabQuality", "QualityMetrics", "detector"
                    ),
                    "RightValue": "cleanlab",
                },
                {
                    "Type": "ConditionGreaterThanOrEqualTo",
                    "LeftValue": _step_ref(
                        "KFoldCleanlabQuality", "QualityMetrics", "cleaned_count"
                    ),
                    "RightValue": config["data_selection"]["target_dataset_min"],
                },
                {
                    "Type": "ConditionGreaterThanOrEqualTo",
                    "LeftValue": _step_ref(
                        "KFoldCleanlabQuality", "QualityMetrics", "minimum_class_count"
                    ),
                    "RightValue": config["data_selection"]["min_images_per_class"],
                },
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
                    "RightValue": config["data_selection"]["target_dataset_min"],
                },
                {
                    "Type": "ConditionLessThanOrEqualTo",
                    "LeftValue": _step_ref(
                        "SelectRecentBrandData", "SelectionMetrics", "dataset_size"
                    ),
                    "RightValue": config["data_selection"]["target_dataset_max"],
                },
                {
                    "Type": "ConditionGreaterThanOrEqualTo",
                    "LeftValue": _step_ref(
                        "SelectRecentBrandData", "SelectionMetrics", "minimum_class_count"
                    ),
                    "RightValue": config["data_selection"]["min_images_per_class"],
                },
            ],
            "IfSteps": [embed_and_deduplicate, label_quality, cleanlab_gate],
            "ElseSteps": [fail_dataset],
        },
    }

    return _regionalize_dlc_uris(
        {
            "Version": PIPELINE_VERSION,
            "Metadata": {
                "BrandId": brand.id,
                "GeneratedBy": "pipeline/export_definition.py",
            },
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


def validate_dry_run_definition(
    definition: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> None:
    """Validate the branch graph and non-negotiable model-release safeguards.

    When ``config`` is provided, the definition's BrandId and BrandNamingPattern
    defaults must match a brand declared in ``config.yaml``. This guards against
    exported definitions that pair one brand's id with another brand's naming
    pattern (which would cause the SelectRecentBrandData step to curate another
    tenant's data).
    """
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
    for parameter in (
        "BrandId",
        "BrandNamingPattern",
        "ModelPackageGroup",
        "UseSpot",
        "Seed",
    ):
        if parameter not in parameters:
            raise ValueError(f"Local DAG is missing required parameter: {parameter}")
    # Safety-critical gate thresholds must NEVER appear as execution parameters, since
    # SageMaker PipelineParameters are overridable per-execution via
    # StartPipelineExecution. They must be baked into the Condition steps as literal
    # RightValues instead.
    _forbidden_gate_parameters = (
        "AccuracyThreshold",
        "FpRateEmergencyThreshold",
        "LatencyBudgetMs",
        "CleanlabFlagThresholdFraction",
        "Level1ConfidenceFloor",
        "CorrectionRateMultiplier",
        "TargetDatasetMin",
        "TargetDatasetMax",
        "MinImagesPerClass",
        "EvaluationMinSampleCount",
        "EvaluationMinPerClassCount",
        "DoneLabel",
        "NotDoneLabel",
    )
    forbidden_present = sorted(name for name in _forbidden_gate_parameters if name in parameters)
    if forbidden_present:
        raise ValueError(
            "Safety-critical gate thresholds must not be exposed as pipeline parameters: "
            f"{forbidden_present}"
        )

    _validate_gate_right_values(definition)

    # Deployment identity (role ARN and project buckets) must be literals baked
    # into the definition, not overridable pipeline parameters. If a future
    # change reintroduces them, StartPipelineExecution callers would regain the
    # ability to redirect the code channel or run jobs under a different role.
    forbidden = FORBIDDEN_PARAMETER_NAMES.intersection(parameters)
    if forbidden:
        raise ValueError(
            "Pipeline identity values must be baked in as literals, not exposed "
            f"as execution parameters: {sorted(forbidden)}"
        )

    # Every RoleArn and every code-channel S3Uri must be a literal string --
    # no ``Get: Parameters.*`` indirection -- so a PipelineParameters override
    # cannot swap in an attacker role or an attacker code bucket.
    def _walk(node: Any, path: str = "$") -> None:
        if isinstance(node, Mapping):
            role = node.get("RoleArn") if "RoleArn" in node else None
            if role is not None and not isinstance(role, str):
                raise ValueError(
                    f"RoleArn at {path} must be a literal string; found parameter reference"
                )
            if (
                node.get("InputName") == "pipeline-code"
                and isinstance(node.get("S3Input"), Mapping)
            ):
                uri = node["S3Input"].get("S3Uri")
                if not isinstance(uri, str) or not uri.startswith("s3://"):
                    raise ValueError(
                        f"pipeline-code S3Uri at {path} must be a literal s3:// URI; "
                        "parameterised code channels enable code-channel hijack"
                    )
            if node.get("ChannelName") == "code":
                uri = (
                    node.get("DataSource", {})
                    .get("S3DataSource", {})
                    .get("S3Uri")
                )
                if not isinstance(uri, str) or not uri.startswith("s3://"):
                    raise ValueError(
                        f"training 'code' channel S3Uri at {path} must be a literal "
                        "s3:// URI; parameterised code channels enable code-channel hijack"
                    )
            for key, value in node.items():
                _walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                _walk(value, f"{path}[{index}]")

    _walk(definition)

    if config is not None:
        brand_id = parameters["BrandId"].get("DefaultValue")
        naming_pattern = parameters["BrandNamingPattern"].get("DefaultValue")
        model_package_group = parameters["ModelPackageGroup"].get("DefaultValue")
        try:
            declared = get_brand(config, brand_id)
        except ValueError as error:
            raise ValueError(
                f"Pipeline definition declares BrandId {brand_id!r} which is not "
                f"present in config.yaml"
            ) from error
        if naming_pattern != declared.naming_pattern:
            raise ValueError(
                f"BrandNamingPattern {naming_pattern!r} does not match brand "
                f"{brand_id!r} (expected {declared.naming_pattern!r}). This usually "
                f"means a single-brand baseline was cloned without substituting "
                f"brand-specific tokens."
            )
        if model_package_group != declared.model_package_group:
            raise ValueError(
                f"ModelPackageGroup {model_package_group!r} does not match brand "
                f"{brand_id!r} (expected {declared.model_package_group!r})."
            )

    register_step = next(
        step
        for step in _walk_steps(definition["Steps"])
        if step["Name"] == "RegisterModelPendingManualApproval"
    )
    status = register_step["Arguments"].get("ModelApprovalStatus")
    if status != "PendingManualApproval":
        raise ValueError("Registered models must remain PendingManualApproval")

    _validate_evaluate_matches_register(definition, register_step)


def _get_step_property_path(value: Any) -> str | None:
    """Return the ``Steps.*`` property path a ``{"Get": "Steps.*"}`` reference
    resolves to, or ``None`` if ``value`` is not such a reference.
    """
    if not isinstance(value, Mapping):
        return None
    target = value.get("Get")
    if not isinstance(target, str) or not target.startswith("Steps."):
        return None
    return target


def _validate_evaluate_matches_register(
    definition: Mapping[str, Any],
    register_step: Mapping[str, Any],
) -> None:
    """Assert EvaluateModel's model input and RegisterModel's ModelDataUrl
    resolve to the same training-step property.

    If the two steps measure and register different artifacts, the release gate
    is decoupled from what gets published (CWE-494). This check requires both
    references to be a literal ``{"Get": "Steps.<Train>.ModelArtifacts.S3ModelArtifacts"}``
    pointing at the same training step, so a bucket-prefix mount cannot be
    silently reintroduced for EvaluateModel.
    """
    evaluate_step = next(
        (
            step
            for step in _walk_steps(definition.get("Steps", []))
            if step.get("Name") == "EvaluateModel"
        ),
        None,
    )
    if evaluate_step is None:
        raise ValueError("Local DAG is missing required step: EvaluateModel")

    processing_inputs = (
        evaluate_step.get("Arguments", {}).get("ProcessingInputs", [])
    )
    model_inputs = [
        entry
        for entry in processing_inputs
        if isinstance(entry, Mapping) and entry.get("InputName") == "model-artifacts"
    ]
    if len(model_inputs) != 1:
        raise ValueError(
            "EvaluateModel must declare exactly one 'model-artifacts' input; "
            f"found {len(model_inputs)}"
        )
    eval_s3_uri = model_inputs[0].get("S3Input", {}).get("S3Uri")
    eval_path = _get_step_property_path(eval_s3_uri)
    if eval_path is None:
        raise ValueError(
            "EvaluateModel 'model-artifacts' S3Uri must be a step-property "
            "reference of the form {'Get': 'Steps.<Train>.ModelArtifacts."
            "S3ModelArtifacts'} so the evaluated artifact is pinned to the "
            f"training-job output; found {eval_s3_uri!r}. Mounting a bucket "
            "prefix here decouples the release gate from the registered model."
        )

    containers = (
        register_step.get("Arguments", {})
        .get("InferenceSpecification", {})
        .get("Containers", [])
    )
    if not containers:
        raise ValueError(
            "RegisterModelPendingManualApproval must declare at least one "
            "InferenceSpecification container with a ModelDataUrl."
        )
    register_path = _get_step_property_path(containers[0].get("ModelDataUrl"))
    if register_path is None:
        raise ValueError(
            "RegisterModel ModelDataUrl must be a step-property reference of the "
            "form {'Get': 'Steps.<Train>.ModelArtifacts.S3ModelArtifacts'}; found "
            f"{containers[0].get('ModelDataUrl')!r}."
        )

    if eval_path != register_path:
        raise ValueError(
            "EvaluateModel's model input and RegisterModel's ModelDataUrl must "
            "resolve to the same training-step property so the gate measures "
            f"what gets registered; EvaluateModel={eval_path!r} vs "
            f"RegisterModel={register_path!r}."
        )
    if not eval_path.endswith(".ModelArtifacts.S3ModelArtifacts"):
        raise ValueError(
            "EvaluateModel/RegisterModel must both reference "
            "'Steps.<Train>.ModelArtifacts.S3ModelArtifacts'; got "
            f"{eval_path!r}."
        )


# Policy-approved ranges for the safety-critical gate RightValues. These are
# inclusive bounds that any brand-overlay configuration must stay within. They are
# the ultimate backstop against a definition-build-time overlay accidentally (or
# maliciously) relaxing a safety gate even when the gate is now a literal rather
# than an execution parameter.
_GATE_RIGHT_VALUE_POLICY: dict[tuple[str, str, str], tuple[float, float]] = {
    ("EvaluateModelQualityGate", "accuracy", "ConditionGreaterThanOrEqualTo"): (0.90, 1.0),
    (
        "EvaluateModelQualityGate",
        "emergency_false_positive_rate",
        "ConditionLessThanOrEqualTo",
    ): (0.0, 0.10),
    ("EvaluateModelQualityGate", "latency_ms", "ConditionLessThanOrEqualTo"): (1.0, 5_000.0),
    ("DriftGate", "confidence", "ConditionGreaterThanOrEqualTo"): (0.50, 1.0),
    ("DriftGate", "correction_rate_multiplier", "ConditionLessThanOrEqualTo"): (1.0, 10.0),
    ("CleanlabQualityGate", "flagged_fraction", "ConditionLessThanOrEqualTo"): (0.0, 0.10),
    ("DatasetBoundsGate", "dataset_size:min", "ConditionGreaterThanOrEqualTo"): (1.0, 1_000_000.0),
    ("DatasetBoundsGate", "dataset_size:max", "ConditionLessThanOrEqualTo"): (1.0, 10_000_000.0),
    ("DatasetBoundsGate", "minimum_class_count", "ConditionGreaterThanOrEqualTo"): (1.0, 100_000.0),
}


def _left_value_json_path(left_value: Mapping[str, Any]) -> str | None:
    json_get = left_value.get("Std:JsonGet") if isinstance(left_value, Mapping) else None
    if not isinstance(json_get, Mapping):
        return None
    path = json_get.get("Path")
    return path if isinstance(path, str) else None


def _validate_gate_right_values(definition: Mapping[str, Any]) -> None:
    """Assert that every safety gate's RightValue is a literal in its policy range.

    SageMaker Condition steps compare a step-produced LeftValue against a RightValue.
    A RightValue that is a ``{"Get": "Parameters.*"}`` reference is caller-overridable
    at StartPipelineExecution time and therefore cannot carry a safety guarantee.
    This check enforces that every gate RightValue is a plain number baked into the
    definition (or, for ``ConditionEquals``, a literal string), and that numeric
    thresholds fall within the policy-approved range.
    """
    seen: set[tuple[str, str, str]] = set()
    for step in _walk_steps(definition.get("Steps", [])):
        if step.get("Type") != "Condition":
            continue
        step_name = step.get("Name", "<unnamed>")
        conditions = step.get("Arguments", {}).get("Conditions", [])
        # Track per-gate occurrences so we can distinguish the two dataset-size bounds
        # (min vs max) which share the same JSON path.
        occurrences: dict[tuple[str, str], int] = {}
        for condition in conditions:
            condition_type = condition.get("Type", "")
            json_path = _left_value_json_path(condition.get("LeftValue", {}))
            right_value = condition.get("RightValue")
            if isinstance(right_value, Mapping):
                raise ValueError(
                    f"Gate {step_name!r} RightValue for {json_path!r} must be a literal "
                    f"number, not a reference: {dict(right_value)!r}. "
                    "Safety-critical thresholds must not be caller-overridable pipeline "
                    "parameters."
                )
            if condition_type == "ConditionEquals" and isinstance(right_value, str):
                # Literal string equality (e.g. ``detector == "cleanlab"``) is baked
                # into the definition and not caller-overridable, so it is allowed.
                continue
            if isinstance(right_value, bool) or not isinstance(right_value, int | float):
                raise ValueError(
                    f"Gate {step_name!r} RightValue for {json_path!r} must be a number; "
                    f"got {right_value!r}"
                )
            if json_path is None:
                # Unknown shape; still require a numeric literal (already checked) but
                # skip range enforcement.
                continue
            lookup_path = json_path
            if step_name == "DatasetBoundsGate" and json_path == "dataset_size":
                occurrences[(step_name, json_path)] = occurrences.get((step_name, json_path), 0) + 1
                suffix = "min" if occurrences[(step_name, json_path)] == 1 else "max"
                lookup_path = f"dataset_size:{suffix}"
            key = (step_name, lookup_path, condition_type)
            policy = _GATE_RIGHT_VALUE_POLICY.get(key)
            if policy is None:
                # No policy declared for this gate/path/comparator -- leave the literal
                # check in place but do not range-enforce.
                continue
            low, high = policy
            if not (low <= float(right_value) <= high):
                raise ValueError(
                    f"Gate {step_name!r} RightValue for {json_path!r} ({right_value}) is "
                    f"outside policy-approved range [{low}, {high}]"
                )
            seen.add(key)

    missing_policies = set(_GATE_RIGHT_VALUE_POLICY).difference(seen)
    if missing_policies:
        raise ValueError(
            "Safety gate(s) missing required Condition(s): "
            f"{sorted(missing_policies)}"
        )


def render_dry_run_json(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    brand_id: str = "white-castle",
) -> str:
    """Return a canonical JSON representation of the complete local DAG."""
    config = load_config(config_path)
    definition = build_dry_run_definition(config_path, brand_id)
    validate_dry_run_definition(definition, config=config)
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
            ConditionEquals,
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
            "TARGET_DATASET_MIN": str(config["data_selection"]["target_dataset_min"]),
            "MIN_IMAGES_PER_CLASS": str(config["data_selection"]["min_images_per_class"]),
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
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=evaluation_step.name,
                    property_file=evaluation_report,
                    json_path="sample_count",
                ),
                right=int(evaluation.get("min_sample_count", 1)),
            ),
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=evaluation_step.name,
                    property_file=evaluation_report,
                    json_path="not_done_sample_count",
                ),
                right=int(evaluation.get("min_per_class_count", 1)),
            ),
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=evaluation_step.name,
                    property_file=evaluation_report,
                    json_path="done_sample_count",
                ),
                right=int(evaluation.get("min_per_class_count", 1)),
            ),
            ConditionLessThanOrEqualTo(
                left=JsonGet(
                    step_name=evaluation_step.name,
                    property_file=evaluation_report,
                    json_path="missing_images",
                ),
                right=0,
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
            ),
            ConditionLessThanOrEqualTo(
                left=JsonGet(
                    step_name=drift_step.name,
                    property_file=drift_report,
                    json_path="correction_rate_multiplier",
                ),
                right=config["drift"]["correction_rate_multiplier"],
            ),
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
            ),
            # Reject any report whose detector is not the real CleanLab
            # detector -- fail closed on silent degradation to the local
            # heuristic fallback.
            ConditionEquals(
                left=JsonGet(
                    step_name=quality_step.name,
                    property_file=quality_report,
                    json_path="detector",
                ),
                right="cleanlab",
            ),
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=quality_step.name,
                    property_file=quality_report,
                    json_path="cleaned_count",
                ),
                right=config["data_selection"]["target_dataset_min"],
            ),
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=quality_step.name,
                    property_file=quality_report,
                    json_path="minimum_class_count",
                ),
                right=config["data_selection"]["min_images_per_class"],
            ),
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
            ),
            ConditionLessThanOrEqualTo(
                left=JsonGet(
                    step_name=selection_step.name,
                    property_file=selection_report,
                    json_path="dataset_size",
                ),
                right=config["data_selection"]["target_dataset_max"],
            ),
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=selection_step.name,
                    property_file=selection_report,
                    json_path="minimum_class_count",
                ),
                right=config["data_selection"]["min_images_per_class"],
            ),
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


# Expected condition count per gate, keyed by step name. Both the local DAG and the
# SDK-emitted definition must agree on this contract so the artifact actually
# deployed enforces every safety floor declared in config.yaml.
_EXPECTED_GATE_CONDITION_COUNTS: Mapping[str, int] = {
    "DatasetBoundsGate": 3,
    "CleanlabQualityGate": 1,
    "DriftGate": 2,
    "EvaluateModelQualityGate": 3,
}


def validate_sdk_pipeline_definition(definition: str | Mapping[str, Any]) -> None:
    """Validate the SDK-emitted pipeline definition against the local DAG's contract.

    The SDK builder has historically been maintained independently of
    ``build_dry_run_definition`` and can silently omit gate conditions enforced by
    the validated local DAG. This function parses the SDK ``Pipeline.definition()``
    JSON (or an equivalent mapping) and asserts the same safety-critical
    invariants ``validate_dry_run_definition`` enforces: required steps are
    present, each gate carries the full set of ``ConditionStep`` conditions, and
    the model registration remains ``PendingManualApproval``.
    """
    if isinstance(definition, str):
        parsed = json.loads(definition)
    else:
        parsed = dict(definition)

    steps = parsed.get("Steps", [])
    flattened = _walk_steps(steps)
    names = [step["Name"] for step in flattened]

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
        "FailDatasetBounds",
        "FailCleanlabQuality",
        "FailDriftGate",
        "FailEvaluationGate",
    }
    missing = required.difference(names)
    if missing:
        raise ValueError(f"SDK pipeline is missing required steps: {sorted(missing)}")

    by_name = {step["Name"]: step for step in flattened}
    for gate_name, expected_count in _EXPECTED_GATE_CONDITION_COUNTS.items():
        gate = by_name.get(gate_name)
        if gate is None:
            raise ValueError(f"SDK pipeline is missing required gate: {gate_name}")
        conditions = gate.get("Arguments", {}).get("Conditions", [])
        if len(conditions) != expected_count:
            raise ValueError(
                f"SDK pipeline gate {gate_name!r} has {len(conditions)} condition(s); "
                f"expected {expected_count} to match the validated local DAG"
            )

    register_step = next(
        (
            step
            for step in flattened
            if step.get("Type") == "RegisterModel"
            or step.get("Name", "").startswith("RegisterModelPendingManualApproval")
        ),
        None,
    )
    if register_step is None:
        raise ValueError("SDK pipeline is missing the RegisterModel step")
    status = register_step.get("Arguments", {}).get("ModelApprovalStatus")
    if status != "PendingManualApproval":
        raise ValueError(
            "SDK pipeline must register models as PendingManualApproval; "
            f"found {status!r}"
        )


if __name__ == "__main__":
    print(render_dry_run_json())

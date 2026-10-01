"""Build, upsert, and (optionally) run the full Miso SageMaker Pipeline.

This constructs the real multi-step DAG the console visualizes:

  SelectRecentBrandData
    -> DatasetBoundsGate
         (pass) EmbedAndDeduplicate -> KFoldCleanlabQuality -> CleanlabQualityGate
                  (pass) DetectDrift -> DriftGate
                           (pass) TrainBrandClassifier -> EvaluateModel
                                    -> EvaluateModelQualityGate
                                         (pass) RegisterModel (PendingManualApproval)
                                         (fail) FailEvaluationGate
                           (fail) FailDriftGate
                  (fail) FailCleanlabQuality
         (fail) FailDatasetBounds

Every step is its own SageMaker job. Thresholds come from a config file
(config.smoke.yaml relaxes the production floors so a 50-image dataset reaches
the manual-approval gate).

Usage:
    python pipeline/deploy_pipeline.py --config config.smoke.yaml --upsert --run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from sagemaker.estimator import Estimator
from sagemaker.image_uris import retrieve
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEPS_DIR = PROJECT_ROOT / "pipeline" / "steps"


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build(config: dict, code_prefix: str, images_uri: str) -> Pipeline:
    region = config["aws"]["region"]
    role = config["aws"]["sagemaker_role_arn"]
    bucket = config["s3"]["pipeline_artifacts_bucket"]
    brand = config["brands"][0]
    brand_id = brand["id"]
    group = brand["model_package_group"]
    sel = config["data_selection"]
    session = PipelineSession()

    base = f"s3://{bucket}/miso-smoke"
    code_channel = f"{base}/{code_prefix}"

    tf_image = retrieve(
        framework="tensorflow",
        region=region,
        version="2.14.1",
        image_scope="training",
        instance_type="ml.m5.xlarge",
    )

    def processor(env: dict | None = None) -> ScriptProcessor:
        return ScriptProcessor(
            image_uri=tf_image,
            command=["python3"],
            role=role,
            instance_count=1,
            instance_type=config["training"]["processing_instance_type"],
            sagemaker_session=session,
            base_job_name="miso",
            env=env or {},
        )

    code_input = ProcessingInput(
        source=code_channel,
        destination="/opt/ml/processing/input/pkgcode",
        input_name="pkgcode",
    )
    images_input = ProcessingInput(
        source=images_uri,
        destination="/opt/ml/processing/input/images",
        input_name="images",
    )

    # ---- Step: SelectRecentBrandData ----
    sel_metrics = PropertyFile(
        name="SelectionMetrics", output_name="sel_metrics", path="metrics.json"
    )
    select_step = ProcessingStep(
        name="SelectRecentBrandData",
        step_args=processor(
            {
                "NAMING_PATTERN": brand["naming_pattern"],
                "MIN_IMAGES_PER_CLASS": str(sel["min_images_per_class"]),
                "TARGET_DATASET_MIN": str(sel["target_dataset_min"]),
                "TARGET_DATASET_MAX": str(sel["target_dataset_max"]),
            }
        ).run(
            code=str(STEPS_DIR / "data_selection.py"),
            inputs=[images_input, code_input],
            outputs=[
                ProcessingOutput(
                    output_name="sel_dataset",
                    source="/opt/ml/processing/output/dataset",
                    destination=f"{base}/pipe/select/dataset",
                ),
                ProcessingOutput(
                    output_name="sel_metrics",
                    source="/opt/ml/processing/output/metrics",
                    destination=f"{base}/pipe/select/metrics",
                ),
            ],
        ),
        property_files=[sel_metrics],
    )

    # ---- Step: EmbedAndDeduplicate ----
    embed_step = ProcessingStep(
        name="EmbedAndDeduplicate",
        step_args=processor(
            {
                "EMBEDDING_DIM": str(config["embedding"]["embedding_dim"]),
                "COSINE_DEDUP_THRESHOLD": str(config["embedding"]["cosine_dedup_threshold"]),
                "BACKBONE_SOURCE": config["embedding"]["backbone_source"],
            }
        ).run(
            code=str(STEPS_DIR / "embedding_dedup.py"),
            inputs=[
                images_input,
                code_input,
                ProcessingInput(
                    source=select_step.properties.ProcessingOutputConfig.Outputs[
                        "sel_dataset"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/input/dataset",
                    input_name="in_dataset",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="embed_dataset",
                    source="/opt/ml/processing/output/dataset",
                    destination=f"{base}/pipe/embed/dataset",
                )
            ],
        ),
    )

    # ---- Step: KFoldCleanlabQuality ----
    quality_metrics = PropertyFile(
        name="QualityMetrics", output_name="quality_metrics", path="metrics.json"
    )
    quality_step = ProcessingStep(
        name="KFoldCleanlabQuality",
        step_args=processor(
            {
                "KFOLD_COUNT": str(config["kfold"]["n_folds"]),
                "CLASSIFIER": config["kfold"]["classifier"],
                "CLEANLAB_FLAG_THRESHOLD_FRACTION": str(
                    config["cleanlab"]["flag_threshold_fraction"]
                ),
            }
        ).run(
            code=str(STEPS_DIR / "kfold_cleanlab.py"),
            inputs=[
                code_input,
                ProcessingInput(
                    source=embed_step.properties.ProcessingOutputConfig.Outputs[
                        "embed_dataset"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/input/dataset",
                    input_name="in_dataset",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="quality_dataset",
                    source="/opt/ml/processing/output/dataset",
                    destination=f"{base}/pipe/quality/dataset",
                ),
                ProcessingOutput(
                    output_name="quality_metrics",
                    source="/opt/ml/processing/output/metrics",
                    destination=f"{base}/pipe/quality/metrics",
                ),
            ],
        ),
        property_files=[quality_metrics],
    )

    # ---- Step: DetectDrift ----
    drift_metrics = PropertyFile(
        name="DriftMetrics", output_name="drift_metrics", path="metrics.json"
    )
    drift_step = ProcessingStep(
        name="DetectDrift",
        step_args=processor(
            {"CONFIDENCE_BASELINE": str(config["drift"]["confidence_baseline"])}
        ).run(
            code=str(STEPS_DIR / "drift_detection.py"),
            inputs=[
                code_input,
                ProcessingInput(
                    source=embed_step.properties.ProcessingOutputConfig.Outputs[
                        "embed_dataset"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/input/dataset",
                    input_name="in_dataset",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="drift_metrics",
                    source="/opt/ml/processing/output/metrics",
                    destination=f"{base}/pipe/drift/metrics",
                )
            ],
        ),
        property_files=[drift_metrics],
    )

    # ---- Step: TrainBrandClassifier ----
    estimator = Estimator(
        image_uri=tf_image,
        role=role,
        instance_count=1,
        instance_type=config["training"]["instance_type"],
        output_path=f"{base}/pipe/train",
        sagemaker_session=session,
        base_job_name="miso-train",
        entry_point=str(STEPS_DIR / "train_step.py"),
        environment={
            "BRAND_ID": brand_id,
            "EPOCHS": str(config["training"]["epochs"]),
            "BATCH_SIZE": str(config["training"]["batch_size"]),
        },
    )
    train_step = TrainingStep(
        name="TrainBrandClassifier",
        step_args=estimator.fit(
            inputs={
                "images": images_uri,
                "dataset": quality_step.properties.ProcessingOutputConfig.Outputs[
                    "quality_dataset"
                ].S3Output.S3Uri,
                "pkgcode": code_channel,
            }
        ),
    )

    # ---- Step: EvaluateModel ----
    eval_metrics = PropertyFile(
        name="EvaluationMetrics", output_name="eval_metrics", path="metrics.json"
    )
    eval_step = ProcessingStep(
        name="EvaluateModel",
        step_args=processor({"IMAGE_SIZE": str(config["training"]["image_size"])}).run(
            code=str(STEPS_DIR / "evaluation.py"),
            inputs=[
                images_input,
                code_input,
                ProcessingInput(
                    source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/input/model",
                    input_name="model",
                ),
                ProcessingInput(
                    source=quality_step.properties.ProcessingOutputConfig.Outputs[
                        "quality_dataset"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/input/dataset",
                    input_name="in_dataset",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="eval_metrics",
                    source="/opt/ml/processing/output/metrics",
                    destination=f"{base}/pipe/eval/metrics",
                )
            ],
        ),
        property_files=[eval_metrics],
    )

    # ---- RegisterModel ----
    model = Model(
        image_uri=tf_image,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
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
        model_package_group_name=group,
        approval_status="PendingManualApproval",
    )

    # ---- Gates ----
    eval_gate = ConditionStep(
        name="EvaluateModelQualityGate",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=eval_step.name, property_file=eval_metrics, json_path="accuracy"
                ),
                right=config["evaluation"]["accuracy_threshold"],
            ),
            ConditionLessThanOrEqualTo(
                left=JsonGet(
                    step_name=eval_step.name,
                    property_file=eval_metrics,
                    json_path="emergency_false_positive_rate",
                ),
                right=config["evaluation"]["fp_rate_emergency_threshold"],
            ),
            ConditionLessThanOrEqualTo(
                left=JsonGet(
                    step_name=eval_step.name, property_file=eval_metrics, json_path="latency_ms"
                ),
                right=config["evaluation"]["latency_budget_ms"],
            ),
        ],
        if_steps=[register_step],
        else_steps=[FailStep(name="FailEvaluationGate", error_message="Evaluation gate failed.")],
    )
    drift_gate = ConditionStep(
        name="DriftGate",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=drift_step.name, property_file=drift_metrics, json_path="confidence"
                ),
                right=config["drift"]["level1_confidence_floor"],
            ),
            ConditionLessThanOrEqualTo(
                left=JsonGet(
                    step_name=drift_step.name,
                    property_file=drift_metrics,
                    json_path="correction_rate_multiplier",
                ),
                right=config["drift"]["correction_rate_multiplier"],
            ),
        ],
        if_steps=[train_step, eval_step, eval_gate],
        else_steps=[FailStep(name="FailDriftGate", error_message="Drift gate failed.")],
    )
    cleanlab_gate = ConditionStep(
        name="CleanlabQualityGate",
        conditions=[
            ConditionLessThanOrEqualTo(
                left=JsonGet(
                    step_name=quality_step.name,
                    property_file=quality_metrics,
                    json_path="flagged_fraction",
                ),
                right=config["cleanlab"]["flag_threshold_fraction"],
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
                    step_name=select_step.name, property_file=sel_metrics, json_path="dataset_size"
                ),
                right=sel["target_dataset_min"],
            ),
            ConditionLessThanOrEqualTo(
                left=JsonGet(
                    step_name=select_step.name, property_file=sel_metrics, json_path="dataset_size"
                ),
                right=sel["target_dataset_max"],
            ),
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=select_step.name,
                    property_file=sel_metrics,
                    json_path="minimum_class_count",
                ),
                right=sel["min_images_per_class"],
            ),
        ],
        if_steps=[embed_step, quality_step, cleanlab_gate],
        else_steps=[
            FailStep(name="FailDatasetBounds", error_message="Dataset bounds gate failed.")
        ],
    )

    return Pipeline(
        name=f"miso-food-classifier-{brand_id}",
        steps=[select_step, dataset_gate],
        sagemaker_session=session,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.smoke.yaml")
    parser.add_argument("--code-prefix", default="code")
    parser.add_argument("--images-uri", required=True)
    parser.add_argument("--upsert", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = build(config, args.code_prefix, args.images_uri)
    role = config["aws"]["sagemaker_role_arn"]

    if args.upsert:
        pipeline.upsert(role_arn=role)
        print(f"[deploy] upserted pipeline: {pipeline.name}")
    if args.run:
        execution = pipeline.start()
        print(f"[deploy] started execution: {execution.arn}")


if __name__ == "__main__":
    main()

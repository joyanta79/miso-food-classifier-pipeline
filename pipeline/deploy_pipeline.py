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
    # Production deployment:
    python pipeline/deploy_pipeline.py --config config.yaml --images-uri s3://... --upsert

    # Smoke run (relaxed overlay) — a smoke pipeline name and an explicit
    # acknowledgement are required so --upsert cannot overwrite the
    # Terraform-managed production pipeline:
    python pipeline/deploy_pipeline.py --config config.smoke.yaml \\
        --images-uri s3://... --upsert --run \\
        --i-understand-this-is-a-smoke-config
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sagemaker.estimator import Estimator
from sagemaker.image_uris import retrieve
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEPS_DIR = PROJECT_ROOT / "pipeline" / "steps"

# Production safety floors are DERIVED from ``config.yaml`` at the repo root --
# the file Terraform consumes -- rather than being duplicated in a hard-coded
# table that can drift. The shared ``load_config`` already enforces structure
# and types; this list only declares which fields are safety-critical and
# which direction (``ge``/``le``) a config value must satisfy to be
# at-least-as-strict-as-production.
#   "ge": production value is a minimum (deploy value must be >= production)
#   "le": production value is a maximum (deploy value must be <= production)
#
# Every field here must be present and numeric in both the production config
# and the deploy config -- a missing or non-numeric field is a HARD ERROR (not
# a silent 'not relaxed'), so an attacker cannot bypass the guard by deleting
# the field or writing a list literal in its place.
_PRODUCTION_GATE_FLOOR_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("data_selection", "min_images_per_class", "ge"),
    ("data_selection", "target_dataset_min", "ge"),
    ("data_selection", "target_dataset_max", "le"),
    ("cleanlab", "flag_threshold_fraction", "le"),
    ("evaluation", "accuracy_threshold", "ge"),
    ("evaluation", "fp_rate_emergency_threshold", "le"),
    ("evaluation", "latency_budget_ms", "le"),
    ("evaluation", "min_sample_count", "ge"),
    ("evaluation", "min_per_class_count", "ge"),
    ("drift", "level1_confidence_floor", "ge"),
    ("drift", "level2_confidence_threshold", "ge"),
    ("drift", "correction_rate_multiplier", "le"),
)

# Label-pair fields: deploy must declare the SAME label-to-class binding as the
# production config so ``emergency_false_positive_rate`` continues to measure
# the dangerous direction. Swapping ``done_label``/``not_done_label`` would
# make the gate measure the harmless direction.
_PRODUCTION_LABEL_FIELDS: tuple[tuple[str, str], ...] = (
    ("evaluation", "done_label"),
    ("evaluation", "not_done_label"),
)

PRODUCTION_PIPELINE_PREFIX = "miso-food-classifier"
SMOKE_PIPELINE_PREFIX = "smoke-miso-food-classifier"


def _production_config() -> dict[str, Any]:
    """Load the repository's production ``config.yaml`` as the floor source.

    Floors are not duplicated in this module -- they come from the same file
    Terraform renders into ``infra/pipeline_definition.<brand>.json``.
    """
    from pipeline.pipeline import DEFAULT_CONFIG_PATH  # noqa: WPS433
    from pipeline.pipeline import load_config as _shared_load_config  # noqa: WPS433

    return _shared_load_config(DEFAULT_CONFIG_PATH)


def _require_section_value(
    source: str, config: dict[str, Any], section: str, key: str
) -> Any:
    """Return a required key from ``config[section]`` or raise ``ValueError``."""
    section_cfg = config.get(section)
    if not isinstance(section_cfg, dict) or key not in section_cfg:
        raise ValueError(
            f"{source} is missing required field {section}.{key!r} "
            "used by the production safety gate"
        )
    return section_cfg[key]


def _coerce_numeric(source: str, section: str, key: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"{source} {section}.{key} must be numeric, got {value!r}"
        )
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{source} {section}.{key} must be numeric, got {value!r}"
        ) from error


def _coerce_label(source: str, section: str, key: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{source} {section}.{key} must be a non-empty string label, got "
            f"{value!r}"
        )
    return value


def load_config(path: str) -> dict[str, Any]:
    """Load the deploy configuration using the shared validator.

    Delegates to :func:`pipeline.pipeline.load_config` so that required sections
    and the brand list are validated before anything is deployed. The import is
    performed lazily to keep this module importable without the shared package
    being on ``sys.path`` at module import time (the script runs from the repo
    root).
    """
    from pipeline.pipeline import load_config as _shared_load_config  # noqa: WPS433

    return _shared_load_config(path)


def _relaxed_fields(config: dict[str, Any]) -> list[str]:
    """Return the list of fields whose values are weaker than the production floor.

    Floors are derived from the repository's production ``config.yaml`` -- there
    is no duplicated hard-coded table that could lag the real production
    thresholds. Missing or non-numeric gate fields in either the production
    config or the deploy config are a HARD ERROR (``ValueError``); they are
    never silently treated as 'not relaxed'.
    """
    production = _production_config()
    weak: list[str] = []
    for section, key, op in _PRODUCTION_GATE_FLOOR_FIELDS:
        floor_raw = _require_section_value(
            "production config.yaml", production, section, key
        )
        value_raw = _require_section_value("deploy config", config, section, key)
        floor = _coerce_numeric("production config.yaml", section, key, floor_raw)
        value = _coerce_numeric("deploy config", section, key, value_raw)
        if op == "ge" and value < floor:
            weak.append(f"{section}.{key}={value_raw} < production floor {floor_raw}")
        elif op == "le" and value > floor:
            weak.append(f"{section}.{key}={value_raw} > production floor {floor_raw}")
    for section, key in _PRODUCTION_LABEL_FIELDS:
        production_label = _coerce_label(
            "production config.yaml",
            section,
            key,
            _require_section_value("production config.yaml", production, section, key),
        )
        deploy_label = _coerce_label(
            "deploy config",
            section,
            key,
            _require_section_value("deploy config", config, section, key),
        )
        if deploy_label != production_label:
            weak.append(
                f"{section}.{key}={deploy_label!r} != production value "
                f"{production_label!r}"
            )
    return weak


def build(
    config: dict,
    code_prefix: str,
    images_uri: str,
    pipeline_name: str | None = None,
) -> Pipeline:
    """Build the Miso SageMaker Pipeline for the first configured brand.

    ``images_uri`` MUST point at this brand's dedicated raw-image prefix
    (``s3://<raw-archive-bucket>/<brand.id>/``), not at the bucket root --
    the step container only cross-checks the ``<brand>`` token in each
    filename as defence in depth. Scope the upload credentials for the raw
    archive to the same prefix (``s3:prefix`` IAM condition) to prevent
    cross-brand data mixing (CWE-639).
    """
    from pipeline.pipeline import get_brand  # noqa: WPS433 (lazy, see load_config)

    region = config["aws"]["region"]
    role = config["aws"]["sagemaker_role_arn"]
    bucket = config["s3"]["pipeline_artifacts_bucket"]
    first_brand_id = config["brands"][0]["id"]
    # Validate the brand against the declared registry group (refuses arbitrary
    # package-group combinations) instead of trusting config["brands"][0] blindly.
    brand = get_brand(config, first_brand_id)
    brand_id = brand.id
    group = brand.model_package_group
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
                "TARGET_DATASET_MIN": str(sel["target_dataset_min"]),
                "MIN_IMAGES_PER_CLASS": str(sel["min_images_per_class"]),
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
            {
                "CONFIDENCE_BASELINE": str(config["drift"]["confidence_baseline"]),
                "LEVEL1_CONFIDENCE_FLOOR": str(config["drift"]["level1_confidence_floor"]),
                "LEVEL2_CONFIDENCE_THRESHOLD": str(
                    config["drift"]["level2_confidence_threshold"]
                ),
                "CORRECTION_RATE_MULTIPLIER": str(
                    config["drift"]["correction_rate_multiplier"]
                ),
            }
        ).run(
            code=str(STEPS_DIR / "drift_detection.py"),
            inputs=[
                code_input,
                # Reads pred_probs.npy persisted by KFoldCleanlabQuality so the
                # drift step can measure real mean confidence instead of
                # falling back to the configured baseline.
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
        step_args=processor(
            {
                "IMAGE_SIZE": str(config["training"]["image_size"]),
                "DONE_LABEL": str(config["evaluation"]["done_label"]),
                "NOT_DONE_LABEL": str(config["evaluation"]["not_done_label"]),
                "MIN_SAMPLE_COUNT": str(config["evaluation"]["min_sample_count"]),
                "MIN_PER_CLASS_COUNT": str(
                    config["evaluation"]["min_per_class_count"]
                ),
            }
        ).run(
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
            # Reject vacuous or dataset-shrinking evaluations. The step itself
            # also exits non-zero in these cases, but we keep the gate conditions
            # as defense-in-depth against a tampered metrics.json.
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=eval_step.name,
                    property_file=eval_metrics,
                    json_path="sample_count",
                ),
                right=int(config["evaluation"]["min_sample_count"]),
            ),
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=eval_step.name,
                    property_file=eval_metrics,
                    json_path="not_done_sample_count",
                ),
                right=int(config["evaluation"]["min_per_class_count"]),
            ),
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=eval_step.name,
                    property_file=eval_metrics,
                    json_path="done_sample_count",
                ),
                right=int(config["evaluation"]["min_per_class_count"]),
            ),
            ConditionLessThanOrEqualTo(
                left=JsonGet(
                    step_name=eval_step.name,
                    property_file=eval_metrics,
                    json_path="missing_images",
                ),
                right=0,
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
            # Reject any report whose detector is not the real CleanLab
            # detector -- fail closed on silent degradation to the local
            # heuristic fallback. ``detector`` is emitted by
            # processing/step3_cleanlab_filter.py alongside
            # ``flagged_fraction``.
            ConditionEquals(
                left=JsonGet(
                    step_name=quality_step.name,
                    property_file=quality_metrics,
                    json_path="detector",
                ),
                right="cleanlab",
            ),
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=quality_step.name,
                    property_file=quality_metrics,
                    json_path="cleaned_count",
                ),
                right=sel["target_dataset_min"],
            ),
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=quality_step.name,
                    property_file=quality_metrics,
                    json_path="minimum_class_count",
                ),
                right=sel["min_images_per_class"],
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
        name=pipeline_name or f"miso-food-classifier-{brand_id}",
        steps=[select_step, dataset_gate],
        sagemaker_session=session,
    )


def _default_pipeline_name(config: dict[str, Any], relaxed: bool) -> str:
    brand_id = config["brands"][0]["id"]
    prefix = SMOKE_PIPELINE_PREFIX if relaxed else PRODUCTION_PIPELINE_PREFIX
    return f"{prefix}-{brand_id}"


def main() -> None:
    parser = argparse.ArgumentParser()
    # --config has no default: a documented command *must* name the config to
    # prevent silently loading the relaxed smoke overlay and overwriting the
    # Terraform-managed production pipeline.
    parser.add_argument("--config", required=True)
    parser.add_argument("--code-prefix", default="code")
    parser.add_argument(
        "--images-uri",
        required=True,
        help=(
            "S3 URI of the raw-image prefix for this brand, e.g. "
            "s3://<raw-archive-bucket>/<brand.id>/. Must be scoped to a "
            "single brand -- do not pass the bucket root (CWE-639)."
        ),
    )
    parser.add_argument(
        "--pipeline-name",
        default=None,
        help=(
            "Explicit SageMaker pipeline name. If omitted, production configs "
            "use 'miso-food-classifier-<brand>' and relaxed/smoke configs use "
            "'smoke-miso-food-classifier-<brand>' so --upsert cannot overwrite "
            "the Terraform-managed production pipeline."
        ),
    )
    parser.add_argument(
        "--i-understand-this-is-a-smoke-config",
        dest="acknowledge_smoke",
        action="store_true",
        help=(
            "Required acknowledgement when --config points at a relaxed overlay "
            "whose gate values are weaker than the production floors."
        ),
    )
    parser.add_argument("--upsert", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    weak_fields = _relaxed_fields(config)
    is_relaxed = bool(weak_fields)

    if is_relaxed and not args.acknowledge_smoke:
        joined = "\n  - ".join(weak_fields)
        parser.error(
            "config has gate values weaker than the production floors:\n  - "
            f"{joined}\n"
            "Re-run with --i-understand-this-is-a-smoke-config to confirm "
            "this is a smoke deployment."
        )

    pipeline_name = args.pipeline_name or _default_pipeline_name(config, is_relaxed)

    # Belt-and-braces: never let --upsert overwrite a Terraform-managed
    # production pipeline, regardless of whether the deploy config looks
    # relaxed. The guard runs UNCONDITIONALLY -- a config that silently
    # weakens an uncovered field (or an operator who thinks they are
    # re-applying production) must still fail closed, because the production
    # pipeline definition is owned by Terraform (infra/main.tf consumes
    # infra/pipeline_definition.<brand>.json) and this script is only meant
    # for the smoke overlay. ``PRODUCTION_PIPELINE_PREFIX`` is a safe
    # over-approximation of the Terraform ``pipeline_name_prefix`` variable
    # (infra/variables.tf); operators who have reassigned that variable must
    # pass ``--pipeline-name`` with a prefix that does not collide.
    if args.upsert:
        brand_id = config["brands"][0]["id"]
        production_name = f"{PRODUCTION_PIPELINE_PREFIX}-{brand_id}"
        collides_with_production = pipeline_name == production_name or (
            pipeline_name.startswith(f"{PRODUCTION_PIPELINE_PREFIX}-")
            and not pipeline_name.startswith(f"{SMOKE_PIPELINE_PREFIX}-")
        )
        if collides_with_production:
            parser.error(
                f"refusing to --upsert pipeline name {pipeline_name!r}: it "
                f"collides with the Terraform-managed production pipeline "
                f"prefix {PRODUCTION_PIPELINE_PREFIX!r}. The production "
                "definition is owned by Terraform (infra/main.tf consumes "
                "infra/pipeline_definition.<brand>.json). Pass --pipeline-name "
                f"with a distinct name (e.g. prefixed {SMOKE_PIPELINE_PREFIX!r})."
            )

    pipeline = build(config, args.code_prefix, args.images_uri, pipeline_name=pipeline_name)
    role = config["aws"]["sagemaker_role_arn"]

    if args.upsert:
        # Final safety net: run the shared SDK-definition validator over the
        # built pipeline before upserting. This mirrors the Terraform export
        # path (``pipeline/export_definition.py``) and catches an SDK builder
        # that silently drops a safety-gate Condition, duplicates a step, or
        # changes the model-registration approval status.
        from pipeline.pipeline import validate_sdk_pipeline_definition  # noqa: WPS433

        validate_sdk_pipeline_definition(pipeline.definition())
        pipeline.upsert(role_arn=role)
        print(f"[deploy] upserted pipeline: {pipeline.name}")
    if args.run:
        execution = pipeline.start()
        print(f"[deploy] started execution: {execution.arn}")


if __name__ == "__main__":
    main()

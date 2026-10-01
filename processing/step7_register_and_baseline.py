"""Register a candidate model and publish its manual-approval request."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any

import yaml

DEFAULT_INPUT_DIR = Path("/opt/ml/processing/input")
DEFAULT_OUTPUT_DIR = Path("/opt/ml/processing/output")
EVENT_SOURCE = "miso.ml.pipeline"
EVENT_DETAIL_TYPE = "ModelApprovalRequested"
PENDING_APPROVAL = "PendingManualApproval"


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def model_group_for_brand(config: dict[str, Any], brand_id: str) -> str:
    for brand in config.get("brands", []):
        if brand.get("id") == brand_id:
            return str(brand["model_package_group"])
    raise ValueError(f"No model package group is configured for brand '{brand_id}'.")


def load_evaluation_report(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as report_file:
        return json.load(report_file)


def build_baseline_statistics(evaluation: dict[str, Any], brand_id: str) -> dict[str, Any]:
    """Make an immutable, model-scoped baseline for downstream monitoring."""
    metrics = evaluation["metrics"]
    return {
        "schema_version": "1.0",
        "brand_id": brand_id,
        "created_at": datetime.now(UTC).isoformat(),
        "sample_count": metrics["sample_count"],
        "label_distribution": metrics["label_distribution"],
        "confidence": metrics.get("confidence_summary", {}),
        "quality_metrics": {
            "accuracy": metrics["accuracy"],
            "done_when_not_done_fp_rate": metrics["done_when_not_done_fp_rate"],
        },
    }


def approval_event(
    *,
    brand_id: str,
    model_package_group_name: str,
    model_package_arn: str,
    baseline_statistics_s3_uri: str,
    evaluation_report_s3_uri: str,
    quality_gate: str,
) -> dict[str, str]:
    """Build the EventBridge entry consumed by the ML lead approval workflow."""
    detail = {
        "schema_version": "1.0",
        "brand_id": brand_id,
        "model_package_group_name": model_package_group_name,
        "model_package_arn": model_package_arn,
        "model_approval_status": PENDING_APPROVAL,
        "quality_gate": quality_gate,
        "baseline_statistics_s3_uri": baseline_statistics_s3_uri,
        "evaluation_report_s3_uri": evaluation_report_s3_uri,
        "required_action": "manual_model_approval",
    }
    return {
        "Source": EVENT_SOURCE,
        "DetailType": EVENT_DETAIL_TYPE,
        "Detail": json.dumps(detail, sort_keys=True),
    }


def _s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key.lstrip('/')}"


def register_and_baseline(
    *,
    config: dict[str, Any],
    brand_id: str,
    model_data_s3_uri: str,
    image_uri: str,
    evaluation_report_s3_uri: str,
    evaluation: dict[str, Any],
    sagemaker_client: Any,
    s3_client: Any,
    events_client: Any,
    event_bus_name: str = "default",
) -> dict[str, Any]:
    """Persist baseline stats, create a PendingManualApproval package, and emit an event."""
    group_name = model_group_for_brand(config, brand_id)
    artifact_bucket = str(config["s3"]["model_artifacts_bucket"])
    baseline_key = f"baselines/{brand_id}/{Path(model_data_s3_uri).stem}/baseline_statistics.json"
    baseline = build_baseline_statistics(evaluation, brand_id)
    s3_client.put_object(
        Bucket=artifact_bucket,
        Key=baseline_key,
        Body=json.dumps(baseline, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    baseline_s3_uri = _s3_uri(artifact_bucket, baseline_key)

    quality_gate = str(evaluation.get("quality_gate", "failed"))
    response = sagemaker_client.create_model_package(
        ModelPackageGroupName=group_name,
        ModelApprovalStatus=PENDING_APPROVAL,
        InferenceSpecification={
            "Containers": [{"Image": image_uri, "ModelDataUrl": model_data_s3_uri}],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        ModelMetrics={
            "ModelQuality": {
                "Statistics": {
                    "ContentType": "application/json",
                    "S3Uri": evaluation_report_s3_uri,
                }
            }
        },
        CustomerMetadataProperties={
            "brand_id": brand_id,
            "quality_gate": quality_gate,
            "manual_approval_required": "true",
        },
        Description=(
            f"{brand_id} candidate; quality gate={quality_gate}; requires manual approval before deployment."
        ),
    )
    model_package_arn = response["ModelPackageArn"]
    event = approval_event(
        brand_id=brand_id,
        model_package_group_name=group_name,
        model_package_arn=model_package_arn,
        baseline_statistics_s3_uri=baseline_s3_uri,
        evaluation_report_s3_uri=evaluation_report_s3_uri,
        quality_gate=quality_gate,
    )
    events_client.put_events(Entries=[{**event, "EventBusName": event_bus_name}])
    return {
        "model_package_arn": model_package_arn,
        "model_package_group_name": group_name,
        "model_approval_status": PENDING_APPROVAL,
        "baseline_statistics_s3_uri": baseline_s3_uri,
        "approval_event": event,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a model candidate and request manual approval.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--brand-id", required=True)
    parser.add_argument("--model-data-s3-uri", required=True)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--evaluation-report-s3-uri", required=True)
    parser.add_argument("--evaluation-report", default=str(DEFAULT_INPUT_DIR / "evaluation.json"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--event-bus-name", default="default")
    parser.add_argument("--dry-run", action="store_true", help="Write a local registration plan without AWS calls.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    evaluation = load_evaluation_report(args.evaluation_report)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dry_run = args.dry_run or config.get("local_mode", {}).get("mock_aws", False)

    if dry_run:
        group_name = model_group_for_brand(config, args.brand_id)
        baseline = build_baseline_statistics(evaluation, args.brand_id)
        (output_dir / "baseline_statistics.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        model_package_arn = f"local://sagemaker/model-package/{group_name}/candidate"
        result = {
            "dry_run": True,
            "model_package_arn": model_package_arn,
            "model_package_group_name": group_name,
            "model_approval_status": PENDING_APPROVAL,
            "baseline_statistics_s3_uri": None,
            "approval_event": approval_event(
                brand_id=args.brand_id,
                model_package_group_name=group_name,
                model_package_arn=model_package_arn,
                baseline_statistics_s3_uri="local://baseline_statistics.json",
                evaluation_report_s3_uri=args.evaluation_report_s3_uri,
                quality_gate=str(evaluation.get("quality_gate", "failed")),
            ),
        }
    else:
        import boto3

        result = register_and_baseline(
            config=config,
            brand_id=args.brand_id,
            model_data_s3_uri=args.model_data_s3_uri,
            image_uri=args.image_uri,
            evaluation_report_s3_uri=args.evaluation_report_s3_uri,
            evaluation=evaluation,
            sagemaker_client=boto3.client("sagemaker", region_name=config["aws"]["region"]),
            s3_client=boto3.client("s3", region_name=config["aws"]["region"]),
            events_client=boto3.client("events", region_name=config["aws"]["region"]),
            event_bus_name=args.event_bus_name,
        )
    (output_dir / "registration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(main(parse_args()), indent=2))

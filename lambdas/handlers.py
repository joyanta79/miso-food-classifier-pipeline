"""Event-driven Lambda handlers for brand training workflows.

All AWS clients are injectable so unit tests and local mode never make network calls.
"""

from __future__ import annotations

import json
import os
from fnmatch import fnmatch
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the shared project configuration without mutating it."""
    path = Path(config_path or os.environ.get("MISO_CONFIG_PATH", _CONFIG_PATH))
    with path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _client(service_name: str, injected_client: Any | None) -> Any:
    if injected_client is not None:
        return injected_client
    import boto3  # Imported lazily so unit tests and local mode never require boto3.

    return boto3.client(service_name)


def _brand_for_key(key: str, config: dict[str, Any]) -> dict[str, Any] | None:
    filename = Path(key).name
    for brand in config["brands"]:
        if fnmatch(filename, brand["naming_pattern"]):
            return brand
    return None


def _s3_objects(event: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize native S3 and EventBridge S3 event shapes."""
    if "Records" in event:
        objects = []
        for record in event["Records"]:
            s3_record = record.get("s3", {})
            bucket = s3_record.get("bucket", {}).get("name")
            key = s3_record.get("object", {}).get("key")
            if bucket and key:
                objects.append({"bucket": bucket, "key": unquote_plus(key)})
        return objects

    detail = event.get("detail", {})
    bucket = detail.get("bucket", {}).get("name")
    key = detail.get("object", {}).get("key")
    return [{"bucket": bucket, "key": unquote_plus(key)}] if bucket and key else []


def s3_trigger_handler(
    event: dict[str, Any],
    _context: Any,
    *,
    eventbridge_client: Any | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish a brand-scoped training request for each recognized S3 upload."""
    resolved_config = config or load_config()
    entries = []
    for uploaded_object in _s3_objects(event):
        brand = _brand_for_key(uploaded_object["key"], resolved_config)
        if brand is None:
            continue
        detail = {
            "brand_id": brand["id"],
            "bucket": uploaded_object["bucket"],
            "key": uploaded_object["key"],
            "trigger": "s3",
        }
        entries.append(
            {
                "Source": "miso.ml.training",
                "DetailType": "BrandTrainingRequested",
                "Detail": json.dumps(detail, sort_keys=True),
                "EventBusName": os.environ.get("MISO_EVENT_BUS_NAME", "default"),
            }
        )

    if not entries:
        return {"accepted": 0, "failed": 0, "message": "No configured brand matched the upload."}

    response = _client("events", eventbridge_client).put_events(Entries=entries)
    return {
        "accepted": len(entries) - response.get("FailedEntryCount", 0),
        "failed": response.get("FailedEntryCount", 0),
        "event_ids": [entry.get("EventId") for entry in response.get("Entries", [])],
    }


def _pipeline_parameters(
    brand_id: str, trigger: str, config: dict[str, Any]
) -> list[dict[str, str]]:
    return [
        {"Name": "BrandId", "Value": brand_id},
        {"Name": "TriggerSource", "Value": trigger},
        {"Name": "TrainingBucket", "Value": config["s3"]["training_bucket"]},
    ]


def eventbridge_trigger_handler(
    event: dict[str, Any],
    _context: Any,
    *,
    sagemaker_client: Any | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Start the per-brand SageMaker pipeline from an EventBridge request."""
    resolved_config = config or load_config()
    detail = event.get("detail", event)
    if isinstance(detail, str):
        detail = json.loads(detail)

    brand_id = detail.get("brand_id")
    if brand_id not in {brand["id"] for brand in resolved_config["brands"]}:
        raise ValueError(f"Unknown brand_id: {brand_id!r}")

    pipeline_name = detail.get(
        "pipeline_name", os.environ.get("MISO_PIPELINE_NAME", "MisoFoodClassifierPipeline")
    )
    response = _client("sagemaker", sagemaker_client).start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineParameters=_pipeline_parameters(
            brand_id, detail.get("trigger", "eventbridge"), resolved_config
        ),
    )
    return {
        "brand_id": brand_id,
        "pipeline_execution_arn": response["PipelineExecutionArn"],
        "trigger": detail.get("trigger", "eventbridge"),
    }


def manual_trigger_handler(
    event: dict[str, Any],
    context: Any,
    *,
    sagemaker_client: Any | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Start a brand pipeline from an authenticated manual invocation."""
    body = event.get("body", event)
    if isinstance(body, str):
        body = json.loads(body)
    brand_id = body.get("brand_id")
    if not brand_id:
        raise ValueError("brand_id is required for a manual training request")

    return eventbridge_trigger_handler(
        {"detail": {**body, "brand_id": brand_id, "trigger": "manual"}},
        context,
        sagemaker_client=sagemaker_client,
        config=config,
    )

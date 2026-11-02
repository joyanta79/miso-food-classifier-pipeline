"""Event-driven Lambda handlers for brand training workflows.

All AWS clients are injectable so unit tests and local mode never make network calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote_plus

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

# EventBridge PutEvents accepts at most 10 entries per call.
_EVENTBRIDGE_MAX_BATCH = 10

# Default window used to derive deterministic idempotency tokens / display
# names when the caller does not pin one. One execution per brand per window
# is the throttle unit enforced against SageMaker.
_DEFAULT_IDEMPOTENCY_WINDOW_SECONDS = 300

# SageMaker pipeline execution states that count as in-flight and should
# therefore suppress a new start for the same brand.
_IN_FLIGHT_PIPELINE_STATES = frozenset({"Executing", "Stopping"})


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


def _chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def s3_trigger_handler(
    event: dict[str, Any],
    _context: Any,
    *,
    eventbridge_client: Any | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish a brand-scoped training request for each recognized S3 upload.

    Multiple uploads for the same brand in a single invocation are coalesced
    into a single ``BrandTrainingRequested`` event so a burst of writes cannot
    amplify into N pipeline executions. ``put_events`` requests are chunked to
    the EventBridge batch limit of 10 so large notifications degrade gracefully
    instead of failing validation.
    """
    resolved_config = config or load_config()
    event_bus = os.environ.get("MISO_EVENT_BUS_NAME", "default")

    # Coalesce per brand: first-wins on representative bucket/key, and record
    # how many uploads collapsed into each entry so downstream observability
    # does not lose the fan-in information.
    coalesced: dict[str, dict[str, Any]] = {}
    for uploaded_object in _s3_objects(event):
        brand = _brand_for_key(uploaded_object["key"], resolved_config)
        if brand is None:
            continue
        brand_id = brand["id"]
        existing = coalesced.get(brand_id)
        if existing is None:
            coalesced[brand_id] = {
                "brand_id": brand_id,
                "bucket": uploaded_object["bucket"],
                "key": uploaded_object["key"],
                "trigger": "s3",
            }
        else:
            existing.setdefault("coalesced_count", 1)
            existing["coalesced_count"] += 1

    entries = [
        {
            "Source": "miso.ml.training",
            "DetailType": "BrandTrainingRequested",
            "Detail": json.dumps(detail, sort_keys=True),
            "EventBusName": event_bus,
        }
        for detail in coalesced.values()
    ]

    if not entries:
        return {"accepted": 0, "failed": 0, "message": "No configured brand matched the upload."}

    events_client = _client("events", eventbridge_client)
    accepted = 0
    failed = 0
    event_ids: list[str | None] = []
    for batch in _chunks(entries, _EVENTBRIDGE_MAX_BATCH):
        response = events_client.put_events(Entries=batch)
        batch_failed = response.get("FailedEntryCount", 0)
        failed += batch_failed
        accepted += len(batch) - batch_failed
        event_ids.extend(entry.get("EventId") for entry in response.get("Entries", []))

    return {
        "accepted": accepted,
        "failed": failed,
        "event_ids": event_ids,
    }


def _pipeline_parameters(
    brand_id: str, trigger: str, config: dict[str, Any]
) -> list[dict[str, str]]:
    return [
        {"Name": "BrandId", "Value": brand_id},
        {"Name": "TriggerSource", "Value": trigger},
        {"Name": "TrainingBucket", "Value": config["s3"]["training_bucket"]},
    ]


def _idempotency_window_seconds() -> int:
    raw = os.environ.get("MISO_IDEMPOTENCY_WINDOW_SECONDS")
    if not raw:
        return _DEFAULT_IDEMPOTENCY_WINDOW_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_IDEMPOTENCY_WINDOW_SECONDS
    return value if value > 0 else _DEFAULT_IDEMPOTENCY_WINDOW_SECONDS


def _pipeline_execution_display_name(
    brand_id: str, pipeline_name: str, now: float | None = None
) -> str:
    """Deterministic display name scoped to brand + current throttle window.

    SageMaker enforces that ``PipelineExecutionDisplayName`` is unique per
    pipeline, so two starts for the same brand that fall inside the same
    window collide and the second call fails fast (idempotent retry) instead
    of booting a second GPU training job.
    """
    window = _idempotency_window_seconds()
    bucket = int((now if now is not None else time.time()) // window)
    digest = hashlib.sha1(
        f"{pipeline_name}:{brand_id}:{bucket}".encode("utf-8")
    ).hexdigest()[:12]
    safe_brand = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in brand_id)[:40]
    return f"brand-{safe_brand}-{bucket}-{digest}"


def _has_in_flight_execution(sagemaker: Any, pipeline_name: str, brand_id: str) -> bool:
    """Return True if a pipeline execution for ``brand_id`` is already running.

    Falls back to ``False`` -- letting ``start_pipeline_execution`` be the sole
    gate -- when the injected client does not implement
    ``list_pipeline_executions`` (e.g. the lightweight test doubles used in
    unit tests and local mode).
    """
    list_fn = getattr(sagemaker, "list_pipeline_executions", None)
    if list_fn is None:
        return False
    try:
        response = list_fn(
            PipelineName=pipeline_name,
            SortBy="CreationTime",
            SortOrder="Descending",
        )
    except Exception:
        # Fail closed on the in-flight check so a throttled or permission-
        # denied List call cannot escalate into unbounded Starts.
        return True
    for summary in response.get("PipelineExecutionSummaries", []) or []:
        if summary.get("PipelineExecutionStatus") not in _IN_FLIGHT_PIPELINE_STATES:
            continue
        display_name = summary.get("PipelineExecutionDisplayName", "")
        if display_name.startswith(f"brand-{brand_id}-") or f"-{brand_id}-" in display_name:
            return True
    return False


def eventbridge_trigger_handler(
    event: dict[str, Any],
    _context: Any,
    *,
    sagemaker_client: Any | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Start the per-brand SageMaker pipeline from an EventBridge request.

    Guards against upload storms by (1) short-circuiting when an execution for
    the same brand is already running and (2) pinning a deterministic
    ``PipelineExecutionDisplayName`` so retries inside the throttle window
    collide instead of launching new GPU training jobs.
    """
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
    trigger = detail.get("trigger", "eventbridge")

    sagemaker = _client("sagemaker", sagemaker_client)
    if _has_in_flight_execution(sagemaker, pipeline_name, brand_id):
        return {
            "brand_id": brand_id,
            "pipeline_execution_arn": None,
            "trigger": trigger,
            "skipped": True,
            "reason": "in_flight_execution",
        }

    display_name = _pipeline_execution_display_name(brand_id, pipeline_name)
    try:
        response = sagemaker.start_pipeline_execution(
            PipelineName=pipeline_name,
            PipelineParameters=_pipeline_parameters(brand_id, trigger, resolved_config),
            PipelineExecutionDisplayName=display_name,
        )
    except TypeError:
        # Test doubles / older clients that do not accept the idempotency
        # kwarg still work; the display name is then best-effort only.
        response = sagemaker.start_pipeline_execution(
            PipelineName=pipeline_name,
            PipelineParameters=_pipeline_parameters(brand_id, trigger, resolved_config),
        )
    return {
        "brand_id": brand_id,
        "pipeline_execution_arn": response["PipelineExecutionArn"],
        "trigger": trigger,
        "pipeline_execution_display_name": display_name,
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

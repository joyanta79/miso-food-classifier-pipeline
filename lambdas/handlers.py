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

# Default prefix for the per-brand SageMaker pipelines deployed by infra/main.tf
# (``aws_sagemaker_pipeline.brand`` names each pipeline
# ``${var.pipeline_name_prefix}-${brand_id}``, with ``miso-food-classifier`` as
# the default prefix).
_DEFAULT_PIPELINE_NAME_PREFIX = "miso-food-classifier"

# Fields a caller may supply to ``manual_trigger_handler``. Anything else is
# rejected so attacker-controlled keys (notably ``pipeline_name``) cannot flow
# through the mass-assigned forwarded detail into ``start_pipeline_execution``.
_MANUAL_TRIGGER_ALLOWED_FIELDS = frozenset({"brand_id"})


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


def _pipeline_name_for_brand(brand_id: str) -> str:
    """Return the pipeline name owned by ``brand_id``.

    The target is always derived server-side from the already-validated
    ``brand_id`` and the deployment-controlled ``MISO_PIPELINE_NAME_PREFIX``
    environment variable (default ``miso-food-classifier``) so a caller cannot
    choose which tenant's pipeline runs.
    """
    prefix = os.environ.get("MISO_PIPELINE_NAME_PREFIX", _DEFAULT_PIPELINE_NAME_PREFIX)
    return f"{prefix}-{brand_id}"


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

    # Pipeline target is derived from the validated ``brand_id`` only. Any
    # ``pipeline_name`` supplied by the caller is ignored to prevent a confused
    # deputy against ``sagemaker:StartPipelineExecution``.
    pipeline_name = _pipeline_name_for_brand(brand_id)
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


def _split_claim_values(value: Any) -> list[str]:
    """Normalize an authorizer claim into a list of individual values.

    Authorizer context values arrive as strings from API Gateway Lambda
    authorizers (even when they are lists in the authorizer source), while
    JWT/Cognito claims may already be lists. Accept both and split on common
    separators so callers can pass ``"brand-a,brand-b"``, ``"brand-a brand-b"``,
    or a JSON array transparently.
    """
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        values: list[str] = []
        for item in value:
            values.extend(_split_claim_values(item))
        return values
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value:
        return []
    # Support JSON-encoded arrays smuggled through a string claim.
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = None
        if isinstance(parsed, list):
            return _split_claim_values(parsed)
    return [
        part.strip()
        for part in value.replace(";", ",").replace(" ", ",").split(",")
        if part.strip()
    ]


def _authorized_brands(event: dict[str, Any]) -> set[str]:
    """Return the set of brand ids the authenticated caller is allowed to drive.

    The caller identity must come from an upstream API Gateway authorizer (JWT,
    Cognito, or Lambda authorizer) or REQUEST authorizer context propagated in
    ``event['requestContext']['authorizer']``. If no authorizer context is
    present the request is treated as unauthenticated and rejected.

    The allowed brand set is sourced from, in priority order:
      * ``allowed_brands`` claim/context value (comma / space / JSON list).
      * ``brand_id`` claim/context value (single brand binding).
      * ``cognito:groups`` claim entries that start with ``brand:`` (e.g.
        ``brand:white-castle``).
    """
    request_context = event.get("requestContext") if isinstance(event, dict) else None
    if not isinstance(request_context, dict):
        raise PermissionError(
            "Unauthenticated manual training request: missing requestContext"
        )
    authorizer = request_context.get("authorizer")
    if not isinstance(authorizer, dict) or not authorizer:
        raise PermissionError(
            "Unauthenticated manual training request: missing authorizer context"
        )

    # JWT / Cognito authorizers nest verified claims under ``claims``; Lambda
    # (REQUEST/TOKEN) authorizers place context fields directly on authorizer.
    claims = authorizer.get("claims") if isinstance(authorizer.get("claims"), dict) else {}
    jwt_block = authorizer.get("jwt") if isinstance(authorizer.get("jwt"), dict) else {}
    jwt_claims = jwt_block.get("claims") if isinstance(jwt_block.get("claims"), dict) else {}

    def _lookup(key: str) -> Any:
        for source in (claims, jwt_claims, authorizer):
            if isinstance(source, dict) and key in source and source[key] not in (None, ""):
                return source[key]
        return None

    allowed: set[str] = set()
    allowed.update(_split_claim_values(_lookup("allowed_brands")))
    allowed.update(_split_claim_values(_lookup("brand_id")))
    for group in _split_claim_values(_lookup("cognito:groups")):
        if group.startswith("brand:") and len(group) > len("brand:"):
            allowed.add(group.split(":", 1)[1])
    return allowed


def manual_trigger_handler(
    event: dict[str, Any],
    context: Any,
    *,
    sagemaker_client: Any | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Start a brand pipeline from an authenticated manual invocation.

    The caller identity must be established upstream (API Gateway JWT/Cognito
    authorizer, Lambda authorizer, or equivalent) and surfaced through
    ``event['requestContext']['authorizer']``. The handler authorizes the
    requested ``brand_id`` against the brand set bound to that principal and
    rejects any brand outside it, regardless of whether the brand is listed in
    ``config.yaml``.
    """
    allowed_brands = _authorized_brands(event)
    if not allowed_brands:
        raise PermissionError(
            "Authenticated principal is not authorized for any brand"
        )

    body = event.get("body", event)
    if isinstance(body, str):
        body = json.loads(body)
    if not isinstance(body, dict):
        raise ValueError("manual training request body must be a JSON object")

    # Reject unknown keys instead of splatting ``body`` into the forwarded
    # detail. This prevents mass-assignment of control-plane fields such as
    # ``pipeline_name`` through to ``start_pipeline_execution``.
    unexpected = set(body) - _MANUAL_TRIGGER_ALLOWED_FIELDS
    if unexpected:
        raise ValueError(
            "Unexpected field(s) in manual training request: "
            + ", ".join(sorted(unexpected))
        )

    brand_id = body.get("brand_id")
    if not brand_id:
        raise ValueError("brand_id is required for a manual training request")
    if brand_id not in allowed_brands:
        # Do not leak which brands the caller is bound to.
        raise PermissionError(
            f"Caller is not authorized to trigger training for brand_id {brand_id!r}"
        )

    return eventbridge_trigger_handler(
        {"detail": {"brand_id": brand_id, "trigger": "manual"}},
        context,
        sagemaker_client=sagemaker_client,
        config=config,
    )

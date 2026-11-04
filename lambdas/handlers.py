"""Event-driven Lambda handlers for brand training workflows.

All AWS clients are injectable so unit tests and local mode never make network calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Iterable
from fnmatch import fnmatch
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus

import yaml

# ---------------------------------------------------------------------------
# Audit logging
#
# Every security-relevant decision in this module (authentication failure,
# authorization denial, brand acceptance/rejection, successful
# ``start_pipeline_execution`` and idempotency/in-flight suppression) emits a
# single structured JSON log record through this logger. The records are
# consumed by a CloudWatch metric filter (``infra/observability.tf``) that
# alarms on repeated authorization denials per principal and by CloudTrail +
# S3 server-access logging so uploads driving ``s3_trigger_handler`` are
# attributable.
#
# Only non-sensitive claim fields are logged (principal id and the resolved
# brand entitlement set); raw tokens, email addresses and other PII are never
# recorded. Caller-controlled strings are JSON-encoded via ``json.dumps`` and
# therefore cannot inject forged line breaks or control characters into the
# log stream.
# ---------------------------------------------------------------------------

_audit_logger = logging.getLogger(__name__)
if not _audit_logger.handlers and not logging.getLogger().handlers:
    # Ensure records are emitted even if Lambda has not configured a root
    # handler (notably in unit tests and local invocations). AWS Lambda itself
    # installs a root handler at startup, so this is a no-op in production.
    _audit_logger.addHandler(logging.StreamHandler())
_audit_logger.setLevel(logging.INFO)

# Event name used for the CloudWatch metric filter.
_AUDIT_EVENT_PREFIX = "miso.audit"


def _audit(event_name: str, outcome: str, **fields: Any) -> None:
    """Emit a single structured audit record.

    ``event_name`` and ``outcome`` are controlled by this module; ``fields``
    may contain caller-supplied values and are JSON-encoded so that any
    embedded newlines, quotes, or control characters cannot split or forge a
    second log record.
    """
    record = {"event": f"{_AUDIT_EVENT_PREFIX}.{event_name}", "outcome": outcome}
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, (set, frozenset)):
            value = sorted(value)
        record[key] = value
    # ``default=str`` keeps the call total: any object that is not JSON-
    # serialisable is coerced to its ``str`` form rather than raising inside a
    # security log path.
    _audit_logger.info(json.dumps(record, sort_keys=True, default=str))


def _principal_from_authorizer(event: Any) -> dict[str, Any]:
    """Extract non-sensitive principal identifiers from an API Gateway event.

    Returns a dict with at most ``principal_id``, ``request_id`` and
    ``source_ip`` populated. No tokens, emails or other sensitive claim
    fields are read.
    """
    info: dict[str, Any] = {}
    if not isinstance(event, dict):
        return info
    request_context = event.get("requestContext")
    if not isinstance(request_context, dict):
        return info
    request_id = request_context.get("requestId")
    if isinstance(request_id, str):
        info["request_id"] = request_id
    identity = request_context.get("identity")
    if isinstance(identity, dict):
        source_ip = identity.get("sourceIp")
        if isinstance(source_ip, str):
            info["source_ip"] = source_ip
    authorizer = request_context.get("authorizer")
    if isinstance(authorizer, dict):
        claims = authorizer.get("claims") if isinstance(authorizer.get("claims"), dict) else {}
        jwt_block = authorizer.get("jwt") if isinstance(authorizer.get("jwt"), dict) else {}
        jwt_claims = jwt_block.get("claims") if isinstance(jwt_block.get("claims"), dict) else {}
        for source in (claims, jwt_claims, authorizer):
            if not isinstance(source, dict):
                continue
            principal = source.get("sub") or source.get("principalId")
            if isinstance(principal, str) and principal:
                info["principal_id"] = principal
                break
    return info

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
            # Rejected brand attribution: the uploaded object does not match
            # any configured naming pattern. Record it so the data-plane
            # upload is attributable even when no pipeline is started.
            _audit(
                "s3_trigger.brand_match",
                "rejected",
                bucket=uploaded_object.get("bucket"),
                key=uploaded_object.get("key"),
                reason="no_matching_brand",
            )
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
            _audit(
                "s3_trigger.brand_match",
                "accepted",
                brand_id=brand_id,
                bucket=uploaded_object["bucket"],
                key=uploaded_object["key"],
            )
        else:
            existing.setdefault("coalesced_count", 1)
            existing["coalesced_count"] += 1
            _audit(
                "s3_trigger.brand_match",
                "coalesced",
                brand_id=brand_id,
                bucket=uploaded_object.get("bucket"),
                key=uploaded_object.get("key"),
            )

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
    # NOTE: Do not submit SageMakerRoleArn/*Bucket here. Those are deployment
    # facts baked into the pipeline definition as literals; passing them as
    # ``PipelineParameters`` would both be rejected by SageMaker (unknown
    # parameter) and reopen the code-channel / role-override hijack path
    # (CWE-494). The ``config`` argument is retained for forward-compatible
    # callers but is intentionally unused.
    del config
    return [
        {"Name": "BrandId", "Value": brand_id},
        {"Name": "TriggerSource", "Value": trigger},
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
        f"{pipeline_name}:{brand_id}:{bucket}".encode()
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
    _principal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Start the per-brand SageMaker pipeline from an EventBridge request.

    Guards against upload storms by (1) short-circuiting when an execution for
    the same brand is already running and (2) pinning a deterministic
    ``PipelineExecutionDisplayName`` so retries inside the throttle window
    collide instead of launching new GPU training jobs.

    ``_principal`` is an internal, non-public parameter used by
    ``manual_trigger_handler`` to attribute the start to the authenticated
    caller in the emitted audit record. External EventBridge invocations
    omit it and are audited as ``trigger='eventbridge'``.
    """
    resolved_config = config or load_config()
    detail = event.get("detail", event)
    if isinstance(detail, str):
        detail = json.loads(detail)

    brand_id = detail.get("brand_id")
    known_brands = {brand["id"] for brand in resolved_config["brands"]}
    if brand_id not in known_brands:
        _audit(
            "eventbridge_trigger.brand",
            "rejected",
            brand_id=brand_id,
            reason="unknown_brand",
            principal=_principal,
        )
        raise ValueError(f"Unknown brand_id: {brand_id!r}")

    # Pipeline target is derived from the validated ``brand_id`` only. Any
    # ``pipeline_name`` supplied by the caller is ignored to prevent a confused
    # deputy against ``sagemaker:StartPipelineExecution``.
    pipeline_name = _pipeline_name_for_brand(brand_id)
    trigger = detail.get("trigger", "eventbridge")
    _audit(
        "eventbridge_trigger.brand",
        "accepted",
        brand_id=brand_id,
        trigger=trigger,
        pipeline_name=pipeline_name,
        principal=_principal,
    )

    sagemaker = _client("sagemaker", sagemaker_client)
    if _has_in_flight_execution(sagemaker, pipeline_name, brand_id):
        _audit(
            "start_pipeline_execution",
            "suppressed",
            brand_id=brand_id,
            trigger=trigger,
            pipeline_name=pipeline_name,
            reason="in_flight_execution",
            principal=_principal,
        )
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
    execution_arn = response["PipelineExecutionArn"]
    _audit(
        "start_pipeline_execution",
        "started",
        brand_id=brand_id,
        trigger=trigger,
        pipeline_name=pipeline_name,
        pipeline_execution_arn=execution_arn,
        pipeline_execution_display_name=display_name,
        principal=_principal,
    )
    return {
        "brand_id": brand_id,
        "pipeline_execution_arn": execution_arn,
        "trigger": trigger,
        "pipeline_execution_display_name": display_name,
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
    principal = _principal_from_authorizer(event)
    request_context = event.get("requestContext") if isinstance(event, dict) else None
    if not isinstance(request_context, dict):
        _audit(
            "authentication",
            "denied",
            reason="missing_request_context",
            request_id=principal.get("request_id"),
            source_ip=principal.get("source_ip"),
            principal_id=principal.get("principal_id"),
        )
        raise PermissionError(
            "Unauthenticated manual training request: missing requestContext"
        )
    authorizer = request_context.get("authorizer")
    if not isinstance(authorizer, dict) or not authorizer:
        _audit(
            "authentication",
            "denied",
            reason="missing_authorizer",
            request_id=principal.get("request_id"),
            source_ip=principal.get("source_ip"),
            principal_id=principal.get("principal_id"),
        )
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
    principal = _principal_from_authorizer(event)
    if not allowed_brands:
        _audit(
            "authorization",
            "denied",
            reason="no_authorized_brands",
            principal_id=principal.get("principal_id"),
            request_id=principal.get("request_id"),
            source_ip=principal.get("source_ip"),
        )
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
        _audit(
            "authorization",
            "denied",
            reason="brand_not_in_principal_binding",
            principal_id=principal.get("principal_id"),
            request_id=principal.get("request_id"),
            source_ip=principal.get("source_ip"),
            requested_brand_id=brand_id,
            allowed_brands=sorted(allowed_brands),
        )
        # Do not leak which brands the caller is bound to.
        raise PermissionError(
            f"Caller is not authorized to trigger training for brand_id {brand_id!r}"
        )

    return eventbridge_trigger_handler(
        {"detail": {"brand_id": brand_id, "trigger": "manual"}},
        context,
        sagemaker_client=sagemaker_client,
        config=config,
        _principal=principal or None,
    )

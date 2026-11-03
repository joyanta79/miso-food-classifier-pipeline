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

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

# Module logger for structured audit records. We emit JSON payloads at INFO
# (allow / skipped) and WARNING (deny) so CloudWatch Logs Insights filters and
# subscription-based alerting can key off ``decision`` and ``event`` without
# parsing free text. The raw authorizer token is never logged -- only the
# resolved principal identifier and the claim source that produced it.
_AUDIT_LOGGER = logging.getLogger("miso.lambdas.audit")
if not _AUDIT_LOGGER.handlers and not logging.getLogger().handlers:
    # Ensure audit records surface even when the host runtime has not
    # configured a root handler (unit tests, local mode). On AWS Lambda the
    # runtime installs a root handler, so this branch is a no-op there.
    logging.basicConfig(level=logging.INFO)
_AUDIT_LOGGER.setLevel(logging.INFO)


def _emit_audit(level: int, payload: dict[str, Any]) -> None:
    """Emit a single-line JSON audit record.

    ``default=str`` guarantees we never raise from a logging call just because
    a claim value is not JSON-serialisable. The ``event`` key is always
    ``manual_trigger`` so operators can filter on it.
    """
    try:
        message = json.dumps(payload, sort_keys=True, default=str)
    except Exception:  # pragma: no cover - defensive
        message = json.dumps({"event": payload.get("event"), "decision": payload.get("decision"), "serialization_error": True})
    _AUDIT_LOGGER.log(level, message)

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
    brand_id: str,
    trigger: str,
    config: dict[str, Any],
    *,
    requester_principal: str | None = None,
) -> list[dict[str, str]]:
    # NOTE: Do not submit SageMakerRoleArn/*Bucket here. Those are deployment
    # facts baked into the pipeline definition as literals; passing them as
    # ``PipelineParameters`` would both be rejected by SageMaker (unknown
    # parameter) and reopen the code-channel / role-override hijack path
    # (CWE-494). The ``config`` argument is retained for forward-compatible
    # callers but is intentionally unused.
    del config
    parameters: list[dict[str, str]] = [
        {"Name": "BrandId", "Value": brand_id},
        {"Name": "TriggerSource", "Value": trigger},
    ]
    if requester_principal:
        # Attribute the execution to the caller so the SageMaker record ties
        # the privileged start to the authenticated principal, not just the
        # brand. Values flowing in from authorizer claims are coerced to a
        # string so SageMaker's parameter validation never rejects the call.
        parameters.append(
            {"Name": "RequesterPrincipal", "Value": str(requester_principal)}
        )
    return parameters


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

    # Pipeline target is derived from the validated ``brand_id`` only. Any
    # ``pipeline_name`` supplied by the caller is ignored to prevent a confused
    # deputy against ``sagemaker:StartPipelineExecution``.
    pipeline_name = _pipeline_name_for_brand(brand_id)
    trigger = detail.get("trigger", "eventbridge")
    # ``requester_principal`` is injected by ``manual_trigger_handler`` so
    # the SageMaker execution record (via ``PipelineParameters``) is
    # attributable to the authenticated caller. Non-manual flows (S3 or
    # EventBridge scheduled rules) have no end-user principal and leave this
    # unset, so the parameter list is unchanged for those callers.
    requester_principal = detail.get("requester_principal")

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
            PipelineParameters=_pipeline_parameters(
                brand_id,
                trigger,
                resolved_config,
                requester_principal=requester_principal,
            ),
            PipelineExecutionDisplayName=display_name,
        )
    except TypeError:
        # Test doubles / older clients that do not accept the idempotency
        # kwarg still work; the display name is then best-effort only.
        response = sagemaker.start_pipeline_execution(
            PipelineName=pipeline_name,
            PipelineParameters=_pipeline_parameters(
                brand_id,
                trigger,
                resolved_config,
                requester_principal=requester_principal,
            ),
        )
    return {
        "brand_id": brand_id,
        "pipeline_execution_arn": response["PipelineExecutionArn"],
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


def _request_metadata(event: dict[str, Any]) -> dict[str, Any]:
    """Pull non-sensitive request metadata for audit records.

    Returns ``request_id`` and ``source_ip`` from ``requestContext`` when
    present. Never includes the raw ``Authorization`` header or any token.
    """
    if not isinstance(event, dict):
        return {"request_id": None, "source_ip": None}
    request_context = event.get("requestContext") if isinstance(event.get("requestContext"), dict) else {}
    identity = request_context.get("identity") if isinstance(request_context.get("identity"), dict) else {}
    http = request_context.get("http") if isinstance(request_context.get("http"), dict) else {}
    source_ip = identity.get("sourceIp") or http.get("sourceIp")
    return {
        "request_id": request_context.get("requestId"),
        "source_ip": source_ip,
    }


def _resolve_caller_identity(event: dict[str, Any]) -> dict[str, Any]:
    """Return the authenticated caller's principal, authorized brands, and claim source.

    This is the single source of truth used by ``manual_trigger_handler`` for
    both the authorization decision and the audit log record. The raw token
    is never read or returned -- only verified claim fields.

    Raises ``PermissionError`` (with an audit record emitted first) when the
    caller cannot be authenticated.
    """
    request_meta = _request_metadata(event)
    request_context = event.get("requestContext") if isinstance(event, dict) else None
    if not isinstance(request_context, dict):
        _emit_audit(
            logging.WARNING,
            {
                "event": "manual_trigger",
                "decision": "deny",
                "reason": "missing_request_context",
                "principal_id": None,
                "claim_source": None,
                "brand_id": None,
                "allowed_brands": [],
                "request_id": request_meta["request_id"],
                "source_ip": request_meta["source_ip"],
            },
        )
        raise PermissionError(
            "Unauthenticated manual training request: missing requestContext"
        )
    authorizer = request_context.get("authorizer")
    if not isinstance(authorizer, dict) or not authorizer:
        _emit_audit(
            logging.WARNING,
            {
                "event": "manual_trigger",
                "decision": "deny",
                "reason": "missing_authorizer",
                "principal_id": None,
                "claim_source": None,
                "brand_id": None,
                "allowed_brands": [],
                "request_id": request_meta["request_id"],
                "source_ip": request_meta["source_ip"],
            },
        )
        raise PermissionError(
            "Unauthenticated manual training request: missing authorizer context"
        )

    # JWT / Cognito authorizers nest verified claims under ``claims``; HTTP
    # API JWT authorizers nest them under ``jwt.claims``; Lambda
    # (REQUEST/TOKEN) authorizers place context fields directly on
    # ``authorizer``. Track which source produced the identity so the audit
    # record pins the claim provenance.
    claims = authorizer.get("claims") if isinstance(authorizer.get("claims"), dict) else {}
    jwt_block = authorizer.get("jwt") if isinstance(authorizer.get("jwt"), dict) else {}
    jwt_claims = jwt_block.get("claims") if isinstance(jwt_block.get("claims"), dict) else {}

    _source_labels = (
        (claims, "claims"),
        (jwt_claims, "jwt.claims"),
        (authorizer, "authorizer"),
    )

    def _lookup(key: str) -> tuple[Any, str | None]:
        for source, label in _source_labels:
            if isinstance(source, dict) and key in source and source[key] not in (None, ""):
                return source[key], label
        return None, None

    allowed: set[str] = set()
    sources_used: set[str] = set()

    allowed_value, allowed_source = _lookup("allowed_brands")
    if allowed_source is not None:
        allowed.update(_split_claim_values(allowed_value))
        sources_used.add(allowed_source)

    brand_value, brand_source = _lookup("brand_id")
    if brand_source is not None:
        allowed.update(_split_claim_values(brand_value))
        sources_used.add(brand_source)

    groups_value, groups_source = _lookup("cognito:groups")
    if groups_source is not None:
        for group in _split_claim_values(groups_value):
            if group.startswith("brand:") and len(group) > len("brand:"):
                allowed.add(group.split(":", 1)[1])
        sources_used.add(groups_source)

    principal_value, principal_source = _lookup("sub")
    if principal_value is None:
        principal_value, principal_source = _lookup("principalId")
    if principal_source is not None:
        sources_used.add(principal_source)

    claim_source = ",".join(sorted(sources_used)) if sources_used else None

    return {
        "principal_id": principal_value,
        "claim_source": claim_source,
        "allowed_brands": allowed,
        "request_id": request_meta["request_id"],
        "source_ip": request_meta["source_ip"],
    }


def _authorized_brands(event: dict[str, Any]) -> set[str]:
    """Return the set of brand ids the authenticated caller is allowed to drive.

    Thin backwards-compatible wrapper around :func:`_resolve_caller_identity`.
    Prefer the richer helper when the caller also needs principal / claim
    source information for audit emission.
    """
    return _resolve_caller_identity(event)["allowed_brands"]


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

    Every allow/deny decision is written as a structured JSON audit record
    before this function returns or raises. The audit record carries the
    resolved principal, the authorized brand set, the claim source, the
    ``requestContext.requestId`` and source IP, and -- on allow -- the
    resulting ``PipelineExecutionArn``. The raw authorizer token is never
    logged.
    """
    identity = _resolve_caller_identity(event)
    allowed_brands = identity["allowed_brands"]
    principal_id = identity["principal_id"]
    claim_source = identity["claim_source"]
    request_id = identity["request_id"]
    source_ip = identity["source_ip"]

    def _audit_base(brand_id: Any) -> dict[str, Any]:
        return {
            "event": "manual_trigger",
            "principal_id": principal_id,
            "claim_source": claim_source,
            "brand_id": brand_id,
            "allowed_brands": sorted(allowed_brands),
            "request_id": request_id,
            "source_ip": source_ip,
        }

    if not allowed_brands:
        _emit_audit(
            logging.WARNING,
            {**_audit_base(None), "decision": "deny", "reason": "no_authorized_brands"},
        )
        raise PermissionError(
            "Authenticated principal is not authorized for any brand"
        )

    body = event.get("body", event)
    if isinstance(body, str):
        body = json.loads(body)
    if not isinstance(body, dict):
        _emit_audit(
            logging.WARNING,
            {**_audit_base(None), "decision": "deny", "reason": "invalid_body"},
        )
        raise ValueError("manual training request body must be a JSON object")

    # Reject unknown keys instead of splatting ``body`` into the forwarded
    # detail. This prevents mass-assignment of control-plane fields such as
    # ``pipeline_name`` through to ``start_pipeline_execution``.
    unexpected = set(body) - _MANUAL_TRIGGER_ALLOWED_FIELDS
    if unexpected:
        _emit_audit(
            logging.WARNING,
            {
                **_audit_base(body.get("brand_id")),
                "decision": "deny",
                "reason": "unexpected_fields",
                "unexpected_fields": sorted(unexpected),
            },
        )
        raise ValueError(
            "Unexpected field(s) in manual training request: "
            + ", ".join(sorted(unexpected))
        )

    brand_id = body.get("brand_id")
    if not brand_id:
        _emit_audit(
            logging.WARNING,
            {**_audit_base(None), "decision": "deny", "reason": "missing_brand_id"},
        )
        raise ValueError("brand_id is required for a manual training request")
    if brand_id not in allowed_brands:
        # Audit the attempt so brand-enumeration probing is alertable, but do
        # not leak the caller's bound brand set in the raised exception.
        _emit_audit(
            logging.WARNING,
            {**_audit_base(brand_id), "decision": "deny", "reason": "brand_not_authorized"},
        )
        raise PermissionError(
            f"Caller is not authorized to trigger training for brand_id {brand_id!r}"
        )

    forwarded_detail: dict[str, Any] = {
        "brand_id": brand_id,
        "trigger": "manual",
    }
    if principal_id is not None:
        # Propagate the principal into the forwarded detail so the SageMaker
        # record is attributable to the authenticated requester, not just the
        # brand. ``eventbridge_trigger_handler`` reads this key.
        forwarded_detail["requester_principal"] = str(principal_id)

    try:
        result = eventbridge_trigger_handler(
            {"detail": forwarded_detail},
            context,
            sagemaker_client=sagemaker_client,
            config=config,
        )
    except Exception as exc:  # audit the failure, then re-raise unchanged
        _emit_audit(
            logging.WARNING,
            {
                **_audit_base(brand_id),
                "decision": "error",
                "reason": type(exc).__name__,
            },
        )
        raise

    _emit_audit(
        logging.INFO,
        {
            **_audit_base(brand_id),
            "decision": "allow",
            "pipeline_execution_arn": result.get("pipeline_execution_arn"),
            "pipeline_execution_display_name": result.get("pipeline_execution_display_name"),
            "skipped": result.get("skipped", False),
        },
    )
    return result

"""Event-driven Lambda handlers for brand training workflows.

All AWS clients are injectable so unit tests and local mode never make network calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterable
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

# EventBridge PutEvents accepts at most 10 entries per call.
_EVENTBRIDGE_MAX_BATCH = 10

# Default window used to derive deterministic idempotency tokens / display
# names when the caller does not pin one. One execution per brand per window
# is the throttle unit enforced against SageMaker.
_DEFAULT_IDEMPOTENCY_WINDOW_SECONDS = 300

# SageMaker pipeline execution states that count as in-flight and should
# therefore suppress a new start for the same brand.
_IN_FLIGHT_PIPELINE_STATES = frozenset({"Executing", "Stopping"})

# Trusted EventBridge sources permitted to invoke ``eventbridge_trigger_handler``.
# Any event whose top-level ``source`` is not in this set, and that does not
# carry the in-process ``_trusted_invocation`` marker set by
# ``manual_trigger_handler`` after it has re-authorized the caller, is rejected.
# This closes the sibling-entry-point bypass: an attacker who can direct-invoke
# ``eventbridge_trigger_handler`` with a hand-crafted ``{"detail": ...}`` payload
# must now also forge a provenance marker that the Lambda runtime does not
# propagate from the caller.
_TRUSTED_EVENTBRIDGE_SOURCES = frozenset({"miso.ml.training"})

# Sentinel used by ``manual_trigger_handler`` to tag the EventBridge-shaped
# event it forwards into ``eventbridge_trigger_handler`` after it has verified
# the authenticated principal. Chosen as an object identity, not a string, so
# a client-supplied JSON payload cannot replicate it.


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


_TRUSTED_INVOCATION_SENTINEL = object()


def _assert_trusted_eventbridge_caller(event: dict[str, Any]) -> None:
    """Reject invocations that cannot be attributed to a trusted event source.

    ``eventbridge_trigger_handler`` performs no caller authentication of its
    own and starts another tenant's SageMaker pipeline on success. Any invoker
    able to shape the raw Lambda event (direct ``lambda:InvokeFunction`` or a
    pass-through API Gateway integration) could otherwise bypass the
    manual-trigger authorization gate just by hand-crafting a bare
    ``{"detail": {"brand_id": ...}}`` payload. The infra-side control for this
    is an ``aws_lambda_permission`` scoped to the EventBridge rule's ARN; this
    code-side check is a defense-in-depth that also closes the gap when the
    function is deployed without that resource policy.
    """
    if not isinstance(event, dict):
        raise PermissionError(
            "eventbridge_trigger_handler received a non-dict event"
        )
    # Internal re-entry from ``manual_trigger_handler`` after it has
    # re-authorized the caller is tagged with an object-identity sentinel the
    # caller cannot forge through a JSON payload.
    if event.get("_trusted_invocation") is _TRUSTED_INVOCATION_SENTINEL:
        return
    # An authorizer block on the event means this invocation was shaped like
    # an API Gateway proxy request; such a request must flow through
    # ``manual_trigger_handler`` so the principal is re-verified. Reject it
    # here so a sibling direct-invoke cannot be used to skip that gate.
    request_context = event.get("requestContext")
    if isinstance(request_context, dict) and request_context.get("authorizer"):
        raise PermissionError(
            "eventbridge_trigger_handler rejected a request shaped as an API "
            "Gateway proxy event; manual invocations must go through "
            "manual_trigger_handler"
        )
    source = event.get("source")
    if not isinstance(source, str) or source not in _TRUSTED_EVENTBRIDGE_SOURCES:
        raise PermissionError(
            "eventbridge_trigger_handler rejected an event from an untrusted "
            f"source: {source!r}"
        )


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

    Caller provenance is enforced via
    :func:`_assert_trusted_eventbridge_caller`: only genuine EventBridge
    deliveries from the configured source (or the in-process re-entry from
    ``manual_trigger_handler``) are accepted.
    """
    _assert_trusted_eventbridge_caller(event)
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


def _principal_brand_map(config: dict[str, Any] | None) -> dict[str, set[str]]:
    """Load the server-side principal-to-brand mapping.

    The mapping is a trusted, deployment-controlled artifact (NOT a value the
    caller can supply inside the request). Two sources are consulted, with the
    environment variable taking precedence so operators can rotate grants
    without redeploying code or config:

      * ``MISO_PRINCIPAL_BRAND_MAP`` env var, a JSON object of the form
        ``{"<sub>": ["brand-a", "brand-b"], ...}``.
      * ``principal_brand_map`` top-level key in ``config.yaml`` with the
        same shape.

    Values may be a list, a comma/space/JSON-array-encoded string, or a single
    brand id. Unknown principals map to the empty set so the caller is
    rejected by the manual-trigger gate without disclosing which principals
    are provisioned.
    """
    mapping: dict[str, set[str]] = {}
    raw_env = os.environ.get("MISO_PRINCIPAL_BRAND_MAP")
    if raw_env:
        try:
            parsed_env = json.loads(raw_env)
        except ValueError:
            parsed_env = None
        if isinstance(parsed_env, dict):
            for principal, brands in parsed_env.items():
                if not isinstance(principal, str) or not principal:
                    continue
                mapping.setdefault(principal, set()).update(_split_claim_values(brands))
    config_map = None
    if isinstance(config, dict):
        config_map = config.get("principal_brand_map")
    if isinstance(config_map, dict):
        for principal, brands in config_map.items():
            if not isinstance(principal, str) or not principal:
                continue
            mapping.setdefault(principal, set()).update(_split_claim_values(brands))
    return mapping


def _principal_from_authorizer(authorizer: dict[str, Any]) -> str | None:
    """Extract a verified principal identifier from an authorizer context.

    Looks for ``sub`` or ``principalId`` in (in order) the top-level
    authorizer dict, its ``claims`` child, and its ``jwt.claims`` child. The
    first non-empty string found wins. Any other type (number, dict, list)
    is ignored so a caller cannot sneak a structured value past a loose
    ``if principal:`` check downstream.
    """
    claims = authorizer.get("claims") if isinstance(authorizer.get("claims"), dict) else {}
    jwt_block = authorizer.get("jwt") if isinstance(authorizer.get("jwt"), dict) else {}
    jwt_claims = jwt_block.get("claims") if isinstance(jwt_block.get("claims"), dict) else {}
    for source in (authorizer, claims, jwt_claims):
        for key in ("sub", "principalId"):
            value = source.get(key) if isinstance(source, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _authorized_brands(
    event: dict[str, Any], config: dict[str, Any] | None = None
) -> set[str]:
    """Return the set of brand ids the authenticated caller is allowed to drive.

    The caller identity must come from an upstream API Gateway authorizer (JWT,
    Cognito, or Lambda authorizer) or REQUEST authorizer context propagated in
    ``event['requestContext']['authorizer']``. If no authorizer context is
    present, or the authorizer context carries no verified principal
    identifier (``sub`` / ``principalId``), the request is rejected as
    unauthenticated.

    The brand grant itself is **not** read from the request. It is resolved
    from a server-side principal-to-brand mapping (see
    :func:`_principal_brand_map`) keyed by the verified principal id, so a
    caller cannot self-elevate by injecting an ``allowed_brands`` or
    ``brand_id`` claim. ``cognito:groups`` entries of the form ``brand:<id>``
    are honored as a defense-in-depth fallback **only when** the IdP-controlled
    group claim is present alongside a verified ``sub`` -- a caller shaping the
    raw event payload cannot forge the Cognito-signed group claim upstream of
    this handler, but if there is any doubt, the server-side mapping should
    be the sole source.
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

    principal = _principal_from_authorizer(authorizer)
    if principal is None:
        raise PermissionError(
            "Unauthenticated manual training request: authorizer context has "
            "no verified principal identifier (sub/principalId)"
        )

    allowed: set[str] = set(_principal_brand_map(config).get(principal, set()))

    # ``cognito:groups`` is written by Cognito from the user's group
    # memberships, not from a value the client sends in the request. Honor
    # it as a secondary binding so operators who manage tenancy via Cognito
    # groups do not have to maintain a parallel mapping, but only after a
    # principal id has been verified above.
    claims = authorizer.get("claims") if isinstance(authorizer.get("claims"), dict) else {}
    jwt_block = authorizer.get("jwt") if isinstance(authorizer.get("jwt"), dict) else {}
    jwt_claims = jwt_block.get("claims") if isinstance(jwt_block.get("claims"), dict) else {}
    for source in (claims, jwt_claims, authorizer):
        if not isinstance(source, dict):
            continue
        for group in _split_claim_values(source.get("cognito:groups")):
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
    ``event['requestContext']['authorizer']``. The handler requires a verified
    principal identifier (``sub`` or ``principalId``) and authorizes the
    requested ``brand_id`` against the brand set bound to that principal in
    the deployment-controlled principal-to-brand mapping (see
    :func:`_principal_brand_map`), rejecting any brand outside it, regardless
    of whether the brand is listed in ``config.yaml``.
    """
    resolved_config = config or load_config()
    allowed_brands = _authorized_brands(event, resolved_config)
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
        {
            "detail": {"brand_id": brand_id, "trigger": "manual"},
            # Object-identity sentinel: ``manual_trigger_handler`` has already
            # verified the caller; signal that to the downstream provenance
            # check so it does not reject the in-process re-entry. A client
            # cannot replicate this through a JSON payload.
            "_trusted_invocation": _TRUSTED_INVOCATION_SENTINEL,
        },
        context,
        sagemaker_client=sagemaker_client,
        config=resolved_config,
    )

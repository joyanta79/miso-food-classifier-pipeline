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

# Upper bound on the ``list_pipeline_executions`` page used as the secondary
# in-flight signal. SageMaker caps ``MaxResults`` at 100; one page sorted by
# ``CreationTime Descending`` is enough to notice any brand currently running
# because the probe only needs to see any in-flight start, not enumerate all
# of history.
_IN_FLIGHT_LIST_PAGE_SIZE = 100

# Hard upper bound on how many pages the probe will walk, so a chatty
# control plane cannot turn the check into an unbounded fan-out.
_IN_FLIGHT_LIST_MAX_PAGES = 5


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


def _safe_brand_token(brand_id: str) -> str:
    """Return the canonicalized brand token used in display names / tags.

    Centralizing the sanitization keeps the duplicate-execution probe,
    display name, and reservation lock all comparing the same value instead
    of mixing the raw ``brand_id`` with its sanitized form.
    """
    return "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in brand_id)[:40]


def _idempotency_bucket(now: float | None = None) -> int:
    window = _idempotency_window_seconds()
    return int((now if now is not None else time.time()) // window)


def _pipeline_execution_display_name(
    brand_id: str, pipeline_name: str, now: float | None = None
) -> str:
    """Deterministic display name scoped to brand + current throttle window.

    SageMaker does **not** enforce that ``PipelineExecutionDisplayName`` is
    unique per pipeline, so this value alone cannot be relied on as a
    duplicate-start guard; the real guards are the DynamoDB conditional
    reservation (``_acquire_execution_reservation``) and the deterministic
    ``ClientRequestToken`` passed to ``start_pipeline_execution``. The
    display name is still useful as a human-readable correlation id and as
    a secondary signal for the list-based probe.
    """
    bucket = _idempotency_bucket(now)
    digest = hashlib.sha1(
        f"{pipeline_name}:{brand_id}:{bucket}".encode()
    ).hexdigest()[:12]
    safe_brand = _safe_brand_token(brand_id)
    return f"brand-{safe_brand}-{bucket}-{digest}"


def _client_request_token(brand_id: str, pipeline_name: str, now: float | None = None) -> str:
    """Deterministic idempotency token for ``start_pipeline_execution``.

    Collapses retries of the same logical trigger (same brand, same throttle
    window) into a single SageMaker-side execution regardless of how many
    concurrent Lambda invocations call Start. The token is derived from the
    sanitized brand so a raw-brand variant cannot escape the dedup.
    SageMaker requires the token to be 1-128 chars and match ``[a-zA-Z0-9-]*``.
    """
    bucket = _idempotency_bucket(now)
    digest = hashlib.sha256(
        f"{pipeline_name}:{brand_id}:{bucket}".encode()
    ).hexdigest()[:32]
    return f"miso-{bucket}-{digest}"


def _has_in_flight_execution(sagemaker: Any, pipeline_name: str, brand_id: str) -> bool:
    """Return True if a pipeline execution for ``brand_id`` is already running.

    This is only a best-effort secondary signal: the authoritative guard is
    the atomic reservation acquired by ``_acquire_execution_reservation`` and
    the deterministic ``ClientRequestToken`` passed to
    ``start_pipeline_execution``. The probe paginates with ``MaxResults`` so
    a long execution history cannot hide an in-flight run on page 2+, and it
    matches against the sanitized brand token used by the display name --
    the raw-``brand_id`` substring compare was both false-negative
    (sanitization mismatch -> duplicate start) and false-positive
    (one brand suppressing another whose id is a substring).

    Falls back to ``False`` -- letting the reservation / ClientRequestToken
    be the sole gates -- when the injected client does not implement
    ``list_pipeline_executions`` (e.g. the lightweight test doubles used in
    unit tests and local mode).
    """
    list_fn = getattr(sagemaker, "list_pipeline_executions", None)
    if list_fn is None:
        return False
    safe_brand = _safe_brand_token(brand_id)
    display_prefix = f"brand-{safe_brand}-"
    next_token: str | None = None
    for _ in range(_IN_FLIGHT_LIST_MAX_PAGES):
        request_kwargs: dict[str, Any] = {
            "PipelineName": pipeline_name,
            "SortBy": "CreationTime",
            "SortOrder": "Descending",
            "MaxResults": _IN_FLIGHT_LIST_PAGE_SIZE,
        }
        if next_token:
            request_kwargs["NextToken"] = next_token
        try:
            response = list_fn(**request_kwargs)
        except TypeError:
            # Older/stub clients may not accept ``MaxResults``/``NextToken``.
            # Retry with just the minimum kwargs so the probe still runs.
            try:
                response = list_fn(
                    PipelineName=pipeline_name,
                    SortBy="CreationTime",
                    SortOrder="Descending",
                )
            except Exception:
                return True
            next_token = None
        except Exception:
            # Fail closed on the in-flight check so a throttled or permission-
            # denied List call cannot escalate into unbounded Starts.
            return True
        for summary in response.get("PipelineExecutionSummaries", []) or []:
            if summary.get("PipelineExecutionStatus") not in _IN_FLIGHT_PIPELINE_STATES:
                continue
            display_name = summary.get("PipelineExecutionDisplayName", "")
            if display_name.startswith(display_prefix):
                return True
        next_token = response.get("NextToken") if isinstance(response, dict) else None
        if not next_token:
            break
    return False


def _reservation_table_name() -> str | None:
    name = os.environ.get("MISO_EXECUTION_LOCK_TABLE")
    return name.strip() if name and name.strip() else None


def _reservation_key(pipeline_name: str, brand_id: str, now: float | None = None) -> str:
    bucket = _idempotency_bucket(now)
    safe_brand = _safe_brand_token(brand_id)
    return f"{pipeline_name}#{safe_brand}#{bucket}"


def _acquire_execution_reservation(
    dynamodb: Any,
    pipeline_name: str,
    brand_id: str,
    now: float | None = None,
) -> tuple[bool, str | None]:
    """Atomically reserve the right to start one execution for a brand/window.

    Uses a DynamoDB ``PutItem`` with ``ConditionExpression`` of
    ``attribute_not_exists(pk)`` so exactly one concurrent trigger wins the
    reservation; all others see ``ConditionalCheckFailedException`` and must
    skip. The item carries a TTL (``expires_at``) so a crashed executor
    cannot wedge the brand forever.

    Returns ``(acquired, reservation_key)``. If ``dynamodb`` is ``None`` or
    the table name is not configured, returns ``(True, None)`` -- the
    deterministic ``ClientRequestToken`` on ``start_pipeline_execution`` is
    still enforced server-side, so this degradation does not re-open the
    duplicate-start race in environments that have not provisioned the lock
    table yet.
    """
    if dynamodb is None:
        return True, None
    table_name = _reservation_table_name()
    if not table_name:
        return True, None
    put_item = getattr(dynamodb, "put_item", None)
    if put_item is None:
        return True, None
    reservation_key = _reservation_key(pipeline_name, brand_id, now)
    now_ts = int(now if now is not None else time.time())
    # TTL slightly larger than the throttle window so a successful Start that
    # outlives the window still blocks concurrent peers for its duration.
    ttl = now_ts + max(_idempotency_window_seconds() * 2, 600)
    try:
        put_item(
            TableName=table_name,
            Item={
                "pk": {"S": reservation_key},
                "pipeline_name": {"S": pipeline_name},
                "brand_id": {"S": _safe_brand_token(brand_id)},
                "created_at": {"N": str(now_ts)},
                "expires_at": {"N": str(ttl)},
            },
            ConditionExpression="attribute_not_exists(pk)",
        )
    except Exception as exc:  # noqa: BLE001 - boto3 exception hierarchy varies
        name = type(exc).__name__
        response_meta = getattr(exc, "response", None) or {}
        error_code = ""
        if isinstance(response_meta, dict):
            error_code = (response_meta.get("Error") or {}).get("Code", "")
        if name == "ConditionalCheckFailedException" or error_code == "ConditionalCheckFailedException":
            return False, reservation_key
        # Fail closed: an unavailable lock table must not silently disable
        # the duplicate-start guard.
        raise
    return True, reservation_key


def _release_execution_reservation(
    dynamodb: Any, reservation_key: str | None
) -> None:
    """Best-effort release used when Start fails after a reservation was taken.

    Terminal-state cleanup is left to the TTL; this helper only unblocks
    peers if the Start call itself failed so a transient SageMaker error does
    not consume the brand's throttle window.
    """
    if dynamodb is None or not reservation_key:
        return
    table_name = _reservation_table_name()
    if not table_name:
        return
    delete_item = getattr(dynamodb, "delete_item", None)
    if delete_item is None:
        return
    try:
        delete_item(
            TableName=table_name,
            Key={"pk": {"S": reservation_key}},
        )
    except Exception:  # noqa: BLE001 - cleanup is best-effort
        return


def eventbridge_trigger_handler(
    event: dict[str, Any],
    _context: Any,
    *,
    sagemaker_client: Any | None = None,
    dynamodb_client: Any | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Start the per-brand SageMaker pipeline from an EventBridge request.

    Duplicate-execution suppression is layered so no single component is a
    load-bearing uniqueness guarantee:

    * Atomic reservation: when ``MISO_EXECUTION_LOCK_TABLE`` is configured a
      DynamoDB ``PutItem`` with ``ConditionExpression=attribute_not_exists``
      wins exactly once per brand/window across concurrent Lambda
      invocations. The TTL releases the slot if the executor crashes.
    * Deterministic ``ClientRequestToken``: SageMaker collapses retries that
      share the token into a single execution, so even a race that bypasses
      the reservation (lock table unavailable, test doubles) still cannot
      launch two GPU jobs for the same logical trigger.
    * Best-effort list probe: a paginated ``list_pipeline_executions`` scan
      is retained as a secondary signal that short-circuits the obvious case
      where a run is already visible on the control plane.
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

    sagemaker = _client("sagemaker", sagemaker_client)
    if _has_in_flight_execution(sagemaker, pipeline_name, brand_id):
        return {
            "brand_id": brand_id,
            "pipeline_execution_arn": None,
            "trigger": trigger,
            "skipped": True,
            "reason": "in_flight_execution",
        }

    # Only reach for a DynamoDB client when a lock table is actually
    # configured, so unit tests and local-mode runs (which never provision
    # one) do not require boto3.
    dynamodb: Any | None = dynamodb_client
    if dynamodb is None and _reservation_table_name():
        dynamodb = _client("dynamodb", None)

    acquired, reservation_key = _acquire_execution_reservation(
        dynamodb, pipeline_name, brand_id
    )
    if not acquired:
        return {
            "brand_id": brand_id,
            "pipeline_execution_arn": None,
            "trigger": trigger,
            "skipped": True,
            "reason": "reservation_held",
        }

    display_name = _pipeline_execution_display_name(brand_id, pipeline_name)
    client_request_token = _client_request_token(brand_id, pipeline_name)
    start_kwargs: dict[str, Any] = {
        "PipelineName": pipeline_name,
        "PipelineParameters": _pipeline_parameters(brand_id, trigger, resolved_config),
        "PipelineExecutionDisplayName": display_name,
        "ClientRequestToken": client_request_token,
    }
    try:
        try:
            response = sagemaker.start_pipeline_execution(**start_kwargs)
        except TypeError:
            # Test doubles / older clients may not accept the idempotency
            # kwargs. Degrade gracefully by dropping them one at a time so
            # the server-side guards are preserved whenever possible.
            fallback_kwargs = dict(start_kwargs)
            for optional in ("ClientRequestToken", "PipelineExecutionDisplayName"):
                fallback_kwargs.pop(optional, None)
                try:
                    response = sagemaker.start_pipeline_execution(**fallback_kwargs)
                    break
                except TypeError:
                    continue
            else:
                raise
    except Exception:
        # Release the reservation so a transient Start failure does not
        # consume the brand's throttle window.
        _release_execution_reservation(dynamodb, reservation_key)
        raise
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
    dynamodb_client: Any | None = None,
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
        dynamodb_client=dynamodb_client,
        config=config,
    )

from __future__ import annotations

import json
from typing import Any

import pytest

from lambdas.handlers import (
    eventbridge_trigger_handler,
    manual_trigger_handler,
    s3_trigger_handler,
)


@pytest.fixture
def config() -> dict[str, Any]:
    return {
        "s3": {"training_bucket": "training-bucket"},
        "brands": [
            {"id": "white-castle", "naming_pattern": "wc_*"},
            {"id": "brand-b", "naming_pattern": "bb_*"},
        ],
    }


class EventBridgeMock:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def put_events(self, *, Entries: list[dict[str, Any]]) -> dict[str, Any]:
        self.entries.extend(Entries)
        return {"FailedEntryCount": 0, "Entries": [{"EventId": "event-1"}]}


class SageMakerMock:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def start_pipeline_execution(self, **kwargs: Any) -> dict[str, str]:
        self.calls.append(kwargs)
        return {"PipelineExecutionArn": "arn:aws:sagemaker:local:pipeline/execution-1"}


def test_s3_trigger_publishes_matched_brand_event(config: dict[str, Any]) -> None:
    eventbridge = EventBridgeMock()
    result = s3_trigger_handler(
        {
            "Records": [
                {
                    "s3": {
                        "bucket": {"name": "incoming-images"},
                        "object": {"key": "raw/wc_burger%20image.jpg"},
                    }
                }
            ]
        },
        None,
        eventbridge_client=eventbridge,
        config=config,
    )

    assert result["accepted"] == 1
    assert len(eventbridge.entries) == 1
    assert json.loads(eventbridge.entries[0]["Detail"]) == {
        "brand_id": "white-castle",
        "bucket": "incoming-images",
        "key": "raw/wc_burger image.jpg",
        "trigger": "s3",
    }


def test_s3_eventbridge_shape_ignores_unknown_brand(config: dict[str, Any]) -> None:
    result = s3_trigger_handler(
        {
            "detail": {
                "bucket": {"name": "incoming-images"},
                "object": {"key": "raw/unknown_image.jpg"},
            }
        },
        None,
        eventbridge_client=EventBridgeMock(),
        config=config,
    )

    assert result == {
        "accepted": 0,
        "failed": 0,
        "message": "No configured brand matched the upload.",
    }


def test_eventbridge_trigger_starts_brand_pipeline(config: dict[str, Any]) -> None:
    sagemaker = SageMakerMock()
    result = eventbridge_trigger_handler(
        {"detail": {"brand_id": "brand-b", "trigger": "s3"}},
        None,
        sagemaker_client=sagemaker,
        config=config,
    )

    assert result["brand_id"] == "brand-b"
    assert len(sagemaker.calls) == 1
    call = sagemaker.calls[0]
    assert call["PipelineName"] == "miso-food-classifier-brand-b"
    assert call["PipelineParameters"] == [
        {"Name": "TriggerSource", "Value": "s3"},
    ]
    # Deterministic display name is attached so retries inside the throttle
    # window collide at SageMaker instead of starting duplicate GPU jobs.
    assert call["PipelineExecutionDisplayName"].startswith("brand-brand-b-")
    assert result["pipeline_execution_display_name"] == call["PipelineExecutionDisplayName"]


def test_eventbridge_trigger_ignores_caller_pipeline_name(
    config: dict[str, Any],
) -> None:
    sagemaker = SageMakerMock()
    eventbridge_trigger_handler(
        {
            "detail": {
                "brand_id": "white-castle",
                "pipeline_name": "attacker-chosen",
                "trigger": "s3",
            }
        },
        None,
        sagemaker_client=sagemaker,
        config=config,
    )

    assert sagemaker.calls[0]["PipelineName"] == "miso-food-classifier-white-castle"


def test_eventbridge_trigger_uses_custom_prefix_env(
    config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MISO_PIPELINE_NAME_PREFIX", "tenant-pipelines")
    sagemaker = SageMakerMock()
    eventbridge_trigger_handler(
        {"detail": {"brand_id": "brand-b", "trigger": "s3"}},
        None,
        sagemaker_client=sagemaker,
        config=config,
    )

    assert sagemaker.calls[0]["PipelineName"] == "tenant-pipelines-brand-b"


def _authed_event(body: Any, allowed_brands: str = "white-castle,brand-b") -> dict[str, Any]:
    """Build an API Gateway proxy event with an authenticated principal."""
    return {
        "body": body,
        "requestContext": {
            "authorizer": {
                "claims": {
                    "sub": "user-123",
                    "allowed_brands": allowed_brands,
                }
            }
        },
    }


def test_manual_trigger_accepts_json_body(config: dict[str, Any]) -> None:
    sagemaker = SageMakerMock()
    result = manual_trigger_handler(
        _authed_event(json.dumps({"brand_id": "white-castle"})),
        None,
        sagemaker_client=sagemaker,
        config=config,
    )

    assert result["trigger"] == "manual"
    assert sagemaker.calls[0]["PipelineName"] == "miso-food-classifier-white-castle"
    assert sagemaker.calls[0]["PipelineParameters"][0] == {
        "Name": "TriggerSource",
        "Value": "manual",
    }


def test_manual_trigger_rejects_unknown_fields(config: dict[str, Any]) -> None:
    sagemaker = SageMakerMock()
    with pytest.raises(ValueError, match="Unexpected field"):
        manual_trigger_handler(
            _authed_event(
                json.dumps(
                    {
                        "brand_id": "white-castle",
                        "pipeline_name": "attacker-chosen",
                    }
                )
            ),
            None,
            sagemaker_client=sagemaker,
            config=config,
        )


def test_s3_trigger_coalesces_uploads_per_brand(config: dict[str, Any]) -> None:
    """A burst of uploads for the same brand collapses to one event."""
    eventbridge = EventBridgeMock()
    records = [
        {
            "s3": {
                "bucket": {"name": "incoming-images"},
                "object": {"key": f"raw/wc_{i}.jpg"},
            }
        }
        for i in range(25)
    ]
    records.extend(
        {
            "s3": {
                "bucket": {"name": "incoming-images"},
                "object": {"key": f"raw/bb_{i}.jpg"},
            }
        }
        for i in range(3)
    )
    result = s3_trigger_handler(
        {"Records": records},
        None,
        eventbridge_client=eventbridge,
        config=config,
    )

    assert len(eventbridge.entries) == 2  # one per brand, not one per object
    assert result["accepted"] == 2
    brands = sorted(json.loads(entry["Detail"])["brand_id"] for entry in eventbridge.entries)
    assert brands == ["brand-b", "white-castle"]


def test_s3_trigger_chunks_put_events_to_batches_of_ten(config: dict[str, Any]) -> None:
    """When >10 brands match in one invocation, put_events is chunked."""

    class BatchingEventBridgeMock:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def put_events(self, *, Entries: list[dict[str, Any]]) -> dict[str, Any]:
            assert len(Entries) <= 10, "EventBridge PutEvents accepts at most 10 entries"
            self.batch_sizes.append(len(Entries))
            return {
                "FailedEntryCount": 0,
                "Entries": [{"EventId": f"e-{i}"} for i in range(len(Entries))],
            }

    many_brand_config = {
        "s3": {"training_bucket": "training-bucket"},
        "brands": [
            {"id": f"brand-{i}", "naming_pattern": f"b{i}_*"} for i in range(15)
        ],
    }
    records = [
        {
            "s3": {
                "bucket": {"name": "incoming-images"},
                "object": {"key": f"raw/b{i}_file.jpg"},
            }
        }
        for i in range(15)
    ]
    eventbridge = BatchingEventBridgeMock()
    result = s3_trigger_handler(
        {"Records": records},
        None,
        eventbridge_client=eventbridge,
        config=many_brand_config,
    )

    assert eventbridge.batch_sizes == [10, 5]
    assert result["accepted"] == 15


def test_eventbridge_trigger_skips_when_execution_in_flight(config: dict[str, Any]) -> None:
    """An already-running execution for the brand suppresses a new start."""

    class SageMakerWithRunningExecution(SageMakerMock):
        def list_pipeline_executions(self, **_: Any) -> dict[str, Any]:
            return {
                "PipelineExecutionSummaries": [
                    {
                        "PipelineExecutionStatus": "Executing",
                        "PipelineExecutionDisplayName": "brand-brand-b-1-abcdef",
                    }
                ]
            }

    sagemaker = SageMakerWithRunningExecution()
    result = eventbridge_trigger_handler(
        {"detail": {"brand_id": "brand-b", "trigger": "s3"}},
        None,
        sagemaker_client=sagemaker,
        config=config,
    )

    assert sagemaker.calls == []  # no new start_pipeline_execution
    assert result["skipped"] is True
    assert result["reason"] == "in_flight_execution"


def test_manual_trigger_rejects_unknown_brand(config: dict[str, Any]) -> None:
    with pytest.raises((PermissionError, ValueError)):
        manual_trigger_handler(
            _authed_event({"brand_id": "not-a-brand"}),
            None,
            sagemaker_client=SageMakerMock(),
            config=config,
        )


def test_manual_trigger_rejects_missing_authorizer(config: dict[str, Any]) -> None:
    with pytest.raises(PermissionError):
        manual_trigger_handler(
            {"body": json.dumps({"brand_id": "white-castle"})},
            None,
            sagemaker_client=SageMakerMock(),
            config=config,
        )


def test_manual_trigger_rejects_empty_authorizer(config: dict[str, Any]) -> None:
    with pytest.raises(PermissionError):
        manual_trigger_handler(
            {
                "body": json.dumps({"brand_id": "white-castle"}),
                "requestContext": {"authorizer": {}},
            },
            None,
            sagemaker_client=SageMakerMock(),
            config=config,
        )


def test_manual_trigger_rejects_brand_outside_principal_binding(
    config: dict[str, Any],
) -> None:
    with pytest.raises(PermissionError):
        manual_trigger_handler(
            _authed_event(
                json.dumps({"brand_id": "brand-b"}),
                allowed_brands="white-castle",
            ),
            None,
            sagemaker_client=SageMakerMock(),
            config=config,
        )


def test_manual_trigger_accepts_cognito_groups_brand_binding(
    config: dict[str, Any],
) -> None:
    sagemaker = SageMakerMock()
    event = {
        "body": json.dumps({"brand_id": "brand-b"}),
        "requestContext": {
            "authorizer": {
                "claims": {
                    "sub": "user-123",
                    "cognito:groups": "brand:brand-b,ops",
                }
            }
        },
    }
    result = manual_trigger_handler(event, None, sagemaker_client=sagemaker, config=config)
    assert result["brand_id"] == "brand-b"


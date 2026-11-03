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
    assert sagemaker.calls == [
        {
            "PipelineName": "miso-food-classifier-brand-b",
            "PipelineParameters": [
                {"Name": "BrandId", "Value": "brand-b"},
                {"Name": "TriggerSource", "Value": "s3"},
                {"Name": "TrainingBucket", "Value": "training-bucket"},
            ],
        }
    ]


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
    assert sagemaker.calls[0]["PipelineParameters"][1] == {
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
    assert sagemaker.calls == []


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

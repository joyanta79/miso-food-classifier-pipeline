import json
from unittest.mock import Mock

from processing.step7_register_and_baseline import (
    EVENT_DETAIL_TYPE,
    EVENT_SOURCE,
    PENDING_APPROVAL,
    approval_event,
    register_and_baseline,
)


def _config():
    return {
        "s3": {"model_artifacts_bucket": "model-artifacts"},
        "brands": [{"id": "white-castle", "model_package_group": "FoodClassifier-WhiteCastle"}],
    }


def _evaluation():
    return {
        "quality_gate": "passed",
        "metrics": {
            "sample_count": 10,
            "label_distribution": {"done": 5, "not_done": 5},
            "confidence_summary": {"count": 10, "mean": 0.91, "min": 0.75, "max": 0.99},
            "accuracy": 0.98,
            "done_when_not_done_fp_rate": 0.02,
        },
    }


def test_registers_pending_manual_approval_persists_baseline_and_emits_event():
    sagemaker = Mock()
    sagemaker.create_model_package.return_value = {"ModelPackageArn": "arn:aws:sagemaker:pkg/123"}
    s3 = Mock()
    events = Mock()

    result = register_and_baseline(
        config=_config(),
        brand_id="white-castle",
        model_data_s3_uri="s3://model-artifacts/models/white-castle/model.tar.gz",
        image_uri="123456.dkr.ecr.us-west-2.amazonaws.com/inference:latest",
        evaluation_report_s3_uri="s3://pipeline-artifacts/evaluation.json",
        evaluation=_evaluation(),
        sagemaker_client=sagemaker,
        s3_client=s3,
        events_client=events,
    )

    registration = sagemaker.create_model_package.call_args.kwargs
    assert registration["ModelPackageGroupName"] == "FoodClassifier-WhiteCastle"
    assert registration["ModelApprovalStatus"] == PENDING_APPROVAL
    assert registration["CustomerMetadataProperties"]["manual_approval_required"] == "true"
    assert registration["ModelMetrics"]["ModelQuality"]["Statistics"]["S3Uri"] == "s3://pipeline-artifacts/evaluation.json"

    baseline_call = s3.put_object.call_args.kwargs
    baseline = json.loads(baseline_call["Body"].decode("utf-8"))
    assert baseline["quality_metrics"]["done_when_not_done_fp_rate"] == 0.02
    assert result["model_approval_status"] == PENDING_APPROVAL

    event_entry = events.put_events.call_args.kwargs["Entries"][0]
    assert event_entry["Source"] == EVENT_SOURCE
    assert event_entry["DetailType"] == EVENT_DETAIL_TYPE
    assert event_entry["EventBusName"] == "default"
    detail = json.loads(event_entry["Detail"])
    assert detail["model_approval_status"] == PENDING_APPROVAL
    assert detail["required_action"] == "manual_model_approval"


def test_approval_event_has_stable_model_approval_shape():
    entry = approval_event(
        brand_id="white-castle",
        model_package_group_name="FoodClassifier-WhiteCastle",
        model_package_arn="arn:aws:sagemaker:pkg/123",
        baseline_statistics_s3_uri="s3://model-artifacts/baseline.json",
        evaluation_report_s3_uri="s3://pipeline-artifacts/evaluation.json",
        quality_gate="passed",
    )

    assert set(entry) == {"Source", "DetailType", "Detail"}
    assert entry["Source"] == EVENT_SOURCE
    assert entry["DetailType"] == EVENT_DETAIL_TYPE
    assert json.loads(entry["Detail"]) == {
        "baseline_statistics_s3_uri": "s3://model-artifacts/baseline.json",
        "brand_id": "white-castle",
        "evaluation_report_s3_uri": "s3://pipeline-artifacts/evaluation.json",
        "model_approval_status": PENDING_APPROVAL,
        "model_package_arn": "arn:aws:sagemaker:pkg/123",
        "model_package_group_name": "FoodClassifier-WhiteCastle",
        "quality_gate": "passed",
        "required_action": "manual_model_approval",
        "schema_version": "1.0",
    }

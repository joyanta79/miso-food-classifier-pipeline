from __future__ import annotations

from typing import Any

import pytest

from migration.datasync_task import DataSyncTask
from monitoring.model_monitor_setup import ModelMonitorSetup


@pytest.fixture
def config() -> dict[str, Any]:
    return {
        "drift": {
            "level1_confidence_floor": 0.65,
            "level2_confidence_threshold": 0.70,
            "correction_rate_multiplier": 3,
        },
        "notifications": {
            "review_topic_arn": "arn:aws:sns:local:review",
            "ml_lead_topic_arn": "arn:aws:sns:local:ml-lead",
            "vp_engineering_topic_arn": "arn:aws:sns:local:vp",
        },
        "training": {"processing_instance_type": "ml.m5.xlarge"},
    }


class CorrectionDbMock:
    def get_correction_rate(self, brand_id: str) -> float:
        assert brand_id == "white-castle"
        return 0.31


class CloudWatchMock:
    def __init__(self) -> None:
        self.metrics: list[dict[str, Any]] = []
        self.alarms: list[dict[str, Any]] = []

    def put_metric_data(self, **kwargs: Any) -> dict[str, Any]:
        self.metrics.append(kwargs)
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def put_metric_alarm(self, **kwargs: Any) -> dict[str, Any]:
        self.alarms.append(kwargs)
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}


class SageMakerMock:
    def create_model_quality_job_definition(self, **kwargs: Any) -> dict[str, str]:
        return {"JobDefinitionArn": f"arn:job/{kwargs['JobDefinitionName']}"}

    def create_monitoring_schedule(self, **kwargs: Any) -> dict[str, str]:
        return {"MonitoringScheduleArn": f"arn:schedule/{kwargs['MonitoringScheduleName']}"}


class DataSyncMock:
    def __init__(self) -> None:
        self.location_requests: list[dict[str, Any]] = []
        self.task_request: dict[str, Any] | None = None

    def create_location_s3(self, **kwargs: Any) -> dict[str, str]:
        self.location_requests.append(kwargs)
        return {"LocationArn": f"arn:location/{len(self.location_requests)}"}

    def create_task(self, **kwargs: Any) -> dict[str, str]:
        self.task_request = kwargs
        return {"TaskArn": "arn:task/1"}

    def start_task_execution(self, *, TaskArn: str) -> dict[str, str]:
        return {"TaskExecutionArn": f"{TaskArn}/execution/1"}


class S3HeadMock:
    def __init__(self, head: dict[str, str]) -> None:
        self.head = head
        self.requests: list[dict[str, str]] = []

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, str]:
        self.requests.append({"Bucket": Bucket, "Key": Key})
        return self.head


def test_drift_levels_include_mock_correction_database_metric(config: dict[str, Any]) -> None:
    monitor = ModelMonitorSetup(config)

    assert monitor.assess_drift(0.69, 0.10, 0.10).level == 1
    assert monitor.assess_drift(0.64, 0.10, 0.10).level == 2
    assessment = monitor.assess_from_correction_db(0.90, CorrectionDbMock(), 0.10, "white-castle")
    assert assessment.level == 3
    assert assessment.reasons == ("correction_rate_spike",)


def test_model_monitor_schedule_signal_and_alarm_specs(config: dict[str, Any]) -> None:
    monitor = ModelMonitorSetup(config)
    job = monitor.model_quality_job_definition(
        job_name="food-monitor",
        endpoint_name="food-classifier",
        role_arn="arn:role/monitor",
        image_uri="image-uri",
        baseline_constraints_uri="s3://baseline/constraints.json",
        baseline_statistics_uri="s3://baseline/statistics.json",
        output_uri="s3://monitor-output/",
    )
    schedule = monitor.monitoring_schedule("food-schedule", "food-monitor")
    cloudwatch = CloudWatchMock()
    response = monitor.apply(
        sagemaker_client=SageMakerMock(),
        cloudwatch_client=cloudwatch,
        job_definition=job,
        schedule=schedule,
        endpoint_name="food-classifier",
    )

    assert job["ModelQualityAppSpecification"]["ProblemType"] == "MulticlassClassification"
    assert schedule["MonitoringScheduleConfig"]["MonitoringType"] == "ModelQuality"
    assert [alarm["Threshold"] for alarm in cloudwatch.alarms] == [1.0, 2.0, 3.0]
    assert [alarm["AlarmActions"] for alarm in cloudwatch.alarms] == [
        ["arn:aws:sns:local:review"],
        ["arn:aws:sns:local:ml-lead"],
        ["arn:aws:sns:local:vp"],
    ]
    assert response["schedule"]["MonitoringScheduleArn"] == "arn:schedule/food-schedule"

    assessment = monitor.assess_drift(0.64, 0.1, 0.1)
    monitor.publish_drift_signal(cloudwatch, "food-classifier", assessment)
    assert cloudwatch.metrics[0]["MetricData"][0]["Value"] == 2


def test_datasync_uses_s3_standard_and_validates_sha256() -> None:
    task = DataSyncTask(
        task_name="archive-migration",
        source_bucket_arn="arn:aws:s3:::source",
        destination_bucket_arn="arn:aws:s3:::destination",
        bucket_access_role_arn="arn:aws:iam::123:role/DataSync",
    )
    datasync = DataSyncMock()
    source = S3HeadMock({"ChecksumSHA256": "same-checksum"})
    destination = S3HeadMock({"ChecksumSHA256": "same-checksum"})
    result = task.start_and_validate(
        datasync,
        source,
        destination,
        [
            {
                "source_bucket": "source",
                "destination_bucket": "destination",
                "source_key": "raw/image.jpg",
            }
        ],
    )

    assert datasync.location_requests[1]["S3StorageClass"] == "STANDARD"
    assert datasync.task_request is not None
    assert datasync.task_request["Options"]["VerifyMode"] == "ONLY_FILES_TRANSFERRED"
    assert result["checksums"][0].matched is True


def test_datasync_rejects_checksum_mismatch() -> None:
    task = DataSyncTask(
        task_name="archive-migration",
        source_bucket_arn="arn:aws:s3:::source",
        destination_bucket_arn="arn:aws:s3:::destination",
        bucket_access_role_arn="arn:aws:iam::123:role/DataSync",
    )
    with pytest.raises(ValueError, match="Checksum mismatch"):
        task.validate_checksums(
            S3HeadMock({"ETag": '"source"'}),
            S3HeadMock({"ETag": '"destination"'}),
            [
                {
                    "source_bucket": "source",
                    "destination_bucket": "destination",
                    "source_key": "raw/image.jpg",
                }
            ],
        )

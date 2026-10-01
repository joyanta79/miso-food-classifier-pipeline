"""SageMaker Model Monitor and CloudWatch drift configuration.

The module builds AWS request specifications separately from applying them.  This
keeps all AWS interactions injectable and makes local mode fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DriftAssessment:
    """A normalized drift decision suitable for metrics and alerting."""

    level: int
    reasons: tuple[str, ...]
    confidence: float
    correction_rate: float


class ModelMonitorSetup:
    """Create model-monitor resources and publish levelled drift measurements."""

    namespace = "MisoRobotics/ModelMonitor"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def assess_drift(
        self,
        confidence: float,
        correction_rate: float,
        baseline_correction_rate: float,
    ) -> DriftAssessment:
        """Classify drift from confidence and human-correction metrics.

        L1 is an early confidence warning, L2 is a confidence-floor breach, and
        L3 is an urgent correction-rate spike.  The highest applicable level wins.
        """
        if baseline_correction_rate < 0:
            raise ValueError("baseline_correction_rate must be non-negative")

        drift_config = self.config["drift"]
        reasons: list[str] = []
        level = 0
        if confidence < drift_config["level2_confidence_threshold"]:
            level = 1
            reasons.append("confidence_below_level1_threshold")
        if confidence < drift_config["level1_confidence_floor"]:
            level = 2
            reasons.append("confidence_below_level2_floor")
        if baseline_correction_rate == 0:
            correction_spike = correction_rate > 0
        else:
            correction_spike = (
                correction_rate
                >= baseline_correction_rate * drift_config["correction_rate_multiplier"]
            )
        if correction_spike:
            level = 3
            reasons.append("correction_rate_spike")

        return DriftAssessment(level, tuple(reasons), confidence, correction_rate)

    def assess_from_correction_db(
        self,
        confidence: float,
        correction_db: Any,
        baseline_correction_rate: float,
        brand_id: str,
    ) -> DriftAssessment:
        """Read the correction metric through a small mock-friendly DB interface."""
        if hasattr(correction_db, "get_correction_rate"):
            correction_rate = correction_db.get_correction_rate(brand_id)
        elif hasattr(correction_db, "query_correction_rate"):
            correction_rate = correction_db.query_correction_rate(brand_id)
        else:
            raise TypeError(
                "correction_db must expose get_correction_rate or query_correction_rate"
            )
        return self.assess_drift(confidence, float(correction_rate), baseline_correction_rate)

    def model_quality_job_definition(
        self,
        *,
        job_name: str,
        endpoint_name: str,
        role_arn: str,
        image_uri: str,
        baseline_constraints_uri: str,
        baseline_statistics_uri: str,
        output_uri: str,
    ) -> dict[str, Any]:
        """Build a SageMaker Model Quality job definition request."""
        return {
            "JobDefinitionName": job_name,
            "ModelQualityAppSpecification": {
                "ImageUri": image_uri,
                "ProblemType": "MulticlassClassification",
            },
            "ModelQualityBaselineConfig": {
                "ConstraintsResource": {"S3Uri": baseline_constraints_uri},
                "StatisticsResource": {"S3Uri": baseline_statistics_uri},
            },
            "ModelQualityJobInput": {
                "EndpointInput": {
                    "EndpointName": endpoint_name,
                    "LocalPath": "/opt/ml/processing/input",
                    "S3InputMode": "File",
                }
            },
            "ModelQualityJobOutputConfig": {
                "MonitoringOutputs": [
                    {"S3Output": {"LocalPath": "/opt/ml/processing/output", "S3Uri": output_uri}}
                ]
            },
            "JobResources": {
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": self.config["training"]["processing_instance_type"],
                    "VolumeSizeInGB": 30,
                }
            },
            "RoleArn": role_arn,
        }

    @staticmethod
    def monitoring_schedule(
        schedule_name: str,
        job_definition_name: str,
        cron_expression: str = "cron(0 * * * ? *)",
    ) -> dict[str, Any]:
        """Build the hourly SageMaker Model Monitor schedule specification."""
        return {
            "MonitoringScheduleName": schedule_name,
            "MonitoringScheduleConfig": {
                "MonitoringJobDefinitionName": job_definition_name,
                "MonitoringType": "ModelQuality",
                "ScheduleConfig": {"ScheduleExpression": cron_expression},
            },
        }

    def drift_alarm_specs(self, endpoint_name: str) -> list[dict[str, Any]]:
        """Build CloudWatch alarms for L1 review, L2 ML lead, and L3 escalation."""
        notifications = self.config["notifications"]
        topics = {
            1: notifications["review_topic_arn"],
            2: notifications["ml_lead_topic_arn"],
            3: notifications["vp_engineering_topic_arn"],
        }
        return [
            {
                "AlarmName": f"miso-{endpoint_name}-model-drift-level-{level}",
                "AlarmDescription": f"Miso model drift reached level {level}",
                "Namespace": self.namespace,
                "MetricName": "DriftLevel",
                "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}],
                "Statistic": "Maximum",
                "Period": 300,
                "EvaluationPeriods": 1,
                "Threshold": float(level),
                "ComparisonOperator": "GreaterThanOrEqualToThreshold",
                "TreatMissingData": "notBreaching",
                "AlarmActions": [topics[level]],
            }
            for level in (1, 2, 3)
        ]

    def publish_drift_signal(
        self,
        cloudwatch_client: Any,
        endpoint_name: str,
        assessment: DriftAssessment,
    ) -> dict[str, Any]:
        """Publish a CloudWatch drift-level measurement through an injected client."""
        return cloudwatch_client.put_metric_data(
            Namespace=self.namespace,
            MetricData=[
                {
                    "MetricName": "DriftLevel",
                    "Value": assessment.level,
                    "Unit": "Count",
                    "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}],
                }
            ],
        )

    def apply(
        self,
        *,
        sagemaker_client: Any,
        cloudwatch_client: Any,
        job_definition: dict[str, Any],
        schedule: dict[str, Any],
        endpoint_name: str,
    ) -> dict[str, Any]:
        """Apply all model-monitor resources through caller-supplied clients."""
        job_response = sagemaker_client.create_model_quality_job_definition(**job_definition)
        schedule_response = sagemaker_client.create_monitoring_schedule(**schedule)
        alarm_responses = [
            cloudwatch_client.put_metric_alarm(**spec)
            for spec in self.drift_alarm_specs(endpoint_name)
        ]
        return {
            "job_definition": job_response,
            "schedule": schedule_response,
            "alarms": alarm_responses,
        }

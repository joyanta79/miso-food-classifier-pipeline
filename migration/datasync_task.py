"""DataSync task construction and checksum validation for raw-image migration."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def validate_checksums(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Compare SHA-256 digests of two local files (used by local-mode checks).

    This is a lightweight, filesystem-level counterpart to
    ``DataSyncTask.validate_checksums`` for verifying migrated artifacts locally.
    """
    source_digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    destination_digest = hashlib.sha256(Path(destination).read_bytes()).hexdigest()
    return {
        "source_sha256": source_digest,
        "destination_sha256": destination_digest,
        "match": source_digest == destination_digest,
    }


@dataclass(frozen=True)
class ChecksumResult:
    source_key: str
    destination_key: str
    algorithm: str
    matched: bool


class DataSyncTask:
    """Create an S3 Standard DataSync task and validate transferred checksums."""

    def __init__(
        self,
        *,
        task_name: str,
        source_bucket_arn: str,
        destination_bucket_arn: str,
        bucket_access_role_arn: str,
        source_prefix: str = "/",
        destination_prefix: str = "/",
    ) -> None:
        self.task_name = task_name
        self.source_bucket_arn = source_bucket_arn
        self.destination_bucket_arn = destination_bucket_arn
        self.bucket_access_role_arn = bucket_access_role_arn
        self.source_prefix = source_prefix
        self.destination_prefix = destination_prefix

    @staticmethod
    def location_request(
        bucket_arn: str,
        role_arn: str,
        subdirectory: str,
        *,
        destination: bool,
    ) -> dict[str, Any]:
        """Build a DataSync S3 location request, explicitly retaining S3 Standard."""
        request: dict[str, Any] = {
            "S3BucketArn": bucket_arn,
            "S3Config": {"BucketAccessRoleArn": role_arn},
            "Subdirectory": subdirectory,
        }
        if destination:
            request["S3StorageClass"] = "STANDARD"
        return request

    def create_locations(self, datasync_client: Any) -> tuple[str, str]:
        """Create source and destination locations via the injected DataSync client."""
        source = datasync_client.create_location_s3(
            **self.location_request(
                self.source_bucket_arn,
                self.bucket_access_role_arn,
                self.source_prefix,
                destination=False,
            )
        )
        destination = datasync_client.create_location_s3(
            **self.location_request(
                self.destination_bucket_arn,
                self.bucket_access_role_arn,
                self.destination_prefix,
                destination=True,
            )
        )
        return source["LocationArn"], destination["LocationArn"]

    def task_request(
        self, source_location_arn: str, destination_location_arn: str
    ) -> dict[str, Any]:
        """Build a transfer request with DataSync verification enabled."""
        return {
            "SourceLocationArn": source_location_arn,
            "DestinationLocationArn": destination_location_arn,
            "Name": self.task_name,
            "Options": {
                "VerifyMode": "ONLY_FILES_TRANSFERRED",
                "OverwriteMode": "ALWAYS",
                "Atime": "NONE",
                "Mtime": "PRESERVE",
                "PreserveDeletedFiles": "PRESERVE",
            },
        }

    def create_task(self, datasync_client: Any) -> str:
        """Create and return a task ARN without constructing an SDK client."""
        source_location_arn, destination_location_arn = self.create_locations(datasync_client)
        response = datasync_client.create_task(
            **self.task_request(source_location_arn, destination_location_arn)
        )
        return response["TaskArn"]

    @staticmethod
    def _checksum(head: dict[str, Any]) -> tuple[str, str]:
        """Use SHA-256 when available; otherwise compare normalized ETags."""
        checksum = head.get("ChecksumSHA256")
        if checksum:
            return "SHA256", checksum
        etag = head.get("ETag")
        if not etag:
            raise ValueError("Object metadata lacks ChecksumSHA256 and ETag")
        return "ETag", str(etag).strip('"')

    def validate_checksums(
        self,
        source_s3_client: Any,
        destination_s3_client: Any,
        objects: Iterable[dict[str, str]],
    ) -> list[ChecksumResult]:
        """Validate source/destination checksums and fail closed on any mismatch."""
        results: list[ChecksumResult] = []
        for item in objects:
            source_key = item["source_key"]
            destination_key = item.get("destination_key", source_key)
            source_head = source_s3_client.head_object(Bucket=item["source_bucket"], Key=source_key)
            destination_head = destination_s3_client.head_object(
                Bucket=item["destination_bucket"], Key=destination_key
            )
            source_algorithm, source_value = self._checksum(source_head)
            destination_algorithm, destination_value = self._checksum(destination_head)
            matched = (
                source_algorithm == destination_algorithm and source_value == destination_value
            )
            result = ChecksumResult(source_key, destination_key, source_algorithm, matched)
            results.append(result)
            if not matched:
                raise ValueError(
                    f"Checksum mismatch for {source_key}: "
                    f"{source_algorithm}={source_value!r}, "
                    f"{destination_algorithm}={destination_value!r}"
                )
        return results

    def start_and_validate(
        self,
        datasync_client: Any,
        source_s3_client: Any,
        destination_s3_client: Any,
        objects: Iterable[dict[str, str]],
        task_arn: str | None = None,
    ) -> dict[str, Any]:
        """Start a task and run explicit post-transfer S3 checksum validation."""
        resolved_task_arn = task_arn or self.create_task(datasync_client)
        execution = datasync_client.start_task_execution(TaskArn=resolved_task_arn)
        checksums = self.validate_checksums(source_s3_client, destination_s3_client, objects)
        return {
            "task_arn": resolved_task_arn,
            "task_execution_arn": execution["TaskExecutionArn"],
            "checksums": checksums,
        }

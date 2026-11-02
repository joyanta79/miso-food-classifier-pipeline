"""DataSync task construction and checksum validation for raw-image migration."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Terminal states for a DataSync TaskExecution. Only ``SUCCESS`` indicates that
# every file in scope has been copied and verified by DataSync; the remaining
# states (``ERROR``) are terminal failures. All other states
# (``LAUNCHING``, ``PREPARING``, ``TRANSFERRING``, ``VERIFYING``, ``QUEUED``)
# are non-terminal and must be polled through.
_DATASYNC_TERMINAL_STATES = frozenset({"SUCCESS", "ERROR"})


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
    destination_version_id: str | None = None


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

    @staticmethod
    def _head_destination(
        destination_s3_client: Any, bucket: str, key: str
    ) -> dict[str, Any]:
        """HEAD the destination object, converting a 404 into a validation failure.

        A missing destination object means the DataSync transfer did not write
        the object we were asked to verify (or the key is wrong). Treat it as
        an integrity-gate failure rather than letting the underlying
        ``ClientError``/404 escape unhandled.
        """
        try:
            return destination_s3_client.head_object(Bucket=bucket, Key=key)
        except Exception as exc:  # pragma: no cover - exercised at integration
            # Detect the common boto3 404 shape without a hard dependency on
            # botocore being importable at module load time.
            response = getattr(exc, "response", None)
            if isinstance(response, dict):
                error = response.get("Error", {})
                code = str(error.get("Code", ""))
                status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
                    raise ValueError(
                        f"Destination object missing for {key!r} in {bucket!r}: "
                        "post-transfer integrity validation failed."
                    ) from exc
            raise

    def validate_checksums(
        self,
        source_s3_client: Any,
        destination_s3_client: Any,
        objects: Iterable[dict[str, str]],
    ) -> list[ChecksumResult]:
        """Validate source/destination checksums and fail closed on any mismatch.

        The destination ``VersionId`` (when S3 versioning is enabled) is
        captured on the ``ChecksumResult`` so the verified digest is pinned to
        the exact object version that was migrated, rather than to a key that
        may be overwritten later.
        """
        results: list[ChecksumResult] = []
        for item in objects:
            source_key = item["source_key"]
            destination_key = item.get("destination_key", source_key)
            source_head = source_s3_client.head_object(Bucket=item["source_bucket"], Key=source_key)
            destination_head = self._head_destination(
                destination_s3_client, item["destination_bucket"], destination_key
            )
            source_algorithm, source_value = self._checksum(source_head)
            destination_algorithm, destination_value = self._checksum(destination_head)
            matched = (
                source_algorithm == destination_algorithm and source_value == destination_value
            )
            destination_version_id = destination_head.get("VersionId")
            result = ChecksumResult(
                source_key,
                destination_key,
                source_algorithm,
                matched,
                destination_version_id,
            )
            results.append(result)
            if not matched:
                raise ValueError(
                    f"Checksum mismatch for {source_key}: "
                    f"{source_algorithm}={source_value!r}, "
                    f"{destination_algorithm}={destination_value!r}"
                )
        return results

    @staticmethod
    def wait_for_execution(
        datasync_client: Any,
        task_execution_arn: str,
        *,
        poll_interval_seconds: float = 15.0,
        timeout_seconds: float = 6 * 60 * 60,
        sleep: Any = time.sleep,
        now: Any = time.monotonic,
    ) -> dict[str, Any]:
        """Block until a DataSync TaskExecution reaches a terminal state.

        Returns the final ``describe_task_execution`` response on ``SUCCESS``.
        Raises ``RuntimeError`` on ``ERROR`` or on timeout. ``StartTaskExecution``
        is asynchronous, so this synchronization is required before any
        post-transfer integrity check can inspect destination bytes.
        """
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        deadline = now() + timeout_seconds
        while True:
            description = datasync_client.describe_task_execution(
                TaskExecutionArn=task_execution_arn
            )
            status = description.get("Status")
            if status == "SUCCESS":
                return description
            if status == "ERROR":
                result = description.get("Result", {}) or {}
                raise RuntimeError(
                    f"DataSync task execution {task_execution_arn} failed with status "
                    f"{status!r}: ErrorCode={result.get('ErrorCode')!r}, "
                    f"ErrorDetail={result.get('ErrorDetail')!r}"
                )
            if status in _DATASYNC_TERMINAL_STATES:
                # Defensive: any other terminal state should abort validation.
                raise RuntimeError(
                    f"DataSync task execution {task_execution_arn} ended in "
                    f"non-success terminal state {status!r}."
                )
            if now() >= deadline:
                raise RuntimeError(
                    f"Timed out after {timeout_seconds:.0f}s waiting for DataSync "
                    f"task execution {task_execution_arn}; last status={status!r}."
                )
            sleep(poll_interval_seconds)

    def start_and_validate(
        self,
        datasync_client: Any,
        source_s3_client: Any,
        destination_s3_client: Any,
        objects: Iterable[dict[str, str]],
        task_arn: str | None = None,
        *,
        poll_interval_seconds: float = 15.0,
        timeout_seconds: float = 6 * 60 * 60,
    ) -> dict[str, Any]:
        """Start a task and run explicit post-transfer S3 checksum validation.

        ``StartTaskExecution`` is asynchronous: it returns a ``TaskExecutionArn``
        as soon as the execution is queued, long before any object has been
        copied. The post-transfer integrity gate below must therefore wait for
        the execution to reach the ``SUCCESS`` terminal state before running
        ``validate_checksums`` -- otherwise it inspects destination state the
        transfer has not written yet, producing either a 404 error or a
        false-positive match against a stale/pre-planted object.
        """
        resolved_task_arn = task_arn or self.create_task(datasync_client)
        execution = datasync_client.start_task_execution(TaskArn=resolved_task_arn)
        task_execution_arn = execution["TaskExecutionArn"]
        self.wait_for_execution(
            datasync_client,
            task_execution_arn,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds,
        )
        checksums = self.validate_checksums(source_s3_client, destination_s3_client, objects)
        return {
            "task_arn": resolved_task_arn,
            "task_execution_arn": task_execution_arn,
            "checksums": checksums,
        }

"""S3 upload Lambda entry point."""

from .handlers import s3_trigger_handler as handler

__all__ = ["handler"]

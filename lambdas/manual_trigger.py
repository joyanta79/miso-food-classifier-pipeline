"""Manual invocation Lambda entry point."""

from .handlers import manual_trigger_handler as handler

__all__ = ["handler"]

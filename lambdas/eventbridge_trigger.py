"""EventBridge Lambda entry point."""

from .handlers import eventbridge_trigger_handler as handler

__all__ = ["handler"]

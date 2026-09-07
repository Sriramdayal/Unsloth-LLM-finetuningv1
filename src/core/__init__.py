"""Backward-compatible re-exports for src.core."""

from ..platform.factory import ModelFactory
from ..platform.model_runner import ModelRunner

__all__ = ["ModelFactory", "ModelRunner"]

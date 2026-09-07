"""Backward-compatible re-export shim.

Moved to :mod:`src.platform.model_runner`. This module is kept so existing
imports (``from src.core.model_runner import ModelRunner``) continue to work.
"""

from ..platform.model_runner import ModelRunner, _best_accel_label

__all__ = ["ModelRunner", "_best_accel_label"]

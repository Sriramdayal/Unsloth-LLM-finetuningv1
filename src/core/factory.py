"""Backward-compatible re-export shim.

Moved to :mod:`src.platform.factory`. This module is kept so existing
imports (``from src.core.factory import ModelFactory``) continue to work.
"""

from ..platform.factory import (
    ModelFactory,
    _best_device,
    _bnb_config_4bit,
    _compute_dtype,
    _has_cuda,
    _has_mps,
    _has_unsloth,
    _platform,
)

__all__ = [
    "ModelFactory",
    "_platform",
    "_has_cuda",
    "_has_mps",
    "_has_unsloth",
    "_best_device",
    "_compute_dtype",
    "_bnb_config_4bit",
]

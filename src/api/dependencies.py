"""Backward-compatible re-export shim.

Moved to :mod:`src.service.dependencies`.
"""

from src.service.dependencies import (  # noqa: F401
    JobRegistry,
    ModelCache,
    job_registry,
    model_cache,
)

__all__ = ["JobRegistry", "ModelCache", "job_registry", "model_cache"]

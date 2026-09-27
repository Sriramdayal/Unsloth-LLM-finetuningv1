"""Backward-compatible re-export shim.

Moved to :mod:`src.service.schemas`.
"""

from src.service.schemas import (  # noqa: F401
    InferenceRequest,
    JobStatusResponse,
    TrainingRequest,
)

__all__ = ["InferenceRequest", "JobStatusResponse", "TrainingRequest"]

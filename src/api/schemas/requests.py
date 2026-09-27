"""Backward-compatible re-export shim.

Moved to :mod:`src.service.schemas.requests`.
"""

from src.service.schemas.requests import (  # noqa: F401
    InferenceRequest,
    JobStatusResponse,
    TrainingRequest,
)

__all__ = ["InferenceRequest", "JobStatusResponse", "TrainingRequest"]

"""Backward-compatible re-export shim.

Moved to :mod:`src.service.routes.training`.
"""

from src.service.routes.training import (  # noqa: F401
    _run_training_job,
    get_status,
    router,
    start_training,
)

__all__ = ["_run_training_job", "get_status", "router", "start_training"]

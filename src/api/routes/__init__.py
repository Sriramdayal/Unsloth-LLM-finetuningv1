"""Backward-compatible re-export shim.

Moved to :mod:`src.service.routes`.
"""

from src.service.routes import health, inference, training  # noqa: F401

__all__ = ["health", "inference", "training"]

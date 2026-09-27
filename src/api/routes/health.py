"""Backward-compatible re-export shim.

Moved to :mod:`src.service.routes.health`.
"""

from src.service.routes.health import health_check, router  # noqa: F401

__all__ = ["health_check", "router"]

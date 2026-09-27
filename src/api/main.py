"""Backward-compatible re-export shim.

Moved to :mod:`src.service.app`. This module is kept so existing
imports (``from src.api.main import create_app``) and entry points
(``uvicorn src.api.main:app``) continue to work.
"""

from src.service.app import app, create_app  # noqa: F401

__all__ = ["app", "create_app"]

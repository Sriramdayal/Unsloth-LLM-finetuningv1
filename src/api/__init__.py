"""Backward-compatible re-export shim.

Moved to :mod:`src.service`. This package is kept so existing
imports (``from src.api.main import create_app``) continue to work.
"""

from src.service.app import app  # noqa: F401

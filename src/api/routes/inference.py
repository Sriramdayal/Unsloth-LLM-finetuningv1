"""Backward-compatible re-export shim.

Moved to :mod:`src.service.routes.inference`.
"""

from src.service.routes.inference import infer, infer_gguf, router  # noqa: F401

__all__ = ["infer", "infer_gguf", "router"]

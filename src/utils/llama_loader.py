"""Backward-compatible re-export shim.

Moved to :mod:`src.platform.dll_bootstrap`. This module is kept so existing
imports (``from src.utils.llama_loader import bootstrap_platform_dlls``)
continue to work.
"""

from ..platform.dll_bootstrap import (
    bootstrap_platform_dlls,
    bootstrap_windows_cuda_dlls,
)

__all__ = ["bootstrap_platform_dlls", "bootstrap_windows_cuda_dlls"]

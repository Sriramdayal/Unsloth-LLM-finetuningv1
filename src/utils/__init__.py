"""Backward-compatible re-exports for src.utils."""

from ..platform.dll_bootstrap import bootstrap_platform_dlls
from ..platform.hardware import HardwareManager

__all__ = ["HardwareManager", "bootstrap_platform_dlls"]

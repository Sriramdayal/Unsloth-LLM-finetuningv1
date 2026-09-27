"""Backward-compatible re-export shim.

Moved to :mod:`src.platform.hardware`. This module is kept so existing
imports (``from src.utils.env import HardwareManager``) continue to work.
"""

from ..platform.hardware import HardwareManager

__all__ = ["HardwareManager"]

"""Cross-platform backend layer: hardware detection, model loading, inference."""

from .dll_bootstrap import bootstrap_platform_dlls, bootstrap_windows_cuda_dlls
from .factory import ModelFactory
from .hardware import HardwareManager
from .model_runner import ModelRunner

__all__ = [
    "HardwareManager",
    "ModelFactory",
    "ModelRunner",
    "bootstrap_platform_dlls",
    "bootstrap_windows_cuda_dlls",
]

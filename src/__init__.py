"""
Unsloth Finetuning - Enterprise Pipeline for LLM Fine-Tuning.

Public API surface. Import core components from here.
"""

__version__ = "0.3.0"

from .config import ModelConfig, TrainConfig
from .data import DataProcessor
from .platform import HardwareManager, ModelFactory, ModelRunner
from .training import train_model, train_mlx

try:
    from .soup import SoupClient, SoupConfig, SoupNotAvailableError
except ImportError:
    pass  # soup package has no heavy deps; this is a safety net

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "DataProcessor",
    "train_model",
    "train_mlx",
    "ModelRunner",
    "ModelFactory",
    "HardwareManager",
    "SoupClient",
    "SoupConfig",
    "SoupNotAvailableError",
    "__version__",
]

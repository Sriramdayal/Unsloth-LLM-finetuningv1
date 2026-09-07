"""Backward-compatible re-export shim.

Moved to :mod:`src.training.orchestrator`. This module is kept so existing
imports (``from src.train import train_model``) continue to work.
"""

from .training.mlx_trainer import train_mlx
from .training.orchestrator import train_model

__all__ = ["train_model", "train_mlx"]

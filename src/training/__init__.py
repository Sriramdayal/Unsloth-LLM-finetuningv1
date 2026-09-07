"""Training orchestration layer: SFTTrainer loop + MLX backend."""

from .mlx_trainer import train_mlx
from .orchestrator import train_model

__all__ = ["train_model", "train_mlx"]

"""Apple MLX training backend (macOS Apple Silicon only).

Exports the HuggingFace dataset to JSONL and shells out to
``python -m mlx_lm.lora`` as a subprocess.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Any, Tuple

try:
    from ..config import ModelConfig, TrainConfig
except ImportError:
    from src.config import ModelConfig, TrainConfig

logger = logging.getLogger(__name__)


def train_mlx(
    dataset, train_config: TrainConfig, model_config: ModelConfig
) -> Tuple[Any, str]:
    """Train using Apple MLX via mlx_lm.lora."""
    logger.info("Exporting dataset to JSONL for MLX...")

    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = os.path.join(tmpdir, "train.jsonl")
        valid_path = os.path.join(tmpdir, "valid.jsonl")

        dataset.to_json(train_path)

        # Simple validation split for MLX requirement
        val_size = min(10, len(dataset))
        if val_size > 0:
            dataset.select(range(val_size)).to_json(valid_path)

        # Calculate iterations if max_steps is not explicitly set
        if train_config.max_steps > 0:
            iters = train_config.max_steps
        else:
            iters = int((len(dataset) / train_config.batch_size) * train_config.num_train_epochs)
            iters = max(10, iters)  # ensure at least some iterations

        cmd = [
            "python", "-m", "mlx_lm.lora",
            "--model", model_config.model_name_or_path,
            "--train",
            "--data", tmpdir,
            "--iters", str(iters),
            "--batch-size", str(train_config.batch_size),
            "--learning-rate", str(train_config.learning_rate),
            "--adapter-path", train_config.output_dir,
        ]

        logger.info(f"Running MLX training command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            logger.info("MLX training completed successfully.")
            return {"status": "success", "backend": "mlx"}, train_config.output_dir
        except subprocess.CalledProcessError as e:
            logger.error(f"MLX training failed: {e}")
            raise RuntimeError(f"MLX training failed: {e}") from e


# Backward-compatible alias for the pre-modularization private name.
_train_mlx = train_mlx

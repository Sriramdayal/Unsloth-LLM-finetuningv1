"""Soup configuration mapping.

Maps this repo's :class:`ModelConfig` / :class:`TrainConfig` fields to the
``soup.yaml`` format consumed by the `soup` CLI (hybrid integration: Soup is
invoked as a subprocess, not imported).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SoupConfig:
    """Minimal soup.yaml configuration.

    Field mapping from this repo's config:
      model_name_or_path → base
      load_in_4bit       → training.quantization ("4bit" / "none")
      lora_r / lora_alpha → training.lora.{r, alpha}
      dataset_name       → data.train
      dataset_style      → data.format
      learning_rate      → training.lr
      num_train_epochs   → training.epochs
    """

    base_model: str = ""
    task: str = "sft"
    backend: str = "unsloth"
    data_train: str = ""
    data_format: str = "alpaca"
    data_max_length: int = 2048
    training_epochs: float = 3
    training_lr: float = 2e-5
    training_batch_size: Any = "auto"
    training_quantization: str = "4bit"
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=list)
    output: str = "./output"

    def to_dict(self) -> Dict[str, Any]:
        """Return the soup.yaml structure as a plain dict."""
        lora: Dict[str, Any] = {"r": self.lora_r, "alpha": self.lora_alpha}
        if self.lora_dropout:
            lora["dropout"] = self.lora_dropout
        if self.lora_target_modules:
            lora["target_modules"] = list(self.lora_target_modules)
        return {
            "base": self.base_model,
            "task": self.task,
            "backend": self.backend,
            "data": {
                "train": self.data_train,
                "format": self.data_format,
                "max_length": self.data_max_length,
            },
            "training": {
                "epochs": self.training_epochs,
                "lr": self.training_lr,
                "batch_size": self.training_batch_size,
                "quantization": self.training_quantization,
                "lora": lora,
            },
            "output": self.output,
        }

    def to_yaml(self, output_path: str) -> str:
        """Write this config to ``output_path`` as soup.yaml. Returns the path."""
        import yaml

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        return output_path

    @classmethod
    def from_model_train_configs(cls, model_config, train_config) -> "SoupConfig":
        """Build a SoupConfig from this repo's ModelConfig + TrainConfig.

        ``target_modules`` are passed through only when customized — otherwise
        they are omitted so Soup auto-detects them per architecture.
        """
        targets = list(getattr(model_config, "target_modules", None) or [])
        try:
            from ..config import ModelConfig as _ModelConfig
        except ImportError:
            try:
                from src.config import ModelConfig as _ModelConfig
            except ImportError:
                _ModelConfig = None
        if _ModelConfig is not None:
            try:
                if targets == _ModelConfig(model_name_or_path="").target_modules:
                    targets = []
            except Exception:
                pass
        return cls(
            base_model=getattr(model_config, "model_name_or_path", ""),
            data_train=getattr(train_config, "dataset_name", ""),
            data_format=getattr(train_config, "dataset_style", "alpaca") or "alpaca",
            data_max_length=getattr(model_config, "max_seq_length", 2048),
            training_epochs=getattr(train_config, "num_train_epochs", 3),
            training_lr=getattr(train_config, "learning_rate", 2e-5),
            training_batch_size=getattr(train_config, "batch_size", "auto"),
            training_quantization=(
                "4bit" if getattr(model_config, "load_in_4bit", True) else "none"
            ),
            lora_r=getattr(model_config, "lora_r", 16),
            lora_alpha=getattr(model_config, "lora_alpha", 16),
            lora_dropout=getattr(model_config, "lora_dropout", 0.0),
            lora_target_modules=targets,
            output=getattr(train_config, "output_dir", "./output"),
        )

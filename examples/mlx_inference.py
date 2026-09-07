"""Inference with the Apple MLX backend (macOS Apple Silicon only).

Usage:
    pip install -e ".[macos]"
    python examples/mlx_inference.py
"""

import sys

from src import ModelConfig, ModelRunner
from src.platform import HardwareManager


def main():
    if sys.platform != "darwin" or not HardwareManager.use_mlx():
        raise SystemExit("MLX backend requires macOS Apple Silicon with mlx-lm installed.")

    config = ModelConfig(model_name_or_path="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    runner = ModelRunner(config)
    # Pass adapter_path="outputs/..." to load MLX LoRA adapters
    runner.setup_for_inference()

    prompt = "Write a haiku about the sea."
    print(f"Prompt: {prompt}")
    print(f"Response: {runner.generate(prompt, max_new_tokens=128)}")


if __name__ == "__main__":
    main()

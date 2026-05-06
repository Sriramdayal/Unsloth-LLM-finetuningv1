"""
Hardware detection and environment utilities — cross-platform.

Supports:
  • Windows  NVIDIA   — CUDA
  • Linux    NVIDIA   — CUDA
  • macOS    Silicon  — Apple MPS
  • macOS    Intel    — CPU
  • Any      CPU      — PyTorch CPU
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from typing import Dict, Union

import torch

logger = logging.getLogger(__name__)


class HardwareManager:
    """
    Centralized utility for hardware detection, memory management,
    and environment validation. Works on Windows, Linux, and macOS.
    """

    @staticmethod
    def get_device() -> str:
        """Returns the best available compute device: cuda | mps | cpu."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def get_torch_dtype() -> torch.dtype:
        """Returns the optimal float dtype for the active accelerator."""
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.float16  # MPS supports float16; bfloat16 is partial
        return torch.float32

    @staticmethod
    def get_memory_stats(device_index: int = 0) -> Dict[str, Union[str, float]]:
        """
        Returns memory/VRAM stats for the active accelerator.

        Returns:
            Dict with keys: status | device | total_gb | reserved_gb |
                            allocated_gb | free_gb | unified_memory (MPS only)
        """
        device = HardwareManager.get_device()

        if device == "cuda":
            gpu_stats = torch.cuda.get_device_properties(device_index)
            reserved = torch.cuda.memory_reserved(device_index)
            allocated = torch.cuda.memory_allocated(device_index)
            return {
                "device": gpu_stats.name,
                "total_gb": round(gpu_stats.total_memory / 1024**3, 2),
                "reserved_gb": round(reserved / 1024**3, 2),
                "allocated_gb": round(allocated / 1024**3, 2),
                "free_gb": round((reserved - allocated) / 1024**3, 2),
            }

        if device == "mps":
            # MPS uses unified memory shared with CPU — report system RAM as proxy
            try:
                import subprocess, json

                result = subprocess.run(
                    ["system_profiler", "SPHardwareDataType", "-json"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                data = json.loads(result.stdout)
                ram_bytes = (
                    int(
                        data["SPHardwareDataType"][0]
                        .get("physical_memory", "0 GB")
                        .replace(" GB", "")
                        .strip()
                    )
                    * 1024**3
                )
                return {
                    "device": "Apple Silicon MPS",
                    "total_gb": round(ram_bytes / 1024**3, 2),
                    "unified_memory": True,
                    "note": "MPS uses unified memory shared with CPU",
                }
            except Exception:
                return {"device": "Apple Silicon MPS", "unified_memory": True}

        return {"status": "cpu_only"}

    @staticmethod
    def log_system_report() -> None:
        """Logs a full environment summary for debugging."""
        logger.info("--- System Hardware Report ---")
        logger.info(f"OS:      {platform.system()} {platform.release()}")
        logger.info(f"Python:  {platform.python_version()}")
        logger.info(f"PyTorch: {torch.__version__}")

        device = HardwareManager.get_device()
        logger.info(f"Primary Device: {device.upper()}")

        if device == "cuda":
            stats = HardwareManager.get_memory_stats()
            logger.info(f"GPU: {stats['device']} ({stats['total_gb']} GB VRAM)")

        elif device == "mps":
            stats = HardwareManager.get_memory_stats()
            logger.info(
                f"GPU: {stats.get('device', 'Apple MPS')}  "
                f"({stats.get('total_gb', '?')} GB unified memory)"
            )

        logger.info(f"Platform backend: {HardwareManager.get_backend_label()}")
        logger.info("------------------------------")

    @staticmethod
    def get_backend_label() -> str:
        """
        Returns a human-readable label describing the active training/inference stack.
        """
        device = HardwareManager.get_device()
        os_name = sys.platform

        if device == "cuda":
            try:
                import importlib.util

                if importlib.util.find_spec("unsloth") and os_name == "linux":
                    return "Linux + CUDA + Unsloth (Triton kernels)"
            except Exception:
                pass
            label = {
                "win32": "Windows + CUDA (bitsandbytes QLoRA + llama.cpp CUBLAS)",
                "linux": "Linux + CUDA (bitsandbytes QLoRA + llama.cpp CUDA)",
            }.get(os_name, f"{os_name} + CUDA")
            return label

        if device == "mps":
            return "macOS + Apple Silicon MPS (transformers float16 + llama.cpp Metal)"

        return "CPU (transformers float32 + llama.cpp CPU)"

    @staticmethod
    def is_unsloth_compatible() -> bool:
        """
        Returns True if the environment fully supports Unsloth Triton kernels.
        Requires: Linux or WSL2 + NVIDIA CUDA GPU + unsloth installed.
        """
        os_name = sys.platform

        if os_name == "win32":
            if os.environ.get("WSL_DISTRO_NAME"):
                # Running inside WSL2 — treated as Linux
                pass
            else:
                logger.info(
                    "Native Windows: Unsloth Triton kernels not available. "
                    "Using bitsandbytes QLoRA + llama.cpp CUBLAS instead."
                )
                return False

        if os_name == "darwin":
            logger.info(
                "macOS: Unsloth Triton kernels not available. "
                "Using MPS (Apple Silicon) or CPU backend instead."
            )
            return False

        return torch.cuda.is_available()

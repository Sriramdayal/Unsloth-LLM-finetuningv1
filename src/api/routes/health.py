"""
Health check endpoint.

Reports GPU availability, hardware backend, and system platform information.

Heavy imports (torch, HardwareManager) are deferred to the endpoint handler
so that importing this module does not trigger a full dependency chain.
"""

import platform as platform_mod

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["system"])
def health_check():
    """
    System health check — returns hardware and runtime info.

    Returns GPU availability, device name, OS platform, and the active
    training/inference backend label (Unsloth / PEFT QLoRA / MPS / CPU).
    """
    import torch

    from src.utils.env import HardwareManager

    gpu_available = torch.cuda.is_available()

    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_name": torch.cuda.get_device_name(0) if gpu_available else None,
        "platform": platform_mod.system(),
        "backend": HardwareManager.get_backend_label(),
    }

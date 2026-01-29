import torch
import os
import platform
import logging

logger = logging.getLogger(__name__)

class HardwareManager:
    """
    Centralized utility for hardware detection, memory management, and environment checks.
    """
    
    @staticmethod
    def get_device():
        """Returns the best available device (cuda, mps, cpu)."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Note: Unsloth doesn't support MPS well yet, but good for general torch logic
            return "mps"
        return "cpu"

    @staticmethod
    def get_torch_dtype():
        """Returns the optimal float type for the current hardware."""
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    @staticmethod
    def get_memory_stats(device_index=0):
        """Returns VRAM usage stats if CUDA is available."""
        if not torch.cuda.is_available():
            return {"status": "cpu_only"}
        
        gpu_stats = torch.cuda.get_device_properties(device_index)
        reserved = torch.cuda.memory_reserved(device_index)
        allocated = torch.cuda.memory_allocated(device_index)
        free = reserved - allocated
        
        return {
            "device": gpu_stats.name,
            "total_gb": round(gpu_stats.total_memory / 1024**3, 2),
            "reserved_gb": round(reserved / 1024**3, 2),
            "allocated_gb": round(allocated / 1024**3, 2),
            "free_gb": round(free / 1024**3, 2),
        }

    @staticmethod
    def log_system_report():
        """Logs a summary of the environment for debugging."""
        logger.info("--- System Hardware Report ---")
        logger.info(f"OS: {platform.system()} {platform.release()}")
        logger.info(f"Python: {platform.python_version()}")
        logger.info(f"PyTorch: {torch.__version__}")
        
        device = HardwareManager.get_device()
        logger.info(f"Primary Device: {device}")
        
        if device == "cuda":
            stats = HardwareManager.get_memory_stats()
            logger.info(f"GPU: {stats['device']} ({stats['total_gb']} GB)")
        logger.info("------------------------------")

    @staticmethod
    def is_unsloth_compatible():
        """Checks if the environment meets Unsloth requirements (Linux or MacOS w/ specific setups)."""
        # Unsloth primarily targets Linux/WSL2 with NVIDIA GPUs
        if platform.system() == "Windows" and "WSL" not in os.environ.get("WSL_DISTRO_NAME", ""):
             # Basic windows might work with some specific wheels but typically we want WSL
             return False
        return torch.cuda.is_available()

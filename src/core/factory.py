"""
Model Factory — Cross-Platform GPU Backend.

Automatically selects the optimal training/inference stack per OS:

  ┌─────────────┬────────────────────────────────────────────────────────┐
  │ Platform    │ Training Backend                                       │
  ├─────────────┼────────────────────────────────────────────────────────┤
  │ Linux/WSL2  │ Unsloth FastLanguageModel (Triton, 2× speed)          │
  │  (NVIDIA)   │   → falls back to bitsandbytes 4-bit QLoRA + PEFT     │
  ├─────────────┼────────────────────────────────────────────────────────┤
  │ Windows     │ bitsandbytes 4-bit NF4 QLoRA + PEFT LoRA              │
  │  (NVIDIA)   │   (no Triton required, pure CUDA)                     │
  ├─────────────┼────────────────────────────────────────────────────────┤
  │ macOS       │ transformers + MPS (Apple Silicon) or CPU (Intel)     │
  │  (Apple)    │   float16 on MPS, float32 on CPU                      │
  │             │   Note: bitsandbytes 4-bit not supported on MPS       │
  └─────────────┴────────────────────────────────────────────────────────┘

Inference path (all platforms):
  • .gguf file  → llama-cpp-python (CUDA/Metal/CPU — auto-detected)
  • HF repo/dir → transformers generate()
"""

from __future__ import annotations

import logging
import platform
import sys
from typing import Optional, Tuple

import torch

from ..config import ModelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Platform detection helpers
# ---------------------------------------------------------------------------

def _platform() -> str:
    """Return normalized platform string: 'linux' | 'windows' | 'darwin'."""
    return sys.platform  # 'linux', 'win32', 'darwin'


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _has_mps() -> bool:
    return (
        _platform() == "darwin"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def _has_unsloth() -> bool:
    """Return True if unsloth is installed (Linux / WSL2 only)."""
    try:
        import importlib.util
        return importlib.util.find_spec("unsloth") is not None
    except Exception:
        return False


def _best_device() -> str:
    if _has_cuda():
        return "cuda"
    if _has_mps():
        return "mps"
    return "cpu"


def _compute_dtype() -> torch.dtype:
    """Return the best float dtype for the active accelerator."""
    if _has_cuda():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if _has_mps():
        return torch.float16   # MPS supports float16; bfloat16 support is partial
    return torch.float32


# ---------------------------------------------------------------------------
# Backend-specific helpers
# ---------------------------------------------------------------------------

def _bnb_config_4bit():
    """Build BitsAndBytesConfig for 4-bit NF4 QLoRA (CUDA only)."""
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_compute_dtype(),
        bnb_4bit_use_double_quant=True,   # Nested quantisation saves ~0.4 GB
    )


def _load_with_unsloth(config: ModelConfig) -> Tuple:
    """
    Load model via Unsloth's FastLanguageModel (Linux/WSL2, Triton path).
    Provides ~2× training speed and 70% less VRAM vs. stock HuggingFace.
    """
    from unsloth import FastLanguageModel  # type: ignore[import]

    logger.info("Factory [Unsloth]: Loading model with Triton-optimised kernels ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name_or_path,
        max_seq_length=config.max_seq_length,
        dtype=None,              # Auto-detect (bfloat16 on Ampere+)
        load_in_4bit=config.load_in_4bit,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Factory [Unsloth]: Model loaded successfully.")
    return model, tokenizer


def _apply_lora_unsloth(model, config: ModelConfig):
    """Apply LoRA via Unsloth's optimised get_peft_model (Linux only)."""
    from unsloth import FastLanguageModel  # type: ignore[import]

    logger.info(
        f"Factory [Unsloth]: Applying LoRA  r={config.lora_r}  "
        f"alpha={config.lora_alpha}  targets={config.target_modules}"
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth gradient checkpointing
        random_state=config.random_state,
        use_rslora=False,
        loftq_config=None,
    )
    logger.info("Factory [Unsloth]: LoRA applied successfully.")
    return model


def _load_with_transformers(config: ModelConfig) -> Tuple:
    """
    Load model via HuggingFace Transformers.
    Supports CUDA (with bitsandbytes 4-bit), MPS (Apple Silicon), and CPU.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    platform_str = _platform()
    device = _best_device()
    use_4bit = config.load_in_4bit and _has_cuda()  # 4-bit only on CUDA

    logger.info(
        f"Factory [HF]: Loading '{config.model_name_or_path}'  "
        f"device={device}  load_in_4bit={use_4bit}"
    )

    if use_4bit and not _has_cuda():
        logger.warning(
            "load_in_4bit=True requested but CUDA is not available. "
            "Falling back to full precision."
        )

    bnb_cfg   = _bnb_config_4bit() if use_4bit else None
    # When using bitsandbytes, dtype is managed internally
    dtype     = None if use_4bit else _compute_dtype()
    # device_map="auto" handles CPU/CUDA/MPS distribution
    device_map = "auto" if device != "mps" else {"": "mps"}

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if bnb_cfg is not None:
        kwargs["quantization_config"] = bnb_cfg

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **kwargs
    )

    logger.info("Factory [HF]: Model + tokenizer loaded successfully.")
    return model, tokenizer


def _apply_lora_peft(model, config: ModelConfig):
    """Apply LoRA via stock PEFT (Windows / macOS / Linux fallback)."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    logger.info(
        f"Factory [PEFT]: Applying LoRA  r={config.lora_r}  "
        f"alpha={config.lora_alpha}  targets={config.target_modules}"
    )

    use_4bit = config.load_in_4bit and _has_cuda()
    if use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    logger.info("Factory [PEFT]: LoRA applied successfully.")
    return model


# ---------------------------------------------------------------------------
# Public Factory
# ---------------------------------------------------------------------------

class ModelFactory:
    """
    Cross-platform factory for loading and configuring LLMs.

    Automatically selects the best available backend:
      • Linux  + NVIDIA + unsloth installed → Unsloth (Triton, fastest)
      • Linux  + NVIDIA + no unsloth        → bitsandbytes 4-bit QLoRA + PEFT
      • Windows + NVIDIA                    → bitsandbytes 4-bit QLoRA + PEFT
      • macOS  + Apple Silicon (MPS)        → transformers float16 + PEFT
      • macOS  + Intel / any CPU            → transformers float32 + PEFT
    """

    @staticmethod
    def create_model_and_tokenizer(config: ModelConfig) -> Tuple:
        """
        Loads the base model + tokenizer using the best available backend.

        Returns:
            Tuple of (model, tokenizer)
        """
        plat = _platform()
        try:
            # ── Linux/WSL2: try Unsloth first ────────────────────────────────
            if plat == "linux" and _has_cuda() and _has_unsloth():
                logger.info("Platform: Linux + CUDA + Unsloth → using Triton-optimised path")
                return _load_with_unsloth(config)

            # ── All other platforms: transformers + optional bitsandbytes ─────
            if plat == "linux" and _has_cuda():
                logger.info("Platform: Linux + CUDA (no Unsloth) → bitsandbytes 4-bit QLoRA")
            elif plat == "win32":
                logger.info("Platform: Windows + CUDA → bitsandbytes 4-bit QLoRA (no Triton)")
            elif plat == "darwin":
                device = "MPS (Apple Silicon)" if _has_mps() else "CPU (Intel)"
                logger.info(f"Platform: macOS → transformers {device}")
            else:
                logger.info(f"Platform: {plat} / CPU → transformers float32")

            return _load_with_transformers(config)

        except Exception as e:
            logger.error(f"Factory: Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    @staticmethod
    def apply_lora(model, config: ModelConfig):
        """
        Applies LoRA adapters using the best available backend.

        Returns:
            PeftModel with LoRA adapters applied
        """
        try:
            # Unsloth LoRA: Linux + CUDA + unsloth installed
            if _platform() == "linux" and _has_cuda() and _has_unsloth():
                return _apply_lora_unsloth(model, config)

            # Standard PEFT: Windows / macOS / Linux fallback
            return _apply_lora_peft(model, config)

        except Exception as e:
            logger.error(f"Factory: Failed to apply LoRA: {e}")
            raise RuntimeError(f"LoRA application failed: {e}") from e

    @staticmethod
    def prepare_for_inference(model):
        """
        Switches model to eval / inference mode (all platforms).

        Returns:
            Model in eval mode
        """
        # Unsloth fast inference mode (Linux only)
        if _platform() == "linux" and _has_cuda() and _has_unsloth():
            try:
                from unsloth import FastLanguageModel  # type: ignore[import]
                FastLanguageModel.for_inference(model)
                logger.info("Factory [Unsloth]: Switched to fast inference mode.")
            except Exception:
                pass  # Fall through to standard eval

        model.eval()
        logger.info("Factory: Model in eval/inference mode.")
        return model

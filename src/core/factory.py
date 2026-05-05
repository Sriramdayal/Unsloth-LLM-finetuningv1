"""
Model Factory — Windows-native CUDA backend.

Uses HuggingFace Transformers + BitsAndBytes (4-bit QLoRA) + PEFT instead of
the Linux-only Unsloth/Triton stack. Fully GPU-accelerated on NVIDIA via CUDA.

Training path : transformers + bitsandbytes + peft  (QLoRA, fp16/bf16)
Inference path: llama-cpp-python (GGUF, CUBLAS)  OR  transformers (safetensors)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from ..config import ModelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bnb_config(load_in_4bit: bool):
    """Build a BitsAndBytesConfig for 4-bit QLoRA when requested."""
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as e:
        raise ImportError("transformers is required. Run: uv pip install transformers") from e

    if not load_in_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",         # NF4 — best accuracy for QLoRA
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported()
                               else torch.float16,
        bnb_4bit_use_double_quant=True,    # Nested quantisation saves ~0.4 GB extra
    )


def _dtype() -> Optional[torch.dtype]:
    """Return the best floating-point dtype for the current GPU."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# ---------------------------------------------------------------------------
# Public Factory
# ---------------------------------------------------------------------------

class ModelFactory:
    """
    Windows-native factory for loading and configuring LLMs.

    Training  → transformers AutoModelForCausalLM + BitsAndBytes 4-bit + PEFT LoRA
    Inference → llama-cpp-python (GGUF / CUBLAS) or transformers (safetensors)
    """

    @staticmethod
    def create_model_and_tokenizer(
        config: ModelConfig,
    ) -> Tuple:
        """
        Loads the base model + tokenizer via HuggingFace Transformers.
        Applies 4-bit quantisation with BitsAndBytes when config.load_in_4bit=True.

        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Factory: Loading model '{config.model_name_or_path}' ...")
        logger.info(f"  load_in_4bit={config.load_in_4bit}  |  max_seq_length={config.max_seq_length}")

        bnb_cfg = _bnb_config(config.load_in_4bit)
        dtype   = None if config.load_in_4bit else _dtype()  # BnB manages dtype internally

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name_or_path,
                trust_remote_code=True,
            )
            # Some tokenizers lack a pad token — fall back to eos
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                quantization_config=bnb_cfg,
                torch_dtype=dtype,
                device_map="auto",           # Automatically spreads across GPU(s)
                trust_remote_code=True,
            )

            logger.info("Factory: Model + tokenizer loaded successfully.")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Factory: Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    @staticmethod
    def apply_lora(model, config: ModelConfig):
        """
        Applies LoRA adapters via PEFT.
        Equivalent to Unsloth's get_peft_model() but uses stock PEFT — runs on Windows.

        Returns:
            PeftModel with LoRA adapters applied
        """
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        logger.info(
            f"Factory: Applying LoRA  r={config.lora_r}  alpha={config.lora_alpha}  "
            f"targets={config.target_modules}"
        )

        try:
            # Required before LoRA when using bitsandbytes 4-bit
            if config.load_in_4bit:
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
            logger.info("Factory: LoRA adapters applied successfully.")
            return model

        except Exception as e:
            logger.error(f"Factory: Failed to apply LoRA: {e}")
            raise RuntimeError(f"LoRA application failed: {e}") from e

    @staticmethod
    def prepare_for_inference(model):
        """
        Switches a transformers model to eval / inference mode.

        Returns:
            Model in eval mode
        """
        model.eval()
        logger.info("Factory: Model switched to inference mode.")
        return model

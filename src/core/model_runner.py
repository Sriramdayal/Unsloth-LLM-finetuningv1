"""
ModelRunner — dual-path inference for Windows-native CUDA.

  Path A (GGUF)  : llama-cpp-python with CUBLAS backend.
                   Pass a .gguf file path as model_name_or_path.
                   GPU-accelerated via llama.cpp precompiled CUDA binaries.
                   No Triton, no WSL required.

  Path B (HF)    : transformers AutoModelForCausalLM (safetensors / LoRA adapter).
                   Used automatically when model_name_or_path is a HF repo ID
                   or a directory containing safetensors files.

Training always uses Path B (transformers + bitsandbytes 4-bit + PEFT).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from .factory import ModelFactory
from ..config import ModelConfig

logger = logging.getLogger(__name__)

# GPU layers to offload in llama.cpp  (-1 = all layers on GPU)
_LLAMA_N_GPU_LAYERS = int(os.environ.get("LLAMA_N_GPU_LAYERS", "-1"))


def _is_gguf(path: str) -> bool:
    """Return True when the path points to a .gguf file."""
    return path.strip().lower().endswith(".gguf")


def _load_llama_cpp(model_path: str, n_ctx: int = 4096):
    """
    Load a GGUF model with llama-cpp-python using the CUBLAS GPU backend.

    On Windows, bootstraps DLL search paths so that llama.dll can find
    cudart64_12.dll from PyTorch's bundled CUDA runtime (no CUDA Toolkit needed).

    Returns:
        llama_cpp.Llama instance
    """
    # ── Windows DLL bootstrap (must happen BEFORE import llama_cpp) ──────────
    from ..utils.llama_loader import bootstrap_windows_cuda_dlls
    bootstrap_windows_cuda_dlls()

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is not installed.\n"
            "Install the CUBLAS build for GPU acceleration:\n"
            "  uv pip install llama-cpp-python "
            "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124"
        ) from exc

    logger.info(f"ModelRunner [GGUF/CUBLAS]: Loading '{model_path}' ...")
    logger.info(f"  n_gpu_layers={_LLAMA_N_GPU_LAYERS}  n_ctx={n_ctx}")

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=_LLAMA_N_GPU_LAYERS,  # -1 = put ALL layers on GPU
        n_ctx=n_ctx,
        verbose=False,
    )
    logger.info("ModelRunner [GGUF/CUBLAS]: Model loaded successfully.")
    return llm


class ModelRunner:
    """
    High-level API for model loading, generation, and lifecycle management.

    Supports two backends transparently:
      * GGUF  → llama-cpp-python (CUBLAS, Windows-native GPU acceleration)
      * HF    → transformers + bitsandbytes + PEFT
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._backend: str = "hf"          # "gguf" | "hf"
        self.is_training_ready: bool = False

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def setup_for_training(self):
        """
        Prepares model + tokenizer for supervised fine-tuning (HF backend).

        Returns:
            Tuple of (model, tokenizer)
        """
        self.model, self.tokenizer = ModelFactory.create_model_and_tokenizer(self.config)
        self.model = ModelFactory.apply_lora(self.model, self.config)
        self._backend = "hf"
        self.is_training_ready = True
        logger.info("ModelRunner: Ready for training (HF/PEFT backend).")
        return self.model, self.tokenizer

    def setup_for_inference(self, adapter_path: Optional[str] = None):
        """
        Loads a model for inference.

        Automatically selects backend:
          • .gguf path → llama-cpp-python (CUBLAS GPU)
          • HF repo / safetensors dir → transformers

        Args:
            adapter_path: Optional path to a LoRA adapter directory (HF backend only).

        Returns:
            self  (for chaining; model/tokenizer stored on self)
        """
        model_path = self.config.model_name_or_path

        if _is_gguf(model_path):
            # ── GGUF / llama.cpp path ─────────────────────────────────
            self._backend = "gguf"
            self.model = _load_llama_cpp(model_path, n_ctx=self.config.max_seq_length)
            self.tokenizer = None   # llama.cpp handles tokenisation internally
        else:
            # ── HF / transformers path ────────────────────────────────
            self._backend = "hf"
            if self.model is None:
                self.model, self.tokenizer = ModelFactory.create_model_and_tokenizer(self.config)

            if adapter_path:
                from peft import PeftModel
                logger.info(f"Loading LoRA adapter from: {adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, adapter_path)

            self.model = ModelFactory.prepare_for_inference(self.model)

        logger.info(f"ModelRunner: Ready for inference (backend={self._backend}).")
        return self

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text from a prompt.

        Dispatches to the appropriate backend automatically.

        Args:
            prompt:         Input text prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature:    Sampling temperature (0 = greedy).

        Returns:
            Generated text string (without the prompt).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call setup_for_inference() first.")

        if self._backend == "gguf":
            return self._generate_gguf(prompt, max_new_tokens, temperature)
        return self._generate_hf(prompt, max_new_tokens, temperature)

    # ------------------------------------------------------------------
    # Backend-specific generation
    # ------------------------------------------------------------------

    def _generate_gguf(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """llama-cpp-python generation (GGUF / CUBLAS)."""
        response = self.model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=max(temperature, 1e-6),  # llama.cpp requires temperature > 0
        )
        return response["choices"][0]["message"]["content"]

    @torch.no_grad()
    def _generate_hf(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """HuggingFace transformers generation."""
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        # Strip the prompt tokens from the output
        generated = outputs[0][inputs.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

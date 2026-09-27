"""
ModelRunner — cross-platform multi-backend inference and training.

Inference backends (auto-selected by model path + platform):
  ┌──────────────────────┬─────────────────────────────────────────────────┐
  │ Model path           │ Backend                                         │
  ├──────────────────────┼─────────────────────────────────────────────────┤
  │ *.gguf               │ llama-cpp-python                                │
  │                      │  • Windows NVIDIA  → CUBLAS (ggml-cuda.dll)    │
  │                      │  • Linux  NVIDIA   → CUDA                      │
  │                      │  • macOS  Silicon  → Metal (GPU)               │
  │                      │  • CPU fallback    → llama.cpp CPU             │
  ├──────────────────────┼─────────────────────────────────────────────────┤
  │ HF repo / directory  │ transformers AutoModelForCausalLM               │
  │                      │  • Linux  + Unsloth → FastLanguageModel        │
  │                      │  • CUDA (any OS)   → bitsandbytes 4-bit        │
  │                      │  • macOS MPS       → float16                   │
  │                      │  • CPU             → float32                   │
  └──────────────────────┴─────────────────────────────────────────────────┘

Training backend (always HF-based):
  • Linux + Unsloth  → FastLanguageModel + unsloth LoRA
  • CUDA (any OS)    → bitsandbytes 4-bit QLoRA + PEFT
  • macOS / CPU      → transformers + PEFT (full precision)

Environment variables:
  LLAMA_N_GPU_LAYERS   Number of llama.cpp layers on GPU. Default -1 (all).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import torch

from ..config import ModelConfig
from .factory import ModelFactory, _has_cuda, _has_mps, _has_unsloth, _platform
from .hardware import HardwareManager

logger = logging.getLogger(__name__)

# GPU layers for llama.cpp:  -1 = all layers on GPU,  0 = CPU-only
_LLAMA_N_GPU_LAYERS = int(os.environ.get("LLAMA_N_GPU_LAYERS", "-1"))


# ---------------------------------------------------------------------------
# GGUF / llama-cpp-python loader
# ---------------------------------------------------------------------------


def _is_gguf(path: str) -> bool:
    """Return True when path ends with .gguf."""
    return path.strip().lower().endswith(".gguf")


def _llama_n_gpu_layers() -> int:
    """
    Compute the correct n_gpu_layers for the current platform.
    Override via LLAMA_N_GPU_LAYERS environment variable.
    """
    if _LLAMA_N_GPU_LAYERS != -1:
        return _LLAMA_N_GPU_LAYERS
    # Auto-detect: use GPU if any accelerator is available
    if _has_cuda() or _has_mps():
        return -1  # all layers on GPU
    return 0  # CPU only


def _load_llama_cpp(model_path: str, n_ctx: int = 4096):
    """
    Load a GGUF model via llama-cpp-python.

    GPU backend is selected automatically based on the installed wheel:
      • Windows NVIDIA  : CUBLAS precompiled wheel (DLL bootstrap applied)
      • Linux  NVIDIA   : CUDA wheel
      • macOS  Silicon  : Metal wheel
      • CPU fallback    : standard wheel

    Returns:
        llama_cpp.Llama instance
    """
    # Bootstrap platform-specific native library paths before importing llama_cpp
    from .dll_bootstrap import bootstrap_platform_dlls

    bootstrap_platform_dlls()

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        _install_hint = _llama_cpp_install_hint()
        raise ImportError(f"llama-cpp-python is not installed.\n{_install_hint}") from exc

    n_gpu = _llama_n_gpu_layers()
    accel = _infer_llama_backend()

    logger.info(
        f"ModelRunner [GGUF/{accel}]: Loading '{model_path}'  "
        f"n_gpu_layers={n_gpu}  n_ctx={n_ctx}"
    )

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu,
        n_ctx=n_ctx,
        verbose=False,
    )
    logger.info(f"ModelRunner [GGUF/{accel}]: Loaded successfully.")
    return llm


def _infer_llama_backend() -> str:
    """Return a human-readable label for the active llama.cpp backend."""
    if _has_cuda():
        return "CUBLAS" if sys.platform == "win32" else "CUDA"
    if _has_mps():
        return "Metal"
    return "CPU"


def _llama_cpp_install_hint() -> str:
    """Return platform-specific install instructions for llama-cpp-python."""
    if sys.platform == "win32" and _has_cuda():
        return (
            "Install the CUBLAS wheel (Windows NVIDIA):\n"
            "  uv pip install llama-cpp-python "
            "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124"
        )
    if sys.platform == "linux" and _has_cuda():
        return (
            "Install the CUDA wheel (Linux NVIDIA):\n"
            "  pip install llama-cpp-python "
            "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124"
        )
    if sys.platform == "darwin":
        return (
            "Install with Metal support (macOS Apple Silicon):\n"
            "  CMAKE_ARGS='-DGGML_METAL=on' pip install llama-cpp-python\n"
            "Or for CPU-only:\n"
            "  pip install llama-cpp-python"
        )
    return "  pip install llama-cpp-python"


# ---------------------------------------------------------------------------
# ModelRunner
# ---------------------------------------------------------------------------


class ModelRunner:
    """
    High-level cross-platform API for model loading, generation, and training.

    Backend selection is fully automatic — just set model_name_or_path:
      • Path ending in .gguf  → llama-cpp-python (CUDA / Metal / CPU)
      • HF repo or directory  → transformers (+ optional PEFT adapter)
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._backend: str = "hf"  # "gguf" | "hf"
        self.is_training_ready: bool = False

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def setup_for_training(self):
        """
        Prepares model + tokenizer for supervised fine-tuning.

        Backend (auto-selected by ModelFactory):
          • Linux + Unsloth   → FastLanguageModel + Unsloth LoRA
          • CUDA (any OS)     → bitsandbytes 4-bit QLoRA + PEFT
          • macOS MPS / CPU   → transformers float16/float32 + PEFT

        Returns:
            Tuple of (model, tokenizer)
        """
        self.model, self.tokenizer = ModelFactory.create_model_and_tokenizer(self.config)
        self.model = ModelFactory.apply_lora(self.model, self.config)
        self._backend = "hf"
        self.is_training_ready = True

        plat_label = {
            "linux": "Linux (Unsloth)" if (_has_cuda() and _has_unsloth()) else "Linux (PEFT)",
            "win32": "Windows (PEFT/QLoRA)",
            "darwin": "macOS (MPS/CPU)",
        }.get(_platform(), _platform())

        logger.info(f"ModelRunner: Ready for training on {plat_label}.")
        return self.model, self.tokenizer

    def setup_for_inference(self, adapter_path: Optional[str] = None):
        """
        Loads a model for inference, auto-selecting the optimal backend.

        Args:
            adapter_path: Optional LoRA adapter directory (HF backend only).

        Returns:
            self  (chainable)
        """
        model_path = self.config.model_name_or_path

        if _is_gguf(model_path):
            # ── GGUF path: llama-cpp-python (CUBLAS / Metal / CPU) ───────────
            self._backend = "gguf"
            self.model = _load_llama_cpp(model_path, n_ctx=self.config.max_seq_length)
            self.tokenizer = None  # llama.cpp handles tokenisation internally

        else:
            # ── HF path: transformers (+ optional PEFT adapter) ──────────────
            if _platform() == "darwin" and HardwareManager.use_mlx():
                self._backend = "mlx"
                import mlx_lm
                logger.info(f"ModelRunner [MLX]: Loading model '{model_path}' with adapter '{adapter_path}'")
                self.model, self.tokenizer = mlx_lm.load(model_path, adapter_path=adapter_path)
                if hasattr(self.tokenizer, "pad_token") and self.tokenizer.pad_token is None:
                    if hasattr(self.tokenizer, "eos_token"):
                        self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self._backend = "hf"
                if self.model is None:
                    self.model, self.tokenizer = ModelFactory.create_model_and_tokenizer(self.config)

                if adapter_path:
                    from peft import PeftModel

                    logger.info(f"Loading LoRA adapter from: {adapter_path}")
                    self.model = PeftModel.from_pretrained(self.model, adapter_path)

                self.model = ModelFactory.prepare_for_inference(self.model)

        logger.info(
            f"ModelRunner: Ready for inference  "
            f"backend={self._backend}  accel={_infer_llama_backend() if self._backend == 'gguf' else _best_accel_label()}"
        )
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
        Generate text from a prompt. Dispatches to the active backend.

        Args:
            prompt:         Input text.
            max_new_tokens: Maximum number of tokens to generate.
            temperature:    Sampling temperature (0 = greedy).

        Returns:
            Generated text (prompt excluded).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call setup_for_inference() first.")

        if self._backend == "gguf":
            return self._generate_gguf(prompt, max_new_tokens, temperature)
        elif self._backend == "mlx":
            return self._generate_mlx(prompt, max_new_tokens, temperature)
        return self._generate_hf(prompt, max_new_tokens, temperature)

    # ------------------------------------------------------------------
    # Backend-specific generation
    # ------------------------------------------------------------------

    def _generate_mlx(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """MLX generation for Apple Silicon."""
        import mlx_lm

        # Apply chat template if supported
        if hasattr(self.tokenizer, "apply_chat_template"):
            formatted_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted_prompt = prompt

        return mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_new_tokens,
            temp=temperature,
            verbose=False,
        )

    def _generate_gguf(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """llama-cpp-python generation (GGUF — CUDA / Metal / CPU)."""
        response = self.model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=max(temperature, 1e-6),  # llama.cpp requires temperature > 0
        )
        return response["choices"][0]["message"]["content"]

    @torch.no_grad()
    def _generate_hf(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """HuggingFace transformers generation (CUDA / MPS / CPU)."""
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # If inputs is a BatchEncoding/dict, move all its tensors to the model's device
        if hasattr(inputs, "keys") and hasattr(inputs, "items"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                temperature=temperature,
                do_sample=temperature > 0,
            )
            prompt_len = inputs["input_ids"].shape[-1]
        else:
            inputs = inputs.to(self.model.device)
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                temperature=temperature,
                do_sample=temperature > 0,
            )
            prompt_len = inputs.shape[-1]

        # Return only the newly generated tokens (strip the prompt)
        generated = outputs[0][prompt_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Internal label helper
# ---------------------------------------------------------------------------


def _best_accel_label() -> str:
    if _has_cuda():
        return "CUDA"
    if _has_mps():
        return "MPS"
    return "CPU"

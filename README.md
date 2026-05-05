
---

# 🚀 LLM Finetuning — Windows-Native GPU Stack (v0.2.0)

**Fast, modular LLM fine-tuning and inference — no WSL, no Triton required.**

<p align="center">
  <img src="https://github.com/user-attachments/assets/6c834e71-ad14-40b0-a26f-27783752c07f" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Backend-llama.cpp_CUBLAS-blue?logo=nvidia" />
  <img src="https://img.shields.io/badge/Training-BitsAndBytes_4bit-green" />
  <img src="https://img.shields.io/badge/LoRA-PEFT-yellow" />
  <img src="https://img.shields.io/badge/Trainer-TRL_SFTTrainer-orange" />
  <img src="https://img.shields.io/badge/Windows-Native_CUDA-brightgreen?logo=windows" />
  <img src="https://img.shields.io/badge/License-MIT-purple" />
</p>

# FINE-TUNING AND INFERENCE NOTEBOOKS:
<p align="center">
  <a href="https://colab.research.google.com/drive/1WpbMOTuuW3E5KtcOTrkJ6AqQ3jTRRaoM?usp=sharing" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
  </a>
</p>
<p align="center">
  <a href="https://colab.research.google.com/drive/1_xWw9L-QgPql7sk94FJ2iJnS7VDp-Mit?usp=sharing" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
  </a>
</p>
<!-- HuggingFace Model Badge -->
<p align="center">
  <a href="https://huggingface.co/black279/Qwen_LeetCoder" target="_blank">
    <img src="https://img.shields.io/badge/HuggingFace-Qwen_LeetCoder-orange?style=flat-square" alt="HuggingFace Model"/>
  </a>
</p>

---

## 📌 Overview

This repository provides a **professional enterprise pipeline** for fine-tuning and running LLMs entirely on **Windows with a native NVIDIA GPU** — no WSL2, no Triton, no Docker required.

| Feature | Details |
|---|---|
| ⚡ **GPU-accelerated inference** | llama.cpp CUBLAS backend — GGUF models run directly on GPU via precompiled C++ kernels |
| 🏋️ **GPU-accelerated training** | bitsandbytes 4-bit NF4 QLoRA + PEFT LoRA + TRL SFTTrainer, all CUDA-native on Windows |
| 🔄 **Dual inference backends** | Auto-selects GGUF/llama.cpp or safetensors/transformers based on file extension |
| 🏗️ **Modular Architecture** | Clean separation: `ModelFactory`, `ModelRunner`, `DataProcessor` |
| 🖥️ **Interactive Studio** | Gradio-based no-code GUI for visual configuration |
| 🛠️ **Robust CLI** | Unified entry point for training and inference |
| 💻 **CPU Mock Mode** | Full pipeline test without any GPU |

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Windows-Native GPU Stack                      │
├──────────────────────────┬──────────────────────────────────────┤
│        TRAINING          │            INFERENCE                  │
│                          │                                       │
│  transformers            │  Path A — GGUF (.gguf file)          │
│  AutoModelForCausalLM    │  └─ llama-cpp-python                 │
│  + BitsAndBytes 4-bit    │     └─ ggml-cuda.dll (CUBLAS)        │
│    (NF4 QLoRA)           │        All layers on GPU  ✅         │
│  + PEFT LoraConfig       │                                       │
│  + TRL SFTTrainer        │  Path B — Safetensors (HF repo)      │
│  + adamw_8bit            │  └─ transformers generate()          │
│  No Triton  ✅           │  └─ Optional: PEFT adapter  ✅       │
│  No WSL2   ✅            │  No Triton  ✅  No WSL2  ✅          │
└──────────────────────────┴──────────────────────────────────────┘
      cudart64_12.dll resolved from PyTorch's bundled CUDA runtime
      — no separate CUDA Toolkit installation required
```

---

## 📦 Installation

We recommend [uv](https://docs.astral.sh/uv/) for fast dependency management.

### 1. Windows — Native GPU Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/Sriramdayal/Unsloth-LLM-finetuningv1.git
cd Unsloth-LLM-finetuningv1

# Create virtual environment and install core deps
uv venv
uv pip install -e ".[gui]"

# Install llama-cpp-python with CUBLAS (GPU-accelerated inference)
# This wheel bundles precompiled ggml-cuda.dll — no CUDA Toolkit needed
uv pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

> **Note:** The `cudart64_12.dll` dependency is resolved automatically from PyTorch's
> bundled CUDA runtime. No separate NVIDIA CUDA Toolkit installation is required.

### 2. Verify GPU is detected

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX 4060 Laptop GPU
```

### 3. Linux / WSL2 Setup (with original Unsloth kernels)

```bash
uv pip install -e ".[unsloth,gpu,gui]"
```

### 4. Docker Setup

```bash
docker compose up --build
```

---

## 🛠 Usage Guide

### 🚀 CLI (Command Line Interface)

**Fine-Tuning (Windows GPU — QLoRA 4-bit):**
```bash
uv run unsloth-cli train \
  --model_name_or_path "unsloth/llama-3-8b-bnb-4bit" \
  --dataset_name "yahma/alpaca-cleaned"
```

**Fine-Tuning from a config file:**
```bash
uv run unsloth-cli train --config config.yaml
```

**Inference — GGUF model (llama.cpp CUBLAS, fastest on Windows):**
```bash
uv run unsloth-cli infer \
  --model "path/to/model.gguf" \
  --prompt "Write a Python binary search implementation."
```

**Inference — HuggingFace / safetensors model:**
```bash
uv run unsloth-cli infer \
  --model "outputs/lora_adapters" \
  --prompt "What is machine learning?"
```

**CPU Mock Mode (test pipeline without GPU):**
```bash
uv run unsloth-cli train \
  --model_name_or_path "any-model-id" \
  --use_mock True \
  --dataset_name "yahma/alpaca-cleaned"
```

### 🎨 GUI (Fine-Tuning Studio)

```bash
uv run unsloth-gui
```

---

## 📘 Python API Guide

```python
from src import ModelConfig, TrainConfig
from src.core.model_runner import ModelRunner
from src.data import DataProcessor

# ── Training ──────────────────────────────────────────────────────────
config = ModelConfig(
    model_name_or_path="unsloth/llama-3-8b-bnb-4bit",
    load_in_4bit=True,   # 4-bit NF4 QLoRA via bitsandbytes
    lora_r=16,
)
runner = ModelRunner(config)
model, tokenizer = runner.setup_for_training()

# ── Inference: GGUF / llama.cpp CUBLAS ───────────────────────────────
gguf_config = ModelConfig(model_name_or_path="path/to/model.gguf")
gguf_runner = ModelRunner(gguf_config)
gguf_runner.setup_for_inference()
print(gguf_runner.generate("Explain gradient descent."))

# ── Inference: HF safetensors + LoRA adapter ─────────────────────────
hf_runner = ModelRunner(config)
hf_runner.setup_for_inference(adapter_path="outputs/lora_adapters")
print(hf_runner.generate("What is a transformer?"))
```

👉 **[Read the Full Python API Documentation](documentation.md)**

---

## 🔧 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLAMA_N_GPU_LAYERS` | `-1` | Number of llama.cpp layers to offload to GPU. `-1` = all layers. |

```bash
# Example: offload only 20 layers (useful if VRAM is limited)
set LLAMA_N_GPU_LAYERS=20
uv run unsloth-cli infer --model model.gguf --prompt "Hello"
```

---

## ⏱ Training Time (Estimates on RTX 4060 8GB)

| Dataset Size | QLoRA 4-bit (this stack) | Notes |
|---|---|---|
| 5k samples | ~10–20 min | Alpaca-style |
| 50k samples | ~2–4 hrs | Alpaca-cleaned |
| 500k samples | ~20–30 hrs | Large corpus |

> Times assume `batch_size=2`, `gradient_accumulation_steps=4`, `lora_r=16`, `max_seq_length=2048`.

### Broader GPU Reference

| GPU | Small (50k) | Large (500k–800k) |
|---|---|---|
| **A100** | 20–40 min | 5–7 hours |
| **RTX 4090** | 1–2 hrs | 18–22 hrs |
| **RTX 4060 / 3090** | 2–4 hrs | 28–34 hrs |
| **Tesla T4** | 4–6 hrs | 55–70 hrs |

> Training time scales linearly with dataset size and LoRA rank.

---

## 🗂️ Project Structure

```
src/
├── cli.py                  # Unified CLI: train + infer subcommands
├── config.py               # ModelConfig / TrainConfig dataclasses
├── data.py                 # DataProcessor (load, format, tokenize)
├── train.py                # train_model() — TRL SFTTrainer wrapper
├── core/
│   ├── factory.py          # ModelFactory: bitsandbytes 4-bit + PEFT LoRA
│   └── model_runner.py     # ModelRunner: GGUF/llama.cpp + HF dual-path
└── utils/
    ├── env.py              # HardwareManager (GPU stats, device detection)
    └── llama_loader.py     # Windows DLL bootstrap for llama-cpp-python

scripts/
├── app.py                  # Gradio GUI
├── run_training.py         # Script entry point
├── run_inference.py        # Script entry point
└── smoke_test.py           # Import validation test
```

---

## 🧪 Smoke Test

Validates all imports and detects GPU correctly:

```bash
uv run python scripts/smoke_test.py
```

---

## 📜 License

MIT

---

## 🔗 Credits

* [llama.cpp](https://github.com/ggerganov/llama.cpp) — C++ inference engine with CUBLAS GPU support
* [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — Python bindings with prebuilt CUDA wheels
* [HuggingFace Transformers](https://github.com/huggingface/transformers)
* [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) — 4-bit quantization
* [PEFT](https://github.com/huggingface/peft) — LoRA adapters
* [TRL](https://github.com/huggingface/trl) — SFTTrainer
* [Unsloth](https://github.com/unslothai/unsloth) — Original inspiration (Linux/WSL2 path)

## Contribution and Issues

Feel free to contribute by opening a Pull Request or raising an issue.

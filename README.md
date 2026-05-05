
---

# 🚀 LLM Finetuning — Cross-Platform GPU Pipeline (v0.2.0)

**Fast, modular LLM fine-tuning and inference on Windows, Linux, and macOS — fully GPU-accelerated.**

<p align="center">
  <img src="https://github.com/user-attachments/assets/6c834e71-ad14-40b0-a26f-27783752c07f" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Linux-Unsloth_Triton-blue?logo=linux&logoColor=white" />
  <img src="https://img.shields.io/badge/Windows-CUBLAS_QLoRA-blue?logo=windows&logoColor=white" />
  <img src="https://img.shields.io/badge/macOS-Metal_MPS-blue?logo=apple&logoColor=white" />
  <img src="https://img.shields.io/badge/Inference-llama.cpp-green" />
  <img src="https://img.shields.io/badge/Training-bitsandbytes_QLoRA-green" />
  <img src="https://img.shields.io/badge/LoRA-PEFT-yellow" />
  <img src="https://img.shields.io/badge/Trainer-TRL_SFTTrainer-orange" />
  <img src="https://img.shields.io/badge/License-MIT-purple" />
</p>

# Fine-Tuning and Inference Notebooks

<p align="center">
  <a href="https://colab.research.google.com/drive/1WpbMOTuuW3E5KtcOTrkJ6AqQ3jTRRaoM?usp=sharing" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab — Fine-Tuning"/>
  </a>
  &nbsp;
  <a href="https://colab.research.google.com/drive/1_xWw9L-QgPql7sk94FJ2iJnS7VDp-Mit?usp=sharing" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab — Inference"/>
  </a>
  &nbsp;
  <a href="https://huggingface.co/black279/Qwen_LeetCoder" target="_blank">
    <img src="https://img.shields.io/badge/HuggingFace-Qwen_LeetCoder-orange?style=flat-square" alt="HuggingFace Model"/>
  </a>
</p>

---

## 📌 Overview

A **professional enterprise pipeline** for fine-tuning and running open-source LLMs across **all major platforms** — the backend is selected automatically with zero configuration required.

| Feature | Details |
|---|---|
| ⚡ **GPU inference** | llama.cpp — CUBLAS (Windows) · CUDA (Linux) · Metal (macOS) |
| 🏋️ **GPU training** | Unsloth Triton (Linux) · bitsandbytes 4-bit QLoRA (Windows/Linux) · MPS (macOS) |
| 🔄 **Dual inference backends** | Auto-selects: `.gguf` → llama.cpp · HF repo/dir → transformers |
| 🤖 **Auto backend selection** | Platform detected at runtime — no config needed |
| 🏗️ **Modular architecture** | Clean separation: `ModelFactory` · `ModelRunner` · `DataProcessor` |
| 🖥️ **Interactive GUI** | Gradio-based no-code fine-tuning studio |
| 🛠️ **Robust CLI** | Unified entry point for training and inference |
| 💻 **CPU mock mode** | Full pipeline test without any GPU |

---

## 🏛️ Architecture

```
┌──────────────────────────┬────────────────────────────────────────────────┐
│ Platform                 │ Training                  Inference             │
├──────────────────────────┼────────────────────────────────────────────────┤
│ 🐧 Linux + NVIDIA        │ Unsloth Triton (2× speed) llama.cpp CUDA       │
│    (Best for training)   │ → bitsandbytes QLoRA ↩    OR transformers      │
├──────────────────────────┼────────────────────────────────────────────────┤
│ 🪟 Windows + NVIDIA      │ bitsandbytes 4-bit NF4    llama.cpp CUBLAS     │
│    (No WSL required)     │ QLoRA + PEFT LoRA         OR transformers      │
├──────────────────────────┼────────────────────────────────────────────────┤
│ 🍎 macOS Apple Silicon   │ transformers float16      llama.cpp Metal      │
│    (M1 / M2 / M3 / M4)  │ MPS device + PEFT LoRA    OR transformers MPS  │
├──────────────────────────┼────────────────────────────────────────────────┤
│ 💻 CPU / macOS Intel     │ transformers float32      llama.cpp CPU        │
│    (No GPU)              │ + PEFT LoRA               OR transformers CPU  │
└──────────────────────────┴────────────────────────────────────────────────┘

  Windows → cudart64_12.dll auto-resolved from PyTorch bundled CUDA runtime
  macOS   → Metal support built into llama-cpp-python wheel
  Linux   → libcudart.so resolved via LD_LIBRARY_PATH (standard CUDA setup)
```

> **All backend switching is automatic.** Set `model_name_or_path` — the factory detects
> your OS, GPU, and installed packages and picks the optimal stack.

---

## 📦 Installation

We recommend [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management.

### 🪟 Windows — Native NVIDIA GPU (No WSL Required)

```bash
git clone https://github.com/Sriramdayal/Unsloth-LLM-finetuningv1.git
cd Unsloth-LLM-finetuningv1

uv venv
uv pip install -e ".[gui]"

# llama-cpp-python CUBLAS wheel — GPU inference, no CUDA Toolkit install needed
uv pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

> **How it works:** `cudart64_12.dll` is auto-resolved from PyTorch's bundled CUDA runtime
> via `src/utils/llama_loader.py`. No NVIDIA CUDA Toolkit installation required.

**Verify your GPU:**
```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True  NVIDIA GeForce RTX 4060 Laptop GPU
```

---

### 🐧 Linux — Full Unsloth Stack (Fastest Training)

```bash
git clone https://github.com/Sriramdayal/Unsloth-LLM-finetuningv1.git
cd Unsloth-LLM-finetuningv1

pip install -e ".[linux,gui]"

# llama-cpp-python CUDA wheel
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

> **Training:** Uses Unsloth Triton kernels — **2× faster** and **70% less VRAM** vs.
> standard transformers. Automatically falls back to bitsandbytes QLoRA if Unsloth
> is not installed.

---

### 🍎 macOS — Apple Silicon (M1 / M2 / M3 / M4)

```bash
git clone https://github.com/Sriramdayal/Unsloth-LLM-finetuningv1.git
cd Unsloth-LLM-finetuningv1

pip install -e ".[macos,gui]"

# llama-cpp-python with Metal GPU acceleration
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

> **Training:** Uses Apple MPS device with `float16`. Note: bitsandbytes 4-bit is not
> yet supported on MPS, so models run in full `float16` precision.
> **Inference:** llama.cpp uses Metal GPU — fast and energy-efficient.

---

### 💻 CPU-Only (Any OS, No GPU)

```bash
pip install -e ".[cpu,gui]"
pip install llama-cpp-python   # Standard CPU build
```

---

### 🐳 Docker

```bash
docker compose up --build
```

---

## 🛠 Usage Guide

### 🚀 CLI — Training

**GPU training (auto-selects backend for your OS):**
```bash
uv run unsloth-cli train \
  --model_name_or_path "unsloth/llama-3-8b-bnb-4bit" \
  --dataset_name "yahma/alpaca-cleaned"
```

**Training from a config file:**
```bash
uv run unsloth-cli train --config config.yaml
```

**CPU Mock Mode (test without a GPU):**
```bash
uv run unsloth-cli train \
  --model_name_or_path "any-model-id" \
  --use_mock True \
  --dataset_name "yahma/alpaca-cleaned"
```

---

### 🔍 CLI — Inference

**GGUF model (llama.cpp — CUBLAS / CUDA / Metal, auto-detected):**
```bash
uv run unsloth-cli infer \
  --model "path/to/model.gguf" \
  --prompt "Write a Python binary search implementation."
```

**HuggingFace safetensors / LoRA adapter:**
```bash
uv run unsloth-cli infer \
  --model "outputs/lora_adapters" \
  --prompt "What is machine learning?"
```

---

### 🎨 GUI — Fine-Tuning Studio

```bash
uv run unsloth-gui
```

---

## 📘 Python API

```python
from src import ModelConfig, TrainConfig
from src.core.model_runner import ModelRunner
from src.data import DataProcessor

# ── Training (backend auto-selected per OS) ──────────────────────────────────
config = ModelConfig(
    model_name_or_path="unsloth/llama-3-8b-bnb-4bit",
    load_in_4bit=True,   # NF4 4-bit QLoRA via bitsandbytes (CUDA/Linux/Windows)
    lora_r=16,
)
runner = ModelRunner(config)
model, tokenizer = runner.setup_for_training()  # Applies LoRA automatically

# ── Inference: GGUF → llama.cpp (CUBLAS / CUDA / Metal) ─────────────────────
gguf_runner = ModelRunner(ModelConfig(model_name_or_path="model.gguf"))
gguf_runner.setup_for_inference()
print(gguf_runner.generate("Explain gradient descent in simple terms."))

# ── Inference: HF safetensors + LoRA adapter ─────────────────────────────────
hf_runner = ModelRunner(config)
hf_runner.setup_for_inference(adapter_path="outputs/lora_adapters")
print(hf_runner.generate("What is a transformer model?"))
```

👉 **[Read the Full Python API Documentation](documentation.md)**

---

## ⚙️ Config File Reference

`config.yaml` example:
```yaml
model_name_or_path: "unsloth/llama-3-8b-bnb-4bit"
load_in_4bit: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
dataset_name: "yahma/alpaca-cleaned"
learning_rate: 0.0002
num_train_epochs: 3
batch_size: 2
gradient_accumulation_steps: 4
output_dir: "outputs/my_model"
push_to_hub: false
```

```bash
uv run unsloth-cli train --config config.yaml
```

---

## 🔧 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLAMA_N_GPU_LAYERS` | `-1` | llama.cpp GPU layers. `-1` = all on GPU, `0` = CPU only. |

```bash
# Limit GPU layers if VRAM is tight
set LLAMA_N_GPU_LAYERS=20          # Windows
export LLAMA_N_GPU_LAYERS=20       # Linux / macOS
```

---

## ⏱ Training Time Estimates
| GPU | Small (50k) | Large (500k–800k) |
|---|---|---|
| **A100 80GB** | 20–40 min | 5–7 hrs |
| **RTX 4090** | 1–2 hrs | 18–22 hrs |
| **RTX 4060 / 3090** | 2–4 hrs | 28–34 hrs |
| **Tesla T4 (Colab)** | 4–6 hrs | 55–70 hrs |

> Linux + Unsloth Triton is ~2× faster across all GPUs compared to this table.

---

## 🗂️ Project Structure

```
src/
├── cli.py                  # Unified CLI: train + infer subcommands
├── config.py               # ModelConfig / TrainConfig dataclasses
├── data.py                 # DataProcessor (load, format, tokenize)
├── train.py                # train_model() — TRL SFTTrainer wrapper
├── core/
│   ├── factory.py          # Cross-platform ModelFactory (Unsloth / QLoRA / MPS)
│   └── model_runner.py     # ModelRunner: GGUF/llama.cpp + HF dual-backend
└── utils/
    ├── env.py              # HardwareManager (CUDA / MPS / CPU detection)
    └── llama_loader.py     # Platform DLL/library bootstrap for llama-cpp-python

scripts/
├── app.py                  # Gradio GUI (unsloth-gui)
├── run_training.py         # Script entry point
├── run_inference.py        # Script entry point
└── smoke_test.py           # Import validation
```

---

## 🧪 Smoke Test

Validates all imports and reports the active backend:

```bash
uv run python scripts/smoke_test.py
```

---

## 📜 License

MIT

---

## 🔗 Credits

* [llama.cpp](https://github.com/ggerganov/llama.cpp) — C++ LLM inference with CUDA / Metal / CPU backends
* [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — Python bindings with prebuilt GPU wheels
* [Unsloth](https://github.com/unslothai/unsloth) — Triton-optimised LoRA training (Linux/WSL2)
* [HuggingFace Transformers](https://github.com/huggingface/transformers)
* [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) — 4-bit NF4 quantization
* [PEFT](https://github.com/huggingface/peft) — LoRA adapters
* [TRL](https://github.com/huggingface/trl) — SFTTrainer

---

## 🤝 Contribution and Issues

Feel free to contribute by opening a Pull Request or raising an issue on
[GitHub](https://github.com/Sriramdayal/Unsloth-LLM-finetuningv1/issues).

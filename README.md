
---

# 🚀 LLM Finetuning — Cross-Platform GPU Pipeline

**Fast, modular LLM fine-tuning and inference on Windows, Linux, and macOS — fully GPU-accelerated.**

<p align="center">
  <img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/a3693aa9-83b7-424a-9d49-c10b7fe24626" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Linux-Unsloth_Triton-blue?logo=linux&logoColor=white" />
  <img src="https://img.shields.io/badge/Windows-CUBLAS_QLoRA-blue?logo=windows&logoColor=white" />
  <img src="https://img.shields.io/badge/macOS-Apple_MLX-blue?logo=apple&logoColor=white" />
  <img src="https://img.shields.io/badge/Inference-llama.cpp-green" />
  <img src="https://img.shields.io/badge/Training-MLX_/_QLoRA-green" />
  <img src="https://img.shields.io/badge/LoRA-PEFT_/_MLX-yellow" />
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
| ⚡ **GPU inference** | Apple MLX / llama.cpp — Metal (macOS) · CUBLAS (Windows) · CUDA (Linux) |
| 🏋️ **GPU training** | Unsloth Triton (Linux) · bitsandbytes 4-bit QLoRA (Windows/Linux) · Apple MLX LoRA (macOS) |
| 🔄 **Dual inference backends** | Auto-selects: `.gguf` → llama.cpp · HF repo/dir → transformers / mlx-lm |
| 🤖 **Auto backend selection** | Platform detected at runtime — no config needed |
| 🧠 **Multi-Agent System (Beta)** | `smolagents` powered AI assistant for model/param selection & coding |
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
│ 🍎 macOS Apple Silicon   │ mlx-lm Apple MLX          mlx-lm generation    │
│    (M1 / M2 / M3 / M4)  │ MLX Native LoRA           OR llama.cpp Metal   │
├──────────────────────────┼────────────────────────────────────────────────┤
│ 💻 CPU / macOS Intel     │ transformers float32      llama.cpp CPU        │
│    (No GPU)              │ + PEFT LoRA               OR transformers CPU  │
└──────────────────────────┴────────────────────────────────────────────────┘

  Windows → cudart64_12.dll auto-resolved from PyTorch bundled CUDA runtime
  macOS   → Apple MLX native integration / Metal support built into llama-cpp-python wheel
  Linux   → libcudart.so resolved via LD_LIBRARY_PATH (standard CUDA setup)
```

> **All backend switching is automatic.** Set `model_name_or_path` — the factory detects
> your OS, GPU, and installed packages and picks the optimal stack.

---

## 📦 Installation

I recommend [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management.

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

> **Training:** Uses Apple's native **MLX** framework via `mlx-lm`. 
> Data is converted on the fly to support MLX's high-speed LoRA tuning.
> **Inference:** Natively runs on MLX for Hugging Face models, or llama.cpp Metal for GGUFs.

---

### 💻 CPU-Only (Any OS, No GPU)

```bash
pip install -e ".[cpu,gui]"
pip install llama-cpp-python   # Standard CPU build
```

---

### 🐳 Docker

Run the entire pipeline inside a Docker container (includes full GPU support via Nvidia runtime).

#### Start the API Server via Docker Compose
Build and run the FastAPI server on port `8000`:
```bash
docker compose up --build
```

#### Run CLI Fine-tuning inside a Running Container
To run fine-tuning tasks inside a running container, use `docker exec` (or `docker compose exec`). You MUST pass the `PYTHONPATH=.` environment variable so that the internal modules are correctly resolved:

```bash
# 1. Get the running container ID or name
docker ps

# 2. Run fine-tuning with CLI parameters
docker exec -it -e PYTHONPATH=. <container_id_or_name> unsloth-cli train \
  --model_name_or_path "unsloth/llama-3-8b-bnb-4bit" \
  --dataset_name "yahma/alpaca-cleaned"

# 3. Or run fine-tuning using a mounted config file
docker exec -it -e PYTHONPATH=. <container_id_or_name> unsloth-cli train --config config.yaml
```

---

## 🛠 Usage Guide

The pipeline detects your operating system and hardware configuration automatically at runtime.

### 🐧 Linux (CUDA / Unsloth)

Linux environments with NVIDIA GPUs run the full **Unsloth Triton** stack for maximum training speed.

* **Training (CLI):**
  ```bash
  uv run unsloth-cli train \
    --model_name_or_path "unsloth/llama-3-8b-bnb-4bit" \
    --dataset_name "yahma/alpaca-cleaned"
  ```
* **Inference — GGUF (via llama.cpp CUDA):**
  ```bash
  uv run unsloth-cli infer \
    --model "path/to/model.gguf" \
    --prompt "Write a Python binary search implementation."
  ```
* **Inference — HuggingFace / LoRA Adapter (via Transformers):**
  ```bash
  uv run unsloth-cli infer \
    --model "outputs/lora_adapters" \
    --prompt "Explain gradient descent in simple terms."
  ```

---

### 🪟 Windows (CUDA / QLoRA)

Windows environments with NVIDIA GPUs run **bitsandbytes 4-bit QLoRA** training natively (no WSL2 required).

* **Training (CLI):**
  ```bash
  uv run unsloth-cli train \
    --model_name_or_path "unsloth/llama-3-8b-bnb-4bit" \
    --dataset_name "yahma/alpaca-cleaned"
  ```
* **Inference — GGUF (via llama.cpp CUBLAS):**
  ```bash
  uv run unsloth-cli infer \
    --model "path/to/model.gguf" \
    --prompt "Write a Python binary search implementation."
  ```
* **Inference — HuggingFace / LoRA Adapter (via Transformers):**
  ```bash
  uv run unsloth-cli infer \
    --model "outputs/lora_adapters" \
    --prompt "Explain gradient descent in simple terms."
  ```

---

### 🍎 macOS (Apple Silicon / MLX)

macOS environments run Apple's native **MLX** (`mlx-lm`) backend for optimal hardware acceleration.

* **Training (CLI):**
  MLX training automatically exports your dataset to JSONL format and runs MLX native LoRA tuning.
  ```bash
  uv run unsloth-cli train \
    --model_name_or_path "mlx-community/Llama-3-8B-Instruct-4bit" \
    --dataset_name "yahma/alpaca-cleaned"
  ```
* **Inference — GGUF (via llama.cpp Metal):**
  ```bash
  uv run unsloth-cli infer \
    --model "path/to/model.gguf" \
    --prompt "Write a Python binary search implementation."
  ```
* **Inference — HuggingFace / MLX (via mlx-lm):**
  Runs natively on the MLX engine:
  ```bash
  uv run unsloth-cli infer \
    --model "mlx-community/Llama-3-8B-Instruct-4bit" \
    --prompt "Explain gradient descent in simple terms."
  ```

---

### 💻 CPU-Only (Any OS, No GPU)

Falls back to standard PyTorch CPU execution and llama.cpp CPU inference.

* **Training (CLI — Mock Mode recommended for testing):**
  ```bash
  uv run unsloth-cli train \
    --model_name_or_path "any-model-id" \
    --use_mock True \
    --dataset_name "yahma/alpaca-cleaned"
  ```
* **Inference — GGUF (via llama.cpp CPU):**
  ```bash
  uv run unsloth-cli infer \
    --model "path/to/model.gguf" \
    --prompt "Write a Python binary search implementation."
  ```

---

### ⚙️ Universal Configurations

* **Training from a config file (Works on all platforms):**
  ```bash
  uv run unsloth-cli train --config config.yaml
  ```

---

### 🤖 Multi-Agent Fine-Tuning Assistant (Beta)

The pipeline includes an intelligent multi-agent system powered by Hugging Face's `smolagents` framework. The agent can suggest the best models for your use case, recommend optimal LoRA parameters, and write training code!

1. Set your Hugging Face Token:
   ```bash
   # Windows
   $env:HF_TOKEN="your_hf_token_here"
   # Linux / macOS
   export HF_TOKEN="your_hf_token_here"
   ```
2. Run the agent:
   ```bash
   uv run python src/finetuning_agent.py
   ```

---

### 🎨 GUI — Fine-Tuning Studio

```bash
uv run unsloth-gui
```

---

### 🌐 REST API

A production-ready FastAPI REST API server is included for programmatically enqueuing training jobs and running inference.

#### Start the API Server
Launch the server locally on port `8000`:
```bash
uv run uvicorn src.api:app
```
*(On Windows, do not use `--reload` due to virtual environment resolving issues during process reloading).*

Once running, the interactive documentation is available at:
* **Interactive UI Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Static ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

#### REST API Endpoints
* **System Health (`GET /api/v1/health`)**: Check current status and active backend.
  ```bash
  curl http://127.0.0.1:8000/api/v1/health
  ```
* **Start Fine-Tuning (`POST /api/v1/train`)**: Start a background training task. Returns a `job_id` immediately.
  ```bash
  curl -X POST http://127.0.0.1:8000/api/v1/train \
    -H "Content-Type: application/json" \
    -d '{"model_name_or_path": "unsloth/llama-3-8b-bnb-4bit", "dataset_name": "yahma/alpaca-cleaned"}'
  ```
* **Job Status (`GET /api/v1/train/{job_id}/status`)**: Poll status, progress percentage, and output directory.
  ```bash
  curl http://127.0.0.1:8000/api/v1/train/<job_id>/status
  ```
* **HuggingFace Inference (`POST /api/v1/infer`)**: Generate text using HuggingFace model or LoRA adapters.
* **GGUF Inference (`POST /api/v1/infer/gguf`)**: Run high-performance local inference via llama.cpp.

---

## 📘 Python API

```python
from src import ModelConfig, TrainConfig
from src.core.model_runner import ModelRunner
from src.data import DataProcessor

# ── Training (backend auto-selected: Unsloth/Linux, QLoRA/Windows, MLX/macOS) ─
config = ModelConfig(
    model_name_or_path="unsloth/llama-3-8b-bnb-4bit",
    load_in_4bit=True,   # NF4 4-bit QLoRA on CUDA; ignored & auto-loaded on macOS MLX
    lora_r=16,
)
runner = ModelRunner(config)
model, tokenizer = runner.setup_for_training()  # Applies LoRA (or configures MLX)

# ── Inference: GGUF → llama.cpp (CUBLAS / CUDA / Metal) ─────────────────────
gguf_runner = ModelRunner(ModelConfig(model_name_or_path="model.gguf"))
gguf_runner.setup_for_inference()
print(gguf_runner.generate("Explain gradient descent in simple terms."))

# ── Inference: HF / MLX + LoRA adapter ───────────────────────────────────────
hf_runner = ModelRunner(config)
# Auto-selects MLX (mlx-lm) on macOS Apple Silicon, transformers on Windows/Linux
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
├── train.py                # train_model() — SFTTrainer / MLX training orchestrator
├── core/
│   ├── factory.py          # Cross-platform ModelFactory (Unsloth / QLoRA / MLX / MPS)
│   └── model_runner.py     # ModelRunner: GGUF/llama.cpp + HF/MLX dual-backend
└── utils/
    ├── env.py              # HardwareManager (CUDA / MLX / MPS / CPU detection)
    └── llama_loader.py     # Platform DLL/library bootstrap for llama-cpp-python

scripts/
├── app.py                  # Gradio GUI (unsloth-gui)
├── run_training.py         # Script entry point
├── run_inference.py        # Script entry point
└── smoke_test.py           # Import validation
```

---

## 🧪 Testing

The repository features both unit/integration tests and a hardware validation smoke test.

### Running unit and integration tests (pytest)
Tests are written with `pytest` and use mocked ML frameworks to validate configuration, API schemas, and CLI workflows quickly without GPU access.

To execute the test suite:
```bash
# Run pytest with uv
uv run pytest

# Run tests and generate coverage reports (terminal and HTML)
./scripts/run_all_tests.sh
```

### Running the Smoke Test
Verifies library imports, resolves CUDA/MPS availability, and reports the active backend:
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
* [Apple MLX](https://github.com/ml-explore/mlx) — Apple Silicon machine learning framework
* [MLX Tuning (mlx-lm)](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) — Apple Silicon LLM inference and fine-tuning
* [HuggingFace Transformers](https://github.com/huggingface/transformers)
* [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) — 4-bit NF4 quantization
* [PEFT](https://github.com/huggingface/peft) — LoRA adapters
* [TRL](https://github.com/huggingface/trl) — SFTTrainer

---

## 🤝 Contribution and Issues

Feel free to contribute by opening a Pull Request or raising an issue on
[GitHub](https://github.com/Sriramdayal/Unsloth-LLM-finetuningv1/issues).

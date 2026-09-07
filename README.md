
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
  &nbsp;
  <a href="https://huggingface.co/sriram279/Leet-Reason-Qwen0.5" target="_blank">
    <img src="https://img.shields.io/badge/HuggingFace-Leet_Reason_Qwen0.5-orange?style=flat-square" alt="HuggingFace Model"/>
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
| 🏗️ **Modular architecture** | Clean subpackages: `platform` · `data` · `training` · `service` · `soup` · `ui` |
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
> via `src/platform/dll_bootstrap.py`. No NVIDIA CUDA Toolkit installation required.

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
from src import DataProcessor, ModelConfig, ModelRunner, TrainConfig, train_model

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

# Custom Dataset Column Mapping (Optional)
# dataset_style: "alpaca"
# dataset_instruction_column: "instruction"
# dataset_output_column: "output"
# dataset_input_column: "input"
```

```bash
uv run unsloth-cli train --config config.yaml
```

### 📋 Custom Dataset Column Mapping

If your dataset does not use standard column names (such as `greengerong/leetcode` which has columns like `content`, `python`, `java`, etc.), you can override the column mappings directly in your `config.yaml`:

```yaml
dataset_style: "alpaca"                 # Standard Alpaca template formatting
dataset_instruction_column: "content"   # Problem/Prompt description
dataset_output_column: "python"         # The target code/explanation output (python, java, c++, javascript)
dataset_input_column: null              # Optional background/context column
```

This prevents the data parser from defaulting to a single-text column match and ensures your instructions and target responses are properly mapped and tokenized.

---

## 🔄 Merging LoRA Weights & Pushing to Hugging Face Hub

Once fine-tuning is complete, you can merge your trained LoRA adapter weights back into the base model (saving it as a standalone 16-bit precision model) and optionally push it directly to the Hugging Face Hub.

Use the `scripts/merge_and_push.py` utility:

### 1. Merge locally
This loads your adapter and its corresponding base model, merges the weights, and saves the full model locally:
```bash
uv run python scripts/merge_and_push.py --adapter outputs/qwen_leetcode
```
*(By default, this will save the merged model to `outputs/qwen_leetcode_merged`.)*

### 2. Merge and push to Hugging Face Hub
To merge the weights and upload the full model to your Hugging Face repository in a single command, specify the `--hub_id` parameter:
```bash
# Set your Hugging Face write token
# Windows PowerShell:
$env:HF_TOKEN="your_hf_write_token_here"

# Linux / macOS / Git Bash:
export HF_TOKEN="your_hf_write_token_here"

# Merge and upload
uv run python scripts/merge_and_push.py \
  --adapter outputs/qwen_leetcode \
  --hub_id "your-username/qwen-0.5b-leetcode"
```

---

## 🍲 Soup Integration (Optional)

[Soup](https://github.com/MakazhanAlpamys/Soup) is used as an **external tool** (subprocess, never imported) for model export and config generation. Install it with:

```bash
pip install -e ".[soup]"
```

### Export a trained model to GGUF

```python
from src.soup import SoupClient

client = SoupClient()
if client.is_available():
    gguf_path = client.export_gguf("./outputs/quickstart", quant="q4_k_m")
```

Or run the example end to end:

```bash
python examples/soup_export.py ./outputs/quickstart --quant q4_k_m
```

### Generate `soup.yaml` from repo configs

```python
from src import ModelConfig, TrainConfig
from src.soup import SoupConfig

soup_config = SoupConfig.from_model_train_configs(model_config, train_config)
soup_config.to_yaml("soup.yaml")  # then: soup train --config soup.yaml
```

See `examples/configs/soup_qwen.yaml` for a complete example equivalent to `examples/configs/train_qwen.yaml`.

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
├── __init__.py               # Public API (v0.3.0): ModelConfig, ModelRunner, train_model, ...
├── cli.py                    # Unified CLI: train + infer subcommands
├── config.py                 # ModelConfig / TrainConfig dataclasses
├── train.py                  # Shim → src.training (backward compat)
├── finetuning_agent.py       # smolagents assistant (standalone)
│
├── platform/                 # Cross-platform backend layer
│   ├── hardware.py           # HardwareManager (CUDA / MLX / MPS / CPU detection)
│   ├── dll_bootstrap.py      # Platform DLL/library bootstrap for llama-cpp-python
│   ├── factory.py            # ModelFactory (Unsloth / QLoRA / MLX / transformers)
│   └── model_runner.py       # ModelRunner: GGUF/llama.cpp + HF/MLX dual-backend
│
├── data/                     # Data ETL layer
│   └── processor.py          # DataProcessor (load, format, tokenize)
│
├── training/                 # Training orchestration
│   ├── orchestrator.py       # train_model() — SFTTrainer loop
│   └── mlx_trainer.py        # train_mlx() — Apple MLX via mlx_lm.lora
│
├── soup/                     # Soup CLI integration (optional, subprocess-based)
│   ├── client.py             # SoupClient — export_gguf() and other soup commands
│   └── config.py             # SoupConfig — generates soup.yaml from repo configs
│
├── service/                  # REST API layer
│   ├── app.py                # FastAPI app factory (create_app)
│   ├── dependencies.py       # ModelCache (LRU) + JobRegistry singletons
│   ├── schemas/requests.py   # Pydantic request/response models
│   └── routes/               # health.py, inference.py, training.py
│
├── ui/                       # Gradio GUI
│   └── gradio_app.py         # Fine-Tuning Studio (unsloth-gui)
│
├── core/ & utils/            # Backward-compat re-export shims (do not add new code)
└── api/                      # Backward-compat re-export shims (real code lives in service/)

examples/
├── quickstart.py             # Minimal end-to-end training run
├── api_training.py           # REST API: enqueue job + poll status (stdlib only)
├── mlx_inference.py          # MLX inference on Apple Silicon
├── data_etl.py               # DataProcessor preview run
├── soup_export.py            # GGUF export via Soup
└── configs/
    ├── train_qwen.yaml       # Example training config (Qwen-0.5B + LeetCode)
    └── soup_qwen.yaml        # Equivalent soup.yaml

scripts/
├── app.py                    # Thin wrapper → src.ui (unsloth-gui entry)
├── run_training.py           # Script entry point
├── run_inference.py          # Script entry point
├── merge_and_push.py         # Merge LoRA weights & push to HF Hub
└── smoke_test.py             # Import validation (new + legacy paths)
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

# Run Soup integration tests only
uv run pytest tests/test_soup.py -v
# (or: make test-soup)

# Run tests and generate coverage reports (terminal and HTML)
./scripts/run_all_tests.sh
```

### Running the Smoke Test
Verifies all package imports — new canonical paths (`src.platform.*`, `src.training.*`,
`src.service.*`, `src.soup.*`, `src.ui.*`) plus legacy backward-compat shims —
and reports the active backend:
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
* [Soup](https://github.com/MakazhanAlpamys/Soup) — Post-training ops CLI (GGUF export, config generation)

---

## 🤝 Contribution and Issues

Feel free to contribute by opening a Pull Request or raising an issue on
[GitHub](https://github.com/Sriramdayal/Unsloth-LLM-finetuningv1/issues).

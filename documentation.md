# Unsloth Finetuning Package — API Documentation (v0.2.0)

This guide covers the Python API for `unsloth-finetuning`.  
The package now uses a **Windows-native GPU stack** — no Triton, no WSL2 required.

| Layer | Technology |
|---|---|
| Training | `unsloth` (Linux CUDA) · `bitsandbytes` 4-bit (Windows/Linux CUDA) · `mlx-lm` (macOS Apple Silicon) |
| Inference (GGUF) | `llama-cpp-python` (Metal on macOS, CUBLAS on Windows, CUDA on Linux) |
| Inference (HF) | `mlx-lm` (macOS Apple Silicon) · `transformers` generate() (Windows/Linux/CPU) |

---

## 1. Installation

### Windows (Native GPU — Recommended)
```bash
uv pip install -e ".[gui]"

# Install llama-cpp-python CUBLAS wheel (GPU-accelerated GGUF inference)
uv pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

### Linux / WSL2 (original Unsloth kernels)
```bash
uv pip install -e ".[unsloth,gpu,gui]"
```

### macOS (Apple Silicon — MLX)
```bash
uv pip install -e ".[macos,gui]"

# Install llama-cpp-python with Metal support (for GGUF inference)
CMAKE_ARGS="-DGGML_METAL=on" uv pip install llama-cpp-python --no-cache
```

---

## 2. ModelRunner API

`ModelRunner` is the high-level facade that handles model loading, LoRA patching,
and inference. It auto-selects the backend based on the model path and hardware:

| `model_name_or_path` ends in | Platform / Acceleration | Backend used |
|---|---|---|
| `.gguf` | All | llama-cpp-python (CUBLAS/CUDA/Metal/CPU) |
| HF repo ID or directory | macOS Apple Silicon (MLX installed) | mlx-lm |
| HF repo ID or directory | Windows / Linux (CUDA) | transformers (bitsandbytes / torch) |
| HF repo ID or directory | Any (CPU) | transformers (torch float32) |

### Training Setup
```python
from src import ModelConfig
from src.core.model_runner import ModelRunner

config = ModelConfig(
    model_name_or_path="unsloth/llama-3-8b-bnb-4bit",
    load_in_4bit=True,   # NF4 4-bit QLoRA via bitsandbytes
    lora_r=16,
    lora_alpha=32,
)

runner = ModelRunner(config)
model, tokenizer = runner.setup_for_training()  # applies LoRA automatically
```

### Inference — GGUF (llama.cpp CUBLAS, fastest on Windows)
```python
from src import ModelConfig
from src.core.model_runner import ModelRunner

config = ModelConfig(model_name_or_path="path/to/model.gguf")
runner = ModelRunner(config)
runner.setup_for_inference()

response = runner.generate("Explain gradient descent in simple terms.")
print(response)
```

### Inference — HuggingFace / safetensors + LoRA adapter
```python
config = ModelConfig(model_name_or_path="meta-llama/Llama-3-8b-hf")
runner = ModelRunner(config)
runner.setup_for_inference(adapter_path="outputs/lora_adapters")

response = runner.generate("Write a Python quicksort.")
print(response)
```

---

## 3. Dataset Preparation

`DataProcessor` handles loading, formatting, and tokenization.

```python
from src import ModelConfig, TrainConfig
from src.data import DataProcessor

model_cfg = ModelConfig(model_name_or_path="unsloth/llama-3-8b-bnb-4bit")
train_cfg = TrainConfig(dataset_name="yahma/alpaca-cleaned")

processor = DataProcessor(model_cfg, train_cfg, tokenizer)
processor.load_dataset(split="train")

# style: "alpaca" | "chat" | "movie_recommender" | "auto"
dataset = processor.format_and_tokenize(style="alpaca")
```

### Dynamic Column Detection
The processor auto-detects dataset structure:
- **Pre-formatted**: `text` or `content` columns used directly.
- **Chat**: `conversations` or `messages` → ChatML format.
- **Instructional**: `instruction`, `input`, `output`.
- **Fallback**: Positional columns (0 = instruction, 1 = output).

---

## 4. ModelFactory (Advanced)

Use `ModelFactory` directly for more control.

```python
from src.core.factory import ModelFactory
from src.config import ModelConfig

config = ModelConfig(
    model_name_or_path="unsloth/llama-3-8b-bnb-4bit",
    load_in_4bit=True,
)

# Load base model with bitsandbytes 4-bit NF4 quantisation
model, tokenizer = ModelFactory.create_model_and_tokenizer(config)

# Apply PEFT LoRA adapters
model = ModelFactory.apply_lora(model, config)

# Switch to inference mode (eval + no grad)
model = ModelFactory.prepare_for_inference(model)
```

### BitsAndBytes Config details
When `load_in_4bit=True`, the factory uses:
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # Best accuracy for QLoRA
    bnb_4bit_compute_dtype=bfloat16,     # or float16 if bfloat16 unsupported
    bnb_4bit_use_double_quant=True,      # Nested quant saves ~0.4 GB extra
)
```

---

## 5. Training Orchestration

`train_model()` wraps TRL's `SFTTrainer` with hardware-aware defaults, or delegates to Apple-native `mlx-lm` on macOS.

```python
from src.train import train_model
from src.config import ModelConfig, TrainConfig

train_cfg = TrainConfig(
    dataset_name="yahma/alpaca-cleaned",
    output_dir="outputs/my_model",
    batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
)

stats, output_path = train_model(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    train_config=train_cfg,
    model_config=model_cfg,
)
```

**Key behaviours:**
- **macOS Apple Silicon (MLX)**: If MLX is installed, training is automatically offloaded to `mlx_lm.lora` via a subprocess, converting the dataset to JSONL on-the-fly.
- **Auto-Precision**: Selects `bf16` if supported by GPU, else `fp16`.
- **Optimized Optimizer**: Uses `adamw_8bit` on CUDA, `adamw_torch` on CPU.
- **Memory Management**: Clears CUDA cache before and after training.
- **Mock Mode**: `model_config.use_mock = True` simulates training without a GPU.

---

## 6. CLI Reference

```
unsloth-cli <subcommand> [options]
```

| Subcommand | Purpose | Key Arguments |
|---|---|---|
| `train` | Start fine-tuning | `--config`, `--model_name_or_path`, `--dataset_name` |
| `infer` | Run inference | `--model`, `--prompt` |

### Config File (`config.yaml`)
```yaml
model_name_or_path: "unsloth/llama-3-8b-bnb-4bit"
load_in_4bit: true
lora_r: 16
lora_alpha: 32
dataset_name: "yahma/alpaca-cleaned"
learning_rate: 0.0002
num_train_epochs: 3
output_dir: "outputs/my_model"
```

```bash
uv run unsloth-cli train --config config.yaml
```

---

## 7. Windows DLL Bootstrap

On Windows, `llama-cpp-python` requires `cudart64_12.dll` to GPU-accelerate inference.
The package resolves this automatically from PyTorch's bundled CUDA runtime —
**no separate CUDA Toolkit install is needed**.

The bootstrap (`src/utils/llama_loader.py`) runs transparently when you call
`runner.setup_for_inference()` with a `.gguf` path. You can also call it manually:

```python
from src.utils.llama_loader import bootstrap_windows_cuda_dlls
bootstrap_windows_cuda_dlls()  # Safe to call multiple times (no-op after first call)
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLAMA_N_GPU_LAYERS` | `-1` | GPU layers for llama.cpp. `-1` = all on GPU. |

---

## 8. HardwareManager

```python
from src.utils.env import HardwareManager

# Print a system report
HardwareManager.log_system_report()

# Get the best device
device = HardwareManager.get_device()  # "cuda" | "mps" | "cpu"

# Get VRAM stats
stats = HardwareManager.get_memory_stats()
# {'device': 'NVIDIA GeForce RTX 4060 ...', 'total_gb': 8.0, ...}
```

## 9. REST API Server

The package includes a FastAPI REST API for programmatically fine-tuning and running inference.

### Starting the Server

To launch the server locally on port `8000`:

```bash
uv run uvicorn src.api:app
```

> [!WARNING]
> On Windows, do **not** use the `--reload` flag (e.g. `uvicorn src.api:app --reload`). The Uvicorn reloading process spawns a child process using Python's `multiprocessing` library, which often uses the global Python installation rather than the active virtual environment, leading to a silent crash (`ModuleNotFoundError: No module named 'torch'`).

Once the API starts and prints `Application startup complete.`, the interactive API documentation (Swagger UI) is available at:
* **Interactive UI Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Static ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

### API Endpoint Reference

#### 1. System Health Check (`GET /api/v1/health`)
Checks API status and reports the active hardware backend.

* **PowerShell**:
  ```powershell
  Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -Method Get
  ```
* **Bash / curl**:
  ```bash
  curl http://127.0.0.1:8000/api/v1/health
  ```
* **Expected Response**:
  ```json
  {
    "status": "healthy",
    "gpu_available": false,
    "gpu_name": null,
    "platform": "Windows",
    "backend": "CPU (transformers float32 + llama.cpp CPU)"
  }
  ```

#### 2. Start Fine-Tuning Job (`POST /api/v1/train`)
Enqueues a fine-tuning job to run in a background thread and immediately returns a `job_id`.

* **PowerShell**:
  ```powershell
  $body = @{
      model_name_or_path = "unsloth/llama-3-8b-bnb-4bit"
      dataset_name = "yahma/alpaca-cleaned"
      lora_r = 16
      learning_rate = 0.0002
      num_train_epochs = 1
      output_dir = "outputs/api_run_model"
  } | ConvertTo-Json

  Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/train" -Method Post -Body $body -ContentType "application/json"
  ```
* **Bash / curl**:
  ```bash
  curl -X POST http://127.0.0.1:8000/api/v1/train \
    -H "Content-Type: application/json" \
    -d '\''{
      "model_name_or_path": "unsloth/llama-3-8b-bnb-4bit",
      "dataset_name": "yahma/alpaca-cleaned",
      "lora_r": 16,
      "learning_rate": 0.0002,
      "num_train_epochs": 1,
      "output_dir": "outputs/api_run_model"
    }'\''
  ```
* **Expected Response**:
  ```json
  {
    "job_id": "fc94d081",
    "status": "queued"
  }
  ```

#### 3. Poll Training Job Status (`GET /api/v1/train/{job_id}/status`)
Retrieves the training state, progress percentage, and output folder location.

* **PowerShell**:
  ```powershell
  Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/train/fc94d081/status" -Method Get
  ```
* **Bash / curl**:
  ```bash
  curl http://127.0.0.1:8000/api/v1/train/fc94d081/status
  ```
* **Expected Response**:
  ```json
  {
    "job_id": "fc94d081",
    "status": "running",
    "progress": 25.5,
    "output_dir": null,
    "error_message": null
  }
  ```

#### 4. Run Hugging Face / PEFT Inference (`POST /api/v1/infer`)
Executes single-prompt text generation using `transformers` with optional LoRA adapters.

* **PowerShell**:
  ```powershell
  $body = @{
      model_path = "unsloth/llama-3-8b-bnb-4bit"
      prompt = "What is fine-tuning?"
      max_tokens = 256
      temperature = 0.7
  } | ConvertTo-Json

  Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/infer" -Method Post -Body $body -ContentType "application/json"
  ```
* **Bash / curl**:
  ```bash
  curl -X POST http://127.0.0.1:8000/api/v1/infer \
    -H "Content-Type: application/json" \
    -d '\''{
      "model_path": "unsloth/llama-3-8b-bnb-4bit",
      "prompt": "What is fine-tuning?",
      "max_tokens": 256,
      "temperature": 0.7
    }'\''
  ```

#### 5. Run GGUF Inference (`POST /api/v1/infer/gguf`)
Runs high-performance inference using local `.gguf` files via `llama.cpp`.

* **PowerShell**:
  ```powershell
  $body = @{
      model_path = "models/Llama-3-8b-Q4_K_M.gguf"
      prompt = "Hello!"
      max_tokens = 64
  } | ConvertTo-Json

  Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/infer/gguf" -Method Post -Body $body -ContentType "application/json"
  ```
* **Bash / curl**:
  ```bash
  curl -X POST http://127.0.0.1:8000/api/v1/infer/gguf \
    -H "Content-Type: application/json" \
    -d '\''{
      "model_path": "models/Llama-3-8b-Q4_K_M.gguf",
      "prompt": "Hello!",
      "max_tokens": 64
    }'\''
  ```


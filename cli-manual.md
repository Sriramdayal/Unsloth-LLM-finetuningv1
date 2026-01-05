# Unsloth Finetuning CLI Manual

This manual provides detailed instructions on how to use the headless command-line interface (CLI) for the Unsloth Enterprise Pipeline. 

The CLI is located at `src/cli.py` and is designed for automated, reproducible training runs without a GUI.

## 1. Quick Start

### Installation
This project uses `uv` for dependency management, but standard `pip` is also supported.

**Using `pip`:**
```bash
pip install -e .
```

**Using `uv` (Recommended):**
```bash
uv sync
```

### Basic Usage
You can run the CLI as a python module or via specific scripts depending on your environment.

**Standard Run (GPU Required):**
```bash
# Via python module
python -m src.cli \
    --model_name_or_path "unsloth/llama-3-8b-bnb-4bit" \
    --dataset_name "yahma/alpaca-cleaned" \
    --num_train_epochs 1
```

**Using `uv`:**
```bash
uv run python -m src.cli ...
```

### Dry Run & Mock Mode (CPU/CI Friendly)
If you are running on a machine without a GPU (like a CI runner or basic VM), you **MUST** use `--use_mock True`. 

Use the `--dry_run True` flag to load the model and dataset, perform validation, and verify formatting *without* starting the actual training loop.

**Example: Running on CPU (Mock Mode)**
```bash
uv run python -m src.cli \
    --model_name_or_path "unsloth/llama-3-8b-bnb-4bit" \
    --dataset_name "yahma/alpaca-cleaned" \
    --dry_run True \
    --use_mock True
```
> **Note**: In Mock Mode, you might see warnings about Unsloth imports or optimizations. These can be safely ignored as Unsloth requires a GPU to fully initialize.

## 2. Configuration Files (Recommended)

For complex setups, it is best to use a YAML or JSON configuration file. This keeps your training parameters version-controlled.

**Example `config.yaml`:**
```yaml
model_name_or_path: "unsloth/llama-3-8b-bnb-4bit"
load_in_4bit: true
max_seq_length: 2048
lora_r: 16
lora_alpha: 16
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

dataset_name: "yahma/alpaca-cleaned"
output_dir: "outputs/production_v1"
batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
num_train_epochs: 3
push_to_hub: false
dry_run: false
```

**Run with Config:**
```bash
python -m src.cli config.yaml
```

## 3. Argument Reference

### General
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dry_run` | bool | `False` | Validate setup without training. |

### Model Configuration (`ModelConfig`)
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model_name_or_path` | str | **Required** | HuggingFace model ID or path. |
| `--load_in_4bit` | bool | `True` | Use 4-bit precision (BitsAndBytes). |
| `--max_seq_length` | int | `2048` | Maximum context length. |
| `--lora_r` | int | `16` | LoRA rank (attention dimension). |
| `--lora_alpha` | int | `16` | LoRA scaling parameter. |
| `--lora_dropout` | float | `0.0` | Dropout probability for LoRA layers. |
| `--target_modules` | list | `[all linear]` | Modules to apply LoRA (e.g. `q_proj`, `k_proj`). |
| `--random_state` | int | `3407` | Random seed for reproducibility. |
| `--use_mock` | bool | `False` | Use dummy model/tokenizer for CPU testing. |

### Training Configuration (`TrainConfig`)
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset_name` | str | **Required** | HuggingFace dataset ID. |
| `--dataset_text_column`| str | `"text"` | Column containing pre-formatted text. |
| `--dataset_num_samples`| int | `None` | Limit samples for debugging. |
| `--output_dir` | str | `"outputs"` | Directory to save checkpoints. |
| `--batch_size` | int | `2` | Batch size per device. |
| `--gradient_accumulation_steps` | int | `4` | Gradient accumulation steps. |
| `--learning_rate` | float | `2e-4` | Initial learning rate. |
| `--num_train_epochs` | float | `1.0` | Total training epochs. |
| `--max_steps` | int | `-1` | Max steps (overrides epochs if > 0). |
| `--push_to_hub` | bool | `False` | Push model to Hub after training. |
| `--hub_model_id` | str | `None` | Repository name on Hub. |

## 4. Troubleshooting

**"AttributeError: 'NoneType' object has no attribute..."**
- This usually means `Unsloth` failed to load because no GPU was detected.
- **Fix**: Use `--use_mock True` if checking logic on CPU, or run on a GPU machine.

**"Unsloth should be imported before [transformers]"**
- This warning is expected if running on CPU/Mock mode where Unsloth fails to initialize fully.
- It can be ignored in Mock Mode.
- On a GPU machine, ensure you are not importing `transformers` manually before `src.cli`.

**"CUDA out of memory"**
- Reduce `--batch_size`.
- Reduce `--max_seq_length`.
- INCREASE `--gradient_accumulation_steps` to maintain effective batch size.

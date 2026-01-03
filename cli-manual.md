# Unsloth Finetuning CLI Manual

This manual provides detailed instructions on how to use the headless command-line interface (CLI) for the Unsloth Enterprise Pipeline. 

The CLI is located at `scripts/cli.py` and is designed for automated, reproducible training runs without a GUI.

## 1. Quick Start

### Installation
Ensure you have installed the package:
```bash
# From PyPI (coming soon)
pip install unsloth-finetuning

# OR from source
pip install -e .
```

### Basic Usage
You can run the CLI directly using the installed command:

```bash
unsloth-cli \
    --model_name_or_path "unsloth/llama-3-8b-bnb-4bit" \
    --dataset_name "yahma/alpaca-cleaned" \
    --num_train_epochs 1
```

Or via python module:
```bash
python -m src.cli ...
```


### Dry Run
Use the `--dry_run` flag to load the model and dataset, perform validation, and verify formatting *without* starting the actual training loop. This is useful for checking configuration before a long job.

```bash
python scripts/cli.py \
    --model_name_or_path "unsloth/llama-3-8b-bnb-4bit" \
    --dataset_name "yahma/alpaca-cleaned" \
    --dry_run True
```

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
```

**Run with Config:**
```bash
python scripts/cli.py config.yaml
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
| `--grad_accum_steps` | int | `4` | Gradient accumulation steps. |
| `--learning_rate` | float | `2e-4` | Initial learning rate. |
| `--num_train_epochs` | float | `1.0` | Total training epochs. |
| `--max_steps` | int | `-1` | Max steps (overrides epochs if > 0). |
| `--push_to_hub` | bool | `False` | Push model to Hub after training. |
| `--hub_model_id` | str | `None` | Repository name on Hub. |

## 4. Advanced: Dataset Support

The CLI supports dynamic dataset columns:
1.  **Pre-formatted**: If your dataset already has a `text` column (or whatever you set `--dataset_text_column` to), formatting is skipped.
2.  **Standard**: Automatically detects `instruction`, `input` (optional), and `output` columns.
3.  **Arbitrary**: If standard columns are not found, it falls back to using the **first column as instruction** and **second column as output**.

## 5. Troubleshooting

**"Dataset missing required columns"**
- Ensure your dataset is not empty.
- If using custom columns, rely on the automatic fallback or rename them to `instruction`/`output`.
- If using pre-formatted data, ensure the column is named `text` or matched by `--dataset_text_column`.

**"CUDA out of memory"**
- Reduce `--batch_size`.
- Reduce `--max_seq_length`.
- INCREASE `--gradient_accumulation_steps` to maintain effective batch size.

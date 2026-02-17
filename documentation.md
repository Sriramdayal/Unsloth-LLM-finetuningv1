# Unsloth Finetuning Package Documentation

This guide provides detailed instructions on how to use `unsloth-finetuning` as a Python library for building custom training pipelines.

## 1. Installation

### From PyPI (Development Version)
```bash
pip install git+https://github.com/Sriramdayal/Unsloth-LLM-finetuningv1.git
```

## 2. Dataset Preparation

The package includes a robust `DataProcessor` that handles loading, formatting, and proper tokenization (generating labels, masking inputs). Edit dataset and model path from huggingface model and datasets
or upload your dataset in hugging face and format it in Alpaca , ChatML and  ShareGPT style  and  accodingly  format and tokenize  in below code as follows as Alpaca , ChatML and  ShareGPT. Also select the 
datasets in huggingface that support these Alpaca , ChatML and  ShareGPT format style for best results and compatibility while finetuning. The factors affecting training time are Lora rank , dataset size ,
Bigger the model parameters. for more read unsloth finetuning documentation 
[Unsloth](https://github.com/unslothai/unsloth)
* [Unsloth](https://github.com/unslothai/unsloth)
* [HuggingFace Transformers](https://github.com/huggingface/transformers)
* [TRL](https://github.com/huggingface/trl)
* [PEFT](https://github.com/huggingface/peft)

### Loading & Formatting
```python
import unsloth
from unsloth import FastLanguageModel
from src.config import ModelConfig, TrainConfig
from src.data import DataProcessor
from transformers import AutoTokenizer

# 1. Setup Configs
model_config = ModelConfig(
    model_name_or_path="unsloth/mistral-7b-bnb-4bit",
    load_in_4bit=True
)
train_config = TrainConfig(
    dataset_name="bowen-upenn/PersonaMem-v2",
    dataset_text_column="text"
)

# 2. Initialize Tokenizer (usually comes from model, using dummy here for logic demo)
tokenizer = AutoTokenizer.from_pretrained("unsloth/mistral-7b-bnb-4bit")

# 3. Process Data
processor = DataProcessor(model_config, train_config, tokenizer)
processor.load_dataset(split="train_text") # Corrected: Specify the 'train_text' split

# Optional: Inspect detected columns
# processor.validate_columns()

# 4. Format & Tokenize
# This applies the prompt template and tokenizes the result
dataset = processor.format_and_tokenize(style="alpaca")
```

### Dynamic Column Support
The processor automatically detects columns.
- **Pre-formatted**: If a `text` column exists, it uses it directly.
- **Automatic**: Scans for `instruction`, `input`, `output`, `prompt`, `response`, etc.
- **Fallback**: Uses the first two columns as `instruction` and `output` if no standard names are found.

## 3. Model Loading & Configuration

We use `unsloth.FastLanguageModel` for optimized loading.

```python
import unsloth
from unsloth import FastLanguageModel

max_seq_len = 2048

# 1. Load Base Model (If not loaded in previous step)
# model, tokenizer = FastLanguageModel.from_pretrained(...)

# 2. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth", 
    random_state=3407,
)
```

## 4. Training

We recommend using HuggingFace TRL's `SFTTrainer`.

```python
import unsloth
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_len,
    args=TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=60, # or num_train_epochs=1
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
    ),
)

trainer.train()
```

## 5. Inference

For inference, use the `FastLanguageModel.for_inference` context.

```python
import unsloth
from unsloth import FastLanguageModel
FastLanguageModel.for_inference(model)

prompt = "Below is an instruction... ### Instruction:\nExplain quantum computing.\n\n### Response:\n"
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
print(tokenizer.decode(outputs[0]))
```

## 6. Exporting

Save your finetuned adapters or merged model.

```python
# Save LoRA adapters only (Lightweight)
model.save_pretrained("outputs/lora_adapters")

# Save Merged Model (GGUF/VLLM ready)
model.save_pretrained_merged("outputs/merged_model", tokenizer, save_method="merged_16bit")
```

## 7. CLI Integration API

The CLI can be controlled programmatically using its configuration dataclasses. This is useful for building custom grid searches or automated orchestration scripts.

### Programmatic Config Loading
```python
import unsloth
from unsloth import FastLanguageModel
from src.config import ModelConfig, TrainConfig
from transformers import HfArgumentParser

parser = HfArgumentParser((ModelConfig, TrainConfig))

# Load from YAML
model_cfg, train_cfg = parser.parse_json_file(json_file="configs/default_config.yaml")

# Override specific parameters
train_cfg.num_train_epochs = 5
train_cfg.learning_rate = 5e-5

# Delegate to training engine
from src.train import train_model
# ... model loading logic ...
```

### CLI Command Layout
The `unsloth-cli` provides a direct interface to the `src.cli:main` function.

| Interface | Input Format | Primary Action |
| :--- | :--- | :--- |
| **Config Mode** | `unsloth-cli <file>.yaml` | Loads all params from file. |
| **Flag Mode** | `unsloth-cli --key val` | Parses individual arguments. |
| **Mixed Mode** | Not supported | CLI flags take precedence when no file is provided. |

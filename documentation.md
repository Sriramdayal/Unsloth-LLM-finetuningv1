# Unsloth Finetuning Package Documentation

This guide provides detailed instructions on how to use `unsloth-finetuning` as a Python library for building custom training pipelines.

## 1. Installation

### From PyPI
```bash
pip install unsloth-finetuning
```

### From Source
```bash
git clone https://github.com/Sriramdayal/Unsloth-LLM-finetuningv1.git
cd Unsloth-LLM-finetuningv1
pip install -e .
```

## 2. Dataset Preparation

The package includes a robust `DataProcessor` that handles loading, formatting, and proper tokenization (generating labels, masking inputs).

### Loading & Formatting
```python
from src.config import ModelConfig, TrainConfig
from src.data import DataProcessor
from transformers import AutoTokenizer

# 1. Setup Configs
model_config = ModelConfig(
    model_name_or_path="unsloth/mistral-7b-bnb-4bit",
    load_in_4bit=True
)
train_config = TrainConfig(
    dataset_name="yahma/alpaca-cleaned",
    dataset_text_column="text"
)

# 2. Initialize Tokenizer (usually comes from model, using dummy here for logic demo)
tokenizer = AutoTokenizer.from_pretrained("unsloth/mistral-7b-bnb-4bit")

# 3. Process Data
processor = DataProcessor(model_config, train_config, tokenizer)
processor.load_dataset()

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
from unsloth import FastLanguageModel

max_seq_len = 2048

# 1. Load Base Model
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/mistral-7b-bnb-4bit",
    load_in_4bit=True,
    max_seq_length=max_seq_len,
    dtype=None,
)

# 2. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_gradient_checkpointing="unsloth",
)
```

## 4. Training

We recommend using HuggingFace TRL's `SFTTrainer`.

```python
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

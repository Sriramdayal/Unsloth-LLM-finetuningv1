
---

# 🚀 LLM Finetuning with Unsloth + LoRA + TRL

**Fast, lightweight, and scalable finetuning for open-source LLMs**

  <img width="3194" height="992" alt="17638338761695712928855779246747" src="https://github.com/user-attachments/assets/6c834e71-ad14-40b0-a26f-27783752c07f" />


<p align="center">
  <img src="https://img.shields.io/badge/Framework-Unsloth-blue?logo=python" />
  <img src="https://img.shields.io/badge/LoRA-PEFT-green" />
  <img src="https://img.shields.io/badge/Trainer-TRL-yellow" />
  <img src="https://img.shields.io/badge/Quantization-4bit-orange" />
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
  <a href="https://huggingface.co/black279/Qwen_LeetCoder" target="_blank">
    <img src="https://img.shields.io/badge/HuggingFace-Qwen_LeetCoder-orange?style=flat-square" alt="HuggingFace Model"/>
  </a>
</p>

---

## 📌 Overview

This repo provides a **general-purpose template** for finetuning HuggingFace models using:

* ⚡ **Unsloth-optimized kernels** (2× faster)
* 🎯 **LoRA adapters** (low-cost parameter-efficient tuning)
* 🧠 **4-bit inference/training** for consumer GPUs
* 🏋️ **TRL’s SFTTrainer**
* 🔄 **Optional LoRA merge/export for deployment**

Works for **instruction tuning, chat models, summarization, domain adaptation**, and more.

---

## 🚀 Dual-Mode Usage

This repository has been refactored into a maintainable Python package structure that supports two primary modes of operation.

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Mode 1: Headless CLI (Automation)

The CLI is designed for automated training pipelines and supports configuration via YAML/JSON files.

**Basic Usage:**
```bash
python scripts/cli.py --model_name_or_path "unsloth/mistral-7b-bnb-4bit" --dataset_name "imdb"
```

**Using a Config File:**
```bash
python scripts/cli.py configs/example.yaml
```

**Dry Run (Verify config without training):**
```bash
python scripts/cli.py --dry_run
```

### 3. Mode 2: No-Code Studio (Interactive)

Launch the Gradio-based web interface for an interactive fine-tuning experience.

```bash
python scripts/app.py
```
*Open your browser at `http://localhost:7860`*

The Studio provides tabs for:
- **Model & Data**: Select base models and datasets.
- **Training Params**: Configure LoRA rank, epochs, learning rate, etc.
- **Monitor**: View real-time training logs.

---

## ☁️ Run on Google Colab

You can easily run this project on Google Colab by cloning the repository.

1.  **Open a new Colab Notebook.**
2.  **Run the following in a code cell to clone and install:**

    ```python
    !git clone https://github.com/Sriramdayal/Unsloth-LLM-finetuningv1.git
    %cd Unsloth-LLM-finetuningv1
    !pip install -r requirements.txt
    ```

3.  **Run Training (CLI Mode):**

    ```python
    !python scripts/cli.py --model_name_or_path "unsloth/mistral-7b-bnb-4bit" --dataset_name "imdb"
    ```

---


---

## 📥 Dataset Preparation

Your dataset should have a `"text"` field for SFT-style training.

```python
from datasets import load_dataset

ds = load_dataset("fromHuggingfacedatasets/dataset_name")

def format_example(row):
    return {
        "text": f"""### Instruction:
{row.get("instruction", "")}

### Input:
{row.get("input", "")}

### Response:
{row.get("output", "")}"""
    }

train_ds = ds["train"].map(format_example)
train_ds = train_ds.remove_columns(
    [c for c in train_ds.column_names if c != "text"]
)
```

---

## 🧠 Load Model with Unsloth

```python
from unsloth import FastLanguageModel

model_name = "fromHuggingfacemodels/model-herewith1bparamsandunslothsupportedmodels"
max_seq_len = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    load_in_4bit=True,
    max_seq_length=max_seq_len,
)
```

---

## 🔌 Add LoRA Adapters

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=128,
    lora_dropout=0,
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",
    ],
    use_gradient_checkpointing="unsloth",
)
```

---

## ⚡ Enable Fast Training

```python
model = FastLanguageModel.for_training(model)
```

---

## 🏋️ Training with TRL

```python
from trl import SFTTrainer
from transformers import TrainingArguments

#tweak or add the parameters according to need, read the TRL,LORA,UNSLOTH documentation for meinfo
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    dataset_text_field="text",
    max_seq_length=max_seq_len,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=25,
        output_dir="outputs",
        optim="adamw_8bit",
        save_strategy="epoch",
    ),
)

trainer.train()
```

---

## � Merge LoRA Weights (Optional)

### Merge LoRA + Base for Deployment
model deployment in huggingface  example: click below badge to visit
<!-- HuggingFace Model Badge -->
  <a href="https://huggingface.co/black279/Qwen_LeetCoder" target="_blank">
    <img src="https://img.shields.io/badge/HuggingFace-Qwen_LeetCoder-orange?style=flat-square" alt="HuggingFace Model"/>
  </a>
</p>

```python
merged = model.merge_and_unload()
merged.save_pretrained("outputs/merged")
tokenizer.save_pretrained("outputs/merged")
```

### Save LoRA only

```python
model.save_pretrained("outputs/lora")
tokenizer.save_pretrained("outputs/lora")
```

---

## 🧪 Inference Example

```python
prompt = "Explain reinforcement learning in simple terms."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## ⏱ Training Time (Estimates)

| GPU               | Small dataset (50k) | Large dataset (500k–800k) |
| ----------------- | ------------------- | ------------------------- |
| **A100**          | 20–40 min           | 5–7 hours                 |
| **RTX 4090**      | 1–2 hrs             | 18–22 hrs                 |
| **3090 / 4070Ti** | 1.5–3 hrs           | 28–34 hrs                 |
| **Tesla T4**      | 4–6 hrs             | 55–70 hrs                 |

> Training time scales linearly with dataset size and LoRA rank.

---

## ⭐ Notes & Tips

* Reduce dataset size if needed:

```python
train_ds = train_ds.shuffle(seed=42).select(range(100_000))
```

* LoRA rank 64 is high quality; use 16–32 for faster training
* Enable `flash-attn` for memory efficiency on long sequences

---

## 📜 License

MIT

---

## 🔗 Credits

* [Unsloth](https://github.com/unslothai/unsloth)
* [HuggingFace Transformers](https://github.com/huggingface/transformers)
* [TRL](https://github.com/huggingface/trl)
* [PEFT](https://github.com/huggingface/peft)

---


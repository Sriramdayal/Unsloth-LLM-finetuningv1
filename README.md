
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

This repo provides a **Pipeline for finetuning models with datasets seamlessly from Huggingface using Unsloth finetuning framework**:

* ⚡ **Unsloth-optimized kernels** (2× faster, 70% less VRAM)
* 🏗️ **Modular Core Architecture**: Unified `ModelRunner` and `ModelFactory` for training and inference.


## ☁️ Run on Google Colab

You can easily run this project on Google Colab by cloning the repository.

1.  **Open a new Colab Notebook.**
2.  **Run the following in a code cell:**
## 📘 Python API Guide

For detailed instructions on using this repository as a Python library (including `DataProcessor`, configuration, and custom training loops), please refer to the dedicated documentation:

👉 **[Read the Python API Documentation](documentation.md)**

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

##  Contribution and issues: 

You can contribute or raise an issue.


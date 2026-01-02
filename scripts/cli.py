"""
Headless Automation CLI for Unsloth Enterprise Pipeline.
Supports generic config loading (YAML/JSON) and dry runs.
"""

import os
import sys
try:
    from unsloth import FastLanguageModel
except ImportError:
    FastLanguageModel = None
except NotImplementedError:
    # Unsloth raises this on CPU
    FastLanguageModel = None
except Exception:
    FastLanguageModel = None

from dataclasses import dataclass
from typing import Optional
import torch
from transformers import HfArgumentParser

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import ModelConfig, TrainConfig
from src.data import DataProcessor
from src.train import train_model

def main():
    parser = HfArgumentParser((ModelConfig, TrainConfig))
    
    # Allow loading from a config file if provided as a single argument
    if len(sys.argv) == 2 and sys.argv[1].endswith((".json", ".yaml", ".yml")):
        model_config, train_config = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_config, train_config = parser.parse_args_into_dataclasses()

    # Add a custom arguments for dry_run via argparse if not in dataclass (cleaner separation)
    # Or checking environment variable, but here we can check a simple flag manually or add to config if needed permanently.
    # For now, let's look for a specialized flag in sys.argv that HfArgumentParser ignores or add a dummy class.
    
    # Actually, let's just add it to the parser in a standard way using a helper dataclass
    @dataclass
    class CLIConfig:
        dry_run: bool = False
    
    parser = HfArgumentParser((ModelConfig, TrainConfig, CLIConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith((".json", ".yaml", ".yml")):
         model_config, train_config, cli_config = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_config, train_config, cli_config = parser.parse_args_into_dataclasses()

    print(f"--- Configuration ---")
    print(f"Model: {model_config.model_name_or_path}")
    print(f"Dataset: {train_config.dataset_name}")
    print(f"Dry Run: {cli_config.dry_run}")
    print(f"Mock Mode: {model_config.use_mock}")
    print(f"---------------------")

    # 1. Load Model & Tokenizer
    if model_config.use_mock:
        print("[MOCK] Loading dummy tokenizer...")
        from transformers import AutoTokenizer
        # Use a small tokenizer (gpt2) for testing data processing without downloading 7B params
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = None # No model needed for mock training loop logic in train.py
    else:
        print("Loading Model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_config.model_name_or_path,
            max_seq_length = model_config.max_seq_length,
            dtype = None,
            load_in_4bit = model_config.load_in_4bit,
        )

        # 2. Add LoRA Adapters
        print("Adding LoRA Adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r = model_config.lora_r,
            target_modules = model_config.target_modules,
            lora_alpha = model_config.lora_alpha,
            lora_dropout = model_config.lora_dropout,
            bias = "none",
            use_gradient_checkpointing = "unsloth", 
            random_state = model_config.random_state,
            use_rslora = False,
            loftq_config = None,
        )

    # 3. Data Processing
    print("Processing Data...")
    processor = DataProcessor(model_config, train_config, tokenizer)
    processor.load_dataset()
    processor.validate_columns()
    
    dataset = processor.format_and_tokenize(style="alpaca")

    if cli_config.dry_run:
        print("Dry run completed successfully. Data and Model loaded. Exiting.")
        return

    # 4. Training
    train_model(model, tokenizer, dataset, train_config, model_config)
    print("Done!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR]: {e}")
        sys.exit(1)

import unsloth
from unsloth import FastLanguageModel
import os
import sys
import logging
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import HfArgumentParser

# Removed sys.path hack as we are now a package
from .config import ModelConfig, TrainConfig
from .data import DataProcessor
from .train import train_model

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelConfig, TrainConfig))
    
    # Allow loading from a config file if provided as a single argument
    if len(sys.argv) == 2 and sys.argv[1].endswith((".json", ".yaml", ".yml")):
        model_config, train_config = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_config, train_config = parser.parse_args_into_dataclasses()

    logger.info("--- Configuration ---")
    logger.info(f"Model: {model_config.model_name_or_path}")
    logger.info(f"Dataset: {train_config.dataset_name}")
    logger.info(f"Dry Run: {train_config.dry_run}")
    logger.info(f"Mock Mode: {model_config.use_mock}")
    logger.info("---------------------")

    # 1. Load Model & Tokenizer
    if model_config.use_mock:
        logger.info("[MOCK] Loading dummy tokenizer...")
        from transformers import AutoTokenizer
        # Use a small tokenizer (gpt2) for testing data processing without downloading 7B params
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = None # No model needed for mock training loop logic in train.py
    else:
        if FastLanguageModel is None:
            raise RuntimeError(
                "FastLanguageModel is not available. This is likely because Unsloth is not installed "
                "correctly or no GPU is detected. Please install Unsloth with GPU support or use "
                "--use_mock True to run in mock mode."
            )
        logger.info("Loading Model...")
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

    if train_config.dry_run:
        logger.info("Dry run completed successfully. Data and Model loaded. Exiting.")
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

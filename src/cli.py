import unsloth
from unsloth import FastLanguageModel
import os
import sys
import logging
import argparse
from transformers import HfArgumentParser

from .config import ModelConfig, TrainConfig
from .data import DataProcessor
from .train import train_model
from .core.factory import ModelFactory
from .utils.env import HardwareManager

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_training(args):
    """Subcommand for training."""
    parser = HfArgumentParser((ModelConfig, TrainConfig))
    
    # Load from config file or CLI
    if args.config:
        model_cfg, train_cfg = parser.parse_json_file(json_file=os.path.abspath(args.config))
    else:
        model_cfg, train_cfg = parser.parse_args_into_dataclasses(args.unknown)

    # 1. Hardware Report
    HardwareManager.log_system_report()

    # 2. Logic: Mock vs Real
    if model_cfg.use_mock:
        logger.info("[MOCK] Running in simulation mode.")
        tokenizer = None # Processor will load dummy if needed
        model = None
    else:
        model, tokenizer = ModelFactory.create_model_and_tokenizer(model_cfg)
        model = ModelFactory.apply_lora(model, model_cfg)

    # 3. Data
    processor = DataProcessor(model_cfg, train_cfg, tokenizer)
    processor.load_dataset()
    dataset = processor.format_and_tokenize()

    if train_cfg.dry_run:
        logger.info("Dry run complete. Exiting.")
        return

    # 4. Train
    train_model(model, tokenizer, dataset, train_cfg, model_cfg)
    logger.info("Process finished successfully.")

def main():
    parser = argparse.ArgumentParser(description="Unsloth LLM Orchestrator")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Train Command
    train_parser = subparsers.add_parser("train", help="Start fine-tuning")
    train_parser.add_argument("--config", type=str, help="Path to YAML/JSON config file")
    
    # Infer Command (Placeholder for future expansion)
    infer_parser = subparsers.add_parser("infer", help="Run inference (experimental)")
    infer_parser.add_argument("--model", type=str, required=True)
    infer_parser.add_argument("--prompt", type=str, required=True)

    args, unknown = parser.parse_known_args()
    args.unknown = unknown

    if args.command == "train":
        run_training(args)
    elif args.command == "infer":
        logger.info("Inference subcommand coming soon.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

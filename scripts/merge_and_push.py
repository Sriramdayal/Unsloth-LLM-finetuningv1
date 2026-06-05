#!/usr/bin/env python
"""
Script to merge LoRA adapter weights with the base LLM model
and optionally push the merged model to the Hugging Face Hub.
"""

import argparse
import json
import logging
import os
import sys
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights with base model and push to HF Hub.")
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to the trained LoRA adapter directory (e.g., outputs/qwen_leetcode)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="HF repo ID or local path of base model (auto-detected from adapter config if omitted)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Local directory to save the merged model (default: <adapter>_merged)",
    )
    parser.add_argument(
        "--hub_id",
        type=str,
        default=None,
        help="Hugging Face repository ID to push the merged model to (e.g., username/repo-name)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face API token (can also be set via HF_TOKEN environment variable)",
    )

    args = parser.parse_args()

    # 1. Resolve base model path from adapter config if not provided
    adapter_path = os.path.abspath(args.adapter)
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        logger.error(f"Adapter config not found at: {config_path}")
        sys.exit(1)

    base_model = args.base_model
    if not base_model:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                base_model = config_data.get("base_model_name_or_path")
        except Exception as e:
            logger.error(f"Failed to read base model name from config: {e}")
            sys.exit(1)

    if not base_model:
        logger.error("Could not auto-detect base model. Please specify --base_model explicitly.")
        sys.exit(1)

    # 2. Resolve output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = adapter_path.rstrip("/\\") + "_merged"
    output_dir = os.path.abspath(output_dir)

    logger.info(f"Base Model:  {base_model}")
    logger.info(f"Adapter Path:{adapter_path}")
    logger.info(f"Output Dir:  {output_dir}")
    if args.hub_id:
        logger.info(f"Pushing to HF Hub Repo: {args.hub_id}")

    # Set HF token if provided
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # 3. Load Base Model and Tokenizer
    # We load in float16 precision (standard for merging) on CPU or GPU
    logger.info("Loading tokenizer...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
        token=hf_token,
    )

    logger.info("Loading base model (in float16 to support merging)...")
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    base_model_loaded = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
        token=hf_token,
    )

    # 4. Load Adapter and Merge
    logger.info("Loading LoRA adapter...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(
        base_model_loaded,
        adapter_path,
        token=hf_token,
    )

    logger.info("Merging LoRA weights with base model...")
    merged_model = model.merge_and_unload()

    # 5. Save Merged Model locally
    logger.info(f"Saving merged model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    logger.info("Local save complete!")

    # 6. Push to Hugging Face Hub if requested
    if args.hub_id:
        logger.info(f"Pushing merged model to HF Hub repository: {args.hub_id}...")
        merged_model.push_to_hub(
            repo_id=args.hub_id,
            token=hf_token,
        )
        tokenizer.push_to_hub(
            repo_id=args.hub_id,
            token=hf_token,
        )
        logger.info("Hugging Face Hub upload complete!")

if __name__ == "__main__":
    main()

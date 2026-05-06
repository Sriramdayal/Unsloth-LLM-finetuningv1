"""
Training orchestration module for Unsloth Enterprise Pipeline.
Wraps TRL's SFTTrainer with pre-flight checks, mock mode, and cleanup.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import TrainingArguments  # Kept for fallback if needed
from trl import SFTConfig, SFTTrainer

try:
    from .config import ModelConfig, TrainConfig
    from .utils.env import HardwareManager
except ImportError:
    from src.config import ModelConfig, TrainConfig
    from src.utils.env import HardwareManager

logger = logging.getLogger(__name__)


def train_model(
    model,
    tokenizer,
    dataset,
    train_config: TrainConfig,
    model_config: ModelConfig,
    callbacks: Optional[List] = None,
) -> Tuple[Any, str]:
    """
    Orchestrates the training lifecycle with safety checks and logging.

    Returns:
        Tuple of (training_stats, output_directory_path)
    """

    # 1. Pre-flight Checks
    HardwareManager.log_system_report()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # 2. Mock Mode Logic
    if model_config.use_mock:
        logger.info("[MOCK] Simulating training loop...")
        return {"status": "mock_success"}, f"{train_config.output_dir}/mock"

    # 3. Determine precision and optimizer based on hardware
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    dataset_num_proc = getattr(train_config, "dataset_num_proc", -1)
    dataset_num_proc = dataset_num_proc if dataset_num_proc > 0 else (os.cpu_count() or 1)

    training_args = SFTConfig(
        per_device_train_batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        warmup_steps=5,
        max_steps=train_config.max_steps,
        num_train_epochs=train_config.num_train_epochs,
        learning_rate=train_config.learning_rate,
        fp16=not use_bf16 and use_cuda,
        bf16=use_bf16,
        logging_steps=1,
        optim="adamw_8bit" if use_cuda else "adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=model_config.random_state,
        output_dir=train_config.output_dir,
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=getattr(train_config, "dataloader_num_workers", 0),
        dataloader_pin_memory=use_cuda,
        dataset_text_field="text",
        max_seq_length=model_config.max_seq_length,
        packing=getattr(train_config, "packing", False),
        dataset_num_proc=dataset_num_proc,
    )

    # 4. Trainer Configuration
    logger.info("Initializing SFTTrainer...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        callbacks=callbacks,
    )

    # 5. Execution
    logger.info("Training started.")
    stats = trainer.train()

    # 6. Persistence
    logger.info(f"Saving final model to {train_config.output_dir}...")
    trainer.save_model(train_config.output_dir)

    if train_config.push_to_hub and train_config.hub_model_id:
        logger.info(f"Pushing model to Hub: {train_config.hub_model_id}")
        trainer.push_to_hub()

    # 7. Cleanup
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return stats, train_config.output_dir

import unsloth
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
import gc
import logging
try:
    from .config import ModelConfig, TrainConfig
    from .utils.env import HardwareManager
except ImportError:
    from src.config import ModelConfig, TrainConfig
    from src.utils.env import HardwareManager

logger = logging.getLogger(__name__)

def train_model(model, tokenizer, dataset, train_config: TrainConfig, model_config: ModelConfig, callbacks=None):
    """
    Orchestrates the training lifecycle with safety checks and logging.
    """
    
    # 1. Pre-flight Checks
    HardwareManager.log_system_report()
    torch.cuda.empty_cache()
    gc.collect()

    # 2. Mock Mode Logic
    if model_config.use_mock:
        logger.info("MOCK: Simulating training loop...")
        return {"status": "mock_success"}, f"{train_config.output_dir}/mock"

    # 3. Trainer Configuration
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = model_config.max_seq_length,
        args = TrainingArguments(
            per_device_train_batch_size = train_config.batch_size,
            gradient_accumulation_steps = train_config.gradient_accumulation_steps,
            warmup_steps = 5,
            max_steps = train_config.max_steps,
            num_train_epochs = train_config.num_train_epochs,
            learning_rate = train_config.learning_rate,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = model_config.random_state,
            output_dir = train_config.output_dir,
            report_to = "none",
        ),
        callbacks = callbacks
    )

    # 4. Execution
    logger.info("Training started.")
    stats = trainer.train()
    
    # 5. Persistence
    logger.info(f"Saving final model to {train_config.output_dir}...")
    trainer.save_model(train_config.output_dir)
    
    if train_config.push_to_hub:
        trainer.push_to_hub()

    # 6. Cleanup
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    return stats, train_config.output_dir

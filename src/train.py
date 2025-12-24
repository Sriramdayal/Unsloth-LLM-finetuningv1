from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel
import torch
from .config import TrainConfig, ModelConfig

def train_model(model, tokenizer, train_dataset, train_config: TrainConfig, model_config: ModelConfig):
    """
    Executes the training loop.
    """
    # Enable unsloth fast kernels
    model = FastLanguageModel.for_training(model)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=model_config.max_seq_length,
        args=TrainingArguments(
            per_device_train_batch_size=train_config.per_device_train_batch_size,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            warmup_steps=train_config.warmup_steps,
            num_train_epochs=train_config.num_train_epochs,
            learning_rate=train_config.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=train_config.logging_steps,
            optim=train_config.optim,
            weight_decay=train_config.weight_decay,
            lr_scheduler_type=train_config.lr_scheduler_type,
            seed=train_config.seed,
            output_dir=train_config.output_dir,
            save_strategy=train_config.save_strategy,
            save_total_limit=train_config.save_total_limit,
            dataloader_pin_memory=False,
            remove_unused_columns=True, # Often needed with SFTTrainer
        ),
    )
    
    trainer_stats = trainer.train()
    return trainer, trainer_stats

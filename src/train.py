"""
Training Engine for Unsloth Enterprise Pipeline.
Handles model training with custom callbacks and memory management.
"""

try:
    from unsloth import FastLanguageModel
except Exception:
    FastLanguageModel = None
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
import gc
import os
try:
    from .config import ModelConfig, TrainConfig
except ImportError:
    from src.config import ModelConfig, TrainConfig

def train_model(model, tokenizer, dataset, train_config: TrainConfig, model_config: ModelConfig, callbacks=None):
    """
    Executes the training loop.
    
    Args:
        model: The Unsloth/HF model.
        tokenizer: The tokenizer.
        dataset: Formatted dataset.
        train_config: Configuration for training.
        model_config: Configuration for model parameters.
        callbacks: List of Hugging Face trainer callbacks.
        
    Returns:
        tuple: (training_stats, saved_model_path)
    """
    
    # 1. Memory Safety
    torch.cuda.empty_cache()
    gc.collect()

    # --- MOCK TRAINING PATH ---
    if model_config.use_mock:
        import time
        import random
        from transformers.trainer_callback import TrainerState, TrainerControl
        
        print("\n[MOCK MODE] Simulating training on CPU...")
        save_path = os.path.join(train_config.output_dir, "mock_model")
        os.makedirs(save_path, exist_ok=True)
        
        # Simulate steps
        total_steps = 10 if train_config.max_steps == -1 else train_config.max_steps
        if total_steps == -1: total_steps = 10 # Fallback
        
        # Initialize callbacks
        state = TrainerState()
        control = TrainerControl()
        state.global_step = 0
        state.max_steps = total_steps
        
        for step in range(total_steps):
            time.sleep(1.0) # Simulate interaction
            state.global_step += 1
            loss = 2.0 - (step * 0.1) + random.uniform(-0.1, 0.1) # Fake loss curve
            
            logs = {"loss": max(0.0, loss), "step": state.global_step}
            
            if callbacks:
                for callback in callbacks:
                    callback.on_log(TrainingArguments(output_dir=train_config.output_dir), state, control, logs=logs)
            
            print(f"Mock Step {step+1}/{total_steps} - Loss: {loss:.4f}")

        print("[MOCK MODE] Simulation complete.")
        return {"global_step": total_steps, "training_loss": 0.5}, save_path
    # --------------------------

    # 2. Setup Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = train_config.dataset_text_column,
        max_seq_length = model_config.max_seq_length,
        dataset_num_proc = 2,
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
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = model_config.random_state,
            output_dir = train_config.output_dir,
            report_to = "none", # Use custom callback for reporting
        ),
        callbacks = callbacks
    )

    # 3. Train
    print("Starting training...")
    trainer_stats = trainer.train()
    
    # 4. Save
    print("Saving model...")
    save_path = os.path.join(train_config.output_dir, "lora_model")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # 5. Push to Hub if requested
    if train_config.push_to_hub and train_config.hub_model_id:
        print(f"Pushing to Hub: {train_config.hub_model_id}")
        model.push_to_hub(train_config.hub_model_id)
        tokenizer.push_to_hub(train_config.hub_model_id)

    # 6. Memory Cleanup
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    return trainer_stats, save_path

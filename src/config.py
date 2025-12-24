from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    # LoRA Parameters
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    random_state: int = 3407

@dataclass
class TrainConfig:
    dataset_name: str = "Mariodb/movie-recommender-dataset"
    output_dir: str = "outputs"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    logging_steps: int = 25
    save_strategy: str = "epoch"
    save_total_limit: int = 2

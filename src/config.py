"""
Configuration definitions for the Unsloth Enterprise Pipeline.
Uses strict dataclasses compatible with HfArgumentParser for robust argument handling.
"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    """
    Configuration for Model Loading and LoRA parameters.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    load_in_4bit: bool = field(
        default=True,
        metadata={"help": "Whether to load the model in 4-bit precision using bitsandbytes."}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "The maximum sequence length for the model."}
    )
    random_state: int = field(
        default=3407,
        metadata={"help": "Random seed for reproducibility."}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA attention dimension (rank)."}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha parameter."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "LoRA dropout probability."}
    )
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "List of module names to apply LoRA to."}
    )
    use_mock: bool = field(
        default=False,
        metadata={"help": "If True, bypasses model loading and simulates training (CPU safe)."}
    )


@dataclass
class TrainConfig:
    """
    Configuration for Data, Training, and Saving.
    """
    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    output_dir: str = field(
        default="outputs",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    batch_size: int = field(
        default=2,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={"help": "The initial learning rate for AdamW."}
    )
    num_train_epochs: float = field(
        default=1.0,
        metadata={"help": "Total number of training epochs to perform."}
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Overrides num_train_epochs."}
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the model to the Hugging Face Hub after training."}
    )
    hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    dataset_text_column: str = field(
        default="text",
        metadata={"help": "The column name in the dataset containing the text data."}
    )
    dataset_num_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Number of samples to use from the dataset for debugging/testing."}
    )

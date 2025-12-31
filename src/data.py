"""
Data Processing Module for Unsloth Enterprise Pipeline.
Handles dataset loading, validation, formatting, and previewing.
"""

try:
    from unsloth.chat_templates import get_chat_template
except Exception:
    get_chat_template = None
from datasets import load_dataset, Dataset
import pandas as pd
from typing import Dict, Optional, List, Any
try:
    from .config import ModelConfig, TrainConfig
except ImportError:
    from src.config import ModelConfig, TrainConfig

class DataProcessor:
    """
    Robust Data Processor for ETL operations.
    """
    def __init__(self, model_config: ModelConfig, train_config: TrainConfig, tokenizer):
        self.model_config = model_config
        self.train_config = train_config
        self.tokenizer = tokenizer
        self.raw_dataset = None
        self.formatted_dataset = None

    def load_dataset(self, split: str = "train"):
        """
        Loads the dataset from Hugging Face or local path.
        """
        try:
            self.raw_dataset = load_dataset(
                self.train_config.dataset_name,
                split=split
            )
            if self.train_config.dataset_num_samples:
                print(f"DEBUG: Selecting {self.train_config.dataset_num_samples} samples for debugging.")
                self.raw_dataset = self.raw_dataset.select(range(self.train_config.dataset_num_samples))
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {self.train_config.dataset_name}: {e}")

    def validate_columns(self, required_columns: List[str]):
        """
        Validates that the dataset contains the required columns.
        """
        if not self.raw_dataset:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        missing = [col for col in required_columns if col not in self.raw_dataset.column_names]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}. Available: {self.raw_dataset.column_names}")

    def get_preview(self, n: int = 3) -> pd.DataFrame:
        """
        Returns a Pandas DataFrame of the first n rows for preview.
        """
        if not self.raw_dataset:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        return self.raw_dataset.select(range(min(n, len(self.raw_dataset)))).to_pandas()

    def format_and_tokenize(self, mapping: Optional[Dict[str, str]] = None, style: str = "alpaca"):
        """
        Formats and tokenizes the dataset.
        
        Args:
            mapping: Dictionary mapping internal keys (instruction, input, output) to dataset columns.
                     Example: {'instruction': 'question', 'input': 'context', 'output': 'answer'}
            style: Formatting style ('alpaca', 'chatml', or 'completion').
        """
        if not self.raw_dataset:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        # Default Alpaca mapping
        if mapping is None:
            mapping = {
                "instruction": "instruction",
                "input": "input",
                "output": "output"
            }

        # Validate mapping keys based on style if needed, but for now strict validation on keys existing in dataset
        self.validate_columns(list(mapping.values()))

        def formatting_prompts_func(examples):
            instructions = examples[mapping["instruction"]]
            inputs = examples[mapping["input"]]
            outputs = examples[mapping["output"]]
            texts = []
            
            # Standard Alpaca Format
            alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
            
            alpaca_prompt_no_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

            for instruction, input_text, output in zip(instructions, inputs, outputs):
                if input_text and str(input_text).strip():
                    text = alpaca_prompt.format(instruction, input_text, output)
                else:
                    text = alpaca_prompt_no_input.format(instruction, output)
                texts.append(text + self.tokenizer.eos_token)
            return { "text": texts, }

        # Apply formatting
        if style == "alpaca":
             self.formatted_dataset = self.raw_dataset.map(
                formatting_prompts_func,
                batched=True,
            )
        else:
            # Placeholder for other styles like ChatML if needed
            # For now, we default to the standard unsloth alpaca logic or custom logic
            pass
            
        return self.formatted_dataset

    def apply_template(self):
        """
        Applies a chat template if the model/tokenizer supports it.
        This is a placeholder for more advanced chat template integration.
        """
        pass

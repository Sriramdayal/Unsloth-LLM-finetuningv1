import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
"""
Data Processing Module for Unsloth Enterprise Pipeline.
Handles dataset loading, validation, formatting, and previewing.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import load_dataset, Dataset
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

    def validate_columns(self, required_columns: Optional[List[str]] = None):
        """
        Validates that the dataset contains the required columns.
        If required_columns is None, it attempts to validate based on available auto-detection logic.
        """
        if not self.raw_dataset:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        if required_columns is not None:
            missing = [col for col in required_columns if col not in self.raw_dataset.column_names]
            if missing:
                raise ValueError(f"Dataset missing required columns: {missing}. Available: {self.raw_dataset.column_names}")
        else:
            # Dynamic Validation
            # 1. Check for pre-formatted text column
            text_column = self.train_config.dataset_text_column
            if text_column in self.raw_dataset.column_names:
                return # Valid

            # 2. Check if auto-mapping works
            try:
                self._auto_detect_mapping()
            except ValueError as e:
                raise ValueError(f"Dataset validation failed. Could not detect standard columns or pre-formatted text column.\nDetails: {e}")

    def get_preview(self, n: int = 3) -> pd.DataFrame:
        """
        Returns a Pandas DataFrame of the first n rows for preview.
        """
        if not self.raw_dataset:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        return self.raw_dataset.select(range(min(n, len(self.raw_dataset)))).to_pandas()

    def _auto_detect_mapping(self) -> Dict[str, str]:
        """
        Heuristically detects column names for instruction, input, and output.
        """
        if not self.raw_dataset:
            raise ValueError("Dataset not loaded.")
        
        columns = self.raw_dataset.column_names
        mapping = {}

        # 1. Detect Instruction Column
        instruction_candidates = ["instruction", "prompt", "question", "query", "input_text"]
        for cand in instruction_candidates:
            if cand in columns:
                mapping["instruction"] = cand
                break
        
        # 2. Detect Output Column
        output_candidates = ["output", "response", "answer", "completion", "solution", "target"]
        for cand in output_candidates:
            if cand in columns:
                mapping["output"] = cand
                break

        # 3. Detect Input Column (Optional)
        input_candidates = ["input", "context", "system", "history"]
        for cand in input_candidates:
            if cand in columns:
                mapping["input"] = cand
                break
        
        # Validation & Fallback
        if "instruction" not in mapping or "output" not in mapping:
             # Try positional fallback
             if len(columns) >= 2:
                  print(f"WARNING: Automatic column detection failed for standard keys. Falling back to positional mapping.")
                  mapping["instruction"] = columns[0]
                  mapping["output"] = columns[1]
                  mapping["input"] = None
                  print(f"Positional Fallback: instruction='{mapping['instruction']}', output='{mapping['output']}'")
             else:
                  if "instruction" not in mapping:
                       raise ValueError(f"Could not automatically detect an 'instruction' column. Candidates checked: {instruction_candidates}. Available columns: {columns}")
                  if "output" not in mapping:
                       raise ValueError(f"Could not automatically detect an 'output' column. Candidates checked: {output_candidates}. Available columns: {columns}")
        
        # Default 'input' to None if not found (will need handling in formatting)
        if "input" not in mapping:
            mapping["input"] = None

        return mapping

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

        # 0. Check for Pre-formatted Text Column
        text_column = self.train_config.dataset_text_column
        if text_column in self.raw_dataset.column_names:
            print(f"DEBUG: Found pre-formatted text column '{text_column}'. Skipping tokenization/formatting step.")
            self.formatted_dataset = self.raw_dataset
            # Ensure the text column is actually named "text" for downstream compatibility if it isn't already
            if text_column != "text":
                self.formatted_dataset = self.formatted_dataset.rename_column(text_column, "text")
            return self.formatted_dataset

        # Auto-detect mapping if not provided
        if mapping is None:
            mapping = self._auto_detect_mapping()
            print(f"DEBUG: Auto-detected mapping: {mapping}")

        # Validate mapping keys based on style if needed, but for now strict validation on keys existing in dataset
        # We only strictly require instruction and output. Input is optional.
        required_cols = [mapping["instruction"], mapping["output"]]
        self.validate_columns(required_cols)

        def formatting_prompts_func(examples):
            instructions = examples[mapping["instruction"]]
            outputs = examples[mapping["output"]]
            
            if mapping["input"] and mapping["input"] in examples:
                inputs = examples[mapping["input"]]
            else:
                inputs = [None] * len(instructions)

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

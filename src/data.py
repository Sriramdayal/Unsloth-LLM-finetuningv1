import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
"""
Data Processing Module for Unsloth Enterprise Pipeline.
Handles dataset loading, validation, formatting, and previewing.
"""

import logging
from typing import Any, Dict, List, Optional
import ast

import pandas as pd
from datasets import load_dataset, Dataset
try:
    from .config import ModelConfig, TrainConfig
except ImportError:
    from src.config import ModelConfig, TrainConfig

class DataProcessor:
    """
    Robust Data Processor for ETL operations.
    Handles conversion of various raw dataset formats into tokenized training data.
    """
    def __init__(self, model_config: ModelConfig, train_config: TrainConfig, tokenizer):
        self.model_config = model_config
        self.train_config = train_config
        self.tokenizer = tokenizer
        self.raw_dataset = None
        self.formatted_dataset = None
        self.logger = logging.getLogger(__name__)

    def load_dataset(self, split: str = "train"):
        """Loads dataset from HF or local path with error handling."""
        try:
            self.raw_dataset = load_dataset(self.train_config.dataset_name, split=split)
            if self.train_config.dataset_num_samples:
                self.logger.info(f"Subsampling dataset to {self.train_config.dataset_num_samples} samples.")
                self.raw_dataset = self.raw_dataset.select(range(self.train_config.dataset_num_samples))
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise RuntimeError(f"Data loading error: {e}")

    def _auto_detect_mapping(self) -> Dict[str, str]:
        """Enhanced heuristic for mapping varied dataset schemas."""
        cols = self.raw_dataset.column_names
        mapping = {}

        # 1. Look for 'text' or 'content' (Pre-formatted)
        for cand in ["text", "content", "full_text"]:
            if cand in cols:
                return {"text": cand}

        # 2. Look for Chat formats (conversations, messages)
        for cand in ["conversations", "messages", "chat"]:
            if cand in cols:
                return {"chat": cand}

        # 3. Standard Instruction/Output
        mapping["instruction"] = self._match_column(cols, ["instruction", "prompt", "question", "input_text"])
        mapping["output"] = self._match_column(cols, ["output", "response", "answer", "completion"])
        mapping["input"] = self._match_column(cols, ["input", "context", "background"], optional=True)

        if not mapping["instruction"] or not mapping["output"]:
            self.logger.warning("Auto-detection failed for standard keys. Using positional 0:instruction, 1:output.")
            mapping["instruction"] = cols[0]
            mapping["output"] = cols[1] if len(cols) > 1 else None

        return mapping

    def _match_column(self, cols, candidates, optional=False):
        for c in candidates:
            if c in cols: return c
        return None if optional else ""

    def format_and_tokenize(self, style: str = "auto"):
        """Formats data based on detected style (alpaca, chatml, etc)."""
        mapping = self._auto_detect_mapping()
        
        # 1. Force Alpaca if requested
        if style == "alpaca":
            self.logger.info("Forcing Alpaca-style formatting.")
            # Ensure we have the necessary columns mapped, even if auto-detect missed them
            if "instruction" not in mapping:
                 # Fallback to positional if explicit alpaca requested but no mapping found
                 cols = self.raw_dataset.column_names
                 mapping["instruction"] = cols[0]
                 mapping["output"] = cols[1] if len(cols) > 1 else None
                 mapping["input"] = cols[2] if len(cols) > 2 else None
            
            return self._apply_alpaca_format(mapping)

        # 2. Movie Recommender Style
        if style == "movie_recommender":
            self.logger.info("Applying Movie Recommender style formatting.")
            return self._apply_movie_recommender_format()

        if "text" in mapping:
            self.logger.info(f"Using pre-formatted column: {mapping['text']}")
            self.formatted_dataset = self.raw_dataset.rename_column(mapping["text"], "text")
            return self.formatted_dataset

        if "chat" in mapping or style == "chat":
            self.logger.info(f"Applying Chat template to column: {mapping.get('chat', 'chat')}")
            
            # Ensure tokenizer has a chat template
            if not getattr(self.tokenizer, "chat_template", None):
                self.logger.warning("Tokenizer has no chat_template set. Applying default 'chatml' template.")
                from unsloth.chat_templates import get_chat_template
                self.tokenizer = get_chat_template(
                    self.tokenizer,
                    chat_template = "chatml",
                    mapping = {"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"},
                    map_eos_token = True,
                )

            def chat_format(example):
                convo = example[mapping.get("chat", "chat")]
                # Simple ChatML wrapper if not already tokenized
                return {"text": self.tokenizer.apply_chat_template(convo, tokenize=False) + self.tokenizer.eos_token}
            self.formatted_dataset = self.raw_dataset.map(chat_format, batched=False)
            return self.formatted_dataset

        # Default fallback
        self.logger.info("Applying Alpaca-style formatting (Default).")
        return self._apply_alpaca_format(mapping)

    def _apply_alpaca_format(self, mapping):
        def alpaca_format(examples):
            instr = examples[mapping["instruction"]]
            out = examples[mapping["output"]]
            inp = examples.get(mapping.get("input"), [None] * len(instr)) if mapping.get("input") else [None] * len(instr)
            
            texts = []
            for i, o, c in zip(instr, out, inp):
                if c:
                    txt = f"### Instruction:\n{i}\n\n### Input:\n{c}\n\n### Response:\n{o}"
                else:
                    txt = f"### Instruction:\n{i}\n\n### Response:\n{o}"
                texts.append(txt + self.tokenizer.eos_token)
            return {"text": texts}

        self.formatted_dataset = self.raw_dataset.map(alpaca_format, batched=True)
        return self.formatted_dataset

    def _safe_parse(self, value):
        try:
            return [item["name"] for item in ast.literal_eval(value)]
        except:
            return []

    def _apply_movie_recommender_format(self):
        def format_row(row):
            genres = ", ".join(self._safe_parse(row["genres"]))
            keywords = ", ".join(self._safe_parse(row["keywords"]))

            overview = row["overview"] if row["overview"] else "No overview available."
            tagline = row["tagline"] if row.get("tagline") else ""

            return {
                "text": f"""### Instruction:
Generate a persuasive movie recommendation for a user. Highlight why they should watch the movie using its genre, themes, storyline, and popularity.

### Movie Metadata:
Title: {row['title']}
Tagline: {tagline}
Overview: {overview}
Genres: {genres}
Keywords: {keywords}
Vote Average: {row['vote_average']}
Vote Count: {row['vote_count']}
Popularity: {row['popularity']}
Release Date: {row['release_date']}
Runtime: {row['runtime']} minutes

### Response:
Here’s why you might enjoy this movie:
""" + self.tokenizer.eos_token
            }

        self.formatted_dataset = self.raw_dataset.map(format_row)
        return self.formatted_dataset

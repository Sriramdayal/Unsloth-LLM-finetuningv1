"""
Data Processing Module for Unsloth Enterprise Pipeline.
Handles dataset loading, validation, formatting, and previewing.
"""

from __future__ import annotations

import ast
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset, load_dataset

try:
    from .config import ModelConfig, TrainConfig
except ImportError:
    from src.config import ModelConfig, TrainConfig

logger = logging.getLogger(__name__)

# Column name candidates for auto-detection (ordered by priority)
_INSTRUCTION_CANDIDATES = ("instruction", "prompt", "question", "input_text")
_OUTPUT_CANDIDATES = ("output", "response", "answer", "completion")
_INPUT_CANDIDATES = ("input", "context", "background")
_TEXT_CANDIDATES = ("text", "content", "full_text")
_CHAT_CANDIDATES = ("conversations", "messages", "chat")


class DataProcessor:
    """
    Robust Data Processor for ETL operations.
    Handles conversion of various raw dataset formats into tokenized training data.
    """

    def __init__(self, model_config: ModelConfig, train_config: TrainConfig, tokenizer):
        self.model_config = model_config
        self.train_config = train_config
        self.tokenizer = tokenizer
        self.raw_dataset: Optional[Dataset] = None
        self.formatted_dataset: Optional[Dataset] = None

    def _get_num_proc(self) -> int:
        n = getattr(self.train_config, "dataset_num_proc", -1)
        return os.cpu_count() or 1 if n == -1 else n

    def load_dataset(self, split: str = "train") -> None:
        """Loads dataset from HF or local path with error handling."""
        try:
            self.raw_dataset = load_dataset(
                self.train_config.dataset_name,
                split=split,
                trust_remote_code=True,
                num_proc=self._get_num_proc(),
            )
            n = self.train_config.dataset_num_samples
            if n and n < len(self.raw_dataset):
                logger.info(f"Subsampling dataset to {n} samples.")
                self.raw_dataset = self.raw_dataset.select(range(n))
            logger.info(
                f"Dataset loaded: {len(self.raw_dataset)} samples, "
                f"columns={self.raw_dataset.column_names}"
            )
        except Exception as e:
            logger.error(f"Failed to load dataset '{self.train_config.dataset_name}': {e}")
            raise RuntimeError(f"Data loading error: {e}") from e

    def _auto_detect_mapping(self) -> Dict[str, str]:
        """Enhanced heuristic for mapping varied dataset schemas."""
        cols = self.raw_dataset.column_names
        mapping: Dict[str, Optional[str]] = {}

        # 1. Pre-formatted text column
        for cand in _TEXT_CANDIDATES:
            if cand in cols:
                return {"text": cand}

        # 2. Chat/conversation format
        for cand in _CHAT_CANDIDATES:
            if cand in cols:
                return {"chat": cand}

        # 3. Standard Instruction/Output
        mapping["instruction"] = self._match_column(cols, _INSTRUCTION_CANDIDATES)
        mapping["output"] = self._match_column(cols, _OUTPUT_CANDIDATES)
        mapping["input"] = self._match_column(cols, _INPUT_CANDIDATES, optional=True)

        if not mapping["instruction"] or not mapping["output"]:
            logger.warning(
                "Auto-detection failed for standard keys. "
                f"Available columns: {cols}. Using positional 0:instruction, 1:output."
            )
            if not cols:
                raise ValueError("Dataset has no columns to map.")
            mapping["instruction"] = cols[0]
            mapping["output"] = cols[1] if len(cols) > 1 else None

        return mapping

    @staticmethod
    def _match_column(cols: List[str], candidates: tuple, optional: bool = False) -> Optional[str]:
        """Match a column name from a list of candidates."""
        for c in candidates:
            if c in cols:
                return c
        return None if optional else ""

    def format_and_tokenize(self, style: str = "auto") -> Dataset:
        """Formats data based on detected style (alpaca, chatml, etc)."""
        if self.raw_dataset is None:
            raise RuntimeError("No dataset loaded. Call load_dataset() first.")

        mapping = self._auto_detect_mapping()

        # 1. Force Alpaca if requested
        if style == "alpaca":
            logger.info("Forcing Alpaca-style formatting.")
            if "instruction" not in mapping or not mapping["instruction"]:
                cols = self.raw_dataset.column_names
                if not cols:
                    raise ValueError("Dataset has no columns to map for Alpaca format.")
                mapping["instruction"] = cols[0]
                mapping["output"] = cols[1] if len(cols) > 1 else None
                mapping["input"] = cols[2] if len(cols) > 2 else None
            return self._apply_alpaca_format(mapping)

        # 2. Movie Recommender Style
        if style == "movie_recommender":
            logger.info("Applying Movie Recommender style formatting.")
            return self._apply_movie_recommender_format()

        # 3. Pre-formatted text column
        if "text" in mapping:
            logger.info(f"Using pre-formatted column: '{mapping['text']}'")
            self.formatted_dataset = self.raw_dataset.rename_column(mapping["text"], "text")
            return self.formatted_dataset

        # 4. Agent Tool Calling format
        if style == "agent":
            logger.info("Applying Agent/Tool Calling format.")
            return self._apply_chat_format(mapping)

        # 5. Chat/conversation format
        if "chat" in mapping or style == "chat":
            return self._apply_chat_format(mapping)

        # 6. Default fallback → Alpaca
        logger.info("Applying Alpaca-style formatting (Default).")
        return self._apply_alpaca_format(mapping)

    def _apply_chat_format(self, mapping: Dict[str, str]) -> Dataset:
        """Apply ChatML template to conversation-style datasets."""
        chat_col = mapping.get("chat", "chat")
        logger.info(f"Applying Chat template to column: '{chat_col}'")

        # Ensure tokenizer has a chat template
        if not getattr(self.tokenizer, "chat_template", None):
            logger.warning(
                "Tokenizer has no chat_template set. Applying default 'chatml' template."
            )
            from unsloth.chat_templates import get_chat_template

            self.tokenizer = get_chat_template(
                self.tokenizer,
                chat_template="chatml",
                mapping={
                    "role": "role",
                    "content": "content",
                    "user": "user",
                    "assistant": "assistant",
                },
                map_eos_token=True,
            )

        eos = self.tokenizer.eos_token
        tokenizer = self.tokenizer

        def chat_format(examples):
            convos = examples[chat_col]
            return {
                "text": [tokenizer.apply_chat_template(c, tokenize=False) + eos for c in convos]
            }

        self.formatted_dataset = self.raw_dataset.map(
            chat_format, batched=True, num_proc=self._get_num_proc()
        )
        return self.formatted_dataset

    def _apply_alpaca_format(self, mapping: Dict[str, str]) -> Dataset:
        """Apply Alpaca-style instruction/input/output template."""
        eos = self.tokenizer.eos_token if self.tokenizer else ""
        instr_col = mapping["instruction"]
        out_col = mapping["output"]
        inp_col = mapping.get("input")

        def alpaca_format(examples):
            instructions = examples[instr_col]
            outputs = examples[out_col]
            inputs = (
                examples.get(inp_col, [None] * len(instructions))
                if inp_col
                else [None] * len(instructions)
            )

            texts = []
            for instr, out, inp in zip(instructions, outputs, inputs):
                if inp:
                    txt = (
                        f"### Instruction:\n{instr}\n\n"
                        f"### Input:\n{inp}\n\n"
                        f"### Response:\n{out}"
                    )
                else:
                    txt = f"### Instruction:\n{instr}\n\n### Response:\n{out}"
                texts.append(txt + eos)
            return {"text": texts}

        self.formatted_dataset = self.raw_dataset.map(
            alpaca_format, batched=True, num_proc=self._get_num_proc()
        )
        return self.formatted_dataset

    @staticmethod
    def _safe_parse(value: str) -> List[str]:
        """Safely parse a JSON-like string of dicts with 'name' keys."""
        try:
            return [item["name"] for item in ast.literal_eval(value)]
        except (ValueError, SyntaxError, TypeError, KeyError):
            return []

    def _apply_movie_recommender_format(self) -> Dataset:
        """Apply movie recommender domain-specific prompt template."""
        eos = self.tokenizer.eos_token if self.tokenizer else ""

        def format_row(examples):
            texts = []
            # Calculate length dynamically based on any available key
            keys = list(examples.keys())
            if not keys:
                return {"text": []}
            n_items = len(examples[keys[0]])

            for i in range(n_items):

                def get_val(key, default=""):
                    return examples.get(key, [default] * n_items)[i] or default

                genres_raw = get_val("genres", "[]")
                keywords_raw = get_val("keywords", "[]")
                genres = ", ".join(
                    self._safe_parse(genres_raw if isinstance(genres_raw, str) else str(genres_raw))
                )
                keywords = ", ".join(
                    self._safe_parse(
                        keywords_raw if isinstance(keywords_raw, str) else str(keywords_raw)
                    )
                )

                overview = get_val("overview", "No overview available.")
                tagline = get_val("tagline", "")
                title = get_val("title", "Unknown")
                vote_avg = get_val("vote_average", "N/A")
                vote_count = get_val("vote_count", "N/A")
                popularity = get_val("popularity", "N/A")
                release_date = get_val("release_date", "N/A")
                runtime = get_val("runtime", "N/A")

                texts.append(f"""### Instruction:
Generate a persuasive movie recommendation for a user. Highlight why they should watch the movie using its genre, themes, storyline, and popularity.

### Movie Metadata:
Title: {title}
Tagline: {tagline}
Overview: {overview}
Genres: {genres}
Keywords: {keywords}
Vote Average: {vote_avg}
Vote Count: {vote_count}
Popularity: {popularity}
Release Date: {release_date}
Runtime: {runtime} minutes

### Response:
Here's why you might enjoy this movie:
""" + eos)
            return {"text": texts}

        self.formatted_dataset = self.raw_dataset.map(
            format_row, batched=True, num_proc=self._get_num_proc()
        )
        return self.formatted_dataset

from unittest.mock import MagicMock

import pytest

from src.config import ModelConfig, TrainConfig
from src.data import DataProcessor


def test_dataprocessor_initialization(mock_model_config, mock_train_config):
    """DataProcessor loads with config and tokenizer"""
    mock_tokenizer = MagicMock()
    processor = DataProcessor(mock_model_config, mock_train_config, mock_tokenizer)
    assert processor.model_config == mock_model_config
    assert processor.train_config == mock_train_config
    assert processor.tokenizer == mock_tokenizer


def test_dataprocessor_match_column():
    """Test column matching utility"""
    cols = ["instruction", "input", "output"]
    matched = DataProcessor._match_column(cols, ("instruction", "prompt"))
    assert matched == "instruction"

    matched_opt = DataProcessor._match_column(cols, ("nonexistent",), optional=True)
    assert matched_opt is None


def test_dataprocessor_safe_parse():
    """Test safe parsing of JSON-like strings"""
    val = "[{'name': 'Action'}, {'name': 'Adventure'}]"
    parsed = DataProcessor._safe_parse(val)
    assert parsed == ["Action", "Adventure"]

    invalid_val = "not a valid string"
    assert DataProcessor._safe_parse(invalid_val) == []

from unittest.mock import MagicMock

import pytest

from src.config import ModelConfig
from src.core.factory import ModelFactory, _best_device, _platform
from src.core.model_runner import ModelRunner


def test_modelfactory_detects_platform():
    """Test platform detection helper functions"""
    plat = _platform()
    assert plat in ["linux", "win32", "darwin"]

    device = _best_device()
    assert device in ["cuda", "mps", "cpu"]


def test_modelrunner_initialization(mock_model_config):
    """ModelRunner initializes with config"""
    runner = ModelRunner(mock_model_config)
    assert runner.config == mock_model_config


def test_modelrunner_setup_inference_mock(mock_model_config, monkeypatch):
    """ModelRunner.setup_for_inference handles missing model gracefully"""
    runner = ModelRunner(mock_model_config)
    monkeypatch.setattr(runner, "model", None)
    monkeypatch.setattr(runner, "tokenizer", None)
    try:
        runner.setup_for_inference()
    except Exception as e:
        # Should raise error indicating model file doesn't exist or cuda
        assert any(
            x in str(e).lower()
            for x in ["model", "cuda", "not found", "llama-cpp-python", "no module"]
        )

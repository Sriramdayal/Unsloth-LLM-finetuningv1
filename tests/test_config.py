import pytest

from src.config import ModelConfig, TrainConfig


def test_model_config_defaults():
    """ModelConfig uses sensible defaults"""
    config = ModelConfig(model_name_or_path="test/model")
    assert config.lora_r == 16
    assert config.lora_alpha == 16
    assert config.lora_dropout == 0.0
    assert config.load_in_4bit is True


def test_model_config_custom_values():
    """ModelConfig accepts custom LoRA params"""
    config = ModelConfig(
        model_name_or_path="test/model", lora_r=64, lora_alpha=128, lora_dropout=0.1
    )
    assert config.lora_r == 64
    assert config.lora_alpha == 128
    assert config.lora_dropout == 0.1


def test_train_config_output_dir():
    """TrainConfig sets output directory"""
    config = TrainConfig(dataset_name="test/dataset", output_dir="outputs/my_experiment")
    assert config.output_dir == "outputs/my_experiment"


def test_train_config_gradient_accumulation():
    """Gradient accumulation defaults to 4"""
    config = TrainConfig(dataset_name="d")
    assert config.gradient_accumulation_steps == 4

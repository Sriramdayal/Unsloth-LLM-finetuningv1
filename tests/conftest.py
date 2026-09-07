"""
Test configuration — isolates API tests from heavy ML dependencies.

Mocks out the heavy ``src`` package-level imports (torch, transformers,
datasets, pandas, pyarrow) so that the test session can collect and run
without loading GPU drivers or triggering native DLL issues on Windows.
"""

import sys
from unittest.mock import MagicMock

# ── Pre-emptive mock of heavy dependencies ────────────────────────────────────
# src/__init__.py eagerly imports DataProcessor, train_model, ModelRunner, etc.
# which cascade into torch → CUDA, pandas → pyarrow (DLL crash on Windows).
# We intercept this by injecting mock modules BEFORE src is imported.

_HEAVY_MODULES = [
    "torch",
    "torch.cuda",
    "torch.backends",
    "torch.backends.mps",
    "transformers",
    "transformers.BitsAndBytesConfig",
    "transformers.AutoModelForCausalLM",
    "transformers.AutoTokenizer",
    "transformers.TrainingArguments",
    "datasets",
    "datasets.Dataset",
    "datasets.load_dataset",
    "peft",
    "peft.LoraConfig",
    "peft.get_peft_model",
    "peft.prepare_model_for_kbit_training",
    "peft.PeftModel",
    "trl",
    "trl.SFTConfig",
    "trl.SFTTrainer",
    "bitsandbytes",
    "accelerate",
    "pandas",
    "pyarrow",
    "pyarrow.dataset",
    "tqdm",
    "tqdm.auto",
    "mlx",
    "mlx.core",
    "mlx_lm",
]

for mod_name in _HEAVY_MODULES:
    if mod_name not in sys.modules:
        mock = MagicMock()
        # torch.cuda.is_available() should return False in tests
        if mod_name == "torch":
            mock.cuda.is_available.return_value = False
            mock.cuda.is_bf16_supported.return_value = False
            mock.backends.mps.is_available.return_value = False
            mock.no_grad.return_value = lambda fn: fn
            mock.float32 = "float32"
            mock.float16 = "float16"
            mock.bfloat16 = "bfloat16"
        sys.modules[mod_name] = mock

# NOTE: src.soup needs no mocks — it is stdlib-only (subprocess/shutil/dataclasses)
# and talks to the `soup` CLI via subprocess, never via import.


# ── Shared fixtures for tests ───────────────────────────────────────────────

import pytest
from fastapi.testclient import TestClient

from src.service.app import create_app
from src.config import ModelConfig, TrainConfig


@pytest.fixture
def client():
    """FastAPI test client"""
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_model_config():
    return ModelConfig(
        model_name_or_path="unsloth/llama-3-8b-bnb-4bit",
        load_in_4bit=True,
        lora_r=16,
        lora_alpha=32,
    )


@pytest.fixture
def mock_train_config():
    return TrainConfig(
        dataset_name="yahma/alpaca-cleaned",
        output_dir="outputs/test_run",
        batch_size=2,
        learning_rate=2e-4,
        num_train_epochs=1,
    )


@pytest.fixture
def sample_dataset_entry():
    """Mock dataset entry for data tests"""
    return {
        "instruction": "What is sepsis?",
        "input": "",
        "output": "Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection.",
    }

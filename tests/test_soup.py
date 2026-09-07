"""
Soup Integration Tests — subprocess client and config mapping.

src.soup is stdlib-only (subprocess/shutil/dataclasses), so these tests need
no heavy ML mocks. subprocess and shutil.which are mocked to avoid requiring
the real `soup` binary.
"""

from subprocess import CalledProcessError, CompletedProcess
from unittest.mock import MagicMock, patch

import pytest

from src import ModelConfig, TrainConfig
from src.soup import SoupClient, SoupConfig, SoupNotAvailableError


# ── SoupClient.is_available ───────────────────────────────────────────────────


class TestIsAvailable:
    def test_available_when_binary_on_path(self):
        with patch("src.soup.client.shutil.which", return_value="/usr/bin/soup"):
            assert SoupClient().is_available() is True

    def test_unavailable_when_binary_missing(self):
        with patch("src.soup.client.shutil.which", return_value=None):
            assert SoupClient().is_available() is False


# ── SoupClient.export_gguf ────────────────────────────────────────────────────


def _completed(args):
    return CompletedProcess(args=args, returncode=0, stdout="", stderr="")


class TestExportGguf:
    def test_builds_correct_command(self):
        client = SoupClient()
        with (
            patch.object(client, "is_available", return_value=True),
            patch("src.soup.client.subprocess.run", return_value=_completed([])) as mock_run,
        ):
            out = client.export_gguf("./outputs/quickstart", quant="q4_k_m",
                                     output_path="model.gguf")
        assert out == "model.gguf"
        cmd = mock_run.call_args[0][0]
        assert cmd == [
            "soup", "export",
            "--model", "./outputs/quickstart",
            "--format", "gguf",
            "--quant", "q4_k_m",
            "--output", "model.gguf",
        ]

    def test_omits_output_flag_when_not_given(self):
        client = SoupClient()
        with (
            patch.object(client, "is_available", return_value=True),
            patch("src.soup.client.subprocess.run", return_value=_completed([])) as mock_run,
        ):
            out = client.export_gguf("./outputs/quickstart")
        assert out == "./outputs/quickstart"
        assert "--output" not in mock_run.call_args[0][0]

    def test_raises_when_binary_missing(self):
        client = SoupClient()
        with patch.object(client, "is_available", return_value=False):
            with pytest.raises(SoupNotAvailableError):
                client.export_gguf("./outputs/quickstart")

    def test_raises_on_subprocess_failure(self):
        client = SoupClient()
        with (
            patch.object(client, "is_available", return_value=True),
            patch(
                "src.soup.client.subprocess.run",
                side_effect=CalledProcessError(1, ["soup"], stderr="boom"),
            ),
        ):
            with pytest.raises(RuntimeError):
                client.export_gguf("./outputs/quickstart")


# ── SoupConfig ────────────────────────────────────────────────────────────────


class TestSoupConfig:
    def test_to_dict_structure(self):
        cfg = SoupConfig(base_model="org/model", data_train="org/dataset")
        d = cfg.to_dict()
        assert d["base"] == "org/model"
        assert d["task"] == "sft"
        assert d["backend"] == "unsloth"
        assert d["data"] == {"train": "org/dataset", "format": "alpaca", "max_length": 2048}
        assert d["training"]["quantization"] == "4bit"
        assert d["training"]["lora"] == {"r": 16, "alpha": 16}
        assert d["output"] == "./output"

    def test_from_model_train_configs_mapping(self):
        model_cfg = ModelConfig(
            model_name_or_path="AdithyaSK/Qwen-0.5b-Code-Reasoning-v1",
            load_in_4bit=True,
            max_seq_length=2048,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
        )
        train_cfg = TrainConfig(
            dataset_name="greengerong/leetcode",
            dataset_style="alpaca",
            output_dir="outputs/qwen_leetcode",
            learning_rate=0.0002,
            batch_size=2,
            num_train_epochs=3.0,
        )
        cfg = SoupConfig.from_model_train_configs(model_cfg, train_cfg)
        d = cfg.to_dict()
        assert d["base"] == "AdithyaSK/Qwen-0.5b-Code-Reasoning-v1"
        assert d["data"]["train"] == "greengerong/leetcode"
        assert d["data"]["format"] == "alpaca"
        assert d["training"]["epochs"] == 3.0
        assert d["training"]["lr"] == 0.0002
        assert d["training"]["lora"] == {"r": 16, "alpha": 32, "dropout": 0.05}

    def test_custom_target_modules_passed_through(self):
        model_cfg = ModelConfig(
            model_name_or_path="m", target_modules=["q_proj", "v_proj"]
        )
        train_cfg = TrainConfig(dataset_name="d")
        cfg = SoupConfig.from_model_train_configs(model_cfg, train_cfg)
        assert cfg.to_dict()["training"]["lora"]["target_modules"] == ["q_proj", "v_proj"]

    def test_to_yaml_round_trip(self, tmp_path):
        yaml = pytest.importorskip("yaml")
        cfg = SoupConfig(base_model="org/model", data_train="org/dataset")
        path = str(tmp_path / "soup.yaml")
        assert cfg.to_yaml(path) == path
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        assert loaded == cfg.to_dict()

    def test_mock_import_surface(self):
        # src/__init__ re-exports the soup names (stdlib-only, always importable)
        import src

        assert src.SoupClient is SoupClient
        assert src.SoupConfig is SoupConfig

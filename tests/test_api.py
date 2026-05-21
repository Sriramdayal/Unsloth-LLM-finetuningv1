"""
API Integration Tests — Unsloth LLM Fine-Tuning API.

Tests use FastAPI's TestClient (synchronous) to validate endpoint contracts
without loading real GPU models. Heavy dependencies (torch, transformers,
datasets, pyarrow) are mocked in conftest.py before any imports occur.
"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.main import create_app

client = TestClient(create_app())


# ── Health ────────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    def test_health_returns_200(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_reports_status_healthy(self):
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_contains_required_fields(self):
        response = client.get("/api/v1/health")
        data = response.json()
        assert "gpu_available" in data
        assert "platform" in data
        assert "backend" in data


# ── Inference ─────────────────────────────────────────────────────────────────


class TestInferenceEndpoint:
    """Tests for POST /api/v1/infer."""

    def test_infer_rejects_missing_model_path(self):
        """model_path is required — omitting it should return 422."""
        response = client.post("/api/v1/infer", json={"prompt": "hi"})
        assert response.status_code == 422

    def test_infer_rejects_empty_prompt(self):
        """prompt must be at least 1 char."""
        response = client.post(
            "/api/v1/infer",
            json={"model_path": "some/model", "prompt": ""},
        )
        assert response.status_code == 422

    def test_infer_rejects_oversized_tokens(self):
        """max_tokens must be <= 2048."""
        response = client.post(
            "/api/v1/infer",
            json={
                "model_path": "some/model",
                "prompt": "hello",
                "max_tokens": 9999,
            },
        )
        assert response.status_code == 422

    @patch("src.api.routes.inference.model_cache")
    def test_infer_success_with_mocked_runner(self, mock_cache):
        """Verify successful inference flow with a mocked model cache."""
        mock_runner = MagicMock()
        mock_runner.generate.return_value = "Mocked response text"

        # Simulate cache hit
        mock_cache.__contains__ = MagicMock(return_value=True)
        mock_cache.get.return_value = mock_runner

        response = client.post(
            "/api/v1/infer",
            json={"model_path": "test/model", "prompt": "What is AI?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Mocked response text"
        assert data["model"] == "test/model"
        assert "tokens_generated" in data


# ── GGUF Inference ────────────────────────────────────────────────────────────


class TestGGUFInferenceEndpoint:
    """Tests for POST /api/v1/infer/gguf."""

    def test_gguf_rejects_non_gguf_path(self):
        """model_path must end in .gguf."""
        response = client.post(
            "/api/v1/infer/gguf",
            json={"model_path": "some/model", "prompt": "hello"},
        )
        assert response.status_code == 400


# ── Training ──────────────────────────────────────────────────────────────────


class TestTrainingEndpoint:
    """Tests for POST /api/v1/train and GET /api/v1/train/{job_id}/status."""

    @patch("src.api.routes.training._run_training_job")
    def test_train_enqueue_returns_job_id(self, mock_train):
        """POST /train should return 200 with a job_id."""
        response = client.post(
            "/api/v1/train",
            json={
                "model_name_or_path": "unsloth/llama-3-8b-bnb-4bit",
                "dataset_name": "yahma/alpaca-cleaned",
                "num_train_epochs": 1,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"

    def test_train_status_not_found(self):
        """Polling a non-existent job_id should return 404."""
        response = client.get("/api/v1/train/nonexistent/status")
        assert response.status_code == 404

    @patch("src.api.routes.training._run_training_job")
    def test_train_status_after_enqueue(self, mock_train):
        """After enqueuing, the status should be 'queued'."""
        enqueue_resp = client.post(
            "/api/v1/train",
            json={
                "model_name_or_path": "unsloth/llama-3-8b-bnb-4bit",
                "dataset_name": "yahma/alpaca-cleaned",
                "num_train_epochs": 1,
            },
        )
        job_id = enqueue_resp.json()["job_id"]

        status_resp = client.get(f"/api/v1/train/{job_id}/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "queued"


# ── Schema Validation ─────────────────────────────────────────────────────────


class TestTrainingRequestDefaults:
    """Verify TrainingRequest default values are applied correctly."""

    @patch("src.api.routes.training._run_training_job")
    def test_defaults_applied(self, mock_train):
        """Posting with no overrides should accept all defaults."""
        response = client.post("/api/v1/train", json={})
        assert response.status_code == 200

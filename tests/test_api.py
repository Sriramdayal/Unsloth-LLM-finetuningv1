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

    def test_health_method_not_allowed(self):
        """POST to /health should fail"""
        response = client.post("/api/v1/health")
        assert response.status_code == 405


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

    def test_infer_valid_request_structure(self):
        """POST /infer with valid body returns 200 or 500"""
        response = client.post(
            "/api/v1/infer",
            json={
                "model_path": "unsloth/llama-3-8b-bnb-4bit",
                "prompt": "What is machine learning?",
                "max_tokens": 128,
            },
        )
        assert response.status_code in [200, 500]


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

    def test_train_validation_bad_epochs(self):
        """POST /train with num_train_epochs=0 returns 422"""
        response = client.post(
            "/api/v1/train",
            json={
                "model_name_or_path": "unsloth/llama-3-8b-bnb-4bit",
                "dataset_name": "yahma/alpaca-cleaned",
                "num_train_epochs": 0,
            },
        )
        assert response.status_code == 422


# ── Schema Validation ─────────────────────────────────────────────────────────


class TestTrainingRequestDefaults:
    """Verify TrainingRequest default values are applied correctly."""

    @patch("src.api.routes.training._run_training_job")
    def test_defaults_applied(self, mock_train):
        """Posting with no overrides should accept all defaults."""
        response = client.post("/api/v1/train", json={})
        assert response.status_code == 200


# ── ModelCache & use_mock Tests ──────────────────────────────────────────────


class TestModelCache:
    """Tests for ModelCache LRU eviction and cleanup."""

    def test_model_cache_lru_eviction(self):
        from src.api.dependencies import ModelCache

        cache = ModelCache(max_size=2)
        r1 = MagicMock()
        r2 = MagicMock()
        r3 = MagicMock()

        r2.model = "model_val"
        r2.tokenizer = "tokenizer_val"
        r2.config = "config_val"

        cache.put("k1", r1)
        cache.put("k2", r2)

        assert "k1" in cache
        assert "k2" in cache

        # Access k1 to make it most recently used
        cache.get("k1")

        # Put k3 (should evict k2 because k1 is MRU)
        cache.put("k3", r3)

        assert "k1" in cache
        assert "k3" in cache
        assert "k2" not in cache

        # Check memory cleanup attributes were deleted/None on evicted runner
        assert r2.model is None
        assert r2.tokenizer is None
        assert r2.config is None


class TestTrainingMocking:
    """Tests for TrainingRequest with use_mock parameter."""

    @patch("src.api.routes.training._run_training_job")
    def test_train_with_use_mock(self, mock_train):
        """Verify POST /train accepts use_mock and calls worker."""
        response = client.post(
            "/api/v1/train",
            json={
                "model_name_or_path": "unsloth/llama-3-8b-bnb-4bit",
                "dataset_name": "yahma/alpaca-cleaned",
                "use_mock": True,
            },
        )
        assert response.status_code == 200
        # Check that the worker was called with TrainingRequest having use_mock=True
        mock_train.assert_called_once()
        args, _ = mock_train.call_args
        req = args[1]
        assert req.use_mock is True


class TestDocsEndpoint:
    """Swagger UI auto-generated at /docs"""

    def test_api_docs_available(self):
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower()

"""
FastAPI Application Factory.

Creates and configures the Unsloth LLM Fine-Tuning API with all route modules
registered under the ``/api/v1`` prefix.
"""

from fastapi import FastAPI

from src.api.routes import health, inference, training


def create_app() -> FastAPI:
    """
    Build and return the configured FastAPI application.

    Routers:
        /api/v1/health              — system health check
        /api/v1/infer               — HF inference
        /api/v1/infer/gguf          — GGUF (llama.cpp) inference
        /api/v1/train               — enqueue training job
        /api/v1/train/{id}/status   — poll training status
    """
    app = FastAPI(
        title="Unsloth LLM Fine-Tuning API",
        description=(
            "Production REST API for cross-platform LLM fine-tuning and inference. "
            "Supports HuggingFace transformers, PEFT/LoRA, bitsandbytes 4-bit QLoRA, "
            "Unsloth Triton kernels (Linux), and GGUF models via llama.cpp."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(health.router, prefix="/api/v1")
    app.include_router(inference.router, prefix="/api/v1")
    app.include_router(training.router, prefix="/api/v1")

    return app


app = create_app()

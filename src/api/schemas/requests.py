"""
Pydantic models for API request/response validation.

Defines strict schemas for inference, training, and job status endpoints.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """Schema for POST /api/v1/infer — single-prompt inference."""

    model_path: str = Field(..., description="HF model ID or local path")
    prompt: str = Field(..., min_length=1, max_length=4096)
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    adapter_path: Optional[str] = None


class TrainingRequest(BaseModel):
    """Schema for POST /api/v1/train — enqueue a fine-tuning job."""

    model_name_or_path: str = Field(default="unsloth/llama-3-8b-bnb-4bit")
    dataset_name: str = Field(default="yahma/alpaca-cleaned")
    lora_r: int = Field(default=16, ge=1, le=128)
    lora_alpha: int = Field(default=32)
    learning_rate: float = Field(default=2e-4)
    num_train_epochs: int = Field(default=3, ge=1, le=10)
    batch_size: int = Field(default=2, ge=1, le=16)
    output_dir: str = Field(default="outputs/api_run")
    push_to_hub: bool = Field(default=False)
    use_mock: bool = Field(default=False)


class JobStatusResponse(BaseModel):
    """Schema for GET /api/v1/train/{job_id}/status — poll training progress."""

    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: Optional[float] = None
    output_dir: Optional[str] = None
    error_message: Optional[str] = None

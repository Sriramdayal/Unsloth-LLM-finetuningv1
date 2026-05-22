"""
Training endpoints — background job management.

POST /train       → enqueue a new fine-tuning job (runs in a background thread)
GET  /train/{id}  → poll job status / progress

Heavy ML imports (torch, transformers, datasets) are deferred to function bodies
so that importing this module does not trigger a full dependency load — this keeps
test collection fast and avoids pyarrow DLL issues on Windows.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.api.dependencies import job_registry
from src.api.schemas.requests import JobStatusResponse, TrainingRequest

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


def _run_training_job(job_id: str, req: TrainingRequest):
    """
    Execute the full training pipeline in a background thread.

    Pipeline:  ModelRunner.setup_for_training() → DataProcessor → train_model()

    This adapts the blueprint's simple ``train_model(config)`` call to match
    the existing function signature which requires separate model, tokenizer,
    dataset, and config objects.
    """
    job_registry.update(job_id, status="running")

    try:
        # Deferred imports — avoid loading heavy ML deps at module level
        from src.config import ModelConfig, TrainConfig
        from src.core.model_runner import ModelRunner
        from src.data import DataProcessor
        from src.train import train_model

        # 1. Build configs from the API request
        model_config = ModelConfig(
            model_name_or_path=req.model_name_or_path,
            load_in_4bit=True,
            lora_r=req.lora_r,
            lora_alpha=req.lora_alpha,
            use_mock=req.use_mock,
        )
        train_config = TrainConfig(
            dataset_name=req.dataset_name,
            output_dir=req.output_dir,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            num_train_epochs=req.num_train_epochs,
            push_to_hub=req.push_to_hub,
        )

        # 2. Load model + apply LoRA
        runner = ModelRunner(model_config)
        model, tokenizer = runner.setup_for_training()

        # 3. Prepare dataset
        processor = DataProcessor(model_config, train_config, tokenizer)
        processor.load_dataset()
        dataset = processor.format_and_tokenize()

        # 4. Train
        _stats, output_dir = train_model(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            train_config=train_config,
            model_config=model_config,
        )

        job_registry.update(
            job_id,
            status="completed",
            progress=100.0,
            output_dir=output_dir,
        )
        logger.info(f"Training job {job_id} completed → {output_dir}")

    except Exception as e:
        logger.exception(f"Training job {job_id} failed")
        job_registry.update(
            job_id,
            status="failed",
            error_message=str(e),
        )


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.post("/train", tags=["training"])
def start_training(req: TrainingRequest, background: BackgroundTasks):
    """
    Enqueue a new fine-tuning job.

    Returns immediately with a ``job_id`` that can be polled via
    ``GET /api/v1/train/{job_id}/status``.
    """
    job_id = str(uuid.uuid4())[:8]
    job_registry.create(job_id)
    background.add_task(_run_training_job, job_id, req)
    logger.info(f"Training job {job_id} enqueued: {req.model_name_or_path}")
    return {"job_id": job_id, "status": "queued"}


@router.get("/train/{job_id}/status", tags=["training"])
def get_status(job_id: str):
    """Poll the status of an enqueued training job."""
    job = job_registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress"),
        output_dir=job.get("output_dir"),
        error_message=job.get("error_message"),
    )

"""
Inference endpoints — HF transformers and GGUF (llama.cpp) backends.

Model/adapter combinations are cached in-memory so repeated requests
to the same model skip the loading step.

Heavy ML imports (torch, transformers) are deferred to function bodies
so that importing this module does not trigger a full dependency load.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.service.dependencies import model_cache
from src.service.schemas.requests import InferenceRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/infer", tags=["inference"])
def infer(req: InferenceRequest):
    """
    Run single-prompt inference using HuggingFace / PEFT backend.

    The model (and optional LoRA adapter) are loaded once and cached for
    subsequent requests with the same ``model_path:adapter_path`` key.
    """
    from src.config import ModelConfig
    from src.platform.model_runner import ModelRunner

    try:
        cache_key = f"{req.model_path}:{req.adapter_path}"

        if cache_key not in model_cache:
            config = ModelConfig(
                model_name_or_path=req.model_path,
                load_in_4bit=True,
            )
            runner = ModelRunner(config)
            runner.setup_for_inference(adapter_path=req.adapter_path)
            model_cache.put(cache_key, runner)

        runner = model_cache.get(cache_key)
        result = runner.generate(
            req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
        )

        return {
            "response": result,
            "model": req.model_path,
            "tokens_generated": len(result.split()),  # approximate word count
        }

    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/infer/gguf", tags=["inference"])
def infer_gguf(req: InferenceRequest):
    """
    Run inference using a GGUF model via the llama.cpp backend.

    The ``model_path`` must point to a ``.gguf`` file (local or downloaded).
    """
    from src.config import ModelConfig
    from src.platform.model_runner import ModelRunner

    if not req.model_path.endswith(".gguf"):
        raise HTTPException(
            status_code=400,
            detail="model_path must point to a .gguf file for the GGUF endpoint",
        )

    try:
        cache_key = f"gguf:{req.model_path}"

        if cache_key not in model_cache:
            config = ModelConfig(
                model_name_or_path=req.model_path,
                load_in_4bit=False,
            )
            runner = ModelRunner(config)
            runner.setup_for_inference()
            model_cache.put(cache_key, runner)

        runner = model_cache.get(cache_key)
        result = runner.generate(
            req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
        )

        return {
            "response": result,
            "model": req.model_path,
            "tokens_generated": len(result.split()),
        }

    except Exception as e:
        logger.exception("GGUF inference failed")
        raise HTTPException(status_code=500, detail=str(e))

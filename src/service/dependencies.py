"""
Shared dependencies for the Unsloth API.

Provides a thread-safe model cache and a job registry shared across route modules.
"""

from __future__ import annotations

import gc
import threading
from collections import OrderedDict
from typing import Any, Dict

import torch


class ModelCache:
    """
    Thread-safe in-memory cache for loaded ModelRunner instances with LRU eviction.

    Keyed by ``model_path:adapter_path`` so the same model/adapter
    combination is loaded only once across requests.
    """

    def __init__(self, max_size: int = 1):
        self._max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str):
        """Return a cached runner or None (thread-safe, updates LRU order)."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: str, runner):
        """Store a runner in the cache (thread-safe, evicts LRU items if full)."""
        with self._lock:
            if key in self._cache:
                self._cache[key] = runner
                self._cache.move_to_end(key)
                return

            # If cache is full, evict LRU items until space is available
            while len(self._cache) >= self._max_size and self._cache:
                lru_key, lru_runner = self._cache.popitem(last=False)
                self._evict_runner(lru_runner)

            self._cache[key] = runner

    def _evict_runner(self, runner):
        """Delete model and tokenizer attributes, and clean up memory."""
        if runner is not None:
            if hasattr(runner, "model"):
                runner.model = None
            if hasattr(runner, "tokenizer"):
                runner.tokenizer = None
            if hasattr(runner, "config"):
                runner.config = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._cache


class JobRegistry:
    """
    Thread-safe registry for background training jobs.

    Each job is stored as a dict with keys: status, progress, output_dir, error_message.
    """

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str) -> Dict[str, Any]:
        """Register a new job with 'queued' status."""
        job = {
            "status": "queued",
            "progress": 0.0,
            "output_dir": None,
            "error_message": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Dict[str, Any] | None:
        """Return job dict or None if not found."""
        return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs):
        """Update fields on an existing job (thread-safe)."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(kwargs)


# ── Singleton instances shared across routes ──────────────────────────────────

model_cache = ModelCache()
job_registry = JobRegistry()

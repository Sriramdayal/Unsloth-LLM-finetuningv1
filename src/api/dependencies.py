"""
Shared dependencies for the Unsloth API.

Provides a thread-safe model cache and a job registry shared across route modules.
"""

from __future__ import annotations

import threading
from typing import Any, Dict


class ModelCache:
    """
    Thread-safe in-memory cache for loaded ModelRunner instances.

    Keyed by ``model_path:adapter_path`` so the same model/adapter
    combination is loaded only once across requests.
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        """Return a cached runner or None."""
        return self._cache.get(key)

    def put(self, key: str, runner):
        """Store a runner in the cache (thread-safe)."""
        with self._lock:
            self._cache[key] = runner

    def __contains__(self, key: str) -> bool:
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

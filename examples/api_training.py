"""Enqueue a training job via the REST API and poll its status (stdlib only).

Usage:
    # Terminal 1: start the API
    python -m src.api.main        # or: uvicorn src.service.app:app
    # Terminal 2:
    python examples/api_training.py
"""

import json
import time
import urllib.request

BASE_URL = "http://localhost:8000/api/v1"


def _post(path, payload):
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def _get(path):
    with urllib.request.urlopen(f"{BASE_URL}{path}") as resp:
        return json.load(resp)


def main():
    # 1. Health check
    print(_get("/health"))

    # 2. Enqueue a mock training job (no GPU needed)
    job = _post(
        "/train",
        {
            "model_name_or_path": "unsloth/llama-3-8b-bnb-4bit",
            "dataset_name": "yahma/alpaca-cleaned",
            "num_train_epochs": 1,
            "use_mock": True,
        },
    )
    job_id = job["job_id"]
    print(f"Enqueued job: {job_id}")

    # 3. Poll until terminal state
    while True:
        status = _get(f"/train/{job_id}/status")
        print(f"status={status['status']} progress={status.get('progress')}")
        if status["status"] in ("completed", "failed"):
            print(status)
            break
        time.sleep(2)


if __name__ == "__main__":
    main()

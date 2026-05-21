"""
Unsloth API — Uvicorn entry point.

Usage:
    python -m src.api                     # via __main__
    uvicorn src.api:app --reload          # direct uvicorn
    gunicorn -k uvicorn.workers.UvicornWorker src.api:app   # production
"""

from src.api.main import app  # noqa: F401  — re-export for uvicorn

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

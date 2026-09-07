"""REST API service layer: FastAPI app factory, job registry, model cache."""

from .app import app, create_app

__all__ = ["app", "create_app"]

"""Soup hybrid integration: subprocess client + config mapping."""

from .client import SoupClient, SoupNotAvailableError
from .config import SoupConfig

__all__ = ["SoupClient", "SoupConfig", "SoupNotAvailableError"]

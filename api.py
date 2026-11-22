"""Compatibility entry point: uvicorn api:app (preferred: src.main:app)."""
from src.main import app, create_app

__all__ = ["app", "create_app"]

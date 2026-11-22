"""Compatibility facade for notebooks and library clients; implementation is in src/."""
from src.core.policy import Settings
from src.core.analytics.models import EngineOutput
from src.config.engine import run_engine

__all__ = ['Settings', 'run_engine', 'EngineOutput']

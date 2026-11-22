"""Trusted candidate-model persistence; never invoked by upload analysis."""
from pathlib import Path
from typing import Any


class JoblibModelArtifacts:
    def save(self, model: Any, path: Path) -> None:
        import joblib
        joblib.dump(model, path)

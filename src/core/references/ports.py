"""Public-reference retrieval contract owned by the references module."""
from typing import Protocol
from .models import ReferenceData


class ReferenceProvider(Protocol):
    def load(self) -> ReferenceData: ...

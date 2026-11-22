"""Fleet input and exported report contracts."""
from typing import Any, Protocol
import pandas as pd


class FleetDecoder(Protocol):
    def decode(self, content: bytes) -> tuple[pd.DataFrame, str]: ...


class ReportRepository(Protocol):
    def read_json(self, name: str) -> dict[str, Any]: ...
    def read_frame(self, name: str) -> pd.DataFrame: ...

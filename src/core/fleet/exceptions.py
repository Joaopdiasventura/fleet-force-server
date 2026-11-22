"""Fleet ingestion validation errors, independent of HTTP."""
from typing import Any
class FleetDataError(ValueError):
    """Fleet schema or source content cannot be ingested."""


class CSVValidationError(FleetDataError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code, self.message, self.details = code, message, details or {}




class ReportUnavailable(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class VehicleNotFound(Exception):
    pass

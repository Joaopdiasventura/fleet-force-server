"""HTTP error envelope used by upload routes and exception handlers."""
from typing import Any
from fastapi.responses import JSONResponse


def upload_error(status: int, code: str, message: str,
                 details: dict[str, Any] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={
        'error': {'code': code, 'message': message, 'details': details or {}}})

"""Central exception-to-HTTP mapping preserving existing route contracts."""
from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from src.core.fleet.exceptions import ReportUnavailable, VehicleNotFound
from src.shared.http import upload_error


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        if request.url.path == '/v1/fleet/analyze':
            return upload_error(422, 'INVALID_REQUEST', 'A CSV file and valid form fields are required.')
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        if request.url.path == '/v1/fleet/analyze':
            return upload_error(exc.status_code, 'INVALID_REQUEST', 'Invalid upload request.')
        return await http_exception_handler(request, exc)

    @app.exception_handler(ReportUnavailable)
    async def unavailable(request: Request, exc: ReportUnavailable) -> JSONResponse:
        return JSONResponse(status_code=503, content={'detail': exc.message})

    @app.exception_handler(VehicleNotFound)
    async def not_found(request: Request, exc: VehicleNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={'detail': 'Veículo não encontrado'})

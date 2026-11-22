"""Multipart HTTP transport only; analysis runs in the worker thread pool."""
import logging
from typing import Annotated, Any
from fastapi import APIRouter, Depends, File, Form, UploadFile
from starlette.concurrency import run_in_threadpool

from src.config.dependencies import get_settings, get_upload_service
from src.config.settings import AppSettings
from src.core.fleet.exceptions import CSVValidationError
from src.shared.http import upload_error
from .uploads import UploadAnalysisService

router = APIRouter(prefix='/v1/fleet')
logger = logging.getLogger(__name__)
UF_CODES = set('AC AL AP AM BA CE DF ES GO MA MT MS MG PA PB PR PE PI RJ RN RS RO RR SC SP SE TO'.split())


@router.post('/analyze', response_model=None, responses={
    400: {'description': 'Invalid state or request parameters'},
    413: {'description': 'Upload exceeds configured size limit'},
    415: {'description': 'Unsupported file format'},
    422: {'description': 'Invalid CSV encoding, schema or numerical data'},
    500: {'description': 'Unexpected analysis failure'},
})
async def analyze(service: Annotated[UploadAnalysisService, Depends(get_upload_service)],
                  settings: Annotated[AppSettings, Depends(get_settings)],
                  file: UploadFile = File(...), uf: str = Form('MG')) -> Any:
    """Analyze a UTF-8 normalized or UFMG CSV directly, without writing shared reports."""
    try:
        uf = uf.strip().upper()
        if uf not in UF_CODES:
            return upload_error(400, 'INVALID_UF', 'uf must be a valid Brazilian state abbreviation.')
        if not file.filename or not file.filename.lower().endswith('.csv'):
            return upload_error(415, 'UNSUPPORTED_FORMAT', 'Upload a file with a .csv extension.')
        content = bytearray()
        while chunk := await file.read(min(64 * 1024, settings.max_upload_bytes + 1)):
            content.extend(chunk)
            if len(content) > settings.max_upload_bytes:
                return upload_error(413, 'UPLOAD_TOO_LARGE', 'The CSV exceeds the upload size limit.',
                                    {'max_bytes': settings.max_upload_bytes})
        return await run_in_threadpool(service.analyze, bytes(content), uf)
    except CSVValidationError as exc:
        return upload_error(422, exc.code, exc.message, exc.details)
    except Exception:
        logger.error('fleet_upload_analysis_failed', extra={'error_code': 'ANALYSIS_FAILED'})
        return upload_error(500, 'ANALYSIS_FAILED', 'An unexpected error prevented analysis.')
    finally:
        await file.close()

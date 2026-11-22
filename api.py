"""API de relatórios exportados e análise isolada de uploads CSV.

  pip install -r requirements-api.txt
  uvicorn api:app --reload --port 8000

Sem autenticação neste exemplo: habilite autenticação/autorização antes de expor publicamente.
"""
from pathlib import Path
import os
import json
import pandas as pd
from fastapi import FastAPI,HTTPException,Query,UploadFile,File,Form,Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.concurrency import run_in_threadpool
from fleet_force_engine.analysis_service import analyze_csv
from fleet_force_engine.csv_adapter import CSVValidationError
import logging

BASE_DIR=Path(__file__).resolve().parent
OUT=Path(os.getenv('FLEET_OUTPUT_DIR',str(BASE_DIR/'outputs')))
MAX_UPLOAD_BYTES=int(os.getenv('FLEET_MAX_UPLOAD_BYTES',10*1024*1024))
app=FastAPI(title='Fleet Force Analytics',version='1.0.0',description='Relatórios exportados e análise direta de CSV pelo motor unificado')


def json_file(name):
    p=OUT/name
    if not p.is_file():raise HTTPException(status_code=503,detail=f'Relatório {name} não gerado: execute run.py')
    return json.loads(p.read_text(encoding='utf-8'))


def csv(name):
    p=OUT/name
    if not p.is_file():raise HTTPException(status_code=503,detail=f'Relatório {name} não gerado')
    return pd.read_csv(p,encoding='utf-8-sig').replace([float('inf'),-float('inf')],float('nan'))


def records(df):
    # Não devolver NaN para clientes JSON.
    return json.loads(df.to_json(orient='records',force_ascii=False))


@app.get('/v1/fleet/summary')
def summary():return json_file('fleet_kpis.json')


@app.get('/v1/fleet/models/quality')
def models_quality():return json_file('fleet_model_metrics.json')


@app.get('/v1/fleet/vehicles')
def vehicles(offset:int=Query(0,ge=0),limit:int=Query(50,ge=1,le=200),
             category:str|None=None,only_priorities:bool=False):
    f=csv('fleet_vehicles.csv')
    if category is not None:f=f.loc[f.vehicle_category.eq(category)]
    if only_priorities:f=f.loc[f.unified_priority_alert.astype(str).str.lower().eq('true')]
    return {'total':int(len(f)),'offset':offset,'limit':limit,'items':records(f.iloc[offset:offset+limit])}


@app.get('/v1/fleet/vehicles/{fleet_id}')
def detail(fleet_id:str):
    f=csv('fleet_vehicles.csv')
    row=f.loc[f.fleet_id.eq(fleet_id.strip().upper())]
    if row.empty:raise HTTPException(status_code=404,detail='Veículo não encontrado')
    return records(row)[0]


@app.get('/v1/fleet/categories')
def categories():return records(csv('fleet_resumo_categorias.csv'))


@app.get('/v1/fleet/priorities')
def priorities():return records(csv('fleet_priority_ranking.csv'))


@app.get('/v1/fleet/reference-prices')
def prices():return records(csv('anp_reference_prices.csv'))


UF_CODES = set('AC AL AP AM BA CE DF ES GO MA MT MS MG PA PB PR PE PI RJ RN RS RO RR SC SP SE TO'.split())


def upload_error(status, code, message, details=None):
    return JSONResponse(status_code=status, content={
        'error': {'code': code, 'message': message, 'details': details or {}}})


@app.exception_handler(RequestValidationError)
async def request_validation_error(request: Request, exc: RequestValidationError):
    if request.url.path == '/v1/fleet/analyze':
        return upload_error(422, 'INVALID_REQUEST', 'A CSV file and valid form fields are required.')
    from fastapi.exception_handlers import request_validation_exception_handler
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    if request.url.path == '/v1/fleet/analyze':
        return upload_error(exc.status_code, 'INVALID_REQUEST', 'Invalid upload request.')
    from fastapi.exception_handlers import http_exception_handler
    return await http_exception_handler(request, exc)


@app.post('/v1/fleet/analyze', responses={
    400: {'description': 'Invalid state or request parameters'},
    413: {'description': 'Upload exceeds configured size limit'},
    415: {'description': 'Unsupported file format'},
    422: {'description': 'Invalid CSV encoding, schema or numerical data'},
    500: {'description': 'Unexpected analysis failure'},
})
async def analyze(file: UploadFile = File(...), uf: str = Form('MG')):
    """Analyze a UTF-8 normalized or UFMG CSV directly, without writing shared reports."""
    try:
        uf = uf.strip().upper()
        if uf not in UF_CODES:
            return upload_error(400, 'INVALID_UF', 'uf must be a valid Brazilian state abbreviation.')
        if not file.filename or not file.filename.lower().endswith('.csv'):
            return upload_error(415, 'UNSUPPORTED_FORMAT', 'Upload a file with a .csv extension.')
        content = bytearray()
        while chunk := await file.read(min(64 * 1024, MAX_UPLOAD_BYTES + 1)):
            content.extend(chunk)
            if len(content) > MAX_UPLOAD_BYTES:
                return upload_error(413, 'UPLOAD_TOO_LARGE', 'The CSV exceeds the upload size limit.',
                                    {'max_bytes': MAX_UPLOAD_BYTES})
        return await run_in_threadpool(analyze_csv, bytes(content), uf)
    except CSVValidationError as exc:
        return upload_error(422, exc.code, exc.message, exc.details)
    except Exception:
        logging.getLogger(__name__).error('Fleet upload analysis failed')
        return upload_error(500, 'ANALYSIS_FAILED', 'An unexpected error prevented analysis.')
    finally:
        await file.close()

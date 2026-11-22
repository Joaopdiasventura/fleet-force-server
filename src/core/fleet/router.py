"""HTTP routes for read-only exported fleet reports."""
from typing import Annotated, Any
from fastapi import APIRouter, Depends, Query

from src.config.dependencies import get_report_service
from .reports import FleetReportService

router = APIRouter(prefix='/v1/fleet')
Reports = Annotated[FleetReportService, Depends(get_report_service)]


@router.get('/summary', response_model=None)
def summary(service: Reports) -> Any:
    return service.summary()


@router.get('/models/quality', response_model=None)
def models_quality(service: Reports) -> Any:
    return service.model_quality()


@router.get('/vehicles', response_model=None)
def vehicles(service: Reports, offset: int = Query(0, ge=0),
             limit: int = Query(50, ge=1, le=200), category: str | None = None,
             only_priorities: bool = False) -> Any:
    return service.vehicles(offset, limit, category, only_priorities)


@router.get('/vehicles/{fleet_id}', response_model=None)
def detail(fleet_id: str, service: Reports) -> Any:
    return service.vehicle(fleet_id)


@router.get('/categories', response_model=None)
def categories(service: Reports) -> Any:
    return service.table('fleet_resumo_categorias.csv')


@router.get('/priorities', response_model=None)
def priorities(service: Reports) -> Any:
    return service.table('fleet_priority_ranking.csv')


@router.get('/reference-prices', response_model=None)
def prices(service: Reports) -> Any:
    return service.table('anp_reference_prices.csv')

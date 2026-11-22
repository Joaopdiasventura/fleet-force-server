"""Explicit composition and FastAPI dependencies; no service container."""
from fastapi import Request

from src.core.fleet.csv_adapter import UploadCSVDecoder
from src.core.fleet.reports import FleetReportService
from src.core.fleet.repositories import FileReportRepository
from src.core.references.providers import ProcessedReferenceProvider
from src.core.analytics.service import AnalysisService
from src.core.analytics.uploads import UploadAnalysisService
from .settings import AppSettings


def build_upload_service(settings: AppSettings) -> UploadAnalysisService:
    return UploadAnalysisService(UploadCSVDecoder(),
                                 ProcessedReferenceProvider(settings.anp_path, settings.pbev_path),
                                 AnalysisService())


def get_settings(request: Request) -> AppSettings:
    return request.app.state.settings


def get_upload_service(request: Request) -> UploadAnalysisService:
    return request.app.state.upload_service


def get_report_service(request: Request) -> FleetReportService:
    return request.app.state.report_service


def build_report_service(settings: AppSettings) -> FleetReportService:
    return FleetReportService(FileReportRepository(settings.output_dir))

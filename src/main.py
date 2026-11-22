from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.dependencies import build_report_service, build_upload_service
from src.config.http import install_error_handlers
from src.config.settings import AppSettings
from src.core.analytics.router import router as analytics_router
from src.core.fleet.router import router as fleet_router
from src.shared.logging import configure_logging


def create_app(settings: AppSettings | None = None) -> FastAPI:
    settings = settings if settings is not None else AppSettings.from_env()
    configure_logging()

    app = FastAPI(
        title='Fleet Force Analytics',
        version='1.0.0',
        description='Relatórios exportados e análise direta de CSV pelo motor unificado'
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://fleet-force.joaopdias.dev.br",
            "http://localhost:4200",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.state.settings = settings
    app.state.upload_service = build_upload_service(settings)
    app.state.report_service = build_report_service(settings)
    install_error_handlers(app)
    app.include_router(fleet_router)
    app.include_router(analytics_router)
    return app


app = create_app()

"""Read-only report queries independent of storage and HTTP."""
from typing import Any

from src.shared.serialization import records
from .exceptions import VehicleNotFound
from .ports import ReportRepository


class FleetReportService:
    def __init__(self, repository: ReportRepository) -> None:
        self.repository = repository

    def summary(self) -> dict[str, Any]:
        return self.repository.read_json('fleet_kpis.json')

    def model_quality(self) -> dict[str, Any]:
        return self.repository.read_json('fleet_model_metrics.json')

    def vehicles(self, offset: int, limit: int, category: str | None,
                 only_priorities: bool) -> dict[str, Any]:
        fleet = self.repository.read_frame('fleet_vehicles.csv')
        if category is not None:
            fleet = fleet.loc[fleet.vehicle_category.eq(category)]
        if only_priorities:
            fleet = fleet.loc[fleet.unified_priority_alert.astype(str).str.lower().eq('true')]
        return {'total': int(len(fleet)), 'offset': offset, 'limit': limit,
                'items': records(fleet.iloc[offset:offset + limit], double_precision=10)}

    def vehicle(self, fleet_id: str) -> dict[str, Any]:
        fleet = self.repository.read_frame('fleet_vehicles.csv')
        row = fleet.loc[fleet.fleet_id.eq(fleet_id.strip().upper())]
        if row.empty:
            raise VehicleNotFound('Veículo não encontrado')
        return records(row, double_precision=10)[0]

    def table(self, name: str) -> list[dict[str, Any]]:
        return records(self.repository.read_frame(name), double_precision=10)

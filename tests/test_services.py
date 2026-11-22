"""Application tests using injected in-memory collaborators and frozen legacy output."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from src.core.analytics.service import AnalysisService
from src.core.analytics.uploads import UploadAnalysisService
from src.core.fleet.csv_adapter import UploadCSVDecoder
from src.core.fleet.exceptions import ReportUnavailable, VehicleNotFound
from src.core.fleet.reports import FleetReportService
from src.core.fleet.service import ingest_fleet_dataframe
from src.core.policy import Settings
from src.core.references.models import ReferenceData

FIXTURES = Path(__file__).parent / 'fixtures'


class EmptyReferences:
    def load(self) -> ReferenceData:
        return ReferenceData()


class MemoryReports:
    def __init__(self) -> None:
        self.fleet = pd.DataFrame([
            {'fleet_id': 'A', 'vehicle_category': 'automovel', 'unified_priority_alert': True},
            {'fleet_id': 'B', 'vehicle_category': 'van', 'unified_priority_alert': False},
        ])

    def read_frame(self, name: str) -> pd.DataFrame:
        return self.fleet.copy()

    def read_json(self, name: str) -> dict:
        return {'vehicles': 2}


class ApplicationServiceTests(unittest.TestCase):
    def test_complete_response_matches_pre_refactor_snapshot(self):
        content = (FIXTURES / 'fleet_raw.csv').read_bytes()
        expected = json.loads((FIXTURES / 'analysis_baseline.json').read_text())
        service = UploadAnalysisService(UploadCSVDecoder(), EmptyReferences(), AnalysisService())
        # Any accidental infrastructure access from the pipeline fails this test.
        with patch('builtins.open', side_effect=AssertionError('filesystem is outside this use case')), \
             patch('pandas.read_csv', side_effect=AssertionError('CSV decoder must stay in memory')):
            actual = service.analyze(content, 'MG')
        actual['analysis_id'] = 'baseline'
        self.assertEqual(actual, expected)

    def test_pipeline_does_not_mutate_inputs_or_reference_tables(self):
        raw = pd.read_csv(FIXTURES / 'fleet_raw.csv')
        fleet, duplicates = ingest_fleet_dataframe(raw)
        original = fleet.copy(deep=True)
        monthly = pd.DataFrame([{'mes': '2025-12', 'uf': 'MG', 'produto': 'GASOLINA', 'preco_medio_brl_l': 6.0}])
        refs = ReferenceData(monthly=monthly)
        result = AnalysisService().execute(fleet, duplicates, refs, Settings(), inference_only=True)
        assert_frame_equal(fleet, original)
        self.assertEqual(monthly.iloc[0].preco_medio_brl_l, 6.0)
        self.assertEqual(result.kpis['total_expense_brl'], 8000)
        priority = result.fleet.loc[result.fleet.priority_alert].iloc[0]
        self.assertAlmostEqual(priority.actionable_excess_liters_est, 2000 / 8 - 2000 / 11)
        self.assertAlmostEqual(priority.peer_cost_gasolina_scenario_brl, (2000 / 8 - 2000 / 11) * 6)

    def test_report_queries_use_repository_contract(self):
        service = FleetReportService(MemoryReports())
        self.assertEqual(service.summary(), {'vehicles': 2})
        self.assertEqual(service.vehicles(0, 50, None, True)['total'], 1)
        self.assertEqual(service.vehicles(0, 50, 'van', False)['items'][0]['fleet_id'], 'B')
        self.assertEqual(service.vehicle(' a ')['fleet_id'], 'A')
        with self.assertRaises(VehicleNotFound):
            service.vehicle('unknown')

    def test_exception_from_repository_is_not_hidden_by_query_service(self):
        repository = MemoryReports()
        with patch.object(repository, 'read_json', side_effect=ReportUnavailable('unavailable')):
            with self.assertRaises(ReportUnavailable):
                FleetReportService(repository).summary()


if __name__ == '__main__':
    unittest.main()

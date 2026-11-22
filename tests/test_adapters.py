"""Integration coverage for filesystem adapters and entry-point composition."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from src.config.engine import run_engine
from src.config.settings import AppSettings, BASE_DIR
from src.core.fleet.adapters import parse_ufmg_pdf
from src.core.fleet.exceptions import ReportUnavailable
from src.core.fleet.repositories import FileReportRepository
from src.core.learning.artifacts import JoblibModelArtifacts
from src.core.references.providers import ProcessedReferenceProvider, FileReferenceProvider, _reference
from src.core.references import providers
from src.main import create_app

FIXTURES = Path(__file__).parent / 'fixtures'


class AdapterTests(unittest.TestCase):
    def setUp(self):
        _reference.cache_clear()

    def test_reference_cache_invalidation_and_request_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'anp.csv'
            path.write_text('mes,uf,produto,preco_medio_brl_l\n2025-12,MG,GASOLINA,6.0\n')
            provider = ProcessedReferenceProvider(path, Path(directory) / 'absent.csv')
            with patch.object(providers, 'load_anp_monthly', wraps=providers.load_anp_monthly) as load:
                first = provider.load()
                first.monthly.loc[0, 'preco_medio_brl_l'] = 99
                self.assertEqual(provider.load().monthly.iloc[0].preco_medio_brl_l, 6.0)
                self.assertEqual(load.call_count, 1)
                path.write_text('mes,uf,produto,preco_medio_brl_l\n2025-12,MG,GASOLINA,12.5\n')
                self.assertEqual(provider.load().monthly.iloc[0].preco_medio_brl_l, 12.5)
                self.assertEqual(load.call_count, 2)

    def test_explicit_zip_rebuild_retains_audits(self):
        monthly = pd.DataFrame([{'mes': '2025-12', 'uf': 'MG', 'produto': 'GASOLINA', 'preco_medio_brl_l': 6}])
        catalog = pd.DataFrame([{'modelo': 'ONIX', 'kmpl_cidade': 10, 'combustivel': 'GASOLINA',
                                 'ano_referencia': 2025, 'confianca_extracao': 'heuristic'}])
        audit = pd.DataFrame([{'status': 'extracted'}])
        with patch('src.core.references.providers.anp_monthly_from_zip', return_value=(monthly, audit)), \
             patch('src.core.references.pbev_extract.extract_pbev_history', return_value=(catalog, audit)):
            data = FileReferenceProvider(source_zip='trusted.zip', rebuild=True).load()
        self.assertEqual(data.price_source, 'ANP_from_zip_sample_mean_monthly')
        self.assertEqual(data.pbev_source, 'historical_pbev_zip_heuristic_layout_extraction')
        self.assertEqual(data.pbev.iloc[0].kmpl_reference, 10)
        self.assertFalse(data.price_audit.empty)
        self.assertFalse(data.pbev_audit.empty)
        self.assertFalse(data.raw_catalog.empty)

    def test_processed_references_take_precedence_over_zip(self):
        with patch('src.core.references.providers.anp_monthly_from_zip', side_effect=AssertionError('must not rebuild')), \
             patch('src.core.references.pbev_extract.extract_pbev_history', side_effect=AssertionError('must not rebuild')):
            data = FileReferenceProvider(AppSettings().anp_path, AppSettings().pbev_path, 'unused.zip').load()
        self.assertFalse(data.monthly.empty)
        self.assertFalse(data.pbev.empty)

    def test_pdf_source_and_file_reports(self):
        raw = parse_ufmg_pdf(BASE_DIR / 'data/despesa-veiculos-ufmg-2025.pdf')
        self.assertEqual(len(raw), 232)
        self.assertEqual(raw.fleet_id.nunique(), 217)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            run_engine(fleet_dataframe=raw, inference_only=True, output_dir=out)
            repository = FileReportRepository(out)
            self.assertEqual(repository.read_json('fleet_kpis.json')['vehicles'], 217)
            self.assertEqual(len(repository.read_frame('fleet_vehicles.csv')), 217)
            self.assertTrue((out / 'fleet_scenarios.csv').is_file())
            with self.assertRaises(ReportUnavailable):
                repository.read_json('missing.json')

    def test_candidate_serialization_is_an_explicit_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate.joblib'
            JoblibModelArtifacts().save({'candidate': 'trusted-test-artifact'}, path)
            self.assertEqual(joblib.load(path), {'candidate': 'trusted-test-artifact'})

    def test_application_factories_have_independent_settings_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(AppSettings(output_dir=Path(directory), max_upload_bytes=8))
            client = TestClient(app)
            self.assertEqual(client.get('/v1/fleet/summary').json(),
                             {'detail': 'Relatório fleet_kpis.json não gerado: execute run.py'})
            self.assertEqual(client.get('/v1/fleet/summary').status_code, 503)
            self.assertEqual(client.get('/v1/fleet/vehicles?limit=0').status_code, 422)
            self.assertEqual(client.post('/v1/fleet/analyze', files={'file': ('a.csv', b'0123456789')}).status_code, 413)
            self.assertEqual(create_app().state.settings.max_upload_bytes, 10 * 1024 * 1024)

    def test_environment_configuration(self):
        with patch.dict(os.environ, {'FLEET_OUTPUT_DIR': '/tmp/fleet-settings-test', 'FLEET_MAX_UPLOAD_BYTES': '1024'}):
            settings = AppSettings.from_env()
        self.assertEqual(settings.max_upload_bytes, 1024)
        self.assertEqual(settings.output_dir, Path('/tmp/fleet-settings-test'))

    def test_cli_writes_legacy_artifacts_from_explicit_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run([sys.executable, str(BASE_DIR / 'run.py'),
                                        '--fleet-csv', str(FIXTURES.resolve() / 'fleet_raw.csv'),
                                        '--out', directory], cwd=directory, text=True,
                                       capture_output=True, check=True, timeout=30)
            response = json.loads(completed.stdout)
            self.assertEqual(response['kpis']['vehicles'], 4)
            self.assertEqual(response['kpis']['total_expense_brl'], 8000)
            self.assertFalse(response['ml_efficiency_approved'])
            self.assertTrue((Path(directory) / 'fleet_vehicles.csv').is_file())
            self.assertTrue((Path(directory) / 'fleet_manifest.json').is_file())


if __name__ == '__main__':
    unittest.main()

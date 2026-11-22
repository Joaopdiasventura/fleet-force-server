import csv
import io
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import api
from fleet_force_engine import Settings, run_engine
from fleet_force_engine import analysis_service
from fleet_force_engine.config import FLEET_COLUMNS
from fleet_force_engine.csv_adapter import UFMG_HEADERS, parse_upload_csv
from fleet_force_engine.serialization import serialize_analysis


class AnalyzeTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def upload(self, content, filename='fleet.csv', **kwargs):
        return self.client.post('/v1/fleet/analyze', files={'file': (filename, content, 'text/csv')}, **kwargs)

    def normalized(self, fleet_id='001', km=2000, kmpl='10', expense=1000):
        return f'fleet_id,model,km,kmpl,total_expense\n{fleet_id},ONIX 1.0,{km},{kmpl},{expense}\n'

    def raw(self, delimiter=';', brazilian=True, duplicate=False):
        buf = io.StringIO()
        writer = csv.writer(buf, delimiter=delimiter)
        # Deliberately reverse report headers: mapping must never be positional.
        headers = list(UFMG_HEADERS)[::-1]
        writer.writerow(headers)
        for i, efficiency in enumerate([8, 10, 11, 12]):
            row = dict.fromkeys(FLEET_COLUMNS, '0')
            row.update(fleet_id=f'ABC{i:04d}', model='ONIX 1.0', os_count='1',
                       km='2.000,00' if brazilian else '2,000.00',
                       kmpl_reported=str(efficiency), total_expense='1.234,56' if brazilian else '1,234.56',
                       fuel_expense='1.000,00' if brazilian else '1,000.00')
            values = [row[UFMG_HEADERS[h]] for h in headers]
            writer.writerow(values)
            if duplicate and i == 0:
                writer.writerow(values)
        return buf.getvalue()

    def test_normalized_comma_and_identifier(self):
        r = self.upload(self.normalized())
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data['vehicles'][0]['fleet_id'], '001')
        self.assertEqual(data['summary']['total_expense_brl'], 1000)
        self.assertEqual(data['vehicles'][0]['liters_est'], 200)
        self.assertEqual(data['metadata']['input_format'], 'normalized')
        for key in ['categories', 'priorities', 'quality', 'comparisons', 'financial', 'reference_prices', 'ml']:
            self.assertIn(key, data)

    def test_ufmg_delimiters_bom_and_locale(self):
        for delimiter in [';', ',']:
            for brazilian in [True, False]:
                with self.subTest(delimiter=delimiter, brazilian=brazilian):
                    content = ('\ufeff' + self.raw(delimiter, brazilian)).encode()
                    r = self.upload(content)
                    self.assertEqual(r.status_code, 200, r.text)
                    data = r.json()
                    self.assertEqual(data['metadata']['input_format'], 'ufmg')
                    self.assertAlmostEqual(data['summary']['total_expense_brl'], 4 * 1234.56)
                    self.assertEqual(data['summary']['peer_priority_cases'], 1)
                    self.assertEqual(len(data['priorities']), 1)

    def test_header_variations(self):
        content = ' FRÓTA ; MODELO ; km percorrido ; km/l ; r$ peças\n abC ;ONIX;2000;10;12,50\n'
        r = self.upload(content)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()['vehicles'][0]['parts'], 12.5)
        self.assertEqual(r.json()['vehicles'][0]['fleet_id'], 'ABC')

    def test_validation_errors(self):
        cases = [
            ('model,km\nONIX,1000', 422, 'INVALID_CSV_SCHEMA'),
            (self.normalized(kmpl='oops'), 422, 'INVALID_NUMERIC_DATA'),
            ('fleet_id;model;km\nA;ONIX;1,234', 422, 'INVALID_NUMERIC_DATA'),
            ('fleet_id;model;km\nA;ONIX;1.234', 422, 'INVALID_NUMERIC_DATA'),
            (self.normalized(kmpl='Infinity'), 422, 'INVALID_NUMERIC_DATA'),
            ('', 422, 'EMPTY_FILE'),
            ('fleet_id,model,km\n', 422, 'EMPTY_FILE'),
            (b'fleet_id,model,km\nA,\xff,1', 422, 'INVALID_ENCODING'),
            ('fleet_id,model,km\nA,ONIX,1,extra', 422, 'MALFORMED_CSV'),
            ('fleet_id,model,km\nA,"ONIX,1', 422, 'MALFORMED_CSV'),
            ('fleet_id,model,km\n,ONIX,1', 422, 'INVALID_CSV_DATA'),
            ('fleet_id,Frota,model,km\nA,A,ONIX,1', 422, 'INVALID_CSV_SCHEMA'),
        ]
        for content, status, code in cases:
            with self.subTest(code=code, content=content):
                r = self.upload(content)
                self.assertEqual(r.status_code, status, r.text)
                self.assertEqual(r.json()['error']['code'], code)
                self.assertNotIn(str(analysis_service.BASE_DIR), r.text)

    def test_file_and_request_validation(self):
        self.assertEqual(self.client.post('/v1/fleet/analyze').status_code, 422)
        self.assertEqual(self.upload(self.normalized(), 'fleet.exe').status_code, 415)
        self.assertEqual(self.upload(self.normalized(), data={'uf': 'ZZ'}).status_code, 400)
        with patch.object(api, 'MAX_UPLOAD_BYTES', 20):
            r = self.upload(self.normalized())
            self.assertEqual(r.status_code, 413)
            self.assertEqual(r.json()['error']['code'], 'UPLOAD_TOO_LARGE')
        r = self.upload(self.normalized(), data={'uf': ' sp '})
        self.assertEqual(r.json()['metadata']['price_reference_state'], 'SP')

    def test_duplicate_audit_and_consolidation(self):
        r = self.upload(self.raw(duplicate=True))
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(len(data['quality']['duplicates']), 2)
        self.assertEqual(data['summary']['total_km'], 8000)
        self.assertEqual(data['summary']['duplicate_review_vehicles'], 1)
        # Different records for same vehicle sum using the existing weighted-efficiency rule.
        content = self.raw() + self.raw().splitlines()[1].replace('2.000,00', '3.000,00') + '\n'
        r = self.upload(content)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()['summary']['total_km'], 11000)
        partial = 'fleet_id,model,km\nA,ONIX,1000\nA,ONIX,2000\n'
        self.assertEqual(self.upload(partial).status_code, 422)

    def test_quality_zero_missing_and_implausible_efficiency(self):
        for km, kmpl in [(0, '10'), (2000, ''), (2000, '0'), (2000, '-1'), (2000, '99')]:
            r = self.upload(self.normalized(km=km, kmpl=kmpl))
            self.assertEqual(r.status_code, 200, r.text)
            vehicle = r.json()['vehicles'][0]
            self.assertTrue(vehicle['requires_review'])
            self.assertFalse(vehicle['priority_alert'])
        r = self.upload(self.normalized())
        self.assertEqual(r.json()['vehicles'][0]['peer_status'], 'insufficient_other_peers')

    def test_optional_expenses_remain_null(self):
        data = self.upload('fleet_id,model,km\n001,ONIX,2000').json()
        self.assertIsNone(data['summary']['total_expense_brl'])
        self.assertIsNone(data['vehicles'][0]['parts'])
        self.assertIsNone(data['categories'][0]['total_expense_brl'])
        self.assertTrue(all(x['total_brl'] is None for x in data['financial']['expense_distribution']))

    def test_no_ml_training_and_missing_references(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / 'missing.csv'
            with patch.object(analysis_service, 'ANP_PATH', missing), patch.object(analysis_service, 'PBEV_PATH', missing), \
                 patch('fleet_force_engine.engine.fit_efficiency', side_effect=AssertionError('must not train')), \
                 patch('fleet_force_engine.engine.fit_price_model', side_effect=AssertionError('must not train')), \
                 patch('fleet_force_engine.engine.add_anomalies', side_effect=AssertionError('must not fit')):
                r = self.upload(self.raw())
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data['ml']['status'], 'deterministic_fallback')
        self.assertFalse(data['ml']['used_for_operational_alerts'])
        self.assertFalse(data['ml']['models']['efficiency']['approved_production'])
        self.assertFalse(data['metadata']['availability']['anp'])
        self.assertEqual(data['reference_prices'], [])
        self.assertEqual(data['comparisons']['public_references'], [])
        self.assertEqual(data['summary']['peer_priority_cases'], 1)

    def test_direct_engine_parity(self):
        content = self.raw(brazilian=False)
        frame, _ = parse_upload_csv(content.encode())
        # Original path-based interface, including candidate gates, with a small deterministic fixture.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fleet.csv'
            frame.to_csv(path, index=False)
            result = run_engine(fleet_csv=path, cfg=Settings(), output_dir=Path(directory) / 'reports')
            self.assertTrue((Path(directory) / 'reports/fleet_kpis.json').is_file())
        with patch.object(analysis_service, 'references', return_value=pd.DataFrame()):
            data = self.upload(content).json()
        expected = serialize_analysis(result, 'direct', 'ufmg', 'MG')
        self.assertEqual(data['summary'], expected['summary'])
        self.assertEqual(data['categories'], expected['categories'])
        for a, b in zip(data['vehicles'], expected['vehicles']):
            for c in ['total_expense', 'liters_est', 'peer_median_kmpl', 'priority_alert', 'cost_per_km',
                      'actionable_excess_liters_est', 'unified_priority_source']:
                self.assertEqual(a[c], b[c])

    def test_concurrent_isolation_no_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            marker = out / 'sentinel'
            marker.write_text('unchanged')
            # Reject any attempt to export models, results, or create our own temporary upload.
            with patch.object(api, 'OUT', out), \
                 patch('fleet_force_engine.engine.export_results', side_effect=AssertionError('must not export')), \
                 patch('tempfile.TemporaryDirectory', side_effect=AssertionError('must stay in memory')):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(self.upload, self.normalized(fleet_id=f'00{i}', expense=i * 1000)) for i in [1, 2]]
                    responses = [f.result() for f in futures]
            self.assertTrue(all(r.status_code == 200 for r in responses))
            self.assertEqual([r.json()['summary']['total_expense_brl'] for r in responses], [1000, 2000])
            self.assertEqual([r.json()['vehicles'][0]['fleet_id'] for r in responses], ['001', '002'])
            self.assertNotEqual(responses[0].json()['analysis_id'], responses[1].json()['analysis_id'])
            self.assertEqual(list(out.iterdir()), [marker])
            self.assertEqual(marker.read_text(), 'unchanged')

    def test_json_and_unexpected_error(self):
        data = self.upload(self.normalized(km=0, kmpl='')).json()
        json.dumps(data, allow_nan=False)
        with patch.object(api, 'analyze_csv', side_effect=RuntimeError('/secret/internal/path')):
            r = self.upload(self.normalized())
            self.assertEqual(r.status_code, 500)
            self.assertNotIn('secret', r.text)
        frame = pd.DataFrame({'x': [np.nan, np.inf, -np.inf]})
        from fleet_force_engine.serialization import records
        self.assertEqual(records(frame), [{'x': None}] * 3)

    def test_upload_file_closed_on_success_and_errors(self):
        from starlette.datastructures import UploadFile
        original_close = UploadFile.close
        closed_files = []

        async def track_close(upload):
            await original_close(upload)
            closed_files.append(upload.file)

        with patch.object(UploadFile, 'close', track_close):
            self.assertEqual(self.upload(self.normalized()).status_code, 200)
            self.assertEqual(self.upload('broken').status_code, 422)
            self.assertEqual(self.upload('broken', 'file.exe').status_code, 415)
            with patch.object(api, 'MAX_UPLOAD_BYTES', 8):
                self.assertEqual(self.upload(self.normalized()).status_code, 413)
            # Exceed the multipart spool threshold to exercise parser-owned disk cleanup.
            self.assertEqual(self.upload('x' * (2 * 1024 * 1024)).status_code, 422)
        self.assertGreaterEqual(len(closed_files), 5)
        self.assertTrue(all(file.closed for file in closed_files))

    def test_unreadable_references_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            corrupt = Path(directory) / 'corrupt.csv'
            corrupt.write_text('not,a,reference\n1,2,3')
            with patch.object(analysis_service, 'ANP_PATH', corrupt), patch.object(analysis_service, 'PBEV_PATH', corrupt):
                r = self.upload(self.normalized())
        self.assertEqual(r.status_code, 200, r.text)
        self.assertFalse(r.json()['metadata']['availability']['anp'])

    def test_get_endpoints_and_openapi(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'fleet_kpis.json').write_text('{"vehicles": 1}')
            (out / 'fleet_model_metrics.json').write_text('{}')
            pd.DataFrame([{'fleet_id': 'ABC', 'vehicle_category': 'automovel', 'unified_priority_alert': True}]).to_csv(out / 'fleet_vehicles.csv', index=False)
            for filename in ['fleet_resumo_categorias.csv', 'fleet_priority_ranking.csv', 'anp_reference_prices.csv']:
                pd.DataFrame({'value': [1]}).to_csv(out / filename, index=False)
            with patch.object(api, 'OUT', out):
                for path in ['summary', 'models/quality', 'vehicles', 'vehicles/ABC', 'categories', 'priorities', 'reference-prices']:
                    self.assertEqual(self.client.get('/v1/fleet/' + path).status_code, 200)
        self.assertEqual(self.client.get('/docs').status_code, 200)
        operation = self.client.get('/openapi.json').json()['paths']['/v1/fleet/analyze']['post']
        self.assertIn('multipart/form-data', operation['requestBody']['content'])


if __name__ == '__main__':
    unittest.main()

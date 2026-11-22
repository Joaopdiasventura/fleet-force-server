"""Guard domain/infrastructure boundaries and the public API description."""
import ast
import json
from pathlib import Path
import unittest

from src.main import create_app

ROOT = Path(__file__).resolve().parents[1]


class ArchitectureTests(unittest.TestCase):
    def test_openapi_contract_matches_original_application(self):
        expected = json.loads((ROOT / 'tests/fixtures/openapi_baseline.json').read_text())
        self.assertEqual(create_app().openapi(), expected)

    def test_domain_rules_and_services_do_not_import_infrastructure(self):
        domain_modules = [
            'src/core/policy.py', 'src/core/fleet/service.py',
            'src/core/fleet/classification.py', 'src/core/fleet/reports.py',
            'src/core/analytics/service.py', 'src/core/analytics/uploads.py',
            'src/core/analytics/calculations.py', 'src/core/analytics/models.py',
            'src/core/references/rules.py', 'src/core/references/models.py',
            'src/core/learning/service.py', 'src/core/learning/efficiency.py',
            'src/core/learning/prices.py', 'src/core/learning/anomalies.py',
        ]
        forbidden = ('fastapi', 'starlette', 'pathlib', 'os', 'joblib', 'src.config', 'fleet_force_engine')
        for module in domain_modules:
            for node in ast.walk(ast.parse((ROOT / module).read_text())):
                imports = []
                if isinstance(node, ast.Import):
                    imports = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imports = [node.module or '']
                for name in imports:
                    self.assertFalse(any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden),
                                     f'{module} imports infrastructure: {name}')

    def test_compatibility_entry_points_delegate_to_single_implementation(self):
        import api
        from src.main import app
        from fleet_force_engine import run_engine, Settings, EngineOutput
        from src.config.engine import run_engine as implementation
        from src.core.policy import Settings as Policy
        from src.core.analytics.models import EngineOutput as Output
        self.assertIs(api.app, app)
        self.assertIs(run_engine, implementation)
        self.assertIs(Settings, Policy)
        self.assertIs(EngineOutput, Output)


if __name__ == '__main__':
    unittest.main()

"""Stable JSON contract built directly from EngineOutput, preserving engine columns."""
import json

from .config import EXPENSE_FIELDS
from .engine import _clean, fuel_scenario_records


def records(frame):
    return json.loads(frame.to_json(orient='records', force_ascii=False, double_precision=15))


def serialize_analysis(result, analysis_id, input_format, uf):
    f = result.fleet
    priorities = f.loc[f.unified_priority_alert].sort_values('unified_priority_liters_est', ascending=False)
    coverage = {c: {'available': int(f[c].notna().sum()), 'vehicles': len(f)}
                for c in ['km', 'kmpl', 'liters_est', 'total_expense', *EXPENSE_FIELDS,
                          'peer_median_kmpl', 'pbev_min_kmpl', 'ml_predicted_kmpl']}
    distribution = [{'field': c, 'total_brl': f[c].sum(min_count=1),
                     'available_vehicles': int(f[c].notna().sum())} for c in EXPENSE_FIELDS]
    scenarios = fuel_scenario_records(result)
    response = {
        'analysis_id': analysis_id, 'status': 'completed',
        'metadata': {**result.metadata, 'source': 'uploaded_csv', 'input_format': input_format,
                     'price_reference_state': uf,
                     'availability': {'anp': not result.prices.empty, 'pbev_catalog': not result.pbev.empty,
                                      'anomaly_analysis': False}},
        'summary': result.kpis, 'vehicles': records(f), 'categories': records(result.categories),
        'priorities': records(priorities),
        'quality': {'issues': records(f.loc[f.requires_review]),
                    'duplicates': records(result.quality_duplicates), 'coverage': coverage},
        'comparisons': {
            'internal': records(f.groupby(['model_norm', 'peer_confidence'], as_index=False)
                                .agg(vehicles=('fleet_id', 'count'), eligible=('valid_efficiency', 'sum'))),
            'public_references': records(f.loc[f.pbev_comparison_confidence.ne('unavailable')])},
        'financial': {'expense_distribution': distribution,
                      'pareto': records(f[['fleet_id', 'model', 'total_expense', 'cost_share_pct',
                                           'cost_cumulative_share_pct', 'pareto_80_flag']]),
                      'fuel_scenarios': scenarios},
        'reference_prices': records(result.prices),
        'ml': {'enabled': False, 'status': 'deterministic_fallback',
               'reason': 'No trusted independently validated model artifact is available',
               'used_for_operational_alerts': False, 'models': result.training_metrics,
               'crossval_predictions': records(result.crossval_predictions),
               'price_backtest': records(result.price_backtest),
               'price_forecast': records(result.price_forecast)},
    }
    return _clean(response)

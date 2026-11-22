"""The single analytical pipeline. No HTTP, paths, downloads or artifact writes."""
from typing import Any

import numpy as np
import pandas as pd

from src.core.policy import Settings
from src.core.learning.service import LearningService
from src.core.references.models import ReferenceData
from src.core.references.rules import match_pbev, add_fuel_scenarios
from .calculations import apply_quality, add_peer_benchmark, add_cost_analytics, summarize
from .models import EngineOutput


def unify_priorities(fleet: pd.DataFrame, kpis: dict[str, Any]) -> pd.DataFrame:
    fleet['unified_priority_alert'] = fleet['priority_alert'] | fleet['ml_priority_alert']
    fleet['unified_priority_source'] = np.select(
        [fleet['priority_alert'], fleet['ml_priority_alert']],
        ['deterministic_peer', 'validated_ml'], default='none')
    fleet['unified_priority_liters_est'] = np.where(
        fleet['priority_alert'], fleet['actionable_excess_liters_est'].fillna(0),
        np.where(fleet['ml_priority_alert'], fleet['ml_excess_liters_est'].clip(lower=0).fillna(0), np.nan))
    kpis['unified_priority_cases'] = int(fleet.unified_priority_alert.sum())
    kpis['validated_ml_reference_vehicles'] = int(
        fleet.active_reference_source.eq('ml_approved_external_validation').sum())
    return fleet


def analysis_metadata(cfg: Settings, references: ReferenceData) -> dict[str, Any]:
    return {
        'version': 'fleet-force-unified-v1', 'analysis_scope': 'annual_report_2025_for_UFMG',
        'currency': 'BRL', 'distance_unit': 'km', 'efficiency_unit': 'km_per_liter',
        'period_aggregation': 'UFMG_2025_annual_per_vehicle',
        'liters_source': 'estimated_from_distance_and_reported_efficiency',
        'price_uf': cfg.price_uf, 'price_source': references.price_source,
        'pbev_source': references.pbev_source,
        'expense_component_coverage': 'unknown_if_missing_columns',
        'model_promotion_policy': 'independent_holdout_required_for_operational_efficiency',
        'reference_priority': 'same_model_peer_first_then_independently_validated_ml',
        'price_scenarios': 'different_fuels_not_additive', 'settings': cfg.to_dict(), 'input_hashes': {},
    }


class AnalysisService:
    def __init__(self, learning: LearningService | None = None) -> None:
        self.learning = learning if learning is not None else LearningService()

    def execute(self, fleet: pd.DataFrame, duplicates: pd.DataFrame, references: ReferenceData,
                cfg: Settings, *, external: pd.DataFrame | None = None,
                inference_only: bool = False) -> EngineOutput:
        fleet = apply_quality(fleet, cfg)
        fleet = add_peer_benchmark(fleet, cfg)
        fleet = add_cost_analytics(fleet, cfg)
        fleet = self.learning.anomalies(fleet, cfg, inference_only)
        fleet = match_pbev(fleet, references.pbev)
        fleet, prices = add_fuel_scenarios(fleet, references.monthly, cfg.price_uf)
        if external is not None:
            external = apply_quality(external, cfg)
        learned = self.learning.evaluate(fleet, references.monthly, cfg, external, inference_only)
        kpis, categories = summarize(learned.fleet)
        fleet = unify_priorities(learned.fleet, kpis)
        return EngineOutput(
            fleet=fleet, categories=categories, kpis=kpis, training_metrics=learned.metrics,
            quality_duplicates=duplicates.copy(deep=True), prices=prices,
            price_forecast=learned.forecast, pbev=references.pbev.copy(deep=True),
            crossval_predictions=learned.crossval, price_backtest=learned.backtest,
            models=learned.models, metadata=analysis_metadata(cfg, references),
        )

"""Learning use cases: gate evaluation or explicit request-safe fallback."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.core.policy import Settings
from .efficiency import fit_efficiency, attach_efficiency_predictions
from .prices import fit_price_model, forecast_next_price
from .anomalies import add_anomalies


@dataclass
class LearningOutput:
    fleet: pd.DataFrame
    models: dict[str, Any]
    metrics: dict[str, Any]
    crossval: pd.DataFrame
    backtest: pd.DataFrame
    forecast: pd.DataFrame


class LearningService:
    def anomalies(self, fleet: pd.DataFrame, cfg: Settings, inference_only: bool) -> pd.DataFrame:
        if not inference_only:
            return add_anomalies(fleet, cfg)
        fleet = fleet.copy()
        fleet['anomaly_status_ml'] = 'not_evaluated_inference_only'
        fleet['anomaly_score_ml'] = np.nan
        fleet['anomaly_flag_ml'] = False
        return fleet

    def evaluate(self, fleet: pd.DataFrame, monthly: pd.DataFrame, cfg: Settings,
                 external: pd.DataFrame | None = None, inference_only: bool = False) -> LearningOutput:
        if inference_only:
            efficiency_model = None
            metrics_eff = {'status': 'not_loaded', 'approved_production': False,
                           'reason': 'no_trusted_independently_validated_artifact',
                           'automatic_recommendations': False}
            crossval = pd.DataFrame()
        else:
            efficiency_model, metrics_eff, crossval = fit_efficiency(fleet, cfg, external)
        fleet = attach_efficiency_predictions(fleet, efficiency_model, metrics_eff, cfg)
        if inference_only:
            price_model = None
            metrics_price = {'status': 'not_loaded', 'approved_production': False,
                             'reason': 'no_trusted_validated_artifact'}
            backtest = pd.DataFrame()
        else:
            price_model, metrics_price, backtest = fit_price_model(monthly, cfg)
        forecast = forecast_next_price(monthly, price_model, metrics_price)
        return LearningOutput(fleet, {'efficiency': efficiency_model, 'price': price_model},
                              {'efficiency': metrics_eff, 'price': metrics_price},
                              crossval, backtest, forecast)

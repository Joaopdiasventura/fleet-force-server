"""In-memory analytical output contract, shared by API and CLI."""
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class EngineOutput:
    fleet: pd.DataFrame
    categories: pd.DataFrame
    kpis: dict[str, Any]
    training_metrics: dict[str, Any]
    quality_duplicates: pd.DataFrame
    prices: pd.DataFrame
    price_forecast: pd.DataFrame
    pbev: pd.DataFrame
    crossval_predictions: pd.DataFrame
    price_backtest: pd.DataFrame
    models: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

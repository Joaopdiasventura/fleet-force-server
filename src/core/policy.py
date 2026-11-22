"""Versioned analytical policy; independent of runtime configuration."""
from dataclasses import dataclass, asdict
from typing import Any

@dataclass(frozen=True)
class Settings:
    min_distance_km: float=500.0
    min_other_peers: int=3
    moderate_overuse: float=.10
    high_overuse: float=.25
    min_priority_liters: float=25.0
    pbev_tolerance: float=.20
    price_uf: str='MG'
    # Exigências objetivas do gate de ML
    ml_min_train_rows: int=150
    ml_min_model_families: int=25
    ml_min_gain: float=.10
    ml_max_category_regression: float=.10
    ml_min_segment_samples: int=8
    ml_min_external_rows: int=40
    # Este relatório contém somente 2025; avaliação cruzada não substitui holdout independente
    ml_require_external_holdout: bool=True
    anomaly_contamination: float=.07
    default_period_months: int=12

    def to_dict(self) -> dict[str, Any]: return asdict(self)

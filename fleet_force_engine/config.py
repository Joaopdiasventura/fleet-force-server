"""Parâmetros versionáveis do Fleet Force. Limiares são regras de triagem, não verdades físicas."""
from dataclasses import dataclass, asdict

EXPENSE_FIELDS = (
    'labor_internal','labor_external','parts','tires','fuel_expense','lubricant','wash'
)
KMPL_BOUNDS = {
    'automovel': (1.,24.), 'picape':(1.,20.), 'van':(1.,20.),
    'caminhao':(.7,15.), 'onibus':(.7,12.), 'outros':(.5,30.),
    'desconhecido':(.5,30.)
}
GENERIC_EXACT = {
    'CAMINHAO','ONIBUS','VAN','MICRO ONIBUS','M POLO','MARCOPOLO',
    'COMIL','RODOVIARIO','ATHENAS','ALLEGRO','REBOQUE','TANQUE',
    'CAVALO MEC','BASCULANTE','TRUCK'
}
GENERIC_TOKENS = {'MARCOPOLO','COMIL','RODOVIARIO','ATHENAS','ALLEGRO'}
FLEET_COLUMNS = [
    'fleet_id','model','os_count','labor_internal','labor_external',
    'parts','tires','fuel_expense','lubricant','wash','km',
    'kmpl_reported','total_expense','cost_per_km'
]
FLEET_BOUNDS = [35,90,180,210,260,310,360,410,455,495,545,605,655,725,780]

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

    def to_dict(self): return asdict(self)

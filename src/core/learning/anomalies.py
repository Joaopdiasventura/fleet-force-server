"""Exploratory unsupervised anomaly analysis; never an operational diagnosis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from src.core.policy import Settings

def add_anomalies(fleet: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    """Isolation Forest apenas para investigação, NUNCA define prioridade financeira."""
    f=fleet.copy()
    f['anomaly_status_ml']='not_evaluated'
    f['anomaly_score_ml']=np.nan
    f['anomaly_flag_ml']=False
    mask=f['valid_efficiency']&f['cost_per_km'].gt(0)
    if mask.sum()<30: return f
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler
    cols=['km','kmpl','cost_per_km','maintenance_cost_per_km','fuel_share_pct','os_per_1000km']
    X=f.loc[mask,cols].copy().replace([np.inf,-np.inf],np.nan)
    X=SimpleImputer(strategy='median').fit_transform(X)
    X=RobustScaler().fit_transform(np.log1p(np.clip(X,0,None)))
    m=IsolationForest(contamination=cfg.anomaly_contamination,n_estimators=150,random_state=42)
    labels=m.fit_predict(X)
    f.loc[mask,'anomaly_flag_ml']=labels==-1
    f.loc[mask,'anomaly_score_ml']=-m.score_samples(X)
    f.loc[mask,'anomaly_status_ml']='exploratory_unsupervised_not_validated'
    return f



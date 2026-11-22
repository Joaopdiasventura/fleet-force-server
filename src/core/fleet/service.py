"""Pure fleet normalization, duplicate audit and consolidation rules."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from .exceptions import FleetDataError
from .fields import FLEET_COLUMNS, EXPENSE_FIELDS
from src.shared.numbers import norm, safe_div, numeric

def consolidate_raw_ufmg(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing=set(FLEET_COLUMNS)-set(raw)
    if missing: raise FleetDataError('Campos de extrato UFMG ausentes: '+str(sorted(missing)))
    raw=raw.copy()
    raw['fleet_id']=raw['fleet_id'].astype(str).str.strip().str.upper()
    for c in FLEET_COLUMNS[2:]:
        if c!='model':raw[c]=pd.to_numeric(raw[c],errors='coerce')
    # Mesma regra main.ipynb: duplicatas exatas não contam duas vezes,
    # mas preserva marcação de revisão no veículo afetado.
    duplicates=raw.loc[raw.duplicated(subset=FLEET_COLUMNS,keep=False)].copy()
    if not duplicates.empty:
        duplicates['audit_status']='pending_source_confirmation'
        duplicates['duplicate_key']=duplicates[FLEET_COLUMNS].astype(str).agg('|'.join,axis=1).map(lambda s:hashlib.sha256(s.encode()).hexdigest()[:16])
    rows=raw.drop_duplicates(subset=FLEET_COLUMNS,keep='first').copy()
    rows['provisional_duplicate_vehicle']=rows['fleet_id'].isin(duplicates['fleet_id'])
    valid=rows['km'].gt(0)&rows['kmpl_reported'].gt(0)&np.isfinite(rows['kmpl_reported'])
    rows['liters_est']=(rows['km']/rows['kmpl_reported']).where(valid)
    rows['km_covered']=rows['km'].where(valid,0.)
    rows['km_uncovered']=rows['km'].where(~valid&rows['km'].gt(0),0.)
    sums=list(EXPENSE_FIELDS)+['total_expense','os_count','km','km_covered','km_uncovered']
    agg={c:lambda s:s.sum(min_count=1) for c in sums}
    agg.update({'liters_est':lambda s:s.sum(min_count=1),'model':'first',
                'provisional_duplicate_vehicle':'max'})
    fleet=rows.groupby('fleet_id',as_index=False).agg(agg)
    fleet['row_count']=rows.groupby('fleet_id').size().reindex(fleet['fleet_id']).values
    fleet['kmpl']=safe_div(fleet['km_covered'],fleet['liters_est']).replace([np.inf,-np.inf],np.nan)
    fleet['kmpl_coverage_pct']=100*safe_div(fleet['km_covered'],fleet['km'])
    fleet['cost_per_km']=safe_div(fleet['total_expense'],fleet['km'])
    fleet['fuel_cost_per_km_historical']=safe_div(fleet['fuel_expense'],fleet['km'])
    fleet['model_norm']=fleet['model'].map(norm)
    fleet['liters_source']='estimated_from_distance_and_reported_kmpl'
    fleet['data_origin']='UFMG_PDF_2025'
    return fleet,duplicates


def ingest_fleet_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df=df.copy()
    if not {'fleet_id','model','km'}.issubset(df.columns):
        raise FleetDataError('CSV exige pelo menos fleet_id, model, km. Para dados detalhados, forneça formato original main.ipynb.')
    if 'kmpl' not in df and 'kmpl_reported' in df:
        df['kmpl']=df['kmpl_reported']
    df['fleet_id']=df['fleet_id'].astype(str).str.strip().str.upper()
    if set(FLEET_COLUMNS).issubset(df.columns): return consolidate_raw_ufmg(df)
    if df['fleet_id'].duplicated().any():
        raise FleetDataError('fleet_id duplicado no CSV agregado: forneça linhas brutas completas para consolidação.')
    for c in list(EXPENSE_FIELDS)+['total_expense','km','kmpl','km_covered','km_uncovered','liters_est','os_count']:
        if c not in df: df[c]=np.nan
    numeric(df,list(EXPENSE_FIELDS)+['total_expense','km','kmpl','km_covered','km_uncovered','liters_est','os_count'])
    # Não converter ausência em zero silenciosamente: mantendo NaN para campos não informados.
    if df['km_covered'].isna().all(): df['km_covered']=df['km'].where(df['kmpl'].gt(0),0.)
    if df['km_uncovered'].isna().all(): df['km_uncovered']=df['km'].where(~df['kmpl'].gt(0),0.)
    if df['liters_est'].isna().all(): df['liters_est']=safe_div(df['km_covered'],df['kmpl'])
    df['kmpl_coverage_pct']=100*safe_div(df['km_covered'],df['km'])
    df['cost_per_km']=safe_div(df['total_expense'],df['km'])
    df['fuel_cost_per_km_historical']=safe_div(df['fuel_expense'],df['km'])
    df['model_norm']=df['model'].map(norm)
    if 'provisional_duplicate_vehicle' not in df: df['provisional_duplicate_vehicle']=False
    else: df['provisional_duplicate_vehicle']=df['provisional_duplicate_vehicle'].astype(str).str.lower().isin(['true','1','sim'])
    if 'row_count' not in df: df['row_count']=1
    if 'data_origin' not in df: df['data_origin']='FLEET_CSV'
    if 'liters_source' not in df: df['liters_source']='estimated_from_distance_and_reported_kmpl'
    return df,pd.DataFrame()



"""US01/US02/US03: regras originais preservadas, com diagnóstico financeiro adicional."""
from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
from src.core.policy import Settings
from src.core.fleet.fields import KMPL_BOUNDS, EXPENSE_FIELDS
from src.shared.numbers import norm, safe_div
from src.core.fleet.classification import category_for_model, generic_model


def apply_quality(fleet: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    f=fleet.copy()
    f['model_norm']=f['model'].map(norm)
    f['vehicle_category']=f['model_norm'].map(category_for_model)
    flags=[]
    for r in f.itertuples(index=False):
        reasons=[]
        if pd.isna(r.km) or r.km<=0: reasons.append('missing_or_zero_distance')
        elif r.km<cfg.min_distance_km: reasons.append('insufficient_distance')
        if pd.notna(r.km_uncovered) and r.km_uncovered>0: reasons.append('incomplete_kmpl_coverage')
        if pd.isna(r.kmpl) or not np.isfinite(r.kmpl): reasons.append('kmpl_unavailable')
        else:
            lower,upper=KMPL_BOUNDS[r.vehicle_category]
            if r.kmpl<lower: reasons.append('kmpl_below_category_screening')
            if r.kmpl>upper: reasons.append('kmpl_above_category_screening')
        if r.provisional_duplicate_vehicle: reasons.append('duplicate_under_review')
        flags.append(reasons)
    f['quality_reason']=[';'.join(x) for x in flags]
    f['requires_review']=[bool(x) for x in flags]
    f['valid_efficiency']=~f['requires_review']
    f['quality_status']=np.where(f['valid_efficiency'],'eligible','review_required')
    f['model_identification_required']=f['model_norm'].map(generic_model)
    return f


def add_peer_benchmark(fleet: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    f=fleet.copy()
    valid=f.loc[f['valid_efficiency']]
    peers={n:g.set_index('fleet_id')['kmpl'] for n,g in valid.groupby('model_norm',sort=False)}
    results=[]
    for row in f.itertuples(index=False):
        group=peers.get(row.model_norm,pd.Series(dtype=float))
        others=group.drop(index=row.fleet_id,errors='ignore') if row.valid_efficiency else pd.Series(dtype=float)
        n=int(len(others)); compare=row.valid_efficiency and n>=cfg.min_other_peers
        median=float(others.median()) if compare else np.nan
        ratio=row.kmpl/median if compare and median>0 else np.nan
        overuse=median/row.kmpl-1 if compare and row.kmpl>0 else np.nan
        excess=row.km_covered/row.kmpl-row.km_covered/median if compare and median>0 else np.nan
        if not row.valid_efficiency: confidence,status='insufficient','review_data_quality'
        elif not compare: confidence,status='insufficient','insufficient_other_peers'
        elif row.model_identification_required: confidence,status='low','inconclusive_generic_model'
        elif overuse>=cfg.high_overuse: confidence,status='moderate','high_relative_consumption_indication'
        elif overuse>=cfg.moderate_overuse: confidence,status='moderate','moderate_relative_consumption_indication'
        elif overuse>0: confidence,status='moderate','slightly_below_peers'
        else: confidence,status='moderate','at_or_above_peers'
        results.append((n,median,ratio,overuse,excess,confidence,status))
    cols=['other_peer_count','peer_median_kmpl','peer_efficiency_ratio','peer_overuse_pct',
          'excess_liters_vs_peer_est','peer_confidence','peer_status']
    f[cols]=pd.DataFrame(results,index=f.index,columns=cols)
    f['peer_reference_method']='same_normalized_model_leave_one_out'
    f['peer_positive_liters_est']=f['excess_liters_vs_peer_est'].clip(lower=0).where(f['peer_confidence'].eq('moderate'))
    f['priority_alert']=(f['peer_confidence'].eq('moderate') & f['peer_overuse_pct'].ge(cfg.moderate_overuse)
                         & f['peer_positive_liters_est'].ge(cfg.min_priority_liters))
    f['actionable_excess_liters_est']=f['peer_positive_liters_est'].where(f['priority_alert'])
    f['exploratory_excess_liters_est']=f['excess_liters_vs_peer_est'].clip(lower=0).where(f['peer_confidence'].eq('low'))
    f['peer_efficiency_percentile']=f['peer_efficiency_ratio'].where(f['peer_confidence'].eq('moderate')).rank(pct=True)*100
    return f


def add_cost_analytics(fleet: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    f=fleet.copy()
    f['maintenance_expense_known']=f[['labor_internal','labor_external','parts','tires','lubricant']].sum(axis=1,min_count=1)
    f['maintenance_cost_per_km']=safe_div(f['maintenance_expense_known'],f['km'])
    f['fuel_share_pct']=100*safe_div(f['fuel_expense'],f['total_expense'])
    f['maintenance_share_pct']=100*safe_div(f['maintenance_expense_known'],f['total_expense'])
    f['os_per_1000km']=1000*safe_div(f['os_count'],f['km'])
    # Sem informação de uma rubrica, não podemos afirmar que a soma foi reconciliada.
    f['expense_components_complete']=f[list(EXPENSE_FIELDS)].notna().all(axis=1)
    components=f[list(EXPENSE_FIELDS)].sum(axis=1,min_count=1)
    f['expense_integrity_delta_brl']=(f['total_expense']-components).where(f['expense_components_complete'])
    f['expense_integrity_alert']=f['expense_integrity_delta_brl'].abs().gt(1.0).fillna(False)
    # Benchmarks financeiros de categoria (não interpretar como veículos tecnicamente equivalentes).
    rec=[]
    eligible=f['km'].ge(cfg.min_distance_km)&f['cost_per_km'].gt(0)&~f['requires_review']
    groups={n:g.set_index('fleet_id')['cost_per_km'] for n,g in f.loc[eligible].groupby('vehicle_category')}
    for r in f.itertuples(index=False):
        others=groups.get(r.vehicle_category,pd.Series(dtype=float)).drop(index=r.fleet_id,errors='ignore')
        if not (r.km>=cfg.min_distance_km and pd.notna(r.cost_per_km)) or len(others)<cfg.min_other_peers:
            rec.append((len(others),np.nan,np.nan,False)); continue
        med=float(others.median()); delta=r.cost_per_km/med-1 if med>0 else np.nan
        rec.append((len(others),med,delta,bool(delta>.50)))
    cc=['category_other_count','category_other_median_cost_km','category_cost_over_median_pct','category_cost_alert']
    f[cc]=pd.DataFrame(rec,columns=cc,index=f.index)
    f['cost_reference_method']='category_leave_one_out_exploratory'
    # Pareto 80/20: participação cumulativa, calculado sobre despesas registradas.
    f=f.sort_values(['total_expense','fleet_id'],ascending=[False,True],na_position='last').reset_index(drop=True)
    total=f['total_expense'].sum(min_count=1)
    f['cost_share_pct']=100*f['total_expense']/total if pd.notna(total) and total>0 else np.nan
    f['cost_cumulative_share_pct']=f['cost_share_pct'].fillna(0).cumsum()
    f['pareto_80_flag']=(f['cost_cumulative_share_pct']-f['cost_share_pct'].fillna(0)<80) & f['total_expense'].gt(0)
    return f


def summarize(fleet: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    f=fleet
    def total(col):return float(f[col].sum(min_count=1)) if col in f else float('nan')
    vehicles=int(f['fleet_id'].nunique()); km=total('km')
    kpis={
        'vehicles':vehicles,
        'eligible_vehicles':int(f['valid_efficiency'].sum()),
        'quality_review_vehicles':int(f['requires_review'].sum()),
        'total_km':km,
        'total_expense_brl':total('total_expense'),
        'fuel_expense_brl':total('fuel_expense'),
        'maintenance_expense_brl':total('maintenance_expense_known'),
        'cost_per_km_brl':total('total_expense')/km if km>0 else None,
        'mean_expense_per_vehicle_brl':total('total_expense')/vehicles if vehicles else None,
        'fuel_share_pct':100*total('fuel_expense')/total('total_expense') if total('total_expense')>0 else None,
        'eligible_peer_comparisons':int(f['peer_confidence'].eq('moderate').sum()),
        'peer_priority_cases':int(f['priority_alert'].sum()),
        'peer_priority_excess_liters_est':total('actionable_excess_liters_est'),
        'high_category_cost_cases':int(f['category_cost_alert'].sum()),
        'financial_integrity_review_cases':int(f['expense_integrity_alert'].sum()),
        'exploratory_anomalies':int(f.get('anomaly_flag_ml',pd.Series(False,index=f.index)).sum()),
        'duplicate_review_vehicles':int(f['provisional_duplicate_vehicle'].sum()),
    }
    groups=(f.groupby('vehicle_category',as_index=False)
        .agg(vehicles=('fleet_id','nunique'),total_km=('km','sum'),total_expense_brl=('total_expense',lambda s:s.sum(min_count=1)),
             fuel_expense_brl=('fuel_expense',lambda s:s.sum(min_count=1)),maintenance_expense_brl=('maintenance_expense_known',lambda s:s.sum(min_count=1)),
             priority_cases=('priority_alert','sum')))
    groups['cost_per_km_brl']=safe_div(groups['total_expense_brl'],groups['total_km'])
    return kpis,groups

"""Filesystem report generation adapter. Not part of the analytical pipeline."""
from __future__ import annotations
from .models import EngineOutput

from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.shared.serialization import clean_json
from src.core.references.rules import fuel_scenario_records
from src.core.learning.artifacts import JoblibModelArtifacts

def export_results(result: EngineOutput, out: str | Path, monthly: pd.DataFrame | None = None, pbev_audit: pd.DataFrame | None = None, price_audit: pd.DataFrame | None = None) -> None:
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    frames={
        'fleet_vehicles.csv':result.fleet,
        'fleet_qualidade.csv':result.fleet.loc[result.fleet.requires_review],
        'fleet_priority_ranking.csv':result.fleet.loc[result.fleet.unified_priority_alert]
            .sort_values('unified_priority_liters_est',ascending=False),
        'fleet_comparison_groups.csv':result.fleet.groupby(['model_norm','peer_confidence'],as_index=False)
            .agg(vehicles=('fleet_id','count'),eligible=('valid_efficiency','sum')),
        'fleet_public_comparisons.csv':result.fleet.loc[result.fleet.pbev_comparison_confidence.ne('unavailable')],
        'fleet_resumo_categorias.csv':result.categories,
        'fleet_pareto_custos.csv':result.fleet[['fleet_id','model','total_expense','cost_share_pct','cost_cumulative_share_pct','pareto_80_flag']],
        'fleet_auditoria_duplicatas.csv':result.quality_duplicates,
        'fleet_ml_backtest_consumo.csv':result.crossval_predictions,
        'anp_reference_prices.csv':result.prices,
        'anp_ml_backtest_temporal.csv':result.price_backtest,
        'anp_ml_previsao_proximo_mes.csv':result.price_forecast,
    }
    if monthly is not None and not monthly.empty:frames['anp_mensal_uf.csv']=monthly
    if pbev_audit is not None and not pbev_audit.empty:frames['pbev_auditoria_extracao.csv']=pbev_audit
    if price_audit is not None and not price_audit.empty:frames['anp_auditoria_arquivos.csv']=price_audit
    # Exportações retrocompatíveis com o contrato do main.ipynb.
    summary=pd.DataFrame([{'indicador':k,'valor':v,'unidade':('BRL' if k.endswith('_brl') else 'km' if k=='total_km' else 'count_or_ratio')}
                          for k,v in result.kpis.items()])
    frames['fleet_summary.csv']=summary
    frames['fleet_scenarios.csv']=pd.DataFrame(fuel_scenario_records(result.kpis, result.prices),columns=[
        'combustivel','uf','mes_referencia','preco_medio_referencia_brl_l','litros_excedentes_est','custo_cenario_brl','metodologia'])
    coverage=pd.DataFrame([{'campo':c,'disponiveis':int(result.fleet[c].notna().sum()),
                           'veiculos':len(result.fleet),
                           'cobertura_pct':float(result.fleet[c].notna().mean()*100)}
                           for c in result.fleet.columns if c in [
                               'km','kmpl','liters_est','total_expense','fuel_expense','os_count','parts','tires',
                               'maintenance_expense_known','peer_median_kmpl','pbev_min_kmpl','ml_predicted_kmpl']])
    frames['fleet_data_coverage.csv']=coverage
    for name,df in frames.items():df.to_csv(out/name,index=False,encoding='utf-8-sig')
    (out/'fleet_kpis.json').write_text(json.dumps(clean_json(result.kpis),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (out/'fleet_model_metrics.json').write_text(json.dumps(clean_json(result.training_metrics),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (out/'fleet_manifest.json').write_text(json.dumps(clean_json(result.metadata),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    models_path=out/'models';models_path.mkdir(exist_ok=True)
    for kind,model in result.models.items():
        if model is not None:JoblibModelArtifacts().save(model,models_path/f'{kind}_candidate.joblib')
    # Contrato útil para API/Angular, representando valores nulos como null.
    preview=result.fleet.head(100).replace([np.inf,-np.inf],np.nan)
    (out/'fleet_veiculos_preview.json').write_text(json.dumps(clean_json(preview.where(pd.notna(preview),None).to_dict(orient='records')),
            ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')



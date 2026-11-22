"""Orquestrador único: dados -> regras -> referências -> ML gate -> artefatos auditáveis."""
from dataclasses import dataclass,field
from pathlib import Path
import json
import hashlib
import math
import numpy as np
import pandas as pd
from .config import Settings,EXPENSE_FIELDS
from .ingest import load_fleet,ingest_fleet_dataframe
from .deterministic import apply_quality,add_peer_benchmark,add_cost_analytics,add_anomalies,summarize
from .references import load_pbev,match_pbev,load_anp_monthly,anp_monthly_from_zip,add_fuel_scenarios
from .learning import fit_efficiency,attach_efficiency_predictions,fit_price_model,forecast_next_price


def _json_default(obj):
    if isinstance(obj,(np.integer,)):return int(obj)
    if isinstance(obj,(np.floating,)):return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj,(np.bool_,)):return bool(obj)
    if isinstance(obj,(pd.Timestamp,)):return obj.isoformat()
    if isinstance(obj,Path):return str(obj)
    raise TypeError(str(type(obj)))


def _clean(o):
    # JSON interoperável: não enviar NaN/Inf para a API Angular.
    if isinstance(o,dict):return {str(k):_clean(v) for k,v in o.items()}
    if isinstance(o,(tuple,list)):return [_clean(v) for v in o]
    if isinstance(o,(np.integer,)):return int(o)
    if isinstance(o,(np.floating,float)):return float(o) if math.isfinite(float(o)) else None
    if isinstance(o,(np.bool_,)):return bool(o)
    if isinstance(o,(pd.Timestamp,)):return o.strftime('%Y-%m-%d')
    if o is pd.NA or o is None:return None
    return o


def _sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


@dataclass
class EngineOutput:
    fleet:pd.DataFrame
    categories:pd.DataFrame
    kpis:dict
    training_metrics:dict
    quality_duplicates:pd.DataFrame
    prices:pd.DataFrame
    price_forecast:pd.DataFrame
    pbev:pd.DataFrame
    crossval_predictions:pd.DataFrame
    price_backtest:pd.DataFrame
    models:dict=field(default_factory=dict)
    metadata:dict=field(default_factory=dict)


def run_engine(*,fleet_pdf=None,fleet_csv=None,pbev_csv=None,anp_monthly_csv=None,
               source_zip=None,external_validation_csv=None,cfg=None,output_dir=None,
               rebuild_references=False,fleet_dataframe=None,inference_only=False,
               reference_data=None):
    cfg=cfg or Settings()
    out=Path(output_dir) if output_dir else None
    if out:out.mkdir(parents=True,exist_ok=True)
    fleet,duplicates=(ingest_fleet_dataframe(fleet_dataframe) if fleet_dataframe is not None
                      else load_fleet(fleet_pdf,csv_path=fleet_csv))
    fleet=apply_quality(fleet,cfg)
    fleet=add_peer_benchmark(fleet,cfg)
    fleet=add_cost_analytics(fleet,cfg)
    if not inference_only:
        fleet=add_anomalies(fleet,cfg)
    else:
        fleet['anomaly_status_ml']='not_evaluated_inference_only'
        fleet['anomaly_score_ml']=np.nan
        fleet['anomaly_flag_ml']=False

    # ANP: importar cache pré-agrupado ou reprocessar o ZIP em chunks.
    price_source='unavailable';price_audit=pd.DataFrame()
    if reference_data is not None:
        monthly=reference_data[0].copy(deep=True)
        if not monthly.empty:price_source='processed_anp_monthly_csv'
    elif anp_monthly_csv and Path(anp_monthly_csv).is_file() and not rebuild_references:
        monthly=load_anp_monthly(anp_monthly_csv)
        price_source='processed_anp_monthly_csv'
    elif source_zip:
        monthly,price_audit=anp_monthly_from_zip(source_zip)
        price_source='ANP_from_zip_sample_mean_monthly'
        if out and not monthly.empty:
            monthly.to_csv(out/'anp_mensal_uf.csv',index=False,encoding='utf-8-sig')
    else:monthly=pd.DataFrame()

    # PBEV: preferir exportação existente; alternativa extrator histórico do ZIP.
    pbev_audit=pd.DataFrame();pbev_source='unavailable'
    if reference_data is not None:
        pbev=reference_data[1].copy(deep=True)
        if not pbev.empty:pbev_source='preprocessed_pbev_csv_needs_semantic_audit'
    elif pbev_csv and Path(pbev_csv).is_file() and not rebuild_references:
        pbev=load_pbev(pbev_csv);pbev_source='preprocessed_pbev_csv_needs_semantic_audit'
    elif source_zip:
        from .pbev_extract import extract_pbev_history
        catalog,pbev_audit=extract_pbev_history(source_zip)
        pbev=load_pbev_dataframe(catalog)
        pbev_source='historical_pbev_zip_heuristic_layout_extraction'
        if out and not catalog.empty:catalog.to_csv(out/'pbev_catalogo_extraido_auditar.csv',index=False,encoding='utf-8-sig')
    else:pbev=pd.DataFrame()
    fleet=match_pbev(fleet,pbev)
    fleet,prices=add_fuel_scenarios(fleet,monthly,cfg.price_uf)

    ext=None
    if external_validation_csv:
        from .ingest import read_fleet_csv
        ext,_=read_fleet_csv(external_validation_csv)
        ext=apply_quality(ext,cfg)
    if inference_only:
        efficiency_model=None
        metrics_eff={'status':'not_loaded','approved_production':False,
                     'reason':'no_trusted_independently_validated_artifact',
                     'automatic_recommendations':False}
        crossval=pd.DataFrame()
    else:
        efficiency_model,metrics_eff,crossval=fit_efficiency(fleet,cfg,ext)
    fleet=attach_efficiency_predictions(fleet,efficiency_model,metrics_eff,cfg)
    if inference_only:
        price_model=None
        metrics_price={'status':'not_loaded','approved_production':False,
                       'reason':'no_trusted_validated_artifact'}
        backtest=pd.DataFrame()
    else:
        price_model,metrics_price,backtest=fit_price_model(monthly,cfg)
    forecast=forecast_next_price(monthly,price_model,metrics_price)
    kpis,category=summarize(fleet)
    # Prioridade unificada: flags estatísticas e de qualidade, nunca ML exploratório.
    fleet['unified_priority_alert']=fleet['priority_alert']|fleet['ml_priority_alert']
    fleet['unified_priority_source']=np.select([fleet['priority_alert'],fleet['ml_priority_alert']],
                                               ['deterministic_peer','validated_ml'],default='none')
    fleet['unified_priority_liters_est']=np.where(fleet['priority_alert'],
            fleet['actionable_excess_liters_est'].fillna(0),np.where(fleet['ml_priority_alert'],
            fleet['ml_excess_liters_est'].clip(lower=0).fillna(0),np.nan))
    kpis['unified_priority_cases']=int(fleet.unified_priority_alert.sum())
    kpis['validated_ml_reference_vehicles']=int(fleet.active_reference_source.eq('ml_approved_external_validation').sum())
    metadata={'version':'fleet-force-unified-v1','analysis_scope':'annual_report_2025_for_UFMG',
              'currency':'BRL','distance_unit':'km','efficiency_unit':'km_per_liter',
              'period_aggregation':'UFMG_2025_annual_per_vehicle',
              'liters_source':'estimated_from_distance_and_reported_efficiency',
              'price_uf':cfg.price_uf,'price_source':price_source,'pbev_source':pbev_source,
              'expense_component_coverage':'unknown_if_missing_columns',
              'model_promotion_policy':'independent_holdout_required_for_operational_efficiency',
              'reference_priority':'same_model_peer_first_then_independently_validated_ml',
              'price_scenarios':'different_fuels_not_additive','settings':cfg.to_dict(),
              'input_hashes':{str(p):_sha256(p) for p in [fleet_pdf,fleet_csv,pbev_csv,anp_monthly_csv,source_zip,external_validation_csv]
                              if p is not None and Path(p).is_file()}}
    result=EngineOutput(fleet=fleet,categories=category,kpis=kpis,
        training_metrics={'efficiency':metrics_eff,'price':metrics_price},
        quality_duplicates=duplicates,prices=prices,price_forecast=forecast,
        pbev=pbev,crossval_predictions=crossval,price_backtest=backtest,
        models={'efficiency':efficiency_model,'price':price_model},metadata=metadata)
    if out:export_results(result,out,monthly,pbev_audit,price_audit)
    return result


def load_pbev_dataframe(catalog):
    """Reutiliza adaptador de CSV para aplicar a mesma validação nos dois caminhos de entrada."""
    if catalog is None or catalog.empty:return pd.DataFrame()
    from .utils import norm
    x=catalog.dropna(subset=['kmpl_cidade']).copy()
    x['kmpl_cidade']=pd.to_numeric(x.kmpl_cidade,errors='coerce')
    x=x.loc[x.kmpl_cidade.between(1,40)]
    return pd.DataFrame({'model_norm':x.modelo.map(norm),'kmpl_reference':x.kmpl_cidade,
                         'fuel':x.combustivel.map(norm),'reference_year':x.ano_referencia,
                         'extraction_confidence':x.confianca_extracao}).drop_duplicates()


def export_results(result,out,monthly=None,pbev_audit=None,price_audit=None):
    import joblib
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
    frames['fleet_scenarios.csv']=pd.DataFrame(fuel_scenario_records(result),columns=[
        'combustivel','uf','mes_referencia','preco_medio_referencia_brl_l','litros_excedentes_est','custo_cenario_brl','metodologia'])
    coverage=pd.DataFrame([{'campo':c,'disponiveis':int(result.fleet[c].notna().sum()),
                           'veiculos':len(result.fleet),
                           'cobertura_pct':float(result.fleet[c].notna().mean()*100)}
                           for c in result.fleet.columns if c in [
                               'km','kmpl','liters_est','total_expense','fuel_expense','os_count','parts','tires',
                               'maintenance_expense_known','peer_median_kmpl','pbev_min_kmpl','ml_predicted_kmpl']])
    frames['fleet_data_coverage.csv']=coverage
    for name,df in frames.items():df.to_csv(out/name,index=False,encoding='utf-8-sig')
    (out/'fleet_kpis.json').write_text(json.dumps(_clean(result.kpis),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (out/'fleet_model_metrics.json').write_text(json.dumps(_clean(result.training_metrics),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (out/'fleet_manifest.json').write_text(json.dumps(_clean(result.metadata),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    models_path=out/'models';models_path.mkdir(exist_ok=True)
    for kind,model in result.models.items():
        if model is not None:joblib.dump(model,models_path/f'{kind}_candidate.joblib')
    # Contrato útil para API/Angular, representando valores nulos como null.
    preview=result.fleet.head(100).replace([np.inf,-np.inf],np.nan)
    (out/'fleet_veiculos_preview.json').write_text(json.dumps(_clean(preview.where(pd.notna(preview),None).to_dict(orient='records')),
            ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def fuel_scenario_records(result):
    """Historical, alternative fuel scenarios shared by exports and HTTP serialization."""
    scenarios=[]
    liters=result.kpis['peer_priority_excess_liters_est']
    for row in result.prices.itertuples(index=False):
        if row.produto in ['GASOLINA','ETANOL','DIESEL S10']:
            scenarios.append({'combustivel':row.produto,'uf':row.uf,'mes_referencia':row.mes,
                              'preco_medio_referencia_brl_l':row.preco_medio_brl_l,
                              'litros_excedentes_est':liters,
                              'custo_cenario_brl':liters*row.preco_medio_brl_l,
                              'metodologia':'alternatives_not_additive_estimate_not_savings'})
    return scenarios

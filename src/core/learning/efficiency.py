"""Treinamento supervisionado, backtests e gates explícitos de promoção do ML.

Nenhuma métrica de validação cruzada no mesmo relatório anual promove consumo a produção.
Preço usa observações anteriores e mede ganho versus persistência (último preço).
"""
from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
from src.core.policy import Settings


def _features_efficiency(frame: pd.DataFrame) -> pd.DataFrame:
    x=frame[['model_norm','vehicle_category','km']].copy()
    x['model_norm']=x.model_norm.fillna('').astype(str)
    x['vehicle_category']=x.vehicle_category.fillna('desconhecido').astype(str)
    x['log_km']=np.log1p(pd.to_numeric(x.km,errors='coerce').clip(lower=0))
    return x[['model_norm','vehicle_category','log_km']]


def _efficiency_pipeline() -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import OneHotEncoder,StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    transform=ColumnTransformer([
        ('model_tokens',TfidfVectorizer(analyzer='char',ngram_range=(2,4),min_df=2,max_features=500),'model_norm'),
        ('category',OneHotEncoder(handle_unknown='ignore'),['vehicle_category']),
        ('distance',StandardScaler(with_mean=False),['log_km'])
    ],remainder='drop')
    return Pipeline([('features',transform),('regressor',Ridge(alpha=10.0))])


def _baseline_prediction(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    med=train.groupby('vehicle_category')['kmpl'].median()
    overall=float(train.kmpl.median())
    return test.vehicle_category.map(med).fillna(overall).to_numpy(dtype=float)


def _segment_metrics(y: Any, pred: Any, baseline: Any, categories: Any, cfg: Settings) -> list[dict[str, Any]]:
    from sklearn.metrics import mean_absolute_error
    rows=[]
    for category in sorted(pd.Series(categories).unique()):
        m=np.asarray(categories)==category
        if m.sum()<cfg.ml_min_segment_samples:continue
        err=mean_absolute_error(np.asarray(y)[m],np.asarray(pred)[m]);base=mean_absolute_error(np.asarray(y)[m],np.asarray(baseline)[m])
        rows.append({'category':str(category),'n':int(m.sum()),'mae_ml':float(err),
                     'mae_baseline':float(base),'relative_gain':float((base-err)/base) if base>0 else 0.})
    return rows


def fit_efficiency(train_fleet: pd.DataFrame, cfg: Settings, external_fleet: pd.DataFrame | None = None) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    """Validação GroupKFold por modelo. Holdout EXTERNO de outro período para promoção.

    O relatório UFMG não mede litros diretamente: km/L é informado e litros são derivados.
    É inadequado inferir economia comprovada apenas com regressão treinada nesse relatório.
    """
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import GroupKFold
    eligible=train_fleet.loc[train_fleet.valid_efficiency & ~train_fleet.model_identification_required].copy()
    eligible=eligible.loc[eligible.kmpl.notna() & eligible.km.gt(0)]
    groups=eligible.model_norm.fillna('')
    info={'status':'not_trained','n_train':int(len(eligible)),'distinct_models':int(groups.nunique()),
          'approved_production':False,'reason':'insufficient_data','validation':'grouped_by_vehicle_model_cv',
          'trained_on':'UFMG_2025_reported_kmpl_not_independent_liters',
          'automatic_recommendations':False}
    if len(eligible)<cfg.ml_min_train_rows or groups.nunique()<cfg.ml_min_model_families:
        return None,info,pd.DataFrame()
    X=_features_efficiency(eligible)
    y=eligible.kmpl.to_numpy(dtype=float)
    n_splits=min(5,groups.nunique())
    oof=np.full(len(eligible),np.nan);bpred=np.full(len(eligible),np.nan)
    for ix_train,ix_test in GroupKFold(n_splits=n_splits).split(X,y,groups=groups):
        tr=eligible.iloc[ix_train];te=eligible.iloc[ix_test]
        candidate=_efficiency_pipeline()
        candidate.fit(_features_efficiency(tr),tr.kmpl)
        oof[ix_test]=candidate.predict(_features_efficiency(te))
        bpred[ix_test]=_baseline_prediction(tr,te)
    mae=float(mean_absolute_error(y,oof));baseline_mae=float(mean_absolute_error(y,bpred))
    gain=(baseline_mae-mae)/baseline_mae if baseline_mae>0 else 0.0
    segments=_segment_metrics(y,oof,bpred,eligible.vehicle_category,cfg)
    # Gate de categorias: não ocultar degradação sistemática de caminhões/van.
    segments_pass=bool(segments) and all(s['relative_gain']>=-cfg.ml_max_category_regression for s in segments)
    cv_pass=bool(gain>=cfg.ml_min_gain and segments_pass)
    info.update({'status':'evaluated','reason':'no_independent_external_period',
                 'mae_ml_kmpl':mae,'mae_baseline_kmpl':baseline_mae,'relative_gain':float(gain),
                 'segment_metrics':segments,'group_cv_pass':cv_pass,'group_cv_folds':n_splits})
    pred_frame=eligible[['fleet_id','model','model_norm','vehicle_category','km','kmpl']].copy()
    pred_frame['ml_oof_kmpl']=oof
    pred_frame['baseline_oof_kmpl']=bpred
    pred_frame['ml_oof_overuse_pct']=np.where(y>0,oof/y-1,np.nan)
    pred_frame['prediction_scope']='out_of_fold_exploratory_not_production'
    final=_efficiency_pipeline()
    final.fit(X,eligible.kmpl)
    info['trained_fleet_ids']=eligible.fleet_id.astype(str).tolist()
    if external_fleet is not None and len(external_fleet):
        ext=external_fleet.copy()
        if 'reference_period' not in ext or not ext['reference_period'].astype(str).str[:4].ne('2025').all():
            info['reason']='external_holdout_must_have_distinct_reference_period_from_2025'
        else:
            ext=ext.loc[ext.valid_efficiency & ~ext.model_identification_required].copy()
            if len(ext)>=cfg.ml_min_external_rows:
                p=final.predict(_features_efficiency(ext));baseline=_baseline_prediction(eligible,ext)
                e=float(mean_absolute_error(ext.kmpl,p));b=float(mean_absolute_error(ext.kmpl,baseline))
                g=(b-e)/b if b>0 else 0.0
                ss=_segment_metrics(ext.kmpl,p,baseline,ext.vehicle_category,cfg)
                external_pass=bool(g>=cfg.ml_min_gain and ss and all(s['relative_gain']>=-cfg.ml_max_category_regression for s in ss))
                info['external_holdout']={'n':len(ext),'mae_ml':e,'mae_baseline':b,'relative_gain':g,'segment_metrics':ss}
                info['approved_production']=bool(cv_pass and external_pass)
                info['reason']='validated_on_independent_period' if info['approved_production'] else 'external_holdout_not_better_than_baseline'
            else:info['reason']='external_holdout_too_small'
    if not cfg.ml_require_external_holdout and cv_pass:
        # Opt-in explícito: mesmo assim chamar de provisório, pois só há CV.
        info['approved_production']=True
        info['reason']='group_cv_only_opt_in_not_independent_validation'
        info['validation_scope']='provisional_without_external_period'
    return final,info,pred_frame


def attach_efficiency_predictions(fleet: pd.DataFrame, model: Any, metrics: dict[str, Any], cfg: Settings, training_source: str = 'UFMG_PDF_2025') -> pd.DataFrame:
    f=fleet.copy()
    f['ml_predicted_kmpl']=np.nan
    f['ml_overuse_pct']=np.nan
    f['ml_excess_liters_est']=np.nan
    f['ml_priority_alert']=False
    f['ml_inference_status']='disabled_not_validated'
    f['active_reference_kmpl']=f['peer_median_kmpl'].where(f.peer_confidence.eq('moderate'))
    f['active_reference_source']=np.where(f.peer_confidence.eq('moderate'),'deterministic_same_model_peer','none')
    if model is None or not metrics.get('approved_production'):return f
    # Não realizar inferência em registros que foram usados para treinar este artefato.
    ids=set(metrics.get('trained_fleet_ids',[]))
    m=(f.valid_efficiency & ~f.fleet_id.astype(str).isin(ids) & ~f.model_identification_required)
    if not m.any():
        f.loc[f.valid_efficiency,'ml_inference_status']='withheld_training_vehicle'
        return f
    raw=model.predict(_features_efficiency(f.loc[m]))
    # Se não há referência confiável, não extrapolar absurdos.
    m_ix=f.loc[m].index
    bounded=(np.isfinite(raw) & (raw>0))
    f.loc[m_ix[bounded],'ml_predicted_kmpl']=raw[bounded]
    f.loc[m_ix[bounded],'ml_inference_status']='approved_external_validation'
    f.loc[m_ix[~bounded],'ml_inference_status']='invalid_prediction'
    m=f['ml_predicted_kmpl'].gt(0)
    f.loc[m,'ml_overuse_pct']=f.loc[m,'ml_predicted_kmpl']/f.loc[m,'kmpl']-1
    f.loc[m,'ml_excess_liters_est']=f.loc[m,'km_covered']/f.loc[m,'kmpl']-f.loc[m,'km_covered']/f.loc[m,'ml_predicted_kmpl']
    can_use=m & f['active_reference_source'].eq('none')
    f.loc[can_use,'active_reference_kmpl']=f.loc[can_use,'ml_predicted_kmpl']
    f.loc[can_use,'active_reference_source']='ml_approved_external_validation'
    f['ml_priority_alert']=(can_use & f['ml_overuse_pct'].ge(cfg.moderate_overuse) &
                            f['ml_excess_liters_est'].ge(cfg.min_priority_liters))
    return f



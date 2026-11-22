"""Temporal price training, approval gates and one-step forecasting."""
from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
from src.core.policy import Settings

def _price_features(monthly: pd.DataFrame) -> pd.DataFrame:
    m=monthly.copy()
    m['period']=pd.to_datetime(m['mes'].astype(str)+'-01',errors='coerce')
    m=m.dropna(subset=['period']).sort_values(['uf','produto','period'])
    out=[]
    for (uf,prod),part in m.groupby(['uf','produto']):
        part=part.drop_duplicates('period',keep='last').set_index('period').sort_index()
        # Cria lacunas explícitas: shift nunca atravessa meses ausentes.
        part=part.reindex(pd.date_range(part.index.min(),part.index.max(),freq='MS'))
        part['uf']=uf;part['produto']=prod;part['period']=part.index
        y=part['preco_medio_brl_l']
        for lag in (1,2,3,6,12):part[f'lag{lag}']=y.shift(lag)
        part['rolling3']=y.shift(1).rolling(3,min_periods=3).mean()
        part['rolling6']=y.shift(1).rolling(6,min_periods=6).mean()
        month=part.index.month
        part['month_sin']=np.sin(2*np.pi*month/12)
        part['month_cos']=np.cos(2*np.pi*month/12)
        out.append(part.reset_index(drop=True))
    features=pd.concat(out,ignore_index=True) if out else pd.DataFrame()
    if features.empty:return features
    return features.dropna(subset=['preco_medio_brl_l','lag1','lag2','lag3','lag6','lag12','rolling3','rolling6']).copy()

PRICE_NUMERIC=['lag1','lag2','lag3','lag6','lag12','rolling3','rolling6','month_sin','month_cos']
PRICE_CATEGORICAL=['uf','produto']

def _price_pipeline() -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    enc=ColumnTransformer([('categories',OneHotEncoder(handle_unknown='ignore',sparse_output=False),PRICE_CATEGORICAL),
                           ('numeric','passthrough',PRICE_NUMERIC)],sparse_threshold=0)
    return Pipeline([('features',enc),('regressor',HistGradientBoostingRegressor(
        max_iter=100,max_leaf_nodes=15,l2_regularization=2.,learning_rate=.06,random_state=42))])


def fit_price_model(monthly: pd.DataFrame, cfg: Settings) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    from sklearn.metrics import mean_absolute_error
    info={'status':'not_trained','approved_production':False,'reason':'insufficient_history',
          'horizon':'one_month_ahead','benchmark':'last_observed_price_lag1'}
    if monthly is None or monthly.empty:return None,info,pd.DataFrame()
    x=_price_features(monthly)
    if len(x)<600 or x.period.nunique()<24:return None,info,pd.DataFrame()
    months=sorted(x.period.unique())
    # Dois blocos cronológicos de 6 meses: nunca treinar no futuro para prever o passado.
    windows=[months[-12:-6],months[-6:]]
    results=[];pred_parts=[]
    for win in windows:
        if len(win)!=6:continue
        cutoff=min(win)
        tr=x[x.period<cutoff];te=x[x.period.isin(win)]
        if len(tr)<500 or len(te)<100:continue
        candidate=_price_pipeline()
        candidate.fit(tr[PRICE_CATEGORICAL+PRICE_NUMERIC],tr['preco_medio_brl_l'])
        pred=np.maximum(.01,candidate.predict(te[PRICE_CATEGORICAL+PRICE_NUMERIC]))
        out=te[['period','uf','produto','preco_medio_brl_l','lag1']].copy()
        out['prediction_ml']=pred;out['window_start']=pd.Timestamp(cutoff).strftime('%Y-%m')
        pred_parts.append(out)
    if len(pred_parts)<2:
        info['reason']='insufficient_temporal_backtest_windows'
        return None,info,pd.DataFrame()
    pred=pd.concat(pred_parts,ignore_index=True)
    truth=pred.preco_medio_brl_l
    err=float(mean_absolute_error(truth,pred.prediction_ml))
    baseline=float(mean_absolute_error(truth,pred.lag1))
    gain=(baseline-err)/baseline if baseline>0 else 0.
    per_fuel=[]
    for prod,g in pred.groupby('produto'):
        e=float(mean_absolute_error(g.preco_medio_brl_l,g.prediction_ml))
        b=float(mean_absolute_error(g.preco_medio_brl_l,g.lag1))
        per_fuel.append({'produto':prod,'n':len(g),'mae_ml':e,'mae_baseline_lag1':b,
                         'relative_gain':(b-e)/b if b>0 else 0.})
    approved=bool(gain>=cfg.ml_min_gain and all(
        seg['relative_gain']>=-cfg.ml_max_category_regression for seg in per_fuel if seg['n']>=40))
    info.update({'status':'evaluated','approved_production':approved,
                 'reason':'two_time_windows_beat_persistence' if approved else 'baseline_persistence_not_beaten_safely',
                 'n_backtest':len(pred),'backtest_windows':2,'mae_ml_brl_l':err,
                 'mae_baseline_brl_l':baseline,'relative_gain':gain,'segment_metrics':per_fuel,
                 'training_end_month':pd.Timestamp(x.period.max()).strftime('%Y-%m')})
    final=_price_pipeline()
    final.fit(x[PRICE_CATEGORICAL+PRICE_NUMERIC],x.preco_medio_brl_l)
    return final,info,pred


def forecast_next_price(monthly: pd.DataFrame, model: Any, metrics: dict[str, Any]) -> pd.DataFrame:
    """Somente próximo mês de cada série, não estimativa de cotação corrente ou vários meses."""
    if model is None or not metrics.get('approved_production') or monthly is None or monthly.empty:
        return pd.DataFrame(columns=['uf','produto','forecast_month','prediction_brl_l','status'])
    m=monthly.copy();m['period']=pd.to_datetime(m['mes']+'-01')
    out=[]
    for (uf,prod),g in m.groupby(['uf','produto']):
        g=g.sort_values('period').drop_duplicates('period',keep='last')
        t=g.period.max();future=t+pd.offsets.MonthBegin(1)
        series=g.set_index('period')['preco_medio_brl_l'].reindex(pd.date_range(t-pd.DateOffset(months=11),t,freq='MS'))
        if series.isna().any():continue
        arr=series.to_numpy(float)
        row={'uf':uf,'produto':prod,'lag1':arr[-1],'lag2':arr[-2],
             'lag3':arr[-3],'lag6':arr[-6],'lag12':arr[-12],
             'rolling3':np.mean(arr[-3:]),'rolling6':np.mean(arr[-6:]),
             'month_sin':np.sin(2*np.pi*future.month/12),'month_cos':np.cos(2*np.pi*future.month/12)}
        p=float(model.predict(pd.DataFrame([row])[PRICE_CATEGORICAL+PRICE_NUMERIC])[0])
        out.append({'uf':uf,'produto':prod,'last_observed_month':t.strftime('%Y-%m'),
                    'forecast_month':future.strftime('%Y-%m'),'prediction_brl_l':round(max(.01,p),4),
                    'status':'validated_one_step_forecast_not_observed_price'})
    return pd.DataFrame(out)

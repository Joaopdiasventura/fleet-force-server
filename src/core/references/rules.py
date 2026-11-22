"""Pure public-reference normalization, matching and historical scenarios."""
from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
from .exceptions import ReferenceDataError
from src.shared.numbers import norm
from src.core.fleet.classification import generic_model

FUEL_PRODUCTS={'GASOLINA':'gasolina','ETANOL':'etanol','DIESEL S10':'diesel_s10'}


def normalize_pbev(data: pd.DataFrame) -> pd.DataFrame:
    out=[]
    if {'model','gas_combined','ethanol_combined'}.issubset(data.columns):
        for r in data.itertuples(index=False):
            if norm(getattr(r,'propulsion','')) in {'ELETRICO','ELETRICO PURO'}:continue
            for fuel,col in [('GASOLINA_OU_DIESEL','gas_combined'),('ETANOL','ethanol_combined')]:
                v=pd.to_numeric(getattr(r,col),errors='coerce')
                if pd.notna(v) and 0<v<40:
                    out.append({'model_norm':norm(r.model),'kmpl_reference':float(v),
                                'fuel':fuel,'reference_year':2026,'extraction_confidence':'layout_2026_needs_semantic_review'})
    elif {'modelo','kmpl_cidade','combustivel'}.issubset(data.columns):
        for r in data.itertuples(index=False):
            v=pd.to_numeric(getattr(r,'kmpl_cidade'),errors='coerce')
            if pd.notna(v) and 0<v<40:
                out.append({'model_norm':norm(r.modelo),'kmpl_reference':float(v),
                            'fuel':norm(r.combustivel),'reference_year':getattr(r,'ano_referencia',None),
                            'extraction_confidence':getattr(r,'confianca_extracao','heuristic_extraction_unreviewed')})
    else: raise ReferenceDataError('PBEV exige CSV extraído por main.ipynb ou pelo notebook de histórico PBEV.')
    ref=pd.DataFrame(out)
    if ref.empty:return ref
    ref=ref.drop_duplicates(subset=['model_norm','kmpl_reference','fuel','reference_year'])
    return ref


def normalize_anp_monthly(data: pd.DataFrame) -> pd.DataFrame:
    x=data.copy()
    required={'mes','uf','produto','preco_medio_brl_l'}
    if not required.issubset(x):raise ReferenceDataError('Tabela ANP mensal sem campos '+str(sorted(required-set(x))))
    x['preco_medio_brl_l']=pd.to_numeric(x['preco_medio_brl_l'],errors='coerce')
    x['mes']=x['mes'].astype(str).str.slice(0,7)
    x['uf']=x['uf'].astype(str).str.upper()
    x['produto']=x['produto'].astype(str).str.upper()
    return x.loc[x['preco_medio_brl_l'].gt(0)].sort_values(['uf','produto','mes']).reset_index(drop=True)


def match_pbev(fleet: pd.DataFrame, pbev: pd.DataFrame) -> pd.DataFrame:
    f=fleet.copy()
    defaults={'pbev_match_method':'unavailable','pbev_min_kmpl':np.nan,
              'pbev_reference_year':np.nan,'pbev_excess_pct':np.nan,
              'pbev_excess_liters_est':np.nan,'pbev_comparison_confidence':'unavailable',
              'pbev_alert_exploratory':False}
    for c,v in defaults.items():f[c]=v
    if pbev is None or pbev.empty:return f
    names=set(pbev['model_norm'].dropna().unique())
    for ix,r in f.iterrows():
        if not r.valid_efficiency or generic_model(r.model_norm) or r.vehicle_category in {'caminhao','onibus','outros'}:continue
        if r.model_norm in names: match,method=r.model_norm,'exact_name_only'
        else:
            candidates=[s for s in names if len(s)>=3 and f' {s} ' in f' {r.model_norm} ']
            if not candidates:continue
            max_length=max(map(len,candidates))
            best=[s for s in candidates if len(s)==max_length]
            if len(best)!=1:continue
            match,method=best[0],'contained_whole_tokens'
        candidates=pbev.loc[pbev.model_norm.eq(match)]
        # Empregar apenas a edição mais recente disponível para o nome, evitando
        # misturar consumo homologado de 2010 com uma versão de 2026.
        if candidates.reference_year.notna().any():
            most_recent=candidates.reference_year.max()
            candidates=candidates.loc[candidates.reference_year.eq(most_recent)]
        if 'fuel' in f.columns and pd.notna(r.get('fuel',None)):
            fuel=norm(r['fuel']); matching=candidates.loc[candidates['fuel'].map(norm).eq(fuel)]
            if not matching.empty:candidates=matching
        # Preserve interpretação conservadora do main.ipynb: mínimo de variantes/fuels.
        minimum=float(candidates.kmpl_reference.min())
        f.at[ix,'pbev_match_method']=method
        f.at[ix,'pbev_min_kmpl']=minimum
        f.at[ix,'pbev_reference_year']=float(candidates.reference_year.max()) if candidates.reference_year.notna().any() else np.nan
        f.at[ix,'pbev_excess_pct']=minimum/r.kmpl-1
        f.at[ix,'pbev_excess_liters_est']=r.km_covered/r.kmpl-r.km_covered/minimum
        f.at[ix,'pbev_comparison_confidence']='exploratory_name_match_unreviewed_version_fuel'
        f.at[ix,'pbev_alert_exploratory']=bool(minimum/r.kmpl-1>.20)
    return f


def latest_prices(monthly: pd.DataFrame, uf: str = 'MG') -> pd.DataFrame:
    if monthly is None or monthly.empty:return pd.DataFrame(columns=['uf','produto','mes','preco_medio_brl_l'])
    subset=monthly.loc[monthly.uf.eq(uf.upper())&monthly.produto.isin(FUEL_PRODUCTS)]
    return subset.sort_values('mes').drop_duplicates(subset=['uf','produto'],keep='last').reset_index(drop=True)


def add_fuel_scenarios(fleet: pd.DataFrame, monthly: pd.DataFrame, uf: str = 'MG') -> tuple[pd.DataFrame, pd.DataFrame]:
    f=fleet.copy()
    prices=latest_prices(monthly,uf)
    for prod,suffix in FUEL_PRODUCTS.items():
        price=prices.loc[prices.produto.eq(prod)]
        value=float(price.iloc[0]['preco_medio_brl_l']) if len(price) else np.nan
        f[f'peer_cost_{suffix}_scenario_brl']=f['actionable_excess_liters_est']*value
        f[f'peer_cost_{suffix}_reference_month']=price.iloc[0]['mes'] if len(price) else None
    f['price_scenario_scope']='alternative_hypothetical_fuels_not_to_sum'
    return f,prices
def load_pbev_dataframe(catalog: pd.DataFrame) -> pd.DataFrame:
    """Reutiliza adaptador de CSV para aplicar a mesma validação nos dois caminhos de entrada."""
    if catalog is None or catalog.empty:return pd.DataFrame()
    x=catalog.dropna(subset=['kmpl_cidade']).copy()
    x['kmpl_cidade']=pd.to_numeric(x.kmpl_cidade,errors='coerce')
    x=x.loc[x.kmpl_cidade.between(1,40)]
    return pd.DataFrame({'model_norm':x.modelo.map(norm),'kmpl_reference':x.kmpl_cidade,
                         'fuel':x.combustivel.map(norm),'reference_year':x.ano_referencia,
                         'extraction_confidence':x.confianca_extracao}).drop_duplicates()


def fuel_scenario_records(kpis: dict[str, Any], prices: pd.DataFrame) -> list[dict[str, Any]]:
    """Historical, alternative fuel scenarios shared by exports and HTTP serialization."""
    scenarios=[]
    liters=kpis['peer_priority_excess_liters_est']
    for row in prices.itertuples(index=False):
        if row.produto in ['GASOLINA','ETANOL','DIESEL S10']:
            scenarios.append({'combustivel':row.produto,'uf':row.uf,'mes_referencia':row.mes,
                              'preco_medio_referencia_brl_l':row.preco_medio_brl_l,
                              'litros_excedentes_est':liters,
                              'custo_cenario_brl':liters*row.preco_medio_brl_l,
                              'metodologia':'alternatives_not_additive_estimate_not_savings'})
    return scenarios

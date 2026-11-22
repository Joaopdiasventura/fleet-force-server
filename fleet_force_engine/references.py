"""Referências brasileiras PBEV (exploratórias) e ANP (valores históricos observados)."""
from pathlib import Path
from zipfile import ZipFile
import re
import numpy as np
import pandas as pd
from .utils import norm,safe_div
from .config import EXPENSE_FIELDS

FUEL_PRODUCTS={'GASOLINA':'gasolina','ETANOL':'etanol','DIESEL S10':'diesel_s10'}


def load_pbev(path):
    if path is None:return pd.DataFrame()
    data=pd.read_csv(path,encoding='utf-8-sig',low_memory=False)
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
    else: raise ValueError('PBEV exige CSV extraído por main.ipynb ou pelo notebook de histórico PBEV.')
    ref=pd.DataFrame(out)
    if ref.empty:return ref
    ref=ref.drop_duplicates(subset=['model_norm','kmpl_reference','fuel','reference_year'])
    return ref


def match_pbev(fleet,pbev):
    f=fleet.copy()
    defaults={'pbev_match_method':'unavailable','pbev_min_kmpl':np.nan,
              'pbev_reference_year':np.nan,'pbev_excess_pct':np.nan,
              'pbev_excess_liters_est':np.nan,'pbev_comparison_confidence':'unavailable',
              'pbev_alert_exploratory':False}
    for c,v in defaults.items():f[c]=v
    if pbev is None or pbev.empty:return f
    from .utils import generic_model
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


def anp_monthly_from_zip(archive_path,chunk_size=180_000):
    """Média amostral por mês/UF/produto. Resistente a ZIP com arquivos semestrais grandes.
    Não equivale a preço transacionado pela frota nem a dados atuais automaticamente.
    """
    accum={}
    audit=[]
    with ZipFile(archive_path) as zf:
        names=[n for n in zf.namelist() if re.search(r'combustiveis/ca-\d{4}-\d{2}\.csv$',n,re.I)]
        for name in sorted(names):
            nrow,nvalid=0,0
            with zf.open(name) as fh:
                chunks=pd.read_csv(fh,sep=';',encoding='latin1',usecols=[
                    'Estado - Sigla','Produto','Data da Coleta','Valor de Venda','Unidade de Medida'
                ],dtype=str,chunksize=chunk_size,low_memory=False)
                for ch in chunks:
                    nrow+=len(ch)
                    price=pd.to_numeric(ch['Valor de Venda'].str.replace(',','.',regex=False),errors='coerce')
                    dates=pd.to_datetime(ch['Data da Coleta'],dayfirst=True,errors='coerce')
                    prod=ch['Produto'].astype(str).str.upper().str.strip()
                    unit=ch['Unidade de Medida'].astype(str).str.lower()
                    ok=(price.gt(0)&price.lt(30)&dates.notna()&prod.isin(FUEL_PRODUCTS)
                        &unit.str.contains('litro',regex=False,na=False))
                    s=pd.DataFrame({'mes':dates.loc[ok].dt.strftime('%Y-%m'),
                                    'uf':ch.loc[ok,'Estado - Sigla'].str.upper().str.strip(),
                                    'produto':prod.loc[ok], 'price':price.loc[ok]})
                    nvalid+=len(s)
                    if s.empty:continue
                    sums=s.groupby(['mes','uf','produto'],as_index=False).agg(total=('price','sum'),observations=('price','size'))
                    for rr in sums.itertuples(index=False):
                        key=(rr.mes,rr.uf,rr.produto)
                        prev_sum,prev_count=accum.get(key,(0.0,0))
                        accum[key]=(prev_sum+float(rr.total),prev_count+int(rr.observations))
            audit.append({'source_file':name,'rows_read':nrow,'rows_valid_liters':nvalid})
    monthly=pd.DataFrame([{'mes':k[0],'uf':k[1],'produto':k[2],
                           'preco_medio_brl_l':s/n,'n_postos_amostrados_linhas':n}
                          for k,(s,n) in accum.items()]).sort_values(['uf','produto','mes'])
    return monthly,pd.DataFrame(audit)


def load_anp_monthly(path):
    if path is None:return pd.DataFrame()
    x=pd.read_csv(path,encoding='utf-8-sig',low_memory=False)
    required={'mes','uf','produto','preco_medio_brl_l'}
    if not required.issubset(x):raise ValueError('Tabela ANP mensal sem campos '+str(sorted(required-set(x))))
    x['preco_medio_brl_l']=pd.to_numeric(x['preco_medio_brl_l'],errors='coerce')
    x['mes']=x['mes'].astype(str).str.slice(0,7)
    x['uf']=x['uf'].astype(str).str.upper()
    x['produto']=x['produto'].astype(str).str.upper()
    return x.loc[x['preco_medio_brl_l'].gt(0)].sort_values(['uf','produto','mes']).reset_index(drop=True)


def latest_prices(monthly,uf='MG'):
    if monthly is None or monthly.empty:return pd.DataFrame(columns=['uf','produto','mes','preco_medio_brl_l'])
    subset=monthly.loc[monthly.uf.eq(uf.upper())&monthly.produto.isin(FUEL_PRODUCTS)]
    return subset.sort_values('mes').drop_duplicates(subset=['uf','produto'],keep='last').reset_index(drop=True)


def add_fuel_scenarios(fleet,monthly,uf='MG'):
    f=fleet.copy()
    prices=latest_prices(monthly,uf)
    for prod,suffix in FUEL_PRODUCTS.items():
        price=prices.loc[prices.produto.eq(prod)]
        value=float(price.iloc[0]['preco_medio_brl_l']) if len(price) else np.nan
        f[f'peer_cost_{suffix}_scenario_brl']=f['actionable_excess_liters_est']*value
        f[f'peer_cost_{suffix}_reference_month']=price.iloc[0]['mes'] if len(price) else None
    f['price_scenario_scope']='alternative_hypothetical_fuels_not_to_sum'
    return f,prices

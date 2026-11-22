"""Processed CSV and historical ZIP reference I/O."""
from __future__ import annotations
from pathlib import Path

from zipfile import ZipFile
import re
import pandas as pd
from .rules import FUEL_PRODUCTS, normalize_pbev, normalize_anp_monthly

def load_pbev(path: str | Path | None) -> pd.DataFrame:
    if path is None:return pd.DataFrame()
    return normalize_pbev(pd.read_csv(path,encoding="utf-8-sig",low_memory=False))

def load_anp_monthly(path: str | Path | None) -> pd.DataFrame:
    if path is None:return pd.DataFrame()
    return normalize_anp_monthly(pd.read_csv(path,encoding="utf-8-sig",low_memory=False))

def anp_monthly_from_zip(archive_path: str | Path, chunk_size: int = 180_000) -> tuple[pd.DataFrame, pd.DataFrame]:
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



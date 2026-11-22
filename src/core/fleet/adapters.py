"""Filesystem CSV and PDF fleet source adapters."""
from __future__ import annotations
from pathlib import Path

import re
import pandas as pd
from .exceptions import FleetDataError
from .fields import FLEET_COLUMNS
from .service import consolidate_raw_ufmg, ingest_fleet_dataframe
from src.shared.numbers import norm, us_number

FLEET_BOUNDS = [35,90,180,210,260,310,360,410,455,495,545,605,655,725,780]


def parse_ufmg_pdf(pdf_path: str | Path) -> pd.DataFrame:
    import pdfplumber
    raw=[]
    with pdfplumber.open(pdf_path) as pdf:
        for page_num,page in enumerate(pdf.pages,1):
            words=page.extract_words(x_tolerance=1,y_tolerance=1.2,keep_blank_chars=False)
            anchors=sorted({round(w['top'],2) for w in words
                            if 40 <= w['x0'] < 90 and re.fullmatch(r'[A-Z0-9]{7}',w['text'])})
            for y in anchors:
                line=[w for w in words if abs(w['top']-y)<=.8 and 35<=w['x0']<780]
                buckets=[[] for _ in FLEET_COLUMNS]
                for w in line:
                    x=(w['x0']+w['x1'])/2
                    idx=next((i for i,(lo,hi) in enumerate(zip(FLEET_BOUNDS[:-1],FLEET_BOUNDS[1:])) if lo<=x<hi),None)
                    if idx is not None: buckets[idx].append(w)
                values=[' '.join(w['text'] for w in sorted(b,key=lambda w:w['x0'])).strip() for b in buckets]
                if re.fullmatch(r'[A-Z0-9]{7}',values[0]):
                    r=dict(zip(FLEET_COLUMNS,values)); r['source_page']=page_num;r['source_y']=y;raw.append(r)
    if not raw: raise FleetDataError('PDF UFMG sem registros reconhecidos. Layout incompatível ou arquivo incorreto.')
    df=pd.DataFrame(raw)
    for c in FLEET_COLUMNS[2:]:
        if c!='model': df[c]=df[c].map(us_number)
    df.insert(0,'source_row_id',range(1,len(df)+1))
    df['model_norm']=df['model'].map(norm)
    return df


def read_fleet_csv(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df=pd.read_csv(path,encoding='utf-8-sig',low_memory=False)
    return ingest_fleet_dataframe(df)


def load_fleet(pdf_path: str | Path | None = None, csv_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Prioriza o CSV gerado pelo projeto original, caso explícito.
    if csv_path:
        return read_fleet_csv(csv_path)
    if pdf_path:
        return consolidate_raw_ufmg(parse_ufmg_pdf(pdf_path))
    raise FleetDataError('Informe --fleet-csv ou --fleet-pdf.')

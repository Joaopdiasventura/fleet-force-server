"""Pure text and numerical normalization primitives."""
from __future__ import annotations
from typing import Iterable

import re
import unicodedata
import numpy as np
import pandas as pd

def norm(value: object) -> str:
    if value is None or pd.isna(value): return ''
    s=unicodedata.normalize('NFKD',str(value))
    s=''.join(c for c in s if not unicodedata.combining(c))
    s=s.upper().replace('&',' AND ')
    return re.sub(r'\s+',' ',re.sub(r'[^A-Z0-9]+',' ',s)).strip()

def us_number(value: object) -> float:
    if value is None: return np.nan
    try: return float(str(value).strip().replace(',',''))
    except (ValueError, TypeError): return np.nan

def br_number(value: object) -> float:
    if value is None or norm(value) in {'','ND','NA','N A','N D'}: return np.nan
    s=str(value).strip()
    if s in {'-','\\'}: return np.nan
    try: return float(s.replace('.','').replace(',','.'))
    except ValueError: return np.nan

def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.to_numeric(numerator,errors='coerce') / pd.to_numeric(denominator,errors='coerce').replace(0,np.nan)

def numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df: df[col]=pd.to_numeric(df[col],errors='coerce')
    return df

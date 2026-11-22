import re
import unicodedata
import numpy as np
import pandas as pd

def norm(value):
    if value is None or pd.isna(value): return ''
    s=unicodedata.normalize('NFKD',str(value))
    s=''.join(c for c in s if not unicodedata.combining(c))
    s=s.upper().replace('&',' AND ')
    return re.sub(r'\s+',' ',re.sub(r'[^A-Z0-9]+',' ',s)).strip()

def us_number(value):
    if value is None: return np.nan
    try: return float(str(value).strip().replace(',',''))
    except (ValueError, TypeError): return np.nan

def br_number(value):
    if value is None or norm(value) in {'','ND','NA','N A','N D'}: return np.nan
    s=str(value).strip()
    if s in {'-','\\'}: return np.nan
    try: return float(s.replace('.','').replace(',','.'))
    except ValueError: return np.nan

def safe_div(numerator, denominator):
    return pd.to_numeric(numerator,errors='coerce') / pd.to_numeric(denominator,errors='coerce').replace(0,np.nan)

def category_for_model(model):
    t=set(norm(model).split())
    if t.intersection({'CAMINHAO','TRUCK','BASCULANTE','CAVALO','CARRETA'}): return 'caminhao'
    if t.intersection({'ONIBUS','MICRO','MARCOPOLO','COMIL','RODOVIARIO','ATHENAS','ALLEGRO'}): return 'onibus'
    if t.intersection({'PICK','PIC','S10','AMAROK','L200','FRONTIER','RANGER','HILUX','SAVEIRO','STRADA'}):return 'picape'
    if t.intersection({'VAN','SPRINTER','DUCATO','TRANSIT','JUMPER','MASTER'}): return 'van'
    if t.intersection({'REBOQUE','TANQUE'}): return 'outros'
    return 'automovel' if norm(model) else 'desconhecido'

def generic_model(value):
    from .config import GENERIC_EXACT,GENERIC_TOKENS
    n=norm(value)
    return n in GENERIC_EXACT or bool(set(n.split()) & GENERIC_TOKENS)

def numeric(df, columns):
    for col in columns:
        if col in df: df[col]=pd.to_numeric(df[col],errors='coerce')
    return df

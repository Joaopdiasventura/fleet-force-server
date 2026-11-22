"""Extrator histórico PBEV oriundo do notebook UFMG enviado.
Nota: extração por heurísticas de layout; NÃO é catálogo auditado registro a registro.
"""
from __future__ import annotations
from typing import Any, Iterable

from pathlib import Path
from collections import defaultdict
from zipfile import ZipFile
import io
import numpy as np
import pandas as pd
import fitz
from src.shared.numbers import norm as norm_text

# Formato: x_marca, x_modelo, etanol_urbano, etanol_rodovia, gas_urbano, gas_rodovia, x_codigo_combustivel
PBEV_LAYOUTS={
 2009:(54,112,None,None,None,None,None),
 2010:(62,119,589,629,674,714,541),2011:(49,110,608,649,692,733,561),
 2012:(42,106,608,648,690,733,560),2013:(88,146,578,601,626,651,379),
 2014:(88,146,577,599,624,649,379),2015:(88,146,576,601,624,648,377),
 2016:(83,156,595,618,639,661,411),2017:(97,142,566,591,614,639,380),
 2018:(97,142,566,591,614,639,380),2019:(97,142,566,591,614,639,380),
 2020:(84,145,589,611,634,657,401),2021:(107,162,596,617,638,660,429),
 2022:(84,115,396,410,423,437,284),2023:(106,145,540,557,576,596,382),
 2024:(90,129,510,526,543,559,351),2025:(93,134,512,528,545,562,335),
 2026:(216,247,475,485,509,519,380),
}
PBEV_BRANDS=set('FIAT CHEVROLET RENAULT HYUNDAI HONDA TOYOTA FORD PEUGEOT CITROEN VOLKSWAGEN VW NISSAN MITSUBISHI KIA BMW AUDI MERCEDES BENZ BYD GWM CHERY CAOA SUZUKI SUBARU LEXUS PORSCHE DODGE RAM VOLVO MINI JAC LIFAN JEEP TROLLER LAND SMART CHANA EFFA JAGUAR CHANGAN IVECO RELY SSANGYONG HAFEI GEELY SEAT AGRALE BRILLIANCE MASERATI FERRARI LAMBORGHINI'.split())

def pdf_n(s: object) -> float:
    try:return float(str(s).replace(',','.'))
    except (ValueError,TypeError):return np.nan

def pdf_cell(ws: list[Any], x: float, tolerance: float = 5.9) -> float:
    candidates=[w for w in ws if abs(w[0]-x)<=tolerance]
    return pdf_n(min(candidates,key=lambda w:abs(w[0]-x))[4]) if candidates else np.nan

def pdf_txt(ws: list[Any], lo: float, hi: float) -> str:
    return ' '.join(w[4] for w in sorted(ws,key=lambda w:w[0]) if lo<=w[0]<hi).strip()

def read_pbev_year(zf: ZipFile, year: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pdfplumber
    layout=PBEV_LAYOUTS[year]
    bx,mx,ec,er,gc,gr,fx=layout
    source=next((n for n in zf.namelist() if n.endswith(f'/{year}.pdf') and 'PBEV' in n),None)
    if source is None:return [],{'ano':year,'status':'ausente','extraidos':0,'rejeitados':0}
    if ec is None:return [],{'ano':year,'status':'layout_nao_suportado','extraidos':0,'rejeitados':0}
    extracted=[];rejected=0
    content=zf.read(source)
    # Algumas edições (2010-2012/2021) estão rotacionadas na camada de texto do MuPDF.
    if year in [2010,2011,2012,2021]:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pages=[]
            for pg in pdf.pages:
                words=pg.extract_words(x_tolerance=.4,y_tolerance=.4)
                anchors=[w for w in words if abs(w['x0']-bx)<17 and norm_text(w['text']) in PBEV_BRANDS]
                page_rows=[]
                seen=[]
                for a in anchors:
                    if any(abs(a['top']-y)<1.4 for y in seen):continue
                    seen.append(a['top'])
                    line=[w for w in words if abs(w['top']-a['top'])<.8]
                    page_rows.append((pg.page_number,[(w['x0'],w['top'],w['x1'],w['bottom'],w['text']) for w in line]))
                pages.extend(page_rows)
    else:
        pdf=fitz.open(stream=content,filetype='pdf')
        pages=[]
        for pg in pdf:
            groups=defaultdict(list)
            for w in pg.get_text('words',sort=False):groups[w[5]].append(w)
            pages.extend((pg.number+1,ws) for ws in groups.values() if any(abs(w[0]-bx)<17 and norm_text(w[4]) in PBEV_BRANDS for w in ws))
        pdf.close()
    for page_no,ws in pages:
        brands=[w for w in ws if abs(w[0]-bx)<17 and norm_text(w[4]) in PBEV_BRANDS]
        if not brands:continue
        brand=min(brands,key=lambda w:abs(w[0]-bx))[4]
        model=pdf_txt(ws,mx-6,mx+48)
        # Deslocamentos variam entre edições; texto auxiliar pode incluir versão, motor e câmbio.
        description=pdf_txt(ws,mx+43,ec-10)
        fc_tokens=[w for w in ws if abs(w[0]-fx)<5.9]
        fuel_code=min(fc_tokens,key=lambda w:abs(w[0]-fx))[4].upper() if fc_tokens else ''
        values=[pdf_cell(ws,x) for x in (ec,er,gc,gr)]
        valid=[(np.isfinite(v) and 3<=v<=40) for v in values]
        if not model or not any(valid) or any(np.isfinite(v) and not (2<v<45) for v in values):
            rejected+=1;continue
        for fuel,city,highway in [('ETANOL',values[0],values[1]),
                                  ('DIESEL' if fuel_code=='D' else 'GASOLINA',values[2],values[3])]:
            if np.isfinite(city) and 3<=city<=40:
                extracted.append({'ano_referencia':year,'pagina':page_no,'marca':norm_text(brand),
                    'modelo':norm_text(model),'descricao':norm_text(description)[:120],
                    'codigo_combustivel':fuel_code,'combustivel':fuel,'kmpl_cidade':city,
                    'kmpl_estrada':highway if np.isfinite(highway) and 3<=highway<=45 else np.nan,
                    'origem':f'PBEV_{year}_PDF','confianca_extracao':'layout_heuristico_revisao_recomendada'})
    audit={'ano':year,'status':'heuristico_requer_auditoria','extraidos':len(extracted),
           'rejeitados':rejected,'pdf':source.split('/')[-1]}
    return extracted,audit



def extract_pbev_history(zip_path: str | Path, years: Iterable[int] = range(2009, 2027)) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows=[];audits=[]
    with ZipFile(zip_path) as zf:
        for year in years:
            records,audit=read_pbev_year(zf,year)
            rows.extend(records);audits.append(audit)
    out=pd.DataFrame(rows)
    if not out.empty:
        out=out.drop_duplicates(subset=['ano_referencia','marca','modelo','descricao','combustivel','kmpl_cidade','kmpl_estrada'])
    return out,pd.DataFrame(audits)

"""Strict upload CSV syntax adapter; consolidation remains in ingest.py."""
from __future__ import annotations

import csv
import io
import math
import re

import numpy as np
import pandas as pd

from .fields import FLEET_COLUMNS, EXPENSE_FIELDS
from .exceptions import CSVValidationError
from src.shared.numbers import norm


# Explicit report-to-engine mapping, independent of column order.
UFMG_HEADERS = {
    'Frota': 'fleet_id',
    'modelo': 'model',
    'Qde. OS': 'os_count',
    'Interna': 'labor_internal',
    'Externa': 'labor_external',
    'R$ Peças': 'parts',
    'R$ Pneus': 'tires',
    'R$ Abast.': 'fuel_expense',
    'R$ Lubri.': 'lubricant',
    'R$ Lavag.': 'wash',
    'Km Percorrido': 'km',
    'KM/L': 'kmpl_reported',
    'Total Despesas': 'total_expense',
    '$/km': 'cost_per_km',
}


def header_key(value: str) -> str:
    if re.fullmatch(r'\s*(?:R\s*)?\$\s*/\s*km\s*', value, re.I):
        return 'COSTPERKM'
    return norm(re.sub(r'(?i)r\s*\$', '', value)).replace(' ', '')


ALIASES = {header_key(k): v for k, v in UFMG_HEADERS.items()}
ALIASES.update({header_key(c): c for c in FLEET_COLUMNS})
ALIASES.update({'PECAS': 'parts', 'PNEUS': 'tires', 'ABAST': 'fuel_expense',
                'LUBRI': 'lubricant', 'LAVAG': 'wash', 'QDEOS': 'os_count',
                'QUANTIDADEOS': 'os_count', 'KMPERCORRIDOS': 'km'})
NUMERIC_COLUMNS = set(EXPENSE_FIELDS) | {
    'os_count', 'km', 'kmpl', 'kmpl_reported', 'total_expense', 'cost_per_km',
    'km_covered', 'km_uncovered', 'liters_est', 'row_count',
}


ALIASES.update({header_key(c): c for c in NUMERIC_COLUMNS})


def parse_number(value: str, column: str, row: int) -> float:
    s = value.strip()
    if s.lower() in {'', '-', 'na', 'n/a', 'nd', 'n/d', 'null', 'nan'}:
        return np.nan
    s = re.sub(r'^R\$\s*', '', s, flags=re.I)
    # Both conventions are supported only when grouping and decimal roles are clear.
    if ',' in s and '.' in s:
        if re.fullmatch(r'[+-]?\d{1,3}(?:\.\d{3})+,\d+', s):
            s = s.replace('.', '').replace(',', '.')
        elif re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+\.\d+', s):
            s = s.replace(',', '')
        else:
            s = ''
    elif ',' in s:
        # A single separator followed by three digits has two plausible meanings.
        if re.fullmatch(r'[+-]?[1-9]\d{0,2},\d{3}', s):
            s = ''
        elif re.fullmatch(r'[+-]?\d+,\d+', s):
            s = s.replace(',', '.')
        elif re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3}){2,}', s):
            s = s.replace(',', '')
        else:
            s = ''
    elif re.fullmatch(r'[+-]?[1-9]\d{0,2}\.\d{3}', s):
        s = ''  # Decimal versus thousands separator is ambiguous.
    elif re.fullmatch(r'[+-]?\d{1,3}(?:\.\d{3}){2,}', s):
        s = s.replace('.', '')
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', s):
        raise CSVValidationError('INVALID_NUMERIC_DATA', 'Invalid or ambiguous numerical value.',
                                 {'row': row, 'column': column})
    value = float(s)
    if not math.isfinite(value):
        raise CSVValidationError('INVALID_NUMERIC_DATA', 'Numerical value must be finite.',
                                 {'row': row, 'column': column})
    return value


def parse_upload_csv(content: bytes) -> tuple[pd.DataFrame, str]:
    if not content or not content.strip():
        raise CSVValidationError('EMPTY_FILE', 'The uploaded CSV is empty.')
    try:
        text = content.decode('utf-8-sig')
    except UnicodeDecodeError:
        raise CSVValidationError('INVALID_ENCODING', 'Use UTF-8 or UTF-8 with BOM.') from None
    if '\x00' in text:
        raise CSVValidationError('MALFORMED_CSV', 'CSV contains invalid null bytes.')
    # Header determines delimiter; numeric punctuation in data cannot affect detection.
    first = text.splitlines()[0] if text.splitlines() else ''
    delimiter = ';' if first.count(';') > first.count(',') else ','
    try:
        rows = list(csv.reader(io.StringIO(text, newline=''), delimiter=delimiter, strict=True))
    except csv.Error:
        raise CSVValidationError('MALFORMED_CSV', 'Malformed CSV quoting or structure.') from None
    rows = [r for r in rows if any(v.strip() for v in r)]
    if not rows:
        raise CSVValidationError('EMPTY_FILE', 'The uploaded CSV is empty.')
    original = rows[0]
    columns = [ALIASES.get(header_key(c), c.strip().lower()) for c in original]
    if len(set(columns)) != len(columns) or any(not c for c in columns):
        raise CSVValidationError('INVALID_CSV_SCHEMA', 'CSV headers must be unique and nonempty.')
    missing = sorted({'fleet_id', 'model', 'km'} - set(columns))
    if missing:
        raise CSVValidationError('INVALID_CSV_SCHEMA', 'The uploaded CSV is missing required columns.',
                                 {'missing_columns': missing, 'accepted_formats': ['ufmg', 'normalized']})
    if len(rows) == 1:
        raise CSVValidationError('EMPTY_FILE', 'The uploaded CSV has no vehicle records.')
    data = []
    for line, values in enumerate(rows[1:], 2):
        if len(values) != len(columns):
            raise CSVValidationError('MALFORMED_CSV', 'Each row must have the same number of fields as the header.', {'row': line})
        record = dict(zip(columns, values))
        for column in NUMERIC_COLUMNS & record.keys():
            record[column] = parse_number(record[column], column, line)
        for column in ('fleet_id', 'model'):
            record[column] = record[column].strip()
            if not record[column]:
                raise CSVValidationError('INVALID_CSV_DATA', 'Vehicle identifier and model cannot be empty.', {'row': line, 'column': column})
        record['fleet_id'] = record['fleet_id'].upper()
        data.append(record)
    input_format = 'ufmg' if any(header_key(c) == 'FROTA' for c in original) else 'normalized'
    return pd.DataFrame(data), input_format


class UploadCSVDecoder:
    def decode(self, content: bytes) -> tuple[pd.DataFrame, str]:
        return parse_upload_csv(content)

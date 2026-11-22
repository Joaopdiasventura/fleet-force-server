"""Request-local analysis with cached, read-only processed public references."""
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import pandas as pd

from . import Settings, run_engine
from .csv_adapter import CSVValidationError, parse_upload_csv
from .references import load_anp_monthly, load_pbev
from .serialization import serialize_analysis

BASE_DIR = Path(__file__).resolve().parents[1]
ANP_PATH = BASE_DIR / 'data/anp_precos_mensais_uf.csv'
PBEV_PATH = BASE_DIR / 'data/pbev_catalogo_historico_extraido_auditar.csv'


@lru_cache(maxsize=4)
def _reference(path, modified, size, kind):
    return (load_anp_monthly if kind == 'anp' else load_pbev)(path)


def references(path, kind):
    try:
        stat = path.stat()
    except OSError:
        return pd.DataFrame()
    try:
        return _reference(str(path), stat.st_mtime_ns, stat.st_size, kind)
    except (OSError, ValueError, UnicodeError):
        # Unusable processed references disable only their dependent analyses.
        return pd.DataFrame()


def analyze_csv(content, uf):
    frame, input_format = parse_upload_csv(content)
    # Match existing ingestion: repeated identifiers require a complete raw layout.
    from .config import FLEET_COLUMNS
    if frame.fleet_id.duplicated().any() and not set(FLEET_COLUMNS).issubset(frame):
        raise CSVValidationError('INVALID_CSV_DATA',
            'Repeated vehicle identifiers require complete raw report columns for consolidation.',
            {'required_for_consolidation': FLEET_COLUMNS})
    result = run_engine(fleet_dataframe=frame, cfg=Settings(price_uf=uf),
                        inference_only=True,
                        reference_data=(references(ANP_PATH, 'anp'), references(PBEV_PATH, 'pbev')))
    result.fleet['data_origin']='uploaded_csv'
    # Never assert that arbitrary client uploads cover the UFMG 2025 period.
    result.metadata.update(analysis_scope='uploaded_fleet', period_aggregation='client_supplied_unspecified_period')
    return serialize_analysis(result, str(uuid4()), input_format, uf)

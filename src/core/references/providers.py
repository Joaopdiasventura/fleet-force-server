"""Reference provider implementations; all filesystem caching lives here."""
from functools import lru_cache
from pathlib import Path
import logging

import pandas as pd

from .adapters import load_anp_monthly, load_pbev, anp_monthly_from_zip
from .models import ReferenceData
from .rules import load_pbev_dataframe

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _reference(path: str, modified: int, size: int, kind: str) -> pd.DataFrame:
    return (load_anp_monthly if kind == 'anp' else load_pbev)(path)


def cached_reference(path: Path, kind: str) -> pd.DataFrame:
    try:
        stat = path.stat()
    except OSError:
        return pd.DataFrame()
    try:
        return _reference(str(path), stat.st_mtime_ns, stat.st_size, kind).copy(deep=True)
    except (OSError, ValueError, UnicodeError):
        logger.warning('reference_unavailable', extra={'reference_kind': kind})
        return pd.DataFrame()


class InMemoryReferenceProvider:
    def __init__(self, monthly: pd.DataFrame, pbev: pd.DataFrame) -> None:
        self.monthly = monthly.copy(deep=True)
        self.pbev = pbev.copy(deep=True)

    def load(self) -> ReferenceData:
        return ReferenceData(
            monthly=self.monthly.copy(deep=True), pbev=self.pbev.copy(deep=True),
            price_source='processed_anp_monthly_csv' if not self.monthly.empty else 'unavailable',
            pbev_source='preprocessed_pbev_csv_needs_semantic_audit' if not self.pbev.empty else 'unavailable',
        )


class ProcessedReferenceProvider:
    """Upload provider: processed files only, never historical ZIP extraction."""
    def __init__(self, anp_path: Path, pbev_path: Path) -> None:
        self.anp_path = anp_path
        self.pbev_path = pbev_path

    def load(self) -> ReferenceData:
        return InMemoryReferenceProvider(
            cached_reference(self.anp_path, 'anp'), cached_reference(self.pbev_path, 'pbev')
        ).load()


class FileReferenceProvider:
    """CLI provider retaining explicit cache/ZIP rebuild selection and audits."""
    def __init__(self, anp_path: str | Path | None = None,
                 pbev_path: str | Path | None = None,
                 source_zip: str | Path | None = None, rebuild: bool = False) -> None:
        self.anp_path, self.pbev_path = anp_path, pbev_path
        self.source_zip, self.rebuild = source_zip, rebuild

    def load(self) -> ReferenceData:
        data = ReferenceData()
        if self.anp_path and Path(self.anp_path).is_file() and not self.rebuild:
            data.monthly = load_anp_monthly(self.anp_path)
            data.price_source = 'processed_anp_monthly_csv'
        elif self.source_zip:
            data.monthly, data.price_audit = anp_monthly_from_zip(self.source_zip)
            data.price_source = 'ANP_from_zip_sample_mean_monthly'
        if self.pbev_path and Path(self.pbev_path).is_file() and not self.rebuild:
            data.pbev = load_pbev(self.pbev_path)
            data.pbev_source = 'preprocessed_pbev_csv_needs_semantic_audit'
        elif self.source_zip:
            from .pbev_extract import extract_pbev_history
            data.raw_catalog, data.pbev_audit = extract_pbev_history(self.source_zip)
            data.pbev = load_pbev_dataframe(data.raw_catalog)
            data.pbev_source = 'historical_pbev_zip_heuristic_layout_extraction'
        return data

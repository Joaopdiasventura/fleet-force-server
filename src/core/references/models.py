"""Reference tables and provenance passed across the application boundary."""
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ReferenceData:
    monthly: pd.DataFrame = field(default_factory=pd.DataFrame)
    pbev: pd.DataFrame = field(default_factory=pd.DataFrame)
    price_source: str = 'unavailable'
    pbev_source: str = 'unavailable'
    price_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    pbev_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_catalog: pd.DataFrame = field(default_factory=pd.DataFrame)

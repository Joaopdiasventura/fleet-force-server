"""Standard JSON cleaning and DataFrame record serialization."""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime('%Y-%m-%d')
    if value is pd.NA or value is None:
        return None
    return value


def records(frame: pd.DataFrame, double_precision: int = 15) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(
        orient='records', force_ascii=False, double_precision=double_precision))

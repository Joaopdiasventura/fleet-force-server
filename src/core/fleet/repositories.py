"""Filesystem implementation of the exported-report repository."""
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .exceptions import ReportUnavailable


class FileReportRepository:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def read_json(self, name: str) -> dict[str, Any]:
        path = self.directory / name
        if not path.is_file():
            raise ReportUnavailable(f'Relatório {name} não gerado: execute run.py')
        return json.loads(path.read_text(encoding='utf-8'))

    def read_frame(self, name: str) -> pd.DataFrame:
        path = self.directory / name
        if not path.is_file():
            raise ReportUnavailable(f'Relatório {name} não gerado')
        return pd.read_csv(path, encoding='utf-8-sig').replace(
            [float('inf'), -float('inf')], float('nan'))

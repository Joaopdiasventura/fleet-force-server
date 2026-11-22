"""Runtime environment configuration; analytical policy lives in core.policy."""
from dataclasses import dataclass, field
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AppSettings:
    output_dir: Path = field(default_factory=lambda: BASE_DIR / 'outputs')
    anp_path: Path = field(default_factory=lambda: BASE_DIR / 'data/anp_precos_mensais_uf.csv')
    pbev_path: Path = field(default_factory=lambda: BASE_DIR / 'data/pbev_catalogo_historico_extraido_auditar.csv')
    max_upload_bytes: int = 10 * 1024 * 1024

    @classmethod
    def from_env(cls) -> 'AppSettings':
        return cls(output_dir=Path(os.getenv('FLEET_OUTPUT_DIR', str(BASE_DIR / 'outputs'))),
                   max_upload_bytes=int(os.getenv('FLEET_MAX_UPLOAD_BYTES', 10 * 1024 * 1024)))

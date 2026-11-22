"""Library/CLI composition facade; the in-memory pipeline owns no infrastructure."""
from pathlib import Path
import pandas as pd

from src.core.policy import Settings
from src.core.fleet.adapters import load_fleet, read_fleet_csv
from src.core.fleet.service import ingest_fleet_dataframe
from src.core.references.providers import FileReferenceProvider, InMemoryReferenceProvider
from src.core.analytics.artifacts import export_results
from src.core.analytics.models import EngineOutput
from src.core.analytics.service import AnalysisService
from src.shared.files import input_hashes


def run_engine(*, fleet_pdf: str | Path | None = None, fleet_csv: str | Path | None = None,
               pbev_csv: str | Path | None = None, anp_monthly_csv: str | Path | None = None,
               source_zip: str | Path | None = None, external_validation_csv: str | Path | None = None,
               cfg: Settings | None = None, output_dir: str | Path | None = None,
               rebuild_references: bool = False, fleet_dataframe: pd.DataFrame | None = None,
               inference_only: bool = False,
               reference_data: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> EngineOutput:
    cfg = cfg or Settings()
    out = Path(output_dir) if output_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    fleet, duplicates = (ingest_fleet_dataframe(fleet_dataframe) if fleet_dataframe is not None
                         else load_fleet(fleet_pdf, csv_path=fleet_csv))
    provider = (InMemoryReferenceProvider(*reference_data) if reference_data is not None
                else FileReferenceProvider(anp_monthly_csv, pbev_csv, source_zip, rebuild_references))
    references = provider.load()
    external = read_fleet_csv(external_validation_csv)[0] if external_validation_csv else None
    result = AnalysisService().execute(fleet, duplicates, references, cfg,
                                       external=external, inference_only=inference_only)
    result.metadata['input_hashes'] = input_hashes([
        fleet_pdf, fleet_csv, pbev_csv, anp_monthly_csv, source_zip, external_validation_csv])
    if out:
        if not references.raw_catalog.empty:
            references.raw_catalog.to_csv(out / 'pbev_catalogo_extraido_auditar.csv', index=False, encoding='utf-8-sig')
        export_results(result, out, references.monthly, references.pbev_audit, references.price_audit)
    return result

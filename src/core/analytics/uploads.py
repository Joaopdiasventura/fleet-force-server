"""Uploaded-fleet use case; infrastructure is supplied by composition."""
from typing import Any
from uuid import uuid4

from src.core.policy import Settings
from src.core.fleet.fields import FLEET_COLUMNS
from src.core.fleet.exceptions import CSVValidationError
from src.core.fleet.ports import FleetDecoder
from src.core.fleet.service import ingest_fleet_dataframe
from src.core.references.ports import ReferenceProvider
from .service import AnalysisService
from .schemas import serialize_analysis


class UploadAnalysisService:
    def __init__(self, decoder: FleetDecoder, references: ReferenceProvider,
                 analysis: AnalysisService) -> None:
        self.decoder = decoder
        self.references = references
        self.analysis = analysis

    def analyze(self, content: bytes, uf: str) -> dict[str, Any]:
        frame, input_format = self.decoder.decode(content)
        if frame.fleet_id.duplicated().any() and not set(FLEET_COLUMNS).issubset(frame):
            raise CSVValidationError('INVALID_CSV_DATA',
                'Repeated vehicle identifiers require complete raw report columns for consolidation.',
                {'required_for_consolidation': FLEET_COLUMNS})
        fleet, duplicates = ingest_fleet_dataframe(frame)
        result = self.analysis.execute(fleet, duplicates, self.references.load(),
                                       Settings(price_uf=uf), inference_only=True)
        result.fleet['data_origin'] = 'uploaded_csv'
        result.metadata.update(analysis_scope='uploaded_fleet',
                               period_aggregation='client_supplied_unspecified_period')
        return serialize_analysis(result, str(uuid4()), input_format, uf)

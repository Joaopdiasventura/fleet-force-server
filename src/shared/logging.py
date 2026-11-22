"""Standard-library structured logging without payloads or vehicle records."""
import json
import logging


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = {'level': record.levelname, 'logger': record.name, 'event': record.getMessage()}
        for name in ('error_code', 'reference_kind'):
            if hasattr(record, name):
                fields[name] = getattr(record, name)
        return json.dumps(fields)


def configure_logging() -> None:
    logger = logging.getLogger('src')
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

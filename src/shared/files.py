"""Filesystem provenance helpers used by entry-point composition only."""
import hashlib
from pathlib import Path
from collections.abc import Iterable


def input_hashes(paths: Iterable[str | Path | None]) -> dict[str, str]:
    hashes = {}
    for path in paths:
        if path is None or not Path(path).is_file():
            continue
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        hashes[str(path)] = digest.hexdigest()
    return hashes

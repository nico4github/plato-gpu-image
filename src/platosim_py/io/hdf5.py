"""HDF5 compatibility layer skeleton."""

from __future__ import annotations

from pathlib import Path


class HDF5Writer:
    """Placeholder writer preserving future legacy layout compatibility."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)

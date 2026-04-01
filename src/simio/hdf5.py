"""HDF5 compatibility layer helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import h5py


LEGACY_OUTPUT_GROUPS: tuple[str, ...] = (
    "/InputParameters",
    "/Images",
    "/BiasMaps",
    "/SmearingMaps",
    "/SubPixelImages",
    "/StarCatalog",
    "/StarPositions",
    "/ACS",
    "/Flatfield",
    "/ThroughputMaps",
)


class HDF5Writer:
    """Create and populate HDF5 outputs with a legacy-compatible layout."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)

    def initialize_file(self, *, overwrite: bool = False) -> None:
        """Create an empty HDF5 file.

        Args:
            overwrite: If False and file exists, raise FileExistsError.
        """
        if self.output_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {self.output_path}")
        with h5py.File(self.output_path, "w"):
            pass

    def ensure_group(self, group_path: str) -> None:
        """Create `group_path` if it does not exist."""
        with h5py.File(self.output_path, "a") as handle:
            handle.require_group(group_path)

    def ensure_legacy_groups(self, groups: Sequence[str] = LEGACY_OUTPUT_GROUPS) -> None:
        """Create a baseline set of legacy output groups."""
        with h5py.File(self.output_path, "a") as handle:
            for group in groups:
                handle.require_group(group)

    def write_root_metadata(self, attributes: Mapping[str, str | int | float]) -> None:
        """Write root-level HDF5 attributes."""
        with h5py.File(self.output_path, "a") as handle:
            for key, value in attributes.items():
                handle.attrs[key] = value

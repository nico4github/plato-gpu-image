from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from platosim_py.io.hdf5 import HDF5Writer, LEGACY_OUTPUT_GROUPS


def test_initialize_file_and_reject_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "out.hdf5"
    writer = HDF5Writer(output)
    writer.initialize_file()
    assert output.exists()

    with pytest.raises(FileExistsError):
        writer.initialize_file(overwrite=False)


def test_ensure_legacy_groups(tmp_path: Path) -> None:
    output = tmp_path / "legacy.hdf5"
    writer = HDF5Writer(output)
    writer.initialize_file()
    writer.ensure_legacy_groups()

    with h5py.File(output, "r") as handle:
        for group in LEGACY_OUTPUT_GROUPS:
            assert group in handle


def test_write_root_metadata(tmp_path: Path) -> None:
    output = tmp_path / "meta.hdf5"
    writer = HDF5Writer(output)
    writer.initialize_file()
    writer.write_root_metadata({"simulator": "plato-gpu-image", "version": 1})

    with h5py.File(output, "r") as handle:
        assert handle.attrs["simulator"] == "plato-gpu-image"
        assert int(handle.attrs["version"]) == 1


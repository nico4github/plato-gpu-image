from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def _script_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "scripts" / "preview_hdf5_side_by_side.py"


def _make_h5(path: Path, *, image_value: float) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_group("/Images")
        handle.create_group("/ACS")
        handle.create_dataset("/Images/image0000000", data=np.full((2, 2), image_value))
        handle.create_dataset("/ACS/time", data=np.array([0.0, 1.0]))


def test_preview_hdf5_side_by_side_script(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.h5"
    local = tmp_path / "local.h5"
    _make_h5(legacy, image_value=1.0)
    _make_h5(local, image_value=2.0)

    cmd = [
        sys.executable,
        str(_script_path()),
        "--legacy",
        str(legacy),
        "--local",
        str(local),
        "--image-index",
        "0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "== Top-level Groups ==" in result.stdout
    assert "== Key Dataset Preview ==" in result.stdout
    assert "/Images/image0000000" in result.stdout


from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py


def _script_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "scripts" / "compare_hdf5_structure.py"


def _make_h5(path: Path, with_extra: bool) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_group("/A")
        handle.create_dataset("/A/x", data=[1, 2, 3])
        if with_extra:
            handle.create_group("/B")


def test_compare_hdf5_structure_script(tmp_path: Path) -> None:
    lhs = tmp_path / "lhs.h5"
    rhs = tmp_path / "rhs.h5"
    _make_h5(lhs, with_extra=True)
    _make_h5(rhs, with_extra=False)

    cmd = [sys.executable, str(_script_path()), str(lhs), str(rhs)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "common_paths=" in result.stdout
    assert "only_lhs=" in result.stdout
    assert "only_rhs=" in result.stdout


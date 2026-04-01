from __future__ import annotations

import os
import subprocess
from pathlib import Path

import h5py
import pytest
import yaml

from platosim_py.core.simulation import Simulation
from platosim_py.io.hdf5 import LEGACY_OUTPUT_GROUPS


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _legacy_root() -> Path:
    return _repo_root().parent / "platosim_develop" / "PlatoSim3"


def _legacy_binary() -> Path:
    return _legacy_root() / "build" / "platosim"


@pytest.mark.skipif(
    os.environ.get("RUN_LEGACY_PARITY") != "1",
    reason="Set RUN_LEGACY_PARITY=1 to enable parity integration tests",
)
def test_hdf5_structure_baseline_parity(tmp_path: Path) -> None:
    legacy_bin = _legacy_binary()
    if not legacy_bin.exists():
        pytest.skip(f"Legacy binary not found: {legacy_bin}")

    source_yaml = _legacy_root() / "inputfiles" / "inputfile.yaml"
    if not source_yaml.exists():
        pytest.skip(f"Legacy input file not found: {source_yaml}")

    config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    config["ObservingParameters"]["NumExposures"] = 1

    config_yaml = tmp_path / "run.yaml"
    legacy_output = tmp_path / "legacy.hdf5"
    legacy_log = tmp_path / "legacy.log"
    py_output = tmp_path / "py.hdf5"
    config_yaml.write_text(yaml.safe_dump(config), encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("PLATO_PROJECT_HOME", str(_legacy_root()))
    result = subprocess.run(
        [str(legacy_bin), str(config_yaml), str(legacy_output), str(legacy_log), "1"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    sim = Simulation.from_legacy_yaml(config_yaml, output_path=py_output)
    sim.run()

    with h5py.File(legacy_output, "r") as legacy_h5, h5py.File(py_output, "r") as py_h5:
        groups_present_in_legacy = [group for group in LEGACY_OUTPUT_GROUPS if group in legacy_h5]
        assert groups_present_in_legacy, "No baseline compatibility groups detected in legacy output"

        for group in groups_present_in_legacy:
            # At this stage we enforce parity only against the overlap of our
            # declared baseline groups and what legacy run produced for this config.
            assert group in py_h5

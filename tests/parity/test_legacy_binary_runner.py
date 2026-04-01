from __future__ import annotations

import os
import subprocess
from pathlib import Path

import h5py
import pytest
import yaml

from tests.output_paths import tagged_output_file


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _legacy_root() -> Path:
    return _repo_root().parent / "platosim_develop" / "PlatoSim3"


def _legacy_binary() -> Path:
    return _legacy_root() / "build" / "platosim"


@pytest.mark.skipif(
    os.environ.get("RUN_LEGACY_PARITY") != "1",
    reason="Set RUN_LEGACY_PARITY=1 to enable slow integration parity runner tests",
)
def test_legacy_binary_can_generate_output(tmp_path: Path) -> None:
    legacy_bin = _legacy_binary()
    if not legacy_bin.exists():
        pytest.skip(f"Legacy binary not found: {legacy_bin}")

    legacy_root = _legacy_root()
    source_yaml = legacy_root / "inputfiles" / "inputfile.yaml"
    if not source_yaml.exists():
        pytest.skip(f"Legacy input file not found: {source_yaml}")

    config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    config["ObservingParameters"]["NumExposures"] = 1

    run_yaml = tmp_path / "legacy_run.yaml"
    run_output = tagged_output_file("legacy", "binary_runner.hdf5")
    run_log = tagged_output_file("legacy", "binary_runner.log")
    if run_output.exists():
        run_output.unlink()
    if run_log.exists():
        run_log.unlink()
    run_yaml.write_text(yaml.safe_dump(config), encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("PLATO_PROJECT_HOME", str(legacy_root))
    cmd = [str(legacy_bin), str(run_yaml), str(run_output), str(run_log), "1"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 0, result.stderr or result.stdout

    assert run_output.exists()
    with h5py.File(run_output, "r") as handle:
        assert "InputParameters" in handle


def test_legacy_binary_version_available_if_present() -> None:
    legacy_bin = _legacy_binary()
    if not legacy_bin.exists():
        pytest.skip(f"Legacy binary not found: {legacy_bin}")

    result = subprocess.run(
        [str(legacy_bin), "-version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "PlatoSim" in result.stdout

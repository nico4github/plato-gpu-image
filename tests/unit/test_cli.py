from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cli_path() -> Path:
    return _repo_root() / "plato-gpu-image.py"


def _write_minimal_valid_config(path: Path) -> None:
    path.write_text(
        "General:\n  ProjectLocation: ENV['PLATO_PROJECT_HOME']\n"
        "ObservingParameters:\n"
        "  NumExposures: 10\n"
        "  BeginExposureNr: 0\n"
        "  CycleTime: 25\n"
        "  StarCatalogFile: inputfiles/starcatalog.txt\n"
        "Sky:\n  SkyBackground:\n    UseConstantSkyBackground: true\n"
        "Platform:\n"
        "  UseJitter: true\n"
        "  JitterSource: FromRedNoise\n"
        "  Orientation:\n    Source: Angles\n"
        "Telescope:\n  GroupID: Custom\n  UseDrift: false\n"
        "Camera:\n  PlateScale: 0.8333\n  IncludeFieldDistortion: true\n"
        "PSF:\n  Model: AnalyticNonGaussian\n"
        "FEE:\n  Temperature: Nominal\n"
        "CCD:\n"
        "  Position: Custom\n"
        "  NumRows: 4510\n"
        "  NumColumns: 4510\n"
        "SubField:\n  NumRows: 100\n  NumColumns: 100\n"
        "RandomSeeds:\n  ReadOutNoiseSeed: 1\n"
        "ControlHDF5Content:\n  WritePixelMaps: true\n",
        encoding="utf-8",
    )


def test_cli_runs_simulation_from_input_yaml(tmp_path: Path) -> None:
    config = tmp_path / "input.yaml"
    output = tmp_path / "run.hdf5"
    _write_minimal_valid_config(config)

    cmd = [
        sys.executable,
        str(_cli_path()),
        "--input",
        str(config),
        "--output",
        str(output),
        "--backend",
        "numpy",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert output.exists()
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["backend"] == "numpy"
    assert payload["output_file"] == str(output)


from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _script_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "scripts" / "validate_legacy_config.py"


def test_validate_script_valid_config(tmp_path: Path) -> None:
    config_file = tmp_path / "valid.yaml"
    config_file.write_text(
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

    cmd = [sys.executable, str(_script_path()), str(config_file)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "VALID:" in result.stdout


def test_validate_script_invalid_config(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("General: {}\n", encoding="utf-8")

    cmd = [sys.executable, str(_script_path()), str(config_file)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "INVALID:" in result.stdout

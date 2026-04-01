from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from config.compatibility import ConfigCompatibilityError
from core.simulation import DEFAULT_EFFECT_ORDER, Simulation


def test_simulation_run_returns_backend_status() -> None:
    sim = Simulation(backend="numpy")
    payload = sim.run()
    assert payload["status"] == "ok"
    assert payload["backend"] == "numpy"
    assert payload["planned_effect_order"] == list(DEFAULT_EFFECT_ORDER)
    assert payload["output_file"] is None


def test_planned_effect_order_contract() -> None:
    sim = Simulation()
    assert sim.planned_effect_order() == DEFAULT_EFFECT_ORDER


def test_simulation_run_creates_output_hdf5(tmp_path: Path) -> None:
    output = tmp_path / "sim_output.hdf5"
    sim = Simulation(output_path=output)
    payload = sim.run()

    assert payload["output_file"] == str(output)
    assert output.exists()

    with h5py.File(output, "r") as handle:
        assert "InputParameters" in handle
        assert "Images" in handle
        assert handle.attrs["simulator"] == "plato-gpu-image"
        assert handle.attrs["backend"] == "numpy"


def test_simulation_from_legacy_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "input.yaml"
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
    sim = Simulation.from_legacy_yaml(config_file)
    assert sim.config is not None
    assert sim.config["ObservingParameters"]["NumExposures"] == 10


def test_simulation_from_legacy_yaml_strict_rejects_invalid(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("General: {}\n", encoding="utf-8")
    with pytest.raises(ConfigCompatibilityError):
        Simulation.from_legacy_yaml(config_file, strict_core_contract=True)

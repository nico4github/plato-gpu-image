from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from config.compatibility import ConfigCompatibilityError
from core.simulation import DEFAULT_EFFECT_ORDER, Simulation
from tests.output_paths import tagged_output_file


def _fixture(name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "tests" / "input_yaml" / name


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
    sim = Simulation.from_legacy_yaml(_fixture("minimal_valid.yaml"))
    assert sim.config is not None
    assert sim.config["ObservingParameters"]["NumExposures"] == 2


def test_simulation_from_legacy_yaml_strict_rejects_invalid(tmp_path: Path) -> None:
    with pytest.raises(ConfigCompatibilityError):
        Simulation.from_legacy_yaml(_fixture("minimal_invalid.yaml"), strict_core_contract=True)


def test_simulation_writes_images_for_configured_exposures(tmp_path: Path) -> None:
    import yaml

    config = yaml.safe_load(_fixture("minimal_valid.yaml").read_text(encoding="utf-8"))
    output = tmp_path / "images.hdf5"
    sim = Simulation(config=config, output_path=output)
    sim.run()

    with h5py.File(output, "r") as handle:
        assert "rawConfigYAML" in handle["InputParameters"]
        assert "image0000005" in handle["Images"]
        assert "image0000006" in handle["Images"]
        assert "smearingMap0000005" in handle["SmearingMaps"]
        assert "biasMap0000005" in handle["BiasMapsLeft"]
        assert "biasMap0000005" in handle["BiasMapsRight"]
        assert "throughputMap0000005" in handle["ThroughputMaps"]
        assert "transmissionEfficiency" in handle["TransmissionEfficiency"]
        assert "skyBackground" in handle["BackgroundMap"]
        assert "PRNU" in handle["Flatfield"]
        assert "time" in handle["Telescope"]
        assert "telescopeRA" in handle["Telescope"]
        image = handle["Images"]["image0000005"][:]
        assert image.shape == (4, 3)
        # background * cycle time => 2 * 10
        assert float(image[0, 0]) == pytest.approx(20.0)
        assert handle["SmearingMaps"]["smearingMap0000005"][:].shape == (30, 3)
        assert handle["BiasMapsLeft"]["biasMap0000005"][:].shape == (25, 15)
        assert handle["ThroughputMaps"]["throughputMap0000005"][:].shape == (4, 3)
        assert handle["TransmissionEfficiency"]["transmissionEfficiency"][:].shape == (2,)
        assert handle["BackgroundMap"]["skyBackground"][:].shape == (2,)
        assert handle["Flatfield"]["PRNU"][:].shape == (4, 3)
        assert handle["Telescope"]["time"][:].shape == (2,)

        assert "time" in handle["ACS"]
        assert "yaw" in handle["ACS"]
        assert handle["ACS"]["time"][:].shape == (2,)
        assert handle["ACS"]["platformRA"][:].shape == (2,)
        raw_yaml = handle["InputParameters"]["rawConfigYAML"][()]
        if isinstance(raw_yaml, bytes):
            raw_yaml = raw_yaml.decode("utf-8")
        assert "ObservingParameters:" in raw_yaml


def test_simulation_skips_images_when_writepixelmaps_false(tmp_path: Path) -> None:
    config = {
        "ObservingParameters": {"NumExposures": 1, "BeginExposureNr": 0, "CycleTime": 1.0},
        "SubField": {"NumRows": 2, "NumColumns": 2},
        "Sky": {"SkyBackground": {"UseConstantSkyBackground": True, "BackgroundValue": 1.0}},
        "ControlHDF5Content": {"WritePixelMaps": False},
    }
    output = tmp_path / "no_images.hdf5"
    sim = Simulation(config=config, output_path=output)
    sim.run()

    with h5py.File(output, "r") as handle:
        assert len(handle["Images"].keys()) == 0


def test_simulation_writes_into_tests_output_file_directory() -> None:
    output = tagged_output_file("local", "simulation_fixture_output.hdf5")
    if output.exists():
        output.unlink()

    sim = Simulation.from_legacy_yaml(_fixture("minimal_valid.yaml"), output_path=output)
    sim.run()

    assert output.exists()
    with h5py.File(output, "r") as handle:
        assert "Images" in handle
        assert "image0000005" in handle["Images"]


def test_simulation_photon_noise_is_deterministic_from_seed(tmp_path: Path) -> None:
    config = {
        "ObservingParameters": {"NumExposures": 1, "BeginExposureNr": 0, "CycleTime": 1.0},
        "SubField": {"NumRows": 2, "NumColumns": 3},
        "Sky": {"SkyBackground": {"UseConstantSkyBackground": True, "BackgroundValue": 10.0}},
        "CCD": {"IncludePhotonNoise": True},
        "RandomSeeds": {"PhotonNoiseSeed": 7},
        "ControlHDF5Content": {"WritePixelMaps": True},
    }
    output = tmp_path / "noise.hdf5"
    sim = Simulation(config=config, output_path=output)
    sim.run()

    with h5py.File(output, "r") as handle:
        observed = handle["Images"]["image0000000"][:]

    rng = np.random.default_rng(7)
    expected = rng.poisson(10.0, size=(2, 3)).astype(np.float32)
    assert np.array_equal(observed, expected)

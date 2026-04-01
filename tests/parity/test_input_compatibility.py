from __future__ import annotations

from pathlib import Path

import pytest

from platosim_py.config.compatibility import (
    ConfigCompatibilityError,
    flatten_paths,
    has_path,
    load_legacy_yaml,
)


def test_legacy_yaml_top_level_mapping(tmp_path: Path) -> None:
    config_file = tmp_path / "input.yaml"
    config_file.write_text(
        "General:\n  ProjectLocation: ENV['PLATO_PROJECT_HOME']\n"
        "ObservingParameters:\n  NumExposures: 10\n"
        "Sky: {}\nPlatform: {}\nTelescope: {}\nCamera: {}\nPSF: {}\nFEE: {}\n"
        "CCD: {}\nSubField: {}\nRandomSeeds: {}\nControlHDF5Content: {}\n",
        encoding="utf-8",
    )
    data = load_legacy_yaml(config_file)
    assert data["ObservingParameters"]["NumExposures"] == 10


def test_missing_required_top_level_section_raises(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("General: {}\n", encoding="utf-8")
    with pytest.raises(ConfigCompatibilityError):
        load_legacy_yaml(config_file)


def test_alias_normalization_ra_dec_pointing(tmp_path: Path) -> None:
    config_file = tmp_path / "legacy_alias.yaml"
    config_file.write_text(
        "General: {}\n"
        "ObservingParameters:\n"
        "  NumExposures: 10\n"
        "  RApointing: 12.5\n"
        "  DecPointing: -4.0\n"
        "Sky: {}\nPlatform:\n  Orientation:\n    Source: Angles\n"
        "Telescope: {}\nCamera: {}\nPSF: {}\nFEE: {}\nCCD: {}\nSubField: {}\n"
        "RandomSeeds: {}\nControlHDF5Content: {}\n",
        encoding="utf-8",
    )
    data = load_legacy_yaml(config_file)
    assert data["Platform"]["Orientation"]["Angles"]["RAPointing"] == 12.5
    assert data["Platform"]["Orientation"]["Angles"]["DecPointing"] == -4.0


def test_flatten_paths_includes_nested_leaf_paths() -> None:
    tree = {"A": {"B": {"C": 1}}, "X": True}
    paths = flatten_paths(tree, leaves_only=True)
    assert "A/B/C" in paths
    assert "X" in paths
    assert "A" not in paths


def test_collect_required_paths_from_reference(tmp_path: Path) -> None:
    config_file = tmp_path / "reference.yaml"
    config_file.write_text(
        "General: {}\n"
        "ObservingParameters:\n"
        "  NumExposures: 5\n",
        encoding="utf-8",
    )
    data = load_legacy_yaml(config_file, required_top_level_sections=())
    paths = flatten_paths(data, leaves_only=True)
    assert "ObservingParameters/NumExposures" in paths


def _local_platosim3_inputfile() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.parent / "platosim_develop" / "PlatoSim3" / "inputfiles" / "inputfile.yaml"


def test_canonical_platosim3_input_parses_when_available() -> None:
    inputfile = _local_platosim3_inputfile()
    if not inputfile.exists():
        pytest.skip(f"Canonical input file not found: {inputfile}")

    data = load_legacy_yaml(inputfile)
    assert data["General"]["ProjectLocation"] == "ENV['PLATO_PROJECT_HOME']"
    assert has_path(data, "Platform/Orientation/Angles/RAPointing")
    assert has_path(data, "ControlHDF5Content/WritePixelMaps")

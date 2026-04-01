from __future__ import annotations

from pathlib import Path

import h5py

from platosim_py.core.simulation import DEFAULT_EFFECT_ORDER, Simulation


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

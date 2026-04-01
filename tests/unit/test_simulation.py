from __future__ import annotations

from platosim_py.core.simulation import Simulation


def test_simulation_run_returns_backend_status() -> None:
    sim = Simulation(backend="numpy")
    payload = sim.run()
    assert payload["status"] == "ok"
    assert payload["backend"] == "numpy"


from __future__ import annotations

from platosim_py.core.simulation import DEFAULT_EFFECT_ORDER, Simulation


def test_simulation_run_returns_backend_status() -> None:
    sim = Simulation(backend="numpy")
    payload = sim.run()
    assert payload["status"] == "ok"
    assert payload["backend"] == "numpy"
    assert payload["planned_effect_order"] == list(DEFAULT_EFFECT_ORDER)


def test_planned_effect_order_contract() -> None:
    sim = Simulation()
    assert sim.planned_effect_order() == DEFAULT_EFFECT_ORDER

"""Core simulation orchestration skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from platosim_py.backends import backend_array_namespace, resolve_backend

DEFAULT_EFFECT_ORDER: tuple[str, ...] = (
    "reset",
    "integrate_light",
    "apply_throughput_efficiency",
    "apply_charge_injection",
    "apply_open_shutter_smearing",
    "add_photon_noise",
    "add_dark_signal",
    "read_out",
    "write_output",
)


@dataclass(slots=True)
class Simulation:
    """Pure-Python simulation entry point (skeleton)."""

    backend: str = "numpy"
    config: dict[str, Any] | None = None

    def array_namespace(self):
        """Return backend array namespace (numpy/cupy)."""
        return backend_array_namespace(self.backend)

    def planned_effect_order(self) -> tuple[str, ...]:
        """Return current execution order contract for one exposure."""
        return DEFAULT_EFFECT_ORDER

    def run(self) -> dict[str, Any]:
        """Run the simulation.

        Returns a run-status payload for now; this method will evolve into the
        full exposure pipeline runner.
        """
        backend_module = resolve_backend(self.backend)
        return {
            "status": "ok",
            "backend": backend_module.name(),
            "planned_effect_order": list(self.planned_effect_order()),
        }

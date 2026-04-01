"""Core simulation orchestration skeleton."""

from __future__ import annotations

from dataclasses import dataclass

from platosim_py.backends import backend_array_namespace, resolve_backend


@dataclass(slots=True)
class Simulation:
    """Pure-Python simulation entry point (skeleton)."""

    backend: str = "numpy"

    def array_namespace(self):
        """Return backend array namespace (numpy/cupy)."""
        return backend_array_namespace(self.backend)

    def run(self) -> dict[str, str]:
        """Run the simulation.

        Returns a run-status payload for now; this method will evolve into the
        full exposure pipeline runner.
        """
        backend_module = resolve_backend(self.backend)
        return {"status": "ok", "backend": backend_module.name()}

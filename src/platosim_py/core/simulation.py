"""Core simulation orchestration skeleton."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Simulation:
    """Pure-Python simulation entry point (skeleton)."""

    backend: str = "numpy"

    def run(self) -> None:
        """Run the simulation.

        This method is intentionally a placeholder for phased implementation.
        """
        return None

"""Core simulation orchestration skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backends import backend_array_namespace, resolve_backend
from config.compatibility import load_core_compatible_yaml, load_legacy_yaml
from simio.hdf5 import HDF5Writer

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
    output_path: str | Path | None = None
    overwrite_output: bool = True

    @classmethod
    def from_legacy_yaml(
        cls,
        config_path: str | Path,
        *,
        backend: str = "numpy",
        output_path: str | Path | None = None,
        strict_core_contract: bool = True,
        overwrite_output: bool = True,
    ) -> "Simulation":
        """Construct simulation object from a legacy YAML configuration."""
        if strict_core_contract:
            config = load_core_compatible_yaml(config_path)
        else:
            config = load_legacy_yaml(config_path)
        return cls(
            backend=backend,
            config=config,
            output_path=output_path,
            overwrite_output=overwrite_output,
        )

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
        output_file = None
        if self.output_path is not None:
            writer = HDF5Writer(self.output_path)
            writer.initialize_file(overwrite=self.overwrite_output)
            writer.ensure_legacy_groups()
            writer.write_root_metadata(
                {
                    "simulator": "plato-gpu-image",
                    "backend": backend_module.name(),
                    "status": "initialized",
                }
            )
            output_file = str(Path(self.output_path))

        return {
            "status": "ok",
            "backend": backend_module.name(),
            "planned_effect_order": list(self.planned_effect_order()),
            "output_file": output_file,
        }

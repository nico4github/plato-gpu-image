"""Core simulation orchestration skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
            if self.config is not None:
                self._write_exposure_images(writer)
            output_file = str(Path(self.output_path))

        return {
            "status": "ok",
            "backend": backend_module.name(),
            "planned_effect_order": list(self.planned_effect_order()),
            "output_file": output_file,
        }

    def _write_exposure_images(self, writer: HDF5Writer) -> None:
        """Write synthetic exposure products from current configuration.

        Current behavior:
        - create one image per configured exposure
        - image content uses configurable constant background baseline
        - optional throughput scaling and simple noise terms
        - emits additional legacy-style maps (bias/smearing/throughput/ACS)
        - preserves legacy image naming convention imageXXXXXXX
        """
        config = self.config or {}
        num_exposures = int(self._cfg(config, "ObservingParameters/NumExposures", default=1))
        begin_exposure = int(
            self._cfg(config, "ObservingParameters/BeginExposureNr", default=0)
        )
        num_rows = int(self._cfg(config, "SubField/NumRows", default=100))
        num_cols = int(self._cfg(config, "SubField/NumColumns", default=100))
        num_bias_rows = int(self._cfg(config, "SubField/NumBiasPrescanRows", default=25))
        num_bias_cols = int(self._cfg(config, "SubField/NumBiasPrescanColumns", default=15))
        num_smearing_rows = int(self._cfg(config, "SubField/NumSmearingOverscanRows", default=30))

        xp = self.array_namespace()
        rng = self._rng(config)

        # ACS vectors (written once for full run)
        if bool(self._cfg(config, "ControlHDF5Content/WriteACS", default=True)):
            self._write_acs(writer, config, num_exposures, begin_exposure)

        write_images = bool(self._cfg(config, "ControlHDF5Content/WritePixelMaps", default=True))
        write_smearing = bool(
            self._cfg(config, "ControlHDF5Content/WriteSmearingMaps", default=True)
        )
        write_bias = bool(self._cfg(config, "ControlHDF5Content/WriteBiasMaps", default=True))
        write_throughput = bool(
            self._cfg(config, "ControlHDF5Content/WriteThroughputMaps", default=True)
        )

        throughput_level = self._throughput_level(config)

        for exposure_nr in range(begin_exposure, begin_exposure + num_exposures):
            image = xp.full(
                (num_rows, num_cols),
                self._background_level(config),
                dtype=xp.float32,
            )
            image *= throughput_level
            image = self._apply_noise_terms(image, config, rng)

            suffix = f"{exposure_nr:07d}"

            if write_images:
                writer.write_dataset("/Images", f"image{suffix}", image, overwrite=True)

            if write_smearing:
                smearing = xp.zeros((num_smearing_rows, num_cols), dtype=xp.float32)
                writer.write_dataset(
                    "/SmearingMaps",
                    f"smearingMap{suffix}",
                    smearing,
                    overwrite=True,
                )

            if write_bias:
                bias_left = xp.zeros((num_bias_rows, num_bias_cols), dtype=xp.float32)
                bias_right = xp.zeros((num_bias_rows, num_bias_cols), dtype=xp.float32)
                writer.write_dataset("/BiasMapsLeft", f"biasMap{suffix}", bias_left, overwrite=True)
                writer.write_dataset("/BiasMapsRight", f"biasMap{suffix}", bias_right, overwrite=True)

            if write_throughput:
                throughput = xp.full((num_rows, num_cols), throughput_level, dtype=xp.float32)
                writer.write_dataset(
                    "/ThroughputMaps",
                    f"throughputMap{suffix}",
                    throughput,
                    overwrite=True,
                )

    def _background_level(self, config: dict[str, Any]) -> float:
        use_constant = bool(
            self._cfg(config, "Sky/SkyBackground/UseConstantSkyBackground", default=True)
        )
        if not use_constant:
            return 0.0

        raw_value = float(self._cfg(config, "Sky/SkyBackground/BackgroundValue", default=0.0))
        if raw_value < 0:
            # Legacy semantics: negative means "compute dynamically".
            # v0 implementation falls back to deterministic zero.
            return 0.0

        cycle_time = float(self._cfg(config, "ObservingParameters/CycleTime", default=1.0))
        return raw_value * cycle_time

    def _throughput_level(self, config: dict[str, Any]) -> float:
        include = bool(self._cfg(config, "CCD/IncludeRelativeTransmissivity", default=True))
        if not include:
            return 1.0
        level = float(
            self._cfg(config, "CCD/RelativeTransmissivity/ExpectedValue", default=1.0)
        )
        return float(max(level, 0.0))

    def _apply_noise_terms(
        self,
        image: Any,
        config: dict[str, Any],
        rng: np.random.Generator,
    ) -> Any:
        xp = self.array_namespace()
        np_image = np.asarray(image.get() if hasattr(image, "get") else image, dtype=np.float32)

        cycle_time = float(self._cfg(config, "ObservingParameters/CycleTime", default=1.0))

        if bool(self._cfg(config, "CCD/IncludeDarkSignal", default=False)):
            dark_current = float(self._cfg(config, "CCD/DarkSignal/DarkCurrent", default=0.0))
            np_image += np.float32(max(dark_current, 0.0) * cycle_time)

        if bool(self._cfg(config, "CCD/IncludePhotonNoise", default=False)):
            lam = np.clip(np_image, 0.0, None)
            np_image = rng.poisson(lam).astype(np.float32)

        if bool(self._cfg(config, "CCD/IncludeReadoutNoise", default=False)):
            sigma = float(self._cfg(config, "CCD/ReadoutNoise", default=0.0))
            if sigma > 0:
                np_image += rng.normal(0.0, sigma, size=np_image.shape).astype(np.float32)

        np.clip(np_image, 0.0, None, out=np_image)
        return xp.asarray(np_image, dtype=xp.float32)

    def _rng(self, config: dict[str, Any]) -> np.random.Generator:
        seed = int(self._cfg(config, "RandomSeeds/PhotonNoiseSeed", default=0))
        if seed < 0:
            # deterministic fallback for reproducibility in tests
            seed = 0
        return np.random.default_rng(seed)

    def _write_acs(
        self,
        writer: HDF5Writer,
        config: dict[str, Any],
        num_exposures: int,
        begin_exposure: int,
    ) -> None:
        cycle_time = float(self._cfg(config, "ObservingParameters/CycleTime", default=1.0))
        time = np.arange(
            begin_exposure * cycle_time,
            (begin_exposure + num_exposures) * cycle_time,
            cycle_time,
            dtype=np.float64,
        )
        if time.size > num_exposures:
            time = time[:num_exposures]

        ra = float(self._cfg(config, "Platform/Orientation/Angles/RAPointing", default=0.0))
        dec = float(self._cfg(config, "Platform/Orientation/Angles/DecPointing", default=0.0))

        writer.write_dataset("/ACS", "time", time, overwrite=True)
        writer.write_dataset("/ACS", "yaw", np.zeros(num_exposures, dtype=np.float32), overwrite=True)
        writer.write_dataset("/ACS", "pitch", np.zeros(num_exposures, dtype=np.float32), overwrite=True)
        writer.write_dataset("/ACS", "roll", np.zeros(num_exposures, dtype=np.float32), overwrite=True)
        writer.write_dataset(
            "/ACS",
            "platformRA",
            np.full(num_exposures, ra, dtype=np.float32),
            overwrite=True,
        )
        writer.write_dataset(
            "/ACS",
            "platformDec",
            np.full(num_exposures, dec, dtype=np.float32),
            overwrite=True,
        )

    @staticmethod
    def _cfg(config: dict[str, Any], path: str, *, default: Any) -> Any:
        node: Any = config
        for key in path.split("/"):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

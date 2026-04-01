"""Core simulation orchestration skeleton."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

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
    config_source_path: str | Path | None = None
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
            config_source_path=config_path,
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
                writer.write_string_dataset(
                    "/InputParameters",
                    "rawConfigYAML",
                    yaml.safe_dump(self.config, sort_keys=True),
                )
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
        self._write_vector_outputs(
            writer=writer,
            config=config,
            num_exposures=num_exposures,
            begin_exposure=begin_exposure,
            num_rows=num_rows,
            num_cols=num_cols,
        )

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
            image = self._inject_star_catalog(image, config)
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

    def _inject_star_catalog(self, image: Any, config: dict[str, Any]) -> Any:
        """Inject synthetic star signal from configured star catalog.

        This is an initial implementation to render meaningful content:
        - reads (RA, Dec, mag) triplets from StarCatalogFile
        - projects stars to subfield using catalog min/max normalization
        - applies a compact 3x3 Gaussian-like kernel per source
        """
        star_path = self._resolve_star_catalog_path(config)
        if star_path is None or not star_path.exists():
            return image

        stars = self._read_star_catalog(star_path)
        if stars.size == 0:
            return image

        xp = self.array_namespace()
        np_image = np.asarray(image.get() if hasattr(image, "get") else image, dtype=np.float32)
        nrows, ncols = np_image.shape

        ra = stars[:, 0]
        dec = stars[:, 1]
        mag = stars[:, 2]

        ra_span = max(float(np.max(ra) - np.min(ra)), 1e-9)
        dec_span = max(float(np.max(dec) - np.min(dec)), 1e-9)

        col = np.clip(((ra - np.min(ra)) / ra_span * (ncols - 1)).astype(int), 0, ncols - 1)
        row = np.clip(((dec - np.min(dec)) / dec_span * (nrows - 1)).astype(int), 0, nrows - 1)

        flux = self._star_flux_from_magnitude(config, mag)

        # Compact PSF-like kernel (normalized).
        kernel = np.array(
            [
                [0.075, 0.123, 0.075],
                [0.123, 0.208, 0.123],
                [0.075, 0.123, 0.075],
            ],
            dtype=np.float32,
        )
        kernel /= float(np.sum(kernel))

        for r, c, f in zip(row, col, flux):
            if f <= 0:
                continue
            for kr in range(-1, 2):
                rr = int(r + kr)
                if rr < 0 or rr >= nrows:
                    continue
                for kc in range(-1, 2):
                    cc = int(c + kc)
                    if cc < 0 or cc >= ncols:
                        continue
                    np_image[rr, cc] += np.float32(f * kernel[kr + 1, kc + 1])

        return xp.asarray(np_image, dtype=xp.float32)

    def _star_flux_from_magnitude(self, config: dict[str, Any], mag: np.ndarray) -> np.ndarray:
        flux_m0 = float(self._cfg(config, "ObservingParameters/Fluxm0", default=1.0e8))
        cycle_time = float(self._cfg(config, "ObservingParameters/CycleTime", default=1.0))

        # Scaled physical-like relation to keep values numerically stable.
        # Relative to a reference 10th magnitude source.
        ref_mag = 10.0
        base = flux_m0 * cycle_time * 1e-8
        rel = np.power(10.0, -0.4 * (mag - ref_mag), dtype=np.float64)
        flux = base * rel
        return np.clip(flux.astype(np.float32), 0.0, 1e7)

    def _resolve_star_catalog_path(self, config: dict[str, Any]) -> Path | None:
        raw = self._cfg(config, "ObservingParameters/StarCatalogFile", default=None)
        if raw is None:
            return None

        candidate = Path(str(raw))
        if candidate.is_absolute():
            return candidate

        roots: list[Path] = []

        # Legacy ENV convention.
        project_home = os.environ.get("PLATO_PROJECT_HOME")
        if project_home:
            roots.append(Path(project_home))

        # Relative to config file location.
        if self.config_source_path is not None:
            source = Path(self.config_source_path).resolve()
            roots.append(source.parent)

        # Relative to repository root.
        roots.append(Path.cwd())

        for root in roots:
            resolved = (root / candidate).resolve()
            if resolved.exists():
                return resolved
        return candidate

    @staticmethod
    def _read_star_catalog(path: Path) -> np.ndarray:
        rows: list[tuple[float, float, float]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) < 3:
                    continue
                try:
                    ra = float(parts[0])
                    dec = float(parts[1])
                    mag = float(parts[2])
                except ValueError:
                    continue
                rows.append((ra, dec, mag))

        if not rows:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def _write_vector_outputs(
        self,
        *,
        writer: HDF5Writer,
        config: dict[str, Any],
        num_exposures: int,
        begin_exposure: int,
        num_rows: int,
        num_cols: int,
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

        if bool(self._cfg(config, "ControlHDF5Content/WriteTransmissionEfficiency", default=True)):
            trans_bol = float(
                self._cfg(config, "Telescope/TransmissionEfficiency/BOL", default=1.0)
            )
            trans_vec = np.full(num_exposures, trans_bol, dtype=np.float32)
            writer.write_dataset(
                "/TransmissionEfficiency",
                "transmissionEfficiency",
                trans_vec,
                overwrite=True,
            )

        if bool(self._cfg(config, "ControlHDF5Content/WriteBackgroundMap", default=True)):
            bg_raw = float(self._cfg(config, "Sky/SkyBackground/BackgroundValue", default=0.0))
            if bg_raw < 0:
                bg_raw = 0.0
            bg_vec = np.full(num_exposures, bg_raw, dtype=np.float32)
            writer.write_dataset("/BackgroundMap", "skyBackground", bg_vec, overwrite=True)

        if bool(self._cfg(config, "ControlHDF5Content/WriteFlatfieldMap", default=True)):
            prnu = np.ones((num_rows, num_cols), dtype=np.float32)
            writer.write_dataset("/Flatfield", "PRNU", prnu, overwrite=True)

        if bool(self._cfg(config, "ControlHDF5Content/WriteTelescopeACS", default=True)):
            ra = float(self._cfg(config, "Platform/Orientation/Angles/RAPointing", default=0.0))
            dec = float(self._cfg(config, "Platform/Orientation/Angles/DecPointing", default=0.0))
            writer.write_dataset("/Telescope", "time", time, overwrite=True)
            writer.write_dataset(
                "/Telescope",
                "telescopeYaw",
                np.zeros(num_exposures, dtype=np.float32),
                overwrite=True,
            )
            writer.write_dataset(
                "/Telescope",
                "telescopePitch",
                np.zeros(num_exposures, dtype=np.float32),
                overwrite=True,
            )
            writer.write_dataset(
                "/Telescope",
                "telescopeRoll",
                np.zeros(num_exposures, dtype=np.float32),
                overwrite=True,
            )
            writer.write_dataset(
                "/Telescope",
                "telescopeRA",
                np.full(num_exposures, ra, dtype=np.float32),
                overwrite=True,
            )
            writer.write_dataset(
                "/Telescope",
                "telescopeDec",
                np.full(num_exposures, dec, dtype=np.float32),
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

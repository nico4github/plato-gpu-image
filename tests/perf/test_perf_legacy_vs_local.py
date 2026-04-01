from __future__ import annotations

import json
import os
import subprocess
import time
import zlib
import struct
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from core.simulation import Simulation
from tests.output_paths import tagged_output_file


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _legacy_root() -> Path:
    return _repo_root().parent / "platosim_develop" / "PlatoSim3"


def _legacy_binary() -> Path:
    return _legacy_root() / "build" / "platosim"


def _prepare_perf_config() -> Path:
    legacy_root = _legacy_root()
    source_yaml = legacy_root / "inputfiles" / "inputfile.yaml"
    cfg = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))

    # Keep this benchmark lightweight but meaningful.
    cfg["ObservingParameters"]["NumExposures"] = 1
    cfg["ObservingParameters"]["BeginExposureNr"] = 0
    cfg["SubField"]["NumRows"] = 100
    cfg["SubField"]["NumColumns"] = 100

    config_path = tagged_output_file("local", "perf_config.yaml")
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return config_path


def _write_preview_png(
    hdf5_path: Path, dataset_path: str, output_png: Path, *, percentile: float = 99.5
) -> None:
    with h5py.File(hdf5_path, "r") as handle:
        if dataset_path not in handle:
            raise KeyError(f"Dataset not found in HDF5: {dataset_path}")
        image = np.asarray(handle[dataset_path][()], dtype=np.float32)

    if image.ndim != 2:
        raise ValueError(f"Expected 2D dataset for preview, got shape {image.shape}")

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(image))
    hi = float(np.percentile(image, percentile))
    if hi <= lo:
        scaled = np.zeros_like(image, dtype=np.uint8)
    else:
        clipped = np.clip(image, lo, hi)
        scaled = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    _write_grayscale_png(output_png, scaled)


def _write_grayscale_png(path: Path, image: np.ndarray) -> None:
    """Write an 8-bit grayscale PNG using stdlib only."""
    if image.ndim != 2:
        raise ValueError(f"PNG writer expects 2D array, got {image.shape}")
    height, width = image.shape
    raw_rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    compressed = zlib.compress(raw_rows, level=9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)  # grayscale
    png_bytes = signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
    path.write_bytes(png_bytes)


@pytest.mark.skipif(
    os.environ.get("RUN_LEGACY_PERF", "1") != "1",
    reason="Set RUN_LEGACY_PERF=1 to enable legacy/local perf benchmark",
)
def test_perf_legacy_vs_local_smoke() -> None:
    legacy_bin = _legacy_binary()
    if not legacy_bin.exists():
        pytest.skip(f"Legacy binary not found: {legacy_bin}")

    config_path = _prepare_perf_config()
    legacy_output = tagged_output_file("legacy", "perf_smoke.hdf5")
    legacy_log = tagged_output_file("legacy", "perf_smoke.log")
    local_output = tagged_output_file("local", "perf_smoke.hdf5")
    report_path = tagged_output_file("local", "perf_report.json")
    legacy_preview = tagged_output_file("legacy", "perf_preview.png")
    local_preview = tagged_output_file("local", "perf_preview.png")

    for artifact in (legacy_output, legacy_log, local_output, report_path, legacy_preview, local_preview):
        if artifact.exists():
            artifact.unlink()

    env = os.environ.copy()
    env.setdefault("PLATO_PROJECT_HOME", str(_legacy_root()))

    t0 = time.perf_counter()
    legacy_rc = subprocess.run(
        [str(legacy_bin), str(config_path), str(legacy_output), str(legacy_log), "1"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    legacy_seconds = time.perf_counter() - t0
    assert legacy_rc.returncode == 0, legacy_rc.stderr or legacy_rc.stdout

    t1 = time.perf_counter()
    sim = Simulation.from_legacy_yaml(
        config_path,
        backend="numpy",
        output_path=local_output,
        strict_core_contract=True,
        overwrite_output=True,
    )
    sim.run()
    local_seconds = time.perf_counter() - t1

    assert legacy_output.exists()
    assert local_output.exists()

    _write_preview_png(legacy_output, "/Images/image0000000", legacy_preview)
    _write_preview_png(local_output, "/Images/image0000000", local_preview)

    speedup_local_over_legacy = (
        (legacy_seconds / local_seconds) if local_seconds > 0 else float("inf")
    )
    report = {
        "benchmark": "legacy_vs_local_smoke",
        "config": str(config_path),
        "legacy_seconds": legacy_seconds,
        "local_seconds": local_seconds,
        "speedup_local_over_legacy": speedup_local_over_legacy,
        "legacy_output": str(legacy_output),
        "local_output": str(local_output),
        "legacy_preview": str(legacy_preview),
        "local_preview": str(local_preview),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    # Sanity expectations only; hard perf budgets would be too environment-dependent.
    assert legacy_seconds > 0
    assert local_seconds > 0

# plato-gpu-image

Standalone repository for a pure-Python implementation of PlatoSim with optional GPU acceleration.

## TL;DR Run A Simulation

Use a legacy PlatoSim YAML input file directly from the command line:

```bash
source .venv/bin/activate
chmod +x ./plato-gpu-image.py
./plato-gpu-image.py \
  --input ../platosim_develop/PlatoSim3/inputfiles/inputfile.yaml \
  --output output/sim_run.hdf5 \
  --backend numpy \
  --strict-core-contract \
  --no-overwrite-output
```

If your shell blocks direct execution, run the same CLI via Python:

```bash
python plato-gpu-image.py --input ../platosim_develop/PlatoSim3/inputfiles/inputfile.yaml --output output/sim_run.hdf5 --backend numpy --strict-core-contract --no-overwrite-output
```

Optional pre-check for input compatibility:

```bash
python scripts/validate_legacy_config.py ../platosim_develop/PlatoSim3/inputfiles/inputfile.yaml
```

## Why This Repo Exists

This project is intentionally separated from the existing `PlatoSim3` repository to:
- keep the original mixed C++/Python codebase clean and stable,
- allow independent release cadence for pure-Python migration,
- support focused benchmarking and validation of CPU/GPU backends.

## Core Requirements

- Pure-Python codebase for project-owned implementation (no project-maintained C++ extensions).
- Optional GPU acceleration through Python CUDA libraries (initially CuPy).
- Full compatibility with existing PlatoSim3 input YAML files.
- Side-by-side parity validation against existing PlatoSim3 behavior and outputs.

## Consolidated Migration Plan

### 1. Architecture

Implement a modular simulation engine directly under `src/`:
- `config`: input parsing, schema normalization, validation, legacy key support.
- `core`: simulation orchestration (`run`, exposure scheduling, effect ordering).
- `models`: physical models (platform, camera, detector, PSF, noise/effects).
- `backends`: array abstraction and backend-specific ops (`numpy`, `cupy`).
- `simio`: HDF5 read/write compatibility with legacy output structure.

### 2. Compatibility (Non-Negotiable)

Input compatibility is a hard acceptance criterion:
- unchanged support for existing YAML structure, key names, nesting, units, semantics,
- same defaults and toggle behavior where defined by PlatoSim3,
- compatibility layer for legacy/alias keys,
- deterministic handling of random seeds aligned with current behavior.

Validation gates:
- golden corpus of legacy config files from `PlatoSim3/inputfiles`,
- parser acceptance tests for all supported key paths,
- simulation parity checks using unchanged input files.

### 3. Phased Delivery

1. Foundation phase
   - package skeleton, typed config model, backend abstraction, HDF5 layout compatibility tests.
2. Core simulation phase
   - exposure loop and effect ordering parity.
3. PSF phase
   - mapped PSF path with FFT on NumPy/SciPy backend,
   - optional CuPy FFT implementation for CUDA.
4. Detector effects phase
   - throughput, smearing, noise models, CTI, saturation, gain/quantization.
5. Performance phase
   - profile hotspots, vectorize kernels, selective Numba on CPU-heavy loops, GPU acceleration expansion.
6. Exit criteria phase
   - parity thresholds met, compatibility tests green, reproducible benchmark reports.

### 4. CUDA Strategy

- CPU NumPy backend is reference and mandatory.
- CuPy backend is optional, runtime-selectable.
- No divergence in scientific interface between CPU and GPU paths.
- Numerical tolerance policy documents acceptable backend differences.

### 5. Repository Hygiene Policy

- Do not implement this migration in `PlatoSim3`.
- Keep this repo standalone and independently versioned.
- Mirror only fixtures/configs required for compatibility and tests.

## Initial Folder Tree

```text
plato-gpu-image/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── benchmarks/
├── configs/
├── docs/
├── scripts/
├── src/
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── numpy_backend.py
│   │   └── cupy_backend.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── compatibility.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── simulation.py
│   ├── simio/
│   │   ├── __init__.py
│   │   └── hdf5.py
│   └── models/
│       └── __init__.py
└── tests/
    ├── parity/
    │   └── test_input_compatibility.py
    └── unit/
        └── __init__.py
```

## Quick Start

```bash
git clone <this-repo-url>
cd plato-gpu-image
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

With CUDA (optional):

```bash
pip install -e .[cuda]
```

## Next Implementation Milestones

- Lock a v1 compatibility schema contract (required path inventory + defaults policy).
- Implement core effect ordering in `Simulation.run()` for one minimal exposure path.
- Add HDF5 output compatibility skeleton for legacy group/dataset naming.
- Expand parity tests with real fixture comparisons against selected PlatoSim3 runs.

## Current Foundation Status

- Compatibility loader implemented with:
  - required top-level section checks,
  - legacy alias normalization (`RApointing`/`DecPointing`),
  - slash-path utilities for schema and validation checks.
- Backend framework implemented:
  - backend resolver (`numpy`/`cupy`),
  - simulation skeleton now resolves backend, reports runtime selection, and exposes a deterministic v1 effect-order contract.
- Benchmarks scaffold implemented:
  - `benchmarks/run_backend_benchmarks.py` runs local NumPy FFT smoke benchmarks now,
  - same CLI path reserved for future CuPy benchmarking on GPU hosts.
- HDF5 output scaffold implemented:
  - legacy group creation helper in `src/simio/hdf5.py`,
  - root metadata writer + unit tests.
- Simulation integration scaffold:
  - `Simulation.run()` can now initialize a legacy-layout HDF5 output file,
  - backend and run metadata are written at file root,
  - a full input snapshot is stored as `/InputParameters/rawConfigYAML`,
  - `Simulation.from_legacy_yaml(...)` loads validated legacy configs directly.
- Automated tests currently pass on CPU-only environment.
- Validation tooling:
  - `scripts/validate_legacy_config.py` validates legacy YAML files against the core compatibility contract.
  - `scripts/compare_hdf5_structure.py` compares group/dataset structure between two HDF5 files.
  - `scripts/preview_hdf5_side_by_side.py` prints side-by-side previews of key legacy vs local datasets (shape + stats).
- Test fixture layout:
  - input YAML fixtures live in `tests/input_yaml/`,
  - generated test outputs are targeted to `tests/output_file/`,
  - output filenames are tagged with `legacy__...` or `local__...`.
- Optional legacy parity runner:
  - `tests/parity/test_legacy_binary_runner.py` can execute the local `PlatoSim3/build/platosim` binary,
  - `tests/parity/test_structure_parity.py` compares baseline HDF5 group overlap between legacy and Python outputs,
  - enable with `RUN_LEGACY_PARITY=1` for integration-level checks.
- Performance benchmark test:
  - `tests/perf/test_perf_legacy_vs_local.py` measures legacy vs local runtime on a lightweight shared config.
  - included in pytest by default when legacy binary is available (set `RUN_LEGACY_PERF=0` to disable).
  - writes `tests/output_file/local__perf_report.json`.
  - pytest terminal summary shows `legacy_seconds`, `local_seconds`, and `local/legacy speedup`.

Quick side-by-side preview example:

```bash
python scripts/preview_hdf5_side_by_side.py \
  --legacy tests/output_file/legacy__structure_parity.hdf5 \
  --local tests/output_file/local__structure_parity.hdf5 \
  --image-index 0
```

## Deferred GPU Validation Checklist

Use this checklist when a CUDA-capable machine becomes available.

### Prerequisites

- NVIDIA GPU with compatible CUDA driver/runtime.
- Fresh virtual environment (recommended).
- CPU test suite already green in this repo.

### Setup On GPU Host

```bash
source .venv/bin/activate
pip install -e .[cuda]
```

### Validation Steps

1. Verify CuPy import and device visibility:

```bash
python -c "import cupy as cp; print(cp.__version__); print(cp.cuda.runtime.getDeviceCount())"
```

2. Run full test suite (CPU + compatibility):

```bash
python -m pytest -q
```

3. Run backend-specific smoke check (to be added under `tests/parity/`):

```bash
python -m pytest -q -k cupy
```

4. Run benchmark script(s) for CPU vs CUDA comparison:

```bash
python benchmarks/run_backend_benchmarks.py --backend numpy
python benchmarks/run_backend_benchmarks.py --backend cupy
```

### Acceptance Criteria

- CuPy backend initializes without fallback errors.
- Compatibility tests with legacy input files still pass.
- Numerical differences remain within documented tolerance.
- GPU backend shows measurable speedup in targeted kernels.

### Notes

- CUDA support remains optional; CPU backend stays the reference path.
- If a GPU test fails, open an issue with hardware/driver/CUDA/CuPy versions.

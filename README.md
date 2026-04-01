# PlatoSim Pure Python GPU

Standalone repository for a pure-Python implementation of PlatoSim with optional GPU acceleration.

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

Implement a modular simulation engine under `src/platosim_py`:
- `config`: input parsing, schema normalization, validation, legacy key support.
- `core`: simulation orchestration (`run`, exposure scheduling, effect ordering).
- `models`: physical models (platform, camera, detector, PSF, noise/effects).
- `backends`: array abstraction and backend-specific ops (`numpy`, `cupy`).
- `io`: HDF5 read/write compatibility with legacy output structure.

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
platosim_py_gpu/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── benchmarks/
├── configs/
├── docs/
├── scripts/
├── src/
│   └── platosim_py/
│       ├── __init__.py
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── numpy_backend.py
│       │   └── cupy_backend.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── compatibility.py
│       ├── core/
│       │   ├── __init__.py
│       │   └── simulation.py
│       ├── io/
│       │   ├── __init__.py
│       │   └── hdf5.py
│       └── models/
│           └── __init__.py
└── tests/
    ├── parity/
    │   └── test_input_compatibility.py
    └── unit/
        └── __init__.py
```

## Quick Start

```bash
git clone <this-repo-url>
cd platosim_py_gpu
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

With CUDA (optional):

```bash
pip install -e .[cuda]
```

## Next Implementation Milestones

- Implement strict + compatibility YAML parser in `config/compatibility.py`.
- Implement backend protocol and default NumPy backend.
- Add parity test harness that runs legacy configs and compares key outputs.

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

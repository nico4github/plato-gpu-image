# AGENTS.md

Guidance for engineers and coding agents working in this repository.

## Mission

Build a pure-Python, GPU-capable PlatoSim equivalent in a standalone repository while preserving compatibility with existing PlatoSim3 inputs.

## Non-Negotiable Constraints

- Keep this repository independent from `PlatoSim3` implementation changes.
- Preserve full compatibility with legacy PlatoSim3 YAML input files.
- Do not introduce project-owned C++ extensions.
- CPU backend must remain first-class even when CUDA is enabled.

## Compatibility Policy

When implementing any feature, maintain:
- existing YAML key paths, structure, and units,
- existing default values and effect toggles,
- equivalent seed and stochastic behavior whenever feasible,
- explicit compatibility tests using unchanged legacy input files.

If behavior cannot be matched exactly, document the delta and add a failing parity test marked with `xfail` plus rationale.

## Backend Policy

- Backend selection must be runtime-configurable.
- `numpy` backend is the reference implementation.
- `cupy` backend is optional acceleration and must not change the public API.
- Any backend-specific numerical tolerance must be documented in tests.

## Definition of Done (per feature)

- Unit tests pass.
- Compatibility/parity tests for affected behavior pass.
- CPU path works without CUDA dependencies installed.
- Documentation updated in `README.md` or `docs/`.

## Test Data And Output Rules

- Keep YAML config fixtures in `tests/input_yaml/`.
- Keep star/catalog and other non-YAML input fixtures in `tests/input_file/`.
- Generated test artifacts must go to `tests/output_file/` (never repo root or `tests/` directly).
- Use tagged filenames for generated artifacts:
  - `legacy__*` for outputs from the legacy C++ binary.
  - `local__*` for outputs from this Python implementation.
- Do not commit generated artifact files (`.hdf5`, `.log`, `.png`, perf reports); keep only fixture inputs and `.gitkeep` placeholders.

## Performance And Preview Rules

- Maintain a perf comparison test between legacy and local implementations.
- Ensure pytest terminal summary exposes perf metrics clearly:
  - `legacy_seconds`
  - `local_seconds`
  - `local/legacy speedup`
- Keep preview image artifacts for quick visual comparison in `tests/output_file/` with tagged names.
- Keep and maintain script-based inspection tools for HDF5 comparison/preview workflows.

## Suggested Work Order

1. Config parser and compatibility normalization.
2. Core simulation orchestration and deterministic ordering.
3. HDF5 writer compatibility layer.
4. PSF and detector effects kernels.
5. CUDA acceleration and benchmark-driven optimization.

## Code Style

- Use type hints on all public functions.
- Prefer small, composable modules.
- Avoid hidden global state.
- Keep random number generators explicit and injectable.

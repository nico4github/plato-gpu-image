#!/usr/bin/env python3
"""CLI entry point for running plato-gpu-image simulations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parent / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from core.simulation import Simulation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to legacy simulation YAML input file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HDF5 path. Default: output/<input-stem>.hdf5",
    )
    parser.add_argument(
        "--backend",
        choices=("numpy", "cupy"),
        default="numpy",
        help="Execution backend. Use cupy only on CUDA-enabled hosts.",
    )
    parser.add_argument(
        "--strict-core-contract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate input against the core compatibility contract.",
    )
    parser.add_argument(
        "--overwrite-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow overwriting an existing output file.",
    )
    return parser


def default_output_path(input_path: Path) -> Path:
    return Path("output") / f"{input_path.stem}.hdf5"


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_path: Path = args.input
    output_path: Path = args.output if args.output is not None else default_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sim = Simulation.from_legacy_yaml(
        input_path,
        backend=args.backend,
        output_path=output_path,
        strict_core_contract=args.strict_core_contract,
        overwrite_output=args.overwrite_output,
    )
    payload = sim.run()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


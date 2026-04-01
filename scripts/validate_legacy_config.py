"""Validate legacy PlatoSim-style YAML configuration files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from platosim_py.config.compatibility import (  # noqa: E402
    CORE_REQUIRED_PATHS,
    ConfigCompatibilityError,
    flatten_paths,
    load_core_compatible_yaml,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to legacy YAML file")
    args = parser.parse_args()

    try:
        config = load_core_compatible_yaml(args.config)
    except ConfigCompatibilityError as exc:
        print(f"INVALID: {exc}")
        return 2

    leaf_paths = flatten_paths(config, leaves_only=True)
    print(f"VALID: {args.config}")
    print(f"core_required_paths={len(CORE_REQUIRED_PATHS)}")
    print(f"leaf_paths={len(leaf_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


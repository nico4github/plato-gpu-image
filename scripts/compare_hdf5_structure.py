"""Compare structure of two HDF5 files."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def collect_paths(h5: h5py.File) -> dict[str, tuple[str, tuple[int, ...] | None]]:
    """Collect group/dataset paths and metadata."""
    paths: dict[str, tuple[str, tuple[int, ...] | None]] = {}

    def _visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        full_path = f"/{name}" if name else "/"
        if isinstance(obj, h5py.Group):
            paths[full_path] = ("group", None)
        elif isinstance(obj, h5py.Dataset):
            paths[full_path] = ("dataset", tuple(obj.shape))

    h5.visititems(_visitor)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lhs", type=Path, help="First HDF5 file")
    parser.add_argument("rhs", type=Path, help="Second HDF5 file")
    args = parser.parse_args()

    with h5py.File(args.lhs, "r") as lhs_h5, h5py.File(args.rhs, "r") as rhs_h5:
        lhs_paths = collect_paths(lhs_h5)
        rhs_paths = collect_paths(rhs_h5)

    lhs_set = set(lhs_paths)
    rhs_set = set(rhs_paths)

    common = sorted(lhs_set & rhs_set)
    only_lhs = sorted(lhs_set - rhs_set)
    only_rhs = sorted(rhs_set - lhs_set)

    print(f"common_paths={len(common)}")
    print(f"only_lhs={len(only_lhs)}")
    print(f"only_rhs={len(only_rhs)}")

    shape_mismatches = 0
    for path in common:
        lhs_info = lhs_paths[path]
        rhs_info = rhs_paths[path]
        if lhs_info != rhs_info:
            shape_mismatches += 1
            print(f"mismatch: {path} lhs={lhs_info} rhs={rhs_info}")

    print(f"metadata_mismatches={shape_mismatches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


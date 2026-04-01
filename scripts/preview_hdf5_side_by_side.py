"""Preview key HDF5 content side-by-side for legacy vs local outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _key_dataset_paths(image_index: int) -> list[str]:
    suffix = f"{image_index:07d}"
    return [
        f"/Images/image{suffix}",
        f"/SmearingMaps/smearingMap{suffix}",
        f"/BiasMapsLeft/biasMap{suffix}",
        f"/BiasMapsRight/biasMap{suffix}",
        f"/ThroughputMaps/throughputMap{suffix}",
        "/TransmissionEfficiency/transmissionEfficiency",
        "/BackgroundMap/skyBackground",
        "/Flatfield/PRNU",
        "/ACS/time",
        "/ACS/yaw",
        "/ACS/pitch",
        "/ACS/roll",
        "/ACS/platformRA",
        "/ACS/platformDec",
        "/Telescope/time",
        "/Telescope/telescopeYaw",
        "/Telescope/telescopePitch",
        "/Telescope/telescopeRoll",
        "/Telescope/telescopeRA",
        "/Telescope/telescopeDec",
        "/InputParameters/rawConfigYAML",
    ]


def _summarize_dataset(dataset: h5py.Dataset, *, max_sample: int) -> str:
    shape = tuple(dataset.shape)
    dtype = str(dataset.dtype)

    # Scalar string / bytes
    if dataset.shape == () and dataset.dtype.kind in ("S", "O", "U"):
        raw = dataset[()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        text = str(raw).replace("\n", "\\n")
        if len(text) > 80:
            text = text[:77] + "..."
        return f"shape={shape} dtype={dtype} value='{text}'"

    if dataset.dtype.kind in ("i", "u", "f", "b"):
        array = np.asarray(dataset[()])
        flat = array.reshape(-1) if array.size else array
        sample_vals = ",".join(f"{v:.3g}" for v in flat[:max_sample]) if array.size else ""
        if array.size:
            return (
                f"shape={shape} dtype={dtype} min={float(np.min(array)):.3g} "
                f"max={float(np.max(array)):.3g} mean={float(np.mean(array)):.3g} "
                f"sample=[{sample_vals}]"
            )
        return f"shape={shape} dtype={dtype} empty"

    return f"shape={shape} dtype={dtype}"


def _get_dataset_summary(handle: h5py.File, path: str, *, max_sample: int) -> str:
    if path not in handle:
        return "MISSING"
    obj = handle[path]
    if not isinstance(obj, h5py.Dataset):
        return f"NOT_DATASET ({type(obj).__name__})"
    return _summarize_dataset(obj, max_sample=max_sample)


def _group_overview(handle: h5py.File) -> set[str]:
    return set(handle.keys())


def _print_group_section(legacy_h5: h5py.File, local_h5: h5py.File) -> None:
    legacy_groups = _group_overview(legacy_h5)
    local_groups = _group_overview(local_h5)
    common = sorted(legacy_groups & local_groups)
    only_legacy = sorted(legacy_groups - local_groups)
    only_local = sorted(local_groups - legacy_groups)

    print("== Top-level Groups ==")
    print(f"common={len(common)} only_legacy={len(only_legacy)} only_local={len(only_local)}")
    if only_legacy:
        print("only_legacy:", ", ".join(only_legacy))
    if only_local:
        print("only_local:", ", ".join(only_local))
    print()


def _status(lhs: str, rhs: str) -> str:
    if lhs == "MISSING" and rhs == "MISSING":
        return "-"
    if lhs == "MISSING" or rhs == "MISSING":
        return "MISSING"
    return "OK" if lhs == rhs else "DIFF"


def _print_key_datasets_section(
    legacy_h5: h5py.File,
    local_h5: h5py.File,
    *,
    image_index: int,
    max_sample: int,
) -> None:
    print("== Key Dataset Preview ==")
    header = f"{'dataset_path':<42} | {'status':<8} | {'legacy_summary':<55} | local_summary"
    print(header)
    print("-" * len(header))
    for path in _key_dataset_paths(image_index):
        legacy_summary = _get_dataset_summary(legacy_h5, path, max_sample=max_sample)
        local_summary = _get_dataset_summary(local_h5, path, max_sample=max_sample)
        st = _status(legacy_summary, local_summary)
        print(f"{path:<42} | {st:<8} | {legacy_summary:<55} | {local_summary}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", required=True, type=Path, help="Legacy HDF5 output path")
    parser.add_argument("--local", required=True, type=Path, help="Local HDF5 output path")
    parser.add_argument(
        "--image-index",
        type=int,
        default=0,
        help="Exposure index used for image-like datasets (default: 0)",
    )
    parser.add_argument(
        "--max-sample",
        type=int,
        default=5,
        help="Max scalar values shown in summary samples (default: 5)",
    )
    args = parser.parse_args()

    with h5py.File(args.legacy, "r") as legacy_h5, h5py.File(args.local, "r") as local_h5:
        _print_group_section(legacy_h5, local_h5)
        _print_key_datasets_section(
            legacy_h5,
            local_h5,
            image_index=args.image_index,
            max_sample=args.max_sample,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


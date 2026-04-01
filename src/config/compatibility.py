"""Legacy PlatoSim3 YAML compatibility helpers.

This module provides a compatibility-focused YAML loader for legacy PlatoSim3
configuration files. It intentionally starts with non-destructive normalization
so downstream simulation logic can remain strict and predictable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


class ConfigCompatibilityError(ValueError):
    """Raised when a legacy config cannot be normalized."""


LEGACY_REQUIRED_TOP_LEVEL_SECTIONS: tuple[str, ...] = (
    "General",
    "ObservingParameters",
    "Sky",
    "Platform",
    "Telescope",
    "Camera",
    "PSF",
    "FEE",
    "CCD",
    "SubField",
    "RandomSeeds",
    "ControlHDF5Content",
)


# Legacy alias paths encountered in historical config variants.
# key: legacy path, value: canonical path.
LEGACY_ALIAS_PATHS: dict[str, str] = {
    "ObservingParameters/RApointing": "Platform/Orientation/Angles/RAPointing",
    "ObservingParameters/DecPointing": "Platform/Orientation/Angles/DecPointing",
}

# Core path contract for v1 compatibility checks.
# This is intentionally smaller than the full schema and captures the minimum
# configuration shape needed to begin deterministic simulation execution.
CORE_REQUIRED_PATHS: tuple[str, ...] = (
    "General/ProjectLocation",
    "ObservingParameters/NumExposures",
    "ObservingParameters/BeginExposureNr",
    "ObservingParameters/CycleTime",
    "ObservingParameters/StarCatalogFile",
    "Sky/SkyBackground/UseConstantSkyBackground",
    "Platform/UseJitter",
    "Platform/JitterSource",
    "Platform/Orientation/Source",
    "Telescope/GroupID",
    "Telescope/UseDrift",
    "Camera/PlateScale",
    "Camera/IncludeFieldDistortion",
    "PSF/Model",
    "FEE/Temperature",
    "CCD/Position",
    "CCD/NumRows",
    "CCD/NumColumns",
    "SubField/NumRows",
    "SubField/NumColumns",
    "RandomSeeds/ReadOutNoiseSeed",
    "ControlHDF5Content/WritePixelMaps",
)


def load_legacy_yaml(
    path: str | Path,
    *,
    required_paths: Iterable[str] | None = None,
    required_top_level_sections: Iterable[str] = LEGACY_REQUIRED_TOP_LEVEL_SECTIONS,
    apply_aliases: bool = True,
) -> dict[str, Any]:
    """Load and normalize a PlatoSim3-compatible YAML file.

    Notes:
        - Performs basic structure validation.
        - Optionally applies known legacy path aliases.
        - Validates required top-level sections by default.
        - Optionally validates a custom list of required leaf/container paths.
    """
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ConfigCompatibilityError("Top-level YAML structure must be a mapping")

    if apply_aliases:
        normalize_legacy_aliases(data)

    ensure_top_level_sections(data, required_top_level_sections)

    if required_paths is not None:
        ensure_paths_exist(data, required_paths)

    return data


def load_core_compatible_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML and enforce the v1 core required-path contract."""
    return load_legacy_yaml(path, required_paths=CORE_REQUIRED_PATHS)


def flatten_paths(tree: dict[str, Any], *, leaves_only: bool = False) -> set[str]:
    """Return all slash-separated paths in a nested mapping.

    Args:
        tree: Nested dictionary-like object.
        leaves_only: If True, include only leaf-value paths.
    """
    output: set[str] = set()

    def _walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            if prefix and not leaves_only:
                output.add(prefix)
            for key, value in node.items():
                child = f"{prefix}/{key}" if prefix else str(key)
                _walk(value, child)
            return
        if prefix:
            output.add(prefix)

    _walk(tree, "")
    return output


def ensure_top_level_sections(
    data: dict[str, Any], sections: Iterable[str] = LEGACY_REQUIRED_TOP_LEVEL_SECTIONS
) -> None:
    """Validate presence of required top-level sections."""
    missing = [section for section in sections if section not in data]
    if missing:
        raise ConfigCompatibilityError(
            "Missing required top-level section(s): " + ", ".join(missing)
        )


def ensure_paths_exist(data: dict[str, Any], required_paths: Iterable[str]) -> None:
    """Validate that all required slash-separated paths exist in `data`."""
    missing: list[str] = []
    for path in required_paths:
        if not has_path(data, path):
            missing.append(path)
    if missing:
        sample = ", ".join(missing[:8])
        if len(missing) > 8:
            sample += f", ... (+{len(missing) - 8} more)"
        raise ConfigCompatibilityError(f"Missing required configuration path(s): {sample}")


def has_path(data: dict[str, Any], path: str) -> bool:
    """Return True if slash-separated `path` exists."""
    try:
        get_path(data, path)
    except (KeyError, TypeError):
        return False
    return True


def get_path(data: dict[str, Any], path: str) -> Any:
    """Get value at slash-separated `path`.

    Raises:
        KeyError: if a path segment is missing.
        TypeError: if traversal encounters a non-mapping node.
    """
    node: Any = data
    for segment in path.split("/"):
        if not isinstance(node, dict):
            raise TypeError(
                f"Cannot traverse into non-mapping node while resolving path '{path}'"
            )
        node = node[segment]
    return node


def set_path(data: dict[str, Any], path: str, value: Any, *, overwrite: bool = False) -> None:
    """Set a slash-separated path, creating missing containers as needed."""
    segments = path.split("/")
    parent = data
    for segment in segments[:-1]:
        current = parent.get(segment)
        if current is None:
            parent[segment] = {}
            current = parent[segment]
        if not isinstance(current, dict):
            raise ConfigCompatibilityError(
                f"Path conflict while setting '{path}': '{segment}' is not a mapping"
            )
        parent = current

    leaf = segments[-1]
    if leaf in parent and not overwrite:
        return
    parent[leaf] = value


def normalize_legacy_aliases(data: dict[str, Any]) -> None:
    """Apply known legacy aliases in-place without overwriting canonical keys."""
    for legacy_path, canonical_path in LEGACY_ALIAS_PATHS.items():
        if not has_path(data, legacy_path):
            continue
        if has_path(data, canonical_path):
            continue
        value = get_path(data, legacy_path)
        set_path(data, canonical_path, value, overwrite=False)


def collect_required_paths_from_reference(
    reference_yaml: str | Path, *, leaves_only: bool = True
) -> set[str]:
    """Extract required paths from a reference YAML file.

    Useful for building parity/compatibility expectations in tests.
    """
    reference = load_legacy_yaml(reference_yaml)
    return flatten_paths(reference, leaves_only=leaves_only)

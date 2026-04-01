"""Legacy PlatoSim3 YAML compatibility helpers (skeleton)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigCompatibilityError(ValueError):
    """Raised when a legacy config cannot be normalized."""


def load_legacy_yaml(path: str | Path) -> dict[str, Any]:
    """Load a PlatoSim3-compatible YAML file.

    Notes:
        This is a skeleton implementation. Validation and normalization
        rules should be expanded as migration progresses.
    """
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigCompatibilityError("Top-level YAML structure must be a mapping")
    return data

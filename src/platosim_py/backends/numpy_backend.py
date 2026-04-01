"""NumPy backend placeholder for array operations."""

from __future__ import annotations

import numpy as np


def name() -> str:
    """Return backend name."""
    return "numpy"


def array_namespace():
    """Return the array namespace for this backend."""
    return np

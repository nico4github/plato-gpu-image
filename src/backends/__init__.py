"""Backend selection helpers."""

from __future__ import annotations

from backends import cupy_backend, numpy_backend
from backends.protocol import ArrayBackendModule

SUPPORTED_BACKENDS: tuple[str, ...] = ("numpy", "cupy")


def resolve_backend(name: str) -> ArrayBackendModule:
    """Resolve backend by name."""
    normalized = name.lower().strip()
    if normalized == "numpy":
        return numpy_backend
    if normalized == "cupy":
        return cupy_backend
    raise ValueError(f"Unsupported backend '{name}'. Supported: {', '.join(SUPPORTED_BACKENDS)}")


def backend_array_namespace(name: str):
    """Resolve backend and return its array namespace object."""
    backend = resolve_backend(name)
    return backend.array_namespace()

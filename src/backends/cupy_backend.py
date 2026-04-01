"""CuPy backend placeholder for CUDA array operations."""

from __future__ import annotations


def name() -> str:
    """Return backend name."""
    return "cupy"


def array_namespace():
    """Return the array namespace for this backend.

    Raises:
        RuntimeError: If CuPy is not available.
    """
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - depends on runtime CUDA setup
        raise RuntimeError("CuPy backend requested but CuPy is not available") from exc
    return cp

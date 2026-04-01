"""Backend protocol for array-compute runtimes."""

from __future__ import annotations

from typing import Any, Protocol


class ArrayBackendModule(Protocol):
    """Minimal protocol for backend modules."""

    def name(self) -> str:  # pragma: no cover - typing protocol signature
        """Backend identifier."""

    def array_namespace(self) -> Any:  # pragma: no cover - typing protocol signature
        """Return the array namespace (e.g., numpy or cupy)."""


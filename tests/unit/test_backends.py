from __future__ import annotations

import pytest

from platosim_py.backends import backend_array_namespace, resolve_backend


def test_resolve_backend_numpy() -> None:
    backend = resolve_backend("numpy")
    assert backend.name() == "numpy"


def test_backend_array_namespace_numpy() -> None:
    xp = backend_array_namespace("numpy")
    arr = xp.array([1, 2, 3], dtype=xp.int32)
    assert int(arr.sum()) == 6


def test_resolve_backend_invalid() -> None:
    with pytest.raises(ValueError):
        resolve_backend("invalid")


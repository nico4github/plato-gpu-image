"""Minimal backend benchmark harness.

This benchmark intentionally focuses on small synthetic operations so it can run
on CPU-only developer machines and on future GPU hosts with the same interface.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Allow running as a local script before package installation.
REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from backends import backend_array_namespace


@dataclass(slots=True)
class BenchmarkResult:
    backend: str
    operation: str
    size: int
    seconds: float


def benchmark_fft(backend: str, size: int) -> BenchmarkResult:
    xp = backend_array_namespace(backend)
    grid = xp.random.random((size, size)).astype(xp.float32)

    start = time.perf_counter()
    fft = xp.fft.fft2(grid)
    _ = xp.fft.ifft2(fft)
    if backend == "cupy":
        xp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        backend=backend,
        operation="fft2+ifft2",
        size=size,
        seconds=elapsed,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="numpy", choices=("numpy", "cupy"))
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()

    result = benchmark_fft(args.backend, args.size)
    print(
        f"backend={result.backend} operation={result.operation} "
        f"size={result.size} seconds={result.seconds:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

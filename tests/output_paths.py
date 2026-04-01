from __future__ import annotations

from pathlib import Path


def output_root() -> Path:
    return Path(__file__).resolve().parents[1] / "tests" / "output_file"


def ensure_output_root() -> Path:
    root = output_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def tagged_output_file(tag: str, name: str) -> Path:
    root = ensure_output_root()
    safe_tag = tag.strip().lower().replace(" ", "_")
    safe_name = name.strip()
    return root / f"{safe_tag}__{safe_name}"


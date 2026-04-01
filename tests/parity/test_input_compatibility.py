from __future__ import annotations

from pathlib import Path

import pytest

from platosim_py.config.compatibility import load_legacy_yaml


def test_legacy_yaml_top_level_mapping(tmp_path: Path) -> None:
    config_file = tmp_path / "input.yaml"
    config_file.write_text("A: 1\nB:\n  C: 2\n", encoding="utf-8")
    data = load_legacy_yaml(config_file)
    assert data["A"] == 1


@pytest.mark.xfail(reason="Full legacy key-semantic compatibility not yet implemented")
def test_full_platosim3_input_compatibility() -> None:
    assert False

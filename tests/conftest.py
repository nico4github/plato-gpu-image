from __future__ import annotations

import json
from pathlib import Path


def _perf_reports() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "tests" / "output_file"
    if not out_dir.exists():
        return []
    return sorted(out_dir.glob("local__perf_report*.json"))


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    reports = _perf_reports()
    terminalreporter.write_sep("=", "Performance Summary")
    if not reports:
        terminalreporter.write_line("No performance report artifacts found.")
        terminalreporter.write_line(
            "Run pytest with legacy perf enabled (RUN_LEGACY_PERF=1) to generate one."
        )
        return

    for report in reports:
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - terminal reporting guard
            terminalreporter.write_line(f"{report.name}: unable to parse ({exc})")
            continue

        benchmark = payload.get("benchmark", report.name)
        legacy_s = payload.get("legacy_seconds")
        local_s = payload.get("local_seconds")
        speedup = payload.get("speedup_local_over_legacy")
        legacy_preview = payload.get("legacy_preview")
        local_preview = payload.get("local_preview")
        terminalreporter.write_line(f"{benchmark}:")
        terminalreporter.write_line(f"  legacy_seconds = {legacy_s:.4f}")
        terminalreporter.write_line(f"  local_seconds  = {local_s:.4f}")
        terminalreporter.write_line(f"  local/legacy speedup = {speedup:.4f}x")
        if legacy_preview:
            terminalreporter.write_line(f"  legacy_preview = {legacy_preview}")
        if local_preview:
            terminalreporter.write_line(f"  local_preview  = {local_preview}")
        terminalreporter.write_line(f"  report = {report}")

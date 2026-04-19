#!/usr/bin/env python
# coding: utf-8
"""
tests/test_log_monitor.py
Run: python -X utf8 tests/test_log_monitor.py
"""

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot.log_monitor import format_alert, run_log_monitor


PASS = "✅"
FAIL = "❌"
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {PASS if ok else FAIL} {name}" + (f"\n       > {detail}" if detail and not ok else ""))
    results.append(ok)


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    logs = root / "logs"
    logs.mkdir()
    state = logs / "log_monitor_state.json"
    summary = logs / "log_monitor_summary.log"
    log_file = logs / "send_reports_20260419.log"

    log_file.write_text("2026-04-19 INFO boot ok\n", encoding="utf-8")
    first = run_log_monitor(
        logs,
        state,
        summary,
        now=datetime(2026, 4, 19, 10, 0, tzinfo=timezone.utc),
    )
    check("first run baselines existing logs", first["initialized"] is True)
    check("first run no alert on old content", first["errors_found"] == 0)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("2026-04-19 ERROR pipeline failed\n")

    second = run_log_monitor(
        logs,
        state,
        summary,
        now=datetime(2026, 4, 19, 12, 0, tzinfo=timezone.utc),
    )
    check("new ERROR detected", second["errors_found"] == 1, str(second))
    check("state json written", json.loads(state.read_text(encoding="utf-8"))["status"] == "alert")
    check("summary log written", "ALERT" in summary.read_text(encoding="utf-8"))
    check("alert text mentions state", "log_monitor_state.json" in format_alert(second))

    third = run_log_monitor(
        logs,
        state,
        summary,
        now=datetime(2026, 4, 19, 14, 0, tzinfo=timezone.utc),
    )
    check("same ERROR is not repeated", third["errors_found"] == 0)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("2026-04-19 ERROR [TEST] expected test-only error\n")
    fourth = run_log_monitor(
        logs,
        state,
        summary,
        now=datetime(2026, 4, 19, 16, 0, tzinfo=timezone.utc),
    )
    check("[TEST] lines ignored", fourth["errors_found"] == 0)

passed = sum(1 for ok in results if ok)
failed = len(results) - passed
print(f"\nИТОГ: {passed}/{len(results)} прошло, {failed} упало")
if failed:
    raise SystemExit(1)

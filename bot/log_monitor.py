#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot/log_monitor.py

Lightweight log monitor for the Telegram bot scheduler.

It stores per-file read offsets so the same error is not reported repeatedly.
On the first run it baselines current log ends and starts reporting only new
lines after that point.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

__VERSION__ = "v1.0.2/24.04.2026"

DEFAULT_PATTERNS = (
    re.compile(r"\bERROR\b", re.IGNORECASE),
    re.compile(r"\bCRITICAL\b", re.IGNORECASE),
    re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"PermissionError", re.IGNORECASE),
    re.compile(r"Unhandled exception", re.IGNORECASE),
    re.compile(r"Job .* raised", re.IGNORECASE),
)

DEFAULT_IGNORE_PATTERNS = (
    re.compile(r"\[TEST\]", re.IGNORECASE),
)

_LEVEL_PREFIX_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:,\d+)?\s*,\s*(INFO|WARNING|ERROR|CRITICAL|DEBUG)\b"
)


@dataclass
class LogFinding:
    file: str
    line: str
    pattern: str


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass


def _append_summary(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def _iter_log_files(logs_dir: Path, patterns: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend(logs_dir.glob(pattern))
    return sorted({p for p in files if p.is_file()}, key=lambda p: p.name)


def _scan_text(
    file_name: str,
    text: str,
    patterns: Iterable[re.Pattern[str]],
    ignore_patterns: Iterable[re.Pattern[str]],
    max_findings: int,
) -> List[LogFinding]:
    findings: List[LogFinding] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(p.search(line) for p in ignore_patterns):
            continue
        matched = next((p.pattern for p in patterns if p.search(line)), "")
        if not matched:
            continue
        level_match = _LEVEL_PREFIX_RE.match(line)
        if level_match and matched in (r"\bERROR\b", r"\bCRITICAL\b"):
            actual_level = level_match.group(1).upper()
            if actual_level not in ("ERROR", "CRITICAL"):
                continue
        findings.append(LogFinding(file=file_name, line=line[-500:], pattern=matched))
        if len(findings) >= max_findings:
            break
    return findings


def run_log_monitor(
    logs_dir: Path,
    state_path: Path,
    summary_path: Path,
    *,
    now: Optional[datetime] = None,
    file_globs: Iterable[str] = ("*.log",),
    max_bytes_per_file: int = 512 * 1024,
    max_findings: int = 20,
    baseline_if_new: bool = True,
) -> Dict[str, Any]:
    """
    Scan new log content and update state.

    Returns a compact dict suitable for state JSON and Telegram alerts.
    """
    now = now or datetime.now().astimezone()
    state = _read_json(state_path)
    positions: Dict[str, int] = {
        str(k): int(v)
        for k, v in (state.get("positions") or {}).items()
        if isinstance(v, int) or str(v).isdigit()
    }
    files = _iter_log_files(logs_dir, file_globs) if logs_dir.exists() else []
    files = [p for p in files if p.resolve() != summary_path.resolve()]
    all_findings: List[LogFinding] = []
    new_positions: Dict[str, int] = dict(positions)
    initialized = False

    for path in files:
        key = path.name
        try:
            size = path.stat().st_size
        except OSError:
            continue

        previous = positions.get(key)
        if previous is None and baseline_if_new:
            new_positions[key] = size
            initialized = True
            continue

        start = int(previous or 0)
        if start > size:
            start = 0
        if size - start > max_bytes_per_file:
            start = max(0, size - max_bytes_per_file)

        try:
            with open(path, "rb") as f:
                f.seek(start)
                chunk = f.read(max_bytes_per_file)
        except OSError:
            continue

        text = chunk.decode("utf-8", errors="replace")
        remaining = max_findings - len(all_findings)
        if remaining > 0:
            all_findings.extend(
                _scan_text(
                    key,
                    text,
                    DEFAULT_PATTERNS,
                    DEFAULT_IGNORE_PATTERNS,
                    remaining,
                )
            )
        new_positions[key] = size

    status = "alert" if all_findings else "ok"
    result = {
        "last_run": now.isoformat(),
        "status": status,
        "initialized": initialized,
        "files_checked": len(files),
        "errors_found": len(all_findings),
        "findings": [finding.__dict__ for finding in all_findings],
        "positions": new_positions,
    }
    _atomic_write_json(state_path, result)

    summary_line = (
        f"{now.isoformat()} {status.upper()} checked={len(files)} "
        f"errors={len(all_findings)} initialized={int(initialized)}"
    )
    if all_findings:
        first = all_findings[0]
        summary_line += f" first={first.file}: {first.line[:180]}"
    _append_summary(summary_path, summary_line)
    return result


def format_alert(result: Dict[str, Any]) -> str:
    findings = result.get("findings") or []
    lines = [
        "🚨 LOG MONITOR",
        f"Проверено файлов: {result.get('files_checked', 0)}",
        f"Новых проблем: {result.get('errors_found', 0)}",
        "",
    ]
    for item in findings[:10]:
        lines.append(f"• {item.get('file', '?')}: {item.get('line', '')[:350]}")
    if len(findings) > 10:
        lines.append(f"… ещё {len(findings) - 10}")
    # v1.0.1 (F-004): подсказка для типовых Telegram-ошибок доставки
    _joined_lines = " ".join(str(item.get("line", "")) for item in findings)
    if "Chat not found" in _joined_lines:
        lines.append("")
        lines.append("⚠ Chat not found: проверьте chat_id в config/managers.json.")
    lines.append("")
    lines.append("Сводка: logs/log_monitor_summary.log")
    lines.append("State: logs/log_monitor_state.json")
    return "\n".join(lines)

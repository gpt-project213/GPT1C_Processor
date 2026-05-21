#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot/log_insights.py
Утилиты для digest по runtime-логам и единой timeline по клиенту.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_LEVEL_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), (?P<level>WARNING|ERROR|CRITICAL) (?P<msg>.*)$"
)
_SYSTEM_RE = re.compile(r"\[(BOT|CRM|COLLECTOR|PIPELINE|STATE|INTEGRATION|STOP_CONTROL)\]")
_DIGEST_SYSTEMS = ("BOT", "CRM", "COLLECTOR", "PIPELINE", "STATE", "INTEGRATION", "STOP_CONTROL", "UNSCOPED")


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize_errors_by_system(
    logs_dir: Path,
    *,
    now: datetime | None = None,
    hours: int = 12,
) -> dict[str, dict[str, int]]:
    now = now or datetime.now()
    since = now - timedelta(hours=hours)
    counts: dict[str, dict[str, int]] = {
        system: {"WARNING": 0, "ERROR": 0, "CRITICAL": 0}
        for system in _DIGEST_SYSTEMS
    }

    for log_path in sorted(logs_dir.glob("*.log")):
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _LEVEL_RE.match(line)
            if not match:
                continue
            try:
                ts = datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if ts < since or ts > now:
                continue
            level = match.group("level")
            msg = match.group("msg")
            system_match = _SYSTEM_RE.search(msg)
            system = system_match.group(1) if system_match else "UNSCOPED"
            counts[system][level] += 1
    return counts


def format_error_digest(
    counts: dict[str, dict[str, int]],
    *,
    now: datetime | None = None,
    hours: int = 12,
) -> str:
    now = now or datetime.now()
    since = now - timedelta(hours=hours)
    lines = [
        "🌅 <b>Ночной digest ошибок</b>",
        f"Период: {since.strftime('%d.%m %H:%M')} - {now.strftime('%d.%m %H:%M')}",
        "",
    ]
    total = 0
    for system in _DIGEST_SYSTEMS:
        stats = counts.get(system, {})
        warn = int(stats.get("WARNING", 0))
        err = int(stats.get("ERROR", 0))
        crit = int(stats.get("CRITICAL", 0))
        total += warn + err + crit
        if warn or err or crit:
            lines.append(f"{system}: WARNING {warn} | ERROR {err} | CRITICAL {crit}")
    if total == 0:
        lines.append("За период WARNING/ERROR/CRITICAL не найдено.")
    return "\n".join(lines)


def read_client_timeline(
    client_key: str,
    *,
    crm_path: Path,
    collector_path: Path,
    limit: int = 40,
) -> list[dict[str, Any]]:
    target = (client_key or "").strip()
    if not target:
        return []

    merged: list[dict[str, Any]] = []
    for rec in _iter_jsonl(crm_path):
        if rec.get("client_key") == target:
            merged.append({"source": "CRM", **rec})
    for rec in _iter_jsonl(collector_path):
        if rec.get("name") == target:
            merged.append({"source": "COLLECTOR", **rec})

    merged.sort(key=lambda row: row.get("ts", ""))
    return merged[-limit:] if len(merged) > limit else merged


def _record_suffix(rec: dict[str, Any]) -> str:
    details: list[str] = []
    for key in ("manager", "claimer", "reason", "batch_id", "amount", "days_overdue", "phone_masked", "notified_count"):
        value = rec.get(key)
        if value not in (None, "", []):
            details.append(f"{key}={value}")
    for key in ("message_text", "text"):
        value = rec.get(key)
        if value:
            details.append(f"{key}={str(value)[:80]}")
            break
    return f" | {', '.join(details)}" if details else ""


def format_client_timeline(client_key: str, records: list[dict[str, Any]]) -> str:
    if not records:
        return f"📭 По клиенту <b>{client_key}</b> событий не найдено."

    lines = [f"🧭 <b>Timeline: {client_key}</b>", ""]
    for rec in records[-20:]:
        ts = rec.get("ts", "")
        try:
            dt = datetime.fromisoformat(ts)
            ts_label = dt.strftime("%d.%m %H:%M")
        except ValueError:
            ts_label = ts[:16]
        source = rec.get("source", "?")
        event = rec.get("event", "?")
        lines.append(f"{ts_label} [{source}] {event}{_record_suffix(rec)}")
    text = "\n".join(lines)
    if len(text) > 3900:
        text = "…\n" + text[-3900:]
    return text

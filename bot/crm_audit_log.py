#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot/crm_audit_log.py
Append-only JSONL audit trail для CRM-процессов.

Версия: 1.0.0 (2026-04-29)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
_TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))
_ROOT = Path(__file__).resolve().parent.parent
_AUDIT_PATH = _ROOT / "logs" / "crm_audit.jsonl"
_LOCK = threading.Lock()


def audit(event: str, **kwargs: Any) -> None:
    record = {
        "ts": datetime.now(tz=_TZ).isoformat(),
        "event": event,
        **kwargs,
    }
    try:
        _AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _LOCK:
            with open(_AUDIT_PATH, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as exc:
        logger.warning("[crm_audit_log] ошибка записи события %s: %s", event, exc)


def read_recent(n: int = 50) -> list[dict[str, Any]]:
    if not _AUDIT_PATH.exists():
        return []
    try:
        lines = _AUDIT_PATH.read_text(encoding="utf-8").splitlines()
        tail = lines[-n:] if len(lines) > n else lines
        return [json.loads(line) for line in tail if line.strip()]
    except Exception as exc:
        logger.warning("[crm_audit_log] ошибка чтения: %s", exc)
        return []

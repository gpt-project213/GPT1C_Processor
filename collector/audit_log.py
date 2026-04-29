#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/audit_log.py
Append-only JSONL-журнал событий коллектора.

Версия: 1.0.0 (2026-04-29)

Назначение:
  Единая точка записи всех observable-событий коллектора.
  Используется для расследований, отладки и проверки что конкретный
  клиент получил/не получил сообщение и по какой причине.

Формат файла: logs/collector_audit.jsonl
  Каждая строка — JSON-объект. Файл никогда не перезаписывается, только дополняется.

Пример записей:
  {"ts":"2026-04-29T17:05:12+05:00","event":"wa_sent","name":"ТОО Ромашка","amount":150000,...}
  {"ts":"2026-04-29T17:05:14+05:00","event":"wa_skipped","name":"ТОО Ромашка","reason":"suppress",...}
  {"ts":"2026-04-29T17:05:15+05:00","event":"batch_approved","batch_id":"b_001","clients":3}

Использование:
  from collector.audit_log import audit
  audit("wa_sent", name="ТОО Ромашка", amount=150000, phone_masked="7700***567")
  audit("wa_skipped", name="ТОО Ромашка", reason="payment_hold")
  audit("batch_approved", batch_id="b_001", clients=3)
"""

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
_AUDIT_PATH = _ROOT / "logs" / "collector_audit.jsonl"
_write_lock = threading.Lock()


def audit(event: str, **kwargs: Any) -> None:
    """Записывает событие в JSONL-журнал.

    Ошибки записи логируются но не пробрасываются — audit не должен ломать основной поток.

    Args:
        event: Код события. Рекомендуемые значения:
               wa_sent, wa_skipped, suppress_set, suppress_cleared,
               batch_created, batch_approved, batch_sent, batch_partially_sent,
               escalation, no_movement_asked, payment_hold_confirmed, payment_hold_cleared
        **kwargs: Произвольные поля записи (name, amount, reason, batch_id, ...).
    """
    record: dict[str, Any] = {
        "ts": datetime.now(tz=_TZ).isoformat(),
        "event": event,
        **kwargs,
    }
    try:
        _AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _write_lock:
            with open(_AUDIT_PATH, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as exc:
        logger.warning("[audit_log] ошибка записи события %s: %s", event, exc)


def read_recent(n: int = 50) -> list[dict[str, Any]]:
    """Возвращает последние n записей из журнала (для диагностики)."""
    if not _AUDIT_PATH.exists():
        return []
    try:
        lines = _AUDIT_PATH.read_text(encoding="utf-8").splitlines()
        tail = lines[-n:] if len(lines) > n else lines
        return [json.loads(line) for line in tail if line.strip()]
    except Exception as exc:
        logger.warning("[audit_log] ошибка чтения: %s", exc)
        return []


def read_for_client(name: str, limit: int = 20) -> list[dict[str, Any]]:
    """Возвращает последние события для конкретного клиента."""
    if not _AUDIT_PATH.exists():
        return []
    try:
        results = []
        lines = _AUDIT_PATH.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("name") == name:
                    results.append(rec)
                    if len(results) >= limit:
                        break
            except json.JSONDecodeError:
                continue
        return list(reversed(results))
    except Exception as exc:
        logger.warning("[audit_log] ошибка чтения для клиента %s: %s", name, exc)
        return []

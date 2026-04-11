#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/dialog_store.py
JSON-хранилище активных диалогов менеджеров с AI Коллектором.

Файл: logs/collector_dialogs.json
Ключ: str(manager_chat_id)
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from zoneinfo import ZoneInfo

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env",
            encoding="utf-8-sig", override=False)

TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
DIALOGS_PATH = _ROOT / "logs" / "collector_dialogs.json"
DIALOG_TTL_HOURS = float(os.getenv("DIALOG_EXPIRE_HOURS", "48"))

# Состояния диалога
STATE_AWAITING_CONFIRM              = "AWAITING_CONFIRM"
STATE_AWAITING_DATA                 = "AWAITING_DATA"
STATE_AWAITING_REJECTION_REASON     = "AWAITING_REJECTION_REASON"
STATE_AWAITING_DATA_CONFIRM         = "AWAITING_DATA_CONFIRM"
STATE_REJECTED_PENDING_ADMIN        = "REJECTED_PENDING_ADMIN"
STATE_DEADLINE_SET                  = "DEADLINE_SET"
STATE_CONFIRMED                     = "CONFIRMED"
STATE_DONE                          = "DONE"
STATE_AWAITING_MANAGER_EXPLANATION  = "AWAITING_MANAGER_EXPLANATION"
STATE_AWAITING_NAME_TEXT            = "AWAITING_NAME_TEXT"
STATE_AWAITING_PHONE_TEXT           = "AWAITING_PHONE_TEXT"

PENDING_STATES = {
    STATE_AWAITING_CONFIRM,
    STATE_AWAITING_DATA,
    STATE_AWAITING_REJECTION_REASON,
    STATE_AWAITING_DATA_CONFIRM,
    STATE_REJECTED_PENDING_ADMIN,
    STATE_DEADLINE_SET,
    STATE_AWAITING_MANAGER_EXPLANATION,
    STATE_AWAITING_NAME_TEXT,
    STATE_AWAITING_PHONE_TEXT,
}

TERMINAL_STATES = {STATE_CONFIRMED, STATE_DONE}


def _now_iso() -> str:
    """Текущее время в ISO формате (Asia/Almaty)."""
    return datetime.now(TZ).isoformat()


def _cleanup_stale_dialogs(dialogs: Dict[str, Any]) -> bool:
    """Переводит просроченные pending-диалоги в DONE с явной записью в лог."""
    now = datetime.now(TZ)
    changed = False
    for key, dialog in dialogs.items():
        if not isinstance(dialog, dict):
            continue
        state = dialog.get("state")
        if state not in PENDING_STATES:
            continue
        ts_raw = dialog.get("control_deadline") or dialog.get("last_reminded") or dialog.get("created")
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(ts_raw)
        except (TypeError, ValueError):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=TZ)
        age_hours = (now - ts).total_seconds() / 3600
        if age_hours <= DIALOG_TTL_HOURS:
            continue
        logger.warning(
            "[%s] stale dialog marked for control after %.1f h (state=%s, manager_chat_id=%s)",
            dialog.get("client_name"),
            age_hours,
            state,
            key,
        )
        dialogs[key]["control_deadline"] = now.isoformat()
        changed = True
    return changed


def load_dialogs() -> Dict[str, Any]:
    """Загружает все диалоги из JSON-файла."""
    if not DIALOGS_PATH.exists():
        return {}
    try:
        with open(DIALOGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if _cleanup_stale_dialogs(data):
                save_dialogs(data)
            return data
        return {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_dialogs(dialogs: Dict[str, Any]) -> None:
    """Атомарно сохраняет диалоги в JSON-файл через tempfile."""
    DIALOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(DIALOGS_PATH.parent),
        suffix=".tmp",
        prefix="collector_dialogs_",
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(dialogs, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, str(DIALOGS_PATH))
    except (OSError, TypeError, ValueError):
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def get_dialog(manager_chat_id: int) -> Optional[Dict[str, Any]]:
    """Возвращает диалог менеджера или None."""
    dialogs = load_dialogs()
    return dialogs.get(str(manager_chat_id))


def set_dialog(manager_chat_id: int, dialog: Dict[str, Any]) -> None:
    """Сохраняет диалог менеджера (полная замена)."""
    dialogs = load_dialogs()
    dialogs[str(manager_chat_id)] = dialog
    save_dialogs(dialogs)


def update_dialog(manager_chat_id: int, **kwargs: Any) -> None:
    """Частичное обновление полей диалога менеджера."""
    dialogs = load_dialogs()
    key = str(manager_chat_id)
    if key not in dialogs:
        return
    dialogs[key].update(kwargs)
    save_dialogs(dialogs)


def remove_dialog(manager_chat_id: int) -> None:
    """Удаляет диалог менеджера."""
    dialogs = load_dialogs()
    key = str(manager_chat_id)
    if key in dialogs:
        del dialogs[key]
        save_dialogs(dialogs)


def new_dialog(
    manager_chat_id: int,
    manager_name: str,
    client_name: str,
    level: int,
    days: int,
    amount: float,
    current_contact: Dict[str, Any],
) -> Dict[str, Any]:
    """Создаёт новую запись диалога и сохраняет её."""
    now = _now_iso()
    dialog: Dict[str, Any] = {
        "client_name":                client_name,
        "manager_name":               manager_name,
        "manager_chat_id":            manager_chat_id,
        "level":                      level,
        "days":                       days,
        "amount":                     amount,
        "current_contact":            current_contact,
        "proposed_contact":           None,
        "state":                      STATE_AWAITING_CONFIRM,
        "rejection_reason":           None,
        "deadline":                   None,
        "remind_count":               0,
        "last_reminded":              now,
        "created":                    now,
        "message_id":                 None,
        # Подтверждение имени/телефона в текущем диалоге
        "name_confirmed":             False,
        "phone_confirmed":            False,
        "awaiting_name_text":         False,
        "awaiting_phone_text":        False,
        # Контроль администратора
        "control_deadline":           None,
        "control_extensions":         0,
        # Ожидание объяснения менеджера
        "awaiting_manager_explanation": False,
    }
    set_dialog(manager_chat_id, dialog)
    return dialog


def get_all_pending() -> List[Dict[str, Any]]:
    """Возвращает все диалоги в незавершённых состояниях."""
    dialogs = load_dialogs()
    result = []
    for dialog in dialogs.values():
        if isinstance(dialog, dict) and dialog.get("state") in PENDING_STATES:
            result.append(dialog)
    return result

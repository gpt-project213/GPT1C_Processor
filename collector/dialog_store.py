#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/dialog_store.py
JSON-хранилище активных диалогов менеджеров с AI Коллектором.

Файл: logs/collector_dialogs.json
Формат (v2):
  {
    "dialogs": {
      "dlg_<stamp>_<hex8>": { "dialog_id": "...", "manager_chat_id": 123, ... }
    },
    "active_by_chat": {
      "123": ["dlg_<stamp>_<hex8>"]
    }
  }

Backward compat: старый плоский формат {str(manager_chat_id): dialog} мигрируется
автоматически при первом чтении.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from dotenv import load_dotenv
from zoneinfo import ZoneInfo

from collector.logging_utils import get_collector_logger

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env",
            encoding="utf-8-sig", override=False)

TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))
logger = get_collector_logger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
DIALOGS_PATH = _ROOT / "logs" / "collector_dialogs.json"
DIALOG_TTL_HOURS = float(os.getenv("DIALOG_EXPIRE_HOURS", "48"))

# ─── Состояния диалога ────────────────────────────────────────────────────────

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

TEXT_AWAITING_STATES = {
    STATE_AWAITING_DATA,
    STATE_AWAITING_REJECTION_REASON,
    STATE_AWAITING_DATA_CONFIRM,
    STATE_AWAITING_MANAGER_EXPLANATION,
    STATE_AWAITING_NAME_TEXT,
    STATE_AWAITING_PHONE_TEXT,
}

TERMINAL_STATES = {STATE_CONFIRMED, STATE_DONE}

_EMPTY_CONTAINER: Dict[str, Any] = {"dialogs": {}, "active_by_chat": {}}


# ─── Внутренние хелперы ───────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(TZ).isoformat()


def _new_dialog_id() -> str:
    stamp = datetime.now(TZ).strftime("%Y%m%d%H%M%S")
    return f"dlg_{stamp}_{uuid4().hex[:8]}"


def _is_old_format(data: Dict[str, Any]) -> bool:
    """Плоский формат: ключи — числовые строки (manager_chat_id), значения — dict-диалоги."""
    if "dialogs" in data or "active_by_chat" in data:
        return False
    for key, val in data.items():
        if not isinstance(val, dict):
            return False
        try:
            int(key)
        except (ValueError, TypeError):
            return False
    return True


def _migrate_old_format(old_data: Dict[str, Any]) -> Dict[str, Any]:
    """Конвертирует старый плоский dict в новый контейнер. Логирует один раз."""
    dialogs: Dict[str, Any] = {}
    active_by_chat: Dict[str, List[str]] = {}
    count = 0
    for chat_key, dialog in old_data.items():
        if not isinstance(dialog, dict):
            continue
        dialog_id = _new_dialog_id()
        dialog["dialog_id"] = dialog_id
        if "manager_chat_id" not in dialog:
            try:
                dialog["manager_chat_id"] = int(chat_key)
            except (ValueError, TypeError):
                continue
        dialogs[dialog_id] = dialog
        chat_str = str(dialog["manager_chat_id"])
        active_by_chat.setdefault(chat_str, []).append(dialog_id)
        count += 1
    if count:
        logger.info("dialog_store: migrated %d dialog(s) from old format to v2", count)
    return {"dialogs": dialogs, "active_by_chat": active_by_chat}


def _cleanup_stale_dialogs(dialogs: Dict[str, Any]) -> bool:
    """Помечает просроченные pending-диалоги. Принимает внутренний dict dialogs, не контейнер."""
    now = datetime.now(TZ)
    changed = False
    for dialog_id, dialog in dialogs.items():
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
            "[%s] stale dialog after %.1f h (state=%s, dialog_id=%s, manager_chat_id=%s)",
            dialog.get("client_name"),
            age_hours,
            state,
            dialog_id,
            dialog.get("manager_chat_id"),
        )
        dialogs[dialog_id]["control_deadline"] = now.isoformat()
        changed = True
    return changed


def _load_container() -> Dict[str, Any]:
    """Читает файл, мигрирует старый формат, запускает stale cleanup."""
    if not DIALOGS_PATH.exists():
        return {"dialogs": {}, "active_by_chat": {}}
    try:
        with open(DIALOGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"dialogs": {}, "active_by_chat": {}}
    if not isinstance(data, dict):
        return {"dialogs": {}, "active_by_chat": {}}

    if _is_old_format(data):
        container = _migrate_old_format(data)
        _save_container(container)
        return container

    container = data
    container.setdefault("dialogs", {})
    container.setdefault("active_by_chat", {})

    if _cleanup_stale_dialogs(container["dialogs"]):
        _save_container(container)
    return container


def _save_container(container: Dict[str, Any]) -> None:
    """Атомарная запись полного контейнера."""
    DIALOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(DIALOGS_PATH.parent),
        suffix=".tmp",
        prefix="collector_dialogs_",
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(container, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, str(DIALOGS_PATH))
    except (OSError, TypeError, ValueError):
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─── Публичный API ────────────────────────────────────────────────────────────

def load_dialogs() -> Dict[str, Any]:
    """Возвращает полный контейнер {"dialogs": {...}, "active_by_chat": {...}}."""
    return _load_container()


def save_dialogs(container: Dict[str, Any]) -> None:
    """Сохраняет полный контейнер."""
    _save_container(container)


def get_dialog(dialog_id: str) -> Optional[Dict[str, Any]]:
    """Возвращает диалог по dialog_id или None."""
    container = _load_container()
    return container["dialogs"].get(dialog_id)


def set_dialog(dialog_id: str, dialog: Dict[str, Any]) -> None:
    """Полная замена диалога по dialog_id."""
    container = _load_container()
    container["dialogs"][dialog_id] = dialog
    _save_container(container)


def update_dialog(dialog_id: str, **kwargs: Any) -> None:
    """Частичное обновление полей диалога по dialog_id."""
    container = _load_container()
    if dialog_id not in container["dialogs"]:
        return
    container["dialogs"][dialog_id].update(kwargs)
    _save_container(container)


def remove_dialog(dialog_id: str) -> None:
    """Удаляет диалог и чистит индекс active_by_chat."""
    container = _load_container()
    dialog = container["dialogs"].pop(dialog_id, None)
    if dialog is None:
        return
    chat_str = str(dialog.get("manager_chat_id", ""))
    ids = container["active_by_chat"].get(chat_str, [])
    ids = [i for i in ids if i != dialog_id]
    if ids:
        container["active_by_chat"][chat_str] = ids
    else:
        container["active_by_chat"].pop(chat_str, None)
    _save_container(container)


# ─── Chat-level helpers ───────────────────────────────────────────────────────

def get_active_dialog_ids(manager_chat_id: int) -> List[str]:
    """Возвращает список dialog_id для данного chat_id (только существующие диалоги)."""
    container = _load_container()
    ids = container["active_by_chat"].get(str(manager_chat_id), [])
    existing = container["dialogs"]
    return [d for d in ids if d in existing]


def get_latest_active_dialog(manager_chat_id: int) -> Optional[Dict[str, Any]]:
    """Возвращает последний (по created) активный диалог менеджера или None."""
    container = _load_container()
    ids = container["active_by_chat"].get(str(manager_chat_id), [])
    candidates = [
        container["dialogs"][d]
        for d in ids
        if d in container["dialogs"]
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.get("created", ""))


def get_text_target_dialog(manager_chat_id: int) -> Optional[Dict[str, Any]]:
    """
    Возвращает единственный диалог менеджера, ожидающий текстового ввода.
    Возвращает None если таких диалогов нет или их больше одного (ambiguity).
    """
    container = _load_container()
    ids = container["active_by_chat"].get(str(manager_chat_id), [])
    waiting = [
        container["dialogs"][d]
        for d in ids
        if d in container["dialogs"]
        and container["dialogs"][d].get("state") in TEXT_AWAITING_STATES
    ]
    return waiting[0] if len(waiting) == 1 else None


def link_dialog_to_chat(manager_chat_id: int, dialog_id: str) -> None:
    """Добавляет dialog_id в индекс active_by_chat для manager_chat_id."""
    container = _load_container()
    chat_str = str(manager_chat_id)
    ids = container["active_by_chat"].setdefault(chat_str, [])
    if dialog_id not in ids:
        ids.append(dialog_id)
    _save_container(container)


def unlink_dialog_from_chat(manager_chat_id: int, dialog_id: str) -> None:
    """Убирает dialog_id из индекса active_by_chat."""
    container = _load_container()
    chat_str = str(manager_chat_id)
    ids = container["active_by_chat"].get(chat_str, [])
    ids = [i for i in ids if i != dialog_id]
    if ids:
        container["active_by_chat"][chat_str] = ids
    else:
        container["active_by_chat"].pop(chat_str, None)
    _save_container(container)


# ─── Создание нового диалога ──────────────────────────────────────────────────

def new_dialog(
    manager_chat_id: int,
    manager_name: str,
    client_name: str,
    level: int,
    days: int,
    amount: float,
    current_contact: Dict[str, Any],
) -> Dict[str, Any]:
    """Создаёт диалог с уникальным dialog_id, сохраняет и индексирует по chat_id."""
    dialog_id = _new_dialog_id()
    now = _now_iso()
    dialog: Dict[str, Any] = {
        "dialog_id":                  dialog_id,
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
        "name_confirmed":             False,
        "phone_confirmed":            False,
        "awaiting_name_text":         False,
        "awaiting_phone_text":        False,
        "control_deadline":           None,
        "control_extensions":         0,
        "awaiting_manager_explanation": False,
    }
    container = _load_container()
    container["dialogs"][dialog_id] = dialog
    chat_str = str(manager_chat_id)
    ids = container["active_by_chat"].setdefault(chat_str, [])
    if dialog_id not in ids:
        ids.append(dialog_id)
    _save_container(container)
    return dialog


# ─── Bulk helpers ─────────────────────────────────────────────────────────────

def get_all_pending() -> List[Dict[str, Any]]:
    """Возвращает все диалоги в незавершённых состояниях."""
    container = _load_container()
    return [
        d for d in container["dialogs"].values()
        if isinstance(d, dict) and d.get("state") in PENDING_STATES
    ]

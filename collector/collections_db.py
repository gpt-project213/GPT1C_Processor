#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collections/collections_db.py
Хранилище состояния коллектора — история контактов, обещания, статусы.

Версия: 1.0.0 (2026-03-16)

Файл хранилища: logs/collector_state.json
Запись атомарная через tempfile (защита от частичной записи).
"""

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from zoneinfo import ZoneInfo

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env",
            encoding="utf-8-sig", override=False)

TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))

ROOT_DIR = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT_DIR / "logs" / "collector_state.json"

logger = logging.getLogger(__name__)


def _today() -> str:
    return datetime.now(tz=TZ).date().isoformat()


def load_state() -> Dict[str, Any]:
    """Загружает состояние коллектора из JSON-файла."""
    if not STATE_PATH.exists():
        return {}
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Ошибка чтения collector_state.json: %s", e)
        return {}


def save_state(state: Dict[str, Any]) -> None:
    """Атомарная запись состояния через tempfile."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            dir=STATE_PATH.parent,
            delete=False,
        ) as tmp:
            json.dump(state, tmp, ensure_ascii=False, indent=2)
            tmp_path = Path(tmp.name)
        tmp_path.replace(STATE_PATH)
    except OSError as e:
        logger.error("Ошибка записи collector_state.json: %s", e)


def _empty_record() -> Dict[str, Any]:
    return {
        "last_contact_date": None,
        "last_contact_channel": None,
        "last_level": 0,
        "last_message_text": None,
        "promise_date": None,
        "promise_amount": None,
        "promise_kept": None,
        "call_result": None,
        "call_transcript": None,
        "response_received": False,
        "last_response_text": None,
        "openclaw_session_id": None,
        "escalated_to_admin": False,
        "history": [],
    }


def get_client_state(name: str) -> Dict[str, Any]:
    """Возвращает запись клиента из хранилища (или пустую если нет)."""
    state = load_state()
    return state.get(name, _empty_record())


def update_after_contact(
    name: str,
    channel: str,
    level: int,
    message: str,
    response: Optional[str] = None,
) -> None:
    """Обновляет запись после отправки сообщения клиенту."""
    state = load_state()
    record = state.get(name, _empty_record())

    today = _today()
    record["last_contact_date"] = today
    record["last_contact_channel"] = channel
    record["last_level"] = level
    record["last_message_text"] = message
    if response is not None:
        record["response_received"] = True
        record["last_response_text"] = response

    # Добавляем в историю (только статусы, без суммы/имени в явном виде)
    record["history"].append({
        "date": today,
        "channel": channel,
        "level": level,
        "sent": True,
        "response": response is not None,
    })

    state[name] = record
    save_state(state)
    logger.info("Обновлено состояние для %s (level=%d, ch=%s)", name, level, channel)


def save_promise(name: str, promise_date: str, amount: Optional[float]) -> None:
    """Сохраняет обещание оплаты."""
    state = load_state()
    record = state.get(name, _empty_record())
    record["promise_date"] = promise_date
    record["promise_amount"] = amount
    record["promise_kept"] = None  # сбрасываем — новое обещание
    state[name] = record
    save_state(state)
    logger.info("Обещание сохранено для %s: дата=%s, сумма=%s", name, promise_date, amount)


def already_contacted_today(name: str) -> bool:
    """True если клиент уже получал сообщение сегодня."""
    record = get_client_state(name)
    return record.get("last_contact_date") == _today()


def get_pending_promises() -> List[Dict[str, Any]]:
    """Возвращает список клиентов с просроченными обещаниями.

    Критерий: promise_date < today и promise_kept is None (не подтверждено и не нарушено).
    """
    state = load_state()
    today = _today()
    result = []
    for name, record in state.items():
        p_date = record.get("promise_date")
        kept = record.get("promise_kept")
        if p_date and kept is None and p_date < today:
            result.append({
                "name": name,
                "promise_date": p_date,
                "promise_amount": record.get("promise_amount"),
                "last_level": record.get("last_level", 0),
            })
    return result


def mark_promise_broken(name: str) -> None:
    """Помечает обещание как нарушенное."""
    state = load_state()
    if name in state:
        state[name]["promise_kept"] = False
        save_state(state)
        logger.info("Обещание нарушено: %s", name)


def mark_escalated(name: str) -> None:
    """Помечает клиента как эскалированного директору."""
    state = load_state()
    record = state.get(name, _empty_record())
    record["escalated_to_admin"] = True
    state[name] = record
    save_state(state)


def save_openclaw_session(name: str, session_id: str) -> None:
    """Сохраняет OpenClaw session_id для клиента."""
    state = load_state()
    record = state.get(name, _empty_record())
    record["openclaw_session_id"] = session_id
    state[name] = record
    save_state(state)


def save_call_result(name: str, call_result: str, transcript: Optional[str]) -> None:
    """Сохраняет результат голосового звонка."""
    state = load_state()
    record = state.get(name, _empty_record())
    record["call_result"] = call_result
    record["call_transcript"] = transcript
    state[name] = record
    save_state(state)

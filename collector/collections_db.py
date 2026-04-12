#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collections/collections_db.py
Хранилище состояния коллектора — история контактов, обещания, статусы.

Версия: 1.0.1 (2026-04-12)

Файл хранилища: logs/collector_state.json
Запись атомарная через tempfile (защита от частичной записи).
"""

import json
import logging
import os
from datetime import date, datetime, timedelta
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
    logger.info("Обещание сохранено: дата=%s", promise_date)


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


# Ключ для хранения даты уведомления менеджера об отсутствии контакта
_MGR_NOTIFY_PREFIX = "__mgr_notify__"


_PHONE_PENDING_PREFIX = "__phone_pending__"


def set_phone_pending(manager_chat_id: int, client_name: str) -> None:
    """Сохраняет ожидание ввода телефона от менеджера для указанного клиента."""
    state = load_state()
    state[_PHONE_PENDING_PREFIX + str(manager_chat_id)] = {
        "client": client_name,
        "date": _today(),
    }
    save_state(state)


def get_phone_pending(manager_chat_id: int) -> Optional[str]:
    """Возвращает имя клиента, для которого менеджер ожидает ввода телефона, или None."""
    state = load_state()
    record = state.get(_PHONE_PENDING_PREFIX + str(manager_chat_id))
    if record and record.get("date") == _today():
        return record.get("client")
    return None


def clear_phone_pending(manager_chat_id: int) -> None:
    """Сбрасывает ожидание ввода телефона для менеджера."""
    state = load_state()
    key = _PHONE_PENDING_PREFIX + str(manager_chat_id)
    if key in state:
        del state[key]
        save_state(state)


_NAME_PENDING_PREFIX = "__name_pending__"


def set_name_pending(manager_chat_id: int, client_name: str) -> None:
    """Сохраняет ожидание ввода исправленного имени от менеджера."""
    state = load_state()
    state[_NAME_PENDING_PREFIX + str(manager_chat_id)] = {
        "client": client_name,
        "date": _today(),
    }
    save_state(state)


def get_name_pending(manager_chat_id: int) -> Optional[str]:
    """Возвращает имя клиента, для которого менеджер ожидает ввода исправления, или None."""
    state = load_state()
    record = state.get(_NAME_PENDING_PREFIX + str(manager_chat_id))
    if record and record.get("date") == _today():
        return record.get("client")
    return None


def clear_name_pending(manager_chat_id: int) -> None:
    """Сбрасывает ожидание ввода исправленного имени для менеджера."""
    state = load_state()
    key = _NAME_PENDING_PREFIX + str(manager_chat_id)
    if key in state:
        del state[key]
        save_state(state)


def already_notified_manager_today(client_name: str) -> bool:
    """True если менеджер уже получал запрос на регистрацию этого клиента сегодня."""
    state = load_state()
    key = _MGR_NOTIFY_PREFIX + client_name
    return (state.get(key) or {}).get("date") == _today()


def mark_manager_notified(client_name: str) -> None:
    """Фиксирует что менеджеру отправлен запрос на регистрацию клиента."""
    state = load_state()
    state[_MGR_NOTIFY_PREFIX + client_name] = {"date": _today()}
    save_state(state)


# ─── Счётчик дней с момента обнаружения долга ────────────────────────────────

_DEBT_DATE_PREFIX = "__debt_since__"


def get_debt_days_since_first_seen(client_name: str) -> int:
    """Возвращает кол-во дней с момента первого обнаружения долга у клиента.

    Независимо от частичных оплат — счётчик не сбрасывается до debt=0.
    Если запись не найдена — регистрирует сегодняшнюю дату и возвращает 0.
    """
    state = load_state()
    key = _DEBT_DATE_PREFIX + client_name
    today_str = _today()

    if key not in state:
        # Первый раз видим этого должника — фиксируем дату
        state[key] = {"first_seen": today_str}
        save_state(state)
        return 0

    first_seen = state[key].get("first_seen", today_str)
    try:
        from datetime import date as _date
        d0 = _date.fromisoformat(first_seen)
        return (datetime.now(tz=TZ).date() - d0).days
    except (ValueError, TypeError):
        return 0


def reset_debt_first_seen(client_name: str) -> None:
    """Сбрасывает счётчик долга (вызывать когда debt стал 0)."""
    state = load_state()
    key = _DEBT_DATE_PREFIX + client_name
    if key in state:
        del state[key]
        save_state(state)


def fix_first_seen_inflation(inflation_date: str = "2026-03-19") -> Dict[str, int]:
    """Пересчитывает first_seen у клиентов, записанных в день первого запуска коллектора.

    Проблема: при первом запуске 157 клиентам выставлен first_seen=2026-03-19,
    хотя реальная просрочка у них другая. Это занижает уровень давления.

    Алгоритм:
      - Находит все __debt_since__* записи с first_seen == inflation_date
      - Для каждого клиента берёт days_overdue из текущего debt JSON
      - Устанавливает first_seen = today - days_overdue
      - Если клиент не найден в debt JSON — сбрасывает first_seen=None
        (при следующем запуске будет инициализирован заново как новый)

    Args:
        inflation_date: Дата первого запуска (YYYY-MM-DD), которую нужно исправить.

    Returns:
        {"fixed": N, "reset": N, "skipped": N}
          fixed   — пересчитано по debt JSON
          reset   — удалено (debt=0 или отсутствует)
          skipped — не найдено в debt JSON, first_seen сброшен на None
    """
    from collector.debt_monitor import load_latest_debt_json, get_overdue_days

    state = load_state()
    today_date = datetime.now(tz=TZ).date()

    # Собираем клиентов с раздутой датой
    to_fix: Dict[str, str] = {}  # client_name → state_key
    for key, val in state.items():
        if key.startswith(_DEBT_DATE_PREFIX):
            if isinstance(val, dict) and val.get("first_seen") == inflation_date:
                client_name = key[len(_DEBT_DATE_PREFIX):]
                to_fix[client_name] = key

    if not to_fix:
        logger.info("fix_first_seen_inflation: нет записей с датой %s", inflation_date)
        return {"fixed": 0, "reset": 0, "skipped": 0}

    # Строим dict name→data из списка клиентов debt JSON
    raw = load_latest_debt_json()
    debt_by_name: Dict[str, Any] = {
        (c.get("name") or c.get("client") or "").strip(): c
        for c in raw.get("clients", [])
        if (c.get("name") or c.get("client") or "").strip()
    }

    fixed = 0
    reset = 0
    skipped = 0

    for client_name, key in to_fix.items():
        client_data = debt_by_name.get(client_name)
        if client_data is not None:
            days = get_overdue_days(client_data)
            if days > 0:
                actual_first_seen = (today_date - timedelta(days=days)).isoformat()
                state[key] = {"first_seen": actual_first_seen}
                fixed += 1
                logger.info(
                    "fix_first_seen: %s → %s (days=%d)", client_name, actual_first_seen, days,
                )
            else:
                # days=0 — долг закрыт, удаляем запись
                del state[key]
                reset += 1
        else:
            # Клиент не найден в текущем debt JSON — сбрасываем
            state[key] = {"first_seen": None}
            skipped += 1
            logger.info("fix_first_seen: %s не найден в debt JSON — сброшен", client_name)

    save_state(state)
    logger.info(
        "fix_first_seen_inflation завершён: fixed=%d, reset=%d, skipped=%d",
        fixed, reset, skipped,
    )
    return {"fixed": fixed, "reset": reset, "skipped": skipped}

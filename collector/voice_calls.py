#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collections/voice_calls.py
Голосовые звонки должникам через Retell AI.

Версия: 1.0.0 (2026-03-16)

API: POST https://api.retellai.com/v2/create-phone-call
Переменные .env: RETELL_API_KEY, RETELL_AGENT_ID, COMPANY_PHONE

Правила:
  - Звонить только при level >= COLLECTOR_CALL_LEVEL (по умолчанию 4)
  - Только 09:00–17:00 Asia/Almaty (не 18:00 — нужно время на разговор)
  - Не в выходные
  - do_not_call=true → только письменные каналы
  - Максимум 1 звонок в день одному клиенту
  - Транскрипт сохраняется в collections_db
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import httpx
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env",
            encoding="utf-8-sig", override=False)

TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))

RETELL_ENABLED  = os.getenv("RETELL_ENABLED", "false").lower() == "true"
RETELL_API_KEY  = os.getenv("RETELL_API_KEY", "")
RETELL_AGENT_ID = os.getenv("RETELL_AGENT_ID", "")
COMPANY_PHONE   = os.getenv("COMPANY_PHONE", "")
CALL_LEVEL_MIN  = int(os.getenv("COLLECTOR_CALL_LEVEL", "4"))
CALL_HOUR_START = int(os.getenv("COLLECTOR_HOUR_START", "9"))
CALL_HOUR_END   = 17  # жёстко: 17:00 (не 18:00)

RETELL_BASE_URL = "https://api.retellai.com"

logger = logging.getLogger(__name__)


def is_call_allowed_time() -> bool:
    """True если сейчас рабочее время для звонков (09:00–17:00, не выходные)."""
    now = datetime.now(tz=TZ)
    if now.weekday() >= 5:
        return False
    return CALL_HOUR_START <= now.hour < CALL_HOUR_END


def initiate_call(
    phone: str,
    client_name: str,
    debt_amount: float,
    days_overdue: int,
    level: int,
    language: str = "ru",
) -> Dict[str, str]:
    """Инициирует голосовой звонок через Retell AI.

    Args:
        phone:        Телефон должника (+77XXXXXXXXX)
        client_name:  Название клиента
        debt_amount:  Сумма долга
        days_overdue: Дней просрочки
        level:        Уровень давления
        language:     "ru" или "kz"

    Returns:
        {"call_id": "...", "status": "initiated"} или
        {"call_id": "", "status": "error: ..."}
    """
    if not RETELL_ENABLED:
        msg = "Голосовые звонки отключены (RETELL_ENABLED=false)"
        logger.info(msg)
        return {"call_id": "", "status": f"skipped: {msg}"}

    if not RETELL_API_KEY or not RETELL_AGENT_ID:
        msg = "RETELL_API_KEY / RETELL_AGENT_ID не заданы"
        logger.warning(msg)
        return {"call_id": "", "status": f"error: {msg}"}

    if not COMPANY_PHONE:
        msg = "COMPANY_PHONE не задан"
        logger.warning(msg)
        return {"call_id": "", "status": f"error: {msg}"}

    if level < CALL_LEVEL_MIN:
        msg = f"level {level} < min {CALL_LEVEL_MIN} — звонок не инициируется"
        logger.info(msg)
        return {"call_id": "", "status": f"skipped: {msg}"}

    if not is_call_allowed_time():
        msg = "Звонки разрешены только 09:00–17:00 в рабочие дни"
        logger.info(msg)
        return {"call_id": "", "status": f"skipped: {msg}"}

    amount_str = f"{debt_amount:,.0f}".replace(",", " ")
    lang_label = "казахском" if language == "kz" else "русском"

    metadata = {
        "client_name": client_name,
        "debt_amount": amount_str,
        "days_overdue": str(days_overdue),
        "language": language,
        "level": str(level),
    }

    payload = {
        "agent_id": RETELL_AGENT_ID,
        "from_number": COMPANY_PHONE,
        "to_number": phone,
        "metadata": metadata,
        "retell_llm_dynamic_variables": {
            "client_name": client_name,
            "debt_amount": amount_str,
            "days_overdue": str(days_overdue),
            "language": lang_label,
        },
    }

    headers = {
        "Authorization": f"Bearer {RETELL_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = httpx.post(
            f"{RETELL_BASE_URL}/v2/create-phone-call",
            headers=headers,
            json=payload,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        call_id = data.get("call_id", "")
        logger.info("Retell звонок инициирован: call_id=%s, клиент=%s", call_id, client_name)
        return {"call_id": call_id, "status": "initiated"}
    except (httpx.HTTPError, KeyError) as e:
        logger.error("Retell API ошибка: %s", e)
        return {"call_id": "", "status": f"error: {e}"}


def get_call_result(call_id: str) -> Dict:
    """Получает результат звонка по call_id.

    Returns:
        {"transcript": "...", "intent": "...", "promise_date": "...", "duration": N}
    """
    if not call_id or not RETELL_API_KEY:
        return {"transcript": "", "intent": "unknown", "promise_date": None, "duration": 0}

    headers = {"Authorization": f"Bearer {RETELL_API_KEY}"}
    try:
        resp = httpx.get(
            f"{RETELL_BASE_URL}/v2/get-call/{call_id}",
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        transcript = data.get("transcript", "")
        duration = data.get("duration_ms", 0) // 1000
        # intent и promise_date — из custom analysis если Retell настроен
        intent = data.get("call_analysis", {}).get("custom_analysis_data", {}).get("intent", "unknown")
        promise_date = data.get("call_analysis", {}).get("custom_analysis_data", {}).get("promise_date")
        return {
            "transcript": transcript,
            "intent": intent,
            "promise_date": promise_date,
            "duration": duration,
        }
    except (httpx.HTTPError, KeyError) as e:
        logger.error("Retell get_call_result ошибка: %s", e)
        return {"transcript": "", "intent": "error", "promise_date": None, "duration": 0}

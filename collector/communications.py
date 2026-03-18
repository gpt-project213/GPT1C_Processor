#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collections/communications.py
Единый gateway для отправки сообщений должникам и уведомлений команде.

Версия: 1.0.0 (2026-03-16)

Каналы:
  WhatsApp — Green API (GREENAPI_ID, GREENAPI_TOKEN из .env)
  Telegram  — существующий бот (TG_BOT_TOKEN / BOT_TOKEN из .env)
  Admin     — уведомление директора (ADMIN_CHAT_ID из .env)

Жёсткие ограничения:
  - Только 09:00–18:00 Asia/Almaty
  - Не в выходные (суббота, воскресенье)
  - Макс. 1 сообщение в день одному клиенту
  - При ошибке доставки — уведомить менеджера, не падать молча
"""

import json
import logging
import os
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env",
            encoding="utf-8-sig", override=False)

TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))

GREENAPI_ID      = os.getenv("GREENAPI_ID", "")
GREENAPI_TOKEN   = os.getenv("GREENAPI_TOKEN", "")
BOT_TOKEN        = os.getenv("TG_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")
ADMIN_CHAT_ID    = os.getenv("ADMIN_CHAT_ID", "")
WHATSAPP_ENABLED = os.getenv("WHATSAPP_ENABLED", "0") == "1"
TEST_MODE        = os.getenv("TEST_MODE", "0") == "1"
TEST_WA_PHONE    = os.getenv("TEST_WA_PHONE", "")

HOUR_START = int(os.getenv("COLLECTOR_HOUR_START", "9"))
HOUR_END   = int(os.getenv("COLLECTOR_HOUR_END", "18"))

logger = logging.getLogger(__name__)


def is_allowed_time() -> bool:
    """True если текущее время попадает в рабочее окно (не выходные, 09–18)."""
    now = datetime.now(tz=TZ)
    if now.weekday() >= 5:  # 5=суббота, 6=воскресенье
        return False
    return HOUR_START <= now.hour < HOUR_END


def send_whatsapp(phone: str, text: str) -> bool:
    """Отправляет сообщение через Green API.

    POST https://api.green-api.com/waInstance{ID}/sendMessage/{TOKEN}
    Возвращает True при успехе.
    WhatsApp временно отключён — установите WHATSAPP_ENABLED=1 в .env для включения.

    В TEST_MODE все сообщения перенаправляются на TEST_WA_PHONE.
    """
    if TEST_MODE and TEST_WA_PHONE:
        logger.info("TEST_MODE: redirecting WhatsApp to %s (original: %s)", TEST_WA_PHONE, phone)
        phone = TEST_WA_PHONE
    if not WHATSAPP_ENABLED:
        logger.info("WhatsApp отключён (WHATSAPP_ENABLED=0) — пропуск отправки: %s", phone)
        return False
    if not GREENAPI_ID or not GREENAPI_TOKEN:
        logger.warning("Green API не настроен (GREENAPI_ID / GREENAPI_TOKEN отсутствуют)")
        return False

    url = f"https://api.green-api.com/waInstance{GREENAPI_ID}/sendMessage/{GREENAPI_TOKEN}"
    # Нормализуем телефон: убираем символы кроме цифр и +
    phone_clean = "".join(c for c in phone if c.isdigit() or c == "+")
    if not phone_clean.startswith("+"):
        phone_clean = "+" + phone_clean
    chat_id = phone_clean.lstrip("+") + "@c.us"

    payload = {"chatId": chat_id, "message": text}
    try:
        resp = httpx.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            logger.info("WhatsApp отправлен: %s", phone)
            return True
        logger.warning("Green API ошибка %d: %s", resp.status_code, resp.text[:200])
        return False
    except (httpx.RequestError, httpx.TimeoutException) as e:
        logger.error("Green API сетевая ошибка: %s", e)
        return False


async def send_telegram(telegram_id: int, text: str) -> bool:
    """Отправляет сообщение через Telegram Bot API.

    Использует существующий BOT_TOKEN из .env.
    """
    if not BOT_TOKEN:
        logger.warning("BOT_TOKEN не задан — Telegram недоступен")
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": telegram_id,
        "text": text,
        "parse_mode": "HTML",
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload)
        if resp.status_code == 200:
            logger.info("Telegram отправлен: chat_id=%s", telegram_id)
            return True
        logger.warning("Telegram ошибка %d: %s", resp.status_code, resp.text[:200])
        return False
    except (httpx.RequestError, httpx.TimeoutException) as e:
        logger.error("Telegram сетевая ошибка: %s", e)
        return False


async def notify_admin(message: str) -> bool:
    """Отправляет уведомление директору (ADMIN_CHAT_ID)."""
    if not ADMIN_CHAT_ID:
        logger.warning("ADMIN_CHAT_ID не задан — эскалация невозможна")
        return False
    try:
        admin_id = int(ADMIN_CHAT_ID)
    except ValueError:
        logger.error("ADMIN_CHAT_ID не является числом: %s", ADMIN_CHAT_ID)
        return False
    return await send_telegram(admin_id, f"🚨 <b>AI Коллектор — эскалация</b>\n\n{message}")


async def notify_manager(manager_chat_id: int, message: str) -> bool:
    """Уведомляет менеджера клиента об ошибке доставки или нарушении обещания."""
    return await send_telegram(manager_chat_id, f"⚠️ <b>AI Коллектор</b>\n\n{message}")


async def send_telegram_with_markup(
    telegram_id: int, text: str, reply_markup: Any
) -> bool:
    """Отправляет сообщение с inline-клавиатурой через Telegram Bot API."""
    if not BOT_TOKEN:
        logger.warning("BOT_TOKEN не задан — Telegram недоступен")
        return False
    import json as _json
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": telegram_id,
        "text": text,
        "parse_mode": "HTML",
        "reply_markup": reply_markup.to_dict() if hasattr(reply_markup, "to_dict") else reply_markup,
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload)
        if resp.status_code == 200:
            logger.info("Telegram+markup отправлен: chat_id=%s", telegram_id)
            return True
        logger.warning("Telegram+markup ошибка %d: %s", resp.status_code, resp.text[:200])
        return False
    except (httpx.RequestError, httpx.TimeoutException) as e:
        logger.error("Telegram+markup сетевая ошибка: %s", e)
        return False


def get_observer_ids(manager_name: str) -> list:
    """Returns chat_ids of manager + supervising subadmin (if any) + all admins."""
    roles_path = Path(__file__).resolve().parent.parent / "config" / "roles.json"
    try:
        with open(roles_path, encoding="utf-8") as f:
            roles = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    ids: set = set()

    # The manager themselves
    mgr_id = roles.get("managers", {}).get(manager_name)
    if mgr_id:
        ids.add(int(mgr_id))

    # Subadmins who supervise this manager
    for sub_id_str, scopes in roles.get("subadmin_scopes", {}).items():
        if manager_name in scopes:
            ids.add(int(sub_id_str))

    # All admins
    for admin_id in roles.get("admins", []):
        ids.add(int(admin_id))

    return sorted(ids)

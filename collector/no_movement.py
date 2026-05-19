#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/no_movement.py
No-movement Saida-first check before sending WhatsApp.

Flow:
  1. collections_engine.run() detects debit==0 and credit==0 (no movements)
  2. ask_saida_about_no_movement() → Саида gets 2 buttons
  3a. nm_paid|key   → payment_hold created, client skipped today
  3b. nm_nopay|key  → admin notified with approve/skip buttons
  4a. nm_adm_send|key → status=approved_send, run(single_client) triggered
  4b. nm_adm_skip|key → status=skipped

State: logs/no_movement_saida_state.json
  { client_norm: { date, name, status, amount, days, mgr_name, mgr_chat_id } }

Statuses:
  pending_saida        — вопрос задан Саиде, ответа нет
  nopay_notified_admin — Саида: нет оплаты, ждём решения руководителя
  approved_send        — руководитель одобрил → run() пропустит nm-блок
  skipped              — руководитель пропустил
  paid                 — Саида подтвердила оплату → payment_hold создан

Note: no top-level import of collections_engine (circular import guard).
"""
from __future__ import annotations

import asyncio
import json
import logging
from collector.logging_utils import get_collector_logger
from collector.payment_hold import strip_manager_prefix as _strip_pfx
import os
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional

logger = get_collector_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
TZ_NAME = os.getenv("TZ", "Asia/Almaty")
try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo(TZ_NAME)
except Exception:  # pragma: no cover
    TZ = None

NM_STATE_FILE = ROOT / "logs" / "no_movement_saida_state.json"
SAIDA_CHAT_ID = int(os.getenv("SAIDA_CHAT_ID", "920236287"))
ADMIN_CHAT_ID_STR = os.getenv("ADMIN_CHAT_ID", "7422963573")


def _now() -> datetime:
    return datetime.now(TZ) if TZ else datetime.now()


def _today() -> str:
    return _now().strftime("%Y-%m-%d")


def _normalize(name: str) -> str:
    return " ".join(str(name or "").lower().split())


# ── State helpers ─────────────────────────────────────────────────────────────

def _load_state() -> Dict[str, Any]:
    try:
        if NM_STATE_FILE.exists():
            data = json.loads(NM_STATE_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _save_state(data: Dict[str, Any]) -> None:
    NM_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8",
            dir=NM_STATE_FILE.parent,
            delete=False, suffix=".tmp",
        ) as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            tmp = fh.name
        os.replace(tmp, NM_STATE_FILE)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def get_nm_state(name: str) -> Optional[Dict[str, Any]]:
    """Return today's state record for this client, or None."""
    rec = _load_state().get(_normalize(name))
    if not isinstance(rec, dict):
        return None
    return rec if rec.get("date") == _today() else None


def was_saida_asked_today(name: str) -> bool:
    return get_nm_state(name) is not None


def _find_by_key(key: str) -> Optional[str]:
    """Find full client name by callback key (first 26 chars of name)."""
    today = _today()
    for _norm, rec in _load_state().items():
        if not isinstance(rec, dict) or rec.get("date") != today:
            continue
        full = rec.get("name", "")
        if full[:26] == key:
            return full
    return None


# ── Public ask ────────────────────────────────────────────────────────────────

async def ask_saida_about_no_movement(
    name: str,
    amount: float,
    days: int,
    mgr_name: str,
    mgr_chat_id: Optional[int] = None,
) -> bool:
    """Send Saida a question with 2 buttons. Save state. Returns True if sent."""
    from collector.communications import send_telegram_with_markup

    key = name[:26]
    amount_str = f"{amount:,.0f}".replace(",", " ")

    try:
        import telegram as _tg
        kb = _tg.InlineKeyboardMarkup([
            [_tg.InlineKeyboardButton("✅ Оплата есть", callback_data=f"nm_paid|{key}")],
            [_tg.InlineKeyboardButton("❌ Оплат нет",   callback_data=f"nm_nopay|{key}")],
        ])
    except Exception as e:
        logger.error("[no_movement] Не удалось создать клавиатуру: %s", e)
        return False

    text = (
        f"❓ <b>Нет движений по клиенту</b>\n\n"
        f"<b>Клиент:</b> {_strip_pfx(name)}\n"
        f"<b>Долг:</b> {amount_str} тг\n"
        f"<b>Без движений:</b> {days} дн.\n"
        f"<b>Менеджер:</b> {mgr_name or '—'}\n\n"
        f"Есть поступления, которые ещё не попали в 1С?"
    )

    ok = await send_telegram_with_markup(SAIDA_CHAT_ID, text, kb)
    if ok:
        state = _load_state()
        state[_normalize(name)] = {
            "date": _today(),
            "name": name,
            "status": "pending_saida",
            "amount": amount,
            "days": days,
            "mgr_name": mgr_name,
            "mgr_chat_id": mgr_chat_id,
        }
        _save_state(state)
        logger.info("[no_movement] вопрос Саиде задан: %s", name)
    return ok


# ── Callback handler ──────────────────────────────────────────────────────────

async def handle_nm_callback(data: str, chat_id: int) -> bool:
    """Handle nm_paid|key / nm_nopay|key / nm_adm_send|key / nm_adm_skip|key."""
    from collector.communications import (
        send_telegram,
        send_telegram_with_markup,
        notify_admin,
    )

    parts = data.split("|", 1)
    if len(parts) != 2:
        return False
    action, key = parts[0], parts[1]

    if action not in ("nm_paid", "nm_nopay", "nm_adm_send", "nm_adm_skip"):
        return False

    admin_id = int(ADMIN_CHAT_ID_STR)

    # Access control
    if action in ("nm_paid", "nm_nopay") and chat_id != SAIDA_CHAT_ID:
        await send_telegram(chat_id, "⚠️ Эта кнопка только для Саиды.")
        return True
    if action in ("nm_adm_send", "nm_adm_skip") and chat_id != admin_id:
        await send_telegram(chat_id, "⚠️ Эта кнопка только для руководителя.")
        return True

    full_name = _find_by_key(key)
    if not full_name:
        logger.warning("[no_movement] клиент не найден по ключу: %s", key)
        await send_telegram(chat_id, "⚠️ Запрос не найден или устарел.")
        return True

    state = _load_state()
    norm = _normalize(full_name)
    rec = state.get(norm) or {}
    amount: float = float(rec.get("amount", 0))
    days: int = int(rec.get("days", 0))
    mgr_name: str = rec.get("mgr_name") or ""
    mgr_chat_id: Optional[int] = rec.get("mgr_chat_id")
    amount_str = f"{amount:,.0f}".replace(",", " ")

    # ── Саида: оплата есть ────────────────────────────────────────────────────
    if action == "nm_paid":
        try:
            from collector.payment_hold import (
                create_manager_payment_request,
                set_sent_to_saida,
                SOURCE_NO_MOVEMENT,
            )
            _nm_hold = create_manager_payment_request(
                manager=mgr_name or "Менеджер",
                client=full_name,
                debt=amount,
                debt_str=f"{amount_str} тг",
                manager_chat_id=mgr_chat_id,
                source=SOURCE_NO_MOVEMENT,
            )
            set_sent_to_saida(_nm_hold["token"])
        except Exception as e:
            logger.error("[no_movement] payment_hold error: %s", e)

        rec["status"] = "paid"
        state[norm] = rec
        _save_state(state)

        await send_telegram(
            chat_id,
            f"✅ Зафиксировала. Ожидаем разноску в 1С по <b>{_strip_pfx(full_name)}</b>."
        )
        await notify_admin(
            f"💳 <b>Нет движений — оплата подтверждена Саидой</b>\n\n"
            f"Клиент: <b>{full_name}</b>\n"
            f"Долг: {amount_str} тг | Менеджер: {mgr_name or '—'}\n\n"
            f"Ожидаем разноску в 1С."
        )
        logger.info("[no_movement] Саида подтвердила оплату: %s", full_name)
        return True

    # ── Саида: оплат нет ──────────────────────────────────────────────────────
    if action == "nm_nopay":
        rec["status"] = "nopay_notified_admin"
        state[norm] = rec
        _save_state(state)

        await send_telegram(chat_id, f"Понял. Уведомляю руководителя по <b>{_strip_pfx(full_name)}</b>.")

        try:
            import telegram as _tg
            kb_adm = _tg.InlineKeyboardMarkup([
                [_tg.InlineKeyboardButton("✅ Отправить WA", callback_data=f"nm_adm_send|{key}")],
                [_tg.InlineKeyboardButton("⏭️ Пропустить",  callback_data=f"nm_adm_skip|{key}")],
            ])
        except Exception:
            kb_adm = None

        adm_text = (
            f"📋 <b>Нет движений — оплат нет</b>\n\n"
            f"Клиент: <b>{full_name}</b>\n"
            f"Долг: {amount_str} тг | {days} дней без движений\n"
            f"Менеджер: {mgr_name or '—'}\n\n"
            f"Саида подтвердила: оплат нет.\n"
            f"Отправить клиенту WhatsApp?"
        )
        if kb_adm:
            await send_telegram_with_markup(admin_id, adm_text, kb_adm)
        else:
            await notify_admin(adm_text)

        logger.info("[no_movement] оплат нет по %s, уведомлён руководитель", full_name)
        return True

    # ── Руководитель: пропустить ──────────────────────────────────────────────
    if action == "nm_adm_skip":
        rec["status"] = "skipped"
        state[norm] = rec
        _save_state(state)
        await send_telegram(chat_id, f"⏭️ Пропущен: <b>{full_name}</b>")
        logger.info("[no_movement] пропущен решением руководителя: %s", full_name)
        return True

    # ── Руководитель: отправить WA ────────────────────────────────────────────
    if action == "nm_adm_send":
        rec["status"] = "approved_send"
        state[norm] = rec
        _save_state(state)
        await send_telegram(chat_id, f"✅ WA по <b>{full_name}</b> отправляется...")
        logger.info("[no_movement] руководитель одобрил WA: %s", full_name)

        # Trigger collector run for this single client (late import — no circular dep)
        try:
            from collector.collections_engine import run as _ce_run
            asyncio.create_task(_ce_run(single_client=full_name))
        except Exception as e:
            logger.error("[no_movement] ошибка запуска run(%s): %s", full_name, e)
        return True

    return False

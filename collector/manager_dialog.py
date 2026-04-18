#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/manager_dialog.py · v1.0.2 · 2026-04-19
Движок диалогов менеджеров с AI Коллектором.

Telegram-взаимодействие через httpx (без python-telegram-bot).
DeepSeek AI для разбора контактных данных и причин отказа.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env",
            encoding="utf-8-sig", override=False)

TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))

_ROOT = Path(__file__).resolve().parent.parent

BOT_TOKEN              = os.getenv("TG_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")
DEEPSEEK_API_KEY       = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL         = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
COLLECTOR_REMINDER_HOURS = float(os.getenv("COLLECTOR_REMINDER_HOURS", "0.5"))
COLLECTOR_DEADLINE_DAYS  = int(os.getenv("COLLECTOR_DEADLINE_DAYS", "5"))
COLLECTOR_MANAGER_SILENCE_HOURS = float(os.getenv("COLLECTOR_MANAGER_SILENCE_HOURS", "2"))
COMPANY_NAME             = os.getenv("COMPANY_NAME", "Минбаракат")
TEST_MODE                = os.getenv("TEST_MODE", "0") == "1"
TEST_TG_CHAT_IDS: List[int] = [
    int(x.strip())
    for x in os.getenv("TEST_TG_CHAT_IDS", "").split(",")
    if x.strip()
]
OPENAI_API_KEY           = os.getenv("OPENAI_API_KEY", "")

logger = logging.getLogger(__name__)

# ─── Telegram helpers ────────────────────────────────────────────────────────

async def _tg_post(method: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Выполняет POST-запрос к Telegram Bot API."""
    if not BOT_TOKEN:
        logger.warning("BOT_TOKEN не задан — Telegram недоступен")
        return None
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload)
        if resp.status_code == 200:
            return resp.json()
        logger.warning("Telegram %s ошибка %d: %s", method, resp.status_code, resp.text[:200])
        return None
    except (httpx.RequestError, httpx.TimeoutException) as e:
        logger.error("Telegram %s сетевая ошибка: %s: %s", method, type(e).__name__, e or repr(e))
        return None


async def _send_msg(
    chat_id: int,
    text: str,
    markup: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Отправляет сообщение; возвращает message_id или None.

    В TEST_MODE сообщения отправляются только на TEST_TG_CHAT_IDS.
    """
    if TEST_MODE:
        if TEST_TG_CHAT_IDS and chat_id not in TEST_TG_CHAT_IDS:
            logger.info(
                "TEST_MODE: redirecting to test recipients — "
                "пропуск отправки chat_id=%d (не в TEST_TG_CHAT_IDS)",
                chat_id,
            )
            return None
    payload: Dict[str, Any] = {
        "chat_id":    chat_id,
        "text":       text,
        "parse_mode": "HTML",
    }
    if markup:
        payload["reply_markup"] = markup
    result = await _tg_post("sendMessage", payload)
    if result and result.get("ok"):
        return result["result"]["message_id"]
    return None


async def _edit_msg(
    chat_id: int,
    message_id: int,
    text: str,
    markup: Optional[Dict[str, Any]] = None,
) -> None:
    """Редактирует существующее сообщение."""
    payload: Dict[str, Any] = {
        "chat_id":    chat_id,
        "message_id": message_id,
        "text":       text,
        "parse_mode": "HTML",
    }
    if markup:
        payload["reply_markup"] = markup
    await _tg_post("editMessageText", payload)


def _inline(rows: List[List[Tuple[str, str]]]) -> Dict[str, Any]:
    """Формирует InlineKeyboardMarkup из списка рядов кнопок [(text, callback_data)]."""
    keyboard = []
    for row in rows:
        keyboard.append([
            {"text": label, "callback_data": data}
            for label, data in row
        ])
    return {"inline_keyboard": keyboard}


def _fmt_amount(amount: float) -> str:
    """Форматирует сумму: 500000 → '500 000'."""
    return f"{amount:,.0f}".replace(",", " ")


# ─── Message builders ────────────────────────────────────────────────────────

def _build_initial_message(dialog: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Строит начальное сообщение с раздельным подтверждением имени и телефона."""
    mid = dialog["manager_chat_id"]
    contact = dialog.get("current_contact") or {}
    phone = contact.get("whatsapp") or contact.get("phone", "не задан")
    display_name = contact.get("display_name") or contact.get("contact_person", "не задано")

    name_confirmations = contact.get("name_confirmations", 0)
    phone_confirmations = contact.get("phone_confirmations", 0)

    name_confirmed  = dialog.get("name_confirmed", False) or name_confirmations >= 3
    phone_confirmed = dialog.get("phone_confirmed", False) or phone_confirmations >= 3

    if name_confirmed and phone_confirmed:
        # Оба подтверждены — показываем финальное подтверждение
        return _build_final_confirm_message(dialog)

    lines = [
        f"📋 <b>AI Коллектор — новый должник</b>\n",
        f"Клиент в 1С: {dialog['client_name']}",
        f"Просрочка: {dialog['days']} дн. | {_fmt_amount(dialog['amount'])} тг | Уровень: {dialog['level']}\n",
        "Если не ответите сразу, бот будет напоминать каждые 30 минут и усиливать тон.\n",
        "Статистика игнора ведётся по каждому менеджеру, видна руководителю и может повлиять на отношения с руководителем.\n",
    ]

    keyboard_rows: List[List[Tuple[str, str]]] = []

    if not name_confirmed:
        lines.append("━━━━━━━━━━━━━━━━")
        lines.append("👤 <b>КАК ОБРАЩАТЬСЯ К КЛИЕНТУ?</b>\n")
        lines.append(f"Сейчас в базе: {display_name}")
        lines.append("(это имя войдёт в сообщение клиенту)\n")
        keyboard_rows.append([
            (f"✅ Верно — {display_name[:20]}", f"col_name_ok_{mid}"),
            ("✏️ Другое название",               f"col_name_edit_{mid}"),
        ])

    if not phone_confirmed:
        lines.append("━━━━━━━━━━━━━━━━")
        lines.append("📞 <b>ТЕЛЕФОН ДЛЯ WHATSAPP:</b>\n")
        lines.append(f"{phone}\n")
        keyboard_rows.append([
            ("✅ Актуален",    f"col_phone_ok_{mid}"),
            ("📞 Изменился",   f"col_phone_edit_{mid}"),
        ])

    keyboard_rows.append([("❌ Не отправлять сейчас", f"col_reject_{mid}")])
    keyboard_rows.append([("❓ Не понимаю, что ответить", f"col_help_{mid}")])

    text = "\n".join(lines)
    markup = _inline(keyboard_rows)
    return text, markup


def _build_final_confirm_message(dialog: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Строит финальное сообщение подтверждения после проверки имени и телефона."""
    mid = dialog["manager_chat_id"]
    contact = dialog.get("current_contact") or {}
    phone = contact.get("whatsapp") or contact.get("phone", "не задан")
    display_name = contact.get("display_name") or contact.get("contact_person", "не задано")

    text = (
        f"✅ <b>Данные подтверждены</b>\n\n"
        f"Клиент: {display_name}\n"
        f"Телефон: {phone}\n"
        f"Просрочка: {dialog['days']} дн. | {_fmt_amount(dialog['amount'])} тг\n\n"
        f"Отправить WhatsApp-уведомление?\n\n"
        f"<i>Если не ответите, запрос будет повторяться каждые 30 минут. "
        f"Статистика игнора видна руководителю.</i>"
    )
    markup = _inline([
        [
            ("🚀 Отправить",        f"col_confirm_{mid}"),
            ("❌ Не отправлять",    f"col_reject_{mid}"),
        ],
        [("❓ Не понимаю, что ответить", f"col_help_{mid}")],
    ])
    return text, markup


def _build_data_confirm_message(dialog: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Строит сообщение с предлагаемыми изменениями контакта + кнопки."""
    mid = dialog["manager_chat_id"]
    proposed = dialog.get("proposed_contact") or {}
    current  = dialog.get("current_contact") or {}

    lines = ["📝 <b>Обновление данных — проверьте и подтвердите</b>\n"]
    field_labels = {
        "phone":          "Телефон",
        "whatsapp":       "WhatsApp",
        "email":          "Email",
        "contact_person": "Контактное лицо",
        "language":       "Язык",
        "do_not_call":    "Не звонить",
        "notes":          "Заметки",
    }
    for field, label in field_labels.items():
        if field in proposed:
            old = current.get(field, "—")
            new = proposed[field]
            lines.append(f"  {label}: <s>{old}</s> → <b>{new}</b>")

    if len(lines) == 1:
        lines.append("  (нет изменений)")

    text = "\n".join(lines)
    markup = _inline([
        [
            ("✅ Подтвердить",    f"col_data_ok_{mid}"),
            ("✏️ Ввести заново",  f"col_data_edit_{mid}"),
        ],
        [("❓ Не понимаю, что ответить", f"col_help_{mid}")],
    ])
    return text, markup


def _build_reminder_text(dialog: Dict[str, Any], count: int) -> str:
    """Строит текст напоминания с нарастающей срочностью."""
    urgency_map = {
        1: "🔔 Напоминание",
        2: "🔔🔔 Повторное напоминание",
        3: "⚠️ Важно! Ожидаем вашего ответа",
    }
    prefix = urgency_map.get(count, f"🚨 Напоминание #{count} — требуется действие")
    return (
        f"{prefix}\n\n"
        f"Клиент: <b>{dialog['client_name']}</b>\n"
        f"Просрочка: {dialog['days']} дн. | Сумма: {_fmt_amount(dialog['amount'])} тг\n\n"
        f"<i>Запрос будет повторяться, пока не будет закрыт. "
        f"Статистика игнора видна руководителю и может повлиять на отношения с руководителем.</i>"
    )


# ─── DeepSeek helpers ────────────────────────────────────────────────────────

async def _parse_contact_with_ai(text: str, current: Dict[str, Any]) -> Dict[str, Any]:
    """Извлекает поля контакта из свободного текста менеджера."""
    if not DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY не задан — парсинг контакта пропущен")
        return {}

    system_prompt = (
        "Ты помощник, который извлекает контактные данные из сообщения менеджера. "
        "Отвечай только JSON, без пояснений."
    )
    user_prompt = (
        f"Текущие данные: {json.dumps(current, ensure_ascii=False)}\n\n"
        f"Сообщение менеджера: {text}\n\n"
        "Верни JSON только с теми полями, которые упомянуты в сообщении. "
        "Возможные поля: phone, whatsapp, email, contact_person, language (ru/kz), "
        "do_not_call (true/false), notes."
    )

    url = "https://api.deepseek.com/v1/chat/completions"
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": 300,
        "temperature": 0.1,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            logger.warning("DeepSeek ошибка %d: %s", resp.status_code, resp.text[:200])
            return {}
        content = resp.json()["choices"][0]["message"]["content"].strip()
        # Убираем markdown-блоки ```json ... ```
        if content.startswith("```"):
            lines = content.splitlines()
            lines = [l for l in lines if not l.startswith("```")]
            content = "\n".join(lines).strip()
        return json.loads(content)
    except (httpx.RequestError, httpx.TimeoutException) as e:
        logger.error("DeepSeek сетевая ошибка: %s: %s", type(e).__name__, e or repr(e))
        return {}
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error("DeepSeek неверный ответ: %s", e)
        return {}


async def _parse_rejection_reason(text: str) -> str:
    """Возвращает причину отказа дословно, без AI-обработки."""
    return text.strip()


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _get_admin_ids() -> List[int]:
    """Загружает список admin chat_id из config/roles.json."""
    roles_path = _ROOT / "config" / "roles.json"
    try:
        with open(roles_path, encoding="utf-8") as f:
            roles = json.load(f)
        admins = roles.get("admins", [])
        return [int(a) for a in admins]
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return []


def _save_contact(client_name: str, contact: Dict[str, Any]) -> None:
    """Сохраняет обновлённый контакт в CRM (config/clients.json)."""
    try:
        from bot.crm_clients import load_clients, save_clients, set_client_details
    except ImportError as e:
        logger.error("CRM недоступна, контакт не сохранён: %s", e)
        return

    primary_phone = (contact.get("whatsapp") or contact.get("phone") or "").strip()
    display_name = (contact.get("display_name") or "").strip()

    data = load_clients()
    clients_db = data.get("clients", {})

    if client_name not in clients_db:
        # Клиент отсутствует в CRM — добавляем базовую запись
        clients_db[client_name] = {
            "manager": contact.get("manager", ""),
            "whatsapp": primary_phone,
            "telegram_id": contact.get("telegram_id", ""),
            "language": contact.get("language", "ru"),
            "do_not_call": contact.get("do_not_call", False),
            "sources": ["collector"],
            "first_seen": datetime.now(tz=TZ).date().isoformat(),
            "last_seen": datetime.now(tz=TZ).date().isoformat(),
        }
        if display_name:
            clients_db[client_name]["display_name"] = display_name
        data["clients"] = clients_db
        save_clients(data)
        logger.info("CRM: добавлен новый клиент через диалог — %s", client_name)
        return

    set_client_details(
        client_name,
        display_name=display_name,
        phone=primary_phone,
    )


# ─── WhatsApp + Notification ──────────────────────────────────────────────────

async def _send_whatsapp_and_notify(dialog: Dict[str, Any]) -> bool:
    """Отправляет WhatsApp-сообщение и уведомляет всех наблюдателей.

    Повторяет попытку до 3 раз с интервалом 5 минут при неудаче.
    После успешной отправки регистрирует клиентский диалог.
    """
    client_name = str(dialog.get("client_name") or "unknown")
    logger.error(
        "[%s] legacy manager_dialog live send blocked; use collector.collections_engine --send-approved --batch-id",
        client_name,
    )
    return False


async def _escalate_to_admin(dialog: Dict[str, Any]) -> None:
    """Отправляет запрос решения по отказу администраторам."""
    mid    = dialog["manager_chat_id"]
    client = dialog["client_name"]
    mgr    = dialog["manager_name"]
    days   = dialog["days"]
    amount = dialog["amount"]
    reason = dialog.get("rejection_reason") or "(не указана)"

    text = (
        f"⚠️ <b>AI Коллектор — менеджер отказывается от отправки</b>\n\n"
        f"Менеджер: {mgr}\n"
        f"Клиент: {client}\n"
        f"Просрочка: {days} дн. | Сумма: {_fmt_amount(amount)} тг\n\n"
        f"Причина: {reason}\n\n"
        f"Принять отказ или установить дедлайн для менеджера?"
    )
    markup = _inline([
        [
            ("✅ Принять отказ",                        f"col_adm_ok_{mid}"),
            ("❌ Не принять — установить дедлайн",      f"col_adm_deny_{mid}"),
        ],
    ])

    admin_ids = _get_admin_ids()
    for admin_id in admin_ids:
        await _send_msg(admin_id, text, markup)


async def _send_deadline_exceeded(dialog: Dict[str, Any]) -> None:
    """Уведомляет о просроченном дедлайне менеджера и администраторов."""
    mid    = dialog["manager_chat_id"]
    client = dialog["client_name"]
    mgr    = dialog["manager_name"]
    days   = dialog["days"]
    amount = dialog["amount"]
    deadline = dialog.get("deadline", "")

    mgr_text = (
        f"🚨 <b>Дедлайн просрочен!</b>\n\n"
        f"Клиент: {client}\n"
        f"Просрочка: {days} дн. | Сумма: {_fmt_amount(amount)} тг\n"
        f"Дедлайн был: {deadline}\n\n"
        f"Необходимо срочно подтвердить или обновить данные."
    )
    _, markup = _build_initial_message(dialog)
    await _send_msg(mid, mgr_text, markup)

    admin_text = (
        f"🚨 <b>AI Коллектор — менеджер просрочил дедлайн</b>\n\n"
        f"Менеджер: {mgr}\n"
        f"Клиент: {client}\n"
        f"Просрочка: {days} дн. | Сумма: {_fmt_amount(amount)} тг\n"
        f"Дедлайн был: {deadline}"
    )
    admin_ids = _get_admin_ids()
    for admin_id in admin_ids:
        await _send_msg(admin_id, admin_text)


# ─── State handlers ───────────────────────────────────────────────────────────

async def _on_confirm(dialog: Dict[str, Any], mid: int) -> None:
    """Менеджер подтвердил актуальность данных — отправить WhatsApp."""
    sent = await _send_whatsapp_and_notify(dialog)
    if not sent:
        await _send_msg(
            mid,
            "⚠️ Отправка не подтверждена. Кейс оставлен активным и не переведён в CONFIRMED.",
        )


async def _on_update(dialog: Dict[str, Any], mid: int) -> None:
    """Менеджер хочет обновить данные."""
    from collector.dialog_store import update_dialog, STATE_AWAITING_DATA
    update_dialog(mid, state=STATE_AWAITING_DATA)
    await _send_msg(
        mid,
        "✏️ <b>Обновление данных</b>\n\n"
        "Отправьте обновлённые данные контакта текстом.\n"
        "Например: телефон +77011234567, контакт Иванов Иван\n\n"
        "<i>Если не отправите данные, бот будет напоминать каждые 30 минут. "
        "Статистика игнора видна руководителю.</i>",
        _inline([[("❓ Не понимаю, что ответить", f"col_help_{mid}")]]),
    )


async def _on_reject_request(dialog: Dict[str, Any], mid: int) -> None:
    """Менеджер хочет отказаться от отправки."""
    from collector.dialog_store import update_dialog, STATE_AWAITING_REJECTION_REASON
    update_dialog(mid, state=STATE_AWAITING_REJECTION_REASON)
    await _send_msg(
        mid,
        "❌ <b>Причина отказа</b>\n\n"
        "Напишите причину, по которой не нужно отправлять сообщение клиенту.\n"
        "Причина будет передана руководителю.\n\n"
        "<i>Если не написать причину, бот будет напоминать каждые 30 минут. "
        "Статистика игнора видна руководителю.</i>",
        _inline([[("❓ Не понимаю, что ответить", f"col_help_{mid}")]]),
    )


async def _on_help(dialog: Dict[str, Any], mid: int) -> None:
    """Объясняет менеджеру текущий запрос и кнопки через жёсткий DeepSeek-помощник."""
    try:
        from collector.manager_help import build_manager_help

        help_text = await build_manager_help(
            area="AI Коллектор — согласование WhatsApp и CRM-данных",
            manager=dialog.get("manager_name", ""),
            client=dialog.get("client_name", ""),
            state=dialog.get("state", ""),
            buttons=[
                "Верно/актуален",
                "Другое название/изменился",
                "Отправить",
                "Не отправлять",
                "Подтвердить",
                "Ввести заново",
            ],
            context={
                "days": dialog.get("days"),
                "amount": dialog.get("amount"),
                "level": dialog.get("level"),
                "name_confirmed": dialog.get("name_confirmed"),
                "phone_confirmed": dialog.get("phone_confirmed"),
                "current_contact": dialog.get("current_contact") or {},
                "remind_count": dialog.get("remind_count", 0),
            },
        )
    except Exception as e:
        logger.warning("manager_dialog help error: %s", e)
        help_text = (
            "Что от вас хотят:\n"
            "Нужно подтвердить данные клиента или объяснить, почему WhatsApp отправлять не надо.\n\n"
            "Что нажать:\n"
            "• Верно/актуален — если имя или телефон подходят.\n"
            "• Другое название/изменился — если нужно исправить данные.\n"
            "• Отправить — если всё проверено и можно готовить WhatsApp.\n"
            "• Не отправлять — если есть причина, её надо написать.\n\n"
            "Что будет если молчать:\n"
            "Бот будет напоминать каждые 30 минут, затем передаст игнор руководителю. "
            "Статистика игнора ведётся по каждому менеджеру и может повлиять на "
            "отношения с руководителем."
        )
    await _send_msg(mid, f"❓ <b>Подсказка по запросу</b>\n\n{help_text}")


async def _on_data_received(
    dialog: Dict[str, Any],
    chat_id: int,
    text: str,
) -> None:
    """Обрабатывает текстовое сообщение менеджера в зависимости от состояния."""
    from collector.dialog_store import (
        update_dialog,
        STATE_AWAITING_REJECTION_REASON,
        STATE_AWAITING_DATA,
        STATE_AWAITING_DATA_CONFIRM,
        STATE_REJECTED_PENDING_ADMIN,
    )
    mid = dialog["manager_chat_id"]
    state = dialog.get("state")

    if state == STATE_AWAITING_REJECTION_REASON:
        reason = await _parse_rejection_reason(text)
        update_dialog(
            mid,
            state=STATE_REJECTED_PENDING_ADMIN,
            rejection_reason=reason,
        )
        dialog["state"] = STATE_REJECTED_PENDING_ADMIN
        dialog["rejection_reason"] = reason
        await _send_msg(
            mid,
            "✅ Причина записана и передана руководителю.\n"
            "Ожидаем решения руководителя.",
        )
        await _escalate_to_admin(dialog)
        return

    if state == STATE_AWAITING_DATA:
        # Сначала показываем "обрабатываю"
        await _send_msg(mid, "⏳ Обрабатываю...")
        current = dialog.get("current_contact") or {}
        proposed = await _parse_contact_with_ai(text, current)

        if not proposed:
            await _send_msg(
                mid,
                "⚠️ Не удалось распознать данные.\n"
                "Попробуйте ещё раз. Пример: телефон +77011234567, контакт Иванов Иван",
            )
            return

        update_dialog(
            mid,
            proposed_contact=proposed,
            state=STATE_AWAITING_DATA_CONFIRM,
        )
        dialog["proposed_contact"] = proposed
        dialog["state"] = STATE_AWAITING_DATA_CONFIRM

        confirm_text, markup = _build_data_confirm_message(dialog)
        msg_id = await _send_msg(mid, confirm_text, markup)
        if msg_id:
            update_dialog(mid, message_id=msg_id)
        return


async def _on_data_confirmed(dialog: Dict[str, Any], mid: int) -> None:
    """Менеджер подтвердил AI-распознанные данные — применить и отправить."""
    from collector.dialog_store import update_dialog, STATE_AWAITING_CONFIRM
    current  = dialog.get("current_contact") or {}
    proposed = dialog.get("proposed_contact") or {}
    merged   = {**current, **proposed}

    _save_contact(dialog["client_name"], merged)
    update_dialog(
        mid,
        state=STATE_AWAITING_CONFIRM,
        current_contact=merged,
        proposed_contact=None,
    )
    dialog["state"]           = STATE_AWAITING_CONFIRM
    dialog["current_contact"] = merged

    await _send_msg(mid, "✅ Данные обновлены.")
    sent = await _send_whatsapp_and_notify(dialog)
    if not sent:
        await _send_msg(
            mid,
            "⚠️ WhatsApp не отправлен. Диалог возвращён в ожидание подтверждения без CONFIRMED.",
        )


async def _on_data_edit(dialog: Dict[str, Any], mid: int) -> None:
    """Менеджер хочет ввести данные заново."""
    from collector.dialog_store import update_dialog, STATE_AWAITING_DATA
    update_dialog(mid, state=STATE_AWAITING_DATA, proposed_contact=None)
    await _send_msg(
        mid,
        "✏️ Введите данные заново.\n"
        "Пример: телефон +77011234567, контакт Иванов Иван",
    )


async def _on_admin_approve_rejection(dialog: Dict[str, Any], mid: int) -> None:
    """Администратор принял причину отказа — закрыть диалог."""
    from collector.dialog_store import update_dialog, STATE_DONE
    update_dialog(mid, state=STATE_DONE)

    # Уведомляем менеджера
    await _send_msg(
        mid,
        f"✅ Руководитель принял причину отказа.\n"
        f"Клиент {dialog['client_name']} снят с контроля.",
    )

    # Уведомляем всех администраторов
    admin_ids = _get_admin_ids()
    for admin_id in admin_ids:
        await _send_msg(
            admin_id,
            f"✅ Отказ одобрен\n\n"
            f"Менеджер: {dialog['manager_name']}\n"
            f"Клиент: {dialog['client_name']}\n"
            f"Причина: {dialog.get('rejection_reason', '(не указана)')}",
        )


async def _on_admin_deny_rejection(dialog: Dict[str, Any], mid: int) -> None:
    """Администратор отклонил отказ — установить дедлайн для менеджера."""
    from collector.dialog_store import update_dialog, STATE_DEADLINE_SET
    deadline_dt  = datetime.now(TZ) + timedelta(days=COLLECTOR_DEADLINE_DAYS)
    deadline_iso = deadline_dt.date().isoformat()
    deadline_fmt = deadline_dt.strftime("%d.%m.%Y")

    update_dialog(
        mid,
        state=STATE_DEADLINE_SET,
        deadline=deadline_iso,
        remind_count=0,
    )
    dialog["state"]    = STATE_DEADLINE_SET
    dialog["deadline"] = deadline_iso

    # Уведомляем менеджера
    _, markup = _build_initial_message(dialog)
    await _send_msg(
        mid,
        f"⚠️ <b>Руководитель не принял отказ</b>\n\n"
        f"Клиент: {dialog['client_name']}\n"
        f"Просрочка: {dialog['days']} дн. | Сумма: {_fmt_amount(dialog['amount'])} тг\n\n"
        f"Установлен дедлайн: <b>{deadline_fmt}</b>\n"
        f"Необходимо подтвердить или обновить данные клиента.",
        markup,
    )

    # Уведомляем всех администраторов
    admin_ids = _get_admin_ids()
    for admin_id in admin_ids:
        await _send_msg(
            admin_id,
            f"📅 Дедлайн установлен\n\n"
            f"Менеджер: {dialog['manager_name']}\n"
            f"Клиент: {dialog['client_name']}\n"
            f"Дедлайн: {deadline_fmt}",
        )


# ─── Vadim control reminder ──────────────────────────────────────────────────

async def _send_control_reminder(dialog: Dict[str, Any]) -> None:
    """Отправляет напоминание администратору (Вадиму) по клиенту на контроле."""
    mid    = dialog["manager_chat_id"]
    client = dialog["client_name"]
    mgr    = dialog["manager_name"]
    days   = dialog["days"]
    amount = dialog["amount"]
    reason = dialog.get("rejection_reason") or "(не указана)"

    text = (
        f"🕐 <b>На контроле — нет результата</b>\n\n"
        f"👤 Менеджер: {mgr}\n"
        f"🏢 Клиент: {client}\n"
        f"📅 Просрочка: теперь {days} дн. | {_fmt_amount(amount)} тг\n\n"
        f"Вчера принята причина:\n"
        f"\"{reason}\"\n\n"
        f"Оплата не поступила. Ваше решение?"
    )
    markup = _inline([
        [("📤 Уведомить клиента сейчас",    f"col_adm_send_{mid}")],
        [("💬 Вызвать менеджера на отчёт",   f"col_adm_call_{mid}")],
        [("⏳ Продлить контроль +1 день",    f"col_adm_extend_{mid}")],
    ])
    admin_ids = _get_admin_ids()
    for admin_id in admin_ids:
        await _send_msg(admin_id, text, markup)


async def _on_admin_send(dialog: Dict[str, Any], mid: int) -> None:
    """Администратор решил отправить уведомление клиенту сейчас."""
    sent = await _send_whatsapp_and_notify(dialog)
    if not sent:
        for admin_id in _get_admin_ids():
            await _send_msg(
                admin_id,
                f"⚠️ Отправка клиенту {dialog['client_name']} не подтверждена. "
                f"Кейс оставлен активным без CONFIRMED.",
            )


async def _on_admin_extend(dialog: Dict[str, Any], mid: int) -> None:
    """Администратор продлевает контроль на +1 день."""
    from collector.dialog_store import update_dialog
    control_extensions = dialog.get("control_extensions", 0) + 1
    old_deadline = dialog.get("control_deadline")
    if old_deadline:
        try:
            from datetime import date
            old_date = datetime.fromisoformat(old_deadline).date()
            new_date = old_date + timedelta(days=1)
            new_deadline = new_date.isoformat()
        except (ValueError, TypeError):
            new_deadline = (datetime.now(TZ) + timedelta(days=1)).date().isoformat()
    else:
        new_deadline = (datetime.now(TZ) + timedelta(days=1)).date().isoformat()

    update_dialog(
        mid,
        control_deadline=new_deadline,
        control_extensions=control_extensions,
    )
    admin_ids = _get_admin_ids()
    for admin_id in admin_ids:
        await _send_msg(
            admin_id,
            f"⏳ Контроль продлён до <b>{new_deadline}</b>\n\n"
            f"Клиент: {dialog['client_name']}\n"
            f"Менеджер: {dialog['manager_name']}",
        )


async def _on_admin_call_manager(dialog: Dict[str, Any], mid: int) -> None:
    """Администратор запрашивает объяснения от менеджера."""
    from collector.dialog_store import update_dialog, STATE_AWAITING_MANAGER_EXPLANATION
    client = dialog["client_name"]
    days   = dialog["days"]
    amount = dialog["amount"]
    reason = dialog.get("rejection_reason") or "(не указана)"

    update_dialog(mid, awaiting_manager_explanation=True, state=STATE_AWAITING_MANAGER_EXPLANATION)

    await _send_msg(
        mid,
        f"🔔 <b>Руководитель запрашивает объяснения</b>\n\n"
        f"Клиент: {client}\n"
        f"Просрочка: {days} дн. | {_fmt_amount(amount)} тг\n"
        f"Принятая причина: \"{reason}\"\n\n"
        f"❓ Вадим требует объяснений.\n"
        f"Напишите ответ ответным сообщением:",
    )


async def _send_admin_timeout_escalation(dialog: Dict[str, Any], age_hours: float) -> None:
    """Передаёт кейс администратору, если менеджер молчит слишком долго."""
    mid = dialog["manager_chat_id"]
    text = (
        f"⏱️ <b>Эскалация из-за молчания менеджера</b>\n\n"
        f"Менеджер: {dialog['manager_name']}\n"
        f"Клиент: {dialog['client_name']}\n"
        f"Просрочка: {dialog['days']} дн. | {_fmt_amount(dialog['amount'])} тг\n\n"
        f"Нет ответа менеджера уже {int(age_hours)} ч.\n"
        f"Кейс передан на решение руководителю."
    )
    markup = _inline([
        [("📤 Уведомить клиента сейчас", f"col_adm_send_{mid}")],
        [("💬 Вызвать менеджера на отчёт", f"col_adm_call_{mid}")],
        [("⏳ Продлить контроль +1 день", f"col_adm_extend_{mid}")],
    ])
    for admin_id in _get_admin_ids():
        await _send_msg(admin_id, text, markup)


async def _on_manager_explanation(dialog: Dict[str, Any], mid: int, text: str) -> None:
    """Обрабатывает объяснение менеджера — форвардит администраторам."""
    from collector.dialog_store import update_dialog, STATE_DEADLINE_SET
    client = dialog["client_name"]
    days   = dialog["days"]
    amount = dialog["amount"]

    update_dialog(mid, awaiting_manager_explanation=False, state=STATE_DEADLINE_SET)

    forward_text = (
        f"💬 <b>Ответ менеджера {dialog['manager_name']}:</b>\n\n"
        f"\"{text}\"\n\n"
        f"Клиент: {client} | {days} дн. | {_fmt_amount(amount)} тг"
    )
    markup = _inline([
        [
            ("📤 Уведомить клиента",    f"col_adm_send_{mid}"),
            ("⏳ +1 день",               f"col_adm_extend_{mid}"),
            ("✅ Принять объяснение",    f"col_adm_ok_{mid}"),
        ],
    ])
    admin_ids = _get_admin_ids()
    for admin_id in admin_ids:
        await _send_msg(admin_id, forward_text, markup)


# ─── Name/Phone confirmation handlers ─────────────────────────────────────────

async def _on_name_ok(dialog: Dict[str, Any], mid: int) -> None:
    """Менеджер подтвердил имя клиента."""
    from collector.dialog_store import update_dialog
    contact = dict(dialog.get("current_contact") or {})
    contact["name_confirmations"] = contact.get("name_confirmations", 0) + 1
    # Сохраняем в debtors_contacts.json
    _save_contact(dialog["client_name"], contact)
    update_dialog(mid, name_confirmed=True, current_contact=contact)
    dialog["name_confirmed"] = True
    dialog["current_contact"] = contact
    # Переотправляем сообщение с обновлённым состоянием
    text, markup = _build_initial_message(dialog)
    await _send_msg(mid, text, markup)


async def _on_name_edit(dialog: Dict[str, Any], mid: int) -> None:
    """Менеджер хочет изменить имя клиента."""
    from collector.dialog_store import update_dialog, STATE_AWAITING_NAME_TEXT
    update_dialog(mid, awaiting_name_text=True, state=STATE_AWAITING_NAME_TEXT)
    contact = dict(dialog.get("current_contact") or {})
    contact["name_confirmations"] = 0
    _save_contact(dialog["client_name"], contact)
    update_dialog(mid, current_contact=contact)
    await _send_msg(
        mid,
        "✏️ <b>Введите имя клиента</b>\n\n"
        "Как обращаться к клиенту в сообщении WhatsApp?\n"
        "Например: ТОО Альфа, Иван Иванов, Магазин у дома",
    )


async def _on_phone_ok(dialog: Dict[str, Any], mid: int) -> None:
    """Менеджер подтвердил телефон клиента."""
    from collector.dialog_store import update_dialog
    contact = dict(dialog.get("current_contact") or {})
    contact["phone_confirmations"] = contact.get("phone_confirmations", 0) + 1
    primary_phone = (contact.get("whatsapp") or contact.get("phone") or "").strip()
    if primary_phone:
        contact["whatsapp"] = primary_phone
        contact["phone"] = primary_phone
        contact["_needs_phone"] = False
    _save_contact(dialog["client_name"], contact)
    update_dialog(mid, phone_confirmed=True, current_contact=contact)
    dialog["phone_confirmed"] = True
    dialog["current_contact"] = contact
    text, markup = _build_initial_message(dialog)
    await _send_msg(mid, text, markup)


async def _on_phone_edit(dialog: Dict[str, Any], mid: int) -> None:
    """Менеджер сообщает о смене телефона клиента."""
    from collector.dialog_store import update_dialog, STATE_AWAITING_PHONE_TEXT
    update_dialog(mid, awaiting_phone_text=True, state=STATE_AWAITING_PHONE_TEXT)
    contact = dict(dialog.get("current_contact") or {})
    contact["phone_confirmations"] = 0
    contact["_needs_phone"] = True
    _save_contact(dialog["client_name"], contact)
    update_dialog(mid, current_contact=contact)
    await _send_msg(
        mid,
        "📞 <b>Введите новый телефон</b>\n\n"
        "Введите актуальный номер WhatsApp клиента.\n"
        "Например: +77011234567 или 87011234567",
    )


# ─── Public API ───────────────────────────────────────────────────────────────

async def start_dialog(
    client: Dict[str, Any],
    contact: Dict[str, Any],
    manager_name: str,
    manager_chat_id: int,
) -> None:
    """Запускает новый диалог с менеджером по конкретному клиенту."""
    from collector.dialog_store import (
        get_dialog, new_dialog, update_dialog,
        STATE_CONFIRMED, STATE_DONE,
    )

    existing = get_dialog(manager_chat_id)

    if existing:
        state = existing.get("state", "")
        # Уже активный диалог по другому клиенту
        if (state not in (STATE_CONFIRMED, STATE_DONE)
                and existing.get("client_name") != client["name"]):
            logger.info(
                "[%s] менеджер %s занят диалогом по %s — пропуск нового",
                client["name"], manager_name, existing.get("client_name"),
            )
            return
        # Тот же клиент, уже подтверждён/закрыт
        if (state in (STATE_CONFIRMED, STATE_DONE)
                and existing.get("client_name") == client["name"]):
            logger.info(
                "[%s] диалог уже завершён (state=%s) — начинаем новый",
                client["name"], state,
            )

    # Создаём новый диалог
    dialog = new_dialog(
        manager_chat_id=manager_chat_id,
        manager_name=manager_name,
        client_name=client["name"],
        level=client.get("level", 1),
        days=client.get("days", 0),
        amount=float(client.get("amount", 0)),
        current_contact=dict(contact),
    )

    text, markup = _build_initial_message(dialog)
    msg_id = await _send_msg(manager_chat_id, text, markup)
    if msg_id:
        update_dialog(manager_chat_id, message_id=msg_id)


async def handle_callback(data: str, chat_id: int, message_id: int) -> bool:
    """Обрабатывает inline callback от Telegram. Возвращает True если обработан."""
    from collector.dialog_store import get_dialog, load_dialogs

    if not data.startswith("col_"):
        return False

    # Парсим callback data: col_<action>_<manager_chat_id>
    parts = data.split("_", 2)
    if len(parts) < 3:
        return False

    action_part = parts[1]
    # Для составных actions (adm_ok, adm_deny, data_ok, data_edit)
    # data: col_adm_ok_12345 → parts = ['col', 'adm', 'ok_12345']
    # Нужна другая схема парсинга

    # Полный список callback паттернов
    callback_map = [
        ("col_confirm_",   "confirm"),
        ("col_update_",    "update"),
        ("col_reject_",    "reject"),
        ("col_data_ok_",   "data_ok"),
        ("col_data_edit_", "data_edit"),
        ("col_adm_ok_",    "adm_ok"),
        ("col_adm_deny_",  "adm_deny"),
        ("col_adm_send_",  "adm_send"),
        ("col_adm_call_",  "adm_call"),
        ("col_adm_extend_","adm_extend"),
        ("col_name_ok_",   "name_ok"),
        ("col_name_edit_", "name_edit"),
        ("col_phone_ok_",  "phone_ok"),
        ("col_phone_edit_","phone_edit"),
        ("col_help_",      "help"),
    ]

    action = None
    mid_str = None
    for prefix, act in callback_map:
        if data.startswith(prefix):
            action  = act
            mid_str = data[len(prefix):]
            break

    if action is None or not mid_str:
        return False

    try:
        mid = int(mid_str)
    except ValueError:
        return False

    dialog = get_dialog(mid)
    if not dialog:
        logger.warning("handle_callback: диалог для %d не найден", mid)
        return True  # Обработан (но нет данных)

    if action == "confirm":
        await _on_confirm(dialog, mid)
    elif action == "update":
        await _on_update(dialog, mid)
    elif action == "reject":
        await _on_reject_request(dialog, mid)
    elif action == "data_ok":
        await _on_data_confirmed(dialog, mid)
    elif action == "data_edit":
        await _on_data_edit(dialog, mid)
    elif action == "adm_ok":
        await _on_admin_approve_rejection(dialog, mid)
    elif action == "adm_deny":
        await _on_admin_deny_rejection(dialog, mid)
    elif action == "adm_send":
        await _on_admin_send(dialog, mid)
    elif action == "adm_call":
        await _on_admin_call_manager(dialog, mid)
    elif action == "adm_extend":
        await _on_admin_extend(dialog, mid)
    elif action == "name_ok":
        await _on_name_ok(dialog, mid)
    elif action == "name_edit":
        await _on_name_edit(dialog, mid)
    elif action == "phone_ok":
        await _on_phone_ok(dialog, mid)
    elif action == "phone_edit":
        await _on_phone_edit(dialog, mid)
    elif action == "help":
        await _on_help(dialog, mid)
    else:
        return False

    return True


async def handle_text_message(chat_id: int, text: str) -> bool:
    """Обрабатывает текстовое сообщение от менеджера. Возвращает True если обработан."""
    from collector.dialog_store import (
        get_dialog,
        STATE_AWAITING_DATA,
        STATE_AWAITING_REJECTION_REASON,
        STATE_AWAITING_CONFIRM,
        STATE_AWAITING_DATA_CONFIRM,
        STATE_REJECTED_PENDING_ADMIN,
        STATE_AWAITING_MANAGER_EXPLANATION,
        STATE_AWAITING_NAME_TEXT,
        STATE_AWAITING_PHONE_TEXT,
    )

    dialog = get_dialog(chat_id)
    if not dialog:
        return False

    state = dialog.get("state")

    if state in (STATE_AWAITING_DATA, STATE_AWAITING_REJECTION_REASON):
        await _on_data_received(dialog, chat_id, text)
        return True

    if state == STATE_AWAITING_NAME_TEXT:
        # Менеджер вводит новое имя клиента
        from collector.dialog_store import update_dialog
        contact = dict(dialog.get("current_contact") or {})
        contact["display_name"] = text.strip()
        contact["name_confirmations"] = 1
        _save_contact(dialog["client_name"], contact)
        update_dialog(
            chat_id,
            current_contact=contact,
            name_confirmed=True,
            awaiting_name_text=False,
            state=STATE_AWAITING_CONFIRM,
        )
        dialog["current_contact"] = contact
        dialog["name_confirmed"] = True
        await _send_msg(chat_id, f"✅ Имя обновлено: <b>{text.strip()}</b>")
        # Показываем следующий шаг
        msg_text, markup = _build_initial_message(dialog)
        await _send_msg(chat_id, msg_text, markup)
        return True

    if state == STATE_AWAITING_PHONE_TEXT:
        # Менеджер вводит новый телефон клиента
        from collector.dialog_store import update_dialog
        phone_clean = "".join(c for c in text if c.isdigit() or c == "+")
        contact = dict(dialog.get("current_contact") or {})
        contact["phone"] = phone_clean
        contact["whatsapp"] = phone_clean
        contact["phone_confirmations"] = 1
        contact["_needs_phone"] = False
        _save_contact(dialog["client_name"], contact)
        update_dialog(
            chat_id,
            current_contact=contact,
            phone_confirmed=True,
            awaiting_phone_text=False,
            state=STATE_AWAITING_CONFIRM,
        )
        dialog["current_contact"] = contact
        dialog["phone_confirmed"] = True
        await _send_msg(chat_id, f"✅ Телефон обновлён: <b>{phone_clean}</b>")
        msg_text, markup = _build_initial_message(dialog)
        await _send_msg(chat_id, msg_text, markup)
        return True

    if state == STATE_AWAITING_MANAGER_EXPLANATION:
        await _on_manager_explanation(dialog, chat_id, text)
        return True

    if state == STATE_AWAITING_CONFIRM:
        await _send_msg(
            chat_id,
            "⬆️ Используйте кнопки выше для ответа.",
        )
        return True

    if state == STATE_AWAITING_DATA_CONFIRM:
        await _send_msg(
            chat_id,
            "⬆️ Используйте кнопки выше для подтверждения или редактирования.",
        )
        return True

    if state == STATE_REJECTED_PENDING_ADMIN:
        await _send_msg(
            chat_id,
            "⏳ Ожидаем решения руководителя.",
        )
        return True

    return False


async def handle_voice_message(chat_id: int, file_id: str) -> bool:
    """Скачивает голосовое сообщение из Telegram и обрабатывает как текст.

    Args:
        chat_id: Telegram chat_id менеджера.
        file_id: file_id голосового сообщения из Telegram.

    Returns:
        True если обработано, False если нет активного диалога.
    """
    if not BOT_TOKEN:
        logger.warning("BOT_TOKEN не задан — голосовые сообщения недоступны")
        return False
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY не задан — транскрипция голоса недоступна")
        return False

    import tempfile as _tempfile

    # Получаем информацию о файле
    file_info_url = f"https://api.telegram.org/bot{BOT_TOKEN}/getFile?file_id={file_id}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(file_info_url)
        if resp.status_code != 200:
            logger.warning("getFile ошибка %d", resp.status_code)
            return False
        file_path = resp.json()["result"]["file_path"]
    except (httpx.RequestError, httpx.TimeoutException, KeyError) as e:
        logger.error("getFile сетевая ошибка: %s: %s", type(e).__name__, e or repr(e))
        return False

    # Скачиваем файл
    download_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
    tmp_path: Optional[str] = None
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            dl_resp = await client.get(download_url)
        if dl_resp.status_code != 200:
            logger.warning("Скачивание голоса ошибка %d", dl_resp.status_code)
            return False

        suffix = ".oga"
        tmp_fd, tmp_path = _tempfile.mkstemp(suffix=suffix, prefix="tg_voice_")
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(dl_resp.content)
        except OSError as e:
            logger.error("Ошибка записи голоса: %s", e)
            return False

        # Транскрибируем через Whisper
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        with open(tmp_path, "rb") as audio_f:
            files = {"file": (f"voice{suffix}", audio_f, "audio/ogg")}
            data = {"model": "whisper-1"}
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    whisper_resp = await client.post(
                        "https://api.openai.com/v1/audio/transcriptions",
                        headers=headers,
                        files=files,
                        data=data,
                    )
                if whisper_resp.status_code != 200:
                    logger.warning("Whisper ошибка %d", whisper_resp.status_code)
                    return False
                transcribed = whisper_resp.json().get("text", "").strip()
            except (httpx.RequestError, httpx.TimeoutException, KeyError) as e:
                logger.error("Whisper сетевая ошибка: %s: %s", type(e).__name__, e or repr(e))
                return False

    except (httpx.RequestError, httpx.TimeoutException) as e:
        logger.error("Скачивание голоса сетевая ошибка: %s: %s", type(e).__name__, e or repr(e))
        return False
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if not transcribed:
        logger.info("Whisper вернул пустую транскрипцию для chat_id=%d", chat_id)
        return False

    logger.info("Голосовое сообщение chat_id=%d транскрибировано: %s...", chat_id, transcribed[:60])
    return await handle_text_message(chat_id, transcribed)


async def send_reminders() -> None:
    """Отправляет напоминания по всем активным диалогам (запускается каждый час)."""
    from collector.dialog_store import (
        get_all_pending, update_dialog,
        STATE_AWAITING_CONFIRM,
        STATE_AWAITING_DATA,
        STATE_AWAITING_REJECTION_REASON,
        STATE_AWAITING_DATA_CONFIRM,
        STATE_REJECTED_PENDING_ADMIN,
        STATE_DEADLINE_SET,
    )

    pending = get_all_pending()
    now = datetime.now(TZ)

    for dialog in pending:
        mid   = dialog.get("manager_chat_id")
        state = dialog.get("state")
        if not mid:
            continue

        # Эскалируем админу молчаливые кейсы, чтобы менеджер не был единственной точкой отказа.
        created_str = dialog.get("created") or dialog.get("last_reminded")
        if created_str:
            try:
                created_dt = datetime.fromisoformat(created_str)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=TZ)
                age_hours = (now - created_dt).total_seconds() / 3600
                if age_hours > COLLECTOR_MANAGER_SILENCE_HOURS and not dialog.get("escalated_at"):
                    logger.warning(
                        "[%s] менеджер молчит %.1f ч — эскалация админу (state=%s)",
                        dialog.get("client_name"), age_hours, state,
                    )
                    update_dialog(
                        mid,
                        control_deadline=now.isoformat(),
                        escalated_at=now.isoformat(),
                    )
                    dialog["control_deadline"] = now.isoformat()
                    dialog["escalated_at"] = now.isoformat()
                    await _send_msg(
                        mid,
                        f"⏱️ Диалог по клиенту <b>{dialog.get('client_name')}</b> "
                        f"передан руководителю (нет ответа {int(age_hours)} ч).",
                    )
                    await _send_admin_timeout_escalation(dialog, age_hours)
                    continue
            except (ValueError, TypeError):
                pass
        if dialog.get("escalated_at"):
            continue

        # Проверяем время последнего напоминания
        last_str = dialog.get("last_reminded")
        if last_str:
            try:
                last_dt = datetime.fromisoformat(last_str)
                if (now - last_dt).total_seconds() < COLLECTOR_REMINDER_HOURS * 3600:
                    continue
            except (ValueError, TypeError):
                pass

        count = (dialog.get("remind_count") or 0) + 1
        update_dialog(mid, remind_count=count, last_reminded=now.isoformat())

        if state in (STATE_AWAITING_CONFIRM, STATE_DEADLINE_SET):
            # Проверяем дедлайн
            if state == STATE_DEADLINE_SET:
                deadline_str = dialog.get("deadline")
                if deadline_str:
                    try:
                        deadline_date = datetime.fromisoformat(deadline_str).date()
                        if now.date() > deadline_date:
                            await _send_deadline_exceeded(dialog)
                            continue
                    except (ValueError, TypeError):
                        pass

            remind_text = _build_reminder_text(dialog, count)
            _, markup = _build_initial_message(dialog)
            await _send_msg(mid, remind_text + "\n\nПожалуйста, нажмите одну из кнопок:", markup)

        elif state == STATE_AWAITING_DATA:
            await _send_msg(
                mid,
                f"⏰ Напоминание #{count}\n\n"
                f"Клиент: <b>{dialog['client_name']}</b>\n"
                f"Отправьте обновлённые данные контакта текстом.",
                _inline([[("❓ Не понимаю, что ответить", f"col_help_{mid}")]]),
            )

        elif state == STATE_AWAITING_REJECTION_REASON:
            await _send_msg(
                mid,
                f"⏰ Напоминание #{count}\n\n"
                f"Клиент: <b>{dialog['client_name']}</b>\n"
                f"Напишите причину отказа от отправки сообщения.",
                _inline([[("❓ Не понимаю, что ответить", f"col_help_{mid}")]]),
            )

        elif state == STATE_AWAITING_DATA_CONFIRM:
            confirm_text, markup = _build_data_confirm_message(dialog)
            await _send_msg(
                mid,
                f"⏰ Напоминание #{count}\n\n{confirm_text}",
                markup,
            )

        elif state == STATE_REJECTED_PENDING_ADMIN:
            # Напоминаем администраторам, не менеджеру
            admin_ids = _get_admin_ids()
            admin_remind = (
                f"⏰ Напоминание #{count} — ожидаем решения\n\n"
                f"Менеджер: {dialog['manager_name']}\n"
                f"Клиент: {dialog['client_name']}\n"
                f"Причина отказа: {dialog.get('rejection_reason', '(не указана)')}"
            )
            markup_adm = _inline([
                [
                    ("✅ Принять отказ",                   f"col_adm_ok_{mid}"),
                    ("❌ Установить дедлайн",               f"col_adm_deny_{mid}"),
                ],
            ])
            for admin_id in admin_ids:
                await _send_msg(admin_id, admin_remind, markup_adm)

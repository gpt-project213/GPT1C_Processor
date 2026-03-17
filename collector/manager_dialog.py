#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/manager_dialog.py
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
COLLECTOR_REMINDER_HOURS = float(os.getenv("COLLECTOR_REMINDER_HOURS", "1"))
COLLECTOR_DEADLINE_DAYS  = int(os.getenv("COLLECTOR_DEADLINE_DAYS", "5"))

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
        logger.error("Telegram %s сетевая ошибка: %s", method, e)
        return None


async def _send_msg(
    chat_id: int,
    text: str,
    markup: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Отправляет сообщение; возвращает message_id или None."""
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
    """Строит начальное сообщение с кнопками подтверждения/обновления/отказа."""
    mid = dialog["manager_chat_id"]
    contact = dialog.get("current_contact") or {}
    phone = contact.get("whatsapp") or contact.get("phone", "не указан")
    person = contact.get("contact_person", "не указан")

    text = (
        f"📋 <b>AI Коллектор — запрос подтверждения</b>\n\n"
        f"Менеджер: {dialog['manager_name']}\n"
        f"Клиент: {dialog['client_name']}\n"
        f"Просрочка: {dialog['days']} дн. | Сумма: {_fmt_amount(dialog['amount'])} тг\n"
        f"Уровень давления: {dialog['level']}\n\n"
        f"Данные в базе:\n"
        f"  📞 Телефон/WhatsApp: {phone}\n"
        f"  👤 Контакт: {person}\n\n"
        f"Данные актуальны? Отправить WhatsApp-напоминание?"
    )
    markup = _inline([
        [
            ("✅ Актуально — отправить", f"col_confirm_{mid}"),
            ("✏️ Обновить данные",       f"col_update_{mid}"),
        ],
        [
            ("❌ Не отправлять сейчас",  f"col_reject_{mid}"),
        ],
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
        f"Просрочка: {dialog['days']} дн. | Сумма: {_fmt_amount(dialog['amount'])} тг"
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
        logger.error("DeepSeek сетевая ошибка: %s", e)
        return {}
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error("DeepSeek неверный ответ: %s", e)
        return {}


async def _parse_rejection_reason(text: str) -> str:
    """Очищает/суммаризирует причину отказа через AI. Fallback — оригинал."""
    if not DEEPSEEK_API_KEY:
        return text.strip()

    url = "https://api.deepseek.com/v1/chat/completions"
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты помощник. Сформулируй причину отказа менеджера кратко "
                    "(1-2 предложения). Отвечай только текстом, без лишних слов."
                ),
            },
            {"role": "user", "content": text},
        ],
        "max_tokens": 150,
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            return text.strip()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except (httpx.RequestError, httpx.TimeoutException, json.JSONDecodeError,
            KeyError, IndexError):
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
    """Сохраняет обновлённый контакт в debtors_contacts.json."""
    contacts_path = _ROOT / "collector" / "debtors_contacts.json"
    try:
        if contacts_path.exists():
            with open(contacts_path, encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)
        else:
            data = {}
    except (OSError, json.JSONDecodeError):
        data = {}

    # Сохраняем _comment если есть
    comment = data.get("_comment")
    data[client_name] = contact
    if comment is not None:
        data["_comment"] = comment

    import tempfile
    contacts_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(contacts_path.parent),
        suffix=".tmp",
        prefix="debtors_contacts_",
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, str(contacts_path))
    except (OSError, TypeError, ValueError) as e:
        logger.error("Ошибка сохранения контакта: %s", e)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ─── WhatsApp + Notification ──────────────────────────────────────────────────

async def _send_whatsapp_and_notify(dialog: Dict[str, Any]) -> None:
    """Отправляет WhatsApp-сообщение и уведомляет всех наблюдателей."""
    from collector.collection_agent import generate_message
    from collector.communications import send_whatsapp, get_observer_ids
    from collector.collections_db import update_after_contact

    client_name  = dialog["client_name"]
    manager_name = dialog["manager_name"]
    contact      = dialog.get("current_contact") or {}
    level        = dialog["level"]
    days         = dialog["days"]
    amount       = dialog["amount"]
    language     = contact.get("language", "ru")
    phone        = contact.get("whatsapp") or contact.get("phone", "")

    # Генерируем текст сообщения
    try:
        message_text = generate_message(
            client_name=client_name,
            debt_amount=amount,
            days_overdue=days,
            level=level,
            language=language,
        )
    except Exception as e:
        logger.error("[%s] generate_message ошибка: %s", client_name, e)
        message_text = f"Уважаемый клиент, у вас просроченная задолженность {_fmt_amount(amount)} тг."

    # Отправка WhatsApp
    wa_ok = False
    if phone:
        try:
            wa_ok = send_whatsapp(phone, message_text)
        except Exception as e:
            logger.error("[%s] send_whatsapp ошибка: %s", client_name, e)

    # Обновляем БД если отправлено
    if wa_ok:
        try:
            update_after_contact(client_name, "whatsapp", level, message_text)
        except Exception as e:
            logger.error("[%s] update_after_contact ошибка: %s", client_name, e)

    # Уведомляем наблюдателей
    status_icon = "✅" if wa_ok else "⚠️"
    status_text = "WhatsApp отправлен" if wa_ok else "WhatsApp НЕ отправлен (ошибка)"
    preview = message_text[:300] + ("..." if len(message_text) > 300 else "")
    notify_text = (
        f"📨 <b>AI Коллектор — подтверждение отправки</b>\n\n"
        f"{status_icon} {status_text}\n\n"
        f"Клиент: {client_name}\n"
        f"Просрочка: {days} дн. | Сумма: {_fmt_amount(amount)} тг\n"
        f"Уровень: {level}\n\n"
        f"Текст сообщения:\n<i>{preview}</i>"
    )

    try:
        observer_ids = get_observer_ids(manager_name)
    except Exception as e:
        logger.error("get_observer_ids ошибка: %s", e)
        observer_ids = []

    for obs_id in observer_ids:
        await _send_msg(obs_id, notify_text)


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
    from collector.dialog_store import update_dialog, STATE_CONFIRMED
    update_dialog(mid, state=STATE_CONFIRMED)
    dialog["state"] = STATE_CONFIRMED
    await _send_whatsapp_and_notify(dialog)


async def _on_update(dialog: Dict[str, Any], mid: int) -> None:
    """Менеджер хочет обновить данные."""
    from collector.dialog_store import update_dialog, STATE_AWAITING_DATA
    update_dialog(mid, state=STATE_AWAITING_DATA)
    await _send_msg(
        mid,
        "✏️ <b>Обновление данных</b>\n\n"
        "Отправьте обновлённые данные контакта текстом.\n"
        "Например: телефон +77011234567, контакт Иванов Иван",
    )


async def _on_reject_request(dialog: Dict[str, Any], mid: int) -> None:
    """Менеджер хочет отказаться от отправки."""
    from collector.dialog_store import update_dialog, STATE_AWAITING_REJECTION_REASON
    update_dialog(mid, state=STATE_AWAITING_REJECTION_REASON)
    await _send_msg(
        mid,
        "❌ <b>Причина отказа</b>\n\n"
        "Напишите причину, по которой не нужно отправлять сообщение клиенту.\n"
        "Причина будет передана руководителю.",
    )


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
    from collector.dialog_store import update_dialog, STATE_CONFIRMED
    current  = dialog.get("current_contact") or {}
    proposed = dialog.get("proposed_contact") or {}
    merged   = {**current, **proposed}

    _save_contact(dialog["client_name"], merged)
    update_dialog(mid, state=STATE_CONFIRMED, current_contact=merged)
    dialog["state"]           = STATE_CONFIRMED
    dialog["current_contact"] = merged

    await _send_msg(mid, "✅ Данные обновлены.")
    await _send_whatsapp_and_notify(dialog)


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

    logger.info("[%s] диалог запущен с менеджером %s", client["name"], manager_name)


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
    )

    dialog = get_dialog(chat_id)
    if not dialog:
        return False

    state = dialog.get("state")

    if state in (STATE_AWAITING_DATA, STATE_AWAITING_REJECTION_REASON):
        await _on_data_received(dialog, chat_id, text)
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
            )

        elif state == STATE_AWAITING_REJECTION_REASON:
            await _send_msg(
                mid,
                f"⏰ Напоминание #{count}\n\n"
                f"Клиент: <b>{dialog['client_name']}</b>\n"
                f"Напишите причину отказа от отправки сообщения.",
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

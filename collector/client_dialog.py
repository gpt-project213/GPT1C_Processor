#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/client_dialog.py
Управление диалогами с должниками через WhatsApp.

Версия: 1.0.8 (2026-04-23)

v1.0.8 (2026-04-23): мягкая обработка ответов клиентов: soft_positive /
  promise_schedule / paid_claim, без ложной фиксации обещаний и без повторной ссылки на 1С.

Хранилище: logs/collector_client_dialogs.json
Ключ: номер телефона (цифры, без +, без @c.us)

Жизненный цикл диалога:
  start_client_dialog() → handle_incoming() → escalate_to_manager()
  state: active | escalated | closed
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from zoneinfo import ZoneInfo

load_dotenv(
    dotenv_path=Path(__file__).resolve().parent.parent / ".env",
    encoding="utf-8-sig",
    override=False,
)

TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))
COMPANY_NAME = os.getenv("COMPANY_NAME", "Минбаракат")

_ROOT = Path(__file__).resolve().parent.parent
_DIALOGS_PATH = _ROOT / "logs" / "collector_client_dialogs.json"
_DELETION_QUEUE_PATH = _ROOT / "logs" / "deletion_queue.json"

logger = logging.getLogger(__name__)


def _schedule_tg_deletion(chat_id: int, message_id: int, delay_hours: int = 24) -> None:
    """Добавляет Telegram-сообщение в очередь авто-удаления (deletion_queue.json)."""
    try:
        now = time.time()
        try:
            with open(_DELETION_QUEUE_PATH, encoding="utf-8") as f:
                queue = json.load(f)
        except (OSError, json.JSONDecodeError):
            queue = {"jobs": []}
        jobs = queue.get("jobs", [])
        jobs.append({
            "chat_id": chat_id,
            "message_id": message_id,
            "due_ts": now + delay_hours * 3600,
            "msg_ts": now,
            "scheduled_at": now,
        })
        queue["jobs"] = jobs[-5000:]
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(_DELETION_QUEUE_PATH.parent), suffix=".tmp")
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(queue, f, ensure_ascii=False)
        os.replace(tmp_path, str(_DELETION_QUEUE_PATH))
    except Exception as e:
        logger.warning("_schedule_tg_deletion error: %s", e)


# ─── Хранилище ───────────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Текущее время в ISO формате (Asia/Almaty)."""
    return datetime.now(TZ).isoformat()


def _load_client_dialogs() -> Dict[str, Any]:
    """Загружает все клиентские диалоги из JSON-файла."""
    if not _DIALOGS_PATH.exists():
        return {}
    try:
        with open(_DIALOGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_client_dialogs(dialogs: Dict[str, Any]) -> None:
    """Атомарно сохраняет клиентские диалоги в JSON-файл через tempfile."""
    _DIALOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(_DIALOGS_PATH.parent),
        suffix=".tmp",
        prefix="client_dialogs_",
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(dialogs, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, str(_DIALOGS_PATH))
    except (OSError, TypeError, ValueError):
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _get_client_dialog(phone: str) -> Optional[Dict[str, Any]]:
    """Возвращает диалог клиента по номеру телефона или None."""
    return _load_client_dialogs().get(phone)


def _set_client_dialog(phone: str, dialog: Dict[str, Any]) -> None:
    """Сохраняет диалог клиента."""
    dialogs = _load_client_dialogs()
    dialogs[phone] = dialog
    _save_client_dialogs(dialogs)


# ─── Языковая детекция ───────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Определяет язык текста по казахским символам.

    Returns:
        'kz' если найдено 2+ казахских специфичных символа, иначе 'ru'.
    """
    kz_chars = set("әғқңөұүі")
    count = sum(1 for c in text.lower() if c in kz_chars)
    return "kz" if count >= 2 else "ru"


# ─── Telegram уведомление менеджера ──────────────────────────────────────────

async def _send_tg(chat_id: int, text: str, reply_markup: Any = None) -> None:
    """Отправляет Telegram-сообщение менеджеру и планирует авто-удаление через 24ч."""
    try:
        if reply_markup is not None:
            from collector.communications import send_telegram_with_markup
            ok = await send_telegram_with_markup(chat_id, text, reply_markup)
            # send_telegram_with_markup не возвращает message_id — удаление не планируем
            # (сообщения с кнопками удалять нежелательно пока кнопки активны)
        else:
            from collector.communications import send_telegram
            message_id = await send_telegram(chat_id, text)
            if message_id:
                _schedule_tg_deletion(chat_id, message_id, delay_hours=24)
    except Exception as e:
        logger.error("Ошибка отправки Telegram chat_id=%d: %s", chat_id, e)


def _gender_pronoun(manager_name: str) -> str:
    """Возвращает 'Она' или 'Он' по окончанию имени менеджера."""
    name = manager_name.strip()
    if name.endswith("а") or name.endswith("я"):
        return "Она"
    return "Он"


def _report_date_context(dialog: Optional[Dict[str, Any]] = None) -> str:
    """Возвращает дату отчёта дебиторки для честного ответа клиенту."""
    if dialog and dialog.get("report_date"):
        return str(dialog["report_date"])
    try:
        from collector.debt_monitor import load_latest_debt_json
        data = load_latest_debt_json()
        period_max = data.get("period_max") if isinstance(data, dict) else None
        if period_max:
            return str(period_max)
        if isinstance(data, dict):
            dates = [
                str(c.get("_period_max"))
                for c in data.get("clients", [])
                if isinstance(c, dict) and c.get("_period_max")
            ]
            if dates:
                return max(dates)
    except Exception as e:
        logger.debug("Не удалось определить дату отчёта дебиторки: %s", e)
    return "последнего отчёта 1С"


def _mentions_recent_unposted_payment(text: str) -> bool:
    """True, если клиент говорит про недавнюю оплату, которая могла не попасть в 1С."""
    t = text.lower()
    markers = (
        "qr", "куар", "киар", "оплат", "упад", "поступ", "за выходные",
        "сегодня", "завтра", "не разнес", "не провел", "не прошло",
    )
    return any(m in t for m in markers) and any(
        m in t for m in ("оплат", "qr", "куар", "киар", "упад", "поступ")
    )


def _fmt_amount(amount: float) -> str:
    """Форматирует сумму: 500000 → '500 000'."""
    return f"{amount:,.0f}".replace(",", " ")


def _fmt_date_display(raw: Optional[str]) -> str:
    """Форматирует ISO-дату для клиентского сообщения."""
    if not raw:
        return ""
    try:
        from datetime import date as _date
        return _date.fromisoformat(str(raw)).strftime("%d.%m.%Y")
    except (TypeError, ValueError):
        return str(raw)


def _payment_schedule_label(raw: Any) -> str:
    """Человекочитаемая формулировка графика оплаты."""
    value = str(raw or "").strip().lower()
    if value in {"daily", "every_day", "ежедневно", "каждый день"}:
        return "ежедневными частичными платежами"
    if value in {"partial", "parts", "частями", "частично"}:
        return "частями"
    return str(raw or "частичными платежами")


def _build_escalation_text(
    dialog: Dict[str, Any],
    reason: str,
    summary: str,
) -> str:
    """Строит текст уведомления менеджеру об эскалации."""
    name = dialog.get("client_name", "—")
    amount = dialog.get("amount", 0)
    days = dialog.get("days", 0)
    exchanges = dialog.get("exchanges", [])
    exchange_count = dialog.get("exchange_count", 0)
    manager_name = dialog.get("manager_name", "менеджеру")
    pronoun = _gender_pronoun(manager_name)

    # Компактная история: только клиентские сообщения в полном виде,
    # бот — одна строка-заглушка чтобы не занимать место.
    last_exchanges = exchanges[-6:] if len(exchanges) > 6 else exchanges
    exchange_lines = []
    bot_shown = False
    for ex in last_exchanges:
        role = ex.get("role", "")
        msg = ex.get("text", "")
        if role == "bot":
            if not bot_shown:
                exchange_lines.append("🤖 Бот: напоминание отправлено")
                bot_shown = True
        elif role == "client":
            exchange_lines.append(f"👤 {msg[:200]}")

    exchanges_block = "\n".join(exchange_lines) if exchange_lines else "(нет переписки)"

    # Обещание если есть
    promise_date = dialog.get("promise_date")
    promise_line = f"\n📅 Обещание оплаты: {promise_date}" if promise_date else ""
    schedule = dialog.get("payment_schedule")
    schedule_line = f"\n🧾 График: {_payment_schedule_label(schedule)}" if schedule else ""

    # Описание намерения
    intent_map = {
        "promise":              "обещал оплатить",
        "promise_without_date": "готов платить, но не назвал дату",
        "promise_schedule":     "предложил график частичных платежей",
        "paid_claim":           "сообщил, что уже оплатил",
        "soft_positive":        "готов платить, но без точной суммы/графика",
        "refusal":              "отказывается платить",
        "delay_request":        "просит отсрочку",
        "question":             "задаёт вопрос о товарах/доставке",
        "identity_question":    "спрашивает кто пишет / откуда номер",
        "unclear":              "неясное намерение",
        "off_topic":            "уходит от темы",
        "requires_human":       "требуется живой человек",
        "limit_reached":        "исчерпан лимит обменов",
    }
    intent_desc = intent_map.get(reason, reason)

    return (
        f"📋 <b>{name}</b> — требуется участие {manager_name}\n\n"
        f"💰 {_fmt_amount(amount)} тг | {days} дн. просрочки{promise_line}{schedule_line}\n"
        f"📌 {intent_desc}\n\n"
        f"💬 Переписка:\n{exchanges_block}\n\n"
        f"⚠️ {summary}"
    )


async def escalate_to_manager(
    dialog: Dict[str, Any],
    reason: str,
    summary: str,
    phone: str,
) -> None:
    """Эскалирует диалог менеджеру и всем наблюдателям.

    Args:
        dialog:  Текущий диалог клиента.
        reason:  Код причины эскалации (promise/refusal/delay_request/...).
        summary: Человекочитаемое описание.
        phone:   Номер телефона клиента.
    """
    manager_chat_id = dialog.get("manager_chat_id")
    manager_name = dialog.get("manager_name", "")

    # Обновляем состояние диалога
    dialog["state"] = "escalated"
    _set_client_dialog(phone, dialog)

    text = _build_escalation_text(dialog, reason, summary)

    # Получаем всех наблюдателей
    try:
        from collector.communications import get_observer_ids
        observer_ids = get_observer_ids(manager_name)
    except Exception as e:
        logger.error("get_observer_ids ошибка: %s", e)
        observer_ids = []

    # Если менеджер не в списке — добавляем его
    if manager_chat_id and manager_chat_id not in observer_ids:
        observer_ids = [manager_chat_id] + observer_ids

    for obs_id in observer_ids:
        await _send_tg(obs_id, text)

    logger.info(
        "[%s] диалог эскалирован менеджеру %s (reason=%s)",
        dialog.get("client_name"), manager_name, reason,
    )


# ─── Ответ бота клиенту ──────────────────────────────────────────────────────

async def _reply_to_client(phone: str, text: str) -> None:
    """Отправляет WhatsApp-ответ клиенту."""
    try:
        from collector.communications import send_whatsapp
        ok = send_whatsapp(phone, text)
        if not ok:
            logger.warning("Не удалось отправить ответ клиенту %s", phone)
    except Exception as e:
        logger.error("Ошибка ответа клиенту %s: %s", phone, e)


# ─── Публичный API ───────────────────────────────────────────────────────────

async def start_client_dialog(
    phone: str,
    client_name: str,
    manager_name: str,
    manager_chat_id: int,
    level: int,
    days: int,
    amount: float,
    message_text: str,
    report_date: str = "",
) -> None:
    """Регистрирует диалог с клиентом после отправки WhatsApp-сообщения.

    Args:
        phone:          Номер телефона (только цифры).
        client_name:    Отображаемое имя клиента.
        manager_name:   Имя менеджера.
        manager_chat_id: Telegram chat_id менеджера.
        level:          Уровень давления 1–5.
        days:           Дней просрочки.
        amount:         Сумма долга.
        message_text:   Текст отправленного сообщения.
    """
    # Нормализуем телефон
    phone_clean = "".join(c for c in phone if c.isdigit())

    now = _now_iso()
    dialog: Dict[str, Any] = {
        "client_name":       client_name,
        "manager_name":      manager_name,
        "manager_chat_id":   manager_chat_id,
        "level":             level,
        "days":              days,
        "amount":            amount,
        "report_date":       report_date,
        "exchanges": [
            {"role": "bot", "text": message_text, "timestamp": now}
        ],
        "exchange_count":    0,  # только ходы клиента
        "state":             "active",
        "created":           now,
        "last_activity":     now,
        "phone_silent_cycles": 0,
        "off_topic_count":   0,
    }
    _set_client_dialog(phone_clean, dialog)
    logger.info(
        "[%s] клиентский диалог зарегистрирован (phone=%s, level=%d)",
        client_name, phone_clean, level,
    )


async def handle_incoming(phone: str, text: str) -> None:
    """Обрабатывает входящее WhatsApp-сообщение от клиента.

    Args:
        phone: Номер телефона клиента (цифры, без @c.us).
        text:  Текст сообщения.
    """
    phone_clean = "".join(c for c in phone if c.isdigit())

    dialogs = _load_client_dialogs()
    dialog = dialogs.get(phone_clean)

    if not dialog:
        logger.info("Неизвестный клиент %s — входящее сообщение проигнорировано", phone_clean)
        return

    if dialog.get("state") != "active":
        logger.info(
            "Диалог %s в состоянии %s — входящее игнорируется",
            phone_clean, dialog.get("state"),
        )
        return

    now = _now_iso()
    manager_name = dialog.get("manager_name", "менеджеру")
    manager_chat_id = dialog.get("manager_chat_id")
    level = dialog.get("level", 1)
    client_name = dialog.get("client_name", "Клиент")
    pronoun = _gender_pronoun(manager_name)

    # Добавляем в историю
    dialog["exchanges"].append({"role": "client", "text": text, "timestamp": now})
    dialog["exchange_count"] = dialog.get("exchange_count", 0) + 1
    dialog["last_activity"] = now
    exchange_count = dialog["exchange_count"]

    # Определяем язык
    language = detect_language(text)

    if _mentions_recent_unposted_payment(text):
        reply = (
            "Спасибо. Если оплата уже прошла, пришлите, пожалуйста, чек или дату и сумму платежа. "
            f"Передадим информацию менеджеру {manager_name}."
        )
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)
        await _reply_to_client(phone_clean, reply)
        return

    # Анализируем через DeepSeek (синхронный вызов — выносим в поток)
    try:
        from collector.collection_agent import analyze_response
        history = dialog.get("exchanges", [])
        analysis = await asyncio.to_thread(
            analyze_response,
            response_text=text,
            manager_name=manager_name,
            conversation_history=history,
        )
    except Exception as e:
        logger.error("analyze_response ошибка: %s", e)
        analysis = {
            "intent": "unclear",
            "promise_date": None,
            "promise_amount": None,
            "requires_human": True,
            "suggested_reply": "",
        }

    intent = analysis.get("intent", "unclear")
    requires_human = analysis.get("requires_human", False)
    suggested_reply = analysis.get("suggested_reply", "")
    promise_date = analysis.get("promise_date")
    promise_amount = analysis.get("promise_amount")
    payment_schedule = analysis.get("payment_schedule") or analysis.get("schedule")

    logger.info(
        "[%s] intent=%s requires_human=%s exchange_count=%d",
        client_name, intent, requires_human, exchange_count,
    )

    # ─── Маршрутизация по намерению ──────────────────────────────────────────

    if exchange_count >= 5:
        # Лимит обменов — передаём менеджеру
        reply = (
            f"Спасибо, передаю вас менеджеру {manager_name}. "
            f"{pronoun} свяжется с вами в ближайшее время."
        )
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)
        await _reply_to_client(phone_clean, reply)
        await escalate_to_manager(
            dialog, "limit_reached",
            f"Достигнут лимит обменов ({exchange_count})", phone_clean,
        )
        return

    if requires_human:
        reply = (
            f"Спасибо, передаю вас менеджеру {manager_name}. "
            f"{pronoun} свяжется с вами в ближайшее время."
        )
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)
        await _reply_to_client(phone_clean, reply)
        await escalate_to_manager(
            dialog, "requires_human",
            "Клиент требует живого человека или ситуация неоднозначна", phone_clean,
        )
        return

    if intent == "paid_claim":
        reply = suggested_reply if suggested_reply else (
            "Спасибо. Если оплата уже прошла, пришлите, пожалуйста, чек или дату и сумму платежа. "
            f"Передадим информацию менеджеру {manager_name}."
        )
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)
        await _reply_to_client(phone_clean, reply)
        return

    if intent == "promise_schedule":
        schedule_code = str(payment_schedule or "partial")
        schedule_text = _payment_schedule_label(schedule_code)
        dialog["payment_schedule"] = schedule_code
        if promise_date:
            dialog["promise_date"] = promise_date
        if promise_amount:
            dialog["promise_amount"] = promise_amount
        first_payment = f" Первый платёж ждём до {_fmt_date_display(promise_date)}." if promise_date else ""
        reply = suggested_reply if suggested_reply else (
            f"Принято: оплата будет {schedule_text}.{first_payment} "
            "Как оплатите — пришлите, пожалуйста, чек."
        )
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        dialog["state"] = "escalated"
        _set_client_dialog(phone_clean, dialog)
        await _reply_to_client(phone_clean, reply)
        summary = f"Клиент предложил график оплаты: {schedule_text}"
        if promise_date:
            summary += f", первый платёж до {promise_date}"
        if promise_amount:
            summary += f", сумма {promise_amount}"
        await escalate_to_manager(dialog, "promise_schedule", summary, phone_clean)
        return

    if intent == "soft_positive":
        if exchange_count >= 2:
            reply = (
                "Понял вас. Тогда ждём ближайшую оплату. "
                "Как оплатите — пришлите, пожалуйста, чек."
            )
            dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
            dialog["state"] = "escalated"
            _set_client_dialog(phone_clean, dialog)
            await _reply_to_client(phone_clean, reply)
            await escalate_to_manager(
                dialog, "soft_positive",
                "Клиент готов платить, но точную сумму или график не назвал", phone_clean,
            )
            return
        reply = suggested_reply if suggested_reply else (
            "Спасибо, понял. Когда планируете первый платёж и примерно какая сумма?"
        )
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)
        await _reply_to_client(phone_clean, reply)
        return

    if intent == "promise":
        # Сохраняем дату обещания в диалог
        if promise_date:
            dialog["promise_date"] = promise_date
        if promise_amount:
            dialog["promise_amount"] = promise_amount
        date_str = f" до {promise_date}" if promise_date else ""
        date_display = _fmt_date_display(promise_date)
        if promise_amount:
            reply = (
                f"Спасибо, договорённость зафиксировал: оплата до {date_display} "
                f"на сумму {_fmt_amount(float(promise_amount))} тг. "
                "Как оплатите — пришлите чек, пожалуйста."
            )
        elif promise_date:
            reply = (
                f"Спасибо, понял. Тогда ждём оплату до {date_display}. "
                "Как оплатите — пришлите, пожалуйста, чек."
            )
        else:
            reply = (
                "Спасибо, понял. Как оплатите — пришлите, пожалуйста, чек."
            )
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        dialog["state"] = "escalated"
        _set_client_dialog(phone_clean, dialog)
        await _reply_to_client(phone_clean, reply)
        summary = f"Клиент обещал оплатить{date_str}"
        if promise_amount:
            summary += f" на сумму {promise_amount}"
        await escalate_to_manager(dialog, "promise", summary, phone_clean)
        return

    if intent == "refusal":
        if exchange_count >= 3:
            reply = (
                f"Понимаю. Передаю информацию менеджеру {manager_name}. "
                f"{pronoun} свяжется с вами для уточнения деталей."
            )
            dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
            _set_client_dialog(phone_clean, dialog)
            await _reply_to_client(phone_clean, reply)
            await escalate_to_manager(
                dialog, "refusal",
                "Клиент отказывается оплачивать", phone_clean,
            )
            return
        else:
            # Первая или вторая попытка — пробуем ещё раз
            if suggested_reply:
                reply = suggested_reply
            else:
                reply = (
                    f"Понимаю вашу ситуацию. Тем не менее, задолженность требует погашения. "
                    f"Можете ли вы указать конкретную дату, когда сможете оплатить?"
                )
            dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
            _set_client_dialog(phone_clean, dialog)
            await _reply_to_client(phone_clean, reply)
            return

    if intent == "delay_request":
        # Уточняем конкретную дату — диалог остаётся активным
        reply = suggested_reply if suggested_reply else (
            "Пожалуйста, укажите конкретную дату оплаты — например, 15.04.2026."
        )
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)
        await _reply_to_client(phone_clean, reply)
        return

    if intent == "question":
        reply = (
            f"Спасибо, передаю вас менеджеру {manager_name}. "
            f"{pronoun} свяжется с вами и ответит на все вопросы."
        )
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)
        await _reply_to_client(phone_clean, reply)
        await escalate_to_manager(
            dialog, "question",
            "Клиент задаёт вопрос о товарах/доставке/счёте", phone_clean,
        )
        return

    if intent == "identity_question":
        # Клиент спрашивает кто пишет — представляемся: компания, точка, менеджер.
        # Дату оплаты НЕ требуем — сначала устанавливаем доверие.
        reply = (
            f"Здравствуйте! Пишет {COMPANY_NAME}, отдел по работе с клиентами.\n"
            f"Обращаемся по задолженности {client_name}.\n"
            f"Ваш менеджер — {manager_name}. Если есть вопросы — можете написать напрямую."
        )
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)
        await _reply_to_client(phone_clean, reply)
        return

    if intent == "promise_without_date":
        # Клиент подтверждает готовность, но без даты — просим уточнить
        reply = suggested_reply if suggested_reply else (
            "Спасибо, понял. Когда планируете первый платёж?"
        )
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)
        await _reply_to_client(phone_clean, reply)
        return

    # intent == "unclear" — возможно off_topic
    off_topic_count = dialog.get("off_topic_count", 0)
    if off_topic_count == 0:
        # Первый раз — мягко возвращаем к теме долга
        reply = (
            "Благодарим за ответ. Вернёмся к вопросу задолженности — "
            "когда вы сможете произвести оплату?"
        )
        dialog["off_topic_count"] = 1
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)
        await _reply_to_client(phone_clean, reply)
    else:
        # Второй раз — эскалируем; пустой bot reply не сохраняем
        _set_client_dialog(phone_clean, dialog)
        await escalate_to_manager(
            dialog, "off_topic",
            "Клиент уклоняется от темы задолженности", phone_clean,
        )


async def mark_phone_silent(phone: str) -> None:
    """Увеличивает счётчик циклов без ответа. При достижении 3 — уведомляет менеджера.

    Вызывается из collections_engine когда клиент не ответил в течение 24 часов.

    Args:
        phone: Номер телефона клиента.
    """
    phone_clean = "".join(c for c in phone if c.isdigit())
    dialog = _get_client_dialog(phone_clean)
    if not dialog:
        return

    dialog["phone_silent_cycles"] = dialog.get("phone_silent_cycles", 0) + 1
    silent_cycles = dialog["phone_silent_cycles"]
    _set_client_dialog(phone_clean, dialog)

    if silent_cycles >= 3:
        manager_chat_id = dialog.get("manager_chat_id")
        client_name = dialog.get("client_name", "—")
        manager_name = dialog.get("manager_name", "")

        msg = (
            f"⚠️ Клиент <b>{client_name}</b> не отвечает {silent_cycles} цикла.\n"
            f"Телефон +{phone_clean} — возможно недействителен.\n"
            f"Уточните контактный номер."
        )

        try:
            from collector.communications import get_observer_ids
            observer_ids = get_observer_ids(manager_name)
        except Exception as e:
            logger.error("get_observer_ids ошибка: %s", e)
            observer_ids = []

        if manager_chat_id and manager_chat_id not in observer_ids:
            observer_ids = [manager_chat_id] + observer_ids

        for obs_id in observer_ids:
            await _send_tg(obs_id, msg)

        logger.info(
            "[%s] клиент молчит %d циклов — менеджер уведомлён",
            client_name, silent_cycles,
        )

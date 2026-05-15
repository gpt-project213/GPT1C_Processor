#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/client_dialog.py
Управление диалогами с должниками через WhatsApp.

Версия: 1.1.11 (2026-05-15)

v1.1.11 (2026-05-15): client-promise из WA-диалога теперь синхронизируется
  с wa_agreed_promises.json через record_client_promise(). Раньше handler
  10:30 (check_broken_agreed_deadlines) видел только manager-promise — клиенты
  типа Гриль Косши / МАСТЕР-КОНДИТЕР с overdue dialog-promise оставались
  невидимыми. F-B1.

v1.1.10 (2026-05-15): silent active dialogs with zero client replies stop
  blocking the next daily cycle forever; after 24h the bot may resend and
  supersede the stale outreach instead of hiding the client from preview.

v1.1.9 (2026-05-14): escalated client dialogs now set a short
  wa_dialog_suppress cooldown by default, so promise/soft-positive/manual
  handoff cases do not re-enter the next preview cycle immediately.

v1.1.8 (2026-05-13): escalation text now separates debt age, contractual
  deferral and effective overdue so deferred clients are not described as
  overdue before their payment term expires.

v1.1.6 (2026-05-12): _is_service_request — regex с \b вместо substring; "расчет"
  больше не ложно срабатывает как "счет". unclear второй раз отправляет
  финальный ответ клиенту перед эскалацией.

v1.1.5 (2026-05-11): neutral acknowledgements в soft_positive больше не
  трактуются как готовность платить; "Ок/Хорошо/Понял" теперь ведут к
  уточняющему вопросу без эскалации и без ожидания чека.

v1.1.4 (2026-05-11): исправлен _normalize_text: слова больше не распадаются на
  отдельные символы, поэтому greeting-guard и связанные intent-checks снова
  сравнивают нормализованный текст корректно.

v1.1.3 (2026-05-11): unified greeting-guard via _normalize_text;
  punctuation-only and double-space greeting variants no longer bypass
  the soft_positive protection.

v1.1.2 (2026-05-11): soft_positive-ветка защищена от преждевременного
  вывода о готовности платить: чистые приветствия (Здравствуйте, Добрый день
  и пр.) теперь получают уточняющий вопрос вместо «ждём оплату».
  Добавлен _is_greeting_only(); _PURE_GREETINGS содержит русские и казахские
  варианты.

v1.1.0 (2026-04-29): paid_claim переведён в отдельное состояние
  awaiting_payment_proof; claim об оплате и вложенные чеки/скрины теперь
  сразу уходят менеджеру/наблюдателям в Telegram с прямой ссылкой на файл.

v1.0.9 (2026-04-29): убраны повторяющиеся ответы в WhatsApp-диалогах:
  короткие подтверждения после просьбы о чеке больше не вызывают новый ответ,
  "счс оплачу/всю" не запускает повторный допрос про сумму/дату,
  сервисные запросы вроде акта сверки сразу эскалируются менеджеру.

v1.0.8 (2026-04-23): мягкая обработка ответов клиентов: soft_positive /
  promise_schedule / paid_claim, без ложной фиксации обещаний и без повторной ссылки на 1С.

Хранилище: logs/collector_client_dialogs.json
Ключ: номер телефона (цифры, без +, без @c.us)

Жизненный цикл диалога:
  start_client_dialog() → handle_incoming() → escalate_to_manager()
  state: active | awaiting_payment_proof | awaiting_manager | escalated | closed
"""

import asyncio
import json
import logging
import re
from collector.logging_utils import get_collector_logger
import os
import tempfile
import time
from datetime import datetime, timedelta
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
_TEST_MODE = os.getenv("COLLECTOR_TEST_MODE", "0").lower() in ("1", "true", "yes")
COMPANY_NAME = os.getenv("COMPANY_NAME", "Минбаракат")
ESCALATION_SUPPRESS_DAYS = int(os.getenv("COLLECTOR_ESCALATION_SUPPRESS_DAYS", "2"))
SILENT_ACTIVE_RESEND_HOURS = float(os.getenv("COLLECTOR_SILENT_ACTIVE_RESEND_HOURS", "24"))

_ROOT = Path(__file__).resolve().parent.parent
_DIALOGS_PATH = _ROOT / "logs" / "collector_client_dialogs.json"
_DELETION_QUEUE_PATH = _ROOT / "logs" / "deletion_queue.json"

logger = get_collector_logger(__name__)
_DIALOG_ACTIVE_STATES = {"active", "awaiting_payment_proof", "awaiting_manager"}


def _mask_phone(phone: str) -> str:
    digits = "".join(c for c in str(phone or "") if c.isdigit())
    if len(digits) <= 4:
        return digits
    return f"{digits[:4]}***{digits[-2:]}"


def _audit(event: str, **kwargs: Any) -> None:
    try:
        from collector.audit_log import audit as _collector_audit
        _collector_audit(event, **kwargs)
    except Exception as exc:
        logger.debug("audit skipped %s: %s", event, exc)


def _set_dialog_followup_suppress(client_name: str, reason: str, days: int) -> None:
    if days <= 0:
        return
    try:
        from collector.collections_db import set_wa_dialog_suppress
        until = (datetime.now(TZ).date() + timedelta(days=days)).isoformat()
        set_wa_dialog_suppress(client_name, reason, until)
    except Exception as exc:
        logger.warning("wa_dialog_suppress (%s): %s", reason, exc)


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


def _parse_dialog_dt(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=TZ)
    return parsed.astimezone(TZ)


def _sync_promise_to_agreed(
    dialog: Dict[str, Any], promise_date: Any, details: str,
) -> bool:
    """F-B1: записывает client-promise в wa_agreed_promises.json.

    Содержит критический `_TEST_MODE` guard. Без него юнит-тесты, которые
    прогоняют handle_incoming с intent=promise, успешно контаминировали
    боевой `logs/wa_agreed_promises.json` (запись "Кайрбек" с deadline
    2026-04-22 → handler 10:30 мог бы выслать ложное «обещание нарушено»).

    Возвращает True если запись действительно сохранена.
    """
    if _TEST_MODE:
        return False
    if not promise_date:
        return False
    try:
        from collector.approval_flow import record_client_promise as _record_promise
        return _record_promise(
            client_name=str(dialog.get("client_name") or ""),
            manager_name=str(dialog.get("manager_name") or ""),
            promise_date=str(promise_date),
            details=details,
        )
    except Exception as _exc:
        logger.warning("_sync_promise_to_agreed failed: %s", _exc)
        return False


def dialog_blocks_new_outreach(dialog: Optional[Dict[str, Any]], *, now: Optional[datetime] = None) -> tuple[bool, str]:
    """Return whether an existing client dialog should block a fresh WA send."""
    if not isinstance(dialog, dict):
        return False, ""

    state = str(dialog.get("state") or "")
    if state not in (*_DIALOG_ACTIVE_STATES, "escalated"):
        return False, state

    ref_now = now or datetime.now(TZ)
    last_activity = _parse_dialog_dt(dialog.get("last_activity") or dialog.get("created"))
    age_hours: Optional[float] = None
    if last_activity is not None:
        age_hours = max(0.0, (ref_now - last_activity).total_seconds() / 3600.0)

    if state == "awaiting_payment_proof" and age_hours is not None and age_hours >= 72:
        return False, "stale_payment_proof"

    if (
        state == "active"
        and int(dialog.get("exchange_count", 0) or 0) == 0
        and age_hours is not None
        and age_hours >= SILENT_ACTIVE_RESEND_HOURS
    ):
        return False, "stale_silent_active"

    return True, state


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


async def _notify_dialog_observers(dialog: Dict[str, Any], text: str) -> None:
    if _TEST_MODE:
        logger.info(
            "COLLECTOR_TEST_MODE: observer notification suppressed for %s",
            dialog.get("client_name", "unknown"),
        )
        return
    manager_chat_id = dialog.get("manager_chat_id")
    manager_name = dialog.get("manager_name", "")
    try:
        from collector.communications import get_observer_ids
        observer_ids = get_observer_ids(manager_name)
    except Exception as e:
        logger.error("get_observer_ids ошибка: %s", e)
        observer_ids = []

    if manager_chat_id and manager_chat_id not in observer_ids:
        observer_ids = [manager_chat_id] + observer_ids

    for obs_id in observer_ids:
        await _send_tg(obs_id, text)


async def _notify_saida_doc_request(
    dialog: Dict[str, Any],
    client_text: str,
    doc_kind: str = "акт сверки",
) -> None:
    """Дублирует Саиде запрос документа от клиента (акт сверки / счёт / накладная).

    Why: Саида ведёт документооборот; ответственный менеджер может задержать,
    а клиент уже ждёт. Параллельное уведомление Саиде в личку ускоряет ответ.
    """
    if _TEST_MODE:
        logger.info(
            "COLLECTOR_TEST_MODE: saida notification suppressed for %s",
            dialog.get("client_name", "unknown"),
        )
        return
    try:
        saida_chat_id = int(os.getenv("SAIDA_CHAT_ID", "920236287"))
    except (TypeError, ValueError):
        logger.warning("SAIDA_CHAT_ID невалиден — пропуск уведомления")
        return
    if not saida_chat_id:
        return

    client_name = dialog.get("client_name", "—")
    manager_name = dialog.get("manager_name", "—")
    preview = (client_text or "").strip()
    if len(preview) > 200:
        preview = preview[:200] + "…"

    text = (
        f"📄 <b>Запрос документа от клиента</b>\n\n"
        f"Клиент: <b>{client_name}</b>\n"
        f"Менеджер: {manager_name}\n"
        f"Запрос: <b>{doc_kind}</b>\n\n"
        f"💬 Сообщение клиента:\n<i>{preview}</i>"
    )
    await _send_tg(saida_chat_id, text)


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
    """True, если клиент говорит про недавнюю оплату, которая могла не попасть в 1С.

    Требует КОНТЕКСТНЫЙ сигнал (QR/временной/1С-статус) + ПЛАТЁЖНЫЙ сигнал.
    "оплат" убран из контекстных маркеров: иначе "Передам на оплату"
    ложно срабатывает, т.к. "оплат" входит как подстрока в "оплату".
    """
    t = text.lower()
    context_markers = (
        "qr", "куар", "киар", "упад", "поступ",
        "за выходные", "сегодня", "завтра",
        "не разнес", "не провел", "не прошло",
    )
    payment_markers = ("оплат", "qr", "куар", "киар", "упад", "поступ")
    return any(m in t for m in context_markers) and any(m in t for m in payment_markers)


def _normalize_text(text: str) -> str:
    lowered = str(text or "").lower().replace("\u0451", "\u0435")
    cleaned = "".join(
        ch if ch.isalnum() or ch.isspace() else " "
        for ch in lowered
    )
    return " ".join(cleaned.split())




def _is_acknowledgement_only(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    acknowledgements = {
        "да",
        "ага",
        "угу",
        "\u043e\u043a",
        "\u043e\u043a\u0435\u0439",
        "\u0445\u043e\u0440\u043e\u0448\u043e",
        "\u043f\u043e\u043d\u044f\u043b",
        "\u043f\u043e\u043d\u044f\u043b\u0430",
        "\u043f\u0440\u0438\u043d\u044f\u043b",
        "\u043f\u0440\u0438\u043d\u044f\u043b\u0430",
        "\u044f\u0441\u043d\u043e",
        "\u043b\u0430\u0434\u043d\u043e",
        "\u0434\u043e\u0433\u043e\u0432\u043e\u0440\u0438\u043b\u0438\u0441\u044c",
        "\u0445\u043e\u0440\u043e\u0448\u043e \u0441\u043f\u0430\u0441\u0438\u0431\u043e",
        "\u043e\u043a \u0441\u043f\u0430\u0441\u0438\u0431\u043e",
    }
    return normalized in acknowledgements


_SERVICE_REQUEST_PATTERNS = [
    r"\b\u0430\u043a\u0442\s+\u0441\u0432\u0435\u0440\u043a",   # \u0430\u043a\u0442 \u0441\u0432\u0435\u0440\u043a\u0438
    r"\b\u0441\u0432\u0435\u0440\u043a",         # \u0441\u0432\u0435\u0440\u043a\u0430, \u0441\u0432\u0435\u0440\u0438\u043c
    r"\b\u0430\u043a\u0442\b",         # \u0430\u043a\u0442 (\u043e\u0442\u0434\u0435\u043b\u044c\u043d\u043e\u0435 \u0441\u043b\u043e\u0432\u043e; \u043d\u0435 "\u0444\u0430\u043a\u0442")
    r"\b\u0441\u0447\u0435\u0442\b",        # \u0441\u0447\u0435\u0442 (\u0431\u0435\u0437 \u0451) \u2014 \u041d\u0415 "\u0440\u0430\u0441\u0447\u0435\u0442" \u0431\u043b\u0430\u0433\u043e\u0434\u0430\u0440\u044f \b
    r"\b\u0441\u0447\u0435\u0442\s+\u0444\u0430\u043a\u0442\u0443\u0440",  # \u0441\u0447\u0435\u0442-\u0444\u0430\u043a\u0442\u0443\u0440\u0430
    r"\b\u043d\u0430\u043a\u043b\u0430\u0434\u043d",       # \u043d\u0430\u043a\u043b\u0430\u0434\u043d\u0430\u044f
    r"\b\u0434\u043e\u0433\u043e\u0432\u043e\u0440",       # \u0434\u043e\u0433\u043e\u0432\u043e\u0440
    r"\b\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442",      # \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u044b
]

def _is_service_request(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return any(re.search(p, normalized) for p in _SERVICE_REQUEST_PATTERNS)


def _shows_imminent_payment_commitment(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    explicit = (
        "\u0441\u0435\u0439\u0447\u0430\u0441 \u043e\u043f\u043b\u0430\u0447\u0443",
        "\u0449\u0430\u0441 \u043e\u043f\u043b\u0430\u0447\u0443",
        "\u0441\u0447\u0441 \u043e\u043f\u043b\u0430\u0447\u0443",
        "\u0441\u0435\u0433\u043e\u0434\u043d\u044f \u043e\u043f\u043b\u0430\u0447\u0443",
        "\u043e\u043f\u043b\u0430\u0447\u0443 \u0441\u0435\u0433\u043e\u0434\u043d\u044f",
        "\u0441\u043a\u043e\u0440\u043e \u043e\u043f\u043b\u0430\u0447\u0443",
        "\u0437\u0430\u043a\u0440\u043e\u044e \u0441\u0435\u0433\u043e\u0434\u043d\u044f",
        "\u0432\u0441\u044e",
        "\u043f\u043e\u043b\u043d\u043e\u0441\u0442\u044c\u044e",
    )
    if normalized in explicit:
        return True
    return any(phrase in normalized for phrase in explicit if " " in phrase)


def _last_bot_text(dialog: Dict[str, Any]) -> str:
    for exchange in reversed(dialog.get("exchanges", [])):
        if exchange.get("role") == "bot":
            return str(exchange.get("text") or "")
    return ""


def _waiting_for_payment_proof(dialog: Dict[str, Any]) -> bool:
    return bool(dialog.get("awaiting_payment_proof"))


def _is_brief_reply(text: str, *, max_words: int = 2, max_chars: int = 18) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    return len(raw) <= max_chars and len(raw.split()) <= max_words


_PURE_GREETINGS: frozenset[str] = frozenset({
    # Русские
    "здравствуйте", "здравствуй",
    "добрый день", "добрый вечер", "добрый",
    "доброе утро",
    "привет",
    # Казахские
    "сәлем", "salem", "салем",
    "сәлеметсіз бе", "сәлеметсіз",
    "саламатсыз ба", "саламатсыз бе", "саламатсызба", "саламатсызбе",
    # Исламское приветствие — варианты кириллицей
    "ассаламалейкум",
    "ассалаумалейкум",
    "ассалам алейкум",
    "ассалам",
    "саламалейкум",
    "ассаляму алейкум",
    "ас саляму алейкум",
    "ассаламу алейкум",
    # Ответное приветствие
    "уалейкум ассалам",
    "уа алейкум ассалам",
    "уалейкумассалам",
    # Латиница
    "assalamu aleykum",
    "assalamualeykum",
    "assalamu alaikum",
    "assalamu alaykum",
    "assalam aleykum",
})


def _is_greeting_only(text: str) -> bool:
    """True if text is a pure greeting without any payment signal."""
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return normalized in _PURE_GREETINGS


def _attachment_note_lines(attachment: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(attachment, dict):
        return []
    att_type = str(attachment.get("type") or "file")
    caption = str(attachment.get("caption") or "").strip()
    file_name = str(attachment.get("file_name") or "").strip()
    download_url = str(attachment.get("download_url") or "").strip()
    lines = [f"📎 Вложение от клиента: <b>{att_type}</b>"]
    if file_name:
        lines.append(f"Файл: <b>{file_name}</b>")
    if caption:
        lines.append(f"Подпись: {caption}")
    if download_url:
        lines.append(f"Ссылка: {download_url}")
    return lines


def _build_payment_claim_note(
    dialog: Dict[str, Any],
    phone: str,
    *,
    client_text: str = "",
    attachment: Optional[Dict[str, Any]] = None,
    proof_received: bool = False,
) -> str:
    client_name = str(dialog.get("client_name") or "—")
    manager_name = str(dialog.get("manager_name") or "—")
    amount = _fmt_amount(float(dialog.get("amount", 0) or 0))
    days = int(dialog.get("days", 0) or 0)
    lines = [
        f"💳 <b>{client_name}</b> сообщил об оплате.",
        f"Менеджер: <b>{manager_name}</b>",
        f"Телефон: <code>+{phone}</code>",
        f"Текущий долг в контуре: <b>{amount} тг</b> | {days} дн.",
    ]
    if client_text:
        lines.append(f"Сообщение клиента: {client_text}")
    if proof_received:
        lines.append("Статус: клиент прислал подтверждение оплаты.")
    else:
        lines.append("Статус: ждём чек / дату и сумму платежа.")
    lines.extend(_attachment_note_lines(attachment))
    return "\n".join(lines)


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
    debt_age_days = int(dialog.get("debt_age_days", days) or 0)
    deferral_days = int(dialog.get("deferral_days", 0) or 0)
    effective_overdue_days = int(dialog.get("effective_overdue_days", days) or 0)
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

    if deferral_days > 0:
        debt_line = (
            f"💰 {_fmt_amount(amount)} тг | Возраст долга: {debt_age_days} дн. "
            f"| Отсрочка: {deferral_days} дн."
        )
        if effective_overdue_days > 0:
            debt_line += f" | Просрочка по отсрочке: {effective_overdue_days} дн."
        else:
            debt_line += " | По отсрочке еще в срок."
    else:
        debt_line = f"💰 {_fmt_amount(amount)} тг | {days} дн. просрочки"

    return (
        f"📋 <b>{name}</b> — требуется участие {manager_name}\n\n"
        f"{debt_line}{promise_line}{schedule_line}\n"
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
    manager_name = dialog.get("manager_name", "")

    # Обновляем состояние диалога
    if dialog.get("state") not in {"awaiting_payment_proof", "awaiting_manager"}:
        dialog["state"] = "escalated"
    _set_dialog_followup_suppress(
        str(dialog.get("client_name") or ""),
        f"escalated_{reason}",
        ESCALATION_SUPPRESS_DAYS,
    )
    _set_client_dialog(phone, dialog)

    text = _build_escalation_text(dialog, reason, summary)
    await _notify_dialog_observers(dialog, text)
    _audit(
        "dialog_escalated",
        name=dialog.get("client_name"),
        phone_masked=_mask_phone(phone),
        reason=reason,
        manager=manager_name,
        state=dialog.get("state"),
    )

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
        if ok:
            _audit("wa_reply_sent", phone_masked=_mask_phone(phone), text_preview=text[:120])
        else:
            _audit("wa_reply_failed", phone_masked=_mask_phone(phone), text_preview=text[:120], reason="send_whatsapp_false")
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
    debt_age_days: int = 0,
    deferral_days: int = 0,
    effective_overdue_days: int = 0,
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

    # Не перезаписываем активный или эскалированный диалог —
    # клиент уже в работе, повторная отправка создаёт дублирование.
    existing = _get_client_dialog(phone_clean)
    carried_silent_cycles = 0
    if existing:
        should_block, reason = dialog_blocks_new_outreach(existing)
        if should_block:
            existing_state = existing.get("state", "")
            logger.info(
                "[%s] диалог уже существует (state=%s, phone=%s) — пропуск",
                client_name, existing_state, phone_clean,
            )
            return
        if reason == "stale_silent_active":
            carried_silent_cycles = int(existing.get("phone_silent_cycles", 0) or 0) + 1
            logger.info(
                "[%s] stale silent dialog no longer blocks resend (phone=%s, cycles=%d)",
                client_name, phone_clean, carried_silent_cycles,
            )
            _audit(
                "dialog_superseded_after_silence",
                name=client_name,
                phone_masked=_mask_phone(phone_clean),
                silent_cycles=carried_silent_cycles,
            )

    now = _now_iso()
    dialog: Dict[str, Any] = {
        "client_name":       client_name,
        "manager_name":      manager_name,
        "manager_chat_id":   manager_chat_id,
        "level":             level,
        "days":              days,
        "debt_age_days":     debt_age_days or days,
        "deferral_days":     deferral_days,
        "effective_overdue_days": effective_overdue_days or days,
        "amount":            amount,
        "report_date":       report_date,
        "exchanges": [
            {"role": "bot", "text": message_text, "timestamp": now}
        ],
        "exchange_count":    0,  # только ходы клиента
        "state":             "active",
        "created":           now,
        "last_activity":     now,
        "phone_silent_cycles": carried_silent_cycles,
        "off_topic_count":   0,
        "awaiting_payment_proof": False,
    }
    _set_client_dialog(phone_clean, dialog)
    _audit("dialog_started", name=client_name, phone_masked=_mask_phone(phone_clean), manager=manager_name, level=level, amount=amount, days=days)
    logger.info(
        "[%s] клиентский диалог зарегистрирован (phone=%s, level=%d)",
        client_name, phone_clean, level,
    )


async def handle_incoming(phone: str, text: str, attachment: Optional[Dict[str, Any]] = None) -> None:
    """Обрабатывает входящее WhatsApp-сообщение от клиента.

    Args:
        phone: Номер телефона клиента (цифры, без @c.us).
        text:  Текст сообщения.
    """
    phone_clean = "".join(c for c in phone if c.isdigit())

    dialogs = _load_client_dialogs()
    dialog = dialogs.get(phone_clean)

    if not dialog:
        _audit("incoming_ignored_no_dialog", phone_masked=_mask_phone(phone_clean), text_preview=text[:120], attachment_type=(attachment or {}).get("type", ""))
        logger.info("Неизвестный клиент %s — входящее сообщение проигнорировано", phone_clean)
        return

    if dialog.get("state") not in _DIALOG_ACTIVE_STATES:
        # Диалог уже эскалирован/закрыт — бот не отвечает,
        # но передаём сообщение клиента менеджеру в TG, чтобы он
        # видел, что написал клиент (иначе возникает «тишина», и менеджер
        # не знает о новой реплике в WA).
        _audit(
            "incoming_forwarded_inactive_state",
            name=dialog.get("client_name"),
            phone_masked=_mask_phone(phone_clean),
            state=dialog.get("state"),
            text_preview=text[:160],
            attachment_type=(attachment or {}).get("type", ""),
        )
        logger.warning(
            "incoming from %s in inactive state=%s — forwarded to manager (no auto-reply)",
            _mask_phone(phone_clean), dialog.get("state"),
        )
        try:
            _client_name = dialog.get("client_name", "Клиент")
            _manager_name = dialog.get("manager_name", "")
            _state = dialog.get("state", "—")
            _preview = (text or "").strip()
            if len(_preview) > 400:
                _preview = _preview[:400] + "…"
            _attach_kind = (attachment or {}).get("type", "")
            _attach_line = f"\n📎 Вложение: {_attach_kind}" if _attach_kind else ""
            fwd_text = (
                f"💬 <b>Новая реплика клиента</b> — {_client_name}\n"
                f"Менеджер: {_manager_name}\n"
                f"Статус диалога: <code>{_state}</code> (бот не отвечает){_attach_line}\n\n"
                f"Клиент пишет:\n<i>{_preview}</i>\n\n"
                f"⚠️ Ответьте клиенту вручную в WhatsApp."
            )
            await _notify_dialog_observers(dialog, fwd_text)
        except Exception as _fe:
            logger.error("forward inactive-state failed: %s", _fe)
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
    _audit("client_reply_received", name=client_name, phone_masked=_mask_phone(phone_clean), state=dialog.get("state"), exchange_count=exchange_count, attachment_type=(attachment or {}).get("type", ""), text_preview=text[:160])

    # Определяем язык
    language = detect_language(text)

    if attachment and dialog.get("state") in {"awaiting_payment_proof", "awaiting_manager"}:
        reply = "Спасибо, подтверждение получили и уже передали менеджеру."
        dialog["state"] = "awaiting_manager"
        dialog["awaiting_payment_proof"] = False
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)
        try:
            from collector.collections_db import set_wa_dialog_suppress
            from datetime import date, timedelta
            _until = (date.today() + timedelta(days=2)).isoformat()
            set_wa_dialog_suppress(client_name, "attachment", _until)
        except Exception as _e:
            logger.warning("wa_dialog_suppress (attachment): %s", _e)
        await _reply_to_client(phone_clean, reply)
        await _notify_dialog_observers(
            dialog,
            _build_payment_claim_note(
                dialog,
                phone_clean,
                client_text=text,
                attachment=attachment,
                proof_received=True,
            ),
        )
        _audit("payment_proof_received", name=client_name, phone_masked=_mask_phone(phone_clean), attachment_type=(attachment or {}).get("type", ""), state=dialog.get("state"))
        return

    if _waiting_for_payment_proof(dialog) and _is_brief_reply(text, max_words=2, max_chars=20):
        _set_client_dialog(phone_clean, dialog)
        return

    if _is_greeting_only(text):
        _greeting_count = dialog.get("off_topic_count", 0)
        if _greeting_count >= 1:
            await escalate_to_manager(
                dialog, "repeated_greeting",
                "Клиент повторно пишет только приветствия — нужен живой менеджер", phone_clean,
            )
            return
        reply = "Здравствуйте. Подскажите, пожалуйста, когда ожидать ближайшую оплату?"
        dialog["off_topic_count"] = _greeting_count + 1
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)
        await _reply_to_client(phone_clean, reply)
        return

    if _is_service_request(text):
        # Подбираем тип документа для уведомления Саиды.
        _norm = _normalize_text(text)
        if "сверк" in _norm or ("акт" in _norm and "сверк" in _norm):
            _doc_kind = "акт сверки"
        elif "наклад" in _norm:
            _doc_kind = "накладная"
        elif "счет фактур" in _norm or "счёт фактур" in _norm or "сф" == _norm.strip():
            _doc_kind = "счёт-фактура"
        elif "счет" in _norm or "счёт" in _norm:
            _doc_kind = "счёт"
        elif "договор" in _norm:
            _doc_kind = "договор"
        else:
            _doc_kind = "документы"

        reply = (
            f"Спасибо, передаю вас менеджеру {manager_name}. "
            f"{pronoun} свяжется с вами и поможет по документам."
        )
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)

        # Suppress 2 дня — пока менеджер шлёт документ и клиент его смотрит,
        # бот не дёргает напоминаниями.
        try:
            from collector.collections_db import set_wa_dialog_suppress
            from datetime import date, timedelta
            _until = (date.today() + timedelta(days=2)).isoformat()
            set_wa_dialog_suppress(client_name, "doc_request", _until)
        except Exception as _e:
            logger.warning("wa_dialog_suppress (doc_request): %s", _e)

        await _reply_to_client(phone_clean, reply)
        await escalate_to_manager(
            dialog, "question",
            f"Клиент запросил {_doc_kind} — нужен менеджер", phone_clean,
        )
        # Дублируем Саиде в личку — она ведёт документооборот.
        await _notify_saida_doc_request(dialog, client_text=text, doc_kind=_doc_kind)
        _audit(
            "doc_request_to_saida",
            name=client_name,
            phone_masked=_mask_phone(phone_clean),
            doc_kind=_doc_kind,
        )
        return

    if _mentions_recent_unposted_payment(text):
        reply = (
            "Спасибо. Если оплата уже прошла, пришлите, пожалуйста, чек или дату и сумму платежа. "
            f"Передадим информацию менеджеру {manager_name}."
        )
        dialog["awaiting_payment_proof"] = True
        dialog["state"] = "awaiting_payment_proof"
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)
        await _reply_to_client(phone_clean, reply)
        await _notify_dialog_observers(
            dialog,
            _build_payment_claim_note(dialog, phone_clean, client_text=text),
        )
        _audit("payment_claim_reported", name=client_name, phone_masked=_mask_phone(phone_clean), source="heuristic_recent_unposted", state=dialog.get("state"))
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

    if intent and intent not in ("unclear", "off_topic"):
        dialog["off_topic_count"] = 0

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

    if intent == "cash_pickup":
        # Клиент предлагает забрать оплату наличными.
        # ВАЖНО: наличку забирает МЕНЕДЖЕР, не Саида — Саиду здесь не уведомляем.
        reply = (
            f"Спасибо, передаю менеджеру {manager_name}. "
            f"{pronoun} согласует, как удобнее принять оплату."
        )
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)
        await _reply_to_client(phone_clean, reply)
        await escalate_to_manager(
            dialog, "cash_pickup",
            "Клиент просит забрать оплату наличными — согласуйте визит и заберите кассу", phone_clean,
        )
        _audit("cash_pickup_request", name=client_name, phone_masked=_mask_phone(phone_clean))
        return

    if intent == "doc_request":
        # AI распознал просьбу о документах (не сматчилось keyword-детектором).
        reply = (
            f"Спасибо, передаю вас менеджеру {manager_name}. "
            f"{pronoun} свяжется с вами и поможет по документам."
        )
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)
        try:
            from collector.collections_db import set_wa_dialog_suppress
            from datetime import date, timedelta
            _until = (date.today() + timedelta(days=2)).isoformat()
            set_wa_dialog_suppress(client_name, "doc_request", _until)
        except Exception as _e:
            logger.warning("wa_dialog_suppress (doc_request, ai): %s", _e)
        await _reply_to_client(phone_clean, reply)
        await escalate_to_manager(
            dialog, "doc_request",
            "Клиент просит документы (распознано AI)", phone_clean,
        )
        await _notify_saida_doc_request(dialog, client_text=text, doc_kind="документы")
        _audit("doc_request_to_saida", name=client_name, phone_masked=_mask_phone(phone_clean), source="ai")
        return

    if intent == "dispute":
        # Клиент оспаривает сумму — диалог отдаём менеджеру, бот замолкает.
        reply = (
            f"Спасибо, передаю менеджеру {manager_name}. "
            f"{pronoun} свяжется с вами и сверит цифры."
        )
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)
        await _reply_to_client(phone_clean, reply)
        await escalate_to_manager(
            dialog, "dispute",
            "Клиент оспаривает сумму или факт долга — нужна сверка", phone_clean,
        )
        _audit("dispute_reported", name=client_name, phone_masked=_mask_phone(phone_clean))
        return

    if intent == "complaint":
        reply = (
            f"Спасибо за обратную связь, передаю менеджеру {manager_name}. "
            f"{pronoun} свяжется с вами по этому вопросу."
        )
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)
        await _reply_to_client(phone_clean, reply)
        await escalate_to_manager(
            dialog, "complaint",
            "Клиент жалуется (товар/доставка/сервис) — нужен живой ответ", phone_clean,
        )
        _audit("complaint_reported", name=client_name, phone_masked=_mask_phone(phone_clean))
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
        dialog["awaiting_payment_proof"] = True
        dialog["state"] = "awaiting_payment_proof"
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)
        try:
            from collector.collections_db import set_wa_dialog_suppress
            from datetime import date, timedelta
            _until = (date.today() + timedelta(days=3)).isoformat()
            set_wa_dialog_suppress(client_name, "paid_claim", _until)
        except Exception as _e:
            logger.warning("wa_dialog_suppress (paid_claim): %s", _e)
        await _reply_to_client(phone_clean, reply)
        await _notify_dialog_observers(
            dialog,
            _build_payment_claim_note(dialog, phone_clean, client_text=text),
        )
        _audit("payment_claim_reported", name=client_name, phone_masked=_mask_phone(phone_clean), source="ai_paid_claim", state=dialog.get("state"))
        return

    if intent == "promise_schedule":
        schedule_code = str(payment_schedule or "partial")
        schedule_text = _payment_schedule_label(schedule_code)
        dialog["payment_schedule"] = schedule_code
        if promise_date:
            dialog["promise_date"] = promise_date
            _sync_promise_to_agreed(
                dialog, promise_date,
                f"WA dialog: график {schedule_text}, первый платёж до {promise_date}",
            )
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
        # Чистое приветствие — guard имеет приоритет над suggested_reply от AI:
        # AI мог вернуть ответ в стиле "ждём оплату" даже на "Здравствуйте".
        if _is_greeting_only(text):
            reply = "Спасибо за ответ. Подскажите, пожалуйста, когда планируете ближайший платёж?"
            dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
            _set_client_dialog(phone_clean, dialog)
            await _reply_to_client(phone_clean, reply)
            return
        if _is_acknowledgement_only(text):
            reply = "Спасибо, понял. Подскажите, пожалуйста, когда планируете ближайший платёж?"
            dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
            _set_client_dialog(phone_clean, dialog)
            await _reply_to_client(phone_clean, reply)
            return
        if exchange_count >= 2 or _is_brief_reply(text, max_words=2, max_chars=18):
            reply = (
                "Понял вас. Тогда ждём ближайшую оплату. "
                "Как оплатите — пришлите, пожалуйста, чек."
            )
            dialog["awaiting_payment_proof"] = True
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
            _sync_promise_to_agreed(
                dialog, promise_date,
                f"WA dialog: клиент обещал оплатить до {promise_date}",
            )
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
        dialog["awaiting_payment_proof"] = True
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
        if _is_brief_reply(text, max_words=2, max_chars=18) or exchange_count >= 2:
            reply = (
                "Понял вас. Тогда ждём ближайшую оплату. "
                "Как оплатите — пришлите, пожалуйста, чек."
            )
            dialog["awaiting_payment_proof"] = True
            dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
            dialog["state"] = "escalated"
            _set_client_dialog(phone_clean, dialog)
            await _reply_to_client(phone_clean, reply)
            await escalate_to_manager(
                dialog, "soft_positive",
                "Клиент подтвердил ближайшую оплату без точной даты", phone_clean,
            )
            return
        # Клиент подтверждает готовность, но без даты — просим уточнить
        reply = suggested_reply if suggested_reply else (
            "Понятно. Уточните, пожалуйста, дату оплаты — когда планируете?"
        )
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)
        await _reply_to_client(phone_clean, reply)
        return

    # intent == "unclear" — AI пробует ответить сам; при втором неясном — эскалирует
    off_topic_count = dialog.get("off_topic_count", 0)
    if off_topic_count == 0:
        # Первый раз — используем AI-ответ если есть, иначе мягкий шаблон
        reply = suggested_reply if suggested_reply else (
            "Благодарим за ответ. Вернёмся к вопросу задолженности — "
            "когда вы сможете произвести оплату?"
        )
        dialog["off_topic_count"] = 1
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)
        await _reply_to_client(phone_clean, reply)
    else:
        # Второй раз — сообщаем клиенту и эскалируем менеджеру
        reply = (
            f"Понял. Передаю вас менеджеру {manager_name} — "
            f"{pronoun} свяжется с вами напрямую."
        )
        dialog["exchanges"].append({"role": "bot", "text": reply, "timestamp": now})
        _set_client_dialog(phone_clean, dialog)
        await _reply_to_client(phone_clean, reply)
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/approval_flow.py
UX согласования рассылки WhatsApp — менеджер → администратор.

Версия: 1.1.11 (2026-05-11)

v1.1.10 (2026-05-11): forensics: поле close_reason фиксирует причину финальной
  закрытия батча (send_window_missed), а escalation_reason теперь не перезаписывается
  при финализации — сохраняется исходная причина эскалации (tight_send_window,
  manager_silence_timeout). Это позволяет при разборе инцидентов видеть обе точки:
  когда и почему батч ушёл к администратору, и когда окно закрылось окончательно.

v1.1.9 (2026-05-07): send-кнопка администратора теперь уважает expires_at
  у уже approved batch. После дедлайна callback больше не делает ложный
  "запускаю отправку", а честно закрывает экран как too_late.

v1.1.5 (2026-05-06): защита от клина admin approve/send: preview_batch_changes теперь
  считается через asyncio.to_thread() с таймаутом, а Telegram editMessageText получил
  wall-time timeout, retry и подробный лог. Это не даёт callback-ветке повесить весь bot polling
  и scheduler при зависшем сетевом edit или медленном diff-расчёте. Также добавлен recovery-хелпер
  для approved batch после рестарта.

v1.1.4 (2026-05-06): добавлена read-only аналитика качества обещаний менеджеров
  по wa_agreed_promises.json: агрегированные счётчики, просроченные активные обещания,
  исполнено/сорвано по каждому менеджеру для директорской сводки.

v1.1.3 (2026-05-06): директор получил отдельный B-lite review-контур для
  «Договорились»: можно принять или отклонить каждую договорённость менеджера
  без перегруза основной сводки. Решения сохраняются в батче и отражаются в
  wa_agreed_promises.json.

v1.1.1 (2026-04-29): manager/admin preview texts now show debt snapshot
  date and age warnings, so approvals are not blind when debt files are old.

v1.1.0 (2026-04-29): stop-клиенты разделены на живой stop-list и старые
  хвостовые долги. Для старых хвостов без новых отгрузок согласование теперь
  показывает отдельные типы legacy_tail_reminder / partial_tail_reminder без
  бессмысленной фразы про ограничение отгрузок.

v1.0.9 (2026-04-22): старые сообщения администратора тоже закрываются при
  появлении нового актуального списка, callback по устаревшему запросу
  блокируется. В ручном списке администратора уже обработанный клиент сразу
  исчезает из экрана, чтобы не путаться при выборе.

v1.0.8 (2026-04-22): новый актуальный батч вытесняет предыдущий активный как
  superseded; manager-callback по закрытому батчу больше не принимается. Если
  менеджеры молчат 1 час, батч автоматически переводится на решение
  администратора без тихого зависания.

v1.0.7 (2026-04-22): финальные send-статусы больше не считаются "активным"
  батчем в load_latest_batch; при expire_old_batches молчавшие менеджеры
  помечаются как timeout, чтобы причина зависания была явно сохранена в state.

v1.0.6 (2026-04-22): администратор может вручную редактировать состав батча
  по клиентам перед финальным подтверждением отправки. Добавлен экран
  "Отправлять / Не отправлять" на базе существующей логики ручного выбора.

v1.0.5 (2026-04-22): добавлена send_admin_preview_notice — информационное
  уведомление администратору сразу при создании батча (без кнопок
  утверждения). Полноценная сводка с кнопками приходит позже из
  send_admin_summary после ответов менеджеров. Фикс для кейса, когда
  менеджеры игнорируют превью и админ никогда не получает сводку.

Жизненный цикл:
  1. create_batch(debtors_by_manager)       → batch dict
  2. send_manager_previews(batch, bot)      → TG кнопки каждому менеджеру
  3. handle_manager_callback(data, ...)     → менеджер одобряет/отклоняет/выбирает
  4. (когда все ответили) send_admin_summary(batch, bot)
  5. handle_admin_callback(data, ...)       → Вадим финально одобряет или отменяет
  6. is_ready_for_send(batch_id)            → True только после admin approve

ВАЖНО: этот модуль НЕ отправляет WhatsApp.
Он только управляет согласованием.
Реальный send — отдельный шаг с WHATSAPP_ENABLED=1 + LIVE_SEND_ALLOWED=1.

State: logs/wa_approval_batches.json
Callback prefix: wa_appr_
"""

import json
import logging
from collector.logging_utils import get_collector_logger
import os
import re
import secrets
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from zoneinfo import ZoneInfo

load_dotenv(
    dotenv_path=Path(__file__).resolve().parent.parent / ".env",
    encoding="utf-8-sig",
    override=False,
)

TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))
_ROOT = Path(__file__).resolve().parent.parent
_BATCHES_PATH   = _ROOT / "logs" / "wa_approval_batches.json"
_PROMISES_PATH  = _ROOT / "logs" / "wa_agreed_promises.json"   # обещания 🤝 по клиентам

BOT_TOKEN = os.getenv("TG_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID", "")

BATCH_EXPIRE_HOURS = int(os.getenv("WA_APPROVAL_EXPIRE_HOURS", "9"))
MANAGER_SILENCE_TIMEOUT_HOURS = int(os.getenv("WA_MANAGER_SILENCE_HOURS", "1"))
# До какого времени директор может нажать "Утвердить" (env: WA_SEND_WINDOW_CUTOFF_HOUR/MINUTE)
SEND_WINDOW_CUTOFF_HOUR   = int(os.getenv("WA_SEND_WINDOW_CUTOFF_HOUR",   "19"))
SEND_WINDOW_CUTOFF_MINUTE = int(os.getenv("WA_SEND_WINDOW_CUTOFF_MINUTE", "30"))

logger = get_collector_logger(__name__)


# ─── State I/O ───────────────────────────────────────────────────────────────

def _load_batches() -> Dict[str, Any]:
    if not _BATCHES_PATH.exists():
        return {}
    try:
        with open(_BATCHES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_batches(batches: Dict[str, Any]) -> None:
    _BATCHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(_BATCHES_PATH.parent),
        suffix=".tmp",
        prefix="wa_approval_",
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(batches, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, str(_BATCHES_PATH))
    except (OSError, TypeError, ValueError):
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_batch(batch_id: str) -> Optional[Dict[str, Any]]:
    return _load_batches().get(batch_id)


def save_batch(batch: Dict[str, Any]) -> None:
    batches = _load_batches()
    batches[batch["batch_id"]] = batch
    _save_batches(batches)


def load_latest_batch() -> Optional[Dict[str, Any]]:
    """Возвращает последний незавершённый батч или None."""
    batches = _load_batches()
    if not batches:
        return None
    final_statuses = {
        "admin_approved",
        "cancelled",
        "expired",
        "too_late",
        "sent",
        "partially_sent",
        "send_failed",
        "send_empty",
        "superseded",
    }
    # Сортируем по created_at desc, берём первый не-финальный
    for bid in sorted(batches.keys(), reverse=True):
        b = batches[bid]
        if b.get("status") not in final_statuses:
            return b
    return None


def get_latest_send_ready_batch() -> Optional[Dict[str, Any]]:
    """Возвращает последний батч, уже утверждённый администратором, но ещё не отправленный.

    Батч с истёкшим expires_at не считается готовым — окно отправки пропущено.
    """
    batches = _load_batches()
    if not batches:
        return None
    now = datetime.now(tz=TZ)
    for bid in sorted(batches.keys(), reverse=True):
        b = batches[bid]
        if (
            b.get("status") == "admin_approved"
            and b.get("admin_status") == "approved"
            and not b.get("send_completed_at")
        ):
            expires = _parse_batch_dt(b.get("expires_at"))
            if expires and now >= expires:
                continue  # окно отправки истекло — не считать send-ready
            return b
    return None


# ─── Batch creation ───────────────────────────────────────────────────────────

def _send_window_cutoff(now: datetime) -> datetime:
    """Дедлайн директора для текущего дня (или следующего если уже прошёл)."""
    cutoff = now.replace(
        hour=SEND_WINDOW_CUTOFF_HOUR,
        minute=SEND_WINDOW_CUTOFF_MINUTE,
        second=0, microsecond=0,
    )
    if cutoff <= now:
        cutoff += timedelta(days=1)
    return cutoff


def _batch_expires_at(now: datetime) -> datetime:
    """Срок жизни батча: min(TTL, дедлайн директора сегодня)."""
    return min(now + timedelta(hours=BATCH_EXPIRE_HOURS), _send_window_cutoff(now))


def create_batch(
    debtors_by_manager: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Создаёт новый батч согласования.

    Args:
        debtors_by_manager: {manager_name: [client_dict, ...]}
            client_dict должен содержать: name, amount, days, level,
                                          phone (optional), language (optional)

    Returns:
        batch dict (ещё не сохранён — вызовите save_batch() отдельно).
    """
    now = datetime.now(tz=TZ)
    batch_id = f"{now.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"

    # Клиенты с сорванными обещаниями — 🤝 для них заблокировано в новом батче
    broken_clients: List[str] = get_second_chance_blocked_clients()

    managers_state: Dict[str, Any] = {}
    for manager_name, clients in debtors_by_manager.items():
        # Пропускаем пустые списки и клиентов без менеджера
        if not manager_name or not manager_name.strip():
            client_names = [c.get("name", "?") for c in clients]
            log_fn = logger.warning
            if client_names and all(str(name).startswith("TEST fixture:") for name in client_names):
                log_fn = logger.info
            log_fn(
                "create_batch: %d клиент(ов) без manager_name — пропускаем: %s",
                len(clients), ", ".join(client_names),
            )
            continue
        if not clients:
            continue

        # Нормализуем клиентов (только нужные поля)
        normalized = []
        for c in clients:
            msg_type = c.get("msg_type")
            reason = c.get("reason")
            if not msg_type or not reason:
                msg_type, reason = _classify_msg_type_and_reason(c)
            phone = c.get("phone") or c.get("whatsapp") or ""
            phone_valid, phone_issue = validate_production_phone(phone, c.get("name", ""))
            normalized.append({
                "name":              c.get("name", "—"),
                "amount":            float(c.get("amount", 0)),
                "days":              int(c.get("days", 0)),
                "level":             int(c.get("level", 0)),
                "opening":           float(c.get("opening", 0) or 0),
                "debit":             float(c.get("debit", 0) or 0),
                "credit":            float(c.get("credit", 0) or 0),
                "payment_silence_days": c.get("payment_silence_days"),
                "report_date":       c.get("report_date", ""),
                "oldest_unpaid_date": c.get("oldest_unpaid_date"),
                "unpaid_parts":      c.get("unpaid_parts", []),
                "ignored_tail_parts": c.get("ignored_tail_parts", []),
                "debt_age_basis":    c.get("debt_age_basis", ""),
                "debt_age_confidence": c.get("debt_age_confidence", ""),
                "active_turnover":   bool(c.get("active_turnover", False)),
                "violation_shipment": bool(c.get("violation_shipment", False)),
                "msg_type":          msg_type,
                "reason":            reason,
                "stop_status":       c.get("stop_status", ""),
                "review_action":     c.get("review_action", "client_approval"),
                "phone":             phone,
                "invalid_phone":     not phone_valid,
                "phone_issue":       phone_issue,
                "language":          c.get("language", "ru"),
            })

        # Клиенты с сорванными обещаниями — для них 🤝 заблокировано
        blocked = [c["name"] for c in normalized if c["name"] in broken_clients]

        managers_state[manager_name] = {
            "clients":              normalized,
            "status":               "pending",   # pending | approved_all | rejected_all | manual_editing | manual_done | timeout
            "approved_names":       [],
            "rejected_names":       [],          # legacy — оставляем для совместимости
            "postponed_names":      [],          # legacy — оставляем для совместимости
            # новые поля с бизнес-семантикой
            "agreed_names":         [],          # 🤝 договорились (убраны из WA, детали обязательны)
            "agreed_details":       {},          # {client_name: {deadline, details, recorded_at}}
            "paid_with_doc_names":  [],          # 💰 оплатил + документ приложен
            "paid_no_doc_names":    [],          # 💰 оплатил, документа нет → Саиде
            "second_chance_used":   blocked,     # автозаполнение из сорванных обещаний
            "waiting_for_proof":    None,        # {client_name, batch_id} — ждём фото от менеджера
            "waiting_for_agreed":   None,        # {client_name, batch_id} — ждём детали договорённости
            "responded_at":         None,
        }

    batch: Dict[str, Any] = {
        "batch_id":        batch_id,
        "created_at":      now.isoformat(),
        "expires_at":      _batch_expires_at(now).isoformat(),
        "status":          "pending_managers",   # pending_managers | pending_admin | admin_approved | cancelled | expired
        "managers":        managers_state,
        "admin_status":    "pending",            # pending | approved | cancelled | postponed
        "admin_approved_at": None,
        "approved_clients": [],
    }
    total_clients = sum(len(v["clients"]) for v in managers_state.values())
    logger.info(
        "Создан батч %s: менеджеров=%d, клиентов=%d",
        batch_id,
        len(managers_state),
        total_clients,
    )
    try:
        from collector.audit_log import audit as _audit
        _audit("batch_created", batch_id=batch_id,
               managers=len(managers_state), clients=total_clients)
    except Exception:
        pass
    return batch


# ─── Payment discipline classifier ───────────────────────────────────────────

_MSG_TYPE_LABELS = {
    "strict_reminder":      "Строгое напоминание",
    "payment_plan_control": "Проверка обещанной оплаты",
    "soft_reminder":        "Мягкое напоминание",
    "stoplist_reminder":    "Напоминание по стоп-листу",
    "legacy_tail_reminder": "Старый долг без движения",
    "partial_tail_reminder": "Старый долг с частичным погашением",
}

_LEVEL_LABELS = {
    1: "мягкое напоминание",
    2: "повторное напоминание",
    3: "жесткое напоминание",
    4: "жесткое напоминание + звонок",
    5: "эскалация руководителю",
}

_PLACEHOLDER_PHONE_KEYS: set[str] = set()


def _fmt_amount(n: float) -> str:
    return f"{n:,.0f}".replace(",", " ")


def _parse_batch_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return dt


def _normalize_phone_key(phone: str) -> str:
    digits = "".join(c for c in str(phone or "") if c.isdigit())
    if len(digits) == 10:
        return "7" + digits
    if len(digits) == 11 and digits.startswith("8"):
        return "7" + digits[1:]
    return digits


def _phone_keys_in_name(client_name: str) -> List[str]:
    keys: List[str] = []
    for raw in re.findall(r"\d[\d\s().-]{8,}\d", str(client_name or "")):
        key = _normalize_phone_key(raw)
        if len(key) == 11 and key.startswith("7"):
            keys.append(key)
    return keys


def validate_production_phone(phone: str, client_name: str = "") -> Tuple[bool, str]:
    """Returns (valid, reason) for production WhatsApp send."""
    key = _normalize_phone_key(phone)
    if not key:
        return False, "invalid_phone:missing"
    if key in _PLACEHOLDER_PHONE_KEYS:
        return False, "invalid_phone:placeholder"
    if len(key) != 11 or not key.startswith("7"):
        return False, "invalid_phone:bad_format"
    if len(set(key[-10:])) <= 2:
        return False, "invalid_phone:suspicious"

    # name_mismatch убран: телефон в 1С-имени — это адресная строка (напр. "тел 87014850191"),
    # а не WhatsApp-контакт. CRM-телефон введён менеджером вручную и является правильным.
    return True, ""


def _classify_msg_type_and_reason(c: Dict[str, Any]) -> tuple:
    """Определяет тип сообщения и причину попадания клиента в список.

    Returns:
        (msg_type, reason) — строки для отображения менеджеру/директору.
    """
    opening   = float(c.get("opening", 0) or 0)
    debit     = float(c.get("debit", 0) or 0)
    credit    = float(c.get("credit", 0) or 0)
    amount    = float(c.get("amount", 0) or 0)
    days      = int(c.get("days", 0) or 0)
    violation = bool(c.get("violation_shipment", False))
    stop_status = str(c.get("stop_status") or "")

    def fmt(n: float) -> str:
        return f"{n:,.0f}".replace(",", " ")

    if stop_status in ("stopped", "auto_stopped"):
        if amount > 0 and opening > 0 and debit == 0 and days >= 20:
            if credit > 0:
                return "partial_tail_reminder", (
                    f"статус {stop_status}, старый хвост: оплата {fmt(credit)} тг, остаток {fmt(amount)} тг"
                )
            return "legacy_tail_reminder", (
                f"статус {stop_status}, старый хвост без движения, остаток {fmt(amount)} тг"
            )
        return "stoplist_reminder", f"статус {stop_status}, долг {fmt(amount)} тг не закрыт"
    if stop_status in ("pending_clearance", "conditional"):
        return "payment_plan_control", f"статус {stop_status}, требуется ручная проверка перед текстом"

    # Заморожен — нет ни отгрузок, ни оплат
    if debit == 0 and credit == 0:
        return "strict_reminder", f"{days}д, нет отгрузок и оплат"

    # Активные отгрузки при старом долге
    if violation and debit > 0:
        if credit > 0 and credit >= debit * 0.85:
            # Платит почти всё что берёт, хвост небольшой
            return "payment_plan_control", (
                f"оборот {fmt(debit)} тг, оплаты {fmt(credit)} тг, хвост {fmt(amount)} тг"
            )
        return "strict_reminder", (
            f"отгрузки {fmt(debit)} тг при начальном долге {fmt(opening)} тг, "
            f"оплаты {fmt(credit)} тг"
        )

    # Есть оплаты, нет новых отгрузок
    if credit > 0 and debit == 0:
        base = opening if opening > 0 else amount
        pct = (credit / base * 100) if base > 0 else 0
        if pct < 15:
            return "soft_reminder", (
                f"оплата {fmt(credit)} тг ({pct:.0f}% от долга), остаток {fmt(amount)} тг"
            )
        return "soft_reminder", f"оплата {fmt(credit)} тг, остаток {fmt(amount)} тг"

    return "strict_reminder", f"{days}д просрочки, долг {fmt(amount)} тг"


# ─── Telegram helpers ─────────────────────────────────────────────────────────

async def _tg_send(chat_id: int, text: str, markup=None, _retries: int = 3) -> Optional[int]:
    """Отправляет Telegram-сообщение; возвращает message_id.

    При сетевой ошибке делает до _retries попыток с задержкой 1–2–4с.
    """
    if not BOT_TOKEN:
        logger.warning("BOT_TOKEN не задан — Telegram недоступен")
        return None
    import asyncio
    import httpx
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload: Dict[str, Any] = {
        "chat_id":    chat_id,
        "text":       text,
        "parse_mode": "HTML",
    }
    if markup:
        payload["reply_markup"] = markup
    for attempt in range(1, _retries + 1):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                return resp.json().get("result", {}).get("message_id")
            logger.warning("TG send error %d: %s", resp.status_code, resp.text[:200])
            return None  # HTTP-ошибка — не ретраим (не сетевая проблема)
        except Exception as e:
            if attempt < _retries:
                delay = 2 ** (attempt - 1)  # 1, 2, 4 секунды
                logger.warning(
                    "TG send attempt %d/%d failed (%s), retry in %ds",
                    attempt, _retries, e, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error("TG send exception (all %d attempts): %s", _retries, e)
    return None


async def _tg_edit(chat_id: int, message_id: int, text: str, markup=None, _retries: int = 3) -> None:
    """Редактирует существующее Telegram-сообщение.

    Важно: это вызывается прямо из callback-веток manager/admin approval. Поэтому
    сетевой edit не должен иметь права повесить весь polling loop. Даём жёсткий
    wall-time timeout и ограниченный retry.
    """
    if not BOT_TOKEN:
        return
    import asyncio
    import httpx
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/editMessageText"
    payload: Dict[str, Any] = {
        "chat_id":    chat_id,
        "message_id": message_id,
        "text":       text,
        "parse_mode": "HTML",
    }
    if markup:
        payload["reply_markup"] = markup
    timeout = httpx.Timeout(15.0, connect=10.0, read=10.0, write=10.0, pool=10.0)
    for attempt in range(1, _retries + 1):
        try:
            logger.info(
                "TG edit start: chat_id=%s message_id=%s attempt=%d/%d",
                chat_id, message_id, attempt, _retries,
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await asyncio.wait_for(client.post(url, json=payload), timeout=20)
            if resp.status_code == 200:
                logger.info(
                    "TG edit ok: chat_id=%s message_id=%s attempt=%d/%d",
                    chat_id, message_id, attempt, _retries,
                )
                return
            logger.warning(
                "TG edit error %d: %s",
                resp.status_code, resp.text[:200],
            )
            return
        except Exception as e:
            if attempt < _retries:
                delay = 2 ** (attempt - 1)
                logger.warning(
                    "TG edit attempt %d/%d failed (%s), retry in %ds",
                    attempt, _retries, e, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error("TG edit exception (all %d attempts): %s", _retries, e)


def _inline_kb(rows: List[List[Tuple[str, str]]]) -> Dict[str, Any]:
    """Строит inline-клавиатуру из списка [(text, callback_data), ...]."""
    return {
        "inline_keyboard": [
            [{"text": t, "callback_data": d} for t, d in row]
            for row in rows
        ]
    }


def _debt_age_text(c: Dict[str, Any]) -> str:
    days = c.get("days", 0)
    text = f"Возраст остатка: {days} дн."
    oldest = c.get("oldest_unpaid_date")
    if oldest:
        text += f" · Остаток с: {oldest}"
    small_old_parts = c.get("ignored_tail_parts") or []
    if small_old_parts:
        amount = sum(float(p.get("amount", 0) or 0) for p in small_old_parts if isinstance(p, dict))
        if amount > 0:
            text += f" · Малый старый остаток: {_fmt_amount(amount)} тг"
    if c.get("active_turnover"):
        text += " · активный оборот"
    return text


def _freshness_lines(batch: Dict[str, Any]) -> List[str]:
    snap = batch.get("debt_snapshot")
    if not isinstance(snap, dict):
        return []
    lines = []
    label = str(snap.get("snapshot_label_ru") or "").strip()
    if label:
        lines.append(f"🗓 Данные дебиторки: <b>{label}</b>")
    max_age_days = snap.get("max_age_days")
    if isinstance(max_age_days, int):
        lines.append(f"⌛ Возраст данных: <b>{max_age_days} дн.</b>")
    if snap.get("has_warning"):
        warn_managers = snap.get("warning_managers") or []
        if warn_managers:
            suffix = f" и ещё {len(warn_managers) - 5}" if len(warn_managers) > 5 else ""
            lines.append(f"⚠️ В snapshot есть старые данные: {', '.join(warn_managers[:5])}{suffix}")
    return lines


# ─── Manager preview ──────────────────────────────────────────────────────────

def _format_manager_preview_text(
    manager_name: str,
    clients: List[Dict[str, Any]],
    batch_id: str,
    batch: Optional[Dict[str, Any]] = None,
) -> str:
    """Формирует текст превью для менеджера."""
    def _fmt(n: float) -> str:
        return f"{n:,.0f}".replace(",", " ")

    lines = [
        f"👋 <b>{manager_name}</b>, добрый день!\n",
        f"Бот предлагает отправить уведомление <b>{len(clients)} клиент(ам)</b>:\n",
    ]
    if batch:
        freshness_lines = _freshness_lines(batch)
        lines.extend(freshness_lines)
        if freshness_lines:
            lines.append("")
    for i, c in enumerate(clients, 1):
        viol_tag = " ⚠️" if c.get("violation_shipment") else ""
        phone_note = (
            f"\n     Тел: <code>{c.get('phone', '—')}</code> ⚠️ {c.get('phone_issue')}"
            if c.get("invalid_phone")
            else ""
        )
        type_label = _MSG_TYPE_LABELS.get(c.get("msg_type", ""), c.get("msg_type", ""))
        lines.append(
            f"  {i}. <b>{c['name']}</b>{viol_tag}\n"
            f"     Долг: {_fmt(c['amount'])} тг · {_debt_age_text(c)}\n"
            f"     Отгрузки: {_fmt(c['debit'])} тг · Оплаты: {_fmt(c['credit'])} тг\n"
            f"     Тип: {type_label}\n"
            f"     Причина: {c.get('reason', '—')}"
            f"{phone_note}"
        )
    deadline_str = ""
    if batch:
        created_raw = batch.get("created_at")
        if created_raw:
            try:
                created_dt = datetime.fromisoformat(created_raw)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=TZ)
                deadline_dt = created_dt + timedelta(hours=MANAGER_SILENCE_TIMEOUT_HOURS)
                deadline_str = deadline_dt.strftime("%H:%M")
            except (ValueError, TypeError):
                pass
    deadline_line = (
        f"⏰ Ответьте до <b>{deadline_str}</b>. Если не успеете — уведомления уйдут автоматически."
        if deadline_str
        else "⏰ У вас 1 час на ответ. Если не успеете — уведомления уйдут автоматически."
    )
    lines += [
        "",
        deadline_line,
        "",
        "Пожалуйста, проверьте список и дайте разрешение на отправку.",
        "",
        "<i>Реальное сообщение клиентам уйдёт только после подтверждения директора.</i>",
    ]
    return "\n".join(lines)


def _manager_main_keyboard(batch_id: str, manager_idx: int) -> Dict[str, Any]:
    """Главная клавиатура менеджера (быстрые кнопки)."""
    b = batch_id
    i = manager_idx
    return _inline_kb([
        [("✅ Разрешить всем отправить",      f"wa_appr_mgr_ok|{b}|{i}")],
        [("🤝 Со всеми договорились",         f"wa_appr_mgr_agree_all|{b}|{i}")],
        [("💰 Все оплатили — разобраться",    f"wa_appr_mgr_paid_all|{b}|{i}")],
        [("✏️ Выбрать вручную",               f"wa_appr_mgr_manual|{b}|{i}")],
    ])


def _client_list_keyboard(
    batch_id: str,
    manager_idx: int,
    clients: List[Dict[str, Any]],
    decisions: Dict[str, str],  # name → keep|skip|later|pending
) -> Dict[str, Any]:
    """Клавиатура пословного выбора клиентов."""
    rows = []
    for ci, c in enumerate(clients):
        name = c["name"]
        status = decisions.get(name, "pending")
        status_icon = {"keep": "✅", "skip": "❌", "later": "⏸", "pending": "◯"}.get(status, "◯")
        short_name = name[:25] + "…" if len(name) > 25 else name
        # Строка клиента
        rows.append([
            (f"{status_icon} {short_name}", f"wa_appr_cli_info|{batch_id}|{manager_idx}|{ci}"),
        ])
        # Кнопки действия
        rows.append([
            ("✅ Отправить",    f"wa_appr_cli_keep|{batch_id}|{manager_idx}|{ci}"),
            ("💰 Оплатил",      f"wa_appr_cli_paid|{batch_id}|{manager_idx}|{ci}"),
            ("🤝 Договорились", f"wa_appr_cli_agree|{batch_id}|{manager_idx}|{ci}"),
        ])
    # Финальная кнопка
    rows.append([("✅ Готово — принять мои выборы", f"wa_appr_mgr_done|{batch_id}|{manager_idx}")])
    return _inline_kb(rows)


async def send_manager_previews(
    batch: Dict[str, Any],
    bot=None,  # если bot не None — используем bot.send_message, иначе httpx
) -> None:
    """Отправляет каждому менеджеру его список клиентов для согласования."""
    managers_cfg = _load_managers_cfg()

    for mgr_idx, (manager_name, mgr_state) in enumerate(batch["managers"].items()):
        if mgr_state.get("status") != "pending":
            continue  # уже ответил

        chat_id = managers_cfg.get(manager_name)
        if not chat_id:
            logger.warning("send_manager_previews: нет chat_id для %s — пропуск", manager_name)
            continue

        clients = mgr_state["clients"]
        text = _format_manager_preview_text(manager_name, clients, batch["batch_id"], batch=batch)
        markup = _manager_main_keyboard(batch["batch_id"], mgr_idx)

        msg_id = await _tg_send(int(chat_id), text, markup)
        if msg_id:
            mgr_state["preview_message_id"] = msg_id
            mgr_state["chat_id"] = int(chat_id)
            logger.info(
                "Превью отправлено менеджеру %s (chat_id=%s, msg_id=%d)",
                manager_name, chat_id, msg_id,
            )
        else:
            logger.error("Не удалось отправить превью менеджеру %s", manager_name)

    save_batch(batch)


async def close_manager_previews(batch: Dict[str, Any], reason_text: str) -> None:
    """Закрывает старые manager-preview сообщения: снимает кнопки и помечает как устаревшие."""
    empty_markup = {"inline_keyboard": []}
    for manager_name, mgr_state in (batch.get("managers") or {}).items():
        chat_id = mgr_state.get("chat_id")
        message_id = mgr_state.get("preview_message_id")
        if not chat_id or not message_id:
            continue
        try:
            await _tg_edit(int(chat_id), int(message_id), reason_text, empty_markup)
        except Exception as e:
            logger.error(
                "[%s] не удалось закрыть preview менеджера %s: %s",
                batch.get("batch_id", "?"), manager_name, e,
            )


async def close_admin_messages(batch: Dict[str, Any], reason_text: str) -> None:
    """Закрывает старые админские сообщения, чтобы по ним нельзя было нажать повторно."""
    empty_markup = {"inline_keyboard": []}
    targets = [
        (batch.get("admin_chat_id"), batch.get("admin_message_id"), "admin_message_id"),
        (batch.get("admin_chat_id"), batch.get("admin_preview_msg_id"), "admin_preview_msg_id"),
    ]
    seen: set[tuple[int, int]] = set()
    for chat_id, message_id, field_name in targets:
        if not chat_id or not message_id:
            continue
        key = (int(chat_id), int(message_id))
        if key in seen:
            continue
        seen.add(key)
        try:
            await _tg_edit(int(chat_id), int(message_id), reason_text, empty_markup)
        except Exception as e:
            logger.error(
                "[%s] не удалось закрыть админское сообщение %s: %s",
                batch.get("batch_id", "?"), field_name, e,
            )


def supersede_batch(
    batch: Dict[str, Any],
    *,
    superseded_by: str,
    reason: str = "replaced_by_new_data",
) -> Dict[str, Any]:
    """Закрывает активный батч как неактуальный перед созданием нового."""
    now_iso = datetime.now(tz=TZ).isoformat()
    batch["status"] = "superseded"
    batch["superseded_at"] = now_iso
    batch["superseded_by"] = superseded_by
    batch["superseded_reason"] = reason
    if batch.get("admin_status") == "pending":
        batch["admin_status"] = "cancelled"
    save_batch(batch)
    logger.info(
        "[%s] батч помечен как superseded -> %s (%s)",
        batch.get("batch_id", "?"),
        superseded_by,
        reason,
    )
    return batch


# ─── Manager callback handling ────────────────────────────────────────────────

def _get_manager_by_idx(batch: Dict[str, Any], idx: int) -> Tuple[Optional[str], Optional[Dict]]:
    """Возвращает (manager_name, mgr_state) по индексу."""
    for i, (name, state) in enumerate(batch["managers"].items()):
        if i == idx:
            return name, state
    return None, None


def _build_decisions(mgr_state: Dict[str, Any]) -> Dict[str, str]:
    """Строит dict {client_name → keep|skip|agreed|paid} из mgr_state."""
    d = {}
    for name in mgr_state.get("approved_names", []):
        d[name] = "keep"
    for name in mgr_state.get("rejected_names", []):   # legacy
        d[name] = "skip"
    for name in mgr_state.get("postponed_names", []):  # legacy
        d[name] = "later"
    for name in mgr_state.get("agreed_names", []):
        d[name] = "agreed"
    for name in mgr_state.get("paid_with_doc_names", []):
        d[name] = "paid"
    for name in mgr_state.get("paid_no_doc_names", []):
        d[name] = "paid"
    return d


def _client_removal_reason(mgr_state: Dict[str, Any], client_name: str) -> Optional[str]:
    """Возвращает причину снятия клиента из WA или None если не снят."""
    if client_name in mgr_state.get("agreed_names", []):
        return "agreed"
    if client_name in mgr_state.get("paid_with_doc_names", []):
        return "paid_doc"
    if client_name in mgr_state.get("paid_no_doc_names", []):
        return "paid_no_doc"
    if client_name in mgr_state.get("rejected_names", []):
        return "rejected"
    return None


def _admin_client_key(manager_name: str, client_name: str) -> str:
    return f"{manager_name}|{client_name}"


def _iter_admin_clients(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for manager_name, mgr_state in batch.get("managers", {}).items():
        for client in mgr_state.get("clients", []):
            item = {**client, "manager": manager_name}
            item["_admin_key"] = _admin_client_key(manager_name, client["name"])
            flat.append(item)
    return flat


def _build_admin_decisions(batch: Dict[str, Any]) -> Dict[str, str]:
    """Строит выбор администратора по клиентам.

    Если админ ещё не начинал ручной выбор, по умолчанию берём решения
    менеджеров: approved -> keep, всё остальное -> skip.
    """
    if "admin_keep_keys" in batch or "admin_skip_keys" in batch:
        keep_keys = set(batch.get("admin_keep_keys") or [])
        skip_keys = set(batch.get("admin_skip_keys") or [])
        decisions: Dict[str, str] = {}
        for item in _iter_admin_clients(batch):
            key = item["_admin_key"]
            decisions[key] = "keep" if key in keep_keys else "skip"
        return decisions

    decisions = {}
    for manager_name, mgr_state in batch.get("managers", {}).items():
        approved       = set(mgr_state.get("approved_names", []))
        rejected       = set(mgr_state.get("rejected_names", []))       # legacy
        agreed         = set(mgr_state.get("agreed_names", []))
        paid_doc       = set(mgr_state.get("paid_with_doc_names", []))
        paid_no_doc    = set(mgr_state.get("paid_no_doc_names", []))
        mgr_timed_out  = mgr_state.get("status") == "timeout"
        for client in mgr_state.get("clients", []):
            key  = _admin_client_key(manager_name, client["name"])
            name = client["name"]
            if name in approved:
                decisions[key] = "keep"
            elif name in agreed or name in paid_doc or name in paid_no_doc or name in rejected:
                # Снят с причиной → не отправляем WA
                decisions[key] = "skip"
            elif mgr_timed_out:
                # Молчание менеджера = согласие: авто-включаем всех без явного снятия
                decisions[key] = "keep"
            else:
                decisions[key] = "skip"
    return decisions


def _save_admin_decisions(batch: Dict[str, Any], decisions: Dict[str, str]) -> None:
    keep_keys = sorted(key for key, value in decisions.items() if value == "keep")
    skip_keys = sorted(key for key, value in decisions.items() if value == "skip")
    batch["admin_keep_keys"] = keep_keys
    batch["admin_skip_keys"] = skip_keys
    batch["admin_reviewed_at"] = datetime.now(tz=TZ).isoformat()


def _get_admin_reviewed_keys(batch: Dict[str, Any]) -> set[str]:
    reviewed = batch.get("admin_reviewed_keys") or []
    return {str(key) for key in reviewed}


def _get_agreed_review_decisions(batch: Dict[str, Any]) -> Dict[str, str]:
    decisions = batch.get("agreed_review_decisions") or {}
    return decisions if isinstance(decisions, dict) else {}


def _iter_agreed_review_clients(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    review_decisions = _get_agreed_review_decisions(batch)
    for manager_name, mgr_state in batch.get("managers", {}).items():
        agreed_details = mgr_state.get("agreed_details", {}) or {}
        names: List[str] = list(mgr_state.get("agreed_names", []) or [])
        for key in review_decisions:
            mgr_key, _, client_name = str(key).partition("|")
            if mgr_key != manager_name or not client_name:
                continue
            if client_name in agreed_details and client_name not in names:
                names.append(client_name)
        for client_name in names:
            detail = agreed_details.get(client_name, {}) or {}
            items.append({
                "manager": manager_name,
                "name": client_name,
                "details": detail.get("details", "—"),
                "deadline": detail.get("deadline", ""),
                "client_key": _admin_client_key(manager_name, client_name),
            })
    return items


def _set_agreed_promise_status(
    client_name: str,
    status: str,
    *,
    batch_id: str,
    manager_name: str,
) -> None:
    promises = _load_promises()
    promise = promises.get(client_name)
    if not isinstance(promise, dict):
        return
    promise["status"] = status
    promise["updated_at"] = datetime.now(tz=TZ).isoformat()
    promise["reviewed_in_batch"] = batch_id
    promise["reviewed_by_manager"] = manager_name
    if status == "accepted":
        promise["accepted_at"] = promise["updated_at"]
    elif status == "rejected":
        promise["rejected_at"] = promise["updated_at"]
    promises[client_name] = promise
    _save_promises(promises)


def _admin_client_list_keyboard(
    batch_id: str,
    flat_clients: List[Dict[str, Any]],
    decisions: Dict[str, str],
    reviewed_keys: Optional[set[str]] = None,
) -> Dict[str, Any]:
    reviewed_keys = reviewed_keys or set()
    rows = []
    for idx, client in enumerate(flat_clients):
        key = client["_admin_key"]
        if key in reviewed_keys:
            continue
        status = decisions.get(key, "skip")
        status_icon = {"keep": "✅", "skip": "❌"}.get(status, "❌")
        label = f"{status_icon} {client['manager']}: {client['name']}"
        short_label = label[:34] + "…" if len(label) > 34 else label
        rows.append([(short_label, f"wa_appr_adm_info|{batch_id}|{idx}")])
        rows.append([
            ("✅ Отправлять", f"wa_appr_adm_cli_keep|{batch_id}|{idx}"),
            ("❌ Не отправлять", f"wa_appr_adm_cli_skip|{batch_id}|{idx}"),
        ])
    rows.append([("✅ Готово — сохранить выбор", f"wa_appr_adm_done|{batch_id}")])
    rows.append([("↩️ Назад к сводке", f"wa_appr_adm_back|{batch_id}")])
    return _inline_kb(rows)


def _format_admin_manual_header(batch: Dict[str, Any], decisions: Dict[str, str]) -> str:
    flat_clients = _iter_admin_clients(batch)
    keep_count = sum(1 for item in flat_clients if decisions.get(item["_admin_key"]) == "keep")
    skip_count = len(flat_clients) - keep_count
    reviewed_count = len(_get_admin_reviewed_keys(batch))
    remaining_count = max(len(flat_clients) - reviewed_count, 0)
    return (
        "✏️ <b>Список клиентов перед отправкой</b>\n\n"
        "Для каждого клиента выберите:\n"
        "  ✅ Отправлять\n"
        "  ❌ Не отправлять\n\n"
        f"Выбрано к отправке: <b>{keep_count}</b>\n"
        f"Исключено: <b>{skip_count}</b>\n"
        f"Осталось разобрать: <b>{remaining_count}</b>\n"
        f"Батч: <code>{batch['batch_id']}</code>"
    )


def _format_admin_client_info(client: Dict[str, Any]) -> str:
    amount = _fmt_amount(float(client.get("amount", 0) or 0))
    debit = _fmt_amount(float(client.get("debit", 0) or 0))
    credit = _fmt_amount(float(client.get("credit", 0) or 0))
    level = int(client.get("level", 0) or 0)
    level_label = _LEVEL_LABELS.get(level, f"уровень {level}")
    msg_type = _MSG_TYPE_LABELS.get(client.get("msg_type", ""), client.get("msg_type", "—"))
    phone = client.get("phone") or "—"
    return (
        f"ℹ️ <b>{client['name']}</b>\n"
        f"Менеджер: <b>{client['manager']}</b>\n"
        f"Долг: {amount} тг\n"
        f"{_debt_age_text(client)}\n"
        f"Отгрузки: {debit} тг\n"
        f"Оплаты: {credit} тг\n"
        f"Какое напоминание планируется: {msg_type}\n"
        f"Тон сообщения: {level_label}\n"
        f"Телефон: <code>{phone}</code>"
    )


def _all_managers_responded(batch: Dict[str, Any]) -> bool:
    for mgr_state in batch["managers"].values():
        status = mgr_state.get("status")
        if status in ("approved_all", "rejected_all", "manual_done", "timeout"):
            continue
        if status == "manual" and mgr_state.get("responded_at"):
            continue
        return False
    return True


async def handle_manager_callback(
    data: str,
    chat_id: int,
    message_id: int,
    bot=None,
) -> bool:
    """Обрабатывает нажатие кнопки менеджером.

    Returns:
        True если callback обработан этим модулем.
    """
    if not data.startswith("wa_appr_mgr_") and not data.startswith("wa_appr_cli_"):
        return False

    parts = data.split("|")
    if len(parts) < 3:
        return False

    action  = parts[0]          # wa_appr_mgr_ok / wa_appr_cli_keep / ...
    batch_id = parts[1]
    mgr_idx  = int(parts[2])

    batch = load_batch(batch_id)
    if not batch:
        logger.warning("handle_manager_callback: батч %s не найден", batch_id)
        await _tg_edit(chat_id, message_id, "⚠️ Запрос устарел. Исходный список уже закрыт.")
        return True
    if batch.get("status") != "pending_managers":
        logger.info(
            "handle_manager_callback: батч %s закрыт для менеджера (status=%s)",
            batch_id,
            batch.get("status"),
        )
        await _tg_edit(
            chat_id,
            message_id,
            "⚠️ Этот запрос уже неактуален.\n\n"
            "Решение по нему уже передано администратору или сформирован новый список.",
        )
        return True

    manager_name, mgr_state = _get_manager_by_idx(batch, mgr_idx)
    if not mgr_state:
        logger.warning("handle_manager_callback: менеджер idx=%d не найден в батче %s", mgr_idx, batch_id)
        return True

    clients = mgr_state["clients"]
    now_iso = datetime.now(tz=TZ).isoformat()

    # ── Быстрые решения ──────────────────────────────────────────────────────

    if action == "wa_appr_mgr_ok":
        mgr_state["status"]         = "approved_all"
        mgr_state["approved_names"] = [c["name"] for c in clients]
        mgr_state["rejected_names"] = []
        mgr_state["postponed_names"] = []
        mgr_state["responded_at"]   = now_iso
        save_batch(batch)

        text = (
            f"✅ <b>Принято!</b>\n\n"
            f"Вы разрешили отправить уведомления всем {len(clients)} клиентам.\n"
            f"Итоговое решение — у директора."
        )
        await _tg_edit(chat_id, message_id, text)
        logger.info("[%s] %s одобрил всех (%d клиентов)", batch_id, manager_name, len(clients))

    elif action == "wa_appr_mgr_agree_all":
        # 🤝 Со всеми договорились — открываем поклиентный режим
        mgr_state["status"] = "manual_editing"
        mgr_state["responded_at"] = None
        save_batch(batch)
        decisions = _build_decisions(mgr_state)
        text = (
            f"🤝 <b>Договорились — разбираем по каждому</b>\n\n"
            f"По каждому клиенту нажмите 🤝 Договорились и укажите детали.\n"
            f"Массово убрать без причины нельзя — нужен срок и условия по каждому.\n\n"
            f"Когда закончите — нажмите <b>«Готово»</b>."
        )
        markup = _client_list_keyboard(batch_id, mgr_idx, clients, decisions)
        await _tg_edit(chat_id, message_id, text, markup)
        return True

    elif action == "wa_appr_mgr_paid_all":
        # 💰 Все оплатили — открываем поклиентный режим для подтверждения каждого
        mgr_state["status"] = "manual_editing"
        mgr_state["responded_at"] = None
        save_batch(batch)
        decisions = _build_decisions(mgr_state)
        text = (
            f"💰 <b>Все оплатили — разбираем по каждому</b>\n\n"
            f"По каждому клиенту нажмите 💰 Оплатил и приложите документ.\n"
            f"Массово подтвердить всех без документов нельзя.\n\n"
            f"Когда закончите — нажмите <b>«Готово»</b>."
        )
        markup = _client_list_keyboard(batch_id, mgr_idx, clients, decisions)
        await _tg_edit(chat_id, message_id, text, markup)
        return True

    elif action == "wa_appr_mgr_view":
        # Показываем список с кнопками по каждому клиенту
        decisions = _build_decisions(mgr_state)
        text = _format_manager_preview_text(manager_name, clients, batch_id, batch=batch)
        markup = _client_list_keyboard(batch_id, mgr_idx, clients, decisions)
        await _tg_edit(chat_id, message_id, text, markup)
        return True

    elif action == "wa_appr_mgr_manual":
        # Режим ручного выбора — показываем список
        decisions = _build_decisions(mgr_state)
        mgr_state["status"] = "manual_editing"
        mgr_state["responded_at"] = None
        save_batch(batch)
        text = (
            f"✏️ <b>Выбор вручную</b>\n\n"
            f"Для каждого клиента нажмите:\n"
            f"  ✅ Отправить — разрешить отправку\n"
            f"  💰 Оплатил — клиент оплатил (нужен документ или проверка)\n"
            f"  🤝 Договорились — нужно указать детали\n\n"
            f"Снять без причины нельзя.\n"
            f"Когда выберете всех — нажмите <b>«Готово»</b>."
        )
        markup = _client_list_keyboard(batch_id, mgr_idx, clients, decisions)
        await _tg_edit(chat_id, message_id, text, markup)
        return True

    elif action == "wa_appr_mgr_done":
        # Менеджер завершил выбор — проверяем что каждый клиент имеет причину
        decisions = _build_decisions(mgr_state)
        valid_decisions = {"keep", "skip", "later", "agreed", "paid"}
        approved  = [c["name"] for c in clients if decisions.get(c["name"]) == "keep"]
        agreed    = mgr_state.get("agreed_names", [])
        paid_doc  = mgr_state.get("paid_with_doc_names", [])
        paid_ndoc = mgr_state.get("paid_no_doc_names", [])
        undecided = [c["name"] for c in clients if decisions.get(c["name"]) not in valid_decisions]

        # Клиенты в состоянии "ждём фото/детали" не считаются завершёнными
        waiting_proof  = mgr_state.get("waiting_for_proof") or {}
        waiting_agreed = mgr_state.get("waiting_for_agreed") or {}
        still_waiting  = []
        if isinstance(waiting_proof, dict) and waiting_proof.get("client_name"):
            still_waiting.append(waiting_proof["client_name"] + " (ждём документ)")
        if isinstance(waiting_agreed, dict) and waiting_agreed.get("client_name"):
            still_waiting.append(waiting_agreed["client_name"] + " (ждём детали)")

        if undecided or still_waiting:
            problem_list = [f"  • {n}" for n in undecided + still_waiting]
            text = (
                f"⚠️ Не все клиенты оформлены.\n\n"
                f"Нужно завершить по {len(undecided) + len(still_waiting)} клиент(ам):\n"
                + "\n".join(problem_list)
                + "\n\nВыберите ✅ Отправить, 💰 Оплатил или 🤝 Договорились по каждому."
            )
            markup = _client_list_keyboard(batch_id, mgr_idx, clients, decisions)
            await _tg_edit(chat_id, message_id, text, markup)
            return True

        mgr_state["status"]         = "manual_done"
        mgr_state["approved_names"] = approved
        mgr_state["responded_at"]   = now_iso
        save_batch(batch)

        text = (
            f"✅ <b>Ваш выбор зафиксирован:</b>\n\n"
            f"  ✅ Отправить:          {len(approved)} кл.\n"
            f"  🤝 Договорились:       {len(agreed)} кл.\n"
            f"  💰 Оплатил + документ: {len(paid_doc)} кл.\n"
            f"  💰 Оплатил без документа: {len(paid_ndoc)} кл.\n\n"
            f"Итоговое решение — у директора."
        )
        await _tg_edit(chat_id, message_id, text)
        logger.info(
            "[%s] %s завершил выбор: отправить=%d, договорились=%d, оплатил_doc=%d, оплатил_nodoc=%d",
            batch_id, manager_name, len(approved), len(agreed), len(paid_doc), len(paid_ndoc),
        )

    # ── Кнопки по конкретному клиенту ────────────────────────────────────────

    elif action in (
        "wa_appr_cli_keep", "wa_appr_cli_skip", "wa_appr_cli_later", "wa_appr_cli_info",
        "wa_appr_cli_paid", "wa_appr_cli_agree",
        "wa_appr_cli_paid_doc", "wa_appr_cli_paid_nodoc",
    ):
        if len(parts) < 4:
            return True
        cli_idx = int(parts[3])
        if cli_idx >= len(clients):
            return True

        client_name = clients[cli_idx]["name"]

        if action == "wa_appr_cli_info":
            c = clients[cli_idx]
            amount_fmt = f"{c['amount']:,.0f}".replace(",", " ")
            await _tg_send(
                chat_id,
                f"ℹ️ <b>{c['name']}</b>\n"
                f"Долг: {amount_fmt} тг\n"
                f"Дней просрочки: {c['days']}\n"
                f"Уровень давления: {c['level']} из 5",
            )
            return True

        # Убираем клиента из всех списков перед добавлением в нужный
        all_lists = (
            "approved_names", "rejected_names", "postponed_names",
            "agreed_names", "paid_with_doc_names", "paid_no_doc_names",
        )
        for lst_key in all_lists:
            lst = mgr_state.get(lst_key, [])
            if client_name in lst:
                lst.remove(client_name)

        if action == "wa_appr_cli_keep":
            mgr_state["approved_names"].append(client_name)

        elif action in ("wa_appr_cli_skip", "wa_appr_cli_later"):
            # legacy-пути — оставляем для совместимости
            mgr_state["rejected_names"].append(client_name)

        elif action == "wa_appr_cli_paid":
            # Первый клик — спрашиваем есть ли документ
            kb = _inline_kb([
                [("📎 Отправлю документ",  f"wa_appr_cli_paid_doc|{batch_id}|{mgr_idx}|{cli_idx}")],
                [("⚠️ Без документа",      f"wa_appr_cli_paid_nodoc|{batch_id}|{mgr_idx}|{cli_idx}")],
            ])
            await _tg_send(
                chat_id,
                f"💰 Клиент <b>{client_name}</b> отмечен как оплативший.\n\n"
                f"Пришли фото чека или выписку следующим сообщением.\n"
                f"Если документа нет — нажми «Без документа».",
                kb,
            )
            return True  # клиент ещё не перемещён — ждём уточнения

        elif action == "wa_appr_cli_paid_doc":
            # Менеджер выбрал "есть документ" — ставим в режим ожидания фото
            mgr_state["waiting_for_proof"] = {
                "client_name": client_name,
                "batch_id":    batch_id,
                "cli_idx":     cli_idx,
            }
            mgr_state["paid_with_doc_names"].append(client_name)
            save_batch(batch)
            await _tg_send(
                chat_id,
                f"📎 Жду фото чека или выписки по клиенту <b>{client_name}</b>.\n"
                f"Отправь документ следующим сообщением.",
            )
            # Обновляем клавиатуру
            decisions = _build_decisions(mgr_state)
            markup = _client_list_keyboard(batch_id, mgr_idx, clients, decisions)
            await _tg_edit(chat_id, message_id,
                f"✏️ <b>Список клиентов</b> — {manager_name}\n\nКогда закончите — нажмите «Готово».",
                markup)
            return True

        elif action == "wa_appr_cli_paid_nodoc":
            # Нет документа — клиент снят из WA, запрос уйдёт Саиде
            mgr_state["paid_no_doc_names"].append(client_name)
            mgr_state["waiting_for_proof"] = None
            save_batch(batch)
            await _tg_send(
                chat_id,
                f"⚠️ <b>{client_name}</b> снят из рассылки.\n"
                f"Запрос на проверку оплаты будет отправлен бухгалтеру.",
            )

        elif action == "wa_appr_cli_agree":
            # Проверяем что менеджер не использовал второй шанс по этому клиенту
            if client_name in mgr_state.get("second_chance_used", []):
                await _tg_send(
                    chat_id,
                    f"🚫 <b>{client_name}</b> — «Договорились» уже было использовано.\n"
                    f"После сорванного обещания снять клиента через «Договорились» нельзя.\n"
                    f"Обратитесь к директору.",
                )
                return True
            # Ставим в режим ожидания деталей
            mgr_state["waiting_for_agreed"] = {
                "client_name": client_name,
                "batch_id":    batch_id,
                "cli_idx":     cli_idx,
            }
            save_batch(batch)
            await _tg_send(
                chat_id,
                f"🤝 Клиент <b>{client_name}</b>.\n\n"
                f"Укажи детали договорённости одним сообщением:\n"
                f"  • дата обещанной оплаты\n"
                f"  • сумма\n"
                f"  • что именно согласовано\n\n"
                f"<i>Пример: «до 15 мая, 300 000 тг, договорился лично»</i>",
            )
            return True  # ждём текстовое сообщение

        save_batch(batch)

        # Обновляем клавиатуру списка
        decisions = _build_decisions(mgr_state)
        markup = _client_list_keyboard(batch_id, mgr_idx, clients, decisions)
        header = (
            f"✏️ <b>Список клиентов</b> — {manager_name}\n\n"
            f"Когда закончите — нажмите «Готово»."
        )
        await _tg_edit(chat_id, message_id, header, markup)
        return True

    else:
        return False

    # После любого быстрого решения: проверяем, все ли менеджеры ответили
    batch = load_batch(batch_id)  # перечитываем — могло измениться
    if _all_managers_responded(batch):
        batch["status"] = "pending_admin"
        save_batch(batch)
        logger.info("[%s] Все менеджеры ответили → pending_admin", batch_id)
        # Отправляем сводку администратору (без bot объекта — через httpx)
        await send_admin_summary(batch)

    return True


# ─── Admin summary ────────────────────────────────────────────────────────────

def _format_admin_summary_text(batch: Dict[str, Any]) -> str:
    """Формирует сводку для администратора."""
    lines = [
        "📋 <b>Согласование рассылки WhatsApp — итог менеджеров</b>\n",
        f"Батч: {batch['batch_id']}\n",
    ]
    freshness_lines = _freshness_lines(batch)
    lines.extend(freshness_lines)
    if freshness_lines:
        lines.append("")
    if batch.get("escalated_to_admin_at"):
        lines += [
            "⚠️ <b>Часть менеджеров не ответила вовремя.</b>",
            "Батч передан вам на ручное решение без ожидания всех ответов.\n",
        ]

    total_send = 0
    total_agreed = 0
    total_paid_doc = 0
    total_paid_ndoc = 0
    total_auto = 0

    for manager_name, mgr_state in batch["managers"].items():
        status    = mgr_state.get("status", "pending")
        approved  = mgr_state.get("approved_names", [])
        agreed    = mgr_state.get("agreed_names", [])
        paid_doc  = mgr_state.get("paid_with_doc_names", [])
        paid_ndoc = mgr_state.get("paid_no_doc_names", [])
        rejected  = mgr_state.get("rejected_names", [])   # legacy
        clients   = mgr_state.get("clients", [])
        agreed_details = mgr_state.get("agreed_details", {})
        invalid_phone_clients = [c["name"] for c in clients if c.get("invalid_phone")]

        # Авто-включённые по таймауту: все без явного снятия с причиной
        removed_set = set(agreed) | set(paid_doc) | set(paid_ndoc) | set(rejected)
        auto_included = [c["name"] for c in clients if c["name"] not in removed_set] if status == "timeout" else []

        if status == "pending":
            status_label = "⏳ не ответил"
        elif status == "approved_all":
            status_label = f"✅ разрешил всех ({len(approved)})"
        elif status == "manual_editing":
            status_label = "✏️ выбирает вручную"
        elif status in ("manual", "manual_done"):
            n_send = len(approved)
            n_agr  = len(agreed)
            n_pd   = len(paid_doc)
            n_pnd  = len(paid_ndoc)
            status_label = f"✏️ выбрал: ✅{n_send} 🤝{n_agr} 💰{n_pd}+{n_pnd}"
        elif status == "timeout":
            status_label = f"🔇 не ответил → авто ({len(auto_included)} кл.)"
        else:
            status_label = status

        lines.append(f"\n<b>{manager_name}</b> — {status_label}")

        if approved:
            lines.append(f"  ✅ Отправить ({len(approved)}):")
            for n in approved:
                lines.append(f"    • {n}")
        if auto_included:
            lines.append(f"  🔇 Авто-включено ({len(auto_included)}):")
            for n in auto_included:
                lines.append(f"    • {n}")
        if agreed:
            lines.append(f"  🤝 Договорились ({len(agreed)}):")
            for n in agreed:
                detail = agreed_details.get(n, {}).get("details", "—")
                lines.append(f"    • {n}: <i>{detail}</i>")
        if paid_doc:
            lines.append(f"  💰 Оплатил + документ ({len(paid_doc)}):")
            for n in paid_doc:
                lines.append(f"    • {n} (документ у директора)")
        if paid_ndoc:
            lines.append(f"  💰 Оплатил без документа ({len(paid_ndoc)}) → Саиде:")
            for n in paid_ndoc:
                lines.append(f"    • {n}")
        if rejected:  # legacy
            lines.append(f"  ❌ Убрано без причины ({len(rejected)}):")
            for n in rejected:
                lines.append(f"    • {n}")
        if invalid_phone_clients:
            lines.append(f"  ⚠️ Нет телефона ({len(invalid_phone_clients)}):")
            for n in invalid_phone_clients:
                lines.append(f"    ⚠️ {n}")

        total_send    += len(approved) + len(auto_included)
        total_agreed  += len(agreed)
        total_paid_doc  += len(paid_doc)
        total_paid_ndoc += len(paid_ndoc)

    sticky_auto = list(batch.get("sticky_auto_clients") or [])
    if sticky_auto:
        lines.append(f"\n<b>Автопродление без нового согласования</b> — {len(sticky_auto)}:")
        for client in sticky_auto:
            lines.append(f"  • {client.get('name', '—')} ({client.get('manager', '—')})")
        total_send += len(sticky_auto)

    admin_decisions = _build_admin_decisions(batch)
    admin_selected = sum(1 for v in admin_decisions.values() if v == "keep") + len(sticky_auto)

    lines += [
        "",
        f"<b>Итого к отправке в WA: {total_send}</b>",
        f"🤝 Договорились: {total_agreed} | 💰 Оплатил+документ: {total_paid_doc} | 💰 Без документа (→Саиде): {total_paid_ndoc}",
        f"<b>Сейчас выбрано вами к отправке:</b> {admin_selected}",
        "",
        "Нажмите <b>«Утвердить отправку»</b>, чтобы разрешить отправку.",
        "<i>Сообщения уйдут только после вашего подтверждения.</i>",
    ]
    return "\n".join(lines)


def _format_admin_detail_text(batch: Dict[str, Any]) -> str:
    """Подробный список всех клиентов для финального решения директора.
    На каждого клиента: имя, менеджер, долг, просрочка, телефон, статус согласования.
    """
    lines = [
        "📋 <b>Итоговый список перед отправкой — подробно</b>\n",
        f"Батч: <code>{batch['batch_id']}</code>\n",
    ]

    total_ready = 0
    total_all   = 0

    for manager_name, mgr_state in batch["managers"].items():
        mgr_status  = mgr_state.get("status", "pending")
        approved_set  = set(mgr_state.get("approved_names", []))
        rejected_set  = set(mgr_state.get("rejected_names", []))
        postponed_set = set(mgr_state.get("postponed_names", []))
        clients = mgr_state.get("clients", [])

        mgr_icon = {
            "approved_all": "✅", "manual": "✅", "manual_done": "✅",
            "rejected_all": "⛔", "pending": "⏳", "manual_editing": "⏳", "timeout": "⏰",
        }.get(mgr_status, "❓")

        lines.append(f"\n<b>{mgr_icon} {manager_name}</b>")

        def _fmt(n: float) -> str:
            return f"{n:,.0f}".replace(",", " ")

        for c in clients:
            name = c["name"]
            phone = c.get("phone", "") or "—"
            if c.get("invalid_phone"):
                phone = f"{phone} ⚠️ {c.get('phone_issue')}"
            viol_tag = " ⚠️" if c.get("violation_shipment") else ""
            type_label = _MSG_TYPE_LABELS.get(c.get("msg_type", ""), c.get("msg_type", ""))
            total_all += 1

            if name in approved_set:
                icon = "✅"
                total_ready += 1
            elif name in rejected_set:
                icon = "❌"
            elif name in postponed_set:
                icon = "⏸"
            else:
                icon = "◯"

            lines.append(
                f"  {icon} <b>{name}</b>{viol_tag}\n"
                f"     Долг: {_fmt(c['amount'])} тг · {_debt_age_text(c)}\n"
                f"     Отгрузки: {_fmt(c.get('debit', 0))} тг · Оплаты: {_fmt(c.get('credit', 0))} тг\n"
                f"     Тип: {type_label} · {c.get('reason', '—')}\n"
                f"     Тел: <code>{phone}</code>"
            )

    sticky_auto = list(batch.get("sticky_auto_clients") or [])
    if sticky_auto:
        lines.append("\n<b>Автопродление без нового согласования</b>")
        for client in sticky_auto:
            lines.append(
                f"  🔁 <b>{client.get('name', '—')}</b>\n"
                f"     Менеджер: {client.get('manager', '—')} · {_debt_age_text(client)}\n"
                f"     Долг: {_fmt(client.get('amount', 0))} тг · Тип: {_MSG_TYPE_LABELS.get(client.get('msg_type', ''), client.get('msg_type', '—'))}"
            )
            total_all += 1
            total_ready += 1

    lines += [
        "",
        f"<b>К отправке: {total_ready}</b> из {total_all}",
        "",
        "Нажмите <b>«✅ Разрешить тестовую отправку»</b> для финального утверждения.",
        "<i>Сообщения уйдут только после вашего подтверждения.</i>",
    ]
    return "\n".join(lines)


def _admin_keyboard(batch_id: str, batch: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    batch = batch or load_batch(batch_id) or {}
    rows = [
        [("✏️ Выбрать клиентов вручную",       f"wa_appr_adm_view|{batch_id}")],
    ]
    agreed_count = sum(len(mgr.get("agreed_names", []) or []) for mgr in (batch.get("managers") or {}).values())
    if agreed_count > 0:
        rows.append([("🤝 Проверить договорённости (%d)" % agreed_count, f"wa_appr_adm_agreed_review|{batch_id}")])
    rows.extend([
        [("✅ Утвердить отправку",             f"wa_appr_adm_ok|{batch_id}")],
        [("❌ Отменить",                       f"wa_appr_adm_no|{batch_id}")],
        [("⏸ Отложить",                       f"wa_appr_adm_later|{batch_id}")],
    ])
    return _inline_kb(rows)


def _format_agreed_review_text(batch: Dict[str, Any]) -> str:
    lines = ["🤝 <b>Договорённости менеджеров — проверка</b>", ""]
    agreed_clients = _iter_agreed_review_clients(batch)
    decisions = _get_agreed_review_decisions(batch)
    if not agreed_clients:
        lines += [
            "Сейчас в батче нет активных договорённостей для проверки.",
            "",
            "Нажмите «Назад к сводке».",
        ]
        return "\n".join(lines)

    current_manager = None
    for item in agreed_clients:
        if item["manager"] != current_manager:
            current_manager = item["manager"]
            if len(lines) > 2:
                lines.append("")
            lines.append(f"<b>{current_manager}</b>:")
        deadline_raw = item.get("deadline")
        deadline_text = ""
        if deadline_raw:
            try:
                deadline_text = datetime.fromisoformat(str(deadline_raw)).strftime("%d.%m.%Y")
            except ValueError:
                deadline_text = str(deadline_raw)
        detail_text = item.get("details") or "—"
        status = decisions.get(item["client_key"], "pending")
        status_prefix = {
            "accepted": "✅ Принято",
            "rejected": "❌ Не принято",
        }.get(status, "◯ На проверке")
        suffix = f" — до {deadline_text}" if deadline_text else ""
        lines.append(f"  • <b>{item['name']}</b>{suffix}")
        lines.append(f"    {detail_text}")
        lines.append(f"    <i>{status_prefix}</i>")
    lines += [
        "",
        "По каждому клиенту: принять договорённость или вернуть клиента в WA этого батча.",
    ]
    return "\n".join(lines)


def _agreed_review_keyboard(batch_id: str, agreed_clients: List[Dict[str, Any]], decisions: Dict[str, str]) -> Dict[str, Any]:
    rows = []
    for idx, item in enumerate(agreed_clients):
        key = item["client_key"]
        state = decisions.get(key, "pending")
        status_icon = {"accepted": "✅", "rejected": "❌", "pending": "◯"}.get(state, "◯")
        label = f"{status_icon} {item['manager']}: {item['name']}"
        short_label = label[:34] + "…" if len(label) > 34 else label
        rows.append([(short_label, f"wa_appr_adm_agreed_info|{batch_id}|{idx}")])
        if state == "pending":
            rows.append([
                ("✅ Принять", f"wa_appr_adm_agreed_ok|{batch_id}|{idx}"),
                ("❌ Не принимаю", f"wa_appr_adm_agreed_no|{batch_id}|{idx}"),
            ])
    rows.append([("↩️ Назад к сводке", f"wa_appr_adm_agreed_back|{batch_id}")])
    return _inline_kb(rows)


def _admin_send_now_keyboard(batch_id: str) -> Dict[str, Any]:
    return _inline_kb([
        [("📤 Отправить сейчас", f"wa_appr_adm_send|{batch_id}")],
        [("✏️ Изменить список клиентов", f"wa_appr_adm_view|{batch_id}")],
    ])


def _format_send_results_text(batch_id: str, results: List[Dict[str, Any]]) -> str:
    sent = sum(1 for r in results if r.get("status") == "sent")
    failed = sum(1 for r in results if r.get("status") == "failed")
    skipped = len(results) - sent - failed
    lines = [
        "📤 <b>Отправка завершена</b>",
        "",
        f"Батч: <code>{batch_id}</code>",
        f"Отправлено: <b>{sent}</b>",
        f"Ошибок: <b>{failed}</b>",
        f"Пропущено: <b>{skipped}</b>",
    ]
    failed_rows = [r for r in results if r.get("status") != "sent"]
    if failed_rows:
        lines.append("")
        lines.append("<b>Не отправилось / требует проверки:</b>")
        for row in failed_rows[:10]:
            name = row.get("name") or "—"
            reason = row.get("reason") or row.get("status") or "неизвестно"
            lines.append(f"  • {name} — {reason}")
    return "\n".join(lines)


def _format_send_blocked_text(batch_id: str, batch: Dict[str, Any], results: Optional[List[Dict[str, Any]]] = None) -> str:
    """Explain why send-now did not produce a real send result."""
    status = str(batch.get("status") or "admin_approved")
    summary = batch.get("send_summary") or {}
    rows = list(results or [])
    lines = [
        "❌ <b>Отправка не выполнена</b>",
        "",
        f"Батч: <code>{batch_id}</code>",
    ]
    if batch.get("send_in_progress"):
        lines.append("Статус: <b>отправка ещё выполняется</b>")
        lines.extend(["", "Проверьте итог через несколько секунд."])
        return "\n".join(lines)

    status_map = {
        "admin_approved": "отправка не стартовала",
        "send_empty":     "нет клиентов к отправке",
        "send_failed":    "отправка завершилась с ошибкой",
    }
    lines.append(f"Статус: <b>{status_map.get(status, status)}</b>")
    if summary:
        lines.append(
            f"Результат: отправлено <b>{summary.get('sent', 0)}</b>, "
            f"ошибок <b>{summary.get('failed', 0)}</b>, "
            f"пропущено <b>{summary.get('skipped', 0)}</b>."
        )
    else:
        lines.append("Итоговые send_results не записаны.")

    reason = next((str(r.get("reason") or "").strip() for r in rows if str(r.get("reason") or "").strip()), "")
    if reason:
        lines.extend(["", f"Причина: <i>{reason}</i>"])
    else:
        lines.extend(["", "Проверьте collector.log и send_reports.log для подробностей."])
    return "\n".join(lines)


async def send_admin_summary(batch: Dict[str, Any], bot=None) -> None:
    """Отправляет итоговую сводку администратору для финального решения."""
    try:
        admin_id = int(ADMIN_CHAT_ID)
    except (ValueError, TypeError):
        logger.error("send_admin_summary: ADMIN_CHAT_ID не задан или некорректен")
        return

    text   = _format_admin_summary_text(batch)
    markup = _admin_keyboard(batch["batch_id"], batch)

    msg_id = await _tg_send(admin_id, text, markup)
    if msg_id:
        batch["admin_message_id"] = msg_id
        batch["admin_chat_id"]    = admin_id
        save_batch(batch)
        logger.info("[%s] Сводка отправлена администратору (msg_id=%d)", batch["batch_id"], msg_id)
    else:
        logger.error("[%s] Не удалось отправить сводку администратору", batch["batch_id"])


async def send_admin_preview_notice(batch: Dict[str, Any], bot=None) -> None:
    """Уведомляет администратора о создании нового батча ДО ответов менеджеров.

    Информационное сообщение без inline-кнопок утверждения.
    Полноценная сводка с кнопками _admin_keyboard придёт отдельно из
    send_admin_summary, когда все менеджеры ответят. Это гарантирует,
    что админ видит батч даже если менеджеры игнорируют превью.
    """
    try:
        admin_id = int(ADMIN_CHAT_ID)
    except (ValueError, TypeError):
        logger.error("send_admin_preview_notice: ADMIN_CHAT_ID не задан или некорректен")
        return

    managers = batch.get("managers") or {}
    total_clients = sum(len(m.get("clients") or []) for m in managers.values())

    lines = [
        "📬 <b>Создан новый батч согласования WhatsApp</b>",
        "",
        f"Батч: <code>{batch.get('batch_id','—')}</code>",
        f"Клиентов всего: <b>{total_clients}</b>",
        f"Менеджеров: <b>{len(managers)}</b>",
        "",
    ]
    freshness_lines = _freshness_lines(batch)
    lines.extend(freshness_lines)
    if freshness_lines:
        lines.append("")
    lines += [
        "<b>Ожидается ответ от менеджеров:</b>",
    ]
    for mgr_name, mgr_state in managers.items():
        cnt = len(mgr_state.get("clients") or [])
        lines.append(f"  • {mgr_name} — {cnt} клиент(ов)")
    if batch.get("replaced_batch_id"):
        lines += [
            "",
            f"⚠️ Предыдущий активный батч <code>{batch['replaced_batch_id']}</code> закрыт как неактуальный.",
        ]
    lines += [
        "",
        "<i>Когда все менеджеры нажмут кнопки — пришлю итоговую сводку с кнопками утверждения.</i>",
        f"<i>Батч активен до: {batch.get('expires_at','—')}</i>",
    ]
    text = "\n".join(lines)

    msg_id = await _tg_send(admin_id, text)
    if msg_id:
        batch["admin_preview_msg_id"] = msg_id
        save_batch(batch)
        logger.info(
            "[%s] Превью-уведомление админу отправлено (msg_id=%d)",
            batch.get("batch_id", "?"), msg_id,
        )
    else:
        logger.error(
            "[%s] Не удалось отправить превью-уведомление админу",
            batch.get("batch_id", "?"),
        )


async def send_admin_auto_ready_notice(batch: Dict[str, Any], bot=None) -> None:
    """Информирует директора о sticky-клиентах, которые готовы к отправке без нового approval."""
    try:
        admin_id = int(ADMIN_CHAT_ID)
    except (ValueError, TypeError):
        logger.error("send_admin_auto_ready_notice: ADMIN_CHAT_ID не задан или некорректен")
        return

    clients = list(batch.get("sticky_auto_clients") or [])
    lines = [
        "🔁 <b>Повторное согласование не требуется</b>",
        "",
        f"Батч: <code>{batch.get('batch_id', '—')}</code>",
        f"К отправке без нового вопроса менеджерам: <b>{len(clients)}</b>",
        "",
        "У этих клиентов с прошлого решения нет новой оплаты, поэтому прошлое разрешение продолжено автоматически.",
    ]
    freshness_lines = _freshness_lines(batch)
    if freshness_lines:
        lines.extend([""] + freshness_lines)
    if clients:
        lines.append("")
        for client in clients:
            lines.append(f"  • {client.get('name', '—')} ({client.get('manager', '—')})")
    lines.extend([
        "",
        "Можно сразу запускать отправку кнопкой ниже.",
    ])
    text = "\n".join(lines)

    msg_id = await _tg_send(admin_id, text, _admin_send_now_keyboard(batch["batch_id"]))
    if msg_id:
        batch["admin_message_id"] = msg_id
        batch["admin_chat_id"] = admin_id
        save_batch(batch)
        logger.info("[%s] Sticky auto-ready notice sent to admin (msg_id=%d)", batch["batch_id"], msg_id)
    else:
        logger.error("[%s] Не удалось отправить sticky auto-ready notice админу", batch["batch_id"])


# ─── Admin callback handling ──────────────────────────────────────────────────

async def handle_admin_callback(
    data: str,
    chat_id: int,
    message_id: int,
    bot=None,
) -> bool:
    """Обрабатывает нажатие кнопки администратором."""
    if not data.startswith("wa_appr_adm_"):
        return False

    parts = data.split("|")
    if len(parts) < 2:
        return False

    action   = parts[0]
    batch_id = parts[1]

    batch = load_batch(batch_id)
    if not batch:
        await _tg_edit(chat_id, message_id, "⚠️ Запрос устарел. Батч не найден.")
        return True

    now_iso = datetime.now(tz=TZ).isoformat()
    if batch.get("status") in ("superseded", "expired", "too_late", "cancelled", "sent", "partially_sent", "send_failed", "send_empty"):
        await _tg_edit(
            chat_id,
            message_id,
            "⚠️ Этот запрос уже закрыт и больше неактуален.\n\n"
            "Если нужен новый список, работайте только с последним сообщением.",
        )
        return True

    if action == "wa_appr_adm_ok":
        logger.info("[%s] admin approve button pressed by chat_id=%s", batch_id, chat_id)
        # Финальное утверждение
        admin_decisions = _build_admin_decisions(batch)
        approved_clients = [dict(client) for client in (batch.get("sticky_auto_clients") or [])]
        for client in _iter_admin_clients(batch):
            if admin_decisions.get(client["_admin_key"]) != "keep":
                continue
            if client.get("invalid_phone"):
                logger.warning(
                    "[%s] admin approve skipped invalid_phone client: %s (%s)",
                    batch_id, client.get("name"), client.get("phone_issue"),
                )
                continue
            approved_clients.append({k: v for k, v in client.items() if k != "_admin_key"})

        batch["status"]            = "admin_approved"
        batch["admin_status"]      = "approved"
        batch["admin_approved_at"] = now_iso
        batch["approved_clients"]  = approved_clients
        save_batch(batch)
        try:
            from collector.collections_db import set_sticky_approval as _set_sticky_approval
            for client in approved_clients:
                msg_type = str(client.get("msg_type") or "")
                if msg_type not in {"strict_reminder", "stoplist_reminder", "legacy_tail_reminder", "partial_tail_reminder"}:
                    continue
                if float(client.get("amount", 0) or 0) <= 0 or float(client.get("debit", 0) or 0) != 0:
                    continue
                _set_sticky_approval(
                    client.get("name", ""),
                    batch_id=batch_id,
                    msg_type=msg_type,
                    amount=float(client.get("amount", 0) or 0),
                    credit=float(client.get("credit", 0) or 0),
                    debit=float(client.get("debit", 0) or 0),
                    stop_status=str(client.get("stop_status") or ""),
                )
        except Exception as _sticky_exc:
            logger.warning("[%s] sticky_approval save skipped: %s", batch_id, _sticky_exc)
        try:
            from collector.audit_log import audit as _audit
            _audit("batch_approved", batch_id=batch_id, clients=len(approved_clients))
        except Exception:
            pass

        _diff_block = ""
        _cancelled_exc = None
        try:
            import asyncio
            from collector.collections_engine import preview_batch_changes as _preview_changes
            logger.info("[%s] admin approve: preview_batch_changes start", batch_id)
            _diff_text = await asyncio.wait_for(
                asyncio.to_thread(_preview_changes, batch_id, approved_clients),
                timeout=10,
            )
            logger.info("[%s] admin approve: preview_batch_changes finish", batch_id)
            if _diff_text:
                _diff_block = f"\n\n⚠️ <b>Данные обновились с момента формирования:</b>\n{_diff_text}"
        except asyncio.CancelledError as _ce:
            # Coroutine was cancelled (e.g. bot shutdown). Capture it — we must
            # still update the Telegram message so the "Send" button appears,
            # then re-raise to let the framework handle cleanup properly.
            _cancelled_exc = _ce
        except asyncio.TimeoutError:
            logger.error("[%s] preview_batch_changes timeout during admin approve", batch_id)
            _diff_block = (
                "\n\n⚠️ <b>Проверка изменений заняла слишком много времени и была пропущена.</b>"
                "\nОтправка не блокируется, но перед отправкой стоит открыть свежую сводку ещё раз."
            )
        except Exception as _de:
            logger.warning("[%s] preview_batch_changes при утверждении: %s", batch_id, _de)

        text = (
            f"✅ <b>Отправка утверждена!</b>\n\n"
            f"К отправке выбрано клиентов: <b>{len(approved_clients)}</b>\n\n"
            + "\n".join(f"  • {c['name']} ({c.get('manager', '—')})" for c in approved_clients)
            + _diff_block
            + "\n\n"
            f"<b>Следующий шаг:</b> можно отправить прямо отсюда кнопкой ниже\n"
            f"или вручную командой:\n"
            f"<code>python -m collector.collections_engine --send-approved --batch-id {batch_id}</code>\n\n"
            f"<i>Предварительно убедитесь, что WHATSAPP_ENABLED=1 и LIVE_SEND_ALLOWED=1 выставлены в .env</i>"
        )
        await _tg_edit(chat_id, message_id, text, _admin_send_now_keyboard(batch_id))
        logger.info(
            "[%s] Администратор УТВЕРДИЛ отправку: %d клиентов",
            batch_id, len(approved_clients),
        )
        if _cancelled_exc is not None:
            raise _cancelled_exc

    elif action == "wa_appr_adm_send":
        logger.info("[%s] admin send button pressed by chat_id=%s status=%s admin_status=%s",
                    batch_id, chat_id, batch.get("status"), batch.get("admin_status"))
        if batch.get("admin_status") != "approved":
            logger.warning("[%s] send rejected: admin_status=%s", batch_id, batch.get("admin_status"))
            await _tg_edit(
                chat_id,
                message_id,
                "⚠️ Отправка недоступна: батч ещё не утверждён администратором.",
                _admin_keyboard(batch_id, batch),
            )
            return True

        expires_at = _parse_batch_dt(batch.get("expires_at"))
        if expires_at and datetime.now(tz=TZ) >= expires_at:
            batch["status"] = "too_late"
            batch["closed_at"] = datetime.now(tz=TZ).isoformat()
            batch["close_reason"] = "send_window_missed"
            batch.setdefault("escalation_reason", "send_window_missed")
            save_batch(batch)
            logger.info(
                "[%s] send rejected: batch expired at %s",
                batch_id, batch.get("expires_at"),
            )
            await _tg_edit(
                chat_id,
                message_id,
                (
                    "⛔ <b>Окно отправки закрыто.</b>\n\n"
                    f"Этот батч был активен до: <b>{batch.get('expires_at', '—')}</b>\n"
                    "После дедлайна отправка по старому батчу не выполняется.\n\n"
                    "Нужен новый актуальный батч."
                ),
            )
            return True

        if batch.get("status") in ("sent", "partially_sent"):
            logger.info("[%s] send skipped: already %s", batch_id, batch.get("status"))
            send_results = batch.get("send_results") or []
            await _tg_edit(chat_id, message_id, _format_send_results_text(batch_id, send_results))
            return True

        from collector.collections_engine import send_approved_batch

        await _tg_edit(chat_id, message_id, "⏳ <b>Запускаю отправку...</b>\n\nЭто займёт несколько секунд.")
        try:
            results = await send_approved_batch(batch_id)
        except Exception as send_exc:
            logger.error("[%s] send_approved_batch error: %s", batch_id, send_exc)
            await _tg_edit(chat_id, message_id, f"❌ <b>Отправка не выполнена</b>\n\n{send_exc}")
            return True
        batch = load_batch(batch_id) or batch
        send_results = batch.get("send_results") or results
        if batch.get("status") in ("sent", "partially_sent"):
            await _tg_edit(chat_id, message_id, _format_send_results_text(batch_id, send_results))
        else:
            await _tg_edit(chat_id, message_id, _format_send_blocked_text(batch_id, batch, send_results))
        logger.info("[%s] Администратор запустил отправку из Telegram: %d результатов", batch_id, len(send_results))

    elif action == "wa_appr_adm_view":
        decisions = _build_admin_decisions(batch)
        flat_clients = _iter_admin_clients(batch)
        _save_admin_decisions(batch, decisions)
        save_batch(batch)
        text = _format_admin_manual_header(batch, decisions)
        markup = _admin_client_list_keyboard(batch_id, flat_clients, decisions, _get_admin_reviewed_keys(batch))
        await _tg_edit(chat_id, message_id, text, markup)

    elif action == "wa_appr_adm_agreed_review":
        agreed_clients = _iter_agreed_review_clients(batch)
        decisions = _get_agreed_review_decisions(batch)
        text = _format_agreed_review_text(batch)
        markup = _agreed_review_keyboard(batch_id, agreed_clients, decisions)
        await _tg_edit(chat_id, message_id, text, markup)

    elif action in ("wa_appr_adm_agreed_ok", "wa_appr_adm_agreed_no", "wa_appr_adm_agreed_info"):
        if len(parts) < 3:
            return True
        cli_idx = int(parts[2])
        agreed_clients = _iter_agreed_review_clients(batch)
        if cli_idx >= len(agreed_clients):
            return True

        agreed_item = agreed_clients[cli_idx]
        client_key = agreed_item["client_key"]
        review_decisions = _get_agreed_review_decisions(batch)

        if action == "wa_appr_adm_agreed_info":
            deadline_raw = agreed_item.get("deadline")
            deadline_text = "—"
            if deadline_raw:
                try:
                    deadline_text = datetime.fromisoformat(str(deadline_raw)).strftime("%d.%m.%Y")
                except ValueError:
                    deadline_text = str(deadline_raw)
            state_label = {
                "accepted": "Принято",
                "rejected": "Не принято",
            }.get(review_decisions.get(client_key, "pending"), "На проверке")
            await _tg_send(
                chat_id,
                (
                    f"🤝 <b>{agreed_item['name']}</b>\n"
                    f"Менеджер: <b>{agreed_item['manager']}</b>\n"
                    f"Срок: <b>{deadline_text}</b>\n"
                    f"Статус: <b>{state_label}</b>\n\n"
                    f"{agreed_item.get('details') or '—'}"
                ),
            )
            return True

        manager_name = agreed_item["manager"]
        client_name = agreed_item["name"]
        mgr_state = (batch.get("managers") or {}).get(manager_name)
        if not mgr_state:
            return True

        if action == "wa_appr_adm_agreed_ok":
            review_decisions[client_key] = "accepted"
            batch["agreed_review_decisions"] = review_decisions
            _set_agreed_promise_status(client_name, "accepted", batch_id=batch_id, manager_name=manager_name)
            if "admin_keep_keys" in batch or "admin_skip_keys" in batch:
                admin_decisions = _build_admin_decisions(batch)
                admin_decisions[client_key] = "skip"
                _save_admin_decisions(batch, admin_decisions)
            save_batch(batch)
            logger.info("[%s] директор принял договорённость: %s / %s", batch_id, manager_name, client_name)

        elif action == "wa_appr_adm_agreed_no":
            review_decisions[client_key] = "rejected"
            batch["agreed_review_decisions"] = review_decisions
            if client_name in mgr_state.get("agreed_names", []):
                mgr_state["agreed_names"].remove(client_name)
            if client_name not in mgr_state.get("approved_names", []):
                mgr_state.setdefault("approved_names", []).append(client_name)
            _set_agreed_promise_status(client_name, "rejected", batch_id=batch_id, manager_name=manager_name)
            if "admin_keep_keys" in batch or "admin_skip_keys" in batch:
                admin_decisions = _build_admin_decisions(batch)
                admin_decisions[client_key] = "keep"
                _save_admin_decisions(batch, admin_decisions)
            save_batch(batch)
            mgr_chat_id = mgr_state.get("chat_id")
            if mgr_chat_id:
                await _tg_send(
                    int(mgr_chat_id),
                    (
                        f"⚠️ Директор не принял договорённость по <b>{client_name}</b>.\n"
                        f"Клиент войдёт в WA-рассылку этого батча.\n"
                        f"Твоя договорённость аннулирована."
                    ),
                )
            logger.info("[%s] директор отклонил договорённость: %s / %s", batch_id, manager_name, client_name)

        agreed_clients = _iter_agreed_review_clients(batch)
        text = _format_agreed_review_text(batch)
        markup = _agreed_review_keyboard(batch_id, agreed_clients, _get_agreed_review_decisions(batch))
        await _tg_edit(chat_id, message_id, text, markup)

    elif action == "wa_appr_adm_agreed_back":
        text = _format_admin_summary_text(batch)
        markup = _admin_keyboard(batch_id, batch)
        await _tg_edit(chat_id, message_id, text, markup)

    elif action in ("wa_appr_adm_cli_keep", "wa_appr_adm_cli_skip", "wa_appr_adm_info"):
        if len(parts) < 3:
            return True
        cli_idx = int(parts[2])
        flat_clients = _iter_admin_clients(batch)
        if cli_idx >= len(flat_clients):
            return True

        client = flat_clients[cli_idx]
        if action == "wa_appr_adm_info":
            await _tg_send(chat_id, _format_admin_client_info(client))
            return True

        decisions = _build_admin_decisions(batch)
        decisions[client["_admin_key"]] = "keep" if action == "wa_appr_adm_cli_keep" else "skip"
        reviewed_keys = _get_admin_reviewed_keys(batch)
        reviewed_keys.add(client["_admin_key"])
        _save_admin_decisions(batch, decisions)
        batch["admin_reviewed_keys"] = sorted(reviewed_keys)
        save_batch(batch)

        text = _format_admin_manual_header(batch, decisions)
        markup = _admin_client_list_keyboard(batch_id, flat_clients, decisions, reviewed_keys)
        await _tg_edit(chat_id, message_id, text, markup)

    elif action == "wa_appr_adm_done":
        decisions = _build_admin_decisions(batch)
        _save_admin_decisions(batch, decisions)
        save_batch(batch)
        text = (
            "✅ <b>Выбор администратора сохранён.</b>\n\n"
            f"К отправке выбрано: <b>{sum(1 for v in decisions.values() if v == 'keep')}</b>\n"
            f"Не отправлять: <b>{sum(1 for v in decisions.values() if v == 'skip')}</b>\n\n"
            "Можно утвердить отправку или вернуться к списку и изменить выбор."
        )
        markup = _admin_keyboard(batch_id, batch)
        await _tg_edit(chat_id, message_id, text, markup)

    elif action == "wa_appr_adm_back":
        text = _format_admin_summary_text(batch)
        markup = _admin_keyboard(batch_id, batch)
        await _tg_edit(chat_id, message_id, text, markup)

    elif action == "wa_appr_adm_no":
        batch["status"]       = "cancelled"
        batch["admin_status"] = "cancelled"
        save_batch(batch)

        text = (
            "❌ <b>Рассылка отменена.</b>\n\n"
            "Ни одно сообщение клиентам не отправлено.\n"
            "Для повторного запуска — новый dry-run завтра."
        )
        await _tg_edit(chat_id, message_id, text)
        logger.info("[%s] Администратор ОТМЕНИЛ рассылку", batch_id)

    elif action == "wa_appr_adm_later":
        batch["admin_status"] = "postponed"
        save_batch(batch)

        text = (
            "⏸ <b>Рассылка отложена.</b>\n\n"
            "Вы можете вернуться к этому запросу в течение дня.\n"
            "Батч будет активен до конца рабочего дня."
        )
        await _tg_edit(chat_id, message_id, text)
        logger.info("[%s] Администратор отложил решение", batch_id)

    else:
        return False

    return True


# ─── Combined callback router ─────────────────────────────────────────────────

async def handle_callback(
    data: str,
    chat_id: int,
    message_id: int,
    bot=None,
) -> bool:
    """Единая точка входа для всех wa_appr_* callback."""
    if data.startswith("wa_appr_mgr_") or data.startswith("wa_appr_cli_"):
        return await handle_manager_callback(data, chat_id, message_id, bot)
    if data.startswith("wa_appr_adm_"):
        return await handle_admin_callback(data, chat_id, message_id, bot)
    return False


# ─── Query helpers ────────────────────────────────────────────────────────────

def is_ready_for_send(batch_id: str) -> bool:
    """True только если администратор утвердил батч."""
    batch = load_batch(batch_id)
    if not batch:
        return False
    return (
        batch.get("admin_status") == "approved"
        and batch.get("status") in ("admin_approved", "partially_sent", "sent")
    )


def get_approved_clients(batch_id: str) -> List[Dict[str, Any]]:
    """Возвращает список одобренных клиентов после admin approve."""
    batch = load_batch(batch_id)
    if not batch or batch.get("admin_status") != "approved":
        return []
    return batch.get("approved_clients", [])


def record_send_results(batch_id: str, results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Stores per-client live-send results back into an approved batch."""
    batch = load_batch(batch_id)
    if not batch:
        return None

    now_iso = datetime.now(tz=TZ).isoformat()
    previous_results = batch.get("send_results") or []
    merged: Dict[str, Dict[str, Any]] = {}

    def _result_key(row: Dict[str, Any]) -> str:
        name = str(row.get("name") or "").strip().lower()
        phone = "".join(c for c in str(row.get("phone") or "") if c.isdigit())
        return f"{name}|{phone}"

    for row in previous_results:
        if isinstance(row, dict):
            merged[_result_key(row)] = row
    for row in results:
        if isinstance(row, dict):
            merged[_result_key(row)] = row

    all_results = list(merged.values())
    sent = sum(1 for r in all_results if r.get("status") == "sent")
    failed = sum(1 for r in all_results if r.get("status") == "failed")
    skipped = sum(1 for r in all_results if r.get("status") == "skipped")
    approved_total = len(batch.get("approved_clients") or [])

    batch["send_results"] = all_results
    batch["send_completed_at"] = now_iso
    batch["send_in_progress"] = False
    batch["send_lock_released_at"] = now_iso
    batch["send_lock_release_reason"] = "record_send_results"
    batch["send_summary"] = {
        "sent": sent,
        "failed": failed,
        "skipped": skipped,
        "total": len(all_results),
        "approved_total": approved_total,
    }
    if not all_results:
        batch["status"] = "send_empty"
    elif failed or skipped:
        batch["status"] = "partially_sent" if sent else "send_failed"
    elif approved_total and len(all_results) < approved_total:
        batch["status"] = "partially_sent"
    else:
        batch["status"] = "sent"
    save_batch(batch)
    try:
        from collector.audit_log import audit as _audit
        _audit(batch["status"],
               batch_id=batch_id, sent=sent, failed=failed,
               skipped=skipped, total=len(all_results))
    except Exception:
        pass
    return batch


def get_pending_managers(batch_id: str) -> List[str]:
    """Возвращает имена менеджеров, которые ещё не ответили."""
    batch = load_batch(batch_id)
    if not batch:
        return []
    return [
        name for name, state in batch["managers"].items()
        if state.get("status") in ("pending", "manual_editing")
    ]


# ─── Обещания 🤝 Договорились — хранение и авто-возврат ──────────────────────

_MONTH_RU = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
_DEFAULT_PROMISE_DAYS = 3   # если дату не удалось извлечь из текста


_WEEKDAY_RU = {
    "понедельник": 0, "пн": 0,
    "вторник": 1, "вт": 1,
    "среда": 2, "среду": 2, "ср": 2,
    "четверг": 3, "чт": 3,
    "пятница": 4, "пятницу": 4, "пятниц": 4, "пт": 4,
    "суббота": 5, "субботу": 5, "сб": 5,
    "воскресенье": 6, "вс": 6,
}


def _extract_deadline_from_text(text: str) -> Optional[date]:
    """Пытается извлечь дату обещания из свободного текста.

    Поддерживает:
      - "15 мая" / "до 15 мая" / "к 15 мая"
      - "15.05" / "15/05" / "15.05.2026"
      - "через 3 дня" / "через неделю" / "через 2 недели"
      - "сегодня" / "завтра" / "послезавтра"
      - "в пятницу" / "до пятницы" / "к понедельнику" / "на среду"
      - "до конца недели" / "до выходных"
      - "5 числа" / "к 5 числу"

    Возвращает None если ничего не распознано (раньше всегда возвращался дефолт
    +3 дня молча — менеджер не знал, что бот не понял дату). Caller теперь
    может переспросить вместо тихого дефолта.
    """
    now = datetime.now(tz=TZ)
    today = now.date()
    t = (text or "").lower()

    # «сегодня»
    if re.search(r"\bсегодня\b", t):
        return today
    # «завтра»
    if re.search(r"\bзавтра\b", t):
        return today + timedelta(days=1)
    # «послезавтра» / «после завтра»
    if re.search(r"\bпосле\s*завтра\b|\bпослезавтра\b", t):
        return today + timedelta(days=2)
    # «через N дней»
    m = re.search(r"через\s+(\d+)\s+дн", t)
    if m:
        return today + timedelta(days=int(m.group(1)))
    # «через неделю / N недель»
    if re.search(r"через\s+неделю", t):
        return today + timedelta(days=7)
    m = re.search(r"через\s+(\d+)\s+недел", t)
    if m:
        return today + timedelta(days=7 * int(m.group(1)))

    # «до конца недели» / «до выходных» → ближайшая пятница
    if re.search(r"до\s+конца\s+недели|до\s+выходн", t):
        ahead = (4 - today.weekday()) % 7
        if ahead == 0:
            ahead = 7
        return today + timedelta(days=ahead)

    # День недели: «в пятницу», «до пятницы», «к понедельнику», «во вторник», «на среду»
    for keyword, wday in _WEEKDAY_RU.items():
        if re.search(rf"\b(в|во|до|к|на)\s+{keyword}", t):
            ahead = (wday - today.weekday()) % 7
            if ahead == 0:
                ahead = 7
            return today + timedelta(days=ahead)

    # «15 мая» / «до 15 мая» / «к 15 мая»
    pattern_ru = r'(\d{1,2})\s+(' + '|'.join(_MONTH_RU.keys()) + r')'
    m = re.search(pattern_ru, text, re.IGNORECASE)
    if m:
        day   = int(m.group(1))
        month = _MONTH_RU[m.group(2).lower()]
        year  = now.year
        try:
            d = date(year, month, day)
            if d < today:
                d = date(year + 1, month, day)
            return d
        except ValueError:
            pass
    # «15.05» / «15/05» / «15.05.2026»
    m = re.search(r'(\d{1,2})[./](\d{1,2})(?:[./](\d{4}))?', text)
    if m:
        try:
            day, month = int(m.group(1)), int(m.group(2))
            year = int(m.group(3)) if m.group(3) else now.year
            d = date(year, month, day)
            if d < today:
                d = date(year + 1, month, day)
            return d
        except ValueError:
            pass
    # «5 числа» / «к 5 числу» — день текущего месяца, иначе следующего
    m = re.search(r'(\d{1,2})\s*числ', t)
    if m:
        try:
            day = int(m.group(1))
            d = date(now.year, now.month, day)
            if d < today:
                if now.month == 12:
                    d = date(now.year + 1, 1, day)
                else:
                    d = date(now.year, now.month + 1, day)
            return d
        except ValueError:
            pass

    return None  # не распознано — пусть caller переспросит


def _extract_deadline_or_default(text: str) -> date:
    """Обратно-совместимый враппер: если парсер вернул None → дефолт.

    Использовать только для legacy-вызовов, где переспрашивать невозможно.
    Новые flow должны использовать _extract_deadline_from_text() и обрабатывать None.
    """
    d = _extract_deadline_from_text(text)
    if d is not None:
        return d
    return (datetime.now(tz=TZ) + timedelta(days=_DEFAULT_PROMISE_DAYS)).date()


def _load_promises() -> Dict[str, Any]:
    try:
        return json.loads(_PROMISES_PATH.read_text(encoding="utf-8")) if _PROMISES_PATH.exists() else {}
    except Exception:
        return {}


def _save_promises(data: Dict[str, Any]) -> None:
    import tempfile
    tmp = _PROMISES_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_PROMISES_PATH)


def save_agreed_promise(
    client_name: str,
    manager_name: str,
    details: str,
    batch_id: str,
) -> date:
    """Сохраняет обещание менеджера. Возвращает извлечённую дату дедлайна.

    Если в тексте дата не распознана, использует дефолт (+_DEFAULT_PROMISE_DAYS).
    Для нового UX-flow (где можно переспросить) — используйте
    `_extract_deadline_from_text()` напрямую и обработайте None.
    """
    deadline = _extract_deadline_or_default(details)
    promises = _load_promises()
    promises[client_name] = {
        "manager":    manager_name,
        "batch_id":   batch_id,
        "details":    details,
        "deadline":   deadline.isoformat(),
        "set_at":     datetime.now(tz=TZ).isoformat(),
        "status":     "active",   # active | accepted | rejected | broken | fulfilled
    }
    _save_promises(promises)
    return deadline


def get_agreed_promise_stats() -> Dict[str, Any]:
    """Агрегирует статистику обещаний менеджеров из wa_agreed_promises.json."""
    promises = _load_promises()
    today = datetime.now(tz=TZ).date()
    statuses = ("active", "accepted", "rejected", "fulfilled", "broken")
    totals: Dict[str, int] = {name: 0 for name in statuses}
    totals.update({"total": 0, "in_control": 0, "overdue_active": 0})
    managers: Dict[str, Dict[str, Any]] = {}

    for client_name, promise in promises.items():
        manager_name = str(promise.get("manager") or "—")
        status = str(promise.get("status") or "active")
        deadline_raw = promise.get("deadline")
        deadline_obj: Optional[date] = None
        if isinstance(deadline_raw, str):
            try:
                deadline_obj = date.fromisoformat(deadline_raw)
            except ValueError:
                deadline_obj = None

        mgr_stats = managers.setdefault(
            manager_name,
            {
                "manager": manager_name,
                "total": 0,
                "active": 0,
                "accepted": 0,
                "rejected": 0,
                "fulfilled": 0,
                "broken": 0,
                "in_control": 0,
                "overdue_active": 0,
                "clients": [],
            },
        )

        mgr_stats["total"] += 1
        totals["total"] += 1
        if status in statuses:
            mgr_stats[status] += 1
            totals[status] += 1
        if status in {"active", "accepted"}:
            mgr_stats["in_control"] += 1
            totals["in_control"] += 1
            if deadline_obj and deadline_obj < today:
                mgr_stats["overdue_active"] += 1
                totals["overdue_active"] += 1

        mgr_stats["clients"].append(
            {
                "name": client_name,
                "status": status,
                "deadline": deadline_obj.isoformat() if deadline_obj else "",
                "details": str(promise.get("details") or ""),
                "batch_id": str(promise.get("batch_id") or ""),
            }
        )

    managers_list = sorted(
        managers.values(),
        key=lambda item: (-item["broken"], -item["fulfilled"], -item["total"], item["manager"].lower()),
    )
    for item in managers_list:
        resolved = item["fulfilled"] + item["broken"]
        item["success_rate"] = round((item["fulfilled"] / resolved) * 100, 1) if resolved else None

    return {
        "generated_at": datetime.now(tz=TZ).isoformat(),
        "totals": totals,
        "managers": managers_list,
    }


def format_agreed_promise_stats_text() -> str:
    """Формирует директорскую сводку качества обещаний менеджеров."""
    stats = get_agreed_promise_stats()
    totals = stats.get("totals", {})
    managers = stats.get("managers", [])

    if not totals.get("total"):
        return "🤝 <b>Обещания менеджеров</b>\n\nПока нет сохранённых договорённостей."

    lines = [
        "🤝 <b>Обещания менеджеров</b>",
        "",
        f"Всего: <b>{totals.get('total', 0)}</b>",
        f"На контроле: <b>{totals.get('in_control', 0)}</b>",
        f"Исполнено: <b>{totals.get('fulfilled', 0)}</b>",
        f"Сорвано: <b>{totals.get('broken', 0)}</b>",
        f"Отклонено директором: <b>{totals.get('rejected', 0)}</b>",
    ]
    if totals.get("overdue_active", 0):
        lines.append(f"Просрочено без закрытия: <b>{totals['overdue_active']}</b>")

    if managers:
        lines.append("")
        lines.append("<b>По менеджерам:</b>")
        for item in managers:
            line = (
                f"• <b>{item['manager']}</b> — всего {item['total']}, "
                f"в работе {item['in_control']}, исполнено {item['fulfilled']}, "
                f"сорвано {item['broken']}, отклонено {item['rejected']}"
            )
            if item.get("overdue_active"):
                line += f", просрочено {item['overdue_active']}"
            if item.get("success_rate") is not None:
                line += f", успех {item['success_rate']:.1f}%"
            lines.append(line)

    return "\n".join(lines)


def get_second_chance_blocked_clients() -> List[str]:
    """Клиенты у которых зафиксировано сорванное обещание — «Договорились» запрещено."""
    promises = _load_promises()
    return [name for name, p in promises.items() if p.get("status") == "broken"]


async def check_broken_agreed_deadlines(bot=None) -> int:
    """Ежедневная проверка: если дедлайн прошёл и долг остался — фиксируем нарушение.

    Уведомляет менеджера и директора. Возвращает число зафиксированных нарушений.
    """
    promises = _load_promises()
    today    = datetime.now(tz=TZ).date()
    broken_count = 0
    changed = False

    # Загружаем актуальную дебиторку чтобы проверить остался ли долг
    try:
        from collector.debt_monitor import load_latest_debt_json
        debt_data = load_latest_debt_json() or {}
        debt_clients = {d.get("name", ""): d.get("debt", 0) for d in debt_data.get("clients", [])}
    except Exception:
        debt_clients = {}

    try:
        admin_id = int(ADMIN_CHAT_ID)
    except (ValueError, TypeError):
        admin_id = 0

    managers_cfg = _load_managers_cfg()

    for client_name, promise in promises.items():
        if promise.get("status") not in {"active", "accepted"}:
            continue
        try:
            deadline = date.fromisoformat(promise["deadline"])
        except (KeyError, ValueError):
            continue
        if today <= deadline:
            continue  # срок ещё не прошёл

        # Срок прошёл — проверяем долг
        debt_remaining = debt_clients.get(client_name, -1)
        if debt_remaining == -1:
            continue  # клиента нет в актуальной дебиторке — возможно оплатил, пропускаем

        if debt_remaining <= 0:
            # Долг погашен — помечаем fulfilled
            promise["status"] = "fulfilled"
            promise["closed_at"] = datetime.now(tz=TZ).isoformat()
            changed = True
            continue

        # Долг остался — обещание нарушено
        promise["status"] = "broken"
        promise["broken_at"] = datetime.now(tz=TZ).isoformat()
        changed = True
        broken_count += 1

        manager_name = promise.get("manager", "")
        details_text = promise.get("details", "—")
        deadline_str = deadline.strftime("%d.%m")

        # Уведомляем менеджера
        mgr_chat = managers_cfg.get(manager_name)
        if mgr_chat and bot:
            try:
                await bot.send_message(
                    chat_id=int(mgr_chat),
                    text=(
                        f"⚠️ <b>Обещание нарушено.</b>\n\n"
                        f"Клиент <b>{client_name}</b> не оплатил к {deadline_str}.\n"
                        f"Твоя договорённость: <i>{details_text}</i>\n\n"
                        f"Клиент вернётся в следующую рассылку автоматически.\n"
                        f"Повторно использовать «Договорились» по нему — нельзя."
                    ),
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.warning("broken promise notify manager error: %s", e)

        # Уведомляем директора
        if admin_id and bot:
            try:
                await bot.send_message(
                    chat_id=admin_id,
                    text=(
                        f"📌 <b>Нарушено обещание оплаты.</b>\n\n"
                        f"Менеджер: <b>{manager_name}</b>\n"
                        f"Клиент: <b>{client_name}</b>\n"
                        f"Договорились: <i>{details_text}</i>\n"
                        f"Срок был: {deadline_str}\n\n"
                        f"Клиент возвращается в рассылку. «Договорились» для менеджера заблокировано."
                    ),
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.warning("broken promise notify admin error: %s", e)

        logger.info("Нарушение обещания: %s (менеджер %s, срок %s)", client_name, manager_name, deadline_str)

    if changed:
        _save_promises(promises)
    return broken_count


# ─── Обработка входящих сообщений от менеджеров (фото/детали) ────────────────

def find_manager_waiting_state(chat_id: int) -> Optional[Dict[str, Any]]:
    """Ищет менеджера с waiting_for_proof или waiting_for_agreed по chat_id.

    Возвращает {batch_id, manager_name, mgr_state, wait_type} или None.
    """
    batches = _load_batches()
    for bid, batch in batches.items():
        if batch.get("status") not in ("pending_managers", "pending_admin"):
            continue
        for mgr_name, mgr_state in batch.get("managers", {}).items():
            if mgr_state.get("chat_id") != chat_id:
                continue
            if mgr_state.get("waiting_for_proof"):
                return {
                    "batch_id":    bid,
                    "batch":       batch,
                    "manager_name": mgr_name,
                    "mgr_state":   mgr_state,
                    "wait_type":   "proof",
                    "wait_data":   mgr_state["waiting_for_proof"],
                }
            if mgr_state.get("waiting_for_agreed"):
                return {
                    "batch_id":    bid,
                    "batch":       batch,
                    "manager_name": mgr_name,
                    "mgr_state":   mgr_state,
                    "wait_type":   "agreed",
                    "wait_data":   mgr_state["waiting_for_agreed"],
                }
    return None


async def handle_manager_proof(
    chat_id: int,
    file_id: str,
    file_type: str,
    admin_chat_id: int,
    bot,
) -> bool:
    """Перехватывает фото/документ от менеджера ожидающего подтверждения оплаты.

    Пересылает директору, снимает waiting_for_proof. Возвращает True если обработано.
    """
    state = find_manager_waiting_state(chat_id)
    if not state or state["wait_type"] != "proof":
        return False

    batch      = state["batch"]
    mgr_state  = state["mgr_state"]
    mgr_name   = state["manager_name"]
    client_name = state["wait_data"].get("client_name", "—")

    mgr_state["waiting_for_proof"] = None
    _save_batches({state["batch_id"]: batch} | _load_batches())

    # Пересылаем директору
    caption = (
        f"💰 <b>Доказательство оплаты</b>\n\n"
        f"Менеджер: {mgr_name}\n"
        f"Клиент: {client_name}\n"
        f"Клиент снят из WA-рассылки. Документ приложен.\n"
        f"Саида не задействована."
    )
    try:
        if file_type == "photo":
            await bot.send_photo(chat_id=admin_chat_id, photo=file_id, caption=caption, parse_mode="HTML")
        else:
            await bot.send_document(chat_id=admin_chat_id, document=file_id, caption=caption, parse_mode="HTML")
    except Exception as e:
        logger.warning("handle_manager_proof: не удалось переслать директору: %s", e)

    await _tg_send(
        chat_id,
        f"✅ Документ по <b>{client_name}</b> отправлен директору. Спасибо.",
    )
    save_batch(batch)
    logger.info("[%s] %s прислал документ оплаты по %s", state["batch_id"], mgr_name, client_name)
    return True


async def handle_manager_agreed_details(chat_id: int, text: str) -> bool:
    """Принимает текст с деталями договорённости от менеджера.

    Сохраняет в agreed_details, снимает waiting_for_agreed. Возвращает True если обработано.

    Если в тексте не нашлась дата — НЕ сохраняем тихо с дефолтом,
    а переспрашиваем менеджера. Это защита от ситуации «договорился завтра 50000»
    где парсер раньше тихо ставил +3 дня и менеджер не знал.
    """
    state = find_manager_waiting_state(chat_id)
    if not state or state["wait_type"] != "agreed":
        return False

    batch       = state["batch"]
    mgr_state   = state["mgr_state"]
    mgr_name    = state["manager_name"]
    client_name = state["wait_data"].get("client_name", "—")
    now_iso     = datetime.now(tz=TZ).isoformat()

    # Сначала проверяем: распознана ли дата? Если нет — переспросить, не сохранять.
    parsed_date = _extract_deadline_from_text(text)
    if parsed_date is None:
        await _tg_send(
            chat_id,
            f"🤔 По <b>{client_name}</b> — не понял когда оплата.\n\n"
            "Напишите дату любым из способов:\n"
            "• <b>завтра</b> / <b>послезавтра</b> / <b>сегодня</b>\n"
            "• <b>в пятницу</b> / <b>до пятницы</b> / <b>к понедельнику</b>\n"
            "• <b>через 3 дня</b> / <b>через неделю</b>\n"
            "• <b>15 мая</b> / <b>до 15 мая</b>\n"
            "• <b>15.05</b> / <b>15.05.2026</b>\n\n"
            f"Например: «договорились до пятницы, 50 000 тг» или «{client_name} завтра 100000».",
        )
        return True

    deadline = save_agreed_promise(
        client_name=client_name,
        manager_name=mgr_name,
        details=text,
        batch_id=state["batch_id"],
    )
    deadline_str = deadline.strftime("%d.%m.%Y")

    mgr_state.setdefault("agreed_details", {})[client_name] = {
        "details":     text,
        "deadline":    deadline.isoformat(),
        "recorded_at": now_iso,
    }
    if client_name not in mgr_state.get("agreed_names", []):
        mgr_state.setdefault("agreed_names", []).append(client_name)
    mgr_state["waiting_for_agreed"] = None

    save_batch(batch)
    await _tg_send(
        chat_id,
        f"🤝 Зафиксировано по <b>{client_name}</b>:\n"
        f"<i>{text}</i>\n\n"
        f"Срок оплаты: <b>{deadline_str}</b>\n\n"
        f"Если оплата не придёт — клиент вернётся в рассылку автоматически.\n"
        f"Повторно использовать «Договорились» по нему будет нельзя.",
    )
    logger.info("[%s] %s договорённость по %s: %s (дедлайн %s)", state["batch_id"], mgr_name, client_name, text[:80], deadline_str)
    return True


# ─── Utility ──────────────────────────────────────────────────────────────────

def _load_managers_cfg() -> Dict[str, Any]:
    path = _ROOT / "config" / "managers.json"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def expire_old_batches() -> int:
    """Финализирует просроченные батчи. Возвращает количество закрытых.

    pending_admin / admin_approved + expired → too_late (окно отправки пропущено).
    Остальные незавершённые + expired         → expired (тихий таймаут).
    """
    batches = _load_batches()
    now = datetime.now(tz=TZ)
    count = 0
    for bid, batch in batches.items():
        if batch.get("status") in (
            "cancelled",
            "expired",
            "too_late",
            "sent",
            "partially_sent",
            "send_failed",
            "send_empty",
            "superseded",
        ):
            continue
        try:
            expires = datetime.fromisoformat(batch["expires_at"])
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=TZ)
            if now > expires:
                if batch.get("status") in ("pending_admin", "admin_approved"):
                    batch["status"] = "too_late"
                    batch["closed_at"] = now.isoformat()
                    batch["close_reason"] = "send_window_missed"
                    batch.setdefault("escalation_reason", "send_window_missed")
                    logger.info("Батч %s помечен как too_late", bid)
                else:
                    batch["status"] = "expired"
                    batch["expired_at"] = now.isoformat()
                    for mgr_state in (batch.get("managers") or {}).values():
                        if mgr_state.get("status") in ("pending", "manual_editing"):
                            mgr_state["status"] = "timeout"
                    logger.info("Батч %s помечен как expired", bid)
                count += 1
        except (KeyError, ValueError):
            pass
    if count:
        _save_batches(batches)
    return count


async def promote_silent_batches_to_admin(bot=None) -> int:
    """Через час молчания менеджеров переводит батч на этап решения администратора.

    Два пути эскалации:
    - Штатный: прошёл MANAGER_SILENCE_TIMEOUT_HOURS → admin summary
    - Узкое окно: до cutoff < MANAGER_SILENCE_TIMEOUT_HOURS → немедленная эскалация,
      менеджеры обходятся, директор решает сам
    - После cutoff → too_late, уведомление "сегодня не состоится"
    """
    batches = _load_batches()
    now = datetime.now(tz=TZ)
    changed_ids: List[str] = []
    too_late_ids: List[str] = []

    today_cutoff = _send_window_cutoff(now)
    hours_to_cutoff = (today_cutoff - now).total_seconds() / 3600
    tight_window = 0 < hours_to_cutoff < MANAGER_SILENCE_TIMEOUT_HOURS

    for bid, batch in batches.items():
        if batch.get("status") != "pending_managers":
            continue
        if batch.get("admin_status") not in (None, "pending"):
            continue
        created_at = _parse_batch_dt(batch.get("created_at"))
        if not created_at:
            continue

        silence_elapsed = now > created_at + timedelta(hours=MANAGER_SILENCE_TIMEOUT_HOURS)

        # Узкое окно — эскалируем немедленно, не ждём таймаута менеджеров
        if tight_window and not silence_elapsed:
            for mgr_state in (batch.get("managers") or {}).values():
                if mgr_state.get("status") in ("pending", "manual_editing"):
                    mgr_state["status"] = "timeout"
            batch["status"] = "pending_admin"
            batch["escalated_to_admin_at"] = now.isoformat()
            batch["escalation_reason"] = "tight_send_window"
            changed_ids.append(bid)
            logger.info(
                "[%s] узкое окно (до cutoff %.1f ч) — немедленная эскалация директору",
                bid, hours_to_cutoff,
            )
            continue

        if not silence_elapsed:
            continue

        # Уже за окном отправки — закрываем батч без эскалации
        if now >= today_cutoff:
            for mgr_state in (batch.get("managers") or {}).values():
                if mgr_state.get("status") in ("pending", "manual_editing"):
                    mgr_state["status"] = "timeout"
            batch["status"] = "too_late"
            batch["closed_at"] = now.isoformat()
            batch["close_reason"] = "send_window_missed"
            batch.setdefault("escalation_reason", "send_window_missed")
            too_late_ids.append(bid)
            logger.info("[%s] батч закрыт: окно отправки %d:%02d пропущено", bid, SEND_WINDOW_CUTOFF_HOUR, SEND_WINDOW_CUTOFF_MINUTE)
            continue

        pending_found = False
        for mgr_state in (batch.get("managers") or {}).values():
            if mgr_state.get("status") in ("pending", "manual_editing"):
                mgr_state["status"] = "timeout"
                pending_found = True
        if not pending_found:
            continue

        batch["status"] = "pending_admin"
        batch["escalated_to_admin_at"] = now.isoformat()
        batch["escalation_reason"] = "manager_silence_timeout"
        changed_ids.append(bid)
        logger.info("[%s] батч эскалирован админу после %d ч молчания", bid, MANAGER_SILENCE_TIMEOUT_HOURS)

    if not changed_ids and not too_late_ids:
        return 0

    _save_batches(batches)

    for bid in too_late_ids:
        try:
            admin_id = int(ADMIN_CHAT_ID)
            await _tg_send(
                admin_id,
                "⏰ <b>Рассылка сегодня не состоится.</b>\n\n"
                f"Батч <code>{bid}</code> создан, но менеджеры не ответили до {SEND_WINDOW_CUTOFF_HOUR}:{SEND_WINDOW_CUTOFF_MINUTE:02d}.\n"
                "Следующий батч будет создан автоматически при поступлении новой дебиторки.",
            )
        except (ValueError, TypeError):
            pass

    for bid in changed_ids:
        batch = load_batch(bid)
        if batch:
            await send_admin_summary(batch, bot)
    return len(changed_ids)

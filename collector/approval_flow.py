#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/approval_flow.py
UX согласования рассылки WhatsApp — менеджер → администратор.

Версия: 1.1.1 (2026-04-29)

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
import os
import re
import secrets
import tempfile
from datetime import datetime, timedelta
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
_BATCHES_PATH = _ROOT / "logs" / "wa_approval_batches.json"

BOT_TOKEN = os.getenv("TG_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID", "")

# Срок жизни батча — до конца рабочего дня (18:00)
BATCH_EXPIRE_HOURS = int(os.getenv("WA_APPROVAL_EXPIRE_HOURS", "9"))
MANAGER_SILENCE_TIMEOUT_HOURS = int(os.getenv("WA_MANAGER_SILENCE_HOURS", "1"))

logger = logging.getLogger(__name__)


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


# ─── Batch creation ───────────────────────────────────────────────────────────

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

        managers_state[manager_name] = {
            "clients":         normalized,
            "status":          "pending",        # pending | approved_all | rejected_all | manual_editing | manual_done | timeout
            "approved_names":  [],
            "rejected_names":  [],
            "postponed_names": [],
            "responded_at":    None,
        }

    batch: Dict[str, Any] = {
        "batch_id":        batch_id,
        "created_at":      now.isoformat(),
        "expires_at":      (now + timedelta(hours=BATCH_EXPIRE_HOURS)).isoformat(),
        "status":          "pending_managers",   # pending_managers | pending_admin | admin_approved | cancelled | expired
        "managers":        managers_state,
        "admin_status":    "pending",            # pending | approved | cancelled | postponed
        "admin_approved_at": None,
        "approved_clients": [],
    }
    logger.info(
        "Создан батч %s: менеджеров=%d, клиентов=%d",
        batch_id,
        len(managers_state),
        sum(len(v["clients"]) for v in managers_state.values()),
    )
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

    name_keys = _phone_keys_in_name(client_name)
    if name_keys and key not in name_keys:
        return False, "invalid_phone:name_mismatch"
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


async def _tg_edit(chat_id: int, message_id: int, text: str, markup=None) -> None:
    """Редактирует существующее Telegram-сообщение."""
    if not BOT_TOKEN:
        return
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
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(url, json=payload)
    except Exception as e:
        logger.error("TG edit exception: %s", e)


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
    lines += [
        "",
        "Пожалуйста, проверьте список и дайте разрешение на отправку.",
        "",
        "<i>Реальное сообщение клиентам уйдёт только после вашего и директора подтверждения.</i>",
    ]
    return "\n".join(lines)


def _manager_main_keyboard(batch_id: str, manager_idx: int) -> Dict[str, Any]:
    """Главная клавиатура менеджера (быстрые кнопки)."""
    b = batch_id
    i = manager_idx
    return _inline_kb([
        [("✅ Разрешить всем отправить",    f"wa_appr_mgr_ok|{b}|{i}")],
        [("👀 Посмотреть список клиентов",  f"wa_appr_mgr_view|{b}|{i}")],
        [("✏️ Выбрать вручную",             f"wa_appr_mgr_manual|{b}|{i}")],
        [("⛔ Не отправлять никому",        f"wa_appr_mgr_no|{b}|{i}")],
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
        # Кнопки действия (только если ещё не решено)
        rows.append([
            ("✅ Оставить", f"wa_appr_cli_keep|{batch_id}|{manager_idx}|{ci}"),
            ("❌ Убрать",   f"wa_appr_cli_skip|{batch_id}|{manager_idx}|{ci}"),
            ("⏸ Позже",    f"wa_appr_cli_later|{batch_id}|{manager_idx}|{ci}"),
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
    """Строит dict {client_name → keep|skip|later} из mgr_state."""
    d = {}
    for name in mgr_state.get("approved_names", []):
        d[name] = "keep"
    for name in mgr_state.get("rejected_names", []):
        d[name] = "skip"
    for name in mgr_state.get("postponed_names", []):
        d[name] = "later"
    return d


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
        approved = set(mgr_state.get("approved_names", []))
        for client in mgr_state.get("clients", []):
            key = _admin_client_key(manager_name, client["name"])
            decisions[key] = "keep" if client["name"] in approved else "skip"
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

    elif action == "wa_appr_mgr_no":
        mgr_state["status"]         = "rejected_all"
        mgr_state["approved_names"] = []
        mgr_state["rejected_names"] = [c["name"] for c in clients]
        mgr_state["postponed_names"] = []
        mgr_state["responded_at"]   = now_iso
        save_batch(batch)

        text = (
            f"⛔ <b>Зафиксировано.</b>\n\n"
            f"Никому из ваших {len(clients)} клиентов сообщения не отправляются.\n"
            f"Если передумаете — обратитесь к директору."
        )
        await _tg_edit(chat_id, message_id, text)
        logger.info("[%s] %s отклонил всех (%d клиентов)", batch_id, manager_name, len(clients))

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
            f"  ✅ Оставить — разрешить отправку\n"
            f"  ❌ Убрать — не отправлять\n"
            f"  ⏸ Позже — отложить до следующего дня\n\n"
            f"Когда выберете всех — нажмите <b>«Готово»</b>."
        )
        markup = _client_list_keyboard(batch_id, mgr_idx, clients, decisions)
        await _tg_edit(chat_id, message_id, text, markup)
        return True

    elif action == "wa_appr_mgr_done":
        # Менеджер завершил ручной выбор
        decisions = _build_decisions(mgr_state)
        approved = [c["name"] for c in clients if decisions.get(c["name"]) == "keep"]
        rejected = [c["name"] for c in clients if decisions.get(c["name"]) == "skip"]
        postponed = [c["name"] for c in clients if decisions.get(c["name"]) == "later"]
        undecided = [c["name"] for c in clients if decisions.get(c["name"]) not in ("keep", "skip", "later")]

        if undecided:
            text = (
                f"⚠️ Не все клиенты размечены.\n\n"
                f"Ещё нужно решить по {len(undecided)} клиент(ам):\n"
                + "\n".join(f"  • {n}" for n in undecided)
                + "\n\nПожалуйста, нажмите кнопки для каждого."
            )
            markup = _client_list_keyboard(batch_id, mgr_idx, clients, decisions)
            await _tg_edit(chat_id, message_id, text, markup)
            return True

        mgr_state["status"]          = "manual_done"
        mgr_state["approved_names"]  = approved
        mgr_state["rejected_names"]  = rejected
        mgr_state["postponed_names"] = postponed
        mgr_state["responded_at"]    = now_iso
        save_batch(batch)

        text = (
            f"✅ <b>Ваш выбор зафиксирован:</b>\n\n"
            f"  Разрешено: {len(approved)} клиент(ов)\n"
            f"  Убрано:   {len(rejected)} клиент(ов)\n"
            f"  Отложено: {len(postponed)} клиент(ов)\n\n"
            f"Итоговое решение — у директора."
        )
        await _tg_edit(chat_id, message_id, text)
        logger.info(
            "[%s] %s завершил ручной выбор: ok=%d, no=%d, later=%d",
            batch_id, manager_name, len(approved), len(rejected), len(postponed),
        )

    # ── Кнопки по конкретному клиенту ────────────────────────────────────────

    elif action in ("wa_appr_cli_keep", "wa_appr_cli_skip", "wa_appr_cli_later", "wa_appr_cli_info"):
        if len(parts) < 4:
            return True
        cli_idx = int(parts[3])
        if cli_idx >= len(clients):
            return True

        client_name = clients[cli_idx]["name"]

        if action == "wa_appr_cli_info":
            # Просто показываем детали клиента без изменения
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

        # Убираем из всех списков (чтобы не дублировалось)
        for lst_key in ("approved_names", "rejected_names", "postponed_names"):
            if client_name in mgr_state[lst_key]:
                mgr_state[lst_key].remove(client_name)

        if action == "wa_appr_cli_keep":
            mgr_state["approved_names"].append(client_name)
        elif action == "wa_appr_cli_skip":
            mgr_state["rejected_names"].append(client_name)
        elif action == "wa_appr_cli_later":
            mgr_state["postponed_names"].append(client_name)

        save_batch(batch)

        # Обновляем клавиатуру списка
        decisions = _build_decisions(mgr_state)
        markup = _client_list_keyboard(batch_id, mgr_idx, clients, decisions)
        header = (
            f"✏️ <b>Список клиентов</b> — {manager_name}\n\n"
            f"Выбираем по каждому клиенту.\n"
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

    total_ok = 0
    total_no = 0
    total_later = 0

    for manager_name, mgr_state in batch["managers"].items():
        status = mgr_state.get("status", "pending")
        approved  = mgr_state.get("approved_names", [])
        rejected  = mgr_state.get("rejected_names", [])
        postponed = mgr_state.get("postponed_names", [])
        clients   = mgr_state.get("clients", [])
        invalid_phone_clients = [c["name"] for c in clients if c.get("invalid_phone")]

        if status == "pending":
            status_label = "⏳ не ответил"
        elif status == "approved_all":
            status_label = f"✅ разрешил всех ({len(approved)})"
        elif status == "rejected_all":
            status_label = f"⛔ отклонил всех ({len(rejected)})"
        elif status == "manual_editing":
            status_label = f"✏️ выбирает вручную"
        elif status in ("manual", "manual_done"):
            status_label = f"✏️ выбрал вручную"
        elif status == "timeout":
            status_label = "⏰ не ответил вовремя"
        else:
            status_label = status

        lines.append(f"\n<b>{manager_name}</b> — {status_label}")

        if approved:
            lines.append(f"  Разрешено ({len(approved)}):")
            for n in approved:
                lines.append(f"    ✅ {n}")
        if rejected:
            lines.append(f"  Убрано ({len(rejected)}):")
            for n in rejected:
                lines.append(f"    ❌ {n}")
        if postponed:
            lines.append(f"  Отложено ({len(postponed)}):")
            for n in postponed:
                lines.append(f"    ⏸ {n}")
        if invalid_phone_clients:
            lines.append(f"  ⚠️ invalid_phone ({len(invalid_phone_clients)}):")
            for n in invalid_phone_clients:
                lines.append(f"    ⚠️ {n}")

        total_ok    += len(approved)
        total_no    += len(rejected)
        total_later += len(postponed)

    admin_decisions = _build_admin_decisions(batch)
    admin_selected = sum(1 for v in admin_decisions.values() if v == "keep")

    lines += [
        "",
        f"<b>Итого к отправке: {total_ok}</b> | убрано: {total_no} | отложено: {total_later}",
        f"<b>Сейчас выбрано вами к отправке:</b> {admin_selected}",
        "",
        "Если нужно, откройте список клиентов и вручную решите, кому отправлять, а кому нет.",
        "Нажмите <b>«Утвердить отправку»</b>, чтобы разрешить отправку выбранных клиентов.",
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

    lines += [
        "",
        f"<b>К отправке: {total_ready}</b> из {total_all}",
        "",
        "Нажмите <b>«✅ Разрешить тестовую отправку»</b> для финального утверждения.",
        "<i>Сообщения уйдут только после вашего подтверждения.</i>",
    ]
    return "\n".join(lines)


def _admin_keyboard(batch_id: str) -> Dict[str, Any]:
    return _inline_kb([
        [("✏️ Выбрать клиентов вручную",       f"wa_appr_adm_view|{batch_id}")],
        [("✅ Утвердить отправку",             f"wa_appr_adm_ok|{batch_id}")],
        [("❌ Отменить",                       f"wa_appr_adm_no|{batch_id}")],
        [("⏸ Отложить",                       f"wa_appr_adm_later|{batch_id}")],
    ])


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


async def send_admin_summary(batch: Dict[str, Any], bot=None) -> None:
    """Отправляет итоговую сводку администратору для финального решения."""
    try:
        admin_id = int(ADMIN_CHAT_ID)
    except (ValueError, TypeError):
        logger.error("send_admin_summary: ADMIN_CHAT_ID не задан или некорректен")
        return

    text   = _format_admin_summary_text(batch)
    markup = _admin_keyboard(batch["batch_id"])

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
    if batch.get("status") in ("superseded", "expired", "cancelled", "sent", "partially_sent", "send_failed", "send_empty"):
        await _tg_edit(
            chat_id,
            message_id,
            "⚠️ Этот запрос уже закрыт и больше неактуален.\n\n"
            "Если нужен новый список, работайте только с последним сообщением.",
        )
        return True

    if action == "wa_appr_adm_ok":
        # Финальное утверждение
        admin_decisions = _build_admin_decisions(batch)
        approved_clients = []
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

        text = (
            f"✅ <b>Отправка утверждена!</b>\n\n"
            f"К отправке выбрано клиентов: <b>{len(approved_clients)}</b>\n\n"
            + "\n".join(f"  • {c['name']} ({c.get('manager', '—')})" for c in approved_clients)
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

    elif action == "wa_appr_adm_send":
        if batch.get("admin_status") != "approved":
            await _tg_edit(
                chat_id,
                message_id,
                "⚠️ Отправка недоступна: батч ещё не утверждён администратором.",
                _admin_keyboard(batch_id),
            )
            return True

        if batch.get("status") in ("sent", "partially_sent"):
            send_results = batch.get("send_results") or []
            await _tg_edit(chat_id, message_id, _format_send_results_text(batch_id, send_results))
            return True

        from collector.collections_engine import send_approved_batch

        results = await send_approved_batch(batch_id)
        batch = load_batch(batch_id) or batch
        send_results = batch.get("send_results") or results
        await _tg_edit(chat_id, message_id, _format_send_results_text(batch_id, send_results))
        logger.info("[%s] Администратор запустил отправку из Telegram: %d результатов", batch_id, len(send_results))

    elif action == "wa_appr_adm_view":
        decisions = _build_admin_decisions(batch)
        flat_clients = _iter_admin_clients(batch)
        _save_admin_decisions(batch, decisions)
        save_batch(batch)
        text = _format_admin_manual_header(batch, decisions)
        markup = _admin_client_list_keyboard(batch_id, flat_clients, decisions, _get_admin_reviewed_keys(batch))
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
        markup = _admin_keyboard(batch_id)
        await _tg_edit(chat_id, message_id, text, markup)

    elif action == "wa_appr_adm_back":
        text = _format_admin_summary_text(batch)
        markup = _admin_keyboard(batch_id)
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


# ─── Utility ──────────────────────────────────────────────────────────────────

def _load_managers_cfg() -> Dict[str, Any]:
    path = _ROOT / "config" / "managers.json"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def expire_old_batches() -> int:
    """Помечает просроченные батчи как expired. Возвращает количество."""
    batches = _load_batches()
    now = datetime.now(tz=TZ)
    count = 0
    for bid, batch in batches.items():
        if batch.get("status") in (
            "admin_approved",
            "cancelled",
            "expired",
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
                batch["status"] = "expired"
                batch["expired_at"] = now.isoformat()
                for mgr_state in (batch.get("managers") or {}).values():
                    if mgr_state.get("status") in ("pending", "manual_editing"):
                        mgr_state["status"] = "timeout"
                count += 1
                logger.info("Батч %s помечен как expired", bid)
        except (KeyError, ValueError):
            pass
    if count:
        _save_batches(batches)
    return count


async def promote_silent_batches_to_admin(bot=None) -> int:
    """Через час молчания менеджеров переводит батч на этап решения администратора."""
    batches = _load_batches()
    now = datetime.now(tz=TZ)
    changed_ids: List[str] = []
    for bid, batch in batches.items():
        if batch.get("status") != "pending_managers":
            continue
        if batch.get("admin_status") not in (None, "pending"):
            continue
        created_at = _parse_batch_dt(batch.get("created_at"))
        if not created_at:
            continue
        if now <= created_at + timedelta(hours=MANAGER_SILENCE_TIMEOUT_HOURS):
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

    if not changed_ids:
        return 0

    _save_batches(batches)

    for bid in changed_ids:
        batch = load_batch(bid)
        if batch:
            await send_admin_summary(batch, bot)
    return len(changed_ids)

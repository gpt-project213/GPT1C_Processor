#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/approval_penalty.py
Штрафные баллы менеджеров за пропуск окна согласования WhatsApp-рассылки
и за неответ на CRM-запрос.

v1.1.2 (2026-05-14): ручной reset штрафов больше не реанимирует старые WA/CRM
  записи. State может хранить wa_reset_floor / crm_reset_floor; batch/pending
  старше этих отметок не поднимаются обратно ни при periodic backfill, ни при
  прямом process_batch_penalties(). Это закрывает повторное начисление штрафов
  после операционного сброса state в середине месяца.

v1.1.1 (2026-05-13): timeout после начатого manager-review теперь считается
  partial, даже если менеджер не добавил никого в approved_names. Раньше
  ветки "Договорились"/"Оплатил без документа"/ожидание деталей ошибочно
  классифицировались как full ignore.

v1.1.0 (2026-05-11): check_crm_ignores() — CRM-игноры в единый счётчик штрафов.
  CRM_IGNORE_MIN_AGE_HOURS (default 22): записи старше порога = игнор.
  Уведомление: та же формула, текст отличается от WA-пропуска.
  source: "wa" | "crm" в каждой записи ignores[].

v1.0.2 (2026-05-11): форматирование чисел через _n(); guard-ветки greeting/ack
  игнорируют suggested_reply; тесты на текст уведомлений.

v1.0.1 (2026-05-11): предупреждение показывает сумму следующего штрафа через
  _penalty_amount(2) вместо хардкода.

v1.0.0 (2026-05-11)

Правила:
  - 1-й пропуск в месяце (WA или CRM): предупреждение, штраф 0 тг
  - 2-й пропуск:                        2 000 тг
  - N-й пропуск (N ≥ 3):               N × 1 000 × 2 тг
  - Частичный WA-пропуск (-10%):        менеджер начал отвечать, но не завершил

Период: календарный месяц (1–последний день).

Триггеры:
  - WA: вызывается после финализации батча (process_batch_penalties)
  - CRM: каждые 30 мин (check_crm_ignores) сканирует crm_pending_state.json
  - Периодически: check_recent_batches — перепроверяет WA-батчи без уведомления
  Отчёт: в последний день месяца — Саиде + админу таблица, менеджерам — их строка.

State: logs/approval_penalty_state.json
"""
from __future__ import annotations

import calendar
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

TZ              = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))
ADMIN_CHAT_ID   = os.getenv("ADMIN_CHAT_ID", "")
SAIDA_CHAT_ID   = int(os.getenv("SAIDA_CHAT_ID", "920236287"))

_ROOT             = Path(__file__).resolve().parent.parent
_BATCHES_PATH     = _ROOT / "logs" / "wa_approval_batches.json"
_STATE_PATH       = _ROOT / "logs" / "approval_penalty_state.json"
_CRM_PENDING_PATH = _ROOT / "logs" / "crm_pending_state.json"

CRM_IGNORE_MIN_AGE_HOURS = float(os.getenv("CRM_IGNORE_MIN_AGE_HOURS", "22"))

TERMINAL_STATUSES = {
    "too_late", "sent", "partially_sent", "send_failed",
    "expired", "cancelled", "send_empty",
}


# ─── Формула штрафа ───────────────────────────────────────────────────────────

def _penalty_amount(ignore_num: int, partial: bool = False) -> int:
    """Штраф за ignore_num-й пропуск в месяце (1-based).

    1-й → 0 (предупреждение)
    N-й (N ≥ 2) → 1 000 × N   (2й=2000, 3й=6000*, 4й=8000*, 5й=10000*)
    * с 3-го дополнительно ×2
    """
    if ignore_num <= 1:
        return 0
    base = 1000 * ignore_num * (2 if ignore_num >= 3 else 1)
    return int(base * 0.9) if partial else base


def _cumulative_penalty(ignores: List[Dict[str, Any]]) -> int:
    return sum(e.get("penalty", 0) for e in ignores)


# ─── State ────────────────────────────────────────────────────────────────────

def _load_state() -> Dict[str, Any]:
    try:
        with open(_STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(state: Dict[str, Any]) -> None:
    tmp = _STATE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(_STATE_PATH)


def _month_key(dt: Optional[datetime] = None) -> str:
    dt = dt or datetime.now(TZ)
    return dt.strftime("%Y-%m")


def _ensure_month(state: Dict[str, Any], month: str) -> None:
    if state.get("month") != month:
        state.clear()
        state["month"] = month
        state["managers"] = {}


def build_reset_state(_now: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Операционный reset штрафов:
    - обнуляет счётчики за текущий месяц;
    - ставит floor-метки, чтобы старые WA/CRM записи не переехали обратно
      из batch/pending backfill.
    """
    now = (_now or datetime.now(TZ)).astimezone(TZ)
    floor = now.isoformat()
    return {
        "month": _month_key(now),
        "managers": {},
        "wa_reset_floor": floor,
        "crm_reset_floor": floor,
    }


def reset_penalty_state(_now: Optional[datetime] = None) -> Dict[str, Any]:
    """Safely reset approval penalties for the current month with backup."""
    now = (_now or datetime.now(TZ)).astimezone(TZ)
    state = build_reset_state(now)
    backup_path = ""
    if _STATE_PATH.exists():
        backup = _STATE_PATH.with_name(
            f"{_STATE_PATH.name}.bak-{now.strftime('%Y%m%d-%H%M%S')}-reset"
        )
        shutil.copy2(_STATE_PATH, backup)
        backup_path = str(backup)
    _save_state(state)
    return {
        "state": state,
        "month": state["month"],
        "backup_path": backup_path,
        "state_path": str(_STATE_PATH),
    }


def _parse_dt(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw)).astimezone(TZ)
    except (ValueError, TypeError):
        return None


def _reset_floor(state: Dict[str, Any], key: str, month: str) -> Optional[datetime]:
    dt = _parse_dt(state.get(key))
    if not dt:
        return None
    return dt if _month_key(dt) == month else None


def _mgr_state(state: Dict[str, Any], manager: str, chat_id: int) -> Dict[str, Any]:
    mgrs = state.setdefault("managers", {})
    if manager not in mgrs:
        mgrs[manager] = {"chat_id": chat_id, "ignores": []}
    elif chat_id:
        mgrs[manager]["chat_id"] = chat_id
    return mgrs[manager]


# ─── Классификация менеджерского статуса в батче ─────────────────────────────

def _classify_manager(mgr_data: Dict[str, Any]) -> Optional[str]:
    """
    Возвращает: 'ok' | 'full' | 'partial' | None (не финализирован).
    """
    status = mgr_data.get("status", "")
    if status == "approved_all":
        return "ok"
    if status == "timeout":
        decision_lists = (
            "approved_names",
            "rejected_names",
            "postponed_names",
            "agreed_names",
            "paid_with_doc_names",
            "paid_no_doc_names",
        )
        has_decision = any(bool(mgr_data.get(key)) for key in decision_lists)
        waiting_for_followup = bool(mgr_data.get("waiting_for_proof")) or bool(mgr_data.get("waiting_for_agreed"))
        has_response_marker = bool(mgr_data.get("responded_at"))
        return "partial" if (has_decision or waiting_for_followup or has_response_marker) else "full"
    return None  # pending / manual_editing — ещё не закрыт


# ─── Уведомление менеджера ────────────────────────────────────────────────────

def _miss_context_line(batch_date: str, source: str) -> str:
    if source == "crm":
        return f"Данные клиента не внесены в срок (CRM-запрос {batch_date})."
    return f"Пропущено окно согласования рассылки {batch_date}."


def _warn_context_line(batch_date: str, source: str) -> str:
    if source == "crm":
        return f"Вы не внесли данные клиента в срок (CRM-запрос {batch_date})."
    return f"Вы не ответили в окне согласования рассылки {batch_date}."


async def _notify_manager(
    manager: str,
    chat_id: int,
    ignore_num: int,
    penalty: int,
    cumulative: int,
    batch_date: str,
    partial: bool,
    bot,
    *,
    source: str = "wa",
) -> None:
    if not chat_id:
        return
    partial_note = " (частичный ответ, −10%)" if partial else ""
    _n = lambda v: f"{v:,}".replace(",", " ")

    if ignore_num == 1:
        next_penalty = _penalty_amount(2)
        text = (
            f"⚠️ <b>Предупреждение</b>\n\n"
            f"{_warn_context_line(batch_date, source)}\n"
            f"Это первый пропуск в этом месяце — штраф не начисляется.\n\n"
            f"⚠️ Следующий пропуск: <b>{_n(next_penalty)} тг</b>"
        )
    else:
        text = (
            f"🔴 <b>Штраф: {_n(penalty)} тг</b>{partial_note}\n\n"
            f"{_miss_context_line(batch_date, source)}\n"
            f"Пропусков за месяц: <b>{ignore_num}</b>\n"
            f"Итого штрафов за месяц: <b>{_n(cumulative)} тг</b>"
        )
    try:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
    except Exception as e:
        logger.warning("approval_penalty: ошибка уведомления %s (chat=%s): %s",
                       manager, chat_id, e)


# ─── Основная функция: обработка закрытого батча ─────────────────────────────

async def process_batch_penalties(batch: Dict[str, Any], bot) -> None:
    """
    Вызывается после финализации батча.
    Находит timeout-менеджеров, обновляет счётчик, уведомляет.
    """
    batch_status = batch.get("status", "")
    if batch_status not in TERMINAL_STATUSES:
        return

    created_raw = batch.get("created_at", "")
    try:
        created_dt = datetime.fromisoformat(created_raw).astimezone(TZ)
    except (ValueError, TypeError):
        created_dt = datetime.now(TZ)
    batch_date = created_dt.strftime("%d.%m.%Y")
    batch_id   = batch.get("batch_id", "")
    month      = _month_key(created_dt)

    state = _load_state()
    _ensure_month(state, month)
    wa_floor = _reset_floor(state, "wa_reset_floor", month)
    if wa_floor and created_dt < wa_floor:
        return

    managers = batch.get("managers", {})
    for mgr_name, mgr_data in managers.items():
        ignore_type = _classify_manager(mgr_data)
        if ignore_type == "ok" or ignore_type is None:
            continue

        chat_id = int(mgr_data.get("chat_id", 0) or 0)
        mgr = _mgr_state(state, mgr_name, chat_id)

        # Не обрабатывать дважды один батч
        if any(e.get("batch_id") == batch_id for e in mgr["ignores"]):
            continue

        partial   = (ignore_type == "partial")
        ignore_num = len(mgr["ignores"]) + 1
        penalty   = _penalty_amount(ignore_num, partial)
        cumulative = _cumulative_penalty(mgr["ignores"]) + penalty

        mgr["ignores"].append({
            "batch_id":  batch_id,
            "date":      batch_date,
            "type":      ignore_type,
            "source":    "wa",
            "penalty":   penalty,
            "partial":   partial,
            "notified":  False,
        })
        _save_state(state)

        await _notify_manager(
            mgr_name, chat_id, ignore_num, penalty, cumulative,
            batch_date, partial, bot, source="wa",
        )
        mgr["ignores"][-1]["notified"] = True
        _save_state(state)

        logger.info(
            "approval_penalty: %s → игнор №%d, штраф %d тг (батч %s)",
            mgr_name, ignore_num, penalty, batch_id,
        )


# ─── Периодическая проверка: свежие батчи без уведомления ────────────────────

async def check_recent_batches(bot) -> None:
    """
    Запускается каждые 30 мин. Проверяет батчи за текущий месяц,
    у которых не была вызвана process_batch_penalties.
    """
    try:
        with open(_BATCHES_PATH, encoding="utf-8") as f:
            batches: Dict[str, Any] = json.load(f)
    except (OSError, json.JSONDecodeError):
        return

    month = _month_key()
    state = _load_state()
    _ensure_month(state, month)
    wa_floor = _reset_floor(state, "wa_reset_floor", month)

    notified_batch_ids = {
        e["batch_id"]
        for mgr in state.get("managers", {}).values()
        for e in mgr.get("ignores", [])
    }

    for batch_id, batch in sorted(batches.items()):
        if batch.get("status") not in TERMINAL_STATUSES:
            continue
        if batch_id in notified_batch_ids:
            continue
        # Только батчи текущего месяца
        created_raw = batch.get("created_at", "")
        try:
            created_dt = datetime.fromisoformat(created_raw).astimezone(TZ)
        except (ValueError, TypeError):
            continue
        if _month_key(created_dt) != month:
            continue
        if wa_floor and created_dt < wa_floor:
            continue

        await process_batch_penalties(batch, bot)


# ─── CRM-игноры: менеджер не ответил на запрос ────────────────────────────────

async def check_crm_ignores(bot, _now: Optional[datetime] = None) -> None:
    """
    Запускается каждые 30 мин (вместе с check_recent_batches).
    Сканирует crm_pending_state.json: записи с created_at старше
    CRM_IGNORE_MIN_AGE_HOURS считаются игнором, если не отмечены ранее.

    Возраст считается от created_at (неизменяем), а НЕ от last_sent —
    иначе crm_phone_reminder_task() обнулял бы таймер на каждом напоминании.

    Проверка выполняется только в рабочие дни и окно 09–19 Almaty,
    чтобы не штрафовать за выходные и ночное время.

    _now: переопределить текущее время (для тестов).
    Идентификатор игнора: crm-{chat_id}-{YYYYMMDD создания} — уникален на дату.
    """
    now = _now or datetime.now(TZ)

    # Только рабочие часы 09–19
    if not (9 <= now.hour < 19):
        return
    # Только рабочие дни
    try:
        from bot.workday_checker import is_holiday_today
        if is_holiday_today():
            return
    except Exception:
        pass  # если модуль недоступен — продолжаем

    try:
        with open(_CRM_PENDING_PATH, encoding="utf-8") as f:
            pending: Dict[str, Any] = json.load(f)
    except (OSError, json.JSONDecodeError):
        return

    if not pending:
        return

    month = _month_key()
    state = _load_state()
    _ensure_month(state, month)
    crm_floor = _reset_floor(state, "crm_reset_floor", month)

    for chat_id_str, entry in pending.items():
        manager = entry.get("manager", "")
        if not manager:
            continue

        # Используем created_at; last_sent постоянно обновляется напоминаниями
        ts_raw = entry.get("created_at") or entry.get("last_sent")
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(ts_raw).astimezone(TZ)
        except (ValueError, TypeError):
            continue
        if crm_floor and ts < crm_floor:
            continue

        # F-05: менеджер явно нажал "Позже" — не штрафуем до истечения paused_until
        paused_raw = entry.get("paused_until")
        if paused_raw:
            try:
                paused_dt = datetime.fromisoformat(paused_raw).astimezone(TZ)
                if now < paused_dt:
                    continue
            except (ValueError, TypeError):
                pass

        age_hours = (now - ts).total_seconds() / 3600
        if age_hours < CRM_IGNORE_MIN_AGE_HOURS:
            continue

        batch_id   = f"crm-{chat_id_str}-{ts.strftime('%Y%m%d')}"
        batch_date = ts.strftime("%d.%m.%Y")
        chat_id    = int(chat_id_str)

        mgr = _mgr_state(state, manager, chat_id)
        if any(e.get("batch_id") == batch_id for e in mgr["ignores"]):
            continue

        ignore_num = len(mgr["ignores"]) + 1
        penalty    = _penalty_amount(ignore_num, partial=False)
        cumulative = _cumulative_penalty(mgr["ignores"]) + penalty

        mgr["ignores"].append({
            "batch_id": batch_id,
            "date":     batch_date,
            "type":     "crm_no_response",
            "source":   "crm",
            "penalty":  penalty,
            "partial":  False,
            "notified": False,
        })
        _save_state(state)

        await _notify_manager(
            manager, chat_id, ignore_num, penalty, cumulative,
            batch_date, False, bot, source="crm",
        )
        mgr["ignores"][-1]["notified"] = True
        _save_state(state)

        logger.info(
            "approval_penalty: %s → CRM-игнор №%d, штраф %d тг (%s)",
            manager, ignore_num, penalty, batch_id,
        )


# ─── Конец месяца: итоговый отчёт ────────────────────────────────────────────

def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", " ") + " тг" if n else "—"


async def send_monthly_penalty_report(bot) -> None:
    """
    Отправляет итоговый отчёт в последний день месяца:
    - Саиде + админу: таблица по всем менеджерам
    - Каждому менеджеру: только его строка
    """
    now   = datetime.now(TZ)
    last_day = calendar.monthrange(now.year, now.month)[1]
    if now.day != last_day:
        return

    state = _load_state()
    month = _month_key()
    if state.get("month") != month:
        logger.info("approval_penalty: нет данных за %s — отчёт не отправляем", month)
        return

    month_label = now.strftime("%B %Y").capitalize()
    managers    = state.get("managers", {})

    # ── Строки таблицы ──
    rows: List[str] = []
    total_penalty = 0
    for mgr_name, mgr in sorted(managers.items()):
        ignores    = mgr.get("ignores", [])
        n_ignores  = len(ignores)
        penalty    = _cumulative_penalty(ignores)
        total_penalty += penalty
        rows.append(
            f"• <b>{mgr_name}</b>: {n_ignores} пропуск(ов) — {_fmt(penalty)}"
        )

    if not rows:
        return

    # ── Отчёт для Саиды и Админа ──
    summary = (
        f"📊 <b>Штрафы за согласование рассылок — {month_label}</b>\n\n"
        + "\n".join(rows)
        + f"\n\n<b>ИТОГО штрафов: {_fmt(total_penalty)}</b>"
    )
    for target_id in [SAIDA_CHAT_ID, ADMIN_CHAT_ID]:
        if not target_id:
            continue
        try:
            await bot.send_message(chat_id=int(target_id),
                                   text=summary, parse_mode="HTML")
        except Exception as e:
            logger.warning("approval_penalty: ошибка отправки сводки (chat=%s): %s",
                           target_id, e)

    # ── Отчёт каждому менеджеру ──
    for mgr_name, mgr in managers.items():
        chat_id = int(mgr.get("chat_id", 0) or 0)
        if not chat_id:
            continue
        ignores  = mgr.get("ignores", [])
        n        = len(ignores)
        penalty  = _cumulative_penalty(ignores)
        mgr_text = (
            f"📊 <b>Штрафы за согласование — {month_label}</b>\n\n"
            f"Пропусков окна согласования: <b>{n}</b>\n"
            f"Сумма штрафов: <b>{_fmt(penalty)}</b>\n\n"
            f"Руководитель учтёт эту сумму при расчёте зарплаты."
        )
        try:
            await bot.send_message(chat_id=chat_id, text=mgr_text, parse_mode="HTML")
        except Exception as e:
            logger.warning("approval_penalty: ошибка отправки менеджеру %s: %s",
                           mgr_name, e)

    logger.info("approval_penalty: месячный отчёт отправлен (%d менеджеров)", len(managers))

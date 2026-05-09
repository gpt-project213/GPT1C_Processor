#!/usr/bin/env python
# coding: utf-8
"""
debt_stop_control.py · v1.0.13 (2026-05-08)

Контроль стоп-листа отгрузки — уведомление Саиды-бухгалтера.

Расписание (рабочие дни):
  14:00 → мониторинг: одобренные исключения перешли 15 дней → авто-стоп
  17:00 → запросы менеджерам по новым кандидатам
  каждые 30 мин → напоминания менеджерам по незакрытым запросам
  19:00 → нет ответа → эскалация руководителю
  22:00 → Саида получает финальный список «не отгружать»

Пороги:
  10+ дней   → 🔴 Молчание — запрос менеджеру
  15+ дней   → 🚫 Авто-стоп даже для одобренных исключений

Исключения:
  - config/weekly_clients.json            (платят раз в неделю)
  - clients.json[payment_terms_days] > 0  (договорная отсрочка)

Нарушители дисциплины (попали в авто-стоп):
  - При следующем появлении в списке — пропускают шаг менеджера,
    идут сразу к руководителю.
  - Сброс статуса нарушителя — только после полной оплаты + решения руководителя.

Состояние:
  reports/debt_stop_state.json    — суточное (сбрасывается каждое утро)
  reports/debt_stop_registry.json — постоянное (одобрения, нарушители, авто-стопы)
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", encoding="utf-8-sig", override=True)
except Exception:
    pass

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from collector.logging_utils import get_stop_logger


LOG = get_stop_logger("debt_stop_control")

# ── Пути ────────────────────────────────────────────────────────────
_THIS = Path(__file__).resolve()
ROOT = _THIS.parent.parent if _THIS.parent.name == "bot" else _THIS.parent
TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))
JSON_DIR   = ROOT / "reports" / "json"
CONFIG_DIR = ROOT / "config"

STATE_FILE        = ROOT / "reports" / "debt_stop_state.json"
REGISTRY_FILE     = ROOT / "reports" / "debt_stop_registry.json"
DELETION_QUEUE    = ROOT / "logs" / "deletion_queue.json"
SAIDA_INTRO_FILE  = ROOT / "reports" / "debt_stop_saida_intro_sent.json"
PAYMENT_HOLDS_FILE = ROOT / "logs" / "saida_payment_holds.json"

# Саида — бухгалтер-оператор
SAIDA_CHAT_ID = int(os.getenv("SAIDA_CHAT_ID", "920236287"))

# Автоудаление сообщений через 24 часа (как у всего бота)
DELETE_AFTER_HOURS = 24

# Пороги (дней молчания)
OVERDUE_MIN   = 10   # 10+ дней: запрос менеджеру
SILENCE_MIN   = 10   # 10+ дней: запрос менеджеру
AUTO_STOP_MIN = 15   # 15+ дней: авто-стоп даже для одобренных

# Минимальная сумма долга — игнорировать мелочь
MIN_DEBT = 50_000.0
STOP_PAID_THRESHOLD = float(os.getenv("STOP_PAID_THRESHOLD", "5000"))

# SLA для подтверждения оплаты Саидой
SAIDA_WARN_HOURS   = int(os.getenv("SAIDA_WARN_HOURS",   "1"))   # первое предупреждение
SAIDA_BYPASS_HOURS = int(os.getenv("SAIDA_BYPASS_HOURS", "2"))   # байпас к директору
SAIDA_STALE_TTL_HOURS = int(os.getenv("SAIDA_STALE_TTL_HOURS", "12"))  # тихо протухает без эскалации

SAIDA_KNOWN_FILE = ROOT / "logs" / "debt_stop_saida_known.json"


# ══════════════════════════════════════════════════════════════════════
# Атомарная работа с JSON
# ══════════════════════════════════════════════════════════════════════

def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".tmp",
        dir=path.parent, delete=False,
    ) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        tmp_path = Path(f.name)
    tmp_path.replace(path)


def _load_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default if default is not None else {}


def _schedule_delete(chat_id: int, message_id: int, msg_ts: float,
                     hours: int = DELETE_AFTER_HOURS) -> None:
    """
    Планирует удаление сообщения через `hours` часов.
    Пишет в тот же deletion_queue.json что и основной бот (janitor подберёт).
    Telegram API позволяет удалять сообщения не старше 48 часов.
    """
    import time as _time
    try:
        due_ts = msg_ts + hours * 3600
        queue  = _load_json(DELETION_QUEUE, {"jobs": []})
        jobs   = queue.get("jobs", [])
        # дедупликация
        for j in jobs:
            if j.get("chat_id") == chat_id and j.get("message_id") == message_id:
                j["due_ts"] = due_ts
                break
        else:
            jobs.append({
                "chat_id":      chat_id,
                "message_id":   message_id,
                "due_ts":       due_ts,
                "msg_ts":       msg_ts,
                "scheduled_at": _time.time(),
            })
        queue["jobs"] = jobs[-5000:]
        _save_json(DELETION_QUEUE, queue)
    except Exception as e:
        LOG.warning("Ошибка планирования удаления msg=%s: %s", message_id, e)


# ══════════════════════════════════════════════════════════════════════
# Суточное состояние
# ══════════════════════════════════════════════════════════════════════

def _kb_saida_help_only() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("❓ Что это значит?", callback_data="dstop_saida_help")
    ]])


def _kb_saida_stop_item(name_key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💰 Оплата получена", callback_data=f"dstop_paid|{name_key}")],
        [InlineKeyboardButton("❓ Что это значит?", callback_data="dstop_saida_help")],
    ])


def load_state() -> Dict[str, Any]:
    state = _load_json(STATE_FILE, {})
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    if state.get("date") != today:
        state = {"date": today, "candidates": {}, "next_id": 1, "saida_sent": False}
    if _sanitize_state_candidates(state):
        save_state(state)
    return state


def save_state(state: Dict[str, Any]) -> None:
    _save_json(STATE_FILE, state)


# ══════════════════════════════════════════════════════════════════════
# Постоянный реестр: одобрения, нарушители, авто-стопы
# ══════════════════════════════════════════════════════════════════════
# Структура записи:
# {
#   "client_name": {
#     "manager": "Алена",
#     "manager_chat_id": 188939016,
#     "approved_at": "2026-03-31",      # дата согласования менеджером
#     "days_at_approval": 9,
#     "debt_at_approval": 1500000,
#     "status": "exception",            # exception | auto_stopped | cleared
#     "auto_stopped_at": null,
#     "days_at_stop": null,
#     "discipline_violation": false,    # true = следующий раз только через руководителя
#     "cleared_at": null
#   }
# }

def load_registry() -> Dict[str, Any]:
    return _load_json(REGISTRY_FILE, {})


def save_registry(reg: Dict[str, Any]) -> None:
    _save_json(REGISTRY_FILE, reg)


# ══════════════════════════════════════════════════════════════════════
# Вспомогательные загрузчики конфигов
# ══════════════════════════════════════════════════════════════════════

def _load_weekly_clients() -> set:
    data = _load_json(CONFIG_DIR / "weekly_clients.json", {})
    return set(data.get("clients", []))


def _load_payment_terms() -> Dict[str, int]:
    """Возвращает {client_name: payment_terms_days} из clients.json."""
    data = _load_json(CONFIG_DIR / "clients.json", {})
    result: Dict[str, int] = {}
    for name, info in data.get("clients", {}).items():
        days = info.get("payment_terms_days", 0)
        if days and int(days) > 0:
            result[name] = int(days)
    return result


def _load_managers() -> Dict[str, int]:
    return {k: int(v) for k, v in _load_json(CONFIG_DIR / "managers.json", {}).items()}


def _get_admin_chat_id() -> int:
    return int(os.getenv("ADMIN_CHAT_ID", "0"))


def _sanitize_state_candidates(state: Dict[str, Any]) -> bool:
    """Удаляет из debt_stop state кандидатов с невалидным manager_chat_id.

    Прод-контур не должен обрабатывать тестовые записи вида manager_chat_id=111
    или кандидатов, где manager→chat_id не совпадает с managers.json.
    """
    candidates = state.get("candidates", {})
    if not isinstance(candidates, dict) or not candidates:
        return False
    managers = _load_managers()
    changed = False
    for cid, rec in list(candidates.items()):
        if not isinstance(rec, dict):
            candidates.pop(cid, None)
            changed = True
            continue
        manager = str(rec.get("manager") or "").strip()
        expected_chat = int(managers.get(manager, 0) or 0)
        actual_chat = int(rec.get("manager_chat_id") or 0)
        if not manager or not expected_chat or actual_chat != expected_chat:
            LOG.warning(
                "Skipping polluted debt_stop candidate cid=%s client=%s manager=%s actual_chat=%s expected_chat=%s",
                cid,
                rec.get("client"),
                manager,
                actual_chat,
                expected_chat,
            )
            candidates.pop(cid, None)
            changed = True
    return changed


# ══════════════════════════════════════════════════════════════════════
# Загрузка дебиторки из JSON
# ══════════════════════════════════════════════════════════════════════

def _get_latest_debt_file(manager: str) -> Optional[Path]:
    """
    Найти актуальный debt_ext файл для менеджера.

    В проекте сосуществуют два семейства manager-specific debt JSON:
    - свежие `debt_ext_Детальный Дебиторы <manager> (...)`
    - старые `debt_ext_Ведомость_по_взаиморасчетам_с_контрагентами_<manager> (...)`

    Номера в скобках несравнимы между этими семействами, поэтому выбор по
    `(... )` может отдавать более старую ведомость вместо свежего detailed debt.
    Для стоп-листа источником истины должен быть самый свежий файл по времени,
    с приоритетом для линии `Детальный Дебиторы`, если она существует.
    """
    detailed = list(JSON_DIR.glob(f"debt_ext_*Детальный Дебиторы {manager}*"))
    if detailed:
        return max(detailed, key=lambda p: p.stat().st_mtime)

    files = list(JSON_DIR.glob(f"debt_ext_*{manager}*"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _get_client_current_state(client_name: str) -> Optional[Dict[str, Any]]:
    """Найти текущий days_silence и debt клиента во всех менеджерских файлах."""
    managers = _load_managers()
    for manager_name in managers:
        path = _get_latest_debt_file(manager_name)
        if not path:
            continue
        data = _load_json(path, {})
        for c in data.get("clients", []):
            if c.get("client", "").strip() == client_name:
                return {
                    "days_silence": int(c.get("days_silence", 0) or 0),
                    "debt": float(c.get("debt", 0) or 0),
                    "manager": manager_name,
                }
    return None


# ══════════════════════════════════════════════════════════════════════
# Форматирование
# ══════════════════════════════════════════════════════════════════════

def _fmt(amount: float) -> str:
    return f"{amount:,.0f}".replace(",", "\u202f") + "\u202f₸"


def _parse_amount(text: str) -> Optional[float]:
    raw = str(text or "").lower().replace("тг", "").replace("тенге", "")
    raw = raw.replace("\u202f", "").replace(" ", "").replace(",", ".")
    raw = re.sub(r"[^0-9.]", "", raw)
    if not raw:
        return None
    try:
        amount = float(raw)
    except ValueError:
        return None
    return amount if amount > 0 else None


# ══════════════════════════════════════════════════════════════════════
# 14:00 — мониторинг реестра: авто-стоп при 15+ днях
# ══════════════════════════════════════════════════════════════════════

async def monitor_exceptions(bot) -> None:
    """
    Ежедневно в 14:00.
    Проверяет клиентов из реестра (статус 'exception'):
    - Если молчание >= 15 дней → переводит в авто-стоп.
    - Уведомляет менеджера и руководителя.
    - Помечает как нарушитель финансовой дисциплины.
    Также проверяет авто-стопы: если долг = 0 → уведомляет руководителя о полной оплате.
    """
    registry = load_registry()
    admin_id = _get_admin_chat_id()
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    auto_stopped_today: List[str] = []
    paid_in_full_today: List[str] = []

    for client_name, rec in registry.items():
        status = rec.get("status", "exception")

        # ── Проверка авто-стопа (15+ дней) ──────────────────────────
        if status == "exception":
            current = _get_client_current_state(client_name)
            if not current:
                continue
            days = current["days_silence"]
            debt = current["debt"]

            if days >= AUTO_STOP_MIN:
                if debt < 2_000:
                    LOG.info("Пропуск авто-стопа %s — долг копеечный (%.2f ₸)", client_name, debt)
                    continue
                rec["status"] = "auto_stopped"
                rec["auto_stopped_at"] = today
                rec["days_at_stop"] = days
                rec["discipline_violation"] = True
                rec["added_by"] = "auto"
                auto_stopped_today.append(client_name)
                LOG.warning("Авто-стоп: %s (%d дн.)", client_name, days)

                mgr_id = rec.get("manager_chat_id", 0)
                msg = (
                    f"🚫 <b>Авто-стоп: {client_name}</b>\n"
                    f"Молчит <b>{days} дн.</b> — договорённость нарушена.\n"
                    f"Долг: <b>{_fmt(debt)}</b>\n"
                    f"Клиент заблокирован для отгрузки до полной оплаты.\n"
                    f"Разблокировка — только через руководителя."
                )
                for chat_id in {mgr_id, admin_id, SAIDA_CHAT_ID}:
                    if chat_id:
                        try:
                            await bot.send_message(
                                chat_id=chat_id,
                                text=msg,
                                parse_mode="HTML",
                                reply_markup=_kb_saida_help_only() if chat_id == SAIDA_CHAT_ID else None,
                            )
                        except Exception as e:
                            LOG.warning("Ошибка уведомления авто-стоп %s (chat=%s): %s",
                                        client_name, chat_id, e)

        # ── Условная отгрузка: долг оплачен → авто-снятие ────────────
        elif status in ("conditional", "allow_after_payment"):
            current = _get_client_current_state(client_name)
            if current and current["debt"] <= STOP_PAID_THRESHOLD:
                limit = float(rec.get("shipment_limit") or 0)
                rec["status"] = "cleared_limited" if limit > 0 else "cleared"
                rec["cleared_at"] = today
                LOG.info("Условие полной оплаты выполнено: %s", client_name)
                mgr_id = rec.get("manager_chat_id", 0)
                if limit > 0:
                    note = (
                        f"✅ <b>{client_name}</b> — старая задолженность закрыта.\n"
                        f"Остаток в 1С: <b>{_fmt(current['debt'])}</b>\n"
                        f"Отгрузка разрешена с новым лимитом: <b>{_fmt(limit)}</b>.\n"
                        f"Больше лимита не отгружать без отдельного разрешения руководителя."
                    )
                else:
                    note = (
                        f"✅ <b>{client_name}</b> — полная оплата видна в 1С.\n"
                        f"Остаток: <b>{_fmt(current['debt'])}</b>\n"
                        f"Клиент снят с контроля. Отгрузка разрешена."
                    )
                for cid_tg in {admin_id, mgr_id, SAIDA_CHAT_ID}:
                    if cid_tg:
                        try:
                            await bot.send_message(
                                chat_id=cid_tg,
                                text=note,
                                parse_mode="HTML",
                                reply_markup=_kb_saida_help_only() if cid_tg == SAIDA_CHAT_ID else None,
                            )
                        except Exception as e:
                            LOG.warning("Ошибка уведомления conditional-cleared %s: %s",
                                        client_name, e)

        # ── Лимит cleared_limited: срок истёк → переводим на рассмотрение ──
        elif status == "cleared_limited":
            limit_expires = rec.get("limit_expires")
            if limit_expires and today >= limit_expires:
                LOG.info("Лимит отгрузки истёк: %s (expires %s)", client_name, limit_expires)
                rec["status"] = "pending_clearance_mgr"
                rec.pop("limit_expires", None)
                paid_in_full_today.append(client_name)
                await _notify_manager_clearance_proposal(client_name, rec, bot)

        # ── Проверка полной оплаты (авто-стоп или утверждённый стоп) ──
        elif status in ("auto_stopped", "stopped", "block_until_payment"):
            current = _get_client_current_state(client_name)
            if not current:
                continue
            if current["debt"] <= STOP_PAID_THRESHOLD:
                rec["status"] = "pending_clearance_mgr"
                paid_in_full_today.append(client_name)
                LOG.info("Оплата обнаружена: %s → уведомление менеджеру", client_name)
                await _notify_manager_clearance_proposal(client_name, rec, bot)

    save_registry(registry)

    if auto_stopped_today:
        LOG.info("Авто-стоп сегодня: %d клиентов: %s", len(auto_stopped_today), auto_stopped_today)
    if paid_in_full_today:
        LOG.info("Полная оплата сегодня: %d клиентов: %s", len(paid_in_full_today), paid_in_full_today)


# ══════════════════════════════════════════════════════════════════════
# 17:00 — сборка кандидатов и запрос менеджерам
# ══════════════════════════════════════════════════════════════════════

def _build_candidates() -> Dict[str, Any]:
    """
    Собирает кандидатов из debt_ext файлов.
    Фильтрует: days_silence < 7, сумма < MIN_DEBT, еженедельные клиенты,
    договорная отсрочка, клиенты уже в реестре (exception / auto_stopped).
    """
    state = load_state()
    if state.get("candidates"):  # уже собраны сегодня
        return state

    weekly   = _load_weekly_clients()
    terms    = _load_payment_terms()
    managers = _load_managers()
    registry = load_registry()

    # Клиенты уже под контролем реестра — не дублировать
    already_controlled = {
        name for name, rec in registry.items()
        if rec.get("status") in (
            "exception", "auto_stopped", "stopped", "conditional",
            "allow_after_payment", "block_until_payment",
            "pending_clearance", "pending_clearance_mgr", "pending_clearance_admin",
            "awaiting_clearance_limit", "awaiting_mgr_limit_input", "awaiting_admin_limit_override",
            "prepayment_only", "blacklisted", "cleared_limited",
        )
    }

    # Нарушители дисциплины (были авто-остановлены ранее, но уже cleared)
    discipline_violators = {
        name for name, rec in registry.items()
        if rec.get("discipline_violation") and rec.get("status") == "cleared"
    }

    next_id = 1
    candidates: Dict[str, Any] = {}

    for manager_name, chat_id in managers.items():
        debt_file = _get_latest_debt_file(manager_name)
        if not debt_file:
            LOG.warning("Нет debt_ext файла для %s", manager_name)
            continue

        data = _load_json(debt_file, {})
        for client in data.get("clients", []):
            name  = client.get("client", "").strip()
            days  = int(client.get("days_silence", 0) or 0)
            debt  = float(client.get("debt", 0) or 0)

            if days < OVERDUE_MIN:
                continue
            if debt < MIN_DEBT:
                continue
            if name in weekly:
                LOG.debug("Исключён (еженедельный): %s", name)
                continue
            if name in already_controlled:
                LOG.debug("Уже в реестре: %s", name)
                continue

            contract_days = terms.get(name, 0)
            if contract_days > 0 and days < contract_days:
                LOG.debug("В рамках договора (%d дн.): %s", contract_days, name)
                continue

            # Нарушитель дисциплины — пропускает шаг менеджера
            skip_manager = name in discipline_violators

            level = "10+"
            cid   = str(next_id)
            next_id += 1
            candidates[cid] = {
                "client":           name,
                "manager":          manager_name,
                "manager_chat_id":  chat_id,
                "days_silence":     days,
                "debt":             debt,
                "level":            level,
                "skip_manager":     skip_manager,
                "manager_msg_id":   None,
                "manager_response": None,
                "response_at":      None,
                "escalated":        False,
                "admin_msg_id":     None,
                "admin_approved":   None,
            }

    state["candidates"] = candidates
    state["next_id"] = next_id
    save_state(state)
    LOG.info("Кандидатов на стоп: %d (нарушители дисц.: %d)", len(candidates),
             sum(1 for c in candidates.values() if c.get("skip_manager")))
    return state


async def send_manager_requests(bot) -> None:
    """17:00 — запросить у каждого менеджера подтверждение по его клиентам."""
    state = _build_candidates()
    candidates = state.get("candidates", {})

    # Нарушители дисциплины сразу к руководителю — менеджеру не отправляем
    to_managers: Dict[str, List[Tuple[str, Any]]] = {}
    for cid, c in candidates.items():
        if c.get("skip_manager") or c.get("manager_response") is not None:
            continue
        mgr = c["manager"]
        to_managers.setdefault(mgr, []).append((cid, c))

    if not to_managers:
        LOG.info("Нет кандидатов для запроса менеджерам")
        return

    for manager_name, items in to_managers.items():
        chat_id = items[0][1]["manager_chat_id"]
        header = (
            f"⚠️ <b>Контроль отгрузки — {manager_name}</b>\n"
            f"У каждого клиента нет оплаты {OVERDUE_MIN}+ дней.\n\n"
            f"✅ <i>Договорились</i> — исключить из стоп-листа сегодня\n"
            f"🚫 <i>Нет, стоп</i> — передать руководителю\n\n"
            f"Если не ответишь с первого раза — бот будет напоминать каждые 30 минут "
            f"и усиливать тон.\n"
            f"Я вижу игнор. Каждый день без ответа фиксируется — "
            f"руководитель получит рекомендацию задержать зарплату на столько же дней.\n"
            f"Если не ответишь до 19:00 — передаётся автоматически."
        )
        try:
            await bot.send_message(chat_id=chat_id, text=header, parse_mode="HTML")
        except Exception as e:
            LOG.warning("Ошибка отправки заголовка менеджеру %s: %s", manager_name, e)

        for cid, c in sorted(items, key=lambda x: -x[1]["days_silence"]):
            icon = "🔴" if c["level"] == "10+" else "⚡"
            text = (
                f"{icon} <b>{c['client']}</b>\n"
                f"Молчит: <b>{c['days_silence']}\u202fдн.</b>  |  "
                f"Долг: <b>{_fmt(c['debt'])}</b>\n\n"
                f"<i>Я вижу игнор. Каждый день без ответа фиксируется — "
                f"руководитель получит рекомендацию задержать зарплату на столько же дней. "
                f"Напоминания идут каждые 30 минут.</i>"
            )
            kb = InlineKeyboardMarkup([[
                InlineKeyboardButton("✅ Договорились", callback_data=f"dstop_yes|{cid}"),
                InlineKeyboardButton("🚫 Нет, стоп",    callback_data=f"dstop_no|{cid}"),
            ], [
                InlineKeyboardButton("❓ Не понимаю, что ответить", callback_data=f"dstop_help|{cid}"),
            ]])
            try:
                msg = await bot.send_message(chat_id=chat_id, text=text,
                                             parse_mode="HTML", reply_markup=kb)
                state["candidates"][cid]["manager_msg_id"] = msg.message_id
                _schedule_delete(chat_id, msg.message_id, msg.date.timestamp())
            except Exception as e:
                LOG.warning("Ошибка отправки клиента %s менеджеру %s: %s",
                            c["client"], manager_name, e)

    save_state(state)
    LOG.info("Запросы отправлены менеджерам: %s", list(to_managers.keys()))


# ══════════════════════════════════════════════════════════════════════
# 19:00 — эскалация нет-ответа → руководителю
# ══════════════════════════════════════════════════════════════════════

async def escalate_unanswered(bot) -> None:
    """
    19:00 — передать руководителю:
    - Тех, кому менеджер не ответил за 2 часа
    - Тех, по кому менеджер сказал «нет, стоп»
    - Нарушителей дисциплины (skip_manager=True)
    """
    state = load_state()
    candidates = state.get("candidates", {})
    admin_id = _get_admin_chat_id()

    if not admin_id:
        LOG.error("ADMIN_CHAT_ID не задан — эскалация невозможна")
        return

    pending = [
        (cid, c) for cid, c in candidates.items()
        if c.get("admin_msg_id") is None and c.get("admin_approved") is None
        and (
            c.get("manager_response") in (None, "no")
            or c.get("skip_manager")
        )
    ]

    if not pending:
        LOG.info("Нет кандидатов для эскалации к руководителю")
        return

    header = (
        f"🛑 <b>Стоп-лист — утверждение ({len(pending)} кл.)</b>\n"
        f"Менеджеры не ответили или подтвердили стоп.\n"
        f"Нажми кнопку по каждому клиенту:"
    )
    try:
        await bot.send_message(chat_id=admin_id, text=header, parse_mode="HTML")
    except Exception as e:
        LOG.warning("Ошибка отправки заголовка руководителю: %s", e)

    for cid, c in sorted(pending, key=lambda x: -x[1]["days_silence"]):
        icon = "🔴" if c["level"] == "10+" else "⚡"

        if c.get("skip_manager"):
            note = "⚠️ Нарушитель фин. дисциплины — без согласования менеджера"
        elif c.get("manager_response") == "no":
            note = f"Менеджер {c['manager']}: нет договорённости"
        elif c.get("manager_response") == "yes":
            note = f"Менеджер {c['manager']}: договорились ✅"
        else:
            note = f"Менеджер {c['manager']} не ответил"

        text = (
            f"{icon} <b>{c['client']}</b>  [{c['manager']}]\n"
            f"Молчит: <b>{c['days_silence']}\u202fдн.</b>  |  "
            f"Долг: <b>{_fmt(c['debt'])}</b>\n"
            f"<i>{note}</i>"
        )
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Разрешить сейчас", callback_data=f"dstop_admin_remove|{cid}"),
            InlineKeyboardButton("📉 После оплаты с лимитом", callback_data=f"dstop_admin_limit_after|{cid}"),
        ], [
            InlineKeyboardButton("🔒 Запретить до оплаты", callback_data=f"dstop_admin_block_until|{cid}"),
            InlineKeyboardButton("🚫 Запретить", callback_data=f"dstop_admin_ok|{cid}"),
        ]])
        try:
            msg = await bot.send_message(chat_id=admin_id, text=text,
                                         parse_mode="HTML", reply_markup=kb)
            state["candidates"][cid]["escalated"]    = True
            state["candidates"][cid]["admin_msg_id"] = msg.message_id
            _schedule_delete(admin_id, msg.message_id, msg.date.timestamp())
        except Exception as e:
            LOG.warning("Ошибка эскалации %s: %s", c["client"], e)

    save_state(state)
    LOG.info("Эскалировано к руководителю: %d", len(pending))


async def send_manager_reminders(bot) -> None:
    """Напоминает менеджерам о незакрытых запросах стоп-листа по нарастающей."""
    state = load_state()
    candidates = state.get("candidates", {})
    now = datetime.now(TZ)
    if not (9 <= now.hour < 23):
        return

    changed = False
    for cid, c in candidates.items():
        if c.get("skip_manager"):
            continue
        if c.get("manager_response") is not None:
            continue
        if c.get("admin_approved") is not None:
            continue
        chat_id = c.get("manager_chat_id")
        if not chat_id:
            continue

        last_raw = c.get("manager_last_reminded") or c.get("response_at")
        if last_raw:
            try:
                last_dt = datetime.fromisoformat(str(last_raw))
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=TZ)
                if now - last_dt < timedelta(minutes=30):
                    continue
            except (TypeError, ValueError):
                pass

        count = int(c.get("manager_remind_count", 0) or 0) + 1
        c["manager_remind_count"] = count
        c["manager_last_reminded"] = now.isoformat()
        changed = True

        if count == 1:
            header = "⏰ <b>Напоминание</b>"
            tone = "Нужно выбрать действие по клиенту."
        elif count == 2:
            header = "⚠️ <b>Повторное напоминание</b>"
            tone = "Запрос всё ещё не закрыт. Ответ обязателен."
        elif count == 3:
            header = "🚨 <b>Последнее предупреждение менеджеру</b>"
            tone = "Если не ответите, вопрос будет у руководителя как неотработанный."
        else:
            header = f"🔥 <b>Просроченный запрос #{count}</b>"
            tone = "Кейс висит без ответа. Требуется действие сейчас."

        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Договорились", callback_data=f"dstop_yes|{cid}"),
            InlineKeyboardButton("🚫 Нет, стоп", callback_data=f"dstop_no|{cid}"),
        ], [
            InlineKeyboardButton("❓ Не понимаю, что ответить", callback_data=f"dstop_help|{cid}"),
        ]])
        text = (
            f"{header}\n\n"
            f"<b>{c['client']}</b>\n"
            f"Молчит: <b>{c['days_silence']}\u202fдн.</b>  |  "
            f"Долг: <b>{_fmt(c['debt'])}</b>\n\n"
            f"{tone}\n"
            f"Пока вы не ответите, запрос остаётся активным.\n"
            f"Я вижу игнор. Каждый день без ответа фиксируется — "
            f"руководитель получит рекомендацию задержать зарплату на столько же дней."
        )
        try:
            msg = await bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode="HTML",
                reply_markup=kb,
            )
            _schedule_delete(chat_id, msg.message_id, msg.date.timestamp())
        except Exception as e:
            LOG.warning("Ошибка напоминания менеджеру %s по %s: %s", c.get("manager"), c.get("client"), e)

        admin_id = _get_admin_chat_id()
        if admin_id and count >= 3 and int(c.get("manager_admin_notice_count", 0) or 0) < count:
            c["manager_admin_notice_count"] = count
            manager_name = c.get("manager")
            mgr_open = [
                item for item in candidates.values()
                if item.get("manager") == manager_name
                and item.get("manager_response") is None
                and item.get("admin_approved") is None
                and not item.get("skip_manager")
            ]
            mgr_total_reminders = sum(int(item.get("manager_remind_count", 0) or 0) for item in mgr_open)
            mgr_max_reminders = max(
                [int(item.get("manager_remind_count", 0) or 0) for item in mgr_open] or [0]
            )
            ignored_lines = []
            for item in sorted(
                mgr_open,
                key=lambda x: int(x.get("manager_remind_count", 0) or 0),
                reverse=True,
            )[:10]:
                ignored_lines.append(
                    f"• {item.get('client')} — "
                    f"{int(item.get('manager_remind_count', 0) or 0)} напомин."
                )
            ignored_details = "\n".join(ignored_lines) or "Нет открытых запросов"
            try:
                await bot.send_message(
                    chat_id=admin_id,
                    text=(
                        f"🚨 <b>Менеджер не выполняет запрос стоп-листа</b>\n\n"
                        f"Менеджер: <b>{manager_name}</b>\n"
                        f"Клиент: <b>{c.get('client')}</b>\n"
                        f"Долг: <b>{_fmt(c.get('debt', 0))}</b>\n"
                        f"Молчит: <b>{c.get('days_silence')}\u202fдн.</b>\n\n"
                        f"Напоминаний менеджеру уже: <b>{count}</b>.\n"
                        f"Ответа нет.\n\n"
                        f"📊 <b>Статистика игнора по менеджеру</b>\n"
                        f"Открытых запросов: <b>{len(mgr_open)}</b>\n"
                        f"Всего напоминаний: <b>{mgr_total_reminders}</b>\n"
                        f"Максимум по одному клиенту: <b>{mgr_max_reminders}</b>\n\n"
                        f"<b>По каждому открытому запросу:</b>\n"
                        f"{ignored_details}"
                    ),
                    parse_mode="HTML",
                )
            except Exception as e:
                LOG.warning("Ошибка уведомления руководителя о молчании менеджера %s: %s", c.get("manager"), e)

    if changed:
        save_state(state)


# ══════════════════════════════════════════════════════════════════════
# 22:00 — финальный список Саиде
# ══════════════════════════════════════════════════════════════════════

async def _send_saida_intro(bot) -> None:
    """
    Отправляет Саиде инструкцию один раз — перед первым стоп-листом.
    Повторно не отправляется (флаг в SAIDA_INTRO_FILE).
    """
    intro_data = _load_json(SAIDA_INTRO_FILE, {})
    if intro_data.get("sent"):
        return

    instruction = (
        "👋 <b>Саида, добрый вечер!</b>\n\n"
        "Я буду присылать тебе запросы на проверку оплаты и список клиентов, "
        "которых <b>нельзя отгружать</b>.\n\n"

        "📋 <b>Запросы на проверку оплаты</b>\n"
        "Когда менеджер говорит что клиент оплатил — ты получаешь запрос с тремя кнопками:\n\n"
        "✅ <b>Да, оплата есть</b> — деньги пришли полностью.\n"
        "   → Клиент снимается со стопа <b>автоматически</b>. Менеджер и директор узнают об этом.\n\n"
        "🔸 <b>Частично</b> — пришла не вся сумма.\n"
        "   → Директор получит уведомление и решит сам.\n\n"
        "❌ <b>Не вижу оплаты</b> — ничего не поступало.\n"
        "   → Клиент остаётся в стопе.\n\n"

        "⏰ <b>Отвечай в течение 4 часов.</b>\n"
        "Через 4ч — придёт напоминание.\n"
        "Через 8ч молчания — уведомления <b>уйдут клиентам автоматически</b>, "
        "а менеджеры и директор узнают что ты не ответила.\n\n"

        "📋 <b>Вечерний стоп-лист (22:00)</b>\n"
        "Каждый вечер получаешь список кого нельзя отгружать завтра.\n"
        "Клиент исчезает из списка сам, как только оплата разносится в 1С.\n\n"

        "❓ Если что-то непонятно — нажми кнопку <b>«Не понимаю»</b> под запросом."
    )

    try:
        msg = await bot.send_message(
            chat_id=SAIDA_CHAT_ID,
            text=instruction,
            parse_mode="HTML"
        )
        _schedule_delete(SAIDA_CHAT_ID, msg.message_id, msg.date.timestamp(),
                         hours=72)  # инструкцию храним 3 дня
        _save_json(SAIDA_INTRO_FILE, {"sent": True, "sent_at": datetime.now(TZ).isoformat()})
        LOG.info("Инструкция Саиде отправлена")
    except Exception as e:
        LOG.warning("Ошибка отправки инструкции Саиде: %s", e)


def _load_saida_known() -> set:
    data = _load_json(SAIDA_KNOWN_FILE, {"known": []})
    return set(data.get("known", []))


def _save_saida_known(known: set) -> None:
    _save_json(SAIDA_KNOWN_FILE, {"known": list(known)})


async def send_saida_final(bot) -> None:
    """22:00 — отправить Саиде ДЕЛЬТУ стоп-листа (только новые клиенты)."""
    state = load_state()
    if state.get("saida_sent"):
        return

    await _send_saida_intro(bot)

    candidates = state.get("candidates", {})
    registry   = load_registry()
    today_str  = datetime.now(TZ).strftime("%d.%m.%Y")
    known      = _load_saida_known()

    stop_statuses = ("auto_stopped", "stopped", "allow_after_payment",
                     "block_until_payment", "prepayment_only", "blacklisted")

    # Из суточного состояния: утверждённые руководителем, только новые
    approved_daily_new = [
        (cid, c) for cid, c in candidates.items()
        if c.get("admin_approved") is True and c.get("client") not in known
    ]

    # Из реестра: авто-стопы и стопы — только НОВЫЕ (Саида ещё не знает)
    auto_stopped_new = [
        (name, rec) for name, rec in registry.items()
        if rec.get("status") in stop_statuses and name not in known
    ]

    total_on_stop = sum(1 for r in registry.values() if r.get("status") in stop_statuses)
    total_new     = len(approved_daily_new) + len(auto_stopped_new)
    already_known_count = total_on_stop - len(auto_stopped_new)

    kb_full = InlineKeyboardMarkup([
        [InlineKeyboardButton("📋 Свежий стоп-лист", callback_data="dstop_saida_stoplist")],
        [InlineKeyboardButton("❓ Инструкция / Не понимаю", callback_data="dstop_saida_help")],
    ])

    if total_new == 0:
        known_info = f"\nВ стоп-листе: {already_known_count} клиентов." if already_known_count > 0 else "\nВсе клиенты в порядке."
        try:
            msg = await bot.send_message(
                chat_id=SAIDA_CHAT_ID,
                text=f"✅ <b>Новых клиентов в стоп-листе нет — {today_str}</b>{known_info}",
                parse_mode="HTML",
                reply_markup=kb_full,
            )
            _schedule_delete(SAIDA_CHAT_ID, msg.message_id, msg.date.timestamp())
        except Exception as e:
            LOG.warning("Ошибка отправки Саиде (нет новых): %s", e)
        state["saida_sent"] = True
        save_state(state)
        return

    header_parts = [f"🚫 <b>Новые в стоп-листе — {today_str}</b>  ({total_new} новых)"]
    if already_known_count > 0:
        header_parts.append(f"<i>Уже в списке (без изменений): {already_known_count}</i>")
    try:
        hdr_msg = await bot.send_message(
            chat_id=SAIDA_CHAT_ID,
            text="\n".join(header_parts),
            parse_mode="HTML",
            reply_markup=kb_full,
        )
        _schedule_delete(SAIDA_CHAT_ID, hdr_msg.message_id, hdr_msg.date.timestamp())
    except Exception as e:
        LOG.warning("Ошибка отправки заголовка Саиде: %s", e)

    all_items: List[Tuple[str, int, str, str, str, float]] = []
    for name, rec in sorted(auto_stopped_new, key=lambda x: x[1].get("days_at_stop", 0), reverse=True):
        status = rec.get("status", "auto_stopped")
        if status == "blacklisted":
            level = "blacklisted"
        elif status == "prepayment_only":
            level = "prepayment_only"
        elif status in ("allow_after_payment",):
            level = "allow_after_payment"
        elif status == "block_until_payment":
            level = "block_until_payment"
        else:
            level = "auto"
        all_items.append((name, rec.get("days_at_stop", 0), "?", rec.get("manager", ""), level,
                          float(rec.get("shipment_limit") or 0)))
        known.add(name)

    for cid, c in sorted(approved_daily_new, key=lambda x: -x[1]["days_silence"]):
        all_items.append((c["client"], c["days_silence"], _fmt(c["debt"]), c["manager"], c["level"], 0.0))
        known.add(c["client"])

    for item in all_items:
        name, days, debt_str, manager, level, limit = item
        if level == "blacklisted":
            icon, note = "⛔", "ЧЁРНЫЙ СПИСОК — не отгружать"
        elif level == "prepayment_only":
            icon, note = "💳", "только 100% предоплата"
        elif level == "auto":
            icon, note = "🚫", "нарушение фин. дисциплины"
        elif level == "allow_after_payment":
            icon = "⏳"
            note = (f"после полной оплаты, лимит {_fmt(limit)}" if limit > 0 else "после полной оплаты")
        elif level == "block_until_payment":
            icon, note = "🔒", "запрет до полной оплаты"
        elif level == "10+":
            icon, note = "🔴", f"долг: {debt_str}"
        else:
            icon, note = "⚡", f"долг: {debt_str}"

        text = (
            f"{icon} <b>{name}</b>\n"
            f"Молчит {days}\u202fдн.  ·  {note}  [{manager}]\n"
            f"<i>Не отгружать до разрешения руководителя</i>"
        )
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("💰 Оплата получена", callback_data=f"dstop_paid|{name[:26]}")
        ]])
        kb = _kb_saida_stop_item(name[:26])
        try:
            item_msg = await bot.send_message(
                chat_id=SAIDA_CHAT_ID, text=text, parse_mode="HTML", reply_markup=kb
            )
            _schedule_delete(SAIDA_CHAT_ID, item_msg.message_id, item_msg.date.timestamp())
        except Exception as e:
            LOG.warning("Ошибка отправки позиции '%s' Саиде: %s", name, e)

    footer = "<i>Нажми «📋 Свежий стоп-лист» для полного актуального списка.</i>"
    try:
        ftr_msg = await bot.send_message(chat_id=SAIDA_CHAT_ID, text=footer, parse_mode="HTML")
        _schedule_delete(SAIDA_CHAT_ID, ftr_msg.message_id, ftr_msg.date.timestamp())
    except Exception as e:
        LOG.warning("Ошибка отправки футера Саиде: %s", e)

    # Предупреждение об уведомлениях без ответа
    try:
        holds_raw = _load_json(PAYMENT_HOLDS_FILE, {})
        now_dt    = datetime.now(TZ)
        unanswered = [
            v for v in holds_raw.values()
            if isinstance(v, dict) and v.get("status") == "pending_saida"
        ]
        if unanswered:
            oldest_days = 0
            for rec in unanswered:
                try:
                    created = datetime.fromisoformat(rec["created_at"])
                    d = (now_dt - created).days
                    if d > oldest_days:
                        oldest_days = d
                except Exception:
                    pass
            warn_text = (
                f"🚨 <b>Саида, я вижу игнор.</b>\n\n"
                f"У тебя {len(unanswered)} неотвеченных запроса на проверку оплаты.\n"
                f"Самый старый — уже {oldest_days} дн. без ответа.\n\n"
                f"Каждый день игнора фиксируется автоматически.\n"
                f"Руководитель получит рекомендацию задержать твою зарплату "
                f"на столько же дней, сколько ты тянешь с ответом.\n\n"
                f"Это касается и остальных: игнор виден всем — последствия те же."
            )
            warn_msg = await bot.send_message(
                chat_id=SAIDA_CHAT_ID,
                text=warn_text,
                parse_mode="HTML",
                reply_markup=_kb_saida_help_only(),
            )
            _schedule_delete(SAIDA_CHAT_ID, warn_msg.message_id, warn_msg.date.timestamp())
            LOG.info("Саиде предупреждение: %d неотвеченных запросов, макс. %d дн.",
                     len(unanswered), oldest_days)
    except Exception as e:
        LOG.warning("Ошибка предупреждения Саиде: %s", e)

    _save_saida_known(known)
    state["saida_sent"] = True
    save_state(state)
    LOG.info("Саиде дельта стоп-листа: %d новых позиций", total_new)


async def send_saida_payment_hold_reminders(bot) -> None:
    """SLA-контроль pending_saida: предупреждение → байпас директору.

    SAIDA_WARN_HOURS   (дефолт 1ч): Саиде — предупреждение с дедлайном.
    SAIDA_BYPASS_HOURS (дефолт 2ч): авто-закрытие холда как "нет подтверждения";
                                      директору  — INFO (без кнопок, решать нечего);
                                      менеджеру  — итог: клиент остаётся в дебиторке;
                                      Саиде      — сообщение о последствиях игнора.
    SAIDA_STALE_TTL_HOURS (дефолт 12ч): если бот добрался до холда слишком поздно,
                                     запись тихо закрывается как неактуальная без уведомлений.
    """
    from collector.payment_hold import _load, _save  # type: ignore
    now = datetime.now(TZ)
    if not (9 <= now.hour < 21):
        return

    try:
        admin_id = int(os.getenv("ADMIN_CHAT_ID", "0"))
    except (ValueError, TypeError):
        admin_id = 0

    data = _load()
    changed = False

    for token, rec in data.items():
        if not isinstance(rec, dict):
            continue
        if rec.get("status") != "pending_saida":
            continue

        try:
            created = datetime.fromisoformat(rec["created_at"])
            if created.tzinfo is None:
                created = created.replace(tzinfo=TZ)
        except (KeyError, ValueError):
            continue

        age_h = (now - created).total_seconds() / 3600
        client   = rec.get("client", "—")
        manager  = rec.get("manager", "—")
        debt_str = rec.get("debt_str") or rec.get("debt", "—")
        mgr_chat = rec.get("manager_chat_id")

        # Приоритетная цепочка — строго взаимоисключающая:
        # TTL → bypass → warning → ничего
        if age_h >= SAIDA_STALE_TTL_HOURS:
            # Слишком старый — не отражает актуальное состояние 1С.
            # Закрываем тихо, без уведомлений.
            rec["status"] = "expired"
            rec["expired_at"] = now.isoformat()
            rec["updated_at"] = now.isoformat()
            rec["expired_reason"] = "stale_ttl"
            data[token] = rec
            changed = True
            LOG.info(
                "Старый payment hold закрыт без уведомлений: client=%s manager=%s age_h=%.1f ttl_h=%s",
                client, manager, age_h, SAIDA_STALE_TTL_HOURS,
            )

        elif age_h >= SAIDA_BYPASS_HOURS and not rec.get("saida_escalated_at"):
            # Время вышло — авто-закрытие, уведомления директору/менеджеру/Саиде.
            from collector.payment_hold import confirm_by_saida as _confirm_saida
            _confirm_saida(token, "none")

            if admin_id:
                try:
                    await bot.send_message(
                        chat_id=admin_id,
                        text=(
                            f"ℹ️ Саида не ответила на запрос по <b>{client}</b> за <b>{age_h:.0f} ч</b>.\n"
                            f"Менеджер: <b>{manager}</b> | Долг: {debt_str}\n\n"
                            f"Холд закрыт автоматически — клиент остаётся в дебиторке."
                        ),
                        parse_mode="HTML",
                    )
                except Exception as e:
                    LOG.warning("send_saida bypass admin error: %s", e)

            if mgr_chat:
                try:
                    await bot.send_message(
                        chat_id=int(mgr_chat),
                        text=(
                            f"ℹ️ Саида не ответила на запрос по <b>{client}</b> за {age_h:.0f} ч.\n"
                            f"Клиент остаётся в дебиторке."
                        ),
                        parse_mode="HTML",
                    )
                except Exception as e:
                    LOG.warning("send_saida bypass mgr error: %s", e)

            try:
                await bot.send_message(
                    chat_id=SAIDA_CHAT_ID,
                    text=(
                        f"🚨 Саида, ты проигнорировала запрос по клиенту <b>{client}</b> — {age_h:.0f} ч без ответа.\n"
                        f"Холд закрыт автоматически. Все последствия — на тебе."
                    ),
                    parse_mode="HTML",
                )
            except Exception as e:
                LOG.warning("send_saida bypass saida msg error: %s", e)

            rec["saida_escalated_at"] = now.isoformat()
            changed = True
            LOG.info("Таймаут Саиды по %s (%.0fч) — авто-закрыт как rejected", client, age_h)

        elif age_h >= SAIDA_WARN_HOURS and not rec.get("saida_warned_at"):
            # Первое предупреждение — только если bypass ещё не наступил.
            remaining = max(0.0, SAIDA_BYPASS_HOURS - age_h)
            from telegram import InlineKeyboardMarkup, InlineKeyboardButton
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Полная оплата",  callback_data=f"payhold_full|{token}")],
                [InlineKeyboardButton("🔸 Частичная",     callback_data=f"payhold_partial|{token}")],
                [InlineKeyboardButton("❌ Оплаты нет",    callback_data=f"payhold_none|{token}")],
            ])
            text = (
                f"⚠️ Саида, ты не подтвердила оплату <b>{client}</b> уже <b>{age_h:.0f} ч</b>.\n\n"
                f"Через <b>{remaining:.0f} ч</b> решение уйдёт автоматически. "
                f"Менеджер <b>{manager}</b> и директор узнают о твоём молчании.\n\n"
                f"Все последствия ошибки — на тебе."
            )
            try:
                await bot.send_message(
                    chat_id=SAIDA_CHAT_ID, text=text,
                    parse_mode="HTML", reply_markup=kb,
                )
                rec["saida_warned_at"] = now.isoformat()
                changed = True
                LOG.info("Саиде предупреждение по %s (%.0fч)", client, age_h)
            except Exception as e:
                LOG.warning("send_saida warn error (%s): %s", client, e)

        # else: age < SAIDA_WARN_HOURS — ничего не делаем

    if changed:
        _save(data)


# ══════════════════════════════════════════════════════════════════════
# Обработчики callback-кнопок
# ══════════════════════════════════════════════════════════════════════

async def handle_dstop_callback(data: str, chat_id: int, bot) -> Optional[str]:
    """
    Точка входа для всех callback с префиксом 'dstop_'.
    Возвращает текст для edit_message_text или None если не обработано.
    """
    if data.startswith("dstop_yes|") or data.startswith("dstop_no|"):
        parts   = data.split("|", 1)
        cid     = parts[1]
        response = "yes" if data.startswith("dstop_yes|") else "no"
        return await _handle_manager_response(cid, response, chat_id, bot)

    if data.startswith("dstop_help|"):
        cid = data.split("|", 1)[1]
        await _send_manager_help(cid, chat_id, bot)
        return None

    if (
        data.startswith("dstop_admin_ok|")
        or data.startswith("dstop_admin_remove|")
        or data.startswith("dstop_admin_limit_after|")
        or data.startswith("dstop_admin_block_until|")
    ):
        parts  = data.split("|", 1)
        cid    = parts[1]
        if data.startswith("dstop_admin_ok|"):
            action = "ok"
        elif data.startswith("dstop_admin_remove|"):
            action = "remove"
        elif data.startswith("dstop_admin_limit_after|"):
            return await _handle_admin_limit_after_request(cid, chat_id, bot)
        else:
            action = "block_until"
        return await _handle_admin_response(cid, action, chat_id, bot)

    if (
        data.startswith("dstop_clear|")
        or data.startswith("dstop_keep|")
        or data.startswith("dstop_limit_clear|")
        or data.startswith("dstop_keep_until_paid|")
    ):
        parts       = data.split("|", 1)
        client_key  = parts[1]
        if data.startswith("dstop_limit_clear|"):
            return await _handle_clearance_limit_request(client_key, chat_id, bot)
        if data.startswith("dstop_clear|"):
            action = "clear"
        else:
            action = "keep"
        return await _handle_clearance(client_key, action, chat_id, bot)

    if data.startswith("dstop_conditional|"):
        client_key = data.split("|", 1)[1]
        return await _handle_conditional_clearance(client_key, chat_id, bot)

    if data.startswith("dstop_paid|"):
        client_key = data.split("|", 1)[1]
        return await _handle_saida_payment(client_key, chat_id, bot)

    if data.startswith("dstop_mgr_paid|"):
        cid = data.split("|", 1)[1]
        return await _handle_mgr_paid_claim(cid, chat_id, bot)

    if data.startswith("dstop_saida_full|"):
        cid = data.split("|", 1)[1]
        return await _handle_saida_confirm_full(cid, chat_id, bot)

    if data.startswith("dstop_saida_partial|"):
        cid = data.split("|", 1)[1]
        return await _handle_saida_confirm_partial(cid, chat_id, bot)

    if data.startswith("dstop_admin_allow|"):
        cid = data.split("|", 1)[1]
        return await _handle_admin_allow_after_saida(cid, chat_id, bot)

    # ── Новая цепочка: менеджер предлагает → руководитель утверждает ──
    if (
        data.startswith("dstop_mgr_cl_clear|")
        or data.startswith("dstop_mgr_cl_prepay|")
        or data.startswith("dstop_mgr_cl_limit|")
        or data.startswith("dstop_mgr_cl_keep|")
    ):
        key    = data.split("|", 1)[1]
        action = data.split("dstop_mgr_cl_")[1].split("|")[0]
        return await _handle_mgr_clearance_proposal(key, action, chat_id, bot)

    if data.startswith("dstop_adm_cl_confirm|"):
        key = data.split("|", 1)[1]
        return await _handle_admin_clearance_confirm(key, chat_id, bot)

    if (
        data.startswith("dstop_adm_cl_clear|")
        or data.startswith("dstop_adm_cl_prepay|")
        or data.startswith("dstop_adm_cl_blacklist|")
        or data.startswith("dstop_adm_cl_limit|")
    ):
        key    = data.split("|", 1)[1]
        action = data.split("dstop_adm_cl_")[1].split("|")[0]
        return await _handle_admin_clearance_override(key, action, chat_id, bot)

    if data == "dstop_saida_stoplist":
        await send_saida_full_stoplist_with_help(bot)
        return ""

    if data == "dstop_saida_help":
        await bot.send_message(
            chat_id=chat_id,
            parse_mode="HTML",
            text=(
                "❓ <b>Как работает стоп-лист</b>\n\n"
                "Каждый вечер я присылаю тебе список клиентов, которых <b>нельзя отгружать</b>.\n\n"
                "🚫 <b>Нарушение фин. дисциплины</b> — менеджер не работает с должником.\n"
                "🔒 <b>Запрет до оплаты</b> — клиент задолжал, нужна полная оплата.\n"
                "⏳ <b>После оплаты</b> — снимается автоматически как только 1С разнесёт платёж.\n"
                "💳 <b>Только предоплата</b> — отгружать можно, но только по факту оплаты.\n"
                "⛔ <b>Чёрный список</b> — никогда, ни при каких условиях.\n\n"
                "💰 <b>Кнопка «Оплата получена»</b> под клиентом:\n"
                "   Менеджер говорит что клиент оплатил → ты получаешь запрос →\n"
                "   «Да» — стоп снимается автоматически.\n"
                "   «Частично» — директор решает сам.\n"
                "   «Не вижу» — клиент остаётся в стопе.\n\n"
                "📋 <b>«Свежий стоп-лист»</b> — актуальный список прямо сейчас.\n\n"
                "⏰ <b>Важно:</b> Отвечай на запросы в течение 4 часов. "
                "Через 8ч молчания уведомления уйдут клиентам автоматически, "
                "а менеджеры и директор узнают что ты не ответила."
            ),
        )
        return None

    return None


async def _send_manager_help(cid: str, chat_id: int, bot) -> None:
    state = load_state()
    c = state.get("candidates", {}).get(cid)
    if not c:
        try:
            await bot.send_message(chat_id=chat_id, text="CRM/стоп-запрос не найден или уже устарел.")
        except Exception:
            pass
        return
    if int(c.get("manager_chat_id") or 0) != int(chat_id):
        return
    try:
        from collector.manager_help import build_manager_help
        help_text = await build_manager_help(
            area="Стоп-лист отгрузки",
            manager=c.get("manager", ""),
            client=c.get("client", ""),
            state="ожидается решение менеджера",
            buttons=["Договорились", "Нет, стоп"],
            context={
                "days_silence": c.get("days_silence"),
                "debt": c.get("debt"),
                "level": c.get("level"),
                "remind_count": c.get("manager_remind_count", 0),
            },
        )
    except Exception as e:
        LOG.warning("dstop manager help error: %s", e)
        help_text = (
            "Что от вас хотят:\n"
            "Нужно выбрать действие по клиенту.\n\n"
            "Что нажать:\n"
            "• Договорились — если есть понятная договорённость. Потом напишите дату, сумму и условия.\n"
            "• Нет, стоп — если договорённости нет или клиент тянет.\n\n"
            "Что будет если молчать:\n"
            "Бот будет напоминать каждые 30 минут, затем передаст руководителю. "
            "Я вижу игнор. Каждый день без ответа фиксируется — "
            "руководитель получит рекомендацию задержать зарплату на столько же дней."
        )
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=f"❓ <b>Подсказка по стоп-листу</b>\n\n{help_text}",
            parse_mode="HTML",
        )
    except Exception as e:
        LOG.warning("Ошибка отправки подсказки менеджеру %s: %s", c.get("manager"), e)


async def _handle_manager_response(cid: str, response: str, chat_id: int, bot) -> str:
    state = load_state()
    c = state["candidates"].get(cid)
    if not c:
        return "❓ Клиент не найден в сегодняшнем списке."
    if c.get("manager_response") is not None:
        return "ℹ️ Ответ уже зафиксирован."

    c["manager_response"] = response
    c["response_at"] = datetime.now(TZ).strftime("%H:%M")

    if response == "yes":
        # Запрашиваем детали договорённости — эскалация к руководителю после ответа
        c["awaiting_detail"] = True
        save_state(state)
        return (
            f"✅ <b>{c['client']}</b> — принято!\n\n"
            f"Уточни детали договорённости одним сообщением:\n"
            f"<i>(дата оплаты, сумма, условия — например: «оплата 15.04, 500к сейчас + 300к через неделю»)</i>"
        )
    else:
        result = f"🚫 Зафиксировано — <b>{c['client']}</b> передан руководителю."

    save_state(state)
    return result


async def _escalate_yes_to_admin(cid: str, c: dict, bot) -> None:
    """Эскалирует к руководителю после получения деталей от менеджера."""
    admin_id = _get_admin_chat_id()
    icon = "🔴" if c["level"] == "10+" else "⚡"
    note = c.get("manager_note", "").strip()
    detail_line = f"\n💬 <b>Детали:</b> {note}" if note else ""
    text = (
        f"{icon} <b>{c['client']}</b>  [{c['manager']}]\n"
        f"Молчит: <b>{c['days_silence']}\u202fдн.</b>  |  "
        f"Долг: <b>{_fmt(c['debt'])}</b>\n"
        f"<i>Менеджер {c['manager']}: договорились ✅</i>"
        f"{detail_line}"
    )
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Разрешить сейчас", callback_data=f"dstop_admin_remove|{cid}"),
        InlineKeyboardButton("📉 После оплаты с лимитом", callback_data=f"dstop_admin_limit_after|{cid}"),
    ], [
        InlineKeyboardButton("🔒 Запретить до оплаты", callback_data=f"dstop_admin_block_until|{cid}"),
        InlineKeyboardButton("🚫 Запретить", callback_data=f"dstop_admin_ok|{cid}"),
    ]])
    if admin_id:
        try:
            msg = await bot.send_message(chat_id=admin_id, text=text,
                                         parse_mode="HTML", reply_markup=kb)
            c["escalated"]    = True
            c["admin_msg_id"] = msg.message_id
            _schedule_delete(admin_id, msg.message_id, msg.date.timestamp())
        except Exception as e:
            LOG.warning("Ошибка эскалации yes-ответа %s: %s", c["client"], e)


async def handle_dstop_detail_message(chat_id: int, text: str, bot) -> bool:
    """Обрабатывает текстовый ответ менеджера с деталями договорённости.

    Вызывается из обработчика текстовых сообщений send_reports.py.
    Возвращает True если сообщение обработано (ждали детали).
    """
    state = load_state()
    candidates = state.get("candidates", {})

    admin_id = _get_admin_chat_id()
    registry = load_registry()

    # ── Ввод лимита менеджером (шаги: сумма → дни) ──
    for client_name, rec in registry.items():
        if rec.get("status") != "awaiting_mgr_limit_input":
            continue
        if int(rec.get("manager_chat_id") or 0) != int(chat_id):
            continue
        proposal = rec.setdefault("clearance_proposal", {})
        step = proposal.get("step", "amount")

        if step == "amount":
            amount = _parse_amount(text)
            if amount is None:
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text="Не понял сумму. Введи число, например: <code>100000</code>",
                        parse_mode="HTML",
                    )
                except Exception:
                    pass
                return True
            proposal["limit_amount"] = amount
            proposal["step"] = "days"
            rec["clearance_proposal"] = proposal
            save_registry(registry)
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"Лимит <b>{_fmt(amount)}</b> записан.\n\n"
                        f"Теперь введи срок (дней), на сколько действует лимит.\n"
                        f"Например: <code>7</code>"
                    ),
                    parse_mode="HTML",
                )
            except Exception:
                pass
            return True

        if step == "days":
            days_raw = re.sub(r"[^0-9]", "", text.strip())
            if not days_raw:
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text="Не понял срок. Введи число дней, например: <code>7</code>",
                        parse_mode="HTML",
                    )
                except Exception:
                    pass
                return True
            proposal["limit_days"] = int(days_raw)
            proposal["step"] = None
            rec["status"] = "pending_clearance_admin"
            rec["clearance_proposal"] = proposal
            save_registry(registry)
            await _forward_mgr_proposal_to_admin(client_name, rec, bot)
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"✅ Предложение передано руководителю:\n"
                        f"📉 Лимит {_fmt(proposal['limit_amount'])} на {proposal['limit_days']} дн."
                    ),
                    parse_mode="HTML",
                )
            except Exception:
                pass
            return True

    # ── Ввод лимита руководителем (override, шаги: сумма → дни) ──
    for client_name, rec in registry.items():
        if rec.get("status") != "awaiting_admin_limit_override":
            continue
        if chat_id != admin_id:
            continue
        proposal = rec.setdefault("clearance_proposal", {})
        step = proposal.get("override_step", "amount")

        if step == "amount":
            amount = _parse_amount(text)
            if amount is None:
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text="Не понял сумму. Введи число, например: <code>100000</code>",
                        parse_mode="HTML",
                    )
                except Exception:
                    pass
                return True
            proposal["limit_amount"] = amount
            proposal["override_step"] = "days"
            rec["clearance_proposal"] = proposal
            save_registry(registry)
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"Лимит <b>{_fmt(amount)}</b> записан.\n\n"
                        f"Теперь введи срок (дней).\nНапример: <code>7</code>"
                    ),
                    parse_mode="HTML",
                )
            except Exception:
                pass
            return True

        if step == "days":
            days_raw = re.sub(r"[^0-9]", "", text.strip())
            if not days_raw:
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text="Не понял срок. Введи число дней, например: <code>7</code>",
                        parse_mode="HTML",
                    )
                except Exception:
                    pass
                return True
            proposal["limit_days"] = int(days_raw)
            proposal["override_step"] = None
            save_registry(registry)
            limit = float(proposal.get("limit_amount") or 0)
            days  = int(days_raw)
            await _apply_final_clearance(client_name, rec, "limit", limit, days, bot)
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"✅ Лимит применён — <b>{client_name}</b>\n"
                        f"📉 {_fmt(limit)} на {days} дн. Менеджер и Саида уведомлены."
                    ),
                    parse_mode="HTML",
                )
            except Exception:
                pass
            return True

    if chat_id == admin_id:
        for client_name, rec in registry.items():
            if rec.get("status") == "awaiting_clearance_limit":
                amount = _parse_amount(text)
                if amount is None:
                    try:
                        await bot.send_message(
                            chat_id=chat_id,
                            text=(
                                "Не понял сумму лимита.\n"
                                "Введите число, например: <code>1000000</code>"
                            ),
                            parse_mode="HTML",
                        )
                    except Exception as e:
                        LOG.warning("Ошибка запроса лимита после оплаты: %s", e)
                    return True
                await _apply_clearance_limit(client_name, rec, amount, bot)
                return True

    for cid, c in candidates.items():
        if c.get("awaiting_shipment_limit") and chat_id == admin_id:
            amount = _parse_amount(text)
            if amount is None:
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=(
                            "Не понял сумму лимита.\n"
                            "Введите число, например: <code>1000000</code>"
                        ),
                        parse_mode="HTML",
                    )
                except Exception as e:
                    LOG.warning("Ошибка запроса лимита отгрузки: %s", e)
                return True

            c["awaiting_shipment_limit"] = False
            c["shipment_limit"] = amount
            c["admin_approved"] = True
            c["admin_decision"] = "allow_after_payment_with_limit"
            save_state(state)
            await _register_allow_after_payment_with_limit(cid, c, amount, bot)
            return True

    # Ищем кандидата с pending-деталью для этого менеджера
    found_cid = None
    for cid, c in candidates.items():
        if c.get("awaiting_detail") and c.get("manager_chat_id") == chat_id:
            found_cid = cid
            break

    if found_cid is None:
        return False

    c = candidates[found_cid]
    c["manager_note"]   = text.strip()
    c["awaiting_detail"] = False
    save_state(state)

    LOG.info("Детали договорённости от менеджера %s: %s", c["manager"], text[:80])

    # Эскалируем к руководителю с деталями
    await _escalate_yes_to_admin(found_cid, c, bot)

    # Подтверждаем менеджеру
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=f"✅ Передано руководителю — <b>{c['client']}</b>\nДетали зафиксированы.",
            parse_mode="HTML",
        )
    except Exception as e:
        LOG.warning("Ошибка подтверждения менеджеру %s: %s", c["manager"], e)

    return True


async def _handle_admin_response(cid: str, action: str, chat_id: int, bot) -> str:
    state = load_state()
    c = state["candidates"].get(cid)
    if not c:
        return "❓ Клиент не найден."
    if c.get("admin_approved") is not None:
        return "ℹ️ Решение уже принято."

    c["admin_approved"] = action in ("ok", "allow_after", "block_until")
    save_state(state)

    mgr_chat_id = c.get("manager_chat_id")
    today = datetime.now(TZ).strftime("%Y-%m-%d")

    if action in ("ok", "block_until"):
        # Стоп — записываем в реестр для отслеживания оплаты
        status_by_action = {
            "ok": "stopped",
            "block_until": "block_until_payment",
        }
        added_by_by_action = {
            "ok": "admin_manual",
            "block_until": "admin_block_until_payment",
        }
        registry = load_registry()
        registry[c["client"]] = {
            "manager":              c["manager"],
            "manager_chat_id":      mgr_chat_id,
            "approved_at":          today,
            "days_at_approval":     c["days_silence"],
            "debt_at_approval":     c["debt"],
            "status":               status_by_action[action],
            "added_by":             added_by_by_action[action],
            "auto_stopped_at":      None,
            "days_at_stop":         c["days_silence"],
            "discipline_violation": False,
            "cleared_at":           None,
        }
        save_registry(registry)
        saida_action_line = {
            "ok": "Не отгружать до отдельного разрешения руководителя.",
            "block_until": (
                f"Не отгружать до полной оплаты в 1С "
                f"(остаток до {_fmt(STOP_PAID_THRESHOLD)}). После оплаты руководитель проверит решение."
            ),
        }[action]
        manager_action_line = {
            "ok": "Отгрузка запрещена.",
            "block_until": (
                f"Отгрузка запрещена до полной оплаты в 1С "
                f"(остаток до {_fmt(STOP_PAID_THRESHOLD)})."
            ),
        }[action]
        # Уведомить Саиду в реальном времени
        try:
            await bot.send_message(
                chat_id=SAIDA_CHAT_ID,
                text=(
                    f"🚫 <b>Стоп-лист обновлён</b>\n"
                    f"<b>{c['client']}</b> — добавлен.\n"
                    f"Молчит {c['days_silence']}\u202fдн., долг: {_fmt(c['debt'])}\n"
                    f"<i>{saida_action_line}</i>"
                ),
                parse_mode="HTML"
            )
        except Exception as e:
            LOG.warning("Ошибка уведомления Саиды о стопе %s: %s", c["client"], e)
        # Уведомить менеджера с кнопкой «Клиент оплатил»
        if mgr_chat_id:
            try:
                kb_mgr = InlineKeyboardMarkup([[
                    InlineKeyboardButton(
                        "💳 Клиент оплатил, ждём разноски",
                        callback_data=f"dstop_mgr_paid|{cid}"
                    )
                ]])
                await bot.send_message(
                    chat_id=mgr_chat_id,
                    text=(
                        f"🚫 <b>Решение руководителя по отгрузке</b>\n"
                        f"{c['client']}\n{manager_action_line}"
                    ),
                    parse_mode="HTML",
                    reply_markup=kb_mgr
                )
            except Exception as e:
                LOG.warning("Ошибка уведомления менеджера о стопе %s: %s", c["client"], e)
        result_labels = {
            "ok": "🚫 Запрет утверждён",
            "block_until": "🔒 Запрет до полной оплаты зафиксирован",
        }
        return f"{result_labels[action]} — <b>{c['client']}</b>"
    else:
        # Разрешил — если менеджер говорил "yes", записываем как exception
        if c.get("manager_response") == "yes":
            registry = load_registry()
            registry[c["client"]] = {
                "manager":              c["manager"],
                "manager_chat_id":      mgr_chat_id,
                "approved_at":          today,
                "days_at_approval":     c["days_silence"],
                "debt_at_approval":     c["debt"],
                "status":               "exception",
                "added_by":             "admin_manual",
                "auto_stopped_at":      None,
                "days_at_stop":         None,
                "discipline_violation": False,
                "cleared_at":           None,
            }
            save_registry(registry)
        # Уведомить менеджера
        if mgr_chat_id:
            try:
                await bot.send_message(
                    chat_id=mgr_chat_id,
                    text=f"✅ <b>Руководитель разрешил отгрузку</b>\n{c['client']}",
                    parse_mode="HTML"
                )
            except Exception as e:
                LOG.warning("Ошибка уведомления менеджера о разрешении %s: %s", c["client"], e)
        return f"✅ Разрешено — <b>{c['client']}</b>"


async def _handle_admin_limit_after_request(cid: str, chat_id: int, bot) -> str:
    """Руководитель выбрал: после полной оплаты разрешить новую отгрузку с лимитом."""
    admin_id = _get_admin_chat_id()
    if chat_id != admin_id:
        return "⛔ Решение по лимиту может принять только руководитель."

    state = load_state()
    c = state["candidates"].get(cid)
    if not c:
        return "❓ Клиент не найден."
    if c.get("admin_approved") is not None:
        return "ℹ️ Решение уже принято."

    c["awaiting_shipment_limit"] = True
    c["limit_requested_at"] = datetime.now(TZ).strftime("%H:%M")
    save_state(state)
    return (
        f"📉 <b>{c['client']}</b>\n\n"
        f"Введите лимит новой отгрузки после полной оплаты.\n"
        f"Например: <code>1000000</code>\n\n"
        f"Старый долг должен быть закрыт в 1С до {_fmt(STOP_PAID_THRESHOLD)}. "
        f"После этого Саида получит разрешение отгружать только в пределах указанного лимита."
    )


async def _register_allow_after_payment_with_limit(
    cid: str,
    c: Dict[str, Any],
    limit: float,
    bot,
) -> None:
    """Сохраняет решение: после закрытия старого долга отгрузка разрешена с лимитом."""
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    registry = load_registry()
    registry[c["client"]] = {
        "manager":              c["manager"],
        "manager_chat_id":      c.get("manager_chat_id"),
        "approved_at":          today,
        "days_at_approval":     c["days_silence"],
        "debt_at_approval":     c["debt"],
        "status":               "allow_after_payment",
        "added_by":             "admin_allow_after_payment_with_limit",
        "shipment_limit":       limit,
        "auto_stopped_at":      None,
        "days_at_stop":         c["days_silence"],
        "discipline_violation": False,
        "cleared_at":           None,
    }
    save_registry(registry)

    mgr_chat_id = c.get("manager_chat_id")
    client = c["client"]
    saida_msg = (
        f"⏳ <b>Отгрузка после полной оплаты с лимитом</b>\n"
        f"<b>{client}</b>\n\n"
        f"Сейчас не отгружать. Старый долг должен быть закрыт в 1С "
        f"(остаток до {_fmt(STOP_PAID_THRESHOLD)}).\n\n"
        f"После закрытия старого долга новая отгрузка разрешена "
        f"только в пределах <b>{_fmt(limit)}</b>.\n"
        f"Больше лимита не отгружать без отдельного разрешения руководителя."
    )
    mgr_msg = (
        f"⏳ <b>Решение руководителя по отгрузке</b>\n"
        f"{client}\n\n"
        f"После полной оплаты старого долга клиенту можно будет отгружать "
        f"с лимитом <b>{_fmt(limit)}</b>."
    )
    admin_msg = (
        f"✅ Зафиксировано — <b>{client}</b>\n"
        f"После полной оплаты: лимит новой отгрузки <b>{_fmt(limit)}</b>."
    )

    for target, msg in ((SAIDA_CHAT_ID, saida_msg), (mgr_chat_id, mgr_msg), (_get_admin_chat_id(), admin_msg)):
        if target:
            try:
                await bot.send_message(chat_id=target, text=msg, parse_mode="HTML")
            except Exception as e:
                LOG.warning("Ошибка уведомления лимита отгрузки %s (chat=%s): %s", client, target, e)


async def _handle_saida_payment(client_key: str, chat_id: int, bot) -> str:
    """
    Саида нажала «Оплата получена» — оплата есть, но ещё не разнесена в 1С.
    Уведомляем руководителя, Саиде подтверждаем что сообщение принято.
    """
    admin_id = _get_admin_chat_id()
    now_str  = datetime.now(TZ).strftime("%H:%M")

    admin_msg = (
        f"💰 <b>Саида: оплата получена (не разнесена в 1С)</b>\n\n"
        f"Клиент: <b>{client_key}</b>\n"
        f"Время: {now_str}\n\n"
        f"Саида планирует отгрузить этого клиента.\n"
        f"Оплата поступила, но ещё не проведена в системе.\n\n"
        f"Если всё верно — ничего делать не нужно.\n"
        f"Если нет — свяжитесь с Саидой."
    )
    if admin_id:
        try:
            adm_msg = await bot.send_message(
                chat_id=admin_id, text=admin_msg, parse_mode="HTML"
            )
            _schedule_delete(admin_id, adm_msg.message_id, adm_msg.date.timestamp())
        except Exception as e:
            LOG.warning("Ошибка уведомления руководителя об оплате %s: %s", client_key, e)

    LOG.info("Саида: оплата получена (не разнесена) — клиент '%s'", client_key)
    return f"✅ Принято. Руководитель уведомлён.\nМожешь отгружать <b>{client_key}</b>."


async def _handle_clearance(client_key: str, action: str, chat_id: int, bot) -> str:
    """Руководитель решает снять/оставить клиента после полной оплаты."""
    registry = load_registry()

    # Ищем по полному имени (ключ мог быть обрезан до 40 символов в callback)
    matched_key = None
    for name in registry:
        if name.startswith(client_key) or name[:26] == client_key[:26]:
            matched_key = name
            break

    if not matched_key:
        return "❓ Клиент не найден в реестре."

    rec = registry[matched_key]
    if rec.get("status") != "pending_clearance":
        return "ℹ️ Статус клиента уже изменён."

    today = datetime.now(TZ).strftime("%Y-%m-%d")
    managers = _load_managers()
    mgr_chat_id = rec.get("manager_chat_id") or managers.get(rec.get("manager", ""), 0)

    if action == "clear":
        rec["status"]     = "cleared"
        rec["cleared_at"] = today
        # discipline_violation остаётся True — при следующем нарушении сразу к руководителю
        save_registry(registry)

        note = (
            f"✅ <b>{matched_key}</b> снят со стопа.\n"
            f"⚠️ Ранее нарушал фин. дисциплину — следующий инцидент сразу к руководителю."
        )
        for cid_tg in {mgr_chat_id, SAIDA_CHAT_ID}:
            if cid_tg:
                try:
                    await bot.send_message(chat_id=cid_tg, text=note, parse_mode="HTML")
                except Exception as e:
                    LOG.warning("Ошибка уведомления о снятии стопа: %s", e)

        return f"✅ {matched_key} снят со стопа. Менеджер и Саида уведомлены."

    else:  # keep
        rec["status"] = "auto_stopped"  # возвращаем в авто-стоп
        save_registry(registry)
        return f"🚫 {matched_key} — оставлен на стопе."


async def _auto_clear_stop_after_saida_full(c: Dict[str, Any], bot) -> bool:
    """Авто-снимает клиента со стопа после полной оплаты, подтверждённой Саидой."""
    client_name = str(c.get("client") or "").strip()
    if not client_name:
        return False

    registry = load_registry()
    rec = registry.get(client_name)
    if not isinstance(rec, dict):
        return False

    today = datetime.now(TZ).strftime("%Y-%m-%d")
    rec["status"] = "cleared"
    rec["cleared_at"] = today
    rec["saida_confirmed_full_at"] = datetime.now(TZ).strftime("%H:%M")
    registry[client_name] = rec
    save_registry(registry)

    try:
        from collector.shipment_control import resolve_decision as _resolve_ship_decision
        _resolve_ship_decision(client_name, "saida_confirmed_full")
    except Exception as e:
        LOG.warning("Ошибка закрытия shipment decision %s после подтверждения Саиды: %s", client_name, e)

    mgr_chat_id = rec.get("manager_chat_id") or c.get("manager_chat_id")
    saida_msg = (
        f"✅ <b>{client_name}</b> снят со стопа автоматически.\n"
        f"Полная оплата подтверждена, можно отгружать."
    )
    mgr_msg = (
        f"✅ <b>{client_name}</b> снят со стопа автоматически.\n"
        f"Саида подтвердила полную оплату. Клиент больше не в стоп-листе."
    )
    for target, msg in ((SAIDA_CHAT_ID, saida_msg), (mgr_chat_id, mgr_msg)):
        if target:
            try:
                await bot.send_message(chat_id=target, text=msg, parse_mode="HTML")
            except Exception as e:
                LOG.warning("Ошибка уведомления об авто-снятии стопа %s (chat=%s): %s", client_name, target, e)

    LOG.info("Авто-снятие стопа после полной оплаты Саиды: %s", client_name)
    return True


async def _handle_conditional_clearance(client_key: str, chat_id: int, bot) -> str:
    """
    Руководитель нажал «⚠️ Условная отгрузка».
    Оплата ещё не разнесена в 1С, но руководитель разрешает отгрузить под условие.
    Статус → 'conditional'. Саида и менеджер уведомляются немедленно.
    """
    registry = load_registry()

    matched_key = None
    for name in registry:
        if name.startswith(client_key) or name[:26] == client_key[:26]:
            matched_key = name
            break

    if not matched_key:
        return "❓ Клиент не найден в реестре."

    rec = registry[matched_key]
    if rec.get("status") not in ("pending_clearance", "auto_stopped", "stopped"):
        return "ℹ️ Условная отгрузка неприменима к текущему статусу клиента."

    today = datetime.now(TZ).strftime("%Y-%m-%d")
    rec["status"] = "conditional"
    rec["conditional_at"] = today
    # discipline_violation сохраняется — при следующем нарушении сразу к руководителю
    save_registry(registry)

    mgr_chat_id = rec.get("manager_chat_id") or _load_managers().get(rec.get("manager", ""), 0)

    saida_msg = (
        f"⚠️ <b>Условная отгрузка разрешена</b>\n"
        f"<b>{matched_key}</b>\n"
        f"Руководитель разрешил отгрузить под условие оплаты.\n"
        f"Оплата ожидается — отгрузи, но контролируй поступление."
    )
    mgr_msg = (
        f"⚠️ <b>Условная отгрузка — {matched_key}</b>\n"
        f"Руководитель разрешил отгрузить клиента под условие оплаты.\n"
        f"Проконтролируй поступление платежа."
    )

    for target, text in [(SAIDA_CHAT_ID, saida_msg), (mgr_chat_id, mgr_msg)]:
        if target:
            try:
                await bot.send_message(chat_id=target, text=text, parse_mode="HTML")
            except Exception as e:
                LOG.warning("Ошибка уведомления условной отгрузки %s (chat=%s): %s",
                            matched_key, target, e)

    LOG.info("Условная отгрузка: %s", matched_key)
    return f"⚠️ Условная отгрузка — <b>{matched_key}</b>. Менеджер и Саида уведомлены."


async def _handle_clearance_limit_request(client_key: str, chat_id: int, bot) -> str:
    """Руководитель выбрал лимит новой отгрузки после уже закрытого долга."""
    if chat_id != _get_admin_chat_id():
        return "⛔ Лимит отгрузки может задать только руководитель."

    registry = load_registry()
    matched_key = None
    for name in registry:
        if name.startswith(client_key) or name[:26] == client_key[:26]:
            matched_key = name
            break

    if not matched_key:
        return "❓ Клиент не найден в реестре."

    rec = registry[matched_key]
    if rec.get("status") != "pending_clearance":
        return "ℹ️ Статус клиента уже изменён."

    rec["status"] = "awaiting_clearance_limit"
    rec["limit_requested_at"] = datetime.now(TZ).strftime("%H:%M")
    save_registry(registry)
    return (
        f"📉 <b>{matched_key}</b>\n\n"
        f"Введите лимит новой отгрузки.\n"
        f"Например: <code>1000000</code>\n\n"
        f"Саида получит разрешение отгружать только в пределах указанного лимита."
    )


async def _apply_clearance_limit(client_name: str, rec: Dict[str, Any], limit: float, bot) -> None:
    """Снимает старый стоп после оплаты и сохраняет лимит новой отгрузки."""
    registry = load_registry()
    stored = registry.get(client_name, rec)
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    stored["status"] = "cleared_limited"
    stored["cleared_at"] = today
    stored["shipment_limit"] = limit
    registry[client_name] = stored
    save_registry(registry)

    mgr_chat_id = stored.get("manager_chat_id") or _load_managers().get(stored.get("manager", ""), 0)
    note = (
        f"✅ <b>{client_name}</b> — отгрузка разрешена с лимитом.\n\n"
        f"Старый долг закрыт.\n"
        f"Новый лимит отгрузки: <b>{_fmt(limit)}</b>.\n"
        f"Больше лимита не отгружать без отдельного разрешения руководителя."
    )
    for target in {_get_admin_chat_id(), mgr_chat_id, SAIDA_CHAT_ID}:
        if target:
            try:
                await bot.send_message(chat_id=target, text=note, parse_mode="HTML")
            except Exception as e:
                LOG.warning("Ошибка уведомления лимита после оплаты %s (chat=%s): %s",
                            client_name, target, e)


# ══════════════════════════════════════════════════════════════════════
# Цепочка: менеджер сообщает об оплате → Саида проверяет → admin решает
# ══════════════════════════════════════════════════════════════════════

async def _handle_mgr_paid_claim(cid: str, chat_id: int, bot) -> str:
    """Менеджер нажал «Клиент оплатил, ждём разноски»."""
    state = load_state()
    c = state["candidates"].get(cid)
    if not c:
        return "❓ Клиент не найден в сегодняшнем списке."
    if c.get("mgr_paid_claimed"):
        return "ℹ️ Вы уже сообщили об оплате этого клиента."

    c["mgr_paid_claimed"] = True
    c["mgr_paid_at"] = datetime.now(TZ).strftime("%H:%M")
    save_state(state)

    admin_id = _get_admin_chat_id()
    now_str = datetime.now(TZ).strftime("%H:%M")

    # Саиде — запрос с двумя кнопками
    kb_saida = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Полная оплата",    callback_data=f"dstop_saida_full|{cid}"),
            InlineKeyboardButton("⚠️ Частичная оплата", callback_data=f"dstop_saida_partial|{cid}"),
        ],
        [InlineKeyboardButton("❓ Что это значит?", callback_data="dstop_saida_help")],
    ])
    try:
        await bot.send_message(
            chat_id=SAIDA_CHAT_ID,
            text=(
                f"💳 <b>Менеджер {c['manager']} сообщает об оплате</b>\n\n"
                f"Клиент: <b>{c['client']}</b>\n"
                f"Время: {now_str}\n\n"
                f"Проверь разноску и подтверди:"
            ),
            parse_mode="HTML",
            reply_markup=kb_saida
        )
    except Exception as e:
        LOG.warning("Ошибка уведомления Саиды об оплате %s: %s", c["client"], e)

    # Руководителю — информационно
    if admin_id:
        try:
            await bot.send_message(
                chat_id=admin_id,
                text=(
                    f"📋 <b>Менеджер {c['manager']}</b> сообщил об оплате:\n"
                    f"<b>{c['client']}</b> ({now_str})\n"
                    f"Саида уведомлена — ожидаем подтверждение разноски."
                ),
                parse_mode="HTML"
            )
        except Exception as e:
            LOG.warning("Ошибка уведомления руководителя о заявке менеджера %s: %s", c["client"], e)

    return f"✅ Саида уведомлена. Ожидайте подтверждения разноски."


async def _handle_saida_confirm_full(cid: str, chat_id: int, bot) -> str:
    """Саида подтвердила полную оплату — руководителю запрос на разрешение."""
    state = load_state()
    c = state["candidates"].get(cid)
    if not c:
        return "❓ Клиент не найден."
    if c.get("saida_payment_confirmed") is not None:
        return "ℹ️ Статус оплаты уже зафиксирован."

    c["saida_payment_confirmed"] = "full"
    save_state(state)

    cleared = await _auto_clear_stop_after_saida_full(c, bot)
    if cleared:
        return f"✅ Полная оплата подтверждена. <b>{c['client']}</b> снят со стопа автоматически."

    LOG.warning("Саида подтвердила полную оплату, но клиент %s не найден в реестре стопа", c.get("client"))
    return (
        f"⚠️ Полная оплата подтверждена, но <b>{c['client']}</b> не найден в реестре стопа.\n"
        f"Проверьте запись вручную."
    )

async def _handle_saida_confirm_partial(cid: str, chat_id: int, bot) -> str:
    """Саида сообщила о частичной оплате — конфликт с заявлением менеджера."""
    state = load_state()
    c = state["candidates"].get(cid)
    if not c:
        return "❓ Клиент не найден."
    if c.get("saida_payment_confirmed") is not None:
        return "ℹ️ Статус оплаты уже зафиксирован."

    c["saida_payment_confirmed"] = "partial"
    save_state(state)

    admin_id = _get_admin_chat_id()
    mgr_chat_id = c.get("manager_chat_id")
    now_str = datetime.now(TZ).strftime("%H:%M")

    # Руководителю — предупреждение о конфликте
    if admin_id:
        try:
            await bot.send_message(
                chat_id=admin_id,
                text=(
                    f"⚠️ <b>Конфликт по оплате!</b>\n\n"
                    f"Менеджер <b>{c['manager']}</b> заявил полную оплату,\n"
                    f"но Саида говорит — <b>частичная</b>.\n\n"
                    f"Клиент: <b>{c['client']}</b>\n"
                    f"Время: {now_str}\n\n"
                    f"Клиент остаётся на стопе."
                ),
                parse_mode="HTML"
            )
        except Exception as e:
            LOG.warning("Ошибка уведомления руководителя о частичной оплате %s: %s", c["client"], e)

    # Менеджеру — оплата не подтверждена
    if mgr_chat_id:
        try:
            await bot.send_message(
                chat_id=mgr_chat_id,
                text=(
                    f"⚠️ <b>Саида: оплата частичная</b>\n"
                    f"{c['client']}\n"
                    f"Разноска не подтверждена. Клиент остаётся на стопе."
                ),
                parse_mode="HTML"
            )
        except Exception as e:
            LOG.warning("Ошибка уведомления менеджера о частичной оплате %s: %s", c["client"], e)

    return f"⚠️ Руководитель уведомлён о конфликте. Клиент остаётся на стопе."


async def _handle_admin_allow_after_saida(cid: str, chat_id: int, bot) -> str:
    """Руководитель разрешил отгрузку после подтверждения Саиды."""
    state = load_state()
    c = state["candidates"].get(cid)
    if not c:
        return "❓ Клиент не найден."
    if c.get("admin_approved") is not None:
        return "ℹ️ Решение уже принято."

    c["admin_approved"] = False  # убираем из стоп-листа
    save_state(state)

    mgr_chat_id = c.get("manager_chat_id")
    now_str = datetime.now(TZ).strftime("%H:%M")

    # Менеджеру — разрешение
    if mgr_chat_id:
        try:
            await bot.send_message(
                chat_id=mgr_chat_id,
                text=f"✅ <b>Руководитель разрешил отгрузку</b>\n{c['client']}",
                parse_mode="HTML"
            )
        except Exception as e:
            LOG.warning("Ошибка уведомления менеджера о разрешении %s: %s", c["client"], e)

    # Саиде — разрешение
    try:
        await bot.send_message(
            chat_id=SAIDA_CHAT_ID,
            text=f"✅ <b>Руководитель разрешил отгрузку</b>\n{c['client']} ({now_str})",
            parse_mode="HTML"
        )
    except Exception as e:
        LOG.warning("Ошибка уведомления Саиды о разрешении %s: %s", c["client"], e)

    return f"✅ Разрешено — <b>{c['client']}</b>. Менеджер и Саида уведомлены."


# ══════════════════════════════════════════════════════════════════════
# Новая цепочка снятия стопа: менеджер → руководитель → финал
# ══════════════════════════════════════════════════════════════════════

async def _notify_manager_clearance_proposal(
    client_name: str, rec: Dict[str, Any], bot
) -> None:
    """Уведомляет менеджера о полной оплате и запрашивает его предложение."""
    mgr_id = rec.get("manager_chat_id") or _load_managers().get(rec.get("manager", ""), 0)
    if not mgr_id:
        await _notify_admin_clearance_fallback(client_name, rec, bot)
        return

    current = _get_client_current_state(client_name)
    debt_line = f"Остаток в 1С: <b>{_fmt(current['debt'])}</b>" if current else ""
    days_on_stop = rec.get("days_at_stop", "?")
    key = client_name[:26]

    kb = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Разрешить полностью", callback_data=f"dstop_mgr_cl_clear|{key}"),
            InlineKeyboardButton("🔒 Только предоплата",   callback_data=f"dstop_mgr_cl_prepay|{key}"),
        ],
        [
            InlineKeyboardButton("📉 С лимитом",          callback_data=f"dstop_mgr_cl_limit|{key}"),
            InlineKeyboardButton("🚫 Держать на стопе",   callback_data=f"dstop_mgr_cl_keep|{key}"),
        ],
    ])
    msg = (
        f"💰 <b>{client_name}</b> — клиент рассчитался.\n"
        f"{debt_line}\n"
        f"Был на стопе {days_on_stop}\u202fдн.\n\n"
        f"Твоё предложение руководителю по дальнейшей работе с клиентом:"
    )
    try:
        await bot.send_message(chat_id=mgr_id, text=msg, parse_mode="HTML", reply_markup=kb)
    except Exception as e:
        LOG.warning("Ошибка уведомления менеджера о снятии стопа %s: %s", client_name, e)
        await _notify_admin_clearance_fallback(client_name, rec, bot)


async def _notify_admin_clearance_fallback(
    client_name: str, rec: Dict[str, Any], bot
) -> None:
    """Уведомляет руководителя напрямую если нет менеджера."""
    admin_id = _get_admin_chat_id()
    if not admin_id:
        return
    current = _get_client_current_state(client_name)
    debt_line = f"Остаток: <b>{_fmt(current['debt'])}</b>" if current else ""
    key = client_name[:26]
    kb = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Разрешить",       callback_data=f"dstop_adm_cl_clear|{key}"),
            InlineKeyboardButton("🔒 Предоплата",      callback_data=f"dstop_adm_cl_prepay|{key}"),
        ],
        [
            InlineKeyboardButton("📉 С лимитом",      callback_data=f"dstop_adm_cl_limit|{key}"),
            InlineKeyboardButton("⛔ Чёрный список",  callback_data=f"dstop_adm_cl_blacklist|{key}"),
        ],
    ])
    msg = (
        f"💰 <b>{client_name}</b> — оплата в 1С.\n{debt_line}\n"
        f"Менеджер не назначен. Выберите решение:"
    )
    try:
        await bot.send_message(chat_id=admin_id, text=msg, parse_mode="HTML", reply_markup=kb)
    except Exception as e:
        LOG.warning("Ошибка уведомления руководителя (fallback) %s: %s", client_name, e)


async def _handle_mgr_clearance_proposal(
    client_key: str, action: str, chat_id: int, bot
) -> str:
    """Менеджер выбрал своё предложение руководителю по снятию стопа."""
    registry = load_registry()
    matched_key = next(
        (n for n in registry if n[:26] == client_key[:26] or n.startswith(client_key)), None
    )
    if not matched_key:
        return "❓ Клиент не найден в реестре."
    rec = registry[matched_key]
    if rec.get("status") != "pending_clearance_mgr":
        return "ℹ️ Статус уже изменён."
    if int(rec.get("manager_chat_id") or 0) != int(chat_id):
        return "⛔ Это решение не для вас."

    if action == "limit":
        rec["status"] = "awaiting_mgr_limit_input"
        rec["clearance_proposal"] = {"step": "amount", "action": "limit",
                                     "limit_amount": None, "limit_days": None}
        save_registry(registry)
        return (
            f"📉 <b>{matched_key}</b>\n\n"
            f"Введи лимит одной отгрузки (₸).\n"
            f"Например: <code>100000</code>"
        )

    rec["status"] = "pending_clearance_admin"
    rec["clearance_proposal"] = {"action": action, "limit_amount": None, "limit_days": None}
    save_registry(registry)

    await _forward_mgr_proposal_to_admin(matched_key, rec, bot)

    labels = {"clear": "✅ Разрешить полностью", "prepay": "🔒 Только предоплата",
               "keep": "🚫 Держать на стопе"}
    return f"Предложение «{labels.get(action, action)}» передано руководителю."


async def _forward_mgr_proposal_to_admin(
    client_name: str, rec: Dict[str, Any], bot
) -> None:
    """Пересылает предложение менеджера руководителю с кнопками подтверждения/изменения."""
    admin_id = _get_admin_chat_id()
    if not admin_id:
        return

    proposal = rec.get("clearance_proposal", {})
    action   = proposal.get("action", "?")
    manager  = rec.get("manager", "?")
    key      = client_name[:26]

    limit_amount = float(proposal.get("limit_amount") or 0)
    limit_days   = proposal.get("limit_days", "?")
    action_labels = {
        "clear":  "✅ Разрешить полностью",
        "prepay": "🔒 Только предоплата",
        "keep":   "🚫 Держать на стопе",
        "limit":  f"📉 С лимитом {_fmt(limit_amount)} на {limit_days} дн.",
    }

    current = _get_client_current_state(client_name)
    debt_line = f"Остаток: <b>{_fmt(current['debt'])}</b>\n" if current else ""

    msg = (
        f"💰 <b>{client_name}</b> — оплата в 1С.\n"
        f"{debt_line}"
        f"Менеджер <b>{manager}</b> предлагает: <b>{action_labels.get(action, action)}</b>\n\n"
        f"Утвердить или изменить:"
    )
    kb = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Согласен",       callback_data=f"dstop_adm_cl_confirm|{key}"),
            InlineKeyboardButton("✅ Разрешить",      callback_data=f"dstop_adm_cl_clear|{key}"),
        ],
        [
            InlineKeyboardButton("📉 С лимитом",     callback_data=f"dstop_adm_cl_limit|{key}"),
            InlineKeyboardButton("🔒 Предоплата",    callback_data=f"dstop_adm_cl_prepay|{key}"),
        ],
        [
            InlineKeyboardButton("⛔ Чёрный список", callback_data=f"dstop_adm_cl_blacklist|{key}"),
        ],
    ])
    try:
        await bot.send_message(chat_id=admin_id, text=msg, parse_mode="HTML", reply_markup=kb)
    except Exception as e:
        LOG.warning("Ошибка отправки предложения руководителю %s: %s", client_name, e)


async def _handle_admin_clearance_confirm(client_key: str, chat_id: int, bot) -> str:
    """Руководитель согласился с предложением менеджера."""
    if chat_id != _get_admin_chat_id():
        return "⛔ Только для руководителя."
    registry = load_registry()
    matched_key = next(
        (n for n in registry if n[:26] == client_key[:26] or n.startswith(client_key)), None
    )
    if not matched_key:
        return "❓ Клиент не найден."
    rec = registry[matched_key]
    if rec.get("status") != "pending_clearance_admin":
        return "ℹ️ Статус уже изменён."

    proposal = rec.get("clearance_proposal", {})
    action   = proposal.get("action", "clear")
    limit    = float(proposal.get("limit_amount") or 0)
    days     = int(proposal.get("limit_days") or 0)

    await _apply_final_clearance(matched_key, rec, action, limit, days, bot)
    return f"✅ Утверждено — <b>{matched_key}</b>. Менеджер и Саида уведомлены."


async def _handle_admin_clearance_override(
    client_key: str, action: str, chat_id: int, bot
) -> str:
    """Руководитель изменяет предложение менеджера."""
    if chat_id != _get_admin_chat_id():
        return "⛔ Только для руководителя."
    registry = load_registry()
    matched_key = next(
        (n for n in registry if n[:26] == client_key[:26] or n.startswith(client_key)), None
    )
    if not matched_key:
        return "❓ Клиент не найден."
    rec = registry[matched_key]

    if action == "limit":
        rec["status"] = "awaiting_admin_limit_override"
        rec.setdefault("clearance_proposal", {})["override_step"] = "amount"
        save_registry(registry)
        return (
            f"📉 <b>{matched_key}</b>\n\n"
            f"Введи лимит одной отгрузки (₸).\n"
            f"Например: <code>100000</code>"
        )

    await _apply_final_clearance(matched_key, rec, action, 0, 0, bot)
    labels = {"clear": "✅ Разрешено", "prepay": "🔒 Только предоплата", "blacklist": "⛔ Чёрный список"}
    return f"{labels.get(action, action)} — <b>{matched_key}</b>. Менеджер и Саида уведомлены."


async def _apply_final_clearance(
    client_name: str,
    rec: Dict[str, Any],
    action: str,
    limit: float,
    days: int,
    bot,
) -> None:
    """Применяет финальное решение и уведомляет менеджера и Саиду."""
    today    = datetime.now(TZ).strftime("%Y-%m-%d")
    registry = load_registry()
    stored   = registry.get(client_name, rec)
    mgr_id   = stored.get("manager_chat_id") or _load_managers().get(stored.get("manager", ""), 0)

    if action == "clear":
        stored["status"]     = "cleared"
        stored["cleared_at"] = today
        saida_text = f"✅ <b>{client_name}</b> снят со стопа. Отгрузка разрешена."
        mgr_text   = f"✅ <b>Руководитель разрешил отгрузку</b>\n{client_name}\nКлиент снят со стопа."
        # Убираем из known-файла — Саида должна знать, что он снят
        known = _load_saida_known()
        known.discard(client_name)
        _save_saida_known(known)

    elif action == "prepay":
        stored["status"]       = "prepayment_only"
        stored["prepay_set_at"] = today
        saida_text = (
            f"💳 <b>{client_name}</b> — только предоплата.\n"
            f"Отгружать только после 100% предоплаты."
        )
        mgr_text = (
            f"💳 <b>Решение руководителя</b>\n{client_name}\n"
            f"Отгрузка только после 100% предоплаты."
        )

    elif action == "limit":
        expires = (
            (datetime.now(TZ) + timedelta(days=days)).strftime("%Y-%m-%d")
            if days > 0 else None
        )
        stored["status"]         = "cleared_limited"
        stored["cleared_at"]     = today
        stored["shipment_limit"] = limit
        stored["limit_days"]     = days
        stored["limit_expires"]  = expires
        exp_str   = f" до {expires}" if expires else ""
        limit_str = _fmt(limit)
        saida_text = (
            f"📉 <b>{client_name}</b> — отгрузка с лимитом.\n"
            f"Одна отгрузка ≤ <b>{limit_str}</b>{exp_str}.\n"
            f"Больше лимита — только с разрешения руководителя."
        )
        mgr_text = (
            f"📉 <b>Решение руководителя</b>\n{client_name}\n"
            f"Отгрузка разрешена с лимитом <b>{limit_str}</b>{exp_str}."
        )

    elif action == "blacklist":
        stored["status"]         = "blacklisted"
        stored["blacklisted_at"] = today
        saida_text = (
            f"⛔ <b>{client_name}</b> — ЧЁРНЫЙ СПИСОК.\n"
            f"Не отгружать ни при каких условиях."
        )
        mgr_text = (
            f"⛔ <b>Решение руководителя</b>\n{client_name}\n"
            f"Клиент в чёрном списке. Отгрузка запрещена."
        )

    else:  # keep
        stored["status"] = "stopped"
        registry[client_name] = stored
        save_registry(registry)
        LOG.info("Клиент %s оставлен на стопе (action=%s)", client_name, action)
        return

    registry[client_name] = stored
    save_registry(registry)
    LOG.info("Финальное решение '%s' для клиента %s", action, client_name)

    for target, t_text in [(mgr_id, mgr_text), (SAIDA_CHAT_ID, saida_text)]:
        if target:
            try:
                await bot.send_message(chat_id=target, text=t_text, parse_mode="HTML")
            except Exception as e:
                LOG.warning("Ошибка уведомления финальное решение %s (chat=%s): %s",
                            client_name, target, e)


# ══════════════════════════════════════════════════════════════════════
# Полный стоп-лист для Саиды по запросу
# ══════════════════════════════════════════════════════════════════════

async def send_saida_full_stoplist_with_help(bot) -> None:
    """Полный стоп-лист Саиды + отдельная help-кнопка над списком."""
    try:
        await bot.send_message(
            chat_id=SAIDA_CHAT_ID,
            text="📋 Актуальный стоп-лист ниже. Если неясно, что делать, нажми кнопку под этим сообщением.",
            reply_markup=_kb_saida_help_only(),
        )
    except Exception as e:
        LOG.warning("РћС€РёР±РєР° help-промпта перед полным стоп-листом Саиды: %s", e)
    await send_saida_full_stoplist(bot)


async def send_saida_full_stoplist(bot) -> None:
    """Полный актуальный стоп-лист для Саиды (по нажатию кнопки)."""
    registry  = load_registry()
    today_str = datetime.now(TZ).strftime("%d.%m.%Y")

    stop_statuses = (
        "auto_stopped", "stopped", "allow_after_payment",
        "block_until_payment", "prepayment_only", "blacklisted", "cleared_limited",
    )
    items = [(n, r) for n, r in registry.items() if r.get("status") in stop_statuses]

    if not items:
        try:
            await bot.send_message(
                chat_id=SAIDA_CHAT_ID,
                text=f"✅ <b>Стоп-лист пуст</b>\nНа {today_str} все клиенты в порядке.",
                parse_mode="HTML",
            )
        except Exception as e:
            LOG.warning("Ошибка отправки полного стоп-листа Саиде: %s", e)
        return

    try:
        hdr = await bot.send_message(
            chat_id=SAIDA_CHAT_ID,
            text=f"📋 <b>Стоп-лист — {today_str}</b>  ({len(items)}\u202fклиентов)",
            parse_mode="HTML",
        )
        _schedule_delete(SAIDA_CHAT_ID, hdr.message_id, hdr.date.timestamp())
    except Exception as e:
        LOG.warning("Ошибка заголовка полного стоп-листа: %s", e)

    for name, rec in sorted(items, key=lambda x: x[1].get("days_at_stop", 0), reverse=True):
        status = rec.get("status", "")
        days   = rec.get("days_at_stop", "?")
        mgr    = rec.get("manager", "?")
        limit  = float(rec.get("shipment_limit") or 0)

        if status == "blacklisted":
            icon, note = "⛔", "ЧЁРНЫЙ СПИСОК — не отгружать"
        elif status == "prepayment_only":
            icon, note = "💳", "только 100% предоплата"
        elif status == "cleared_limited":
            exp = rec.get("limit_expires", "")
            icon = "📉"
            note = f"лимит {_fmt(limit)}" + (f" до {exp}" if exp else "")
        elif status == "allow_after_payment":
            icon = "⏳"
            note = (f"после оплаты, лимит {_fmt(limit)}" if limit > 0 else "после полной оплаты")
        elif status == "block_until_payment":
            icon, note = "🔒", "запрет до полной оплаты"
        else:
            icon, note = "🚫", "не отгружать"

        text = (
            f"{icon} <b>{name}</b>\n"
            f"Молчит {days}\u202fдн. · {note} [{mgr}]\n"
            f"<i>Не отгружать без разрешения руководителя</i>"
        )
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("💰 Оплата получена", callback_data=f"dstop_paid|{name[:26]}")
        ]])
        try:
            item_msg = await bot.send_message(
                chat_id=SAIDA_CHAT_ID, text=text, parse_mode="HTML", reply_markup=kb
            )
            _schedule_delete(SAIDA_CHAT_ID, item_msg.message_id, item_msg.date.timestamp())
        except Exception as e:
            LOG.warning("Ошибка позиции '%s' полного стоп-листа: %s", name, e)

#!/usr/bin/env python
# coding: utf-8
"""
debt_stop_control.py · v1.0.3 (2026-04-13)

Контроль стоп-листа отгрузки — уведомление Саиды-бухгалтера.

Расписание (рабочие дни):
  14:00 → мониторинг: одобренные исключения перешли 15 дней → авто-стоп
  17:00 → запросы менеджерам по новым кандидатам
  каждые 30 мин → напоминания менеджерам по незакрытым запросам
  19:00 → нет ответа → эскалация руководителю
  22:00 → Саида получает финальный список «не отгружать»

Пороги:
   7–9 дней  → ⚡ Просрочка — запрос менеджеру
  10+ дней   → 🔴 Молчание  — запрос менеджеру
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

LOG = logging.getLogger("debt_stop_control")

# ── Пути ────────────────────────────────────────────────────────────
_THIS = Path(__file__).resolve()
ROOT = _THIS.parent.parent if _THIS.parent.name == "bot" else _THIS.parent
TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))
JSON_DIR   = ROOT / "reports" / "json"
CONFIG_DIR = ROOT / "config"

STATE_FILE      = ROOT / "reports" / "debt_stop_state.json"
REGISTRY_FILE   = ROOT / "reports" / "debt_stop_registry.json"
DELETION_QUEUE  = ROOT / "logs" / "deletion_queue.json"
SAIDA_INTRO_FILE = ROOT / "reports" / "debt_stop_saida_intro_sent.json"

# Саида — бухгалтер-оператор
SAIDA_CHAT_ID = int(os.getenv("SAIDA_CHAT_ID", "920236287"))

# Автоудаление сообщений через 24 часа (как у всего бота)
DELETE_AFTER_HOURS = 24

# Пороги (дней молчания)
OVERDUE_MIN   = 7    # 7–9 дней: запрос менеджеру
SILENCE_MIN   = 10   # 10+ дней: запрос менеджеру
AUTO_STOP_MIN = 15   # 15+ дней: авто-стоп даже для одобренных

# Минимальная сумма долга — игнорировать мелочь
MIN_DEBT = 50_000.0
FULL_PAYMENT_THRESHOLD = float(os.getenv("SHIPMENT_FULL_PAYMENT_THRESHOLD", "1000"))


# ══════════════════════════════════════════════════════════════════════
# Атомарная работа с JSON
# ══════════════════════════════════════════════════════════════════════

def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


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

def load_state() -> Dict[str, Any]:
    state = _load_json(STATE_FILE, {})
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    if state.get("date") != today:
        state = {"date": today, "candidates": {}, "next_id": 1, "saida_sent": False}
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


# ══════════════════════════════════════════════════════════════════════
# Загрузка дебиторки из JSON
# ══════════════════════════════════════════════════════════════════════

def _get_latest_debt_file(manager: str) -> Optional[Path]:
    """Найти последний debt_ext файл для данного менеджера (по номеру в скобках)."""
    files = list(JSON_DIR.glob(f"debt_ext_*_{manager}*"))
    if not files:
        return None

    def _seq(p: Path) -> int:
        m = re.search(r"\((\d+)\)", p.stem)
        return int(m.group(1)) if m else 0

    return max(files, key=_seq)


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
                            await bot.send_message(chat_id=chat_id, text=msg, parse_mode="HTML")
                        except Exception as e:
                            LOG.warning("Ошибка уведомления авто-стоп %s (chat=%s): %s",
                                        client_name, chat_id, e)

        # ── Условная отгрузка: долг оплачен → авто-снятие ────────────
        elif status in ("conditional", "allow_after_payment"):
            current = _get_client_current_state(client_name)
            if current and current["debt"] <= FULL_PAYMENT_THRESHOLD:
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
                            await bot.send_message(chat_id=cid_tg, text=note, parse_mode="HTML")
                        except Exception as e:
                            LOG.warning("Ошибка уведомления conditional-cleared %s: %s",
                                        client_name, e)

        # ── Проверка полной оплаты (авто-стоп или утверждённый стоп) ──
        elif status in ("auto_stopped", "stopped", "block_until_payment"):
            current = _get_client_current_state(client_name)
            if not current:
                continue
            if current["debt"] <= FULL_PAYMENT_THRESHOLD:
                rec["status"] = "pending_clearance"
                paid_in_full_today.append(client_name)
                LOG.info("Полная оплата: %s", client_name)

                kb = InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton(
                            "✅ Разрешить сейчас",
                            callback_data=f"dstop_clear|{client_name[:26]}"
                        ),
                        InlineKeyboardButton(
                            "📉 Разрешить с лимитом",
                            callback_data=f"dstop_limit_clear|{client_name[:26]}"
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            "🔒 Запретить до оплаты",
                            callback_data=f"dstop_keep_until_paid|{client_name[:26]}"
                        ),
                        InlineKeyboardButton(
                            "🚫 Запретить",
                            callback_data=f"dstop_keep|{client_name[:26]}"
                        ),
                    ],
                ])
                if status == "auto_stopped":
                    note = f"⚠️ Ранее нарушил финансовую дисциплину ({rec.get('days_at_stop', '?')} дн. молчания).\n"
                else:
                    note = f"Был на стопе {rec.get('days_at_stop', '?')} дн.\n"
                msg = (
                    f"💰 <b>{client_name}</b> — полностью рассчитался.\n"
                    f"Остаток в 1С: <b>{_fmt(current['debt'])}</b> "
                    f"(полная оплата считается до {_fmt(FULL_PAYMENT_THRESHOLD)}).\n"
                    f"{note}"
                    f"Выбери решение по отгрузке:"
                )
                if admin_id:
                    try:
                        await bot.send_message(chat_id=admin_id, text=msg,
                                               parse_mode="HTML", reply_markup=kb)
                    except Exception as e:
                        LOG.warning("Ошибка уведомления о полной оплате %s: %s", client_name, e)

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
        if rec.get("status") in ("exception", "auto_stopped", "pending_clearance",
                                  "stopped", "conditional", "allow_after_payment",
                                  "block_until_payment", "awaiting_clearance_limit")
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

            level = "10+" if days >= SILENCE_MIN else "7-9"
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
                f"<i>Ответьте сразу: если запрос останется без ответа, "
                f"напоминания будут повторяться каждые 30 минут.</i>"
            )
            kb = InlineKeyboardMarkup([[
                InlineKeyboardButton("✅ Договорились", callback_data=f"dstop_yes|{cid}"),
                InlineKeyboardButton("🚫 Нет, стоп",    callback_data=f"dstop_no|{cid}"),
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
        ]])
        text = (
            f"{header}\n\n"
            f"<b>{c['client']}</b>\n"
            f"Молчит: <b>{c['days_silence']}\u202fдн.</b>  |  "
            f"Долг: <b>{_fmt(c['debt'])}</b>\n\n"
            f"{tone}\n"
            f"Пока вы не ответите, запрос остаётся активным."
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
        "Я буду присылать тебе каждый вечер список клиентов, которых <b>нельзя отгружать</b> завтра.\n\n"

        "📋 <b>Как это работает:</b>\n"
        "1. Каждый вечер в 22:00 ты получаешь список.\n"
        "2. Утром перед оформлением отгрузки — проверь его.\n"
        "3. Если клиент в списке — <b>не отгружай</b>, пока нет разрешения.\n\n"

        "⚡ <b>Что означают значки:</b>\n"
        "🚫 — клиент давно не платит и нарушил договорённость. Стоп до полной оплаты.\n"
        "🔴 — молчит 10 и более дней. Руководитель утвердил стоп.\n"
        "⚡ — молчит 7–9 дней. Руководитель утвердил стоп.\n\n"

        "💡 <b>Если знаешь, что оплата уже пришла, но ещё не проведена в 1С:</b>\n"
        "Нажми кнопку <b>«Оплата получена»</b> под именем клиента в списке.\n"
        "Руководитель сразу получит уведомление и подтвердит отгрузку.\n\n"

        "❗ <b>Важно:</b>\n"
        "Список обновляется каждый день автоматически.\n"
        "Клиент исчезает из списка сам, как только оплата разносится в 1С.\n\n"

        "Если что-то непонятно — напиши руководителю. 🙂"
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


async def send_saida_final(bot) -> None:
    """22:00 — отправить Саиде утверждённый стоп-лист."""
    state = load_state()
    if state.get("saida_sent"):
        return

    # Первый запуск — отправить инструкцию перед списком
    await _send_saida_intro(bot)

    candidates = state.get("candidates", {})
    registry   = load_registry()
    today_str  = datetime.now(TZ).strftime("%d.%m.%Y")

    # Из суточного состояния: утверждённые руководителем
    approved_daily = [
        (cid, c) for cid, c in candidates.items()
        if c.get("admin_approved") is True
    ]

    # Из реестра: авто-стопы и утверждённые стопы (включая сегодняшние)
    auto_stopped = [
        (name, rec) for name, rec in registry.items()
        if rec.get("status") in ("auto_stopped", "stopped", "allow_after_payment", "block_until_payment")
    ]

    total = len(approved_daily) + len(auto_stopped)

    if total == 0:
        try:
            msg = await bot.send_message(
                chat_id=SAIDA_CHAT_ID,
                text=f"✅ <b>Стоп-лист на {today_str} пуст</b>\nВсе клиенты в порядке.",
                parse_mode="HTML"
            )
            _schedule_delete(SAIDA_CHAT_ID, msg.message_id, msg.date.timestamp())
        except Exception as e:
            LOG.warning("Ошибка отправки Саиде (пусто): %s", e)
        state["saida_sent"] = True
        save_state(state)
        return

    header = f"🚫 <b>Не отгружать — {today_str}</b>  ({total} позиций)\n"
    try:
        hdr_msg = await bot.send_message(
            chat_id=SAIDA_CHAT_ID, text=header, parse_mode="HTML"
        )
        _schedule_delete(SAIDA_CHAT_ID, hdr_msg.message_id, hdr_msg.date.timestamp())
    except Exception as e:
        LOG.warning("Ошибка отправки заголовка Саиде: %s", e)

    # Каждый клиент — отдельное сообщение с кнопкой «Оплата получена»
    all_items: List[Tuple[str, int, str, str, str, float]] = []  # name, days, debt_fmt, manager, level, limit
    for name, rec in sorted(auto_stopped, key=lambda x: x[1].get("days_at_stop", 0), reverse=True):
        status = rec.get("status", "auto_stopped")
        level = status if status in ("allow_after_payment", "block_until_payment") else "auto"
        all_items.append((
            name, rec.get("days_at_stop", 0), "?", rec.get("manager", ""), level,
            float(rec.get("shipment_limit") or 0),
        ))
    for cid, c in sorted(approved_daily, key=lambda x: -x[1]["days_silence"]):
        all_items.append((c["client"], c["days_silence"], _fmt(c["debt"]), c["manager"], c["level"], 0.0))

    for item in all_items:
        name, days, debt_str, manager, level, limit = item
        if level == "auto":
            icon = "🚫"
            note = "нарушение фин. дисциплины"
        elif level == "allow_after_payment":
            icon = "⏳"
            if limit > 0:
                note = f"после полной оплаты, лимит новой отгрузки {_fmt(limit)}"
            else:
                note = f"после полной оплаты (до {_fmt(FULL_PAYMENT_THRESHOLD)})"
        elif level == "block_until_payment":
            icon = "🔒"
            note = f"запрет до полной оплаты (до {_fmt(FULL_PAYMENT_THRESHOLD)})"
        elif level == "10+":
            icon = "🔴"
            note = f"долг: {debt_str}"
        else:
            icon = "⚡"
            note = f"долг: {debt_str}"

        text = (
            f"{icon} <b>{name}</b>\n"
            f"Молчит {days}\u202fдн.  ·  {note}  [{manager}]\n"
            f"<i>Не отгружать до разрешения руководителя</i>"
        )
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton(
                "💰 Оплата получена",
                callback_data=f"dstop_paid|{name[:26]}"
            )
        ]])
        try:
            item_msg = await bot.send_message(
                chat_id=SAIDA_CHAT_ID, text=text,
                parse_mode="HTML", reply_markup=kb
            )
            _schedule_delete(SAIDA_CHAT_ID, item_msg.message_id, item_msg.date.timestamp())
        except Exception as e:
            LOG.warning("Ошибка отправки позиции '%s' Саиде: %s", name, e)

    footer = "<i>Список обновляется каждый день. Клиент исчезнет сам после оплаты.</i>"
    try:
        ftr_msg = await bot.send_message(
            chat_id=SAIDA_CHAT_ID, text=footer, parse_mode="HTML"
        )
        _schedule_delete(SAIDA_CHAT_ID, ftr_msg.message_id, ftr_msg.date.timestamp())
    except Exception as e:
        LOG.warning("Ошибка отправки футера Саиде: %s", e)

    state["saida_sent"] = True
    save_state(state)
    LOG.info("Саиде отправлен стоп-лист: %d позиций", total)


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

    return None


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
    if chat_id == admin_id:
        registry = load_registry()
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
                f"(остаток до {_fmt(FULL_PAYMENT_THRESHOLD)}). После оплаты руководитель проверит решение."
            ),
        }[action]
        manager_action_line = {
            "ok": "Отгрузка запрещена.",
            "block_until": (
                f"Отгрузка запрещена до полной оплаты в 1С "
                f"(остаток до {_fmt(FULL_PAYMENT_THRESHOLD)})."
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
        f"Старый долг должен быть закрыт в 1С до {_fmt(FULL_PAYMENT_THRESHOLD)}. "
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
        f"(остаток до {_fmt(FULL_PAYMENT_THRESHOLD)}).\n\n"
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
    kb_saida = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Полная оплата",    callback_data=f"dstop_saida_full|{cid}"),
        InlineKeyboardButton("⚠️ Частичная оплата", callback_data=f"dstop_saida_partial|{cid}"),
    ]])
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

    admin_id = _get_admin_chat_id()
    now_str = datetime.now(TZ).strftime("%H:%M")

    # Руководителю — запрос на разрешение отгрузки
    if admin_id:
        kb_admin = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Разрешить отгрузку", callback_data=f"dstop_admin_allow|{cid}"),
            InlineKeyboardButton("🚫 Отказать",           callback_data=f"dstop_admin_ok|{cid}"),
        ]])
        try:
            await bot.send_message(
                chat_id=admin_id,
                text=(
                    f"💰 <b>Саида подтвердила полную оплату</b>\n\n"
                    f"Клиент: <b>{c['client']}</b>  [{c['manager']}]\n"
                    f"Время: {now_str}\n\n"
                    f"Разрешить отгрузку?"
                ),
                parse_mode="HTML",
                reply_markup=kb_admin
            )
        except Exception as e:
            LOG.warning("Ошибка уведомления руководителя о полной оплате %s: %s", c["client"], e)

    return f"✅ Руководитель уведомлён. Ожидайте решения."


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

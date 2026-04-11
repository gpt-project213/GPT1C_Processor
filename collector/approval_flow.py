#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/approval_flow.py
UX согласования рассылки WhatsApp — менеджер → администратор.

Версия: 1.0.2 (2026-04-11)

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
    # Сортируем по created_at desc, берём первый не-финальный
    for bid in sorted(batches.keys(), reverse=True):
        b = batches[bid]
        if b.get("status") not in ("admin_approved", "cancelled", "expired"):
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
    batch_id = now.strftime("%Y%m%d-%H%M")

    managers_state: Dict[str, Any] = {}
    for manager_name, clients in debtors_by_manager.items():
        # Пропускаем пустые списки и клиентов без менеджера
        if not manager_name or not manager_name.strip():
            logger.warning(
                "create_batch: %d клиент(ов) без manager_name — пропускаем",
                len(clients),
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
            normalized.append({
                "name":              c.get("name", "—"),
                "amount":            float(c.get("amount", 0)),
                "days":              int(c.get("days", 0)),
                "level":             int(c.get("level", 0)),
                "opening":           float(c.get("opening", 0) or 0),
                "debit":             float(c.get("debit", 0) or 0),
                "credit":            float(c.get("credit", 0) or 0),
                "violation_shipment": bool(c.get("violation_shipment", False)),
                "msg_type":          msg_type,
                "reason":            reason,
                "stop_status":       c.get("stop_status", ""),
                "review_action":     c.get("review_action", "client_approval"),
                "phone":             c.get("phone") or c.get("whatsapp") or "",
                "language":          c.get("language", "ru"),
            })

        managers_state[manager_name] = {
            "clients":         normalized,
            "status":          "pending",        # pending | approved_all | rejected_all | manual | timeout
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
    "payment_plan_control": "Контроль графика",
    "soft_reminder":        "Мягкое напоминание",
    "stoplist_reminder":    "Стоп-лист",
}


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

async def _tg_send(chat_id: int, text: str, markup=None) -> Optional[int]:
    """Отправляет Telegram-сообщение; возвращает message_id."""
    if not BOT_TOKEN:
        logger.warning("BOT_TOKEN не задан — Telegram недоступен")
        return None
    import httpx
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload: Dict[str, Any] = {
        "chat_id":    chat_id,
        "text":       text,
        "parse_mode": "HTML",
    }
    if markup:
        payload["reply_markup"] = markup
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload)
        if resp.status_code == 200:
            return resp.json().get("result", {}).get("message_id")
        logger.warning("TG send error %d: %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.error("TG send exception: %s", e)
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


# ─── Manager preview ──────────────────────────────────────────────────────────

def _format_manager_preview_text(
    manager_name: str,
    clients: List[Dict[str, Any]],
    batch_id: str,
) -> str:
    """Формирует текст превью для менеджера."""
    def _fmt(n: float) -> str:
        return f"{n:,.0f}".replace(",", " ")

    lines = [
        f"👋 <b>{manager_name}</b>, добрый день!\n",
        f"Бот предлагает отправить уведомление <b>{len(clients)} клиент(ам)</b>:\n",
    ]
    for i, c in enumerate(clients, 1):
        viol_tag = " ⚠️" if c.get("violation_shipment") else ""
        type_label = _MSG_TYPE_LABELS.get(c.get("msg_type", ""), c.get("msg_type", ""))
        lines.append(
            f"  {i}. <b>{c['name']}</b>{viol_tag}\n"
            f"     Долг: {_fmt(c['amount'])} тг · Просрочка: {c['days']} дн. · L{c['level']}\n"
            f"     Отгрузки: {_fmt(c['debit'])} тг · Оплаты: {_fmt(c['credit'])} тг\n"
            f"     Тип: {type_label}\n"
            f"     Причина: {c.get('reason', '—')}"
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
        text = _format_manager_preview_text(manager_name, clients, batch["batch_id"])
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


def _all_managers_responded(batch: Dict[str, Any]) -> bool:
    for mgr_state in batch["managers"].values():
        if mgr_state.get("status") == "pending":
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
        await _tg_edit(chat_id, message_id, "⚠️ Запрос устарел. Батч не найден.")
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
        text = _format_manager_preview_text(manager_name, clients, batch_id)
        markup = _client_list_keyboard(batch_id, mgr_idx, clients, decisions)
        await _tg_edit(chat_id, message_id, text, markup)
        return True

    elif action == "wa_appr_mgr_manual":
        # Режим ручного выбора — показываем список
        decisions = _build_decisions(mgr_state)
        mgr_state["status"] = "manual"
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

        mgr_state["status"]          = "manual"
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

    total_ok = 0
    total_no = 0
    total_later = 0

    for manager_name, mgr_state in batch["managers"].items():
        status = mgr_state.get("status", "pending")
        approved  = mgr_state.get("approved_names", [])
        rejected  = mgr_state.get("rejected_names", [])
        postponed = mgr_state.get("postponed_names", [])
        clients   = mgr_state.get("clients", [])

        if status == "pending":
            status_label = "⏳ не ответил"
        elif status == "approved_all":
            status_label = f"✅ разрешил всех ({len(approved)})"
        elif status == "rejected_all":
            status_label = f"⛔ отклонил всех ({len(rejected)})"
        elif status == "manual":
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

        total_ok    += len(approved)
        total_no    += len(rejected)
        total_later += len(postponed)

    lines += [
        "",
        f"<b>Итого к отправке: {total_ok}</b> | убрано: {total_no} | отложено: {total_later}",
        "",
        "Нажмите <b>«Утвердить»</b>, чтобы разрешить отправку одобренных клиентов.",
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
            "approved_all": "✅", "manual": "✅",
            "rejected_all": "⛔", "pending": "⏳", "timeout": "⏰",
        }.get(mgr_status, "❓")

        lines.append(f"\n<b>{mgr_icon} {manager_name}</b>")

        def _fmt(n: float) -> str:
            return f"{n:,.0f}".replace(",", " ")

        for c in clients:
            name = c["name"]
            phone = c.get("phone", "") or "—"
            days  = c.get("days", 0)
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
                f"     Долг: {_fmt(c['amount'])} тг · Просрочка: {days} дн. · L{c.get('level', '?')}\n"
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
        [("✅ Разрешить тестовую отправку",    f"wa_appr_adm_ok|{batch_id}")],
        [("👀 Показать список подробнее",      f"wa_appr_adm_view|{batch_id}")],
        [("❌ Отменить",                       f"wa_appr_adm_no|{batch_id}")],
        [("⏸ Отложить",                       f"wa_appr_adm_later|{batch_id}")],
    ])


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

    if action == "wa_appr_adm_ok":
        # Финальное утверждение
        approved_clients = []
        for mgr_name, mgr_state in batch["managers"].items():
            approved_names = set(mgr_state.get("approved_names", []))
            for c in mgr_state["clients"]:
                if c["name"] in approved_names:
                    approved_clients.append({**c, "manager": mgr_name})

        batch["status"]            = "admin_approved"
        batch["admin_status"]      = "approved"
        batch["admin_approved_at"] = now_iso
        batch["approved_clients"]  = approved_clients
        save_batch(batch)

        text = (
            f"✅ <b>Отправка утверждена!</b>\n\n"
            f"Одобрено клиентов: <b>{len(approved_clients)}</b>\n\n"
            + "\n".join(f"  • {c['name']} ({c.get('manager', '—')})" for c in approved_clients)
            + "\n\n"
            f"<b>Следующий шаг:</b> запустить отправку:\n"
            f"<code>python -m collector.collections_engine --send-approved --batch-id {batch_id}</code>\n\n"
            f"<i>Предварительно убедитесь, что WHATSAPP_ENABLED=1 и LIVE_SEND_ALLOWED=1 выставлены в .env</i>"
        )
        await _tg_edit(chat_id, message_id, text)
        logger.info(
            "[%s] Администратор УТВЕРДИЛ отправку: %d клиентов",
            batch_id, len(approved_clients),
        )

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

    elif action == "wa_appr_adm_view":
        # Подробный список всех клиентов: телефон + статус согласования менеджера
        text   = _format_admin_detail_text(batch)
        markup = _admin_keyboard(batch_id)
        await _tg_edit(chat_id, message_id, text, markup)

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
    return batch.get("status") == "admin_approved" and batch.get("admin_status") == "approved"


def get_approved_clients(batch_id: str) -> List[Dict[str, Any]]:
    """Возвращает список одобренных клиентов после admin approve."""
    batch = load_batch(batch_id)
    if not batch or batch.get("status") != "admin_approved":
        return []
    return batch.get("approved_clients", [])


def record_send_results(batch_id: str, results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Stores per-client live-send results back into an approved batch."""
    batch = load_batch(batch_id)
    if not batch:
        return None

    now_iso = datetime.now(tz=TZ).isoformat()
    sent = sum(1 for r in results if r.get("status") == "sent")
    failed = sum(1 for r in results if r.get("status") == "failed")
    skipped = sum(1 for r in results if r.get("status") == "skipped")

    batch["send_results"] = results
    batch["send_completed_at"] = now_iso
    batch["send_summary"] = {
        "sent": sent,
        "failed": failed,
        "skipped": skipped,
        "total": len(results),
    }
    if not results:
        batch["status"] = "send_empty"
    elif failed or skipped:
        batch["status"] = "partially_sent" if sent else "send_failed"
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
        if state.get("status") == "pending"
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
        if batch.get("status") in ("admin_approved", "cancelled", "expired"):
            continue
        try:
            expires = datetime.fromisoformat(batch["expires_at"])
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=TZ)
            if now > expires:
                batch["status"] = "expired"
                count += 1
                logger.info("Батч %s помечен как expired", bid)
        except (KeyError, ValueError):
            pass
    if count:
        _save_batches(batches)
    return count

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collections/collections_engine.py
Главный оркестратор AI-Коллектора долгов.

Версия: 1.4.0 (2026-04-12)

CLI:
  python -m collector.collections_engine --dry-run
  python -m collector.collections_engine --preview
  python -m collector.collections_engine --send-approved --batch-id 20260412-120000-ab12
  python -m collector.collections_engine --send-approved --batch-id 20260412-120000-ab12 --client "ТОО Альфа"
  python -m collector.collections_engine --check-promises

Жёсткие ограничения (из ТЗ):
  - Звонки и сообщения только 09:00–18:00, не в выходные
  - Макс. 1 сообщение в день одному должнику
  - do_not_call=true → только WhatsApp/Telegram
  - Все тексты через DeepSeek
  - Данные только локально
  - При ошибке отправки — уведомить ADMIN_CHAT_ID
  - --dry-run обязателен перед --send
"""

import argparse
import asyncio
import io
import json
import logging
import os
import sys
from datetime import datetime, timedelta

# Windows: принудительно UTF-8 для stdout/stderr
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from zoneinfo import ZoneInfo

# Добавляем корень проекта в sys.path для standalone запуска
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(dotenv_path=_ROOT / ".env", encoding="utf-8-sig", override=False)

TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))
LOGS_DIR = _ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Настройка логирования
_log_file = LOGS_DIR / f"collector_{datetime.now(tz=TZ).strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s, %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

DUPLICATE_DIALOG_HOURS = float(os.getenv("COLLECTOR_DUPLICATE_DIALOG_HOURS", "24"))

from collector.debt_monitor import (
    classify_debtors,
    load_contacts,
    load_latest_debt_json,
    match_client,
)
from collector.collections_db import (
    _DEBT_DATE_PREFIX,
    already_contacted_today,
    get_debt_days_since_first_seen,
    get_pending_promises,
    load_state,
    save_state,
    mark_escalated,
    mark_promise_broken,
    reset_debt_first_seen,
    save_call_result,
    save_promise,
    update_after_contact,
)
from collector.collection_agent import analyze_response, generate_message
from collector.communications import (
    is_allowed_time,
    notify_admin,
    notify_manager,
    send_whatsapp,
    send_telegram,
)
from collector.voice_calls import (
    get_call_result,
    initiate_call,
    is_call_allowed_time,
)

# Менеджеры — читаем из config/managers.json для notify_manager
_MANAGERS_CACHE: Optional[Dict[str, Any]] = None


def _today_str() -> str:
    from datetime import date
    return date.today().isoformat()


def _client_report_date(client: Dict[str, Any], days: int = 0) -> str:
    report_date = str(client.get("report_date") or client.get("period_max") or client.get("_period_max") or "")
    if report_date:
        return report_date
    oldest_unpaid = str(client.get("oldest_unpaid_date") or "")
    if oldest_unpaid and days:
        try:
            return (datetime.fromisoformat(oldest_unpaid).date() + timedelta(days=days)).isoformat()
        except ValueError:
            return ""
    return ""


def _live_send_allowed(reason_prefix: str = "LIVE SEND BLOCKED") -> bool:
    if not is_allowed_time():
        logger.error("%s: outside allowed time window", reason_prefix)
        return False
    _wa_live = os.getenv("WHATSAPP_ENABLED", "0").lower() in ("1", "true", "yes")
    _send_ok = os.getenv("LIVE_SEND_ALLOWED", "0").lower() in ("1", "true", "yes")
    if not _wa_live:
        logger.error("%s: WHATSAPP_ENABLED=0", reason_prefix)
        return False
    if not _send_ok:
        logger.error("%s: LIVE_SEND_ALLOWED is not enabled", reason_prefix)
        return False
    return True


def _load_managers() -> Dict[str, Any]:
    global _MANAGERS_CACHE
    if _MANAGERS_CACHE is None:
        path = _ROOT / "config" / "managers.json"
        try:
            with open(path, encoding="utf-8") as f:
                _MANAGERS_CACHE = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Ошибка чтения managers.json: %s", e)
            _MANAGERS_CACHE = {}
    return _MANAGERS_CACHE


def _get_manager_chat_id(manager_name: str) -> Optional[int]:
    managers = _load_managers()
    chat_id = managers.get(manager_name)
    if chat_id is not None:
        try:
            return int(chat_id)
        except (ValueError, TypeError):
            pass
    return None


def _apply_collector_day_policy(client: Dict[str, Any], name: str, *, use_first_seen: bool) -> Dict[str, Any]:
    """Apply first_seen day inflation only when residual debt age is unavailable."""
    basis = str(client.get("debt_age_basis") or "")
    if basis in ("movements_fifo", "opening_fallback", "no_debt"):
        return dict(client)

    real_days = get_debt_days_since_first_seen(name) if use_first_seen else 0
    days_source = int(client.get("days", 0) or 0)
    real_days = min(real_days, days_source + 7)
    from collector.debt_monitor import _level_for_days
    level = max(int(client.get("level", 0) or 0), _level_for_days(real_days))
    return dict(client, level=level, days=max(days_source, real_days))


def _get_client_manager_from_crm(client_name: str) -> str:
    """Ищет менеджера клиента в clients.json когда контакт не найден в contacts."""
    try:
        from bot.crm_clients import load_clients
        data = load_clients()
        clients_db = data.get("clients", {})
        if client_name in clients_db:
            return clients_db[client_name].get("manager", "")
        c_lower = client_name.lower().strip()
        for key, info in clients_db.items():
            if key.lower().strip() == c_lower:
                return info.get("manager", "")
    except Exception as e:
        logger.warning("_get_client_manager_from_crm error: %s", e)
    return ""


def _load_stop_registry_safe() -> Dict[str, Any]:
    try:
        from bot.debt_stop_control import load_registry as _dsc_registry
        reg = _dsc_registry()
        return reg if isinstance(reg, dict) else {}
    except Exception as e:
        logger.debug("stop-registry load error: %s", e)
        return {}


def _get_stop_record(name: str, registry: Dict[str, Any]) -> Dict[str, Any]:
    rec = registry.get(name)
    if isinstance(rec, dict):
        return rec
    name_l = name.lower().strip()
    for key, value in registry.items():
        if key.lower().strip() == name_l and isinstance(value, dict):
            return value
    return {}


def _flag_enabled(data: Optional[Dict[str, Any]], *keys: str) -> bool:
    if not isinstance(data, dict):
        return False
    return any(bool(data.get(key)) for key in keys)


def _fmt_amount(n: float) -> str:
    return f"{n:,.0f}".replace(",", " ")


def _collector_candidate_decision(
    client: Dict[str, Any],
    contact: Optional[Dict[str, Any]],
    stop_rec: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Business decision for preview shortlist.

    This is intentionally separate from debt_stop_control: stop-list still blocks
    shipment, but it is not a collector skip reason when debt remains open.
    """
    amount = float(client.get("amount", 0) or 0)
    days = int(client.get("days", 0) or 0)
    opening = float(client.get("opening", 0) or 0)
    debit = float(client.get("debit", 0) or 0)
    credit = float(client.get("credit", 0) or 0)
    stop_status = str((stop_rec or {}).get("status") or "")

    if amount <= 0:
        return {"action": "skip", "reason": "долг закрыт"}

    try:
        from collector.payment_hold import get_hold_for_client
        hold = get_hold_for_client(str(client.get("name") or client.get("client") or ""))
    except Exception:
        hold = None
    if hold:
        return {
            "action": "skip",
            "reason": "Саида подтвердила оплату, ждём разноски в 1С",
            "payment_hold_status": hold.get("status", ""),
        }

    if _flag_enabled(contact, "do_not_notify", "do_not_write", "do_not_contact", "collector_skip"):
        return {"action": "skip", "reason": "ручной запрет уведомления в CRM"}
    if _flag_enabled(stop_rec, "do_not_notify", "do_not_write", "do_not_contact", "collector_skip"):
        return {"action": "skip", "reason": "ручной запрет уведомления в stop-registry"}

    if stop_status in ("stopped", "auto_stopped"):
        return {
            "action": "client_approval",
            "msg_type": "stoplist_reminder",
            "reason": f"{stop_status}: долг {_fmt_amount(amount)} тг не закрыт",
            "stop_status": stop_status,
        }

    if stop_status in ("pending_clearance", "conditional"):
        return {
            "action": "manager_review",
            "msg_type": "payment_plan_control",
            "reason": f"{stop_status}: нужен ручной контроль перед сообщением клиенту",
            "stop_status": stop_status,
        }

    if credit > 0 and amount < 100_000 and amount <= credit * 0.10:
        return {
            "action": "skip",
            "reason": (
                f"малый остаток после крупной оплаты: "
                f"оплата {_fmt_amount(credit)} тг, остаток {_fmt_amount(amount)} тг"
            ),
        }

    # Healthy payer heuristic: paying at least as much as current shipments and
    # keeping only a small tail should not receive a hard collector message.
    base_debt = opening if opening > 0 else amount
    small_tail = base_debt > 0 and amount <= base_debt * 0.25
    if debit > 0 and credit >= debit and small_tail:
        return {
            "action": "skip",
            "reason": (
                f"платёжная дисциплина выглядит нормальной: "
                f"отгрузки {_fmt_amount(debit)} тг, оплаты {_fmt_amount(credit)} тг, "
                f"остаток {_fmt_amount(amount)} тг"
            ),
        }

    if credit > 0 and debit == 0:
        pct = (credit / base_debt * 100) if base_debt > 0 else 0
        if pct < 20 or amount >= 100_000 or days >= 10:
            return {
                "action": "client_approval",
                "msg_type": "payment_plan_control" if amount >= 500_000 else "soft_reminder",
                "reason": (
                    f"есть оплата {_fmt_amount(credit)} тг ({pct:.0f}% от долга), "
                    f"но остаток {_fmt_amount(amount)} тг не закрыт"
                ),
            }
        return {"action": "skip", "reason": "есть существенная оплата, мягкий контроль пока не нужен"}

    if debit > 0:
        if credit > 0:
            return {
                "action": "client_approval",
                "msg_type": "payment_plan_control" if credit >= debit * 0.85 else "strict_reminder",
                "reason": (
                    f"клиент продолжает движение при долге: "
                    f"отгрузки {_fmt_amount(debit)} тг, оплаты {_fmt_amount(credit)} тг, "
                    f"остаток {_fmt_amount(amount)} тг"
                ),
            }
        return {
            "action": "manager_review",
            "msg_type": "strict_reminder",
            "reason": f"есть отгрузки {_fmt_amount(debit)} тг при незакрытом долге",
        }

    return {
        "action": "client_approval",
        "msg_type": "strict_reminder",
        "reason": f"{days}д просрочки, оплат и отгрузок нет, долг {_fmt_amount(amount)} тг",
    }


_NO_PHONE_WARN_PREFIX = "__no_phone_warn__"


async def _warn_manager_no_phone(
    client_name: str,
    level: int,
    amount: float,
    days: int,
    manager_name: str,
    manager_chat_id: Optional[int],
    dry_run: bool,
) -> None:
    """
    Предупреждает менеджера: у проблемного клиента нет телефона → добавь номер.
    При повторном предупреждении (warn_count >= 2) — эскалация руководителю.
    """
    from collector.collections_db import load_state, save_state

    state = load_state()
    key = f"{_NO_PHONE_WARN_PREFIX}{client_name}"
    today = _today_str()

    warn_info = state.get(key, {})
    last_warned = warn_info.get("last_warned", "")
    warn_count = warn_info.get("warn_count", 0)
    first_warned = warn_info.get("first_warned", today)

    # Уже предупреждали сегодня — не дублировать
    if last_warned == today:
        logger.info("[%s] менеджер уже предупреждён сегодня — пропуск", client_name)
        return

    warn_count += 1
    state[key] = {
        "first_warned": first_warned,
        "last_warned": today,
        "warn_count": warn_count,
        "manager": manager_name,
    }

    amount_str = f"{int(amount):,}".replace(",", " ")

    warn_text = (
        f"📵 <b>Проблемный клиент без контакта</b>\n\n"
        f"👤 <b>{client_name}</b>\n"
        f"💰 Долг: {amount_str} тг  |  Молчит: {days} дн.\n"
        f"🔴 Уровень риска: {level}/5\n\n"
        f"Телефон клиента <b>отсутствует в базе</b> — "
        f"сообщение не может быть отправлено.\n\n"
        f"Внесите номер командой:\n"
        f"<code>/phone {client_name} 87XXXXXXXXX</code>\n\n"
        f"⚠️ Если номер не будет внесён, руководитель получит "
        f"уведомление об этом клиенте."
    )

    if dry_run:
        logger.info(
            "[%s] DRY-RUN: предупреждение менеджеру %s (warn #%d)",
            client_name, manager_name or "?", warn_count,
        )
    elif manager_chat_id:
        await notify_manager(manager_chat_id, warn_text)
        logger.info(
            "[%s] предупреждение отправлено менеджеру %s (warn #%d)",
            client_name, manager_name, warn_count,
        )
    else:
        logger.warning(
            "[%s] нет chat_id для менеджера %s — предупреждение не отправлено",
            client_name, manager_name or "?",
        )

    # 2+ предупреждения → телефон до сих пор не внесён → эскалация
    if warn_count >= 2 and not dry_run:
        admin_text = (
            f"📵 <b>Нет телефона: менеджер {manager_name or '?'} игнорирует</b>\n\n"
            f"👤 {client_name}\n"
            f"💰 Долг: {amount_str} тг  |  Молчит: {days} дн.\n"
            f"Предупреждений отправлено: {warn_count} "
            f"(первое: {first_warned})\n\n"
            f"Телефон клиента до сих пор не добавлен в базу."
        )
        await notify_admin(admin_text)
        logger.warning(
            "[%s] эскалация руководителю — телефон не внесён (warn_count=%d)",
            client_name, warn_count,
        )

    if not dry_run:
        save_state(state)


def daily_summary(processed: List[Dict], total_classified: int = 0, dry_run: bool = False) -> str:
    """Формирует ежедневную сводку работы коллектора."""
    total = len(processed)
    sent = sum(1 for r in processed if r.get("sent"))
    promised = [r for r in processed if r.get("promise_received")]
    broken = [r for r in processed if r.get("promise_broken")]
    no_contact = [r for r in processed if r.get("no_contacts")]
    escalated = [r for r in processed if r.get("escalated")]
    skipped = total_classified - total if total_classified > total else 0

    mode_label = "🔇 DRY-RUN (сообщения НЕ отправлялись)" if dry_run else "✅ LIVE"
    lines = [
        f"📊 <b>AI Коллектор — ежедневная сводка</b> {mode_label}",
        f"",
        f"Классифицировано должников 1–5: {total_classified}",
        f"Обработано коллектором: {total}",
        f"Пропущено фильтрами: {skipped}",
        f"Отправлено сообщений: {sent}",
    ]
    if promised:
        lines.append(f"Обещали оплату: {len(promised)}")
        for r in promised[:5]:
            lines.append(f"  • {r['name']} → {r.get('promise_date', '?')}")
    if broken:
        lines.append(f"Нарушили обещание: {len(broken)}")
        for r in broken[:5]:
            lines.append(f"  • {r['name']}")
    if no_contact:
        lines.append(f"📵 Нет контактов (телефона/TG) — предупреждены менеджеры: {len(no_contact)}")
    if escalated:
        lines.append(f"Эскалировано директору: {len(escalated)}")
    return "\n".join(lines)


async def _process_single(
    client: Dict[str, Any],
    contact: Dict[str, Any],
    dry_run: bool,
    msg_type: str = "",
) -> Dict[str, Any]:
    """Обрабатывает одного должника: генерация + диалог с менеджером + звонок."""
    name = client["name"]
    display_name = contact.get("display_name") or name
    level = client["level"]
    days = client["days"]
    amount = client["amount"]
    language = contact.get("language", "ru")
    phone = contact.get("whatsapp") or contact.get("phone", "")
    tg_id = contact.get("telegram_id")
    manager_name = contact.get("manager", "")
    do_not_call = contact.get("do_not_call", False)
    manager_chat_id = _get_manager_chat_id(manager_name) if manager_name else None
    report_date = _client_report_date(client, days)

    result: Dict[str, Any] = {
        "name": name,
        "level": level,
        "days": days,
        "amount": amount,
        "sent": False,
        "promise_received": False,
        "promise_broken": False,
        "escalated": False,
        "no_contacts": False,
    }

    from collector.dialog_store import load_dialogs as _load_dialogs_state

    for _dialog in _load_dialogs_state().values():
        if not isinstance(_dialog, dict):
            continue
        if _dialog.get("client_name") != name:
            continue
        _state = _dialog.get("state", "")
        if _state not in ("CONFIRMED", "DONE"):
            logger.info("[%s] уже есть активный диалог по клиенту (state=%s) — дубль пропущен", name, _state)
            return result
        _ts_raw = _dialog.get("last_reminded") or _dialog.get("created")
        if not _ts_raw:
            logger.info("[%s] уже есть завершённый диалог по клиенту (state=%s) — дубль пропущен", name, _state)
            result["sent"] = True
            result["via_dialog"] = True
            return result
        try:
            _ts = datetime.fromisoformat(_ts_raw)
            if _ts.tzinfo is None:
                _ts = _ts.replace(tzinfo=TZ)
        except (TypeError, ValueError):
            logger.info("[%s] уже есть завершённый диалог по клиенту (state=%s) — дубль пропущен", name, _state)
            result["sent"] = True
            result["via_dialog"] = True
            return result
        if (datetime.now(TZ) - _ts).total_seconds() < DUPLICATE_DIALOG_HOURS * 3600:
            logger.info("[%s] клиент уже обработан %.1f ч назад (state=%s) — дубль пропущен", name, (datetime.now(TZ) - _ts).total_seconds() / 3600, _state)
            result["sent"] = True
            result["via_dialog"] = True
            return result

    # Проверяем занятость менеджера ДО дорогого API-вызова (BUG-B2 fix)
    if manager_chat_id:
        from collector.dialog_store import get_dialog as _get_dialog_state
        _dialog_pre = _get_dialog_state(manager_chat_id)

        if _dialog_pre and _dialog_pre.get("client_name") == name:
            _state_pre = _dialog_pre.get("state", "")
            if _state_pre in ("CONFIRMED", "DONE"):
                # WhatsApp уже отправлен через диалог
                result["sent"] = True
                result["via_dialog"] = True
                return result
            else:
                # Диалог в процессе — ждём менеджера
                logger.info(
                    "[%s] диалог в состоянии %s — ожидаем менеджера", name, _state_pre
                )
                return result

        elif _dialog_pre and _dialog_pre.get("state") not in ("CONFIRMED", "DONE", None):
            # FIX-2: менеджер занят другим диалогом — пропускаем клиента до следующего цикла.
            # Direct send убран: он обходил manager approval для всех остальных клиентов менеджера.
            logger.info(
                "[%s] менеджер %s занят диалогом по %s — пропуск до следующего запуска",
                name, manager_name, _dialog_pre.get("client_name"),
            )
            return result

    # Генерируем текст сообщения (только когда реально нужен)
    text = generate_message(
        client_name=display_name,
        debt_amount=amount,
        days_overdue=days,
        level=level,
        language=language,
        manager_name=manager_name,
        msg_type=msg_type,
        report_date=report_date,
    )
    logger.info("[%s] level=%d days=%d | текст: %s...", name, level, days, text[:60])

    if dry_run:
        result["sent"] = True
        result["message_text"] = text
        return result

    # OP-4 guard: прямая отправка разрешена только если менеджер известен.
    # Без manager_name невозможно отследить, кому принадлежит клиент.
    if not manager_name:
        logger.warning(
            "[%s] direct send запрещён: manager_name пустой — клиент пропущен (OP-4)",
            name,
        )
        return result

    # Если нет chat_id менеджера — прямая отправка (старый путь)
    if not manager_chat_id:
        wa_ok = False
        if phone:
            wa_ok = send_whatsapp(phone, text)
        tg_ok = False
        if tg_id:
            try:
                tg_ok = await send_telegram(int(tg_id), text)
            except (ValueError, TypeError):
                logger.error("[%s] некорректный tg_id=%s", name, tg_id)
                tg_ok = False
        sent = wa_ok or tg_ok
        if sent:
            update_after_contact(name, "whatsapp" if wa_ok else "telegram", level, text)
            result["sent"] = True
            # Регистрируем клиентский диалог если WhatsApp отправлен
            if wa_ok and phone:
                try:
                    from collector.client_dialog import start_client_dialog
                    phone_clean = "".join(c for c in phone if c.isdigit())
                    await start_client_dialog(
                        phone=phone_clean,
                        client_name=name,
                        manager_name=manager_name,
                        manager_chat_id=0,
                        level=level,
                        days=days,
                        amount=amount,
                        message_text=text,
                        report_date=report_date,
                    )
                except Exception as e:
                    logger.error("[%s] start_client_dialog ошибка: %s", name, e)
    else:
        # Запускаем новый диалог (предпроверка выше прошла — менеджер свободен)
        from collector.manager_dialog import start_dialog as _start_manager_dialog
        await _start_manager_dialog(client, contact, manager_name, manager_chat_id)
        result["dialog_started"] = True
        logger.info("[%s] диалог запущен с менеджером %s", name, manager_name)
        return result

    # Звонок только когда диалог подтверждён (state CONFIRMED) или прямая отправка
    if not do_not_call and phone and is_call_allowed_time():
        call_res = initiate_call(
            phone=phone,
            client_name=name,
            debt_amount=amount,
            days_overdue=days,
            level=level,
            language=language,
        )
        if call_res.get("call_id"):
            result["call_id"] = call_res["call_id"]

    # Level 5 → эскалация директору
    if level >= 5:
        await notify_admin(
            f"Клиент <b>{name}</b> — просрочка {days} дней, сумма {amount:,.0f} тенге.\n"
            f"Уровень 5: требуется ручное вмешательство."
        )
        mark_escalated(name)
        result["escalated"] = True

    return result


async def run(dry_run: bool = False, single_client: Optional[str] = None) -> None:
    """Основной цикл обработки должников."""
    if not is_allowed_time() and not dry_run:
        logger.info("Вне рабочего времени (09:00–18:00, пн–пт) — пропуск")
        return

    # ── SAFEGUARD: двойной замок перед любой live-отправкой ──────────────────
    # Оба флага должны быть явно установлены в .env.
    # WHATSAPP_ENABLED=1 — разрешает технически.
    # LIVE_SEND_ALLOWED=1 — явное подтверждение руководителя «да, отправляй».
    # Без обоих флагов ни один WhatsApp не уйдёт, даже если код дойдёт сюда.
    if not dry_run:
        _wa_live = os.getenv("WHATSAPP_ENABLED", "0").lower() in ("1", "true", "yes")
        _send_ok = os.getenv("LIVE_SEND_ALLOWED", "0").lower() in ("1", "true", "yes")
        if not _wa_live:
            logger.error(
                "LIVE SEND BLOCKED: WHATSAPP_ENABLED=0 в .env. "
                "Для реальной отправки нужны оба флага: WHATSAPP_ENABLED=1 И LIVE_SEND_ALLOWED=1."
            )
            return
        if not _send_ok:
            logger.error(
                "LIVE SEND BLOCKED: LIVE_SEND_ALLOWED не установлен (0 или отсутствует). "
                "Это явная защита от случайной отправки. "
                "Установите LIVE_SEND_ALLOWED=1 только после проверки dry-run и "
                "личного решения руководителя."
            )
            return
        logger.info(
            "LIVE SEND ALLOWED: WHATSAPP_ENABLED=1 + LIVE_SEND_ALLOWED=1 — "
            "отправка явно разрешена руководителем."
        )
    # ── /SAFEGUARD ────────────────────────────────────────────────────────────

    debt_data = load_latest_debt_json()
    if not debt_data:
        logger.warning("Нет данных дебиторки — завершаем")
        await notify_admin("⚠️ AI Коллектор: нет данных дебиторки для обработки")
        return

    debtors = classify_debtors(debt_data)
    try:
        from collector.payment_hold import sync_holds_with_debtors
        sync_holds_with_debtors(debtors)
    except Exception as _e:
        logger.debug("payment hold sync skipped: %s", _e)
    # CRM: объединяем clients.json + debtors_contacts.json для поиска телефонов
    try:
        from bot.crm_clients import load_contacts_compat as _crm_contacts
        contacts = _crm_contacts()
    except Exception:
        contacts = load_contacts()
    processed: List[Dict] = []

    # Уведомления о нарушениях: отгрузка при наличии долга — вина менеджера
    # BUG-C2/C3 fix: батчинг — одна загрузка state, один save, одно сообщение
    # на менеджера и одна сводка для админа (вместо N отдельных сообщений).
    if not dry_run:
        _vstate = load_state()  # загружаем state ОДИН раз до цикла

        # Группируем ненотифицированные нарушения по менеджеру
        _violations_by_mgr: Dict[str, List[Dict[str, Any]]] = {}
        _violations_no_mgr: List[Dict[str, Any]] = []
        for _client in debtors:
            if not _client.get("violation_shipment"):
                continue
            _vkey = f"__violation_notified__{_client['name']}"
            if _vstate.get(_vkey, {}).get("date") == _today_str():
                continue  # уже уведомляли сегодня
            _vmgr = _client.get("manager", "")
            if _vmgr:
                _violations_by_mgr.setdefault(_vmgr, []).append(_client)
            else:
                _violations_no_mgr.append(_client)

        # Отправляем одно сообщение каждому менеджеру со списком его нарушений
        from collector.communications import send_telegram as _send_tg
        for _vmgr, _vclients in _violations_by_mgr.items():
            _vmgr_chat_id = _get_manager_chat_id(_vmgr)
            if _vmgr_chat_id:
                _lines = [
                    f"🚨 <b>Нарушение кредитной политики — {len(_vclients)} клиент(ов)</b>\n",
                    "Отгрузка произведена при непогашенной задолженности:\n",
                ]
                for _vc in _vclients:
                    _lines.append(
                        f"  • <b>{_vc['name']}</b> — {_vc['amount']:,.0f} тг ({_vc['days']} дн.)"
                    )
                _lines.append("\nПрошу урегулировать ситуацию с каждым клиентом лично.")
                await _send_tg(_vmgr_chat_id, "\n".join(_lines))

        # Одна сводка для админа по всем нарушениям
        _all_violations = [
            _c for _vl in _violations_by_mgr.values() for _c in _vl
        ] + _violations_no_mgr
        if _all_violations:
            _admin_lines = [
                f"🚨 <b>Нарушения кредитной политики — {len(_all_violations)} случай</b>\n",
            ]
            for _vc in _all_violations:
                _admin_lines.append(
                    f"  • {_vc.get('manager', '?')}: <b>{_vc['name']}</b>"
                    f" — {_vc['amount']:,.0f} тг ({_vc['days']} дн.)"
                )
            await notify_admin("\n".join(_admin_lines))

            # Помечаем всех сразу — одна запись state
            for _vc in _all_violations:
                _vstate[f"__violation_notified__{_vc['name']}"] = {"date": _today_str()}
                logger.warning(
                    "Нарушение — %s (менеджер=%s, сумма=%.0f тг, дней=%d)",
                    _vc["name"], _vc.get("manager", ""),
                    _vc.get("amount", 0), _vc.get("days", 0),
                )
            save_state(_vstate)  # ОДИН save после всего цикла

    # Сбрасываем счётчик дней для клиентов, чей долг погашен:
    # если клиент есть в нашем state (был должником), но пропал из текущих
    # debt-файлов — значит долг закрыт. Без сброса уровень давления будет
    # завышен при следующем появлении клиента в дебиторке.
    _current_names = {
        (c.get("name") or c.get("client") or "").strip()
        for c in debt_data.get("clients", [])
        if (c.get("name") or c.get("client") or "").strip()
    }
    for _key in list(load_state().keys()):
        if _key.startswith(_DEBT_DATE_PREFIX):
            _client_name = _key[len(_DEBT_DATE_PREFIX):]
            if _client_name not in _current_names:
                reset_debt_first_seen(_client_name)
                logger.info("Долг погашен — сброс счётчика дней: %s", _client_name)

    for client in debtors:
        name = client["name"]

        client = _apply_collector_day_policy(client, name, use_first_seen=not dry_run)
        level = int(client.get("level", 0) or 0)

        # Фильтр по одному клиенту если задан
        if single_client and single_client.lower() not in name.lower():
            continue

        # Level 0 — пропускаем
        if level == 0:
            continue

        # PHASE 4: stopped/auto_stopped блокирует отгрузки, НЕ уведомления коллектора.
        # Явный collector-блок — только через do_not_notify / collector_skip в registry.
        # _bypass_active_guard=True синхронизирует run() с _collector_candidate_decision(),
        # которая для stopped/auto_stopped возвращает client_approval до проверки debit/credit.
        _dsc_rec = None
        _msg_type = ""
        _bypass_active_guard = False
        try:
            from bot.debt_stop_control import load_registry as _dsc_registry
            _dsc_rec = _dsc_registry().get(name)
            if _dsc_rec:
                if _flag_enabled(_dsc_rec, "do_not_notify", "do_not_write",
                                 "do_not_contact", "collector_skip"):
                    logger.info("[%s] collector_skip в stop-registry — пропуск", name)
                    continue
                _shipment_status = _dsc_rec.get("status", "")
                if _shipment_status in ("stopped", "auto_stopped"):
                    _msg_type = "stoplist_reminder"
                    _bypass_active_guard = True
                    logger.info(
                        "[%s] shipment_status=%s — отгрузки заблокированы, "
                        "уведомление разрешено (долг не закрыт)",
                        name, _shipment_status,
                    )
                elif _shipment_status in ("pending_clearance", "conditional"):
                    _msg_type = "payment_plan_control"
                    _bypass_active_guard = True
        except Exception as _e:
            logger.debug("Ошибка проверки stop-registry: %s", _e)

        # FIX-1: активные клиенты (покупают ИЛИ платят) — коллектор не трогает.
        # Исключение: клиенты на shipment-стопе — active guard не применяется,
        # т.к. _collector_candidate_decision() (preview path) тоже его не применяет.
        _debit_val  = client.get("debit", 0.0) or 0.0
        _credit_val = client.get("credit", 0.0) or 0.0
        if not _bypass_active_guard and (_debit_val > 0 or _credit_val > 0):
            logger.info("[%s] пропуск — клиент активен (debit=%.0f, credit=%.0f)",
                        name, _debit_val, _credit_val)
            continue
        if client.get("amount", 0) <= 0:
            logger.info("[%s] пропуск — долг погашен или отрицательный (amount=%.0f)",
                        name, client.get("amount", 0))
            continue
        try:
            from collector.payment_hold import get_hold_for_client
            _payment_hold = get_hold_for_client(name)
        except Exception:
            _payment_hold = None
        if _payment_hold:
            logger.info("[%s] пропуск — Саида подтвердила оплату, ждём разноски в 1С", name)
            continue

        # Уже контактировали сегодня — пропускаем
        if not dry_run and already_contacted_today(name):
            logger.info("[%s] уже обработан сегодня — пропуск", name)
            continue

        # Ищем контакты (из CRM — clients.json + debtors_contacts.json)
        contact = match_client(name, contacts)

        # Случай 1: клиент вообще не найден в базе контактов
        if not contact:
            manager_name = _get_client_manager_from_crm(name)
            manager_chat_id = _get_manager_chat_id(manager_name) if manager_name else None
            await _warn_manager_no_phone(
                name, level, client["amount"], client["days"],
                manager_name, manager_chat_id, dry_run,
            )
            processed.append({"name": name, "level": level, "no_contacts": True,
                               "sent": False, "promise_received": False,
                               "promise_broken": False, "escalated": False})
            continue

        # Случай 2: контакт найден, но нет ни WhatsApp, ни Telegram
        _phone = (contact.get("whatsapp") or contact.get("phone", "")).strip()
        _tg_id = str(contact.get("telegram_id") or "").strip()
        if not _phone and not _tg_id:
            manager_name = contact.get("manager", "") or _get_client_manager_from_crm(name)
            manager_chat_id = _get_manager_chat_id(manager_name) if manager_name else None
            await _warn_manager_no_phone(
                name, level, client["amount"], client["days"],
                manager_name, manager_chat_id, dry_run,
            )
            processed.append({"name": name, "level": level, "no_contacts": True,
                               "sent": False, "promise_received": False,
                               "promise_broken": False, "escalated": False})
            continue

        result = await _process_single(client, contact, dry_run, msg_type=_msg_type)
        processed.append(result)

    # Итоговая сводка → администратору
    summary = daily_summary(processed, total_classified=len(debtors), dry_run=dry_run)
    logger.info("Сводка:\n%s", summary)
    if not dry_run:
        await notify_admin(summary)


async def check_promises() -> None:
    """Проверяет просроченные обещания оплаты и уведомляет менеджеров."""
    pending = get_pending_promises()
    if not pending:
        logger.info("Просроченных обещаний нет")
        return

    logger.info("Просроченных обещаний: %d", len(pending))
    try:
        from bot.crm_clients import load_contacts_compat as _crm_contacts
        contacts = _crm_contacts()
    except Exception:
        contacts = load_contacts()

    for item in pending:
        name = item["name"]
        promise_date = item["promise_date"]
        logger.warning("[%s] обещал оплатить %s — не оплатил", name, promise_date)
        mark_promise_broken(name)

        # Уведомляем менеджера клиента
        contact = match_client(name, contacts)
        manager_name = contact.get("manager", "") if contact else ""
        if manager_name:
            chat_id = _get_manager_chat_id(manager_name)
            if chat_id:
                await notify_manager(
                    chat_id,
                    f"❌ Клиент <b>{name}</b> обещал оплатить <b>{promise_date}</b>, "
                    f"но оплата не поступила.",
                )


async def _send_approved_client(client: Dict[str, Any]) -> Dict[str, Any]:
    name = str(client.get("name") or "").strip()
    manager_name = str(client.get("manager") or "").strip()
    phone = str(client.get("phone") or client.get("whatsapp") or "").strip()

    result = {
        "name": name,
        "manager": manager_name,
        "phone": phone,
        "status": "skipped",
        "reason": "",
    }

    if not name:
        result["reason"] = "missing client name"
        return result
    try:
        from collector.payment_hold import get_hold_for_client
        hold = get_hold_for_client(name)
    except Exception:
        hold = None
    if hold:
        result["reason"] = "Saida confirmed payment, waiting for 1C posting"
        return result
    if not phone:
        result["reason"] = "missing WhatsApp phone in approved batch"
        return result
    from collector.approval_flow import validate_production_phone
    phone_valid, phone_issue = validate_production_phone(phone, name)
    if not phone_valid:
        result["reason"] = phone_issue
        return result
    if already_contacted_today(name):
        result["reason"] = "already contacted today"
        return result

    amount = float(client.get("amount", 0) or 0)
    days = int(client.get("days", 0) or 0)
    level = int(client.get("level", 0) or 0)
    language = str(client.get("language") or "ru")
    msg_type = str(client.get("msg_type") or "")
    report_date = _client_report_date(client, days)

    text = generate_message(
        client_name=name,
        debt_amount=amount,
        days_overdue=days,
        level=level,
        language=language,
        manager_name=manager_name,
        msg_type=msg_type,
        report_date=report_date,
    )

    if not send_whatsapp(phone, text):
        result["status"] = "failed"
        result["reason"] = "send_whatsapp returned false"
        return result

    update_after_contact(name, "whatsapp", level, text)
    result["status"] = "sent"
    result["reason"] = "sent from admin-approved batch"

    try:
        from collector.client_dialog import start_client_dialog
        phone_clean = "".join(c for c in phone if c.isdigit())
        await start_client_dialog(
            phone=phone_clean,
            client_name=name,
            manager_name=manager_name,
            manager_chat_id=_get_manager_chat_id(manager_name) or 0,
            level=level,
            days=days,
            amount=amount,
            message_text=text,
            report_date=report_date,
        )
    except Exception as e:
        logger.error("[%s] start_client_dialog after approved send error: %s", name, e)

    return result


async def send_approved_batch(batch_id: str, single_client: Optional[str] = None) -> List[Dict[str, Any]]:
    """Sends WhatsApp only to clients stored in an admin-approved batch."""
    from collector.approval_flow import get_approved_clients, is_ready_for_send, record_send_results

    if not is_ready_for_send(batch_id):
        logger.error("send-approved blocked: batch %s is not admin-approved", batch_id)
        return []
    if not _live_send_allowed("SEND-APPROVED BLOCKED"):
        return []

    clients = get_approved_clients(batch_id)
    if single_client:
        needle = single_client.lower()
        clients = [c for c in clients if needle in str(c.get("name", "")).lower()]

    logger.info(
        "send-approved: batch=%s candidates=%d%s",
        batch_id, len(clients),
        f" client_filter={single_client!r}" if single_client else "",
    )

    results = []
    for client in clients:
        results.append(await _send_approved_client(client))

    record_send_results(batch_id, results)
    logger.info("send-approved: batch=%s results=%s", batch_id, results)
    return results


async def run_approval_preview(single_client: Optional[str] = None) -> Optional[str]:
    """Формирует список кандидатов и отправляет менеджерам на согласование.

    Это первый шаг перед реальной отправкой:
    1. Запускает dry-run → получает список кандидатов
    2. Группирует по менеджерам
    3. Создаёт батч согласования (approval_flow)
    4. Отправляет каждому менеджеру его список через Telegram
    5. После ответа всех менеджеров — отправляет сводку администратору

    Returns:
        batch_id если батч создан, None если кандидатов нет.

    ВАЖНО: WhatsApp не отправляется. Это только UX согласования.
    Реальная отправка — отдельный шаг после admin approve + WHATSAPP_ENABLED=1.
    """
    from collector.approval_flow import create_batch, save_batch, send_manager_previews

    debt_data = load_latest_debt_json()
    if not debt_data:
        logger.warning("run_approval_preview: нет данных дебиторки")
        return None

    debtors = classify_debtors(debt_data)
    try:
        from collector.payment_hold import sync_holds_with_debtors
        sync_holds_with_debtors(debtors)
    except Exception as _e:
        logger.debug("payment hold sync skipped: %s", _e)
    try:
        from bot.crm_clients import load_contacts_compat as _crm_contacts
        contacts = _crm_contacts()
    except Exception:
        contacts = load_contacts()

    # Preview shortlist: stop-list and debit/credit are not absolute skips here.
    # They become explicit business reasons shown to manager/admin.
    debtors_by_manager: Dict[str, List[Dict]] = {}
    stop_registry = _load_stop_registry_safe()

    for client in debtors:
        name = client["name"]

        client = _apply_collector_day_policy(client, name, use_first_seen=True)
        level = int(client.get("level", 0) or 0)

        if single_client and single_client.lower() not in name.lower():
            continue
        if level == 0:
            continue

        if client.get("amount", 0) <= 0:
            continue

        # Уже контактировали сегодня
        if already_contacted_today(name):
            continue

        # Ищем контакты
        contact = match_client(name, contacts)
        stop_rec = _get_stop_record(name, stop_registry)
        decision = _collector_candidate_decision(client, contact, stop_rec)
        if decision.get("action") == "skip":
            logger.info(
                "run_approval_preview: [%s] skip — %s",
                name, decision.get("reason", ""),
            )
            continue

        _phone = ((contact or {}).get("whatsapp") or (contact or {}).get("phone", "")).strip()
        _tg_id = str((contact or {}).get("telegram_id") or "").strip()
        if not _phone and not _tg_id:
            manager_name = ""
            if contact:
                manager_name = (contact.get("manager", "") or "").strip()
            if not manager_name:
                manager_name = str((stop_rec or {}).get("manager") or "").strip()
            if not manager_name:
                manager_name = _get_client_manager_from_crm(name)
            manager_chat_id = _get_manager_chat_id(manager_name) if manager_name else None
            await _warn_manager_no_phone(
                name, level, client["amount"], client["days"],
                manager_name, manager_chat_id, dry_run=False,
            )
            logger.info(
                "run_approval_preview: [%s] no phone/telegram — manager warning requested",
                name,
            )
            continue

        manager_name = (contact or {}).get("manager", "").strip()
        if not manager_name:
            manager_name = str((stop_rec or {}).get("manager") or "").strip()
        if not manager_name:
            manager_name = _get_client_manager_from_crm(name)
        if not manager_name:
            logger.warning(
                "run_approval_preview: [%s] нет manager_name — пропуск (требует admin approval). "
                "Добавьте менеджера в debtors_contacts.json или CRM.",
                name,
            )
            continue

        # HIGH-3 fix: менеджер без chat_id не может получить preview → batch зависнет.
        # Такие клиенты не включаются в batch; лог виден в precheck.
        _preview_chat_id = _get_manager_chat_id(manager_name)
        if not _preview_chat_id:
            logger.info(
                "run_approval_preview: [%s] пропуск — missing_manager_chat_id "
                "для менеджера %s",
                name, manager_name,
            )
            continue

        debtors_by_manager.setdefault(manager_name, []).append({
            "name":               name,
            "amount":             client.get("amount", 0),
            "days":               client.get("days", 0),
            "level":              level,
            "opening":            client.get("opening", 0) or 0,
            "debit":              client.get("debit", 0) or 0,
            "credit":             client.get("credit", 0) or 0,
            "payment_silence_days": client.get("payment_silence_days"),
            "report_date":        client.get("report_date", ""),
            "oldest_unpaid_date": client.get("oldest_unpaid_date"),
            "unpaid_parts":       client.get("unpaid_parts", []),
            "debt_age_basis":     client.get("debt_age_basis", ""),
            "debt_age_confidence": client.get("debt_age_confidence", ""),
            "active_turnover":    client.get("active_turnover", False),
            "violation_shipment": client.get("violation_shipment", False),
            "phone":              _phone,
            "language":           (contact or {}).get("language", "ru"),
            "msg_type":           decision.get("msg_type"),
            "reason":             decision.get("reason", ""),
            "stop_status":        decision.get("stop_status", str((stop_rec or {}).get("status") or "")),
            "review_action":      decision.get("action", "client_approval"),
        })

    if not debtors_by_manager:
        logger.info("run_approval_preview: кандидатов нет — батч не создан")
        return None

    total = sum(len(v) for v in debtors_by_manager.values())
    logger.info(
        "run_approval_preview: создаём батч — %d менеджеров, %d клиентов",
        len(debtors_by_manager), total,
    )

    batch = create_batch(debtors_by_manager)
    save_batch(batch)
    await send_manager_previews(batch)

    logger.info("run_approval_preview: батч %s создан и отправлен менеджерам", batch["batch_id"])
    return batch["batch_id"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AI Debt Collector — Минбаракат",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python -m collector.collections_engine --dry-run\n"
            "  python -m collector.collections_engine --preview\n"
            "  python -m collector.collections_engine --send-approved --batch-id 20260412-120000-ab12\n"
            "  python -m collector.collections_engine --send-approved --batch-id 20260412-120000-ab12 --client 'ТОО Альфа'\n"
            "  python -m collector.collections_engine --check-promises"
        ),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Показать что будет отправлено, НЕ отправлять")
    parser.add_argument("--preview", action="store_true",
                        help="Сформировать список и отправить менеджерам на согласование (без WhatsApp)")
    parser.add_argument("--send", action="store_true",
                        help="Выполнить реальную отправку сообщений")
    parser.add_argument("--send-approved", action="store_true",
                        help="Отправить WhatsApp только клиентам из admin-approved batch")
    parser.add_argument("--batch-id", type=str, default=None,
                        help="ID approval batch для --send-approved")
    parser.add_argument("--client", type=str, default=None,
                        help="Обработать только одного клиента (подстрока имени)")
    parser.add_argument("--check-promises", action="store_true",
                        help="Проверить просроченные обещания оплаты")
    parser.add_argument("--fix-first-seen", action="store_true",
                        help="Одноразовый патч: пересчитать first_seen у клиентов с датой 2026-03-19")
    parser.add_argument("--resend-preview", action="store_true",
                        help="Повторно отправить превью менеджерам из зависшего батча (требует --batch-id)")
    args = parser.parse_args()

    if (
        not args.dry_run and not args.send and not args.send_approved
        and not args.check_promises and not args.preview and not args.fix_first_seen
        and not args.resend_preview
    ):
        parser.print_help()
        return 0

    if args.resend_preview:
        if not args.batch_id:
            logger.error("--resend-preview requires --batch-id")
            return 1
        from collector.approval_flow import load_batch, send_manager_previews
        batch = load_batch(args.batch_id)
        if not batch:
            logger.error("Батч %s не найден", args.batch_id)
            return 1
        logger.info("Повторная отправка превью для батча %s", args.batch_id)
        asyncio.run(send_manager_previews(batch))
        pending = [m for m, s in batch["managers"].items() if not s.get("preview_message_id")]
        if pending:
            logger.warning("Не удалось отправить: %s", ", ".join(pending))
        else:
            logger.info("Все превью отправлены успешно")
        return 0

    if args.preview:
        batch_id = asyncio.run(run_approval_preview(single_client=args.client))
        if batch_id:
            logger.info("Батч согласования создан: %s", batch_id)
        else:
            logger.info("Нет кандидатов для согласования")
        return 0

    if args.check_promises:
        asyncio.run(check_promises())
        return 0

    if args.fix_first_seen:
        from collector.collections_db import fix_first_seen_inflation
        result = fix_first_seen_inflation()
        logger.info(
            "fix-first-seen: исправлено=%d, удалено=%d, сброшено=%d",
            result["fixed"], result["reset"], result["skipped"],
        )
        return 0

    if args.send_approved:
        if not args.batch_id:
            logger.error("--send-approved requires --batch-id")
            return 1
        asyncio.run(send_approved_batch(args.batch_id, single_client=args.client))
        return 0

    # Phase 2: legacy live send is disabled; controlled live uses --send-approved.
    if args.send:
        logger.error(
            "--send disabled for Phase 2 controlled live. "
            "Use --send-approved --batch-id <id> after admin approval."
        )
        return 1

    dry_run = args.dry_run or not args.send
    if dry_run and not args.dry_run:
        logger.info("ВНИМАНИЕ: запуск без --send — автоматически dry-run режим")

    asyncio.run(run(dry_run=dry_run, single_client=args.client))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

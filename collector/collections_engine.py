#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collections/collections_engine.py
Главный оркестратор AI-Коллектора долгов.

Версия: 1.0.5 (2026-04-09)

CLI:
  python -m collector.collections_engine --dry-run
  python -m collector.collections_engine --send
  python -m collector.collections_engine --send --client "ТОО Альфа"
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
from datetime import datetime

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
        f"Пропущено (стоп-лист): {skipped}",
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
            # Менеджер занят диалогом по другому клиенту — пропуск без API-вызова
            logger.info(
                "[%s] менеджер %s занят диалогом по %s — пропуск",
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
    )
    logger.info("[%s] level=%d days=%d | текст: %s...", name, level, days, text[:60])

    if dry_run:
        result["sent"] = True
        result["message_text"] = text
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

    debt_data = load_latest_debt_json()
    if not debt_data:
        logger.warning("Нет данных дебиторки — завершаем")
        await notify_admin("⚠️ AI Коллектор: нет данных дебиторки для обработки")
        return

    debtors = classify_debtors(debt_data)
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

        # Пересчитываем уровень через собственный счётчик дней с первой отгрузки.
        # days_silence из 1С сбрасывается на любую оплату — это ненадёжно.
        # Наш счётчик считает дни с момента ПЕРВОГО обнаружения долга у клиента
        # и сбрасывается только при полном погашении (debt=0).
        # BUG-C5 fix: dry-run не должен регистрировать first_seen в state.
        real_days = 0 if dry_run else get_debt_days_since_first_seen(name)
        # Уровень — максимум из 1С-дней и наших дней (берём наибольший)
        from collector.debt_monitor import _level_for_days
        level = max(client["level"], _level_for_days(real_days))
        client = dict(client, level=level, days=max(client["days"], real_days))

        # Фильтр по одному клиенту если задан
        if single_client and single_client.lower() not in name.lower():
            continue

        # Level 0 — пропускаем
        if level == 0:
            continue

        # Клиент на ручном/авто стопе — пропускаем, управляется debt_stop_control
        try:
            from bot.debt_stop_control import load_registry as _dsc_registry
            _dsc_reg = _dsc_registry()
            _dsc_rec = _dsc_reg.get(name)
            if _dsc_rec and _dsc_rec.get("status") in (
                "stopped", "auto_stopped", "pending_clearance", "conditional"
            ):
                logger.info("[%s] в стоп-листе (статус: %s) — пропуск коллектора",
                            name, _dsc_rec["status"])
                continue
        except Exception as _e:
            logger.debug("Ошибка проверки stop-registry: %s", _e)

        # Фильтр: пропускаем только клиентов с нулевым или отрицательным долгом.
        # Управление исключениями — через стоп-лист (debt_stop_control).
        # Покупка (debit > 0) не означает что долг погашен — контакт нужен.
        if client.get("amount", 0) <= 0:
            logger.info("[%s] пропуск — долг погашен или отрицательный (amount=%.0f)",
                        name, client.get("amount", 0))
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

        result = await _process_single(client, contact, dry_run)
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AI Debt Collector — Минбаракат",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python -m collector.collections_engine --dry-run\n"
            "  python -m collector.collections_engine --send\n"
            "  python -m collector.collections_engine --send --client 'ТОО Альфа'\n"
            "  python -m collector.collections_engine --check-promises"
        ),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Показать что будет отправлено, НЕ отправлять")
    parser.add_argument("--send", action="store_true",
                        help="Выполнить реальную отправку сообщений")
    parser.add_argument("--client", type=str, default=None,
                        help="Обработать только одного клиента (подстрока имени)")
    parser.add_argument("--check-promises", action="store_true",
                        help="Проверить просроченные обещания оплаты")
    args = parser.parse_args()

    if not args.dry_run and not args.send and not args.check_promises:
        parser.print_help()
        return 0

    if args.check_promises:
        asyncio.run(check_promises())
        return 0

    dry_run = args.dry_run or not args.send
    if dry_run and not args.dry_run:
        logger.info("ВНИМАНИЕ: запуск без --send — автоматически dry-run режим")

    asyncio.run(run(dry_run=dry_run, single_client=args.client))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

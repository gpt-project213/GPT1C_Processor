#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collections/collections_engine.py
Главный оркестратор AI-Коллектора долгов.

Версия: 1.0.0 (2026-03-16)

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

from collector.debt_monitor import (
    classify_debtors,
    load_contacts,
    load_latest_debt_json,
    match_client,
)
from collector.collections_db import (
    _DEBT_DATE_PREFIX,
    already_contacted_today,
    already_notified_manager_today,
    get_debt_days_since_first_seen,
    get_pending_promises,
    load_state,
    mark_escalated,
    mark_manager_notified,
    mark_promise_broken,
    reset_debt_first_seen,
    save_call_result,
    save_promise,
    set_phone_pending,
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


def daily_summary(processed: List[Dict], dry_run: bool = False) -> str:
    """Формирует ежедневную сводку работы коллектора."""
    total = len(processed)
    sent = sum(1 for r in processed if r.get("sent"))
    promised = [r for r in processed if r.get("promise_received")]
    broken = [r for r in processed if r.get("promise_broken")]
    no_contact = [r for r in processed if r.get("no_contacts")]
    escalated = [r for r in processed if r.get("escalated")]

    mode_label = "🔇 DRY-RUN (сообщения НЕ отправлялись)" if dry_run else "✅ LIVE"
    lines = [
        f"📊 <b>AI Коллектор — ежедневная сводка</b> {mode_label}",
        f"",
        f"Всего должников уровней 1–5: {total}",
        f"Обработано сегодня: {sent}",
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
        lines.append(f"Нет контактов в справочнике: {len(no_contact)}")
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

    # Генерируем текст сообщения (всегда — для dry-run и логирования)
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
            tg_ok = await send_telegram(int(tg_id), text)
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
        # Используем диалог с менеджером
        from collector.dialog_store import get_dialog as _get_dialog_state
        from collector.manager_dialog import start_dialog as _start_manager_dialog

        dialog = _get_dialog_state(manager_chat_id)

        if dialog and dialog.get("client_name") == name:
            state = dialog.get("state", "")
            if state in ("CONFIRMED", "DONE"):
                # WhatsApp уже отправлен через диалог
                result["sent"] = True
                result["via_dialog"] = True
                return result
            else:
                # Диалог в процессе — ждём менеджера
                logger.info(
                    "[%s] диалог в состоянии %s — ожидаем менеджера", name, state
                )
                return result

        elif dialog and dialog.get("state") not in ("CONFIRMED", "DONE", None):
            # Менеджер занят диалогом по другому клиенту
            logger.info(
                "[%s] менеджер %s занят диалогом по %s — пропуск",
                name, manager_name, dialog.get("client_name"),
            )
            return result

        else:
            # Запускаем новый диалог
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
    contacts = load_contacts()
    processed: List[Dict] = []

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
        real_days = get_debt_days_since_first_seen(name)
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

        # Уже контактировали сегодня — пропускаем
        if not dry_run and already_contacted_today(name):
            logger.info("[%s] уже обработан сегодня — пропуск", name)
            continue

        # Ищем контакты
        contact = match_client(name, contacts)
        if not contact:
            logger.info("[%s] нет в справочнике контактов — пропуск", name)
            processed.append({"name": name, "level": level, "no_contacts": True,
                               "sent": False, "promise_received": False,
                               "promise_broken": False, "escalated": False})
            # Авторегистрация + уведомление менеджера (не чаще 1 раза в день)
            if not dry_run and not already_notified_manager_today(name):
                mgr_name = client.get("manager", "")
                mgr_chat_id = _get_manager_chat_id(mgr_name) if mgr_name else None

                # Авторегистрируем клиента в реестре без телефона
                from collector.registry_manager import auto_register_client
                auto_register_client(
                    name=name,
                    manager=mgr_name,
                    amount=client["amount"],
                    days=client["days"],
                    violation=bool(client.get("violation_shipment")),
                )

                if mgr_chat_id:
                    violation_flag = (
                        "⚠️ <b>НАРУШЕНИЕ: отгрузка при наличии долга!</b>\n\n"
                        if client.get("violation_shipment") else ""
                    )
                    reg_msg = (
                        f"🤖 <b>Новый должник внесён в реестр автоматически</b>\n\n"
                        f"{violation_flag}"
                        f"Клиент: <b>{name}</b>\n"
                        f"Долг: <b>{client['amount']:,.0f} тг</b>\n"
                        f"Дней просрочки: <b>{client['days']}</b> (уровень {level})\n\n"
                        f"📞 Чтобы бот мог подготовить напоминание для этого клиента, "
                        f"нужен его номер WhatsApp.\n\n"
                        f"⚠️ <b>Без вашего одобрения клиенту ничего не уйдёт.</b> "
                        f"Бот пришлёт вам текст на проверку — вы сами решаете, отправлять или нет.\n\n"
                        f"🔴 <b>ВАЖНО:</b> Убедитесь, что номер принадлежит именно "
                        f"этому клиенту — ошибка приведёт к тому, что бот будет "
                        f"беспокоить постороннего человека!"
                    )
                    # Сохраняем pending-состояние и отправляем с кнопками
                    set_phone_pending(mgr_chat_id, name)
                    from collector.collections_db import set_name_pending as _set_name_pending
                    _set_name_pending(mgr_chat_id, name)
                    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
                    keyboard = InlineKeyboardMarkup([[
                        InlineKeyboardButton(
                            "📞 Внести телефон клиента",
                            callback_data="reg_phone",
                        ),
                        InlineKeyboardButton(
                            "✏️ Исправить имя",
                            callback_data="reg_name",
                        ),
                    ]])
                    from collector.communications import send_telegram_with_markup
                    await send_telegram_with_markup(mgr_chat_id, reg_msg, keyboard)
                    mark_manager_notified(name)
                    logger.info(
                        "[%s] авторегистрация + уведомление менеджеру %s отправлено",
                        name, mgr_name,
                    )
            continue

        result = await _process_single(client, contact, dry_run)
        processed.append(result)

    # Итоговая сводка → администратору
    summary = daily_summary(processed, dry_run=dry_run)
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

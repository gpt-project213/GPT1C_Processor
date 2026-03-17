#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/test_dialog.py
Тестовый запуск диалога коллектора.

Отправляет тебе в Telegram диалог менеджера с кнопками —
точно так же как увидит менеджер при реальной работе.
WhatsApp уходит на TEST_WA_PHONE из .env.

Использование:
  python -X utf8 tools/test_dialog.py
  python -X utf8 tools/test_dialog.py --client "ТОО Альфа Трейд"
  python -X utf8 tools/test_dialog.py --client "ТОО Пример" --days 20 --amount 750000
  python -X utf8 tools/test_dialog.py --list        # показать клиентов в базе
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT / ".env", encoding="utf-8-sig", override=False)

# Проверяем TEST_MODE
TEST_MODE = os.getenv("TEST_MODE", "0") == "1"
TEST_TG   = os.getenv("TEST_TG_CHAT_IDS", "")
TEST_WA   = os.getenv("TEST_WA_PHONE", "")

CONTACTS_PATH = ROOT / "config" / "debtors_contacts.json"


def load_contacts() -> dict:
    try:
        with open(CONTACTS_PATH, encoding="utf-8") as f:
            d = json.load(f)
        return {k: v for k, v in d.items() if not k.startswith("_")}
    except (OSError, json.JSONDecodeError) as e:
        print(f"Ошибка чтения contacts: {e}")
        return {}


def print_preflight():
    print("\n" + "═"*55)
    print("  AI Collector — Тестовый запуск диалога")
    print("═"*55)
    print(f"  TEST_MODE       : {'✅ включён' if TEST_MODE else '❌ ВЫКЛЮЧЕН — включи TEST_MODE=1 в .env'}")
    print(f"  TEST_TG_CHAT_IDS: {TEST_TG or '❌ не задан'}")
    print(f"  TEST_WA_PHONE   : {TEST_WA or '❌ не задан (WhatsApp не уйдёт)'}")
    print("═"*55 + "\n")
    if not TEST_MODE:
        print("СТОП: TEST_MODE=0 — добавь TEST_MODE=1 в .env и запусти снова.")
        sys.exit(1)
    if not TEST_TG:
        print("СТОП: TEST_TG_CHAT_IDS не задан в .env")
        sys.exit(1)


async def run_test(client_name: str, days: int, amount: float):
    contacts = load_contacts()

    if client_name not in contacts:
        print(f"Клиент '{client_name}' не найден в базе.")
        print("Доступные клиенты:")
        for k, v in contacts.items():
            print(f"  - {k}  (менеджер: {v.get('manager','?')})")
        sys.exit(1)

    contact = contacts[client_name]
    manager_name = contact.get("manager", "")
    phone        = contact.get("phone", "")

    # В TEST_MODE сообщение уйдёт на TEST_TG_CHAT_IDS (тебе)
    # Для теста используем первый ID из TEST_TG_CHAT_IDS как manager_chat_id
    try:
        manager_chat_id = int(TEST_TG.split(",")[0].strip())
    except (ValueError, IndexError):
        print("Ошибка: TEST_TG_CHAT_IDS некорректен")
        sys.exit(1)

    from collector.debt_monitor import _level_for_days
    level = _level_for_days(days)

    print(f"  Клиент     : {client_name}")
    print(f"  Менеджер   : {manager_name} → в TEST_MODE → тебе ({manager_chat_id})")
    print(f"  Телефон WA : {phone} → в TEST_MODE → {TEST_WA or 'не задан'}")
    print(f"  Просрочка  : {days} дней | Уровень: {level} | {amount:,.0f} тг")
    print()

    if level == 0:
        print(f"  Уровень 0 (менее 10 дней) — коллектор не работает.")
        print(f"  Попробуй --days 15 или больше.")
        sys.exit(0)

    # Удаляем старый диалог если был
    from collector.dialog_store import get_dialog, remove_dialog
    existing = get_dialog(manager_chat_id)
    if existing:
        print(f"  Найден старый диалог по '{existing.get('client_name')}' — удаляем...")
        remove_dialog(manager_chat_id)

    print("  Отправляю диалог в Telegram...")

    from collector.manager_dialog import start_dialog
    client = {
        "name":   client_name,
        "level":  level,
        "days":   days,
        "amount": amount,
    }
    await start_dialog(client, contact, manager_name, manager_chat_id)

    print()
    print("  ✅ Сообщение отправлено! Проверь Telegram.")
    print()
    print("  Что делать дальше:")
    print("  1. В Telegram нажми кнопки — проверь что они работают")
    print("  2. Нажми [✅ Актуально — отправить] → WhatsApp уйдёт на TEST_WA_PHONE")
    print("  3. Нажми [❌ Не отправлять] → введи причину → Вадиму придёт запрос")
    print("  4. Попробуй отправить голосовое вместо текста")
    print()


def main():
    parser = argparse.ArgumentParser(description="Тестовый запуск диалога коллектора")
    parser.add_argument("--client",  default="ТОО Пример",
                        help="Имя клиента из базы (default: ТОО Пример)")
    parser.add_argument("--days",    type=int,   default=15,
                        help="Дней просрочки (default: 15)")
    parser.add_argument("--amount",  type=float, default=500000,
                        help="Сумма долга в тенге (default: 500000)")
    parser.add_argument("--list",    action="store_true",
                        help="Показать всех клиентов в базе")
    args = parser.parse_args()

    if args.list:
        contacts = load_contacts()
        print("\nКлиенты в базе контактов:")
        for k, v in contacts.items():
            nc = v.get("name_confirmations", 0)
            pc = v.get("phone_confirmations", 0)
            print(f"  {k}")
            print(f"    телефон: {v.get('phone','нет')} | менеджер: {v.get('manager','?')} "
                  f"| имя подтв: {nc}/3 | тел подтв: {pc}/3")
        return

    print_preflight()
    asyncio.run(run_test(args.client, args.days, args.amount))


if __name__ == "__main__":
    main()

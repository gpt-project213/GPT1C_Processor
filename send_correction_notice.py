#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Одноразовое сообщение менеджерам — исправление ошибочных уведомлений.
Запуск: python send_correction_notice.py
"""
import io
import json
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import httpx
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env", encoding="utf-8-sig")

BOT_TOKEN = os.getenv("TG_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")
MANAGERS_PATH = ROOT_DIR / "config" / "managers.json"

MESSAGE = """⚠️ <b>Техническое уведомление</b>

Сегодня утром AI Коллектор отправил вам запросы на ввод телефонов для <b>некоторых клиентов с очень маленьким долгом</b> (менее 5 000 тг) — это была техническая ошибка.

Такие запросы можно <b>проигнорировать</b>. Вводить телефоны для клиентов с долгом менее 5 000 тг не нужно — система теперь их исключает автоматически.

Запросы на телефоны для клиентов с <b>реальным долгом ≥ 5 000 тг</b> остаются актуальными.

Приносим извинения за неудобство. 🤝"""


def send(chat_id: int, name: str) -> bool:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    resp = httpx.post(url, json={
        "chat_id": chat_id,
        "text": MESSAGE,
        "parse_mode": "HTML",
    }, timeout=15)
    if resp.status_code == 200:
        print(f"✅ Отправлено: {name} (chat_id={chat_id})")
        return True
    else:
        print(f"❌ Ошибка для {name}: {resp.status_code} — {resp.text[:200]}")
        return False


def main():
    if not BOT_TOKEN:
        print("❌ BOT_TOKEN не найден в .env")
        return

    managers = json.loads(MANAGERS_PATH.read_text(encoding="utf-8"))
    print(f"Рассылка {len(managers)} менеджерам...\n")

    ok = 0
    for name, chat_id in managers.items():
        if send(int(chat_id), name):
            ok += 1

    print(f"\nГотово: {ok}/{len(managers)} доставлено.")


if __name__ == "__main__":
    main()

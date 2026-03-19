#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Одноразовая рассылка — объяснение AI Коллектора всем менеджерам.
Запуск: python send_collector_intro.py
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

MESSAGE = """📢 <b>AI Коллектор — ваш новый помощник</b>

Добрый день! 👋

С сегодняшнего дня в компании запущен <b>AI-помощник по работе с дебиторской задолженностью</b>.

<b>Что он делает:</b>
Каждый день в 9:00 он автоматически проверяет должников и отправляет им вежливые напоминания об оплате — по WhatsApp или Telegram. Без вашего участия.

<b>Почему он написал вам сегодня:</b>
По некоторым клиентам отсутствуют контактные данные (телефон). Без телефона помощник не может связаться с должником.

<b>Что нужно от вас:</b>
В сообщениях выше вы получили список клиентов без контактов. Нажмите кнопку <b>«📞 Внести телефон клиента»</b> под каждым из них и укажите номер.

Формат: <code>+7 XXX XXX XX XX</code> или просто <code>87XXXXXXXXX</code>

⚠️ <b>Важно:</b> вводите правильный номер — бот будет писать именно на него от имени компании. После заполнения он начнёт работать автоматически.

Спасибо! 🤝"""


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

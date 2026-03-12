#!/usr/bin/env python
# coding: utf-8
"""
send_tg.py · v2.4 (2025-09-05, Asia/Almaty)

Назначение:
- Низкоуровневая отправка в Telegram: длинный текст (с разбиением) и файлы
- Поддержка inline-меню под документом: [Детальный] [Анализ ИИ] [Архив]

Окружение (.env):
- TG_BOT_TOKEN, ADMIN_CHAT_ID
- AI_TG_SPLIT=true/false         (делить длинные сообщения; по умолчанию true)
- AI_TG_CHUNK=3500               (размер куска в символах)
- AI_TG_SLEEP_MS=400             (пауза между кусками, мс)
- AI_TG_PRE=false                (true → оборачивать каждый кусок в <pre>)
"""

from __future__ import annotations

import os
import time
import json
import html
from pathlib import Path
from typing import Iterable, List, Optional

# Загрузка .env (BOM-safe, override)
try:
    import dotenv  # type: ignore
    dotenv.load_dotenv(encoding="utf-8-sig", override=True)
except Exception:
    pass

import requests

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID", "").strip()
API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

def _assert_ready():
    if not TG_BOT_TOKEN:
        raise RuntimeError("TG_BOT_TOKEN не задан в .env")
    if not ADMIN_CHAT_ID:
        raise RuntimeError("ADMIN_CHAT_ID не задан в .env")

def _bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except (ValueError, TypeError):
        return default

AI_TG_SPLIT   = _bool_env("AI_TG_SPLIT", True)
AI_TG_CHUNK   = _int_env("AI_TG_CHUNK", 3500)
AI_TG_SLEEPMS = _int_env("AI_TG_SLEEP_MS", 400)
AI_TG_PRE     = _bool_env("AI_TG_PRE", False)

# ──────────────────────────────────────────────────────────────────
# Вспомогательное: аккуратное разбиение текста
def _chunk_text(s: str, limit: int) -> List[str]:
    if len(s) <= limit:
        return [s]
    out: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        j = min(i + limit, n)
        # стараемся резать по переводу строки/пробелу
        k = s.rfind("\n", i, j)
        if k == -1:
            k = s.rfind(" ", i, j)
        if k == -1 or k <= i + int(limit * 0.5):
            k = j
        out.append(s[i:k])
        i = k
    return out

# ──────────────────────────────────────────────────────────────────
# Клавиатура под документом
def _build_menu() -> dict:
    # Простой не показываем внизу под файлом
    return {
        "inline_keyboard": [
            [{"text": "Детальный",  "callback_data": "ext"}],
            [{"text": "Анализ ИИ",  "callback_data": "ai"}],
            [{"text": "Архив",      "callback_data": "arch"}],
        ]
    }

# ──────────────────────────────────────────────────────────────────
# Публичные функции (именно эти импортирует ai_analyzer.py)
def send_long_text(text: str, chat_id: Optional[str] = None, parse_html: bool = True) -> bool:
    """
    Отправляет длинный текст в Telegram, при необходимости делит на части.
    Учитывает .env: AI_TG_SPLIT, AI_TG_CHUNK, AI_TG_SLEEP_MS, AI_TG_PRE.
    """
    _assert_ready()
    chat = chat_id or ADMIN_CHAT_ID
    chunks = _chunk_text(text, AI_TG_CHUNK) if (AI_TG_SPLIT or len(text) > AI_TG_CHUNK) else [text]

    for idx, part in enumerate(chunks, 1):
        data = {
            "chat_id": chat,
        }
        if parse_html:
            if AI_TG_PRE:
                # В режиме <pre> экранируем HTML, чтобы не сломать разметку
                safe = html.escape(part)
                data["text"] = f"<pre>{safe}</pre>"
            else:
                data["text"] = part
            data["parse_mode"] = "HTML"
        else:
            data["text"] = part

        r = requests.post(f"{API}/sendMessage", data=data, timeout=90)
        r.raise_for_status()

        if idx < len(chunks):
            time.sleep(max(0, AI_TG_SLEEPMS) / 1000.0)

    return True

def send_text(text: str, chat_id: Optional[str] = None, parse_html: bool = True) -> bool:
    """
    Короткий синоним — отправка одного сообщения (без явного разбиения).
    Оставлен для обратной совместимости.
    """
    _assert_ready()
    data = {
        "chat_id": chat_id or ADMIN_CHAT_ID,
        "text": text,
        "parse_mode": "HTML" if parse_html else None
    }
    r = requests.post(f"{API}/sendMessage", data=data, timeout=60)
    r.raise_for_status()
    return True

def send_file(file_path: str | Path, chat_id: Optional[str] = None, caption: Optional[str] = None, with_menu: bool = False) -> bool:
    """
    Отправляет документ (HTML/EXCEL и т.п.). При with_menu=True добавляет inline-меню
    непосредственно к документу.
    """
    _assert_ready()
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    data = {
        "chat_id": chat_id or ADMIN_CHAT_ID,
        "caption": caption or "",
        "parse_mode": "HTML"
    }
    if with_menu:
        data["reply_markup"] = json.dumps(_build_menu(), ensure_ascii=False)

    with p.open("rb") as f:
        r = requests.post(f"{API}/sendDocument", data=data, files={"document": f}, timeout=180)
        r.raise_for_status()
    print(f"TG: file OK → {p}")
    return True

# ───────── CLI для ручной проверки ─────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("send_tg: ручная отправка админу")
    ap.add_argument("--text", help="текст/HTML для отправки")
    ap.add_argument("--file", help="путь к файлу")
    ap.add_argument("--caption", help="подпись к файлу")
    ap.add_argument("--with-menu", action="store_true", help="добавить inline-меню под документом")
    ap.add_argument("--no-html", action="store_true", help="не использовать parse_mode=HTML")
    ap.add_argument("--chat-id", help="явно указать chat_id (по умолчанию ADMIN_CHAT_ID)")
    args = ap.parse_args()

    if args.text:
        send_long_text(args.text, chat_id=args.chat_id, parse_html=not args.no_html)
        print("TG: text OK")
    elif args.file:
        send_file(args.file, chat_id=args.chat_id, caption=args.caption, with_menu=args.with_menu)
    else:
        ap.print_help()

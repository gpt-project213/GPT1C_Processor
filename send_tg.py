#!/usr/bin/env python
# coding: utf-8
"""
send_tg.py · v2.4.2 (2026-04-22, Asia/Almaty)

Назначение:
- Низкоуровневая отправка в Telegram: длинный текст (с разбиением) и файлы
- Поддержка inline-меню под документом: [Детальный] [Анализ ИИ] [Архив]

Изменения v2.4.2:
- Fix F-TG-001: send_file читает файл в bytes и передаёт в _post_tg как
  tuple (name, bytes, mime). Без этого на 5xx/Timeout retry отправил бы 0 байт
  (file-handle уже прочитан первой попыткой).
- Fix F-TG-002: CLI ветка --file печатает "TG: file OK" для симметрии с --text.

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
import logging
from pathlib import Path
from typing import Iterable, List, Optional

# Загрузка .env (BOM-safe, override)
try:
    import dotenv  # type: ignore
    dotenv.load_dotenv(encoding="utf-8-sig", override=True)
except Exception:
    pass

import requests
from requests import Response

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID", "").strip()

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

def _api_base() -> str:
    return f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

def _post_tg(method: str, *, data=None, files=None, timeout: int = 60, retries: int = 3) -> Response:
    url = f"{_api_base()}/{method}"
    last_exc: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, data=data, files=files, timeout=timeout)
            if r.status_code == 429 and attempt < retries:
                retry_after = 1
                try:
                    retry_after = int((r.json().get("parameters") or {}).get("retry_after") or 1)
                except (ValueError, TypeError, AttributeError):
                    retry_after = 1
                time.sleep(max(1, retry_after))
                continue
            if 500 <= r.status_code < 600 and attempt < retries:
                time.sleep(attempt)
                continue
            r.raise_for_status()
            return r
        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
            if attempt >= retries:
                raise
            time.sleep(attempt)

    if last_exc:
        raise last_exc
    raise RuntimeError(f"Telegram request failed after retries: {method}")

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
    if parse_html and not AI_TG_PRE and len(text) > AI_TG_CHUNK:
        raise ValueError(
            "send_long_text: long HTML text cannot be safely split; "
            "use AI_TG_PRE=1 or parse_html=False"
        )
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

        _post_tg("sendMessage", data=data, timeout=90)

        if idx < len(chunks):
            time.sleep(max(0, AI_TG_SLEEPMS) / 1000.0)

    return True

def send_text(text: str, chat_id: Optional[str] = None, parse_html: bool = True) -> bool:
    """
    Короткий синоним — отправка одного сообщения (без явного разбиения).
    Оставлен для обратной совместимости.
    """
    _assert_ready()
    data: dict = {
        "chat_id": chat_id or ADMIN_CHAT_ID,
        "text": text,
    }
    if parse_html:
        data["parse_mode"] = "HTML"
    _post_tg("sendMessage", data=data, timeout=60)
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
    }
    if caption:
        data["caption"] = caption
        data["parse_mode"] = "HTML"
    if with_menu:
        data["reply_markup"] = json.dumps(_build_menu(), ensure_ascii=False)

    # Fix F-TG-001: читаем файл в память — при retry в _post_tg file-handle
    # иначе остался бы прочитанным, и вторая попытка отправила бы 0 байт.
    file_bytes = p.read_bytes()
    _post_tg(
        "sendDocument",
        data=data,
        files={"document": (p.name, file_bytes, "application/octet-stream")},
        timeout=180,
    )
    logging.getLogger(__name__).info("TG: file OK → %s", p)
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
        print("TG: file OK")  # Fix F-TG-002: UX-сигнал после успешной отправки
    else:
        ap.print_help()

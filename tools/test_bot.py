#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/test_bot.py
Минимальный тест-бот для проверки диалога коллектора.
Обрабатывает кнопки col_*, текст и голосовые.

Запуск:
  python -X utf8 tools/test_bot.py

Ctrl+C для остановки.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT / ".env", encoding="utf-8-sig", override=False)

# ── Логирование в консоль ──────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Убираем шум от httpx/telegram
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)

logger = logging.getLogger("test_bot")

BOT_TOKEN = os.getenv("TG_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")
TEST_MODE = os.getenv("TEST_MODE", "0") == "1"

if not BOT_TOKEN:
    print("СТОП: TG_BOT_TOKEN не задан в .env")
    sys.exit(1)
if not TEST_MODE:
    print("СТОП: TEST_MODE=0 — добавь TEST_MODE=1 в .env")
    sys.exit(1)

from telegram import Update
from telegram.ext import (
    Application, CallbackQueryHandler, MessageHandler,
    filters, ContextTypes,
)


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""
    chat_id = q.message.chat.id
    logger.info("CALLBACK от %s: %s", chat_id, data)

    if not data.startswith("col_"):
        logger.info("  → не col_* — игнорируем")
        return

    try:
        from collector.manager_dialog import handle_callback as col_cb
        handled = await col_cb(data, chat_id, q.message.message_id)
        if handled:
            logger.info("  → обработан collector")
        else:
            logger.warning("  → НЕ обработан (handle_callback вернул False)")
    except Exception as e:
        logger.error("  → ОШИБКА: %s", e, exc_info=True)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text or ""
    logger.info("ТЕКСТ от %s: %s", chat_id, repr(text[:80]))

    try:
        from collector.manager_dialog import handle_text_message as col_text
        handled = await col_text(chat_id, text)
        if handled:
            logger.info("  → обработан collector")
        else:
            logger.info("  → нет активного диалога, пропускаем")
    except Exception as e:
        logger.error("  → ОШИБКА: %s", e, exc_info=True)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    voice = update.message.voice
    logger.info("ГОЛОСОВОЕ от %s: file_id=%s duration=%ss",
                chat_id, voice.file_id, voice.duration)

    try:
        from collector.manager_dialog import handle_voice_message as col_voice
        handled = await col_voice(chat_id, voice.file_id)
        if handled:
            logger.info("  → голос обработан (Whisper)")
        else:
            logger.info("  → нет активного диалога")
    except Exception as e:
        logger.error("  → ОШИБКА: %s", e, exc_info=True)


async def post_init(app: Application):
    me = await app.bot.get_me()
    logger.info("=" * 55)
    logger.info("Тест-бот запущен: @%s", me.username)
    logger.info("Жди нажатий кнопок... Ctrl+C для остановки")
    logger.info("=" * 55)


def main():
    print("\n" + "="*55)
    print("  AI Collector — Тест-бот")
    print("="*55)
    print(f"  TEST_MODE: ✅ включён")
    print(f"  Обрабатываю: col_* кнопки, текст, голосовые")
    print("="*55 + "\n")

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

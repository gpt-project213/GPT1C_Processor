#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/whatsapp_poller.py
Green API polling — получает входящие сообщения WhatsApp каждые 30 секунд.

Версия: 1.0.2 (2026-04-08)

Endpoints (используется instance-specific URL, напр. https://7107.api.greenapi.com):
  GET  https://{ID[:4]}.api.greenapi.com/waInstance{ID}/receiveNotification/{TOKEN}
  DELETE https://{ID[:4]}.api.greenapi.com/waInstance{ID}/deleteNotification/{TOKEN}/{receiptId}
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

load_dotenv(
    dotenv_path=Path(__file__).resolve().parent.parent / ".env",
    encoding="utf-8-sig",
    override=False,
)

TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))

GREENAPI_ID    = os.getenv("GREENAPI_ID", "")
GREENAPI_TOKEN = os.getenv("GREENAPI_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TEST_MODE      = os.getenv("TEST_MODE", "0") == "1"
TEST_WA_PHONE  = os.getenv("TEST_WA_PHONE", "")

# Каждый инстанс имеет свой поддомен: первые 4 цифры ID → 7107.api.greenapi.com
# Может быть переопределён через GREENAPI_URL в .env
_GREENAPI_BASE = os.getenv(
    "GREENAPI_URL",
    f"https://{GREENAPI_ID[:4]}.api.greenapi.com" if GREENAPI_ID else "https://api.green-api.com"
)

logger = logging.getLogger(__name__)


def _extract_phone(sender: str) -> str:
    """Извлекает номер телефона из формата '77011234567@c.us' → '77011234567'."""
    return sender.split("@")[0] if "@" in sender else sender


async def transcribe_audio(audio_url: str) -> str:
    """Скачивает аудио по URL и транскрибирует через OpenAI Whisper.

    Args:
        audio_url: Прямая ссылка на аудиофайл от Green API.

    Returns:
        Транскрибированный текст или пустая строка при ошибке.
    """
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY не задан — транскрипция недоступна")
        return ""

    # Скачиваем аудио во временный файл
    tmp_path: Optional[str] = None
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(audio_url)
        if resp.status_code != 200:
            logger.warning("Ошибка скачивания аудио %d: %s", resp.status_code, audio_url[:80])
            return ""

        suffix = ".ogg"  # Green API обычно отдаёт ogg/opus
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="wa_audio_")
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(resp.content)
        except OSError as e:
            logger.error("Ошибка записи аудио во временный файл: %s", e)
            return ""

        # Отправляем в Whisper
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        with open(tmp_path, "rb") as audio_file:
            files = {"file": (f"audio{suffix}", audio_file, "audio/ogg")}
            data = {"model": "whisper-1"}
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    whisper_resp = await client.post(
                        "https://api.openai.com/v1/audio/transcriptions",
                        headers=headers,
                        files=files,
                        data=data,
                    )
                if whisper_resp.status_code == 200:
                    result = whisper_resp.json().get("text", "").strip()
                    logger.info("Whisper транскрипция: %s...", result[:60])
                    return result
                logger.warning(
                    "Whisper ошибка %d: %s",
                    whisper_resp.status_code,
                    whisper_resp.text[:200],
                )
                return ""
            except (httpx.RequestError, httpx.TimeoutException) as e:
                logger.error("Whisper сетевая ошибка: %s: %s", type(e).__name__, e or repr(e))
                return ""

    except (httpx.RequestError, httpx.TimeoutException) as e:
        logger.error("Ошибка скачивания аудио: %s", e)
        return ""
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


async def poll_once() -> None:
    """Получает одно уведомление из Green API и обрабатывает его.

    В TEST_MODE обрабатывает только сообщения от TEST_WA_PHONE.
    Всегда удаляет уведомление после обработки (или при ошибке).
    """
    if not GREENAPI_ID or not GREENAPI_TOKEN:
        logger.debug("Green API не настроен (GREENAPI_ID/GREENAPI_TOKEN) — пропуск")
        return

    receive_url = (
        f"{_GREENAPI_BASE}/waInstance{GREENAPI_ID}"
        f"/receiveNotification/{GREENAPI_TOKEN}"
    )

    receipt_id: Optional[int] = None
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            resp = await client.get(receive_url)

        if resp.status_code != 200:
            logger.warning("Green API receiveNotification ошибка %d", resp.status_code)
            return

        data = resp.json()
        if data is None:
            # Нет уведомлений — нормальная ситуация
            return

        receipt_id = data.get("receiptId")
        body = data.get("body", {})

        webhook_type = body.get("typeWebhook", "")
        logger.info("Green API уведомление: type=%s receiptId=%s", webhook_type, receipt_id)

        if webhook_type == "incomingMessageReceived":
            sender_data = body.get("senderData", {})
            sender_raw = sender_data.get("sender", "")
            phone = _extract_phone(sender_raw)

            if TEST_MODE:
                if not TEST_WA_PHONE or phone != TEST_WA_PHONE.lstrip("+"):
                    logger.info(
                        "TEST_MODE: redirecting to test recipients — "
                        "пропуск сообщения от %s (ожидается %s)",
                        phone, TEST_WA_PHONE,
                    )
                    # Удаляем уведомление и выходим
                    await _delete_notification(receipt_id)
                    return

            message_data = body.get("messageData", {})
            msg_type = message_data.get("typeMessage", "")
            text = ""

            if msg_type == "textMessage":
                text = message_data.get("textMessageData", {}).get("textMessage", "")
                logger.info("Входящий текст от %s: %s...", phone, text[:60])

            elif msg_type == "extendedTextMessage":
                text = message_data.get("extendedTextMessageData", {}).get("text", "")
                logger.info("Входящий extendedText от %s: %s...", phone, text[:60])

            elif msg_type == "audioMessage":
                download_url = (
                    message_data.get("fileMessageData", {}).get("downloadUrl", "")
                )
                if download_url:
                    logger.info("Входящее аудио от %s — транскрибирую...", phone)
                    text = await transcribe_audio(download_url)
                    if not text:
                        text = "[аудио не распознано]"
                else:
                    text = "[аудио без ссылки]"

            else:
                # Изображения, документы и т.д.
                text = f"[клиент прислал {msg_type}]"
                logger.info("Входящий %s от %s", msg_type, phone)

            if phone and text:
                try:
                    from collector.client_dialog import handle_incoming
                    await handle_incoming(phone, text)
                except Exception as e:
                    logger.error("handle_incoming ошибка для %s: %s", phone, e)

        else:
            # Игнорируем прочие типы webhook (статусы доставки и т.д.)
            logger.debug("Игнорируем webhook type=%s", webhook_type)

    except (httpx.RequestError, httpx.TimeoutException) as e:
        logger.debug("Green API сетевая ошибка: %s: %s", type(e).__name__, e or repr(e))
    except (KeyError, ValueError, TypeError) as e:
        logger.error("Ошибка разбора уведомления Green API: %s", e)
    finally:
        if receipt_id is not None:
            await _delete_notification(receipt_id)


async def _delete_notification(receipt_id: int) -> None:
    """Удаляет уведомление из очереди Green API."""
    if not GREENAPI_ID or not GREENAPI_TOKEN:
        return
    delete_url = (
        f"{_GREENAPI_BASE}/waInstance{GREENAPI_ID}"
        f"/deleteNotification/{GREENAPI_TOKEN}/{receipt_id}"
    )
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.delete(delete_url)
        if resp.status_code == 200:
            logger.debug("Уведомление %s удалено", receipt_id)
        else:
            logger.warning(
                "Ошибка удаления уведомления %s: статус %d",
                receipt_id, resp.status_code,
            )
    except (httpx.RequestError, httpx.TimeoutException) as e:
        logger.error("Ошибка удаления уведомления %s: %s: %s", receipt_id, type(e).__name__, e or repr(e))

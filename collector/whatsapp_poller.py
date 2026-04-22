#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/whatsapp_poller.py
Green API polling — получает входящие сообщения WhatsApp каждые 30 секунд.

Версия: 1.1.4 (2026-04-22)

v1.1.4 (2026-04-22): AssemblyAI теперь требует `speech_models` как непустой
  список. Контракт запроса обновлён, чтобы входящие голосовые снова
  распознавались.

v1.1.3 (2026-04-22): убран устаревший параметр `speech_model` из запроса
  AssemblyAI transcript; из-за него входящие голосовые сообщения падали с 400.

Endpoints (используется instance-specific URL, напр. https://7107.api.greenapi.com):
  GET  https://{ID[:4]}.api.greenapi.com/waInstance{ID}/receiveNotification/{TOKEN}
  DELETE https://{ID[:4]}.api.greenapi.com/waInstance{ID}/deleteNotification/{TOKEN}/{receiptId}
"""

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime
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
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
ASSEMBLYAI_POLL_SECONDS = int(os.getenv("ASSEMBLYAI_POLL_SECONDS", "18"))
ASSEMBLYAI_SPEECH_MODELS = ["universal-2"]
SAVE_WA_AUDIO = os.getenv("SAVE_WA_AUDIO", "1").lower() in ("1", "true", "yes")
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


_CLIENT_DIALOGS_PATH = Path(__file__).resolve().parents[1] / "logs" / "collector_client_dialogs.json"
_AUDIO_ARCHIVE_DIR = Path(__file__).resolve().parents[1] / "logs" / "whatsapp_audio"


def _has_active_collector_dialog(phone: str) -> bool:
    """Возвращает True если у этого номера есть активный диалог коллектора.

    Диалог появляется только когда бот сам отправил должнику сообщение.
    Это правильный гейт: должник пишет Саиде лично — диалога нет → пропускаем.
    Только ответы на сообщения коллектора обрабатываются.
    """
    phone_digits = "".join(c for c in phone if c.isdigit())
    if not phone_digits:
        return False
    try:
        if not _CLIENT_DIALOGS_PATH.exists():
            return False
        data = json.loads(_CLIENT_DIALOGS_PATH.read_text(encoding="utf-8"))
        return phone_digits in data
    except Exception:
        pass
    return False


async def _download_audio_to_temp(audio_url: str, archive_label: str = "") -> tuple[Optional[str], str]:
    """Скачивает аудио Green API во временный файл."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(audio_url)
    if resp.status_code != 200:
        logger.warning("Ошибка скачивания аудио %d: %s", resp.status_code, audio_url[:80])
        return None, ".ogg"

    suffix = ".ogg"  # Green API обычно отдаёт ogg/opus
    if SAVE_WA_AUDIO:
        try:
            _AUDIO_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            safe_label = "".join(c if c.isalnum() else "_" for c in archive_label).strip("_")
            stamp = datetime.now(tz=TZ).strftime("%Y%m%d_%H%M%S")
            archive_name = f"{stamp}_{safe_label or 'unknown'}{suffix}"
            archive_path = _AUDIO_ARCHIVE_DIR / archive_name
            archive_path.write_bytes(resp.content)
            logger.info("WhatsApp аудио сохранено: %s", archive_path)
        except OSError as e:
            logger.warning("Не удалось сохранить WhatsApp аудио локально: %s", e)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="wa_audio_")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(resp.content)
    except OSError as e:
        logger.error("Ошибка записи аудио во временный файл: %s", e)
        try:
            os.close(tmp_fd)
        except OSError:
            pass
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return None, suffix
    return tmp_path, suffix


async def _transcribe_with_assemblyai_file(tmp_path: str) -> str:
    """Транскрибирует локальный аудиофайл через AssemblyAI."""
    if not ASSEMBLYAI_API_KEY:
        logger.warning("ASSEMBLYAI_API_KEY не задан — AssemblyAI недоступен")
        return ""

    headers = {"authorization": ASSEMBLYAI_API_KEY}
    try:
        audio_bytes = Path(tmp_path).read_bytes()
    except OSError as e:
        logger.error("AssemblyAI: не удалось прочитать аудиофайл: %s", e)
        return ""

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            upload_resp = await client.post(
                "https://api.assemblyai.com/v2/upload",
                headers=headers,
                content=audio_bytes,
            )
            if upload_resp.status_code not in (200, 201):
                logger.warning(
                    "AssemblyAI upload ошибка %d: %s",
                    upload_resp.status_code,
                    upload_resp.text[:200],
                )
                return ""

            upload_url = upload_resp.json().get("upload_url")
            if not upload_url:
                logger.warning("AssemblyAI upload не вернул upload_url")
                return ""

            transcript_resp = await client.post(
                "https://api.assemblyai.com/v2/transcript",
                headers=headers,
                json={
                    "audio_url": upload_url,
                    "speech_models": ASSEMBLYAI_SPEECH_MODELS,
                    "language_detection": True,
                },
            )
            if transcript_resp.status_code not in (200, 201):
                logger.warning(
                    "AssemblyAI transcript ошибка %d: %s",
                    transcript_resp.status_code,
                    transcript_resp.text[:200],
                )
                return ""

            transcript_id = transcript_resp.json().get("id")
            if not transcript_id:
                logger.warning("AssemblyAI transcript не вернул id")
                return ""

            deadline = ASSEMBLYAI_POLL_SECONDS
            while deadline > 0:
                await asyncio.sleep(2)
                deadline -= 2
                status_resp = await client.get(
                    f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                    headers=headers,
                )
                if status_resp.status_code != 200:
                    logger.warning(
                        "AssemblyAI status ошибка %d: %s",
                        status_resp.status_code,
                        status_resp.text[:200],
                    )
                    return ""

                payload = status_resp.json()
                status = payload.get("status")
                if status == "completed":
                    text = (payload.get("text") or "").strip()
                    lang = payload.get("language_code") or "auto"
                    logger.info("AssemblyAI транскрипция (%s): %s...", lang, text[:60])
                    return text
                if status == "error":
                    logger.warning("AssemblyAI ошибка распознавания: %s", payload.get("error"))
                    return ""

            logger.warning("AssemblyAI не успел вернуть транскрипцию за %d сек", ASSEMBLYAI_POLL_SECONDS)
            return ""
    except (httpx.RequestError, httpx.TimeoutException) as e:
        logger.error("AssemblyAI сетевая ошибка: %s: %s", type(e).__name__, e or repr(e))
        return ""
    except (KeyError, ValueError, TypeError) as e:
        logger.error("AssemblyAI ошибка разбора ответа: %s", e)
        return ""


async def transcribe_audio(audio_url: str, archive_label: str = "") -> str:
    """Скачивает аудио по URL и транскрибирует через AssemblyAI.

    Args:
        audio_url:     Прямая ссылка на аудиофайл от Green API.
        archive_label: Метка для имени архивного файла (номер телефона).

    Returns:
        Транскрибированный текст или пустая строка при ошибке.
    """
    if not ASSEMBLYAI_API_KEY:
        logger.warning("ASSEMBLYAI_API_KEY не задан — транскрипция недоступна")
        return ""

    tmp_path: Optional[str] = None
    try:
        tmp_path, _ = await _download_audio_to_temp(audio_url, archive_label=archive_label)
        if not tmp_path:
            return ""
        return await _transcribe_with_assemblyai_file(tmp_path)

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

            # Саида использует личный номер для Green API — все её входящие сообщения
            # (личные контакты, должники пишущие ей напрямую) проходят через бота.
            # Пропускаем всё, у чего нет активного диалога коллектора — бот мог отправить
            # сообщение только тем, кому сам написал первым. Это защищает личную переписку
            # Саиды от перехвата и не тратит квоту Whisper на чужие аудио.
            if not _has_active_collector_dialog(phone):
                logger.info("Нет активного диалога коллектора для номера — пропуск")
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
                    text = await transcribe_audio(download_url, archive_label=phone)
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

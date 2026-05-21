#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/whatsapp_poller.py
Green API polling — получает входящие сообщения WhatsApp каждые 30 секунд.

Версия: 1.1.7 (2026-04-29)

v1.1.6 (2026-04-29): входящие document/image/video сообщения теперь
  передают в client_dialog метаданные вложения и downloadUrl, чтобы чек
  или иное доказательство оплаты можно было сразу переслать менеджеру.

v1.1.5 (2026-04-23): номер WhatsApp теперь выделен только под бота,
  поэтому старый privacy-гейт "только активный диалог" стал опциональным
  через WA_REQUIRE_ACTIVE_DIALOG=1. По умолчанию входящие аудио доходят до
  транскрипции; неизвестные номера всё равно безопасно игнорируются в
  client_dialog.handle_incoming().

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
from collector.logging_utils import get_collector_logger
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

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
MINAI_WA_PHONE  = os.getenv("MINAI_WA_PHONE", "")
DARYA_WA_PHONE  = os.getenv("DARYA_WA_PHONE", "")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
ASSEMBLYAI_POLL_SECONDS = int(os.getenv("ASSEMBLYAI_POLL_SECONDS", "18"))
ASSEMBLYAI_SPEECH_MODELS = ["universal-2"]
SAVE_WA_AUDIO = os.getenv("SAVE_WA_AUDIO", "1").lower() in ("1", "true", "yes")
TEST_MODE      = os.getenv("TEST_MODE", "0") == "1"
TEST_WA_PHONE  = os.getenv("TEST_WA_PHONE", "")
WA_REQUIRE_ACTIVE_DIALOG = os.getenv("WA_REQUIRE_ACTIVE_DIALOG", "0").lower() in ("1", "true", "yes")

# Каждый инстанс имеет свой поддомен: первые 4 цифры ID → 7107.api.greenapi.com
# Может быть переопределён через GREENAPI_URL в .env
_GREENAPI_BASE = os.getenv(
    "GREENAPI_URL",
    f"https://{GREENAPI_ID[:4]}.api.greenapi.com" if GREENAPI_ID else "https://api.green-api.com"
)

logger = get_collector_logger(__name__)


def _mask_phone(phone: str) -> str:
    digits = "".join(c for c in str(phone or "") if c.isdigit())
    if len(digits) <= 4:
        return digits
    return f"{digits[:4]}***{digits[-2:]}"


def _audit(event: str, **kwargs: Any) -> None:
    try:
        from collector.audit_log import audit as _collector_audit
        _collector_audit(event, **kwargs)
    except Exception as exc:
        logger.debug("audit skipped %s: %s", event, exc)


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


def _should_process_incoming(phone: str) -> bool:
    """Гейт входящих WhatsApp-сообщений.

    Старый режим для личного номера Саиды включается через
    WA_REQUIRE_ACTIVE_DIALOG=1. Для выделенного бот-номера по умолчанию
    пропускаем входящие дальше: неизвестные номера безопасно отсекает
    client_dialog.handle_incoming().
    """
    return True if not WA_REQUIRE_ACTIVE_DIALOG else _has_active_collector_dialog(phone)


def _extract_attachment(message_data: Dict[str, Any], msg_type: str) -> Dict[str, str]:
    candidates = [
        message_data.get("fileMessageData", {}) or {},
        message_data.get("documentMessageData", {}) or {},
        message_data.get("imageMessageData", {}) or {},
        message_data.get("videoMessageData", {}) or {},
    ]
    download_url = ""
    caption = ""
    file_name = ""
    mime_type = ""
    for block in candidates:
        if not isinstance(block, dict):
            continue
        download_url = download_url or str(block.get("downloadUrl") or "")
        caption = caption or str(block.get("caption") or "")
        file_name = file_name or str(block.get("fileName") or block.get("file_name") or "")
        mime_type = mime_type or str(block.get("mimeType") or block.get("mime_type") or "")
    return {
        "type": msg_type,
        "download_url": download_url,
        "caption": caption,
        "file_name": file_name,
        "mime_type": mime_type,
    }


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


def _extract_contact_card(message_data: Dict[str, Any], msg_type: str) -> Optional[Dict[str, Any]]:
    """Извлекает телефон и имя из contactMessage / contactsArrayMessage."""
    contacts = []
    if msg_type == "contactMessage":
        c = (
            message_data.get("contactMessageData", {})
            or message_data.get("contactMessage", {})
            or {}
        )
        if isinstance(c, dict) and c:
            contacts = [c]
    elif msg_type == "contactsArrayMessage":
        arr = (
            message_data.get("contactsArrayMessageData", {})
            or message_data.get("contactsArrayMessage", {})
            or {}
        )
        if isinstance(arr, dict):
            contacts = arr.get("contacts", []) or []

    if not contacts:
        return None

    first = contacts[0] if isinstance(contacts[0], dict) else {}
    display_name = str(first.get("displayName") or "").strip()
    vcard = str(first.get("vcard") or "").strip()

    phone = ""
    if vcard:
        for line in vcard.splitlines():
            if "TEL" in line.upper():
                raw = line.rsplit(":", 1)[-1].strip()
                digits = "".join(c for c in raw if c.isdigit())
                if len(digits) >= 10:
                    if digits.startswith("8") and len(digits) == 11:
                        digits = "7" + digits[1:]
                    phone = digits
                    break

    if not phone:
        return None

    return {"type": "contact_card", "phone": phone, "name": display_name}


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

            # ── Жёсткая изоляция личных каналов от коллектора ───────────────
            # Эти номера НИКОГДА не попадают в коллектор — ни при каких условиях.
            _phone_digits = "".join(c for c in phone if c.isdigit())

            def _norm(p: str) -> str:
                """77XXXXXXXXX / 87XXXXXXXXX / 7XXXXXXXXX → последние 10 цифр."""
                d = "".join(c for c in p if c.isdigit())
                return d[-10:] if len(d) >= 10 else d

            _personal_norm_set = {
                _norm(p) for p in (MINAI_WA_PHONE, DARYA_WA_PHONE) if p
            }
            # Сравниваем по 10 цифрам — одинаково для 77.../87.../7...
            _is_personal = _norm(_phone_digits) in _personal_norm_set
            # Для личного номера старый privacy-гейт можно вернуть через
            # WA_REQUIRE_ACTIVE_DIALOG=1. Для выделенного бот-номера входящие
            # пропускаются до handle_incoming(), где неизвестные номера
            # безопасно игнорируются без ответа клиенту.
            # Минай — отдельный контур напоминалок, не проходит через collector
            _minai_digits = "".join(c for c in (MINAI_WA_PHONE or "") if c.isdigit())
            if _minai_digits and _norm(_phone_digits) == _norm(_minai_digits):
                _msg_data = body.get("messageData", {})
                _msg_type = _msg_data.get("typeMessage", "")
                _raw_text = (
                    _msg_data.get("textMessageData", {}).get("textMessage", "")
                    or _msg_data.get("extendedTextMessageData", {}).get("text", "")
                    or _msg_data.get("buttonsResponseMessage", {}).get("selectedDisplayText", "")
                )
                _is_audio = _msg_type == "audioMessage"
                if not _raw_text and _is_audio:
                    _dl_url = _msg_data.get("fileMessageData", {}).get("downloadUrl", "")
                    if _dl_url:
                        logger.info("Голосовое от Минай — транскрибирую...")
                        _transcribed = await transcribe_audio(_dl_url, archive_label="minai")
                        try:
                            from bot.minai_reminders import handle_minai_audio
                            await handle_minai_audio(_transcribed or "")
                        except Exception as _me:
                            logger.error("minai_reminders audio ошибка: %s", _me)
                elif _raw_text:
                    try:
                        from bot.minai_reminders import handle_minai_response
                        await handle_minai_response(_raw_text)
                    except Exception as _me:
                        logger.error("minai_reminders ошибка: %s", _me)
                await _delete_notification(receipt_id)
                return

            # Второй личный канал напоминаний
            _darya_digits = "".join(c for c in (DARYA_WA_PHONE or "") if c.isdigit())
            if _darya_digits and _norm(_phone_digits) == _norm(_darya_digits):
                _darya_msg_data = body.get("messageData", {})
                _darya_type = _darya_msg_data.get("typeMessage", "")
                _darya_text = (
                    _darya_msg_data.get("textMessageData", {}).get("textMessage", "")
                    or _darya_msg_data.get("extendedTextMessageData", {}).get("text", "")
                    or _darya_msg_data.get("buttonsResponseMessage", {}).get("selectedDisplayText", "")
                )
                if not _darya_text and _darya_type == "audioMessage":
                    _dl = _darya_msg_data.get("fileMessageData", {}).get("downloadUrl", "")
                    if _dl:
                        _tr = await transcribe_audio(_dl, archive_label="darya")
                        try:
                            from bot.darya_reminders import handle_darya_audio
                            await handle_darya_audio(_tr or "")
                        except Exception as _de:
                            logger.error("darya audio error: %s", _de)
                elif _darya_text:
                    try:
                        from bot.darya_reminders import handle_darya_response
                        await handle_darya_response(_darya_text)
                    except Exception as _de:
                        logger.error("darya response error: %s", _de)
                await _delete_notification(receipt_id)
                return

            # Финальный барьер: личный номер не должен добраться сюда никогда
            if _is_personal:
                logger.warning("Личный номер достиг коллектора — заблокирован. phone_digits=%s", _phone_digits[-4:])
                await _delete_notification(receipt_id)
                return

            if not _should_process_incoming(phone):
                _audit("wa_incoming_skipped", phone_masked=_mask_phone(phone), reason="no_active_dialog", receipt_id=receipt_id)
                logger.info("Нет активного диалога коллектора для номера — пропуск")
                return

            message_data = body.get("messageData", {})
            msg_type = message_data.get("typeMessage", "")
            text = ""
            attachment = None

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

            elif msg_type in ("contactMessage", "contactsArrayMessage"):
                contact_card = _extract_contact_card(message_data, msg_type)
                if contact_card:
                    attachment = contact_card
                    text = f"[карточка контакта: {contact_card.get('name') or contact_card.get('phone')}]"
                    logger.info(
                        "Карточка контакта от %s: name=%s phone=...%s",
                        phone, contact_card.get("name"), contact_card.get("phone", "")[-4:],
                    )
                else:
                    text = f"[клиент прислал {msg_type} — номер не извлечён]"
                    logger.info("Входящий %s от %s (номер не извлечён)", msg_type, phone)

            else:
                # Изображения, документы и т.д.
                attachment = _extract_attachment(message_data, msg_type)
                caption = str((attachment or {}).get("caption") or "").strip()
                text = caption if caption else f"[клиент прислал {msg_type}]"
                logger.info("Входящий %s от %s", msg_type, phone)

            if phone and text:
                _audit("wa_incoming_received", phone_masked=_mask_phone(phone), msg_type=msg_type, receipt_id=receipt_id, has_attachment=bool(attachment), text_preview=text[:160])
                try:
                    from collector.client_dialog import handle_incoming
                    await handle_incoming(phone, text, attachment=attachment)
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
            _audit("wa_notification_deleted", receipt_id=receipt_id)
            logger.debug("Уведомление %s удалено", receipt_id)
        else:
            logger.warning(
                "Ошибка удаления уведомления %s: статус %d",
                receipt_id, resp.status_code,
            )
    except (httpx.RequestError, httpx.TimeoutException) as e:
        logger.error("Ошибка удаления уведомления %s: %s: %s", receipt_id, type(e).__name__, e or repr(e))

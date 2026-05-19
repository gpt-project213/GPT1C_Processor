"""WhatsApp-напоминалка для Минай.

Полностью изолированный контур — не пересекается с collector/CRM.
Вся логика через WhatsApp (Green API), Telegram игнорируется.

Scheduler вызывает check_and_send() каждые 5 минут.
whatsapp_poller роутит входящие от MINAI_WA_PHONE сюда.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH  = ROOT / "logs" / "minai_reminder_state.json"
CUSTOM_PATH = ROOT / "logs" / "minai_custom_reminders.json"
TZ   = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))
LOG  = logging.getLogger("minai_reminders")

MINAI_PHONE = os.getenv("MINAI_WA_PHONE", "")

# ── Встроенное расписание ──────────────────────────────────────────────────

# Каждая запись: day=число месяца, hour=час отправки (Almaty),
# critical=True → повторяет каждые 2-3ч пока нет подтверждения
BUILTIN: Dict[str, Dict[str, Any]] = {
    # ── Интернет ──────────────────────────────────────────────────────────
    "internet_warn": {
        "label":    "Интернет (предупреждение)",
        "text":     "Доброе утро, Минай! 🌅\n\n🌐 Завтра 18-е — последний день оплатить интернет.\nОплатите сегодня, чтобы не было перебоев в работе бота.",
        "day":      17,
        "hour":     10,
        "critical": False,
        "group":    "internet",
    },
    "internet_due": {
        "label":    "Интернет (крайний срок)",
        "text":     "Доброе утро, Минай! 🌅\n\n🚨 СЕГОДНЯ крайний срок — оплатить интернет!\nБез оплаты бот встанет и вся аналитика компании остановится.",
        "day":      18,
        "hour":     10,
        "critical": True,
        "resend":   [13, 16, 19],
        "group":    "internet",
    },
    # ── Аренда ────────────────────────────────────────────────────────────
    "rent_early": {
        "label":    "Аренда (3 дня до срока)",
        "text":     "Доброе утро, Минай! 🌅\n\n🏠 Через 2 дня — крайний срок оплаты аренды (до 5-го числа).\nПодготовьте оплату заранее.",
        "day":      3,
        "hour":     10,
        "critical": False,
        "group":    "rent",
    },
    "rent_warn": {
        "label":    "Аренда (завтра срок)",
        "text":     "Доброе утро, Минай! 🌅\n\n🏠 Завтра последний день — оплатить аренду (до 5-го)!\nУспейте сегодня — завтра уже будет поздно.",
        "day":      4,
        "hour":     10,
        "critical": False,
        "group":    "rent",
    },
    "rent_due": {
        "label":    "Аренда (крайний срок)",
        "text":     "Доброе утро, Минай! 🌅\n\n🚨 СЕГОДНЯ последний день — оплатить аренду!\nНе откладывайте, срок истекает сегодня.",
        "day":      5,
        "hour":     10,
        "critical": True,
        "resend":   [13, 16, 19],
        "group":    "rent",
    },
    # ── Налоги по зарплате ────────────────────────────────────────────────
    "taxes_early": {
        "label":    "Налоги по зарплате (3 дня до срока)",
        "text":     "Доброе утро, Минай! 🌅\n\n💼 Через 2 дня — крайний срок уплаты налогов по зарплате (до 25-го числа).\nПодготовьте платежи заранее.",
        "day":      23,
        "hour":     10,
        "critical": False,
        "group":    "taxes",
    },
    "taxes_warn": {
        "label":    "Налоги по зарплате (завтра срок)",
        "text":     "Доброе утро, Минай! 🌅\n\n💼 Завтра последний день — налоги по зарплате (до 25-го)!\nОплатите сегодня, чтобы не было штрафов.",
        "day":      24,
        "hour":     10,
        "critical": False,
        "group":    "taxes",
    },
    "taxes_due": {
        "label":    "Налоги по зарплате (крайний срок)",
        "text":     "Доброе утро, Минай! 🌅\n\n🚨 СЕГОДНЯ последний день — налоги по зарплате!\nНеоплата грозит штрафом. Оплатите обязательно сегодня.",
        "day":      25,
        "hour":     10,
        "critical": True,
        "resend":   [13, 16, 19],
        "group":    "taxes",
    },
}

# ── DeepSeek промпт для разбора нового напоминания ───────────────────────

_DEEPSEEK_PROMPT = """\
Ты помощник, который разбирает напоминания на русском языке.

Пользователь написал: «{user_text}»

Извлеки:
1. text — краткое описание действия (1-2 слова или короткая фраза, например «Оплатить газ»)
2. schedule — одно из:
   - "monthly:N" — каждый месяц до N-го числа
   - "weekly:mon/tue/wed/thu/fri/sat/sun" — каждую неделю в день
   - "daily" — каждый день
   - "once:YYYY-MM-DD" — один раз в конкретную дату
3. hour — час напоминания (число 0-23, по умолчанию 9)

Ответь ТОЛЬКО JSON без объяснений:
{"text": "...", "schedule": "...", "hour": 9}

Если невозможно разобрать — ответь: {"error": "непонятно"}
"""

# ── Хранилище ──────────────────────────────────────────────────────────────

def _load_state() -> Dict[str, Any]:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {}
    except Exception:
        return {}


def _save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_custom() -> Dict[str, Any]:
    try:
        return json.loads(CUSTOM_PATH.read_text(encoding="utf-8")) if CUSTOM_PATH.exists() else {}
    except Exception:
        return {}


def _save_custom(data: Dict[str, Any]) -> None:
    CUSTOM_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _all_reminders() -> Dict[str, Dict[str, Any]]:
    combined = dict(BUILTIN)
    combined.update(_load_custom())
    return combined


def _now() -> datetime:
    return datetime.now(TZ)


def _month_key() -> str:
    return _now().strftime("%Y-%m")


def _state_key(rid: str, month: Optional[str] = None) -> str:
    return f"{rid}_{month or _month_key()}"

# ── Отправка WhatsApp ──────────────────────────────────────────────────────

def _send_plain(text: str) -> bool:
    if not MINAI_PHONE:
        return False
    try:
        from collector.communications import send_whatsapp
        return send_whatsapp(MINAI_PHONE, text)
    except Exception as e:
        LOG.error("Ошибка отправки Минай: %s", e)
        return False


def _send_buttons(text: str, buttons: list[dict]) -> bool:
    """Отправляет WhatsApp с кнопками (макс 3) через Green API sendButtons."""
    if not MINAI_PHONE:
        return False
    if os.getenv("COLLECTOR_TEST_MODE") == "1":
        LOG.info("TEST_MODE: minai reminder не отправляется (%s)", text[:60])
        return True
    try:
        from collector.communications import send_whatsapp_buttons
        return send_whatsapp_buttons(MINAI_PHONE, text, buttons)
    except Exception as e:
        LOG.error("Ошибка отправки кнопок Минай: %s", e)
        return _send_plain(text)   # fallback без кнопок


def _btn(bid: str, label: str) -> dict:
    return {"buttonId": bid, "buttonText": label}


def _main_buttons() -> list[dict]:
    return [_btn("done", "✅ Сделала"), _btn("later", "⏰ Позже"), _btn("add", "➕ Добавить")]


_HINT = "\n\n_Нажмите кнопку или напишите/скажите голосовым_ 🎤"


def _snooze_buttons() -> list[dict]:
    return [_btn("snooze_2h", "Через 2 часа"), _btn("snooze_4h", "Через 4 часа"), _btn("snooze_18", "В 18:00")]


def _add_buttons() -> list[dict]:
    return [_btn("add_yes", "✅ Да, правильно"), _btn("add_no", "❌ Нет, не так"), _btn("add_skip", "Пропустить")]

# ── Повторные отправки (persistence) ──────────────────────────────────────

def _send_reminder(rid: str) -> bool:
    reminders = _all_reminders()
    r = reminders.get(rid)
    if not r:
        return False
    ok = _send_buttons(r["text"] + _HINT, _main_buttons())
    if ok:
        state = _load_state()
        sk = _state_key(rid)
        entry = state.get(sk) or {}
        entry.update({"status": "sent", "sent_at": _now().isoformat(), "last_sent_hour": _now().hour})
        state[sk] = entry
        _save_state(state)
        LOG.info("Напоминание отправлено Минай: %s", rid)
    return ok

# ── Проверка и отправка (вызывается каждые 5 мин) ─────────────────────────

def check_and_send() -> None:
    if not MINAI_PHONE:
        return
    now  = _now()
    state = _load_state()
    reminders = _all_reminders()

    for rid, r in reminders.items():
        day = r.get("day")
        # Кастомные расписания — only monthly:N поддерживаем для MVP
        if day is None:
            schedule = r.get("schedule", "")
            if schedule.startswith("monthly:"):
                day = int(schedule.split(":")[1])
            else:
                continue

        if now.day != day:
            continue

        sk = _state_key(rid)
        entry = state.get(sk) or {}
        status = entry.get("status")

        if status == "confirmed":
            continue

        if status == "snoozed":
            until = entry.get("snoozed_until", "")
            if until and now.isoformat() < until:
                continue
            _send_reminder(rid)
            continue

        send_hour = r.get("hour", 9)
        if now.hour >= send_hour and status not in ("sent", "snoozed", "confirmed"):
            _send_reminder(rid)
            continue

        # Повторные отправки для critical
        if r.get("critical") and r.get("resend") and status == "sent":
            last_h = entry.get("last_sent_hour", 0)
            for rh in r["resend"]:
                if now.hour >= rh > last_h:
                    _send_reminder(rid)
                    break

# ── Обработка входящих от Минай ───────────────────────────────────────────

async def handle_minai_audio(transcribed: str) -> None:
    """Получает транскрипцию голосового, отправляет Минай на подтверждение."""
    if not transcribed or transcribed == "[аудио не распознано]":
        _send_plain("Не смогла распознать голосовое. Попробуйте написать текстом.")
        return
    state = _load_state()
    state["__pending_audio_text__"] = transcribed
    _save_state(state)
    _send_buttons(
        f"Вы сказали:\n«{transcribed}»\n\nПравильно?",
        [_btn("audio_yes", "✅ Да, верно"), _btn("audio_no", "❌ Нет, ошиблась")],
    )


async def handle_minai_response(text: str) -> bool:
    """Роутит входящее сообщение от Минай.
    Возвращает True если сообщение обработано.
    """
    t = text.strip()
    state = _load_state()
    now   = _now()

    # Подтверждение транскрипции голосового
    if state.get("__pending_audio_text__"):
        pending_audio = state["__pending_audio_text__"]
        if _match(t, ("✅", "Да", "audio_yes", "верно")):
            del state["__pending_audio_text__"]
            _save_state(state)
            await handle_minai_response(pending_audio)   # обрабатываем как обычный текст
        elif _match(t, ("❌", "Нет", "audio_no", "ошиблась")):
            del state["__pending_audio_text__"]
            _save_state(state)
            _send_plain("Хорошо — можете сказать голосовым ещё раз или написать текстом 🎤✍️")
        return True

    # Проверяем ожидает ли бот текст нового напоминания от Минай
    if state.get("__awaiting_add__"):
        await _process_add_text(t, state)
        return True

    # Ожидание подтверждения распознанного напоминания
    if state.get("__pending_confirm__"):
        pending = state["__pending_confirm__"]
        if _match(t, ("✅", "Да", "add_yes", "правильно")):
            _save_custom_reminder(pending)
            del state["__pending_confirm__"]
            _save_state(state)
            _send_buttons(
                f"✅ Добавила: «{pending['text']}»\nБуду напоминать в нужное время.",
                [_btn("add", "➕ Ещё добавить"), _btn("nothing", "Всё, спасибо")],
            )
        elif _match(t, ("❌", "Нет", "add_no", "не так")):
            del state["__pending_confirm__"]
            _save_state(state)
            _send_plain("Хорошо, попробуйте написать иначе. Например:\n«напомни оплатить газ 10-го числа»")
            state["__awaiting_add__"] = True
            _save_state(state)
        else:
            _skip_add(state)
        return True

    # Найти активный reminder этого дня
    active_rid = _active_today(state)

    # Кнопки основного напоминания
    if _match(t, ("✅", "Сделала", "done")):
        if active_rid:
            _mark_confirmed(active_rid, state)
        _send_buttons(
            "Отлично, записала ✅\nЧто-нибудь ещё добавить в напоминания?",
            [_btn("add", "➕ Добавить"), _btn("nothing", "Всё, спасибо")],
        )
        return True

    if _match(t, ("⏰", "Позже", "later")):
        _send_buttons("Когда напомнить?", _snooze_buttons())
        return True

    if _match(t, ("Через 2", "snooze_2h")):
        _set_snooze(active_rid, state, hours=2)
        _send_plain("Напомню через 2 часа ⏰")
        return True

    if _match(t, ("Через 4", "snooze_4h")):
        _set_snooze(active_rid, state, hours=4)
        _send_plain("Напомню через 4 часа ⏰")
        return True

    if _match(t, ("18:00", "snooze_18")):
        _set_snooze_at(active_rid, state, hour=18)
        _send_plain("Напомню в 18:00 ⏰")
        return True

    if _match(t, ("➕", "Добавить", "add")):
        _send_plain(
            "Можете написать или сказать голосовым — что напомнить 🎤✍️\n\n"
            "Примеры:\n"
            "• «оплатить газ 10-го числа каждый месяц»\n"
            "• «позвонить бухгалтеру каждый понедельник»\n"
            "• «продлить лицензию 25 июня»"
        )
        state["__awaiting_add__"] = True
        _save_state(state)
        return True

    if _match(t, ("Всё", "спасибо", "nothing")):
        _send_plain("Хорошо 👍")
        return True

    # Инициатива Минай — свободный текст вне контекста кнопок
    # Пробуем распознать как напоминание
    if _looks_like_reminder(t):
        await _process_add_text(t, state)
        return True

    return False   # не распознали — передаём дальше


# ── Вспомогательные функции ────────────────────────────────────────────────

def _match(text: str, keywords: tuple) -> bool:
    tl = text.lower()
    return any(k.lower() in tl for k in keywords)


def _looks_like_reminder(text: str) -> bool:
    """Эвристика: фраза похожа на просьбу добавить напоминание."""
    triggers = ("напомни", "напоминание", "не забыть", "добавь", "поставь напомин")
    return any(t in text.lower() for t in triggers)


def _active_today(state: Dict[str, Any]) -> Optional[str]:
    now = _now()
    for rid, r in _all_reminders().items():
        day = r.get("day") or (
            int(r["schedule"].split(":")[1]) if str(r.get("schedule", "")).startswith("monthly:") else None
        )
        if day and now.day == day:
            sk = _state_key(rid)
            if state.get(sk, {}).get("status") not in ("confirmed", None):
                return rid
    return None


def _mark_confirmed(rid: str, state: Dict[str, Any]) -> None:
    sk = _state_key(rid)
    entry = state.get(sk) or {}
    entry.update({"status": "confirmed", "confirmed_at": _now().isoformat()})
    state[sk] = entry
    _save_state(state)


def _set_snooze(rid: Optional[str], state: Dict[str, Any], hours: int) -> None:
    if not rid:
        return
    sk = _state_key(rid)
    entry = state.get(sk) or {}
    entry.update({"status": "snoozed", "snoozed_until": (_now() + timedelta(hours=hours)).isoformat()})
    state[sk] = entry
    _save_state(state)


def _set_snooze_at(rid: Optional[str], state: Dict[str, Any], hour: int) -> None:
    if not rid:
        return
    target = _now().replace(hour=hour, minute=0, second=0, microsecond=0)
    if target <= _now():
        target += timedelta(days=1)
    sk = _state_key(rid)
    entry = state.get(sk) or {}
    entry.update({"status": "snoozed", "snoozed_until": target.isoformat()})
    state[sk] = entry
    _save_state(state)


def _skip_add(state: Dict[str, Any]) -> None:
    for k in ("__awaiting_add__", "__pending_confirm__"):
        state.pop(k, None)
    _save_state(state)
    _send_plain("Хорошо, пропускаем 👍")


async def _process_add_text(text: str, state: Dict[str, Any]) -> None:
    """DeepSeek разбирает свободный текст в структурированное напоминание."""
    state.pop("__awaiting_add__", None)
    _save_state(state)

    parsed = await _deepseek_parse(text)
    if not parsed or "error" in parsed:
        _send_plain(
            "Не смогла разобрать. Попробуйте написать или сказать голосовым чётче 🎤✍️\n"
            "Например: «напомни оплатить [что] [когда]»"
        )
        return

    r_text = parsed.get("text", "").strip()
    schedule = parsed.get("schedule", "monthly:1")
    hour_val = int(parsed.get("hour", 9))

    # Формируем читаемое описание
    sched_human = _schedule_human(schedule)
    confirm_text = (
        f"Правильно понял?\n\n"
        f"📌 {r_text}\n"
        f"🕐 {sched_human}, в {hour_val:02d}:00"
    )

    state["__pending_confirm__"] = {
        "text": r_text,
        "schedule": schedule,
        "hour": hour_val,
    }
    _save_state(state)
    _send_buttons(confirm_text, _add_buttons())


async def _deepseek_parse(user_text: str) -> Optional[Dict[str, Any]]:
    try:
        import os as _os
        import httpx as _httpx

        api_key = _os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            return None

        prompt = _DEEPSEEK_PROMPT.format(user_text=user_text)
        payload = {
            "model": _os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 200,
        }
        resp = _httpx.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=15,
        )
        if resp.status_code != 200:
            LOG.warning("DeepSeek error %d: %s", resp.status_code, resp.text[:200])
            return None
        content = resp.json()["choices"][0]["message"]["content"].strip()
        # Убираем markdown-обёртку если есть
        if content.startswith("```"):
            content = content.split("```")[1].lstrip("json").strip()
        return json.loads(content)
    except Exception as e:
        LOG.warning("DeepSeek parse error: %s", e)
        return None


def _schedule_human(schedule: str) -> str:
    if schedule.startswith("monthly:"):
        return f"каждый месяц до {schedule.split(':')[1]}-го"
    if schedule.startswith("weekly:"):
        days = {"mon": "пн", "tue": "вт", "wed": "ср", "thu": "чт", "fri": "пт", "sat": "сб", "sun": "вс"}
        return f"каждую неделю ({days.get(schedule.split(':')[1], schedule)})"
    if schedule == "daily":
        return "каждый день"
    if schedule.startswith("once:"):
        return f"один раз {schedule.split(':')[1]}"
    return schedule


def _save_custom_reminder(pending: Dict[str, Any]) -> None:
    custom = _load_custom()
    import hashlib
    rid = "custom_" + hashlib.md5(pending["text"].encode()).hexdigest()[:8]
    schedule = pending["schedule"]
    day = None
    if schedule.startswith("monthly:"):
        day = int(schedule.split(":")[1])
    custom[rid] = {
        "label": pending["text"],
        "text": f"📌 Напоминание: {pending['text']}",
        "schedule": schedule,
        "day": day,
        "hour": int(pending.get("hour", 9)),
        "critical": False,
        "group": "custom",
    }
    _save_custom(custom)
    LOG.info("Добавлено кастомное напоминание: %s → %s", rid, pending["text"])

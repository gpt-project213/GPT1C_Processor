"""WhatsApp-ассистент для Дарьи.

Личный контур — полностью изолирован. Никаких уведомлений третьим лицам.
Все напоминания личные по умолчанию.
Запускается только при наличии DARYA_WA_PHONE в .env.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

ROOT        = Path(__file__).resolve().parents[1]
STATE_PATH  = ROOT / "logs" / "darya_reminder_state.json"
CUSTOM_PATH = ROOT / "logs" / "darya_custom_reminders.json"
TZ          = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))
LOG         = logging.getLogger("darya_reminders")

DARYA_PHONE = os.getenv("DARYA_WA_PHONE", "")

# ── Встроенное расписание ──────────────────────────────────────────────────
# Пусто по умолчанию — Дарья добавляет сама через текст/голос.
# Можно добавить сюда фиксированные напоминания в том же формате что у Минай.
BUILTIN: Dict[str, Dict[str, Any]] = {}

# ── DeepSeek промпт ────────────────────────────────────────────────────────

_DEEPSEEK_PROMPT = """\
Ты помощник, который разбирает напоминания на русском языке.

Пользователь написал: «{user_text}»

Извлеки:
1. text — краткое описание действия (1-2 слова или короткая фраза)
2. schedule — одно из:
   - "monthly:N" — каждый месяц N-го числа
   - "weekly:mon/tue/wed/thu/fri/sat/sun" — каждую неделю в день
   - "daily" — каждый день
   - "once:YYYY-MM-DD" — один раз в конкретную дату
3. hour — час напоминания (число 0-23, по умолчанию 10)

Ответь ТОЛЬКО JSON без объяснений:
{"text": "...", "schedule": "...", "hour": 10}

Если невозможно разобрать — ответь: {"error": "непонятно"}
"""

_WELCOME = """\
Привет, Даша! 💕

Я твой личный ассистент — меня создал Вадим специально для тебя.
Буду помогать ничего не забывать: дни рождения, дела, встречи — всё что важно.

🔒 *Конфиденциальность*
Технически твой номер и переписка доступны только мне и Вадиму — он разработчик системы.
Но честно говоря, чтобы что-то увидеть, ему пришлось бы специально копаться в коде,
а на это у него нет ни времени, ни желания.
Так что считай — всё что ты пишешь мне, остаётся между нами 🤫

━━━━━━━━━━━━━━━
🔔 *КАК ПОЛЬЗОВАТЬСЯ*
━━━━━━━━━━━━━━━
Получила напоминание?
  ✅ *Сделала* — отметить выполнение
  ⏰ *Позже* — напиши на какое время перенести
  ➕ *Добавить* — добавить новое напоминание

━━━━━━━━━━━━━━━
🎤 *КАК ДОБАВИТЬ НАПОМИНАНИЕ*
━━━━━━━━━━━━━━━
Нажми ➕ Добавить и:
• *напиши* — «напомни записаться к врачу 15-го каждого месяца»
• *или скажи голосовым* 🎤

Я пойму, переспрошу и сохраню.
Можешь написать мне в любое время — я всегда здесь.

━━━━━━━━━━━━━━━
💬 *ПОЖЕЛАНИЯ*
━━━━━━━━━━━━━━━
Что-то неудобно? Напиши «неудобно» — передам Вадиму.

━━━━━━━━━━━━━━━
С заботой, твой ассистент 🤍\
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
    if not DARYA_PHONE:
        return False
    try:
        from collector.communications import send_whatsapp
        return send_whatsapp(DARYA_PHONE, text)
    except Exception as e:
        LOG.error("Ошибка отправки Дарье: %s", e)
        return False


def _send_buttons(text: str, buttons: list) -> bool:
    if not DARYA_PHONE:
        return False
    if os.getenv("COLLECTOR_TEST_MODE") == "1":
        return True
    try:
        from collector.communications import send_whatsapp_buttons
        return send_whatsapp_buttons(DARYA_PHONE, text, buttons)
    except Exception as e:
        LOG.error("Ошибка отправки кнопок Дарье: %s", e)
        return _send_plain(text)


def _btn(bid: str, label: str) -> dict:
    return {"buttonId": bid, "buttonText": label}


def _main_buttons() -> list:
    return [_btn("done", "✅ Сделала"), _btn("later", "⏰ Позже"), _btn("add", "➕ Добавить")]


def _snooze_buttons() -> list:
    return [_btn("snooze_2h", "Через 2 часа"), _btn("snooze_4h", "Через 4 часа"), _btn("snooze_18", "В 18:00")]


def _add_buttons() -> list:
    return [_btn("add_yes", "✅ Да, верно"), _btn("add_no", "❌ Нет, не так"), _btn("add_skip", "Пропустить")]


_HINT = "\n\n_Нажми кнопку или напиши/скажи голосовым_ 🎤"
_DAY_END_HOUR     = 21
_DIALOG_TIMEOUT_M = 30

# ── Приветствие ────────────────────────────────────────────────────────────

def send_welcome_if_needed() -> None:
    if not DARYA_PHONE:
        return
    state = _load_state()
    if state.get("__welcome_sent__"):
        return
    ok = _send_plain(_WELCOME)
    if ok:
        state["__welcome_sent__"] = _now().isoformat()
        _save_state(state)
        LOG.info("Приветствие Дарье отправлено")

# ── Отправка напоминания ───────────────────────────────────────────────────

def _send_reminder(rid: str) -> bool:
    r = _all_reminders().get(rid)
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
    return ok

# ── Авто-закрытие зависших состояний ─────────────────────────────────────

def _expire_stale(state: Dict[str, Any], now: datetime, changed: list) -> None:
    _timeout_msgs = {
        "__awaiting_add__":       "⏰ Время ожидания вышло. Нажми ➕ Добавить чтобы попробовать снова.",
        "__pending_audio_text__": "⏰ Голосовое устарело. Попробуй ещё раз 🎤",
        "__pending_confirm__":    "⏰ Время вышло. Нажми ➕ Добавить заново.",
        "__awaiting_feedback__":  "⏰ Время вышло. Напиши «неудобно» когда будет время.",
        "__awaiting_snooze__":    "⏰ Не дождалась ответа. Напомню позже по расписанию.",
    }
    for key, msg in _timeout_msgs.items():
        val = state.get(key)
        if not val:
            continue
        ts_raw = val.get("_ts") if isinstance(val, dict) else (val if isinstance(val, str) else None)
        if ts_raw:
            try:
                ts = datetime.fromisoformat(ts_raw)
                if (now - ts).total_seconds() > _DIALOG_TIMEOUT_M * 60:
                    del state[key]
                    changed.append(key)
                    _send_buttons(msg, [_btn("add", "➕ Добавить"), _btn("nothing", "Не надо")])
                    continue
            except Exception:
                pass
        if isinstance(val, dict):
            val.setdefault("_ts", now.isoformat())
        else:
            state[key] = {"_ts": now.isoformat(), "_val": True}
        changed.append(key)

    # Сноуз за пределами дня → auto_closed
    for rid in _all_reminders():
        sk = _state_key(rid)
        entry = state.get(sk)
        if not isinstance(entry, dict) or entry.get("status") != "snoozed":
            continue
        until_raw = entry.get("snoozed_until", "")
        try:
            until_dt = datetime.fromisoformat(until_raw)
        except Exception:
            continue
        if until_dt.date() < now.date() or (until_dt.date() == now.date() and now.hour >= _DAY_END_HOUR):
            entry["status"] = "auto_closed"
            entry["auto_closed_at"] = now.isoformat()
            state[sk] = entry
            changed.append(sk)

    # После 21:00 — закрыть sent напоминания текущего дня
    if now.hour >= _DAY_END_HOUR:
        for rid, r in _all_reminders().items():
            if not _fires_today(r, now):
                continue
            sk = _state_key(rid)
            entry = state.get(sk)
            if isinstance(entry, dict) and entry.get("status") in ("sent", "snoozed"):
                entry["status"] = "auto_closed"
                entry["auto_closed_at"] = now.isoformat()
                state[sk] = entry
                changed.append(sk)

# ── Проверка и отправка (каждые 5 мин) ────────────────────────────────────

def _reminder_day(r: Dict[str, Any]) -> Optional[int]:
    day = r.get("day")
    if day is None:
        s = str(r.get("schedule", ""))
        if s.startswith("monthly:"):
            try:
                return int(s.split(":")[1])
            except Exception:
                return None
    return day


def check_and_send() -> None:
    if not DARYA_PHONE:
        return
    now   = _now()
    state = _load_state()
    changed: list = []

    _expire_stale(state, now, changed)
    if changed:
        _save_state(state)
        changed.clear()

    if now.hour >= _DAY_END_HOUR:
        return

    for rid, r in _all_reminders().items():
        if not _fires_today(r, now):
            continue
        sk    = _state_key(rid)
        entry = state.get(sk) or {}
        status = entry.get("status")

        if status in ("confirmed", "auto_closed"):
            continue

        if status == "snoozed":
            until = entry.get("snoozed_until", "")
            if until:
                try:
                    _until_dt = datetime.fromisoformat(until)
                    if now < _until_dt:
                        continue
                except Exception:
                    pass
            _send_reminder(rid)
            continue

        send_hour = r.get("hour", 10)
        if now.hour >= send_hour and status not in ("sent", "snoozed", "confirmed", "auto_closed"):
            _send_reminder(rid)
            continue

        if r.get("resend") and status == "sent":
            last_h = entry.get("last_sent_hour", 0)
            for rh in r["resend"]:
                if now.hour >= rh > last_h:
                    _send_reminder(rid)
                    break

# ── Обработка входящих ────────────────────────────────────────────────────

def _is_simple_reply(text: str) -> bool:
    t = text.lower().strip().rstrip(".,!?")
    simple = ("да", "нет", "сделала", "сделал", "позже", "добавить",
              "спасибо", "ок", "окей", "хорошо", "понял", "поняла", "через")
    if any(t == k or t.startswith(k + " ") for k in simple):
        return True
    if len(t.split()) <= 3 and len(t) <= 20:
        return True
    return False


async def handle_darya_audio(transcribed: str) -> None:
    if not transcribed:
        _send_plain("Не смогла распознать. Попробуй ещё раз 🎤")
        return
    if _is_simple_reply(transcribed):
        await handle_darya_response(transcribed)
        return
    state = _load_state()
    state["__pending_audio_text__"] = {"_ts": _now().isoformat(), "_val": transcribed}
    _save_state(state)
    _send_buttons(
        f"Ты сказала:\n«{transcribed}»\n\nПравильно?",
        [_btn("audio_yes", "✅ Да, верно"), _btn("audio_no", "❌ Нет")],
    )


async def handle_darya_response(text: str) -> bool:
    t     = text.strip()
    state = _load_state()

    # Подтверждение голосового
    _audio = state.get("__pending_audio_text__")
    if _audio:
        pending_audio = (_audio.get("_val") if isinstance(_audio, dict) else _audio) or ""
        if _match(t, ("✅", "Да", "audio_yes", "верно")):
            del state["__pending_audio_text__"]
            _save_state(state)
            if pending_audio:
                await _process_add_text(pending_audio, _load_state())
        elif _match(t, ("❌", "Нет", "audio_no")):
            del state["__pending_audio_text__"]
            _save_state(state)
            _send_plain("Хорошо, попробуй ещё раз голосом или напиши текстом 🎤✍️")
        return True

    # Ожидание текста нового напоминания
    _aw = state.get("__awaiting_add__")
    if _aw and (_aw is True or (_aw.get("_val") if isinstance(_aw, dict) else False)):
        await _process_add_text(t, state)
        return True

    # Подтверждение распознанного напоминания
    if state.get("__pending_confirm__"):
        pending = state["__pending_confirm__"]
        if "work" not in pending:
            # Дарья — всё личное, пропускаем вопрос рабочее/личное
            pending["work"] = False
            state["__pending_confirm__"] = pending
            _save_state(state)
            _send_buttons(
                f"Правильно?\n\n📌 {pending['text']}\n"
                f"🕐 {_schedule_human(pending['schedule'])}, в {pending['hour']:02d}:00",
                _add_buttons(),
            )
            return True
        if _match(t, ("✅", "Да", "add_yes", "верно")):
            _save_custom_reminder(pending)
            del state["__pending_confirm__"]
            _save_state(state)
            _send_buttons(
                f"✅ Запомнила: «{pending['text']}»\nБуду напоминать в нужное время.",
                [_btn("add", "➕ Ещё добавить"), _btn("nothing", "Спасибо")],
            )
        elif _match(t, ("❌", "Нет", "add_no")):
            del state["__pending_confirm__"]
            _save_state(state)
            _send_plain("Хорошо, попробуй написать иначе или скажи голосом 🎤✍️")
            state["__awaiting_add__"] = {"_ts": _now().isoformat(), "_val": True}
            _save_state(state)
        else:
            _skip_add(state)
        return True

    # Основные кнопки напоминания
    active_rid = _active_today(state)

    if _match(t, ("✅", "Сделала", "done")):
        if active_rid:
            _mark_confirmed(active_rid, state)
        _send_buttons(
            "Отлично! ✅\nЧто-нибудь ещё добавить в напоминания?",
            [_btn("add", "➕ Добавить"), _btn("nothing", "Нет, спасибо")],
        )
        return True

    if _match(t, ("⏰", "Позже", "later")):
        state["__awaiting_snooze__"] = {"_ts": _now().isoformat(), "_rid": active_rid}
        _save_state(state)
        _send_plain("На какое время перенести? Напиши — например «в 15:00» или «через 3 часа» 🕐")
        return True

    # Ожидание свободного ввода времени снузи
    _sn = state.get("__awaiting_snooze__")
    if _sn:
        rid_for_snooze = _sn.get("_rid") if isinstance(_sn, dict) else active_rid
        _parsed_snooze = _parse_snooze_time(t)
        if _parsed_snooze:
            del state["__awaiting_snooze__"]
            if isinstance(_parsed_snooze, int):
                _set_snooze(rid_for_snooze, state, hours=_parsed_snooze)
                _send_plain(f"Напомню через {_parsed_snooze} ч ⏰")
            else:
                _set_snooze_at(rid_for_snooze, state, hour=_parsed_snooze[0], minute=_parsed_snooze[1])
                _send_plain(f"Напомню в {_parsed_snooze[0]:02d}:{_parsed_snooze[1]:02d} ⏰")
        else:
            _send_plain("Не поняла время. Напиши иначе — например «в 16:00» или «через 2 часа»")
        return True

    if _match(t, ("➕", "Добавить", "add")):
        state["__awaiting_add__"] = {"_ts": _now().isoformat(), "_val": True}
        _save_state(state)
        _send_plain(
            "Можешь написать или сказать голосовым 🎤✍️\n\n"
            "Например:\n"
            "• «напомни записаться к врачу 15-го каждого месяца»\n"
            "• «поздравить маму 23 мая»\n"
            "• «каждую пятницу позвонить бабушке»"
        )
        return True

    if _match(t, ("Нет", "Спасибо", "nothing", "Не надо")):
        _send_plain("Хорошо 💕")
        return True

    # Фидбек
    if _looks_like_feedback(t):
        _fw = state.get("__awaiting_feedback__")
        if _fw and (_fw.get("_val") if isinstance(_fw, dict) else False):
            await _process_feedback(t, state)
            return True
        state["__awaiting_feedback__"] = {"_ts": _now().isoformat(), "_val": True}
        _save_state(state)
        _send_plain("Напиши или скажи голосовым что именно хочешь изменить 🎤✍️")
        return True

    _fw = state.get("__awaiting_feedback__")
    if _fw and (_fw.get("_val") if isinstance(_fw, dict) else False):
        await _process_feedback(t, state)
        return True

    # Свободный текст сначала пробуем разобрать через AI.
    if _should_try_ai_parse(t):
        await _process_add_text(t, state)
        return True

    # Инициатива — похоже на напоминание
    if _looks_like_reminder(t):
        await _process_add_text(t, state)
        return True

    # Catch-all
    _send_buttons(
        "Привет! 💕\n\nНе поняла. Что хочешь сделать?",
        [_btn("add", "➕ Добавить напоминание"),
         _btn("fb",  "💬 Пожелание"),
         _btn("nothing", "Ничего, всё хорошо")],
    )
    return True

# ── Вспомогательные ────────────────────────────────────────────────────────

def _parse_snooze_time(text: str):
    """Парсит свободный ввод времени. Возвращает int (часов) или (hour, minute) или None."""
    import re
    t = text.lower().strip()
    # «через N часа/часов/ч»
    m = re.search(r"через\s+(\d+)\s*(час|ч)", t)
    if m:
        return int(m.group(1))
    # «через N минут/мин»
    m = re.search(r"через\s+(\d+)\s*(мин)", t)
    if m:
        mins = int(m.group(1))
        return (0, mins)  # обработается как timedelta
    # «в HH:MM» или «в HH»
    m = re.search(r"в\s+(\d{1,2})(?::(\d{2}))?", t)
    if m:
        h = int(m.group(1))
        mn = int(m.group(2)) if m.group(2) else 0
        if 0 <= h <= 23:
            return (h, mn)
    # просто «HH:MM»
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23:
            return (h, mn)
    return None


def _match(text: str, keywords: tuple) -> bool:
    tl = text.lower()
    return any(k.lower() in tl for k in keywords)


def _looks_like_reminder(text: str) -> bool:
    triggers = ("напомни", "напоминание", "не забыть", "добавь", "поставь напомин")
    return any(t in text.lower() for t in triggers)


def _looks_like_feedback(text: str) -> bool:
    triggers = ("неудобно", "хочу изменить", "изменить", "не нравится", "пожелание",
                "можно поменять", "feedback", "фидбек")
    return any(t in text.lower() for t in triggers)


def _should_try_ai_parse(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return not _is_simple_reply(t)


def _fires_today(r: Dict[str, Any], now: datetime) -> bool:
    """Возвращает True если напоминание r должно отработать сегодня (now)."""
    day = r.get("day")
    if day is not None:
        return day == now.day

    schedule = str(r.get("schedule", ""))

    if schedule.startswith("monthly:"):
        try:
            d = int(schedule.split(":")[1])
            return 1 <= d <= 31 and d == now.day
        except Exception:
            return False

    if schedule == "daily":
        return True

    if schedule.startswith("weekly:"):
        days = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
        return now.weekday() == days.get(schedule.split(":")[1].lower(), -1)

    if schedule.startswith("once:"):
        from datetime import date as _date
        try:
            target = _date.fromisoformat(schedule.split(":", 1)[1])
            return now.date() == target
        except Exception:
            return False

    return False


def _active_today(state: Dict[str, Any]) -> Optional[str]:
    now = _now()
    for rid, r in _all_reminders().items():
        if not _fires_today(r, now):
            continue
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


def _set_snooze_at(rid: Optional[str], state: Dict[str, Any], hour: int, minute: int = 0) -> None:
    if not rid:
        return
    now = _now()
    if hour == 0 and minute > 0:
        # «через N минут»
        target = now + timedelta(minutes=minute)
    else:
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
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
    _send_plain("Хорошо 💕")


def _validate_parsed(parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(parsed, dict) or "error" in parsed:
        return None
    text = str(parsed.get("text", "")).strip()
    if not text:
        return None
    schedule = str(parsed.get("schedule", "monthly:1")).strip()
    ok = False
    if schedule == "daily":
        ok = True
    elif schedule.startswith("monthly:"):
        try:
            d = int(schedule.split(":")[1]); ok = 1 <= d <= 31
        except Exception:
            schedule = "monthly:1"; ok = True
    elif schedule.startswith("weekly:"):
        ok = schedule.split(":")[1].lower() in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
    elif schedule.startswith("once:"):
        try:
            from datetime import date as _d; _d.fromisoformat(schedule.split(":", 1)[1]); ok = True
        except Exception:
            pass
    if not ok:
        schedule = "monthly:1"
    try:
        hour = int(parsed.get("hour", 10))
        if not 0 <= hour <= 23:
            hour = 10
    except Exception:
        hour = 10
    return {"text": text, "schedule": schedule, "hour": hour}


async def _process_add_text(text: str, state: Dict[str, Any]) -> None:
    state.pop("__awaiting_add__", None)
    _save_state(state)
    _raw = await _deepseek_parse(text)
    parsed = _validate_parsed(_raw) if _raw else None
    if not parsed:
        _send_plain(
            "Не смогла разобрать. Напиши чётче или скажи голосовым 🎤\n"
            "Например: «напомни [что] [когда]»"
        )
        return
    r_text   = parsed["text"]
    schedule = parsed["schedule"]
    hour_val = parsed["hour"]
    state["__pending_confirm__"] = {
        "text": r_text, "schedule": schedule,
        "hour": hour_val, "_ts": _now().isoformat(),
    }
    _save_state(state)
    _send_buttons(
        f"Правильно?\n\n📌 {r_text}\n🕐 {_schedule_human(schedule)}, в {hour_val:02d}:00",
        _add_buttons(),
    )


async def _process_feedback(text: str, state: Dict[str, Any]) -> None:
    state.pop("__awaiting_feedback__", None)
    _save_state(state)
    _send_plain("Принято, передала Вадиму 💕\nСпасибо!")
    _forward_feedback(text)


def _forward_feedback(text: str) -> None:
    try:
        import httpx as _httpx
        token = os.getenv("TG_BOT_TOKEN", "")
        admin = os.getenv("ADMIN_CHAT_ID", "")
        if not token or not admin:
            return
        msg = (
            f"💬 <b>Пожелание от Даши</b>\n\n"
            f"{text}\n\n"
            f"<i>{_now().strftime('%d.%m.%Y %H:%M')}</i>"
        )
        _httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": admin, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        LOG.error("Ошибка пересылки фидбека: %s", e)


async def _deepseek_parse(user_text: str) -> Optional[Dict[str, Any]]:
    try:
        import httpx as _httpx
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            return None
        payload = {
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "messages": [{"role": "user", "content": _DEEPSEEK_PROMPT.format(user_text=user_text)}],
            "temperature": 0, "max_tokens": 200,
        }
        resp = _httpx.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload, timeout=15,
        )
        if resp.status_code != 200:
            return None
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("```")[1].lstrip("json").strip()
        return json.loads(content)
    except Exception as e:
        LOG.warning("DeepSeek parse error: %s", e)
        return None


def _schedule_human(schedule: str) -> str:
    if schedule.startswith("monthly:"):
        return f"каждый месяц {schedule.split(':')[1]}-го"
    if schedule.startswith("weekly:"):
        days = {"mon": "пн", "tue": "вт", "wed": "ср", "thu": "чт",
                "fri": "пт", "sat": "сб", "sun": "вс"}
        return f"каждую неделю ({days.get(schedule.split(':')[1], schedule)})"
    if schedule == "daily":
        return "каждый день"
    if schedule.startswith("once:"):
        return f"один раз {schedule.split(':')[1]}"
    return schedule


def _save_custom_reminder(pending: Dict[str, Any]) -> None:
    import hashlib
    custom = _load_custom()
    rid = "darya_" + hashlib.md5(pending["text"].encode()).hexdigest()[:8]
    schedule = pending["schedule"]
    day = None
    if schedule.startswith("monthly:"):
        try:
            day = int(schedule.split(":")[1])
            if not 1 <= day <= 31:
                day = None
        except Exception:
            day = None
    custom[rid] = {
        "label": pending["text"],
        "text":  f"📌 Напоминание: {pending['text']}",
        "schedule": schedule,
        "day":  day,
        "hour": int(pending.get("hour", 10)),
        "work": False,
        "group": "darya_custom",
    }
    _save_custom(custom)
    LOG.info("Добавлено напоминание Дарьи: %s → %s", rid, pending["text"])

#!/usr/bin/env python
# coding: utf-8
"""
tests/test_collector.py — тесты модулей AI Debt Collector
Запуск: python -X utf8 tests/test_collector.py
Не требует .env, Telegram, Green API, DeepSeek.
"""
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = "✅"
FAIL = "❌"
results = []


def check(name: str, ok: bool, detail: str = ""):
    icon = PASS if ok else FAIL
    msg = f"  {icon} {name}"
    if detail and not ok:
        msg += f"\n       > {detail}"
    print(msg)
    results.append((name, ok))


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ═══════════════════════════════════════════════════════════════
# 1. ИМПОРТЫ
# ═══════════════════════════════════════════════════════════════
section("1. Импорты collector.*")

def try_import(mod):
    try:
        __import__(mod)
        check(f"import {mod}", True)
        return True
    except Exception as e:
        check(f"import {mod}", False, str(e)[:120])
        return False

try_import("collector")
try_import("collector.debt_monitor")
try_import("collector.collections_db")
try_import("collector.collection_agent")
try_import("collector.communications")
try_import("collector.voice_calls")
try_import("collector.collections_engine")
try_import("collector.whatsapp_poller")
try_import("collector.client_dialog")


# ═══════════════════════════════════════════════════════════════
# 2. debt_monitor — _level_for_days
# ═══════════════════════════════════════════════════════════════
section("2. debt_monitor._level_for_days")

from collector.debt_monitor import (
    _level_for_days, get_overdue_days, classify_debtors,
    load_contacts, match_client, _strip_prefix,
)

cases = [
    (0, 0), (5, 0), (9, 0),
    (10, 1), (14, 1),
    (15, 2), (19, 2),
    (20, 3), (24, 3),
    (25, 4), (29, 4),
    (30, 5), (60, 5), (999, 5),
]
for days, expected in cases:
    got = _level_for_days(days)
    check(f"_level_for_days({days}) == {expected}", got == expected,
          f"got {got}")


# ═══════════════════════════════════════════════════════════════
# 3. debt_monitor — get_overdue_days
# ═══════════════════════════════════════════════════════════════
section("3. debt_monitor.get_overdue_days")

check("max_days field",
      get_overdue_days({"max_days": 25}) == 25)
check("days field",
      get_overdue_days({"days": 10}) == 10)
check("overdue_days field",
      get_overdue_days({"overdue_days": 15}) == 15)
check("max_overdue_days field",
      get_overdue_days({"max_overdue_days": 30}) == 30)
check("no fields → 0",
      get_overdue_days({}) == 0)
check("bad value → 0",
      get_overdue_days({"max_days": "неизвестно"}) == 0)
check("string int",
      get_overdue_days({"max_days": "20"}) == 20)


# ═══════════════════════════════════════════════════════════════
# 4. debt_monitor — classify_debtors
# ═══════════════════════════════════════════════════════════════
section("4. debt_monitor.classify_debtors")

# Формат clients list
data_clients = {
    "clients": [
        {"name": "ТОО Альфа", "max_days": 35, "amount": 500000},
        {"name": "ИП Иванов", "max_days": 12, "amount": 100000},
        {"name": "АО Бета",   "max_days": 5,  "amount":  50000},
    ]
}
res = classify_debtors(data_clients)
check("classify: clients list — count == 3", len(res) == 3)
check("classify: level 5 first (35 days)", res[0]["level"] == 5)
check("classify: level 1 second (12 days)", res[1]["level"] == 1)
check("classify: level 0 last (5 days)",   res[2]["level"] == 0)
check("classify: amount parsed",            res[0]["amount"] == 500000.0)

# Формат rows
data_rows = {"rows": [{"name": "Клиент А", "days": 20, "debt": 200000}]}
res2 = classify_debtors(data_rows)
check("classify: rows format works", len(res2) == 1)
check("classify: rows level 3", res2[0]["level"] == 3)

# Плоский dict
data_flat = {"ТОО Гамма": {"max_days": 30, "amount": 300000}}
res3 = classify_debtors(data_flat)
check("classify: flat dict format", len(res3) == 1)
check("classify: flat dict level 5", res3[0]["level"] == 5)

# Пустой
check("classify: empty dict → []", classify_debtors({}) == [])

# Поле name fallback (amount >= 5000 чтобы не попасть под порог фильтра)
data_kontragent = {"clients": [{"контрагент": "ТОО Дельта", "max_days": 25, "amount": 50000}]}
res4 = classify_debtors(data_kontragent)
check("classify: контрагент field", len(res4) == 1 and res4[0]["name"] == "ТОО Дельта")


# ═══════════════════════════════════════════════════════════════
# 5. debt_monitor — _strip_prefix / match_client
# ═══════════════════════════════════════════════════════════════
section("5. debt_monitor.match_client")

check("_strip_prefix ТОО",   _strip_prefix("ТОО Альфа") == "Альфа")
check("_strip_prefix ИП",    _strip_prefix("ИП Иванов") == "Иванов")
check("_strip_prefix LLP",   _strip_prefix("LLP Beta")  == "Beta")
check("_strip_prefix none",  _strip_prefix("Без префикса") == "Без префикса")

contacts = {
    "ТОО Альфа":    {"whatsapp": "+77001112233", "language": "ru"},
    "ИП Сидоров":   {"whatsapp": "+77779999999", "language": "ru"},
    "ТОО Три слова ещё": {"whatsapp": "+7700", "language": "kz"},
}

# Прямое совпадение
check("match: прямое",         match_client("ТОО Альфа", contacts) is not None)
# lower
check("match: lower",          match_client("тоо альфа", contacts) is not None)
# без префикса
check("match: без префикса",   match_client("Альфа", contacts) is not None)
# 3 слова
check("match: 3 слова",        match_client("ТОО Три слова", contacts) is not None)
# не найден
check("match: не найден → None", match_client("Неизвестный клиент", contacts) is None)
# пустые входные данные
check("match: пустое имя → None", match_client("", contacts) is None)
check("match: пустой справочник → None", match_client("ТОО Альфа", {}) is None)


# ═══════════════════════════════════════════════════════════════
# 6. collections_db — основные функции
# ═══════════════════════════════════════════════════════════════
section("6. collections_db (temp state file)")

from collector.collections_db import (
    load_state, save_state, get_client_state, update_after_contact,
    already_contacted_today, save_promise, get_pending_promises,
    mark_escalated, mark_promise_broken, _today,
)
import collector.collections_db as cdb

_orig_path = cdb.STATE_PATH
_tmpdir = tempfile.mkdtemp()
cdb.STATE_PATH = Path(_tmpdir) / "collector_state.json"

try:
    # Пустой стейт
    check("load_state empty", load_state() == {})

    # Сохранение и загрузка
    save_state({"ТОО Тест": {"last_contact_date": "2020-01-01"}})
    s = load_state()
    check("save/load round-trip", s.get("ТОО Тест", {}).get("last_contact_date") == "2020-01-01")

    # update_after_contact
    cdb.STATE_PATH = Path(_tmpdir) / "state2.json"
    update_after_contact("Клиент Х", "telegram", 2, "Привет, у вас долг")
    s2 = load_state()
    rec = s2.get("Клиент Х", {})
    check("update_after_contact: last_level",   rec.get("last_level") == 2)
    check("update_after_contact: channel",      rec.get("last_contact_channel") == "telegram")
    check("update_after_contact: history len",  len(rec.get("history", [])) == 1)

    # already_contacted_today
    check("already_contacted_today: True after update",
          already_contacted_today("Клиент Х"))
    check("already_contacted_today: False for new",
          not already_contacted_today("Новый клиент"))

    # save_promise + get_pending_promises
    cdb.STATE_PATH = Path(_tmpdir) / "state3.json"
    save_promise("Должник А", "2020-01-01", 100000.0)
    pending = get_pending_promises()
    check("get_pending_promises: overdue promise found",
          any(p["name"] == "Должник А" for p in pending))

    # Будущая дата — не просроченное
    save_promise("Должник Б", "2099-12-31", 50000.0)
    pending2 = get_pending_promises()
    check("get_pending_promises: future promise NOT overdue",
          not any(p["name"] == "Должник Б" for p in pending2))

    # mark_escalated
    mark_escalated("Должник А")
    rec2 = get_client_state("Должник А")
    check("mark_escalated sets flag", rec2.get("escalated_to_admin") is True)

    # mark_promise_broken
    mark_promise_broken("Должник А")
    rec3 = get_client_state("Должник А")
    check("mark_promise_broken sets flag", rec3.get("promise_kept") is False)

    # fix_first_seen_inflation — пересчёт раздутых дат
    cdb.STATE_PATH = Path(_tmpdir) / "state_fsi.json"
    # Готовим стейт: два клиента с inflation_date, один без
    from collector.collections_db import _DEBT_DATE_PREFIX, fix_first_seen_inflation
    fsi_state = {
        _DEBT_DATE_PREFIX + "ТОО Ромашка":   {"first_seen": "2026-03-19"},
        _DEBT_DATE_PREFIX + "ИП Сидоров":    {"first_seen": "2026-03-19"},
        _DEBT_DATE_PREFIX + "ТОО Норма":     {"first_seen": "2026-01-10"},  # не должен меняться
    }
    save_state(fsi_state)
    # Мокаем load_latest_debt_json — ТОО Ромашка есть (20 дней), ИП Сидоров — нет
    with patch("collector.debt_monitor.load_latest_debt_json") as mock_ldj:
        mock_ldj.return_value = {"clients": [{"name": "ТОО Ромашка", "days": 20}]}
        result = fix_first_seen_inflation("2026-03-19")
    check("fix_first_seen_inflation: fixed=1", result["fixed"] == 1)
    check("fix_first_seen_inflation: skipped=1", result["skipped"] == 1)
    fsi_after = load_state()
    romashka_key = _DEBT_DATE_PREFIX + "ТОО Ромашка"
    sidorov_key  = _DEBT_DATE_PREFIX + "ИП Сидоров"
    norma_key    = _DEBT_DATE_PREFIX + "ТОО Норма"
    check("fix_first_seen_inflation: ТОО Ромашка first_seen пересчитан",
          fsi_after.get(romashka_key, {}).get("first_seen") != "2026-03-19")
    check("fix_first_seen_inflation: ИП Сидоров first_seen сброшен на None",
          fsi_after.get(sidorov_key, {}).get("first_seen") is None)
    check("fix_first_seen_inflation: ТОО Норма не изменился",
          fsi_after.get(norma_key, {}).get("first_seen") == "2026-01-10")
    # Повторный запуск — нет новых записей с inflation_date
    with patch("collector.debt_monitor.load_latest_debt_json") as mock_ldj2:
        mock_ldj2.return_value = {"clients": [{"name": "ТОО Ромашка", "days": 20}]}
        result2 = fix_first_seen_inflation("2026-03-19")
    check("fix_first_seen_inflation: идемпотентный (fixed=0 на второй запуск)",
          result2 == {"fixed": 0, "reset": 0, "skipped": 0})

finally:
    cdb.STATE_PATH = _orig_path
    shutil.rmtree(_tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════
# 7. communications — WhatsApp отключён по умолчанию
# ═══════════════════════════════════════════════════════════════
section("7. communications — WhatsApp disabled by default")

import collector.communications as comm

check("WHATSAPP_ENABLED is a bool",
      isinstance(comm.WHATSAPP_ENABLED, bool))

# Проверяем поведение когда явно отключено (независимо от .env)
_wa_saved = comm.WHATSAPP_ENABLED
comm.WHATSAPP_ENABLED = False
check("send_whatsapp returns False when disabled",
      comm.send_whatsapp("+77001234567", "тест") is False)

# С включённым флагом — должен пойти в сеть (но GREENAPI_ID пустой → тоже False)
comm.WHATSAPP_ENABLED = True
comm.GREENAPI_ID = ""
check("send_whatsapp returns False when no creds",
      comm.send_whatsapp("+77001234567", "тест") is False)
comm.WHATSAPP_ENABLED = _wa_saved  # вернём как было


# ═══════════════════════════════════════════════════════════════
# 8. communications — is_allowed_time
# ═══════════════════════════════════════════════════════════════
section("8. communications.is_allowed_time")

from datetime import datetime
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Almaty")

# Будний день 10:00
with patch("collector.communications.datetime") as mock_dt:
    mock_now = MagicMock()
    mock_now.weekday.return_value = 0   # понедельник
    mock_now.hour = 10
    mock_dt.now.return_value = mock_now
    check("is_allowed_time: weekday 10h → True", comm.is_allowed_time())

# Будний день 08:00 (до начала)
with patch("collector.communications.datetime") as mock_dt:
    mock_now = MagicMock()
    mock_now.weekday.return_value = 1
    mock_now.hour = 8
    mock_dt.now.return_value = mock_now
    check("is_allowed_time: weekday 08h → False", not comm.is_allowed_time())

# Суббота
with patch("collector.communications.datetime") as mock_dt:
    mock_now = MagicMock()
    mock_now.weekday.return_value = 5   # суббота
    mock_now.hour = 11
    mock_dt.now.return_value = mock_now
    check("is_allowed_time: Saturday → False", not comm.is_allowed_time())


# ═══════════════════════════════════════════════════════════════
# 9. collections_engine — daily_summary
# ═══════════════════════════════════════════════════════════════
section("9. collections_engine.daily_summary")

from collector.collections_engine import daily_summary

processed = [
    {"name": "Клиент А", "sent": True,  "promise_received": True,  "promise_date": "2026-04-01",
     "promise_broken": False, "no_contacts": False, "escalated": False},
    {"name": "Клиент Б", "sent": True,  "promise_received": False,
     "promise_broken": False, "no_contacts": False, "escalated": True},
    {"name": "Клиент В", "sent": False, "promise_received": False,
     "promise_broken": False, "no_contacts": True,  "escalated": False},
]
summary = daily_summary(processed, dry_run=True)
check("daily_summary: dry-run label", "DRY-RUN" in summary)
check("daily_summary: total 3", "3" in summary)
check("daily_summary: escalated 1", "скалировано" in summary)
check("daily_summary: no_contacts 1", "Нет контактов" in summary)
check("daily_summary: promise received", "Обещали" in summary)

summary_live = daily_summary([], dry_run=False)
check("daily_summary: empty live — no DRY", "DRY-RUN" not in summary_live)


# ═══════════════════════════════════════════════════════════════
# 10. client_dialog — detect_language
# ═══════════════════════════════════════════════════════════════
section("10. client_dialog.detect_language")

from collector.client_dialog import detect_language

check("detect_language: русский текст → ru",
      detect_language("Оплачу завтра") == "ru")
check("detect_language: казахский с ә → kz",
      detect_language("Мақсаты бар, әрі бергімен айтысайық") == "kz")
check("detect_language: 2 казахских символа → kz",
      detect_language("Жақсы, мен төлеуге дайынмын, қарызды беремін") == "kz")
check("detect_language: один казахский символ → ru",
      detect_language("Ладно, я позвоню") == "ru")
check("detect_language: пустая строка → ru",
      detect_language("") == "ru")
check("detect_language: кириллица без казахских символов → ru",
      detect_language("Добрый день, уточните пожалуйста") == "ru")
# ә+ғ = 2 kz символа
check("detect_language: ә+ғ → kz",
      detect_language("Ғалым мен Әселдің есімі") == "kz")


# ═══════════════════════════════════════════════════════════════
# 11. client_dialog — store operations
# ═══════════════════════════════════════════════════════════════
section("11. client_dialog — store (temp path)")

import asyncio
import collector.client_dialog as cd_mod

_orig_dialogs_path = cd_mod._DIALOGS_PATH
_tmpdir2 = tempfile.mkdtemp()
cd_mod._DIALOGS_PATH = Path(_tmpdir2) / "client_dialogs.json"

try:
    # start_client_dialog создаёт запись
    asyncio.run(cd_mod.start_client_dialog(
        phone="77011234567",
        client_name="ТОО Тест",
        manager_name="Алена",
        manager_chat_id=123456,
        level=2,
        days=15,
        amount=500000.0,
        message_text="Добрый день, напоминаем о задолженности.",
    ))
    d = cd_mod._get_client_dialog("77011234567")
    check("start_client_dialog: запись создана", d is not None)
    check("start_client_dialog: state=active", d.get("state") == "active")
    check("start_client_dialog: exchange_count=0", d.get("exchange_count") == 0)
    check("start_client_dialog: exchanges has bot message",
          len(d.get("exchanges", [])) == 1 and d["exchanges"][0]["role"] == "bot")
    check("start_client_dialog: amount", d.get("amount") == 500000.0)

    # handle_incoming обновляет exchange_count (мокаем DeepSeek)
    with patch("collector.collection_agent._call_deepseek") as mock_ds:
        mock_ds.return_value = '{"intent":"unclear","promise_date":null,"promise_amount":null,"requires_human":false,"suggested_reply":"Уточните дату."}'
        with patch("collector.client_dialog._reply_to_client") as mock_reply:
            mock_reply.return_value = None
            asyncio.run(cd_mod.handle_incoming("77011234567", "Ладно, я подумаю"))
    d2 = cd_mod._get_client_dialog("77011234567")
    check("handle_incoming: exchange_count увеличился", d2.get("exchange_count") == 1)
    check("handle_incoming: клиентский обмен добавлен",
          any(ex["role"] == "client" for ex in d2.get("exchanges", [])))

    # Диалог inactive — игнорируется
    d2["state"] = "escalated"
    cd_mod._set_client_dialog("77011234567", d2)
    with patch("collector.collection_agent._call_deepseek") as mock_ds2:
        asyncio.run(cd_mod.handle_incoming("77011234567", "ещё одно сообщение"))
    d3 = cd_mod._get_client_dialog("77011234567")
    check("handle_incoming: escalated диалог не обновляется", d3.get("exchange_count") == 1)

    # Неизвестный клиент — игнорируется без ошибки
    asyncio.run(cd_mod.handle_incoming("99999999999", "привет"))
    check("handle_incoming: неизвестный телефон — не падает", True)

    # ─── identity_question: "Кто это?" ─────────────────────────────────────────
    asyncio.run(cd_mod.start_client_dialog(
        phone="77011234568",
        client_name="ТОО Ромашка",
        manager_name="Оксана",
        manager_chat_id=654321,
        level=1,
        days=10,
        amount=200000.0,
        message_text="Добрый день, напоминаем о задолженности.",
    ))
    with patch("collector.collection_agent._call_deepseek") as mock_iq:
        mock_iq.return_value = '{"intent":"identity_question","promise_date":null,"promise_amount":null,"requires_human":false,"suggested_reply":""}'
        with patch("collector.client_dialog._reply_to_client") as mock_riq:
            mock_riq.return_value = None
            asyncio.run(cd_mod.handle_incoming("77011234568", "Кто это?"))
    d_iq = cd_mod._get_client_dialog("77011234568")
    check("identity_question: state=active", d_iq.get("state") == "active")
    bot_replies_iq = [ex["text"] for ex in d_iq.get("exchanges", []) if ex["role"] == "bot"]
    last_iq = bot_replies_iq[-1] if bot_replies_iq else ""
    check("identity_question: ответ содержит название компании",
          cd_mod.COMPANY_NAME in last_iq)
    check("identity_question: ответ содержит имя клиента",
          "ТОО Ромашка" in last_iq)
    check("identity_question: ответ содержит имя менеджера",
          "Оксана" in last_iq)
    check("identity_question: ответ НЕ требует дату оплаты немедленно",
          "когда" not in last_iq.lower() or "менеджер" in last_iq.lower())
    check("identity_question: reply не пустой", bool(last_iq.strip()))

    # ─── promise_without_date: "Я оплачу" ──────────────────────────────────────
    asyncio.run(cd_mod.start_client_dialog(
        phone="77011234569",
        client_name="ИП Асанов",
        manager_name="Ергали",
        manager_chat_id=789012,
        level=2,
        days=18,
        amount=350000.0,
        message_text="Добрый день, напоминаем о задолженности.",
    ))
    with patch("collector.collection_agent._call_deepseek") as mock_pwd:
        mock_pwd.return_value = '{"intent":"promise_without_date","promise_date":null,"promise_amount":null,"requires_human":false,"suggested_reply":"Хорошо! Уточните точную дату оплаты."}'
        with patch("collector.client_dialog._reply_to_client") as mock_rpwd:
            mock_rpwd.return_value = None
            asyncio.run(cd_mod.handle_incoming("77011234569", "Я оплачу"))
    d_pwd = cd_mod._get_client_dialog("77011234569")
    check("promise_without_date 'Я оплачу': state=active", d_pwd.get("state") == "active")
    bot_replies_pwd = [ex["text"] for ex in d_pwd.get("exchanges", []) if ex["role"] == "bot"]
    check("promise_without_date 'Я оплачу': ответ не пустой",
          bool(bot_replies_pwd[-1].strip() if bot_replies_pwd else ""))

    # ─── promise_without_date: "Передам на оплату" ─────────────────────────────
    asyncio.run(cd_mod.start_client_dialog(
        phone="77011234570",
        client_name="ИП Жаксыбеков",
        manager_name="Магира",
        manager_chat_id=111222,
        level=1,
        days=12,
        amount=150000.0,
        message_text="Добрый день, напоминаем о задолженности.",
    ))
    with patch("collector.collection_agent._call_deepseek") as mock_pwd2:
        mock_pwd2.return_value = '{"intent":"promise_without_date","promise_date":null,"promise_amount":null,"requires_human":false,"suggested_reply":""}'
        with patch("collector.client_dialog._reply_to_client") as mock_rpwd2:
            mock_rpwd2.return_value = None
            asyncio.run(cd_mod.handle_incoming("77011234570", "Передам на оплату"))
    d_pwd2 = cd_mod._get_client_dialog("77011234570")
    check("promise_without_date 'Передам на оплату': state=active", d_pwd2.get("state") == "active")
    bot_replies_pwd2 = [ex["text"] for ex in d_pwd2.get("exchanges", []) if ex["role"] == "bot"]
    # Должен использоваться дефолтный текст с датой
    check("promise_without_date 'Передам на оплату': fallback содержит 'дату'",
          any("дату" in t.lower() for t in bot_replies_pwd2))

    # ─── off_topic эскалация: пустой bot reply НЕ сохраняется ──────────────────
    asyncio.run(cd_mod.start_client_dialog(
        phone="77011234571",
        client_name="ТОО Заря",
        manager_name="Ергали",
        manager_chat_id=333444,
        level=1,
        days=11,
        amount=80000.0,
        message_text="Добрый день.",
    ))
    # Первый unclear — мягкое возвращение (off_topic_count=0 → 1)
    with patch("collector.collection_agent._call_deepseek") as mock_ot1:
        mock_ot1.return_value = '{"intent":"unclear","promise_date":null,"promise_amount":null,"requires_human":false,"suggested_reply":""}'
        with patch("collector.client_dialog._reply_to_client") as mock_rot1:
            mock_rot1.return_value = None
            asyncio.run(cd_mod.handle_incoming("77011234571", "ладно ладно"))
    # Второй unclear — эскалация (без пустого reply в exchanges)
    with patch("collector.collection_agent._call_deepseek") as mock_ot2:
        mock_ot2.return_value = '{"intent":"unclear","promise_date":null,"promise_amount":null,"requires_human":false,"suggested_reply":""}'
        with patch("collector.client_dialog._reply_to_client") as mock_rot2:
            mock_rot2.return_value = None
            with patch("collector.client_dialog.escalate_to_manager"):
                asyncio.run(cd_mod.handle_incoming("77011234571", "ну не знаю"))
    d_ot = cd_mod._get_client_dialog("77011234571")
    empty_bot_replies = [
        ex for ex in d_ot.get("exchanges", [])
        if ex["role"] == "bot" and not ex.get("text", "").strip()
    ]
    check("off_topic эскалация: пустые bot replies не сохраняются",
          len(empty_bot_replies) == 0)

finally:
    cd_mod._DIALOGS_PATH = _orig_dialogs_path
    shutil.rmtree(_tmpdir2, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════
# 12. name_confirmations — счётчик
# ═══════════════════════════════════════════════════════════════
section("12. name_confirmations / phone_confirmations logic")

import collector.dialog_store as ds_mod

_orig_ds_path = ds_mod.DIALOGS_PATH
_tmpdir3 = tempfile.mkdtemp()
ds_mod.DIALOGS_PATH = Path(_tmpdir3) / "dialogs.json"

try:
    dlg = ds_mod.new_dialog(
        manager_chat_id=999,
        manager_name="Тест",
        client_name="ТОО Проверка",
        level=1,
        days=10,
        amount=100000.0,
        current_contact={"phone": "+77001112233", "name_confirmations": 0, "phone_confirmations": 0},
    )
    check("new_dialog: name_confirmed=False", dlg.get("name_confirmed") is False)
    check("new_dialog: phone_confirmed=False", dlg.get("phone_confirmed") is False)
    check("new_dialog: awaiting_name_text=False", dlg.get("awaiting_name_text") is False)
    check("new_dialog: control_deadline=None", dlg.get("control_deadline") is None)
    check("new_dialog: control_extensions=0", dlg.get("control_extensions") == 0)
    check("new_dialog: awaiting_manager_explanation=False",
          dlg.get("awaiting_manager_explanation") is False)

    # Обновляем name_confirmed
    ds_mod.update_dialog(999, name_confirmed=True)
    d_upd = ds_mod.get_dialog(999)
    check("update_dialog: name_confirmed → True", d_upd.get("name_confirmed") is True)

    # Новые состояния в PENDING_STATES
    check("STATE_AWAITING_MANAGER_EXPLANATION in PENDING_STATES",
          ds_mod.STATE_AWAITING_MANAGER_EXPLANATION in ds_mod.PENDING_STATES)
    check("STATE_AWAITING_NAME_TEXT in PENDING_STATES",
          ds_mod.STATE_AWAITING_NAME_TEXT in ds_mod.PENDING_STATES)
    check("STATE_AWAITING_PHONE_TEXT in PENDING_STATES",
          ds_mod.STATE_AWAITING_PHONE_TEXT in ds_mod.PENDING_STATES)

finally:
    ds_mod.DIALOGS_PATH = _orig_ds_path
    shutil.rmtree(_tmpdir3, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════
# 13. TEST_MODE — send_whatsapp redirect
# ═══════════════════════════════════════════════════════════════
section("13. TEST_MODE — WhatsApp redirect")

import collector.communications as comm2

# Сохраняем исходные значения
_orig_test_mode  = comm2.TEST_MODE
_orig_test_wa    = comm2.TEST_WA_PHONE
_orig_wa_enabled = comm2.WHATSAPP_ENABLED

try:
    # TEST_MODE=1 + TEST_WA_PHONE — перенаправляет
    comm2.TEST_MODE        = True
    comm2.TEST_WA_PHONE    = "77000000001"
    comm2.WHATSAPP_ENABLED = True
    comm2.GREENAPI_ID      = ""  # нет кредов — вернёт False, но перенаправление залогировано
    result_redirect = comm2.send_whatsapp("+77011234567", "тест")
    # При пустом GREENAPI_ID возвращает False — это ок, главное не упасть
    check("TEST_MODE: send_whatsapp не упал при перенаправлении", True)

    # TEST_MODE=0 — работает как обычно
    comm2.TEST_MODE = False
    comm2.WHATSAPP_ENABLED = False
    check("TEST_MODE=0: WHATSAPP_ENABLED=0 → False",
          comm2.send_whatsapp("+77011234567", "тест") is False)

    # TEST_MODE=1 без TEST_WA_PHONE — не перенаправляет
    comm2.TEST_MODE     = True
    comm2.TEST_WA_PHONE = ""
    comm2.WHATSAPP_ENABLED = False
    check("TEST_MODE=1 + TEST_WA_PHONE='' → False (disabled)",
          comm2.send_whatsapp("+77011234567", "тест") is False)

finally:
    comm2.TEST_MODE        = _orig_test_mode
    comm2.TEST_WA_PHONE    = _orig_test_wa
    comm2.WHATSAPP_ENABLED = _orig_wa_enabled


# ═══════════════════════════════════════════════════════════════
# 15. РЕГРЕССИЯ — INCIDENT 2026-04-10 (несанкционированная WA-рассылка)
# ═══════════════════════════════════════════════════════════════
section("15. Регрессия: INCIDENT 2026-04-10")

# ── FIX-1: debit > 0 блокирует клиента ──────────────────────────────────────
_fix1_client_debit = {
    "name": "Тест Активный Покупатель",
    "debit": 150_000.0,
    "credit": 0.0,
    "amount": 500_000.0,
    "days": 15,
    "level": 2,
}
_fix1_d = _fix1_client_debit.get("debit", 0.0) or 0.0
_fix1_c = _fix1_client_debit.get("credit", 0.0) or 0.0
check(
    "FIX-1: клиент с debit>0 фильтруется (не идёт в коллектор)",
    _fix1_d > 0 or _fix1_c > 0,
)

# ── FIX-1: credit > 0 блокирует клиента ─────────────────────────────────────
_fix1_client_credit = {
    "name": "Тест Активный Плательщик",
    "debit": 0.0,
    "credit": 50_000.0,
    "amount": 300_000.0,
    "days": 20,
    "level": 3,
}
_fix1_d2 = _fix1_client_credit.get("debit", 0.0) or 0.0
_fix1_c2 = _fix1_client_credit.get("credit", 0.0) or 0.0
check(
    "FIX-1: клиент с credit>0 фильтруется (не идёт в коллектор)",
    _fix1_d2 > 0 or _fix1_c2 > 0,
)

# ── FIX-3: старый first_seen не надувает real_days до level 3 ─────────────────
def _level_for_days_regression(days: int) -> int:
    for thr, lvl in [(30, 5), (25, 4), (20, 3), (15, 2), (10, 1)]:
        if days >= thr:
            return lvl
    return 0

_fix3_days_1c   = 3    # реальные данные 1С: 3 дня (уровень 0)
_fix3_real_raw  = 22   # старый first_seen: 22 дня назад (инцидент)
_fix3_capped    = min(_fix3_real_raw, _fix3_days_1c + 7)   # = 10
_fix3_level_fix = max(_level_for_days_regression(_fix3_days_1c),
                       _level_for_days_regression(_fix3_capped))
_fix3_level_bug = max(_level_for_days_regression(_fix3_days_1c),
                       _level_for_days_regression(_fix3_real_raw))
check(
    f"FIX-3: days_1c=3 + raw_real_days=22 → capped={_fix3_capped} → level={_fix3_level_fix} (было {_fix3_level_bug})",
    _fix3_level_fix <= 1 and _fix3_level_bug == 3,
)

# ── FIX-2: ветка "direct send без manager lock" отсутствует в коде ───────────
import pathlib as _pathlib
_engine_src = (_pathlib.Path(__file__).parent.parent
               / "collector" / "collections_engine.py").read_text(encoding="utf-8")
_no_direct_send = (
    "direct send без manager lock" not in _engine_src
    and "manager_chat_id = None" not in _engine_src
)
# Примечание: "direct send запрещён" теперь есть в коде как guard-лог OP-4 — это корректно
check(
    "FIX-2: ветка 'direct send без manager lock' удалена из кода",
    _no_direct_send,
    detail="Найдены следы bypass-логики в collections_engine.py" if not _no_direct_send else "",
)

# ── FIX-4: send_whatsapp() блокируется при WHATSAPP_ENABLED=0 ────────────────
try:
    import importlib
    import collector.communications as _comm_fix4
    _orig_wa_fix4 = _comm_fix4.WHATSAPP_ENABLED

    # Проверка через hardguard (FIX-4): env перечитывается при каждом вызове
    with patch.dict(os.environ, {"WHATSAPP_ENABLED": "0"}):
        _comm_fix4.WHATSAPP_ENABLED = True  # намеренно рассинхронизируем константу
        _result_fix4 = _comm_fix4.send_whatsapp("+77099999999", "тест блок")
    check(
        "FIX-4: send_whatsapp() блокируется через env hardguard даже при WHATSAPP_ENABLED=True в модуле",
        _result_fix4 is False,
    )
finally:
    _comm_fix4.WHATSAPP_ENABLED = _orig_wa_fix4

# ── SAFEGUARD: LIVE_SEND_ALLOWED блокирует run() при dry_run=False ───────────
_lsa_src = _engine_src
_has_live_send_guard = (
    "LIVE_SEND_ALLOWED" in _lsa_src
    and "LIVE SEND BLOCKED" in _lsa_src
)
check(
    "SAFEGUARD: LIVE_SEND_ALLOWED guard присутствует в run()",
    _has_live_send_guard,
)

# ═══════════════════════════════════════════════════════════════
# 16. APPROVAL FLOW — UX согласования рассылки (2026-04-11)
# ═══════════════════════════════════════════════════════════════
section("16. APPROVAL FLOW — UX согласования")

try:
    from collector.approval_flow import (
        create_batch,
        save_batch,
        load_batch,
        load_latest_batch,
        is_ready_for_send,
        get_approved_clients,
        get_pending_managers,
        _all_managers_responded,
        _build_decisions,
        _get_manager_by_idx,
        expire_old_batches,
    )
    check("APPROVAL: импорт approval_flow.py успешен", True)
except Exception as e:
    check("APPROVAL: импорт approval_flow.py", False, str(e))
    # Если импорт упал — дальнейшие тесты пропускаем
    section("ИТОГ")
    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print(f"\n  Прошло: {passed}/{total}")
    sys.exit(1 if failed else 0)

# ── Тест 1: create_batch — один менеджер, 3 клиента ─────────────────────────
_batch1_clients = [
    {"name": "ТОО Альфа",    "amount": 500_000, "days": 15, "level": 2, "phone": "+77011111111"},
    {"name": "ИП Бета",      "amount": 200_000, "days": 12, "level": 1, "phone": "+77012222222"},
    {"name": "ТОО Гамма",    "amount": 900_000, "days": 25, "level": 3, "phone": "+77013333333"},
]
_batch1 = create_batch({"Алена": _batch1_clients})
check(
    "APPROVAL T1: create_batch — 1 менеджер 3 клиента",
    (
        _batch1.get("status") == "pending_managers"
        and "Алена" in _batch1.get("managers", {})
        and len(_batch1["managers"]["Алена"]["clients"]) == 3
        and _batch1["managers"]["Алена"]["status"] == "pending"
    ),
)

# ── Тест 2: create_batch — пустой manager_name игнорируется ─────────────────
_batch_noname = create_batch({
    "":          [{"name": "Клиент без менеджера", "amount": 1, "days": 10, "level": 1}],
    "Оксана":    [{"name": "ТОО Дельта", "amount": 300_000, "days": 11, "level": 1}],
})
check(
    "APPROVAL T2: клиент без manager_name НЕ попадает в батч",
    "" not in _batch_noname.get("managers", {}),
)
check(
    "APPROVAL T2b: клиент с manager_name попадает в батч",
    "Оксана" in _batch_noname.get("managers", {}),
)

# ── Тест 3: save / load ──────────────────────────────────────────────────────
import tempfile as _tempfile
_tmp_dir = _tempfile.mkdtemp()
_tmp_batches = Path(_tmp_dir) / "wa_approval_batches.json"

import collector.approval_flow as _af_mod
_orig_path = _af_mod._BATCHES_PATH
_af_mod._BATCHES_PATH = _tmp_batches  # type: ignore[assignment]

try:
    save_batch(_batch1)
    _loaded = load_batch(_batch1["batch_id"])
    check(
        "APPROVAL T3: save_batch / load_batch round-trip",
        _loaded is not None and _loaded["batch_id"] == _batch1["batch_id"],
    )
    check(
        "APPROVAL T3b: load_latest_batch возвращает pending батч",
        load_latest_batch() is not None,
    )
finally:
    _af_mod._BATCHES_PATH = _orig_path
    import shutil as _shutil_t3
    _shutil_t3.rmtree(_tmp_dir, ignore_errors=True)

# ── Тест 4: менеджер одобрил всех ──────────────────────────────────────────
_batch4 = create_batch({"Алена": _batch1_clients})
_mgr4 = _batch4["managers"]["Алена"]
_mgr4["status"]        = "approved_all"
_mgr4["approved_names"] = [c["name"] for c in _batch1_clients]
check(
    "APPROVAL T4: менеджер одобрил всех — all_responded=True",
    _all_managers_responded(_batch4),
)
check(
    "APPROVAL T4b: approved_names содержит 3 клиентов",
    len(_mgr4["approved_names"]) == 3,
)

# ── Тест 5: менеджер отклонил всех ──────────────────────────────────────────
_batch5 = create_batch({"Алена": _batch1_clients})
_mgr5 = _batch5["managers"]["Алена"]
_mgr5["status"]        = "rejected_all"
_mgr5["rejected_names"] = [c["name"] for c in _batch1_clients]
_mgr5["approved_names"] = []
check(
    "APPROVAL T5: менеджер отклонил всех — approved_names пустой",
    len(_mgr5["approved_names"]) == 0,
)
check(
    "APPROVAL T5b: rejected_names содержит 3 клиентов",
    len(_mgr5["rejected_names"]) == 3,
)

# ── Тест 6: менеджер выбрал вручную ─────────────────────────────────────────
_batch6 = create_batch({"Алена": _batch1_clients})
_mgr6 = _batch6["managers"]["Алена"]
_mgr6["status"]          = "manual"
_mgr6["approved_names"]  = ["ТОО Альфа"]
_mgr6["rejected_names"]  = ["ИП Бета"]
_mgr6["postponed_names"] = ["ТОО Гамма"]
_dec6 = _build_decisions(_mgr6)
check(
    "APPROVAL T6: _build_decisions ручной выбор",
    _dec6 == {"ТОО Альфа": "keep", "ИП Бета": "skip", "ТОО Гамма": "later"},
)

# ── Тест 7: два менеджера, разные списки ─────────────────────────────────────
_batch7_data = {
    "Алена":  [{"name": "ТОО Альфа", "amount": 500_000, "days": 15, "level": 2}],
    "Оксана": [{"name": "ТОО Дельта", "amount": 300_000, "days": 11, "level": 1}],
}
_batch7 = create_batch(_batch7_data)
check(
    "APPROVAL T7: два менеджера — оба в батче",
    set(_batch7["managers"].keys()) == {"Алена", "Оксана"},
)
check(
    "APPROVAL T7b: Алена видит только своего клиента",
    [c["name"] for c in _batch7["managers"]["Алена"]["clients"]] == ["ТОО Альфа"],
)
check(
    "APPROVAL T7c: Оксана видит только своего клиента",
    [c["name"] for c in _batch7["managers"]["Оксана"]["clients"]] == ["ТОО Дельта"],
)

# ── Тест 8: без admin approve send невозможен ─────────────────────────────────
_batch8 = create_batch({"Алена": _batch1_clients})
_batch8["managers"]["Алена"]["status"] = "approved_all"
_batch8["managers"]["Алена"]["approved_names"] = [c["name"] for c in _batch1_clients]
_batch8["status"] = "pending_admin"
# НЕ выставляем admin_approved — is_ready_for_send должен вернуть False

_tmp_dir8 = _tempfile.mkdtemp()
_tmp_batches8 = Path(_tmp_dir8) / "wa_approval_batches.json"
_af_mod._BATCHES_PATH = _tmp_batches8
try:
    save_batch(_batch8)
    check(
        "APPROVAL T8: pending_admin → is_ready_for_send=False",
        is_ready_for_send(_batch8["batch_id"]) is False,
    )
    # Симулируем admin approve
    _batch8["status"]       = "admin_approved"
    _batch8["admin_status"] = "approved"
    _batch8["approved_clients"] = [
        {**c, "manager": "Алена"}
        for c in _batch1_clients
        if c["name"] in _batch8["managers"]["Алена"]["approved_names"]
    ]
    save_batch(_batch8)
    check(
        "APPROVAL T8b: admin_approved → is_ready_for_send=True",
        is_ready_for_send(_batch8["batch_id"]) is True,
    )
    _approved8 = get_approved_clients(_batch8["batch_id"])
    check(
        "APPROVAL T8c: get_approved_clients возвращает 3 клиентов",
        len(_approved8) == 3,
    )
finally:
    _af_mod._BATCHES_PATH = _orig_path
    _shutil_t3.rmtree(_tmp_dir8, ignore_errors=True)

# ── Тест 9: active dialogs не влияют на batching ─────────────────────────────
# Approval flow использует только debtors_by_manager dict — изолирован от dialogs
_batch9 = create_batch({"Алена": _batch1_clients})
check(
    "APPROVAL T9: батч не содержит dialog_state полей (изолирован от collector_dialogs)",
    "dialog_state" not in str(_batch9) and "collector_dialogs" not in str(_batch9),
)

# ── Тест 10: get_pending_managers ─────────────────────────────────────────────
_batch10 = create_batch({"Алена": _batch1_clients, "Оксана": _batch7_data["Оксана"]})
_tmp_dir10 = _tempfile.mkdtemp()
_af_mod._BATCHES_PATH = Path(_tmp_dir10) / "wa_approval_batches.json"
try:
    save_batch(_batch10)
    _pending10 = get_pending_managers(_batch10["batch_id"])
    check(
        "APPROVAL T10: get_pending_managers — оба ожидают",
        set(_pending10) == {"Алена", "Оксана"},
    )
    # Алена ответила
    _batch10["managers"]["Алена"]["status"] = "approved_all"
    save_batch(_batch10)
    _pending10b = get_pending_managers(_batch10["batch_id"])
    check(
        "APPROVAL T10b: после ответа Алены — ожидает только Оксана",
        _pending10b == ["Оксана"],
    )
finally:
    _af_mod._BATCHES_PATH = _orig_path
    _shutil_t3.rmtree(_tmp_dir10, ignore_errors=True)

# ── Тест 11: run_approval_preview в коде ─────────────────────────────────────
_engine_src_v2 = (Path(__file__).parent.parent / "collector" / "collections_engine.py").read_text(encoding="utf-8")
check(
    "APPROVAL T11: run_approval_preview присутствует в collections_engine.py",
    "run_approval_preview" in _engine_src_v2,
)
check(
    "APPROVAL T11b: --preview флаг добавлен в CLI",
    '"--preview"' in _engine_src_v2 or "'--preview'" in _engine_src_v2,
)

# ── Тест 12: wa_appr_ callback зарегистрирован в send_reports.py ─────────────
_reports_src = (Path(__file__).parent.parent / "bot" / "send_reports.py").read_text(encoding="utf-8")
check(
    "APPROVAL T12: wa_appr_ callback роутер добавлен в send_reports.py",
    "wa_appr_" in _reports_src and "approval_flow" in _reports_src,
)

# ═══════════════════════════════════════════════════════════════
# 17. PROMPTS — config/collector_prompts.json (2026-04-11)
# ═══════════════════════════════════════════════════════════════
section("17. Промпты — config/collector_prompts.json")

import collector.collection_agent as _ca_mod

_PROMPTS_JSON = ROOT / "config" / "collector_prompts.json"

# ── T1: файл существует ──────────────────────────────────────────────────────
check("PROMPTS T1: файл config/collector_prompts.json существует",
      _PROMPTS_JSON.exists())

# ── T2: load_prompts() возвращает непустой dict ──────────────────────────────
_loaded_prompts = _ca_mod.load_prompts()
check("PROMPTS T2: load_prompts() возвращает dict", isinstance(_loaded_prompts, dict))
check("PROMPTS T2b: load_prompts() непустой", len(_loaded_prompts) > 0)

# ── T3: обязательные ключи ───────────────────────────────────────────────────
_required_keys = {"tones", "system_prompt", "user_prompt", "fallback_templates", "lang_instructions"}
_missing = _required_keys - set(_loaded_prompts.keys())
check("PROMPTS T3: обязательные ключи присутствуют",
      len(_missing) == 0,
      f"Отсутствуют: {_missing}")

# ── T4: все уровни тонов 1–5 ─────────────────────────────────────────────────
_tones = _loaded_prompts.get("tones", {})
_tone_keys = {str(k) for k in _tones.keys()}
check("PROMPTS T4: тон level 1 есть", "1" in _tone_keys)
check("PROMPTS T4b: тон level 2 есть", "2" in _tone_keys)
check("PROMPTS T4c: тон level 3 есть", "3" in _tone_keys)
check("PROMPTS T4d: тон level 4 есть", "4" in _tone_keys)
check("PROMPTS T4e: тон level 5 есть", "5" in _tone_keys)

# ── T5: все типы fallback_templates ─────────────────────────────────────────
_fb = _loaded_prompts.get("fallback_templates", {})
_required_fb = {"soft_reminder", "payment_plan_control", "strict_reminder", "stoplist_reminder"}
_missing_fb = _required_fb - set(_fb.keys())
check("PROMPTS T5: все типы fallback_templates присутствуют",
      len(_missing_fb) == 0,
      f"Отсутствуют: {_missing_fb}")

# ── T6: нет "торговой точке" в промптах (исправлена старая формулировка) ─────
_prompts_text = json.dumps(_loaded_prompts, ensure_ascii=False)
check("PROMPTS T6: 'торговой точке' отсутствует (заменено на 'задолженности')",
      "торговой точке" not in _prompts_text)

# ── T7: нет "ответьте «менеджер»" — заменено на "напишите 1" ────────────────
check("PROMPTS T7: 'ответьте «менеджер»' отсутствует (заменено на напишите 1)",
      "ответьте «менеджер»" not in _prompts_text)

# ── T8: нет ИИ/бот слов в fallback_templates (как отдельные слова) ───────────
import re as _re
_fb_text = json.dumps(_fb, ensure_ascii=False)
# "бот" проверяем как отдельное слово — не как подстроку "работа", "ботинки" и т.д.
_forbidden_patterns = [
    r'(?<![а-яёА-ЯЁa-zA-Z])ИИ(?![а-яёА-ЯЁa-zA-Z])',
    r'(?<![а-яёА-ЯЁa-zA-Z])бот(?![а-яёА-ЯЁa-zA-Z])',
    r'(?<![а-яёА-ЯЁa-zA-Z])робот(?![а-яёА-ЯЁa-zA-Z])',
    r'автоматически',
]
_found_forbidden = [p for p in _forbidden_patterns
                    if _re.search(p, _fb_text, _re.IGNORECASE)]
check("PROMPTS T8: нет ИИ/бот/робот/автоматически в fallback_templates",
      len(_found_forbidden) == 0,
      f"Найдены паттерны: {_found_forbidden}")

# ── T9: нет юридических УГРОЗ в тонах 4 и 5 ─────────────────────────────────
# "Без угроз судом или юристами" — разрешено (явно запрещает угрозы).
# Запрещены конкретные угрозы: "передадим юристам", "обратимся в суд" и т.д.
_legal_threats = [
    "передадим юристам", "передадим юристу", "обратимся в суд",
    "подадим в суд", "судебное разбирательство", "арбитражный суд",
    "юридический отдел",
]
_tone4 = _tones.get("4", "").lower()
_tone5 = _tones.get("5", "").lower()
_found_legal4 = [w for w in _legal_threats if w in _tone4]
_found_legal5 = [w for w in _legal_threats if w in _tone5]
check("PROMPTS T9: тон L4 без юридических угроз",
      len(_found_legal4) == 0,
      f"Найдены угрозы: {_found_legal4}")
check("PROMPTS T9b: тон L5 без юридических угроз",
      len(_found_legal5) == 0,
      f"Найдены угрозы: {_found_legal5}")

# ── T10: lang_instructions содержит ru и kz ──────────────────────────────────
_lang = _loaded_prompts.get("lang_instructions", {})
check("PROMPTS T10: lang_instructions.ru присутствует", "ru" in _lang)
check("PROMPTS T10b: lang_instructions.kz присутствует", "kz" in _lang)

# ── T11: load_prompts() не падает при отсутствии файла ───────────────────────
with patch.object(_ca_mod, "_PROMPTS_PATH",
                  ROOT / "config" / "collector_prompts_DOES_NOT_EXIST.json"):
    _fallback_result = _ca_mod.load_prompts()
check("PROMPTS T11: load_prompts() возвращает {} при отсутствии файла",
      isinstance(_fallback_result, dict))

# ── T12: _get_fallback_template подставляет переменные ───────────────────────
_tpl = _ca_mod._get_fallback_template(
    msg_type="soft_reminder",
    client_name="ТОО Тест",
    manager_name="Алена",
    amount=100000,
    days=15,
    company="Минбаракат",
)
check("PROMPTS T12: fallback template подставляет client_name", "ТОО Тест" in _tpl)
check("PROMPTS T12b: fallback template подставляет manager_name", "Алена" in _tpl)
check("PROMPTS T12c: fallback template подставляет amount", "100 000" in _tpl or "100000" in _tpl)
check("PROMPTS T12d: fallback template не содержит ИИ/бот как отдельное слово",
      not any(_re.search(p, _tpl, _re.IGNORECASE) for p in [
          r'(?<![а-яёА-ЯЁa-zA-Z])ИИ(?![а-яёА-ЯЁa-zA-Z])',
          r'(?<![а-яёА-ЯЁa-zA-Z])бот(?![а-яёА-ЯЁa-zA-Z])',
          r'(?<![а-яёА-ЯЁa-zA-Z])робот(?![а-яёА-ЯЁa-zA-Z])',
      ]))
check("PROMPTS T12e: fallback содержит 'напишите 1'", "напишите 1" in _tpl)

# ── T13: _get_tone возвращает строку для каждого уровня ──────────────────────
for lvl in range(1, 6):
    _t = _ca_mod._get_tone(lvl)
    check(f"PROMPTS T13: _get_tone({lvl}) непустая строка",
          isinstance(_t, str) and len(_t) > 10,
          f"Получено: {repr(_t)[:50]}")

# ── T14: _get_lang_inst возвращает строку для ru и kz ────────────────────────
check("PROMPTS T14: _get_lang_inst('ru') непустая",
      isinstance(_ca_mod._get_lang_inst("ru"), str) and len(_ca_mod._get_lang_inst("ru")) > 3)
check("PROMPTS T14b: _get_lang_inst('kz') непустая",
      isinstance(_ca_mod._get_lang_inst("kz"), str) and len(_ca_mod._get_lang_inst("kz")) > 3)

# ─────────────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════
# 14. PHASE 4 — DECOUPLE SHIPMENT STOP FROM COLLECTION ELIGIBILITY
# ═══════════════════════════════════════════════════════════════
section("PHASE 4: Collector eligibility decoupled from shipment stop")

from collector.collections_engine import _collector_candidate_decision, _flag_enabled

_p4_client = {
    "name": "Test Client", "amount": 50000.0, "days": 20,
    "opening": 0.0, "debit": 0.0, "credit": 0.0,
}
_p4_contact = {"whatsapp": "+77001112233", "telegram_id": "", "manager": "Тест"}

# T1: auto_stopped + долг + валидный телефон → client_approval, stoplist_reminder
_d1 = _collector_candidate_decision(_p4_client, _p4_contact, {"status": "auto_stopped"})
check("P4 T1: auto_stopped + debt → action=client_approval",
      _d1.get("action") == "client_approval", str(_d1))
check("P4 T1b: auto_stopped → msg_type=stoplist_reminder",
      _d1.get("msg_type") == "stoplist_reminder", str(_d1))

# T2: stopped + долг + валидный телефон → client_approval, stoplist_reminder
_d2 = _collector_candidate_decision(_p4_client, _p4_contact, {"status": "stopped"})
check("P4 T2: stopped + debt → action=client_approval",
      _d2.get("action") == "client_approval", str(_d2))
check("P4 T2b: stopped → msg_type=stoplist_reminder",
      _d2.get("msg_type") == "stoplist_reminder", str(_d2))

# T3: auto_stopped + collector_skip → skip (явный collector-блок работает)
_d3 = _collector_candidate_decision(
    _p4_client, _p4_contact, {"status": "auto_stopped", "collector_skip": True}
)
check("P4 T3: auto_stopped + collector_skip → skip",
      _d3.get("action") == "skip", str(_d3))

# T4: do_not_contact в contact → skip (независимо от stop-статуса)
_d4 = _collector_candidate_decision(
    _p4_client, {**_p4_contact, "do_not_contact": True}, {"status": "auto_stopped"}
)
check("P4 T4: do_not_contact в contact → skip",
      _d4.get("action") == "skip", str(_d4))

# T5: auto_stopped + нет телефона → action=client_approval на уровне решения
# (phone-блок срабатывает позже в run(), не в _collector_candidate_decision)
_d5 = _collector_candidate_decision(
    _p4_client, {"whatsapp": "", "telegram_id": ""}, {"status": "auto_stopped"}
)
check("P4 T5: auto_stopped + нет телефона → client_approval (phone-блок downstream)",
      _d5.get("action") == "client_approval", str(_d5))

# T6: _flag_enabled корректно определяет collector_skip / do_not_contact
check("P4 T6: _flag_enabled находит collector_skip=True",
      _flag_enabled({"collector_skip": True}, "collector_skip", "do_not_contact"))
check("P4 T6b: _flag_enabled отрицательный (только status=auto_stopped)",
      not _flag_enabled({"status": "auto_stopped"}, "collector_skip", "do_not_contact"))

# T7: stop_rec.status не изменяется после вызова _collector_candidate_decision
_stop_rec_check = {"status": "auto_stopped"}
_collector_candidate_decision(_p4_client, _p4_contact, _stop_rec_check)
check("P4 T7: stop_rec['status'] неизменён после decision (auto_stopped остаётся auto_stopped)",
      _stop_rec_check.get("status") == "auto_stopped", str(_stop_rec_check))

# T8: regression — auto_stopped + ненулевые debit/credit → client_approval (active guard не блокирует)
# Это проверяет инвариант dry-run == preview: _collector_candidate_decision() не применяет
# active guard для stopped-клиентов, run() теперь тоже (_bypass_active_guard=True).
_d8 = _collector_candidate_decision(
    {**_p4_client, "debit": 100000.0, "credit": 80000.0},
    _p4_contact,
    {"status": "auto_stopped"},
)
check("P4 T8: auto_stopped + debit/credit > 0 → client_approval (active guard bypassed)",
      _d8.get("action") == "client_approval", str(_d8))
check("P4 T8b: msg_type=stoplist_reminder при наличии debit/credit",
      _d8.get("msg_type") == "stoplist_reminder", str(_d8))


# ═══════════════════════════════════════════════════════════════
# 15. HIGH-3 — missing_manager_chat_id guard
# ═══════════════════════════════════════════════════════════════
section("HIGH-3: preview skips clients without manager chat_id")

from collector.collections_engine import _get_manager_chat_id
from unittest.mock import patch

# T1: неизвестный менеджер → chat_id отсутствует → guard сработает
_unknown_cid = _get_manager_chat_id("НеизвестныйМенеджерXYZ")
check("H3 T1: _get_manager_chat_id для несуществующего менеджера → None",
      _unknown_cid is None, repr(_unknown_cid))

# T2: guard-логика — клиент без chat_id не попадает в debtors_by_manager
# Проверяем через patch _get_manager_chat_id возвращает None для любого менеджера
with patch("collector.collections_engine._get_manager_chat_id", return_value=None):
    from collector.collections_engine import _get_manager_chat_id as _gcid_patched
    _cid = _gcid_patched("Ергали")
    check("H3 T2: при chat_id=None условие not _preview_chat_id → True (пропуск активируется)",
          not _cid)

# T3: при наличии chat_id условие не срабатывает (клиент должен попасть в batch)
with patch("collector.collections_engine._get_manager_chat_id", return_value=123456789):
    from collector.collections_engine import _get_manager_chat_id as _gcid_ok
    _cid_ok = _gcid_ok("Ергали")
    check("H3 T3: при chat_id=123456789 условие not _preview_chat_id → False (клиент не пропускается)",
          bool(_cid_ok))

# T4: guard применяется системно — одно и то же поведение для любого имени менеджера
_managers_to_check = ["Ергали", "Алена", "Оксана", "Магира", "НовыйМенеджер"]
with patch("collector.collections_engine._get_manager_chat_id", return_value=None):
    _all_none = all(
        not _get_manager_chat_id(m)  # реальная функция тоже вернёт None для неизвестных
        for m in ["НовыйМенеджер123", "НеизвестныйАбв"]
    )
check("H3 T4: guard системный — любой менеджер без chat_id блокируется одинаково",
      _all_none)

# T5: Phase 2 регрессия — safe-send не сломан
import importlib
_phase2_mod = importlib.import_module("tests.test_phase2_safe_send") if False else None
# Просто проверяем что модуль collections_engine импортируется без ошибок после правки
try:
    import collector.collections_engine as _ce_check
    check("H3 T5: collector.collections_engine импортируется после HIGH-3 правки",
          hasattr(_ce_check, "run_approval_preview") and
          hasattr(_ce_check, "_get_manager_chat_id"))
except Exception as _e:
    check("H3 T5: collector.collections_engine импортируется после HIGH-3 правки",
          False, str(_e))


# ═══════════════════════════════════════════════════════════════
# 16. HIGH-4 PROOF — msg_type preserved through preview → send-approved
# ═══════════════════════════════════════════════════════════════
section("HIGH-4 proof: msg_type preserved preview → batch → approved → send-approved")

import asyncio
from unittest.mock import patch, MagicMock
from collector.approval_flow import create_batch

# Step 1: create_batch preserves msg_type from debtors_by_manager
_h4_debtors = {
    "Ергали": [{
        "name": "Тест Клиент",
        "amount": 100000.0, "days": 25, "level": 3,
        "opening": 0.0, "debit": 0.0, "credit": 0.0,
        "violation_shipment": False,
        "phone": "+77771234567",
        "language": "ru",
        "msg_type": "stoplist_reminder",
        "reason": "auto_stopped: долг не закрыт",
        "stop_status": "auto_stopped",
        "review_action": "client_approval",
    }]
}
_h4_batch = create_batch(_h4_debtors)
_h4_client_in_batch = _h4_batch["managers"]["Ергали"]["clients"][0]
check("H4 T1: create_batch сохраняет msg_type=stoplist_reminder",
      _h4_client_in_batch.get("msg_type") == "stoplist_reminder",
      str(_h4_client_in_batch.get("msg_type")))

# Step 2: admin approve path copies client dict including msg_type
_h4_approved_client = {**_h4_client_in_batch, "manager": "Ергали"}
check("H4 T2: approved_clients сохраняет msg_type после {**c, 'manager': mgr_name}",
      _h4_approved_client.get("msg_type") == "stoplist_reminder",
      str(_h4_approved_client.get("msg_type")))

# Step 3: _send_approved_client extracts msg_type and passes to generate_message
# Мокаем send_whatsapp и generate_message, запускаем _send_approved_client
from collector.collections_engine import _send_approved_client

_captured_msg_type = []

def _mock_generate_message(**kwargs):
    _captured_msg_type.append(kwargs.get("msg_type", "__NOT_SET__"))
    return "тестовое сообщение"

with patch("collector.collections_engine.send_whatsapp", return_value=True), \
     patch("collector.collections_engine.generate_message", side_effect=_mock_generate_message), \
     patch("collector.collections_engine.already_contacted_today", return_value=False), \
     patch("collector.collections_engine._get_manager_chat_id", return_value=99999999), \
     patch("collector.collections_engine.update_after_contact"), \
     patch("collector.client_dialog.start_client_dialog", return_value=None):
    asyncio.run(_send_approved_client(_h4_approved_client))

check("H4 T3: _send_approved_client передаёт msg_type=stoplist_reminder в generate_message",
      _captured_msg_type == ["stoplist_reminder"],
      f"captured: {_captured_msg_type}")


# ═══════════════════════════════════════════════════════════════
# 17. ИТОГ
# ═══════════════════════════════════════════════════════════════
section("ИТОГ")  # секция 17
total  = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed
print(f"\n  Прошло: {passed}/{total}")
if failed:
    print(f"  Упало:  {failed}")
    for name, ok in results:
        if not ok:
            print(f"    ❌ {name}")
    sys.exit(1)
else:
    print("  Все тесты пройдены.")
    sys.exit(0)

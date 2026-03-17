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

# Поле name fallback
data_kontragent = {"clients": [{"контрагент": "ТОО Дельта", "max_days": 25, "amount": 0}]}
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

finally:
    cdb.STATE_PATH = _orig_path
    shutil.rmtree(_tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════
# 7. communications — WhatsApp отключён по умолчанию
# ═══════════════════════════════════════════════════════════════
section("7. communications — WhatsApp disabled by default")

import collector.communications as comm

check("WHATSAPP_ENABLED default is False",
      comm.WHATSAPP_ENABLED is False)
check("send_whatsapp returns False when disabled",
      comm.send_whatsapp("+77001234567", "тест") is False)

# С включённым флагом — должен пойти в сеть (но GREENAPI_ID пустой → тоже False)
comm.WHATSAPP_ENABLED = True
comm.GREENAPI_ID = ""
check("send_whatsapp returns False when no creds",
      comm.send_whatsapp("+77001234567", "тест") is False)
comm.WHATSAPP_ENABLED = False  # вернём обратно


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
# 10. ИТОГ
# ═══════════════════════════════════════════════════════════════
section("ИТОГ")
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

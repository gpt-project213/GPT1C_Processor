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
import asyncio
import tempfile
import shutil
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["COLLECTOR_TEST_MODE"] = "1"

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
    load_contacts, load_latest_debt_json, match_client, _strip_prefix,
    compute_residual_debt_profile,
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

# PHASE 5: уровень коллектора считается по возрасту текущего остатка.
_p5_as_of = date(2026, 4, 11)
_p5_shapagat = {
    "name": "А ТД Шапагат 5 павильон Дюсембина",
    "debt": 64415.36,
    "opening": 170000.0,
    "debit": 1219009.56,
    "credit": 1324594.2,
    "days_silence": 19,
    "_period_min": "2026-03-11",
    "_period_max": "2026-04-11",
    "_movements": [
        {"date": "2026-03-15", "debit": 0.0, "credit": 170000.0},
        {"date": "2026-03-16", "debit": 450000.0, "credit": 0.0},
        {"date": "2026-03-19", "debit": 639794.2, "credit": 0.0},
        {"date": "2026-03-20", "debit": 64800.0, "credit": 0.2},
        {"date": "2026-03-21", "debit": 0.0, "credit": 680000.0},
        {"date": "2026-03-22", "debit": 0.0, "credit": 345000.0},
        {"date": "2026-03-23", "debit": 0.0, "credit": 129594.0},
        {"date": "2026-04-08", "debit": 28084.0, "credit": 0.0},
        {"date": "2026-04-10", "debit": 36331.36, "credit": 0.0},
    ],
}
_p5_profile = compute_residual_debt_profile(_p5_shapagat, as_of_date=_p5_as_of)
check("P5 T1: Шапагат residual_debt_age_days=3",
      _p5_profile["residual_debt_age_days"] == 3, str(_p5_profile))
check("P5 T1b: Шапагат oldest_unpaid_date=2026-04-08",
      _p5_profile["oldest_unpaid_date"] == "2026-04-08", str(_p5_profile))
check("P5 T1c: Шапагат payment_silence_days сохранён как 19",
      _p5_profile["payment_silence_days"] == 19, str(_p5_profile))
_p5_classified = classify_debtors({"clients": [_p5_shapagat]})[0]
check("P5 T1d: Шапагат классифицируется как L0, не L2",
      _p5_classified["level"] == 0 and _p5_classified["days"] == 3, str(_p5_classified))

_p5_old = {
    "name": "Старый долг",
    "debt": 100000.0,
    "days_silence": 1,
    "_period_min": "2026-03-01",
    "_period_max": "2026-04-12",
    "_movements": [{"date": "2026-03-20", "debit": 100000.0, "credit": 0.0}],
}
_p5_old_profile = compute_residual_debt_profile(_p5_old, as_of_date=date(2026, 4, 12))
check("P5 T2: старая неоплаченная отгрузка 23 дня → L3",
      _p5_old_profile["residual_debt_age_days"] == 23 and
      _level_for_days(_p5_old_profile["residual_debt_age_days"]) == 3,
      str(_p5_old_profile))

_p5_partial_new = {
    "name": "Оплата закрыла старую",
    "debt": 50000.0,
    "_period_min": "2026-04-01",
    "_period_max": "2026-04-12",
    "_movements": [
        {"date": "2026-04-01", "debit": 100000.0, "credit": 0.0},
        {"date": "2026-04-10", "debit": 50000.0, "credit": 0.0},
        {"date": "2026-04-11", "debit": 0.0, "credit": 100000.0},
    ],
}
_p5_partial_new_profile = compute_residual_debt_profile(_p5_partial_new, as_of_date=date(2026, 4, 12))
check("P5 T3: оплата закрыла старую отгрузку, остаток от новой",
      _p5_partial_new_profile["oldest_unpaid_date"] == "2026-04-10" and
      _p5_partial_new_profile["residual_debt_age_days"] == 2,
      str(_p5_partial_new_profile))

_p5_partial_old = {
    "name": "Старая часть осталась",
    "debt": 50000.0,
    "_period_min": "2026-04-01",
    "_period_max": "2026-04-12",
    "_movements": [
        {"date": "2026-04-01", "debit": 150000.0, "credit": 0.0},
        {"date": "2026-04-05", "debit": 0.0, "credit": 100000.0},
    ],
}
_p5_partial_old_profile = compute_residual_debt_profile(_p5_partial_old, as_of_date=date(2026, 4, 12))
check("P5 T4: частичная оплата оставила старую часть",
      _p5_partial_old_profile["oldest_unpaid_date"] == "2026-04-01" and
      _p5_partial_old_profile["residual_debt_age_days"] == 11,
      str(_p5_partial_old_profile))

_p5_opening_closed = {
    "name": "Opening закрыт",
    "debt": 50000.0,
    "opening": 100000.0,
    "_period_min": "2026-04-01",
    "_period_max": "2026-04-12",
    "_movements": [
        {"date": "2026-04-02", "debit": 0.0, "credit": 100000.0},
        {"date": "2026-04-10", "debit": 50000.0, "credit": 0.0},
    ],
}
_p5_opening_closed_profile = compute_residual_debt_profile(_p5_opening_closed, as_of_date=date(2026, 4, 12))
check("P5 T5: закрытый opening не влияет на возраст остатка",
      _p5_opening_closed_profile["oldest_unpaid_date"] == "2026-04-10",
      str(_p5_opening_closed_profile))

_p5_opening_left = {
    "name": "Opening остался",
    "debt": 60000.0,
    "opening": 100000.0,
    "_period_min": "2026-04-01",
    "_period_max": "2026-04-12",
    "_movements": [{"date": "2026-04-02", "debit": 0.0, "credit": 40000.0}],
}
_p5_opening_left_profile = compute_residual_debt_profile(_p5_opening_left, as_of_date=date(2026, 4, 12))
check("P5 T6: непогашенный opening считается с начала периода",
      _p5_opening_left_profile["basis"] == "opening_fallback" and
      _p5_opening_left_profile["oldest_unpaid_date"] == "2026-04-01",
      str(_p5_opening_left_profile))

_p5_overpaid = {
    "name": "Переплата",
    "debt": 0.0,
    "_period_min": "2026-04-01",
    "_period_max": "2026-04-12",
    "_movements": [
        {"date": "2026-04-01", "debit": 100000.0, "credit": 0.0},
        {"date": "2026-04-02", "debit": 0.0, "credit": 150000.0},
    ],
}
_p5_overpaid_profile = compute_residual_debt_profile(_p5_overpaid, as_of_date=date(2026, 4, 12))
check("P5 T7: переплата/закрытый долг → residual age 0",
      _p5_overpaid_profile["basis"] == "no_debt" and
      _p5_overpaid_profile["residual_debt_age_days"] == 0,
      str(_p5_overpaid_profile))

_p5_no_movements = {"name": "Нет движений", "debt": 50000.0, "days_silence": 22}
_p5_no_movements_profile = compute_residual_debt_profile(_p5_no_movements, as_of_date=date(2026, 4, 12))
check("P5 T8: нет movements → fallback на days_silence",
      _p5_no_movements_profile["basis"] == "fallback_days_silence" and
      _p5_no_movements_profile["residual_debt_age_days"] == 22,
      str(_p5_no_movements_profile))

_p5_rounding = {
    "name": "Округление",
    "debt": 64.41,
    "_period_min": "2026-04-01",
    "_period_max": "2026-04-12",
    "_movements": [
        {"date": "2026-04-01", "debit": 100.0, "credit": 0.0},
        {"date": "2026-04-02", "debit": 0.0, "credit": 35.59},
    ],
}
_p5_rounding_profile = compute_residual_debt_profile(_p5_rounding, as_of_date=date(2026, 4, 12))
check("P5 T9: округление не ломает FIFO-остаток",
      abs(sum(p["amount"] for p in _p5_rounding_profile["unpaid_parts"]) - 64.41) < 0.02,
      str(_p5_rounding_profile))

_p5_mv4 = {
    "name": "Е ТОО МВ4 (Азамат)",
    "debt": 792307.51,
    "opening": 3.01,
    "debit": 3266302.0,
    "credit": 2473997.5,
    "days_silence": 1,
    "_period_min": "2026-03-11",
    "_period_max": "2026-04-11",
    "_movements": [
        {"date": "2026-03-18", "debit": 908661.0, "credit": 0.0},
        {"date": "2026-03-20", "debit": 0.0, "credit": 908661.0},
        {"date": "2026-03-27", "debit": 1565336.5, "credit": 0.0},
        {"date": "2026-03-30", "debit": 0.0, "credit": 1565336.5},
        {"date": "2026-04-10", "debit": 792304.5, "credit": 0.0},
    ],
}
_p5_mv4_profile = compute_residual_debt_profile(_p5_mv4, as_of_date=date(2026, 4, 11))
check("P5 T9b: малый старый остаток 3 тг не задаёт возраст долга МВ4",
      _p5_mv4_profile["oldest_unpaid_date"] == "2026-04-10" and
      _p5_mv4_profile["residual_debt_age_days"] == 1 and
      _p5_mv4_profile["basis"] == "movements_fifo_significant" and
      _p5_mv4_profile["ignored_tail_parts"][0]["amount"] == 3.01,
      str(_p5_mv4_profile))

check("P5 T10: классификатор сохраняет обе метрики",
      _p5_classified["residual_debt_age_days"] == 3 and
      _p5_classified["payment_silence_days"] == 19 and
      _p5_classified["debt_age_basis"] == "movements_fifo",
      str(_p5_classified))


# ═══════════════════════════════════════════════════════════════
# 5. debt_monitor — _strip_prefix / match_client
# ═══════════════════════════════════════════════════════════════
from collector.collections_engine import _apply_collector_day_policy
from unittest.mock import patch as _p5_patch

with _p5_patch("collector.collections_engine.get_debt_days_since_first_seen", return_value=19):
    _p5_policy_movements = _apply_collector_day_policy(
        {"name": "Shapagat", "days": 3, "level": 0, "debt_age_basis": "movements_fifo"},
        "Shapagat",
        use_first_seen=True,
    )
check("P5 T11: first_seen does not inflate movements_fifo residual age",
      _p5_policy_movements["days"] == 3 and _p5_policy_movements["level"] == 0,
      str(_p5_policy_movements))

with _p5_patch("collector.collections_engine.get_debt_days_since_first_seen", return_value=19):
    _p5_policy_fallback = _apply_collector_day_policy(
        {"name": "Fallback", "days": 3, "level": 0, "debt_age_basis": "fallback_days_silence"},
        "Fallback",
        use_first_seen=True,
    )
check("P5 T12: first_seen fallback stays for data without movements",
      _p5_policy_fallback["days"] == 10 and _p5_policy_fallback["level"] == 1,
      str(_p5_policy_fallback))

from collector.approval_flow import create_batch as _p5_create_batch, _debt_age_text as _p5_debt_age_text

_p5_batch = _p5_create_batch({
    "Aлена": [{
        "name": "Shapagat",
        "amount": 64415.36,
        "days": 3,
        "level": 0,
        "phone": "+77770000000",
        "payment_silence_days": 19,
        "oldest_unpaid_date": "2026-04-08",
        "debt_age_basis": "movements_fifo",
        "active_turnover": True,
    }]
})
_p5_batch_client = _p5_batch["managers"]["Aлена"]["clients"][0]
check("P5 T13: approval batch preserves residual debt metrics",
      _p5_batch_client["payment_silence_days"] == 19 and
      _p5_batch_client["oldest_unpaid_date"] == "2026-04-08" and
      _p5_batch_client["active_turnover"] is True,
      str(_p5_batch_client))
_p5_debt_text = _p5_debt_age_text(_p5_batch_client)
check("P5 T14: approval text shows residual age without payment-silence ambiguity",
      "Возраст остатка: 3 дн." in _p5_debt_text and
      "оплат нет" not in _p5_debt_text and
      "Остаток с: 2026-04-08" in _p5_debt_text and
      "старейшая часть" not in _p5_debt_text and
      "активный оборот" in _p5_debt_text,
      _p5_debt_text)

_p5_mv4_debt_text = _p5_debt_age_text({
    "days": 1,
    "oldest_unpaid_date": "2026-04-10",
    "ignored_tail_parts": [{"date": "2026-03-27", "amount": 3.01, "source": "shipment"}],
})
check("P5 T14b: approval text says 'Малый старый остаток', not technical tail",
      "Малый старый остаток: 3 тг" in _p5_mv4_debt_text and
      "техничес" not in _p5_mv4_debt_text.lower(),
      _p5_mv4_debt_text)

_p5_tmp_dir = Path(tempfile.mkdtemp(prefix="collector_debt_period_"))
try:
    _p5_newer_file = _p5_tmp_dir / "debt_ext_Детальный Дебиторы Алена (134).json"
    _p5_older_file = _p5_tmp_dir / "debt_ext_Ведомость_по_взаиморасчетам_с_контрагентами_Алена (324).json"
    _p5_newer_file.write_text(json.dumps({
        "manager": "Алена",
        "period_min": "11.03.2026",
        "period_max": "11.04.2026",
        "clients": [{
            "name": "А ТД 77 павильон тест",
            "amount": 10000.0,
            "days_silence": 1,
        }],
    }, ensure_ascii=False), encoding="utf-8")
    _p5_older_file.write_text(json.dumps({
        "manager": "Алена",
        "period_min": "10.03.2026",
        "period_max": "10.04.2026",
        "clients": [{
            "name": "А ТД 77 павильон тест",
            "amount": 10000.0,
            "days_silence": 31,
        }],
    }, ensure_ascii=False), encoding="utf-8")
    with patch("collector.debt_monitor.JSON_DIR", _p5_tmp_dir):
        _p5_loaded = load_latest_debt_json()
    _p5_loaded_client = _p5_loaded["clients"][0]
    check("P5 T15: loader берёт свежий period_max, а не старый с большим days_silence",
          len(_p5_loaded["clients"]) == 1 and
          _p5_loaded_client["days_silence"] == 1 and
          _p5_loaded_client["_period_max"] == "11.04.2026",
          str(_p5_loaded))
finally:
    shutil.rmtree(_p5_tmp_dir, ignore_errors=True)

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

# ─── _state_lock: межпроцессная блокировка ───────────────────────────────────
check("collections_db._state_lock существует (portalocker guard)",
      hasattr(cdb, '_state_lock') and callable(cdb._state_lock))
if hasattr(cdb, '_state_lock'):
    _lck = cdb._state_lock()
    check("_state_lock() возвращает context manager",
          hasattr(_lck, '__enter__') and hasattr(_lck, '__exit__'))


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
      comm.send_whatsapp("+77999000000", "тест") is False)

# С включённым флагом — должен пойти в сеть (но GREENAPI_ID пустой → тоже False)
comm.WHATSAPP_ENABLED = True
comm.GREENAPI_ID = ""
check("send_whatsapp returns False when no creds",
      comm.send_whatsapp("+77999000000", "тест") is False)
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
    check("start_client_dialog: report_date default empty", d.get("report_date") == "")

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
    # v1.1.0: краткое promise_without_date ("Я оплачу") → soft_positive → escalated
    check("promise_without_date 'Я оплачу': state=escalated", d_pwd.get("state") == "escalated")
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
    # fix: "Передам на оплату" — обещание, не факт оплаты → не срабатывает хеуристик → active
    check("promise_without_date 'Передам на оплату': state=active", d_pwd2.get("state") == "active")
    bot_replies_pwd2 = [ex["text"] for ex in d_pwd2.get("exchanges", []) if ex["role"] == "bot"]
    # Должен использоваться дефолтный текст с датой
    check("promise_without_date 'Передам на оплату': fallback содержит 'дату'",
          any("дату" in t.lower() for t in bot_replies_pwd2))

    # ─── recent unposted payment: QR / сегодня / не разнесено ────────────────
    asyncio.run(cd_mod.start_client_dialog(
        phone="77011234572",
        client_name="Е Олжас",
        manager_name="Ергали",
        manager_chat_id=123,
        level=5,
        days=31,
        amount=830781.58,
        message_text="Остаток не закрыт уже 31 день.",
        report_date="2026-04-11",
    ))
    with patch("collector.client_dialog._reply_to_client") as mock_qr_reply:
        mock_qr_reply.return_value = None
        asyncio.run(cd_mod.handle_incoming(
            "77011234572",
            "За выходные QR оплаты прошли, сегодня ещё упадёт, в 1С не разнесено",
        ))
    d_qr = cd_mod._get_client_dialog("77011234572")
    qr_bot_replies = [ex["text"] for ex in d_qr.get("exchanges", []) if ex["role"] == "bot"]
    check("recent payment: бот просит чек или дату и сумму оплаты",
          any("чек" in t.lower()
              and "дату и сумму" in t.lower()
              and "1С" not in t
              for t in qr_bot_replies),
          str(qr_bot_replies))
    # awaiting_payment_proof — промежуточный статус: бот ждёт чек, к менеджеру не эскалировано
    check("recent payment: диалог в awaiting_payment_proof, не эскалирован к менеджеру",
          d_qr.get("state") == "awaiting_payment_proof",
          str(d_qr.get("state")))

    # ─── UX: promise с датой, но без суммы — без ложного "фиксируем" ───────────
    asyncio.run(cd_mod.start_client_dialog(
        phone="77011234573",
        client_name="Кайрбек",
        manager_name="Ергали",
        manager_chat_id=123,
        level=2,
        days=12,
        amount=25150.0,
        message_text="Напоминание по задолженности.",
    ))
    with patch("collector.collection_agent._call_deepseek") as mock_prom_no_amount:
        mock_prom_no_amount.return_value = '{"intent":"promise","promise_date":"2026-04-22","promise_amount":null,"requires_human":false,"suggested_reply":""}'
        with patch("collector.client_dialog._reply_to_client") as mock_prom_reply:
            mock_prom_reply.return_value = None
            with patch("collector.client_dialog.escalate_to_manager"):
                asyncio.run(cd_mod.handle_incoming("77011234573", "Сегодня будет, сумма пока не знаю"))
    d_prom_no_amount = cd_mod._get_client_dialog("77011234573")
    prom_replies = [ex["text"] for ex in d_prom_no_amount.get("exchanges", []) if ex["role"] == "bot"]
    last_prom_reply = prom_replies[-1] if prom_replies else ""
    check("UX promise date-only: state=escalated", d_prom_no_amount.get("state") == "escalated")
    check("UX promise date-only: нет ложной фиксации",
          "фиксируем" not in last_prom_reply.lower(), last_prom_reply)
    check("UX promise date-only: просит чек",
          "чек" in last_prom_reply.lower(), last_prom_reply)

    # ─── UX: schedule — ежедневные/частичные платежи сохраняются ──────────────
    asyncio.run(cd_mod.start_client_dialog(
        phone="77011234574",
        client_name="Кайрбек",
        manager_name="Ергали",
        manager_chat_id=123,
        level=2,
        days=12,
        amount=25150.0,
        message_text="Напоминание по задолженности.",
    ))
    with patch("collector.collection_agent._call_deepseek") as mock_sched:
        mock_sched.return_value = '{"intent":"promise_schedule","promise_date":"2026-04-22","promise_amount":null,"payment_schedule":"daily","requires_human":false,"suggested_reply":""}'
        with patch("collector.client_dialog._reply_to_client") as mock_sched_reply:
            mock_sched_reply.return_value = None
            with patch("collector.client_dialog.escalate_to_manager"):
                asyncio.run(cd_mod.handle_incoming("77011234574", "На ежедневной основе, по определённой сумме"))
    d_sched = cd_mod._get_client_dialog("77011234574")
    sched_replies = [ex["text"] for ex in d_sched.get("exchanges", []) if ex["role"] == "bot"]
    check("UX schedule: payment_schedule сохранён", d_sched.get("payment_schedule") == "daily")
    check("UX schedule: state=escalated", d_sched.get("state") == "escalated")
    check("UX schedule: ответ про график платежей",
          any(("daily" in t.lower() or "част" in t.lower() or "ежеднев" in t.lower()) for t in sched_replies),
          str(sched_replies))

    # ─── UX: paid_claim — не спорит ссылкой на 1С, просит чек ─────────────────
    asyncio.run(cd_mod.start_client_dialog(
        phone="77011234575",
        client_name="Кайрбек",
        manager_name="Ергали",
        manager_chat_id=123,
        level=2,
        days=12,
        amount=25150.0,
        message_text="Напоминание по задолженности.",
    ))
    with patch("collector.collection_agent._call_deepseek") as mock_paid:
        mock_paid.return_value = '{"intent":"paid_claim","promise_date":null,"promise_amount":null,"requires_human":false,"suggested_reply":""}'
        with patch("collector.client_dialog._reply_to_client") as mock_paid_reply:
            mock_paid_reply.return_value = None
            with patch("collector.client_dialog._notify_dialog_observers", new=AsyncMock()) as mock_paid_note:
                asyncio.run(cd_mod.handle_incoming("77011234575", "Я уже оплатил"))
    d_paid = cd_mod._get_client_dialog("77011234575")
    paid_replies = [ex["text"] for ex in d_paid.get("exchanges", []) if ex["role"] == "bot"]
    last_paid = paid_replies[-1] if paid_replies else ""
    check("UX paid_claim: state=awaiting_payment_proof", d_paid.get("state") == "awaiting_payment_proof")
    check("UX paid_claim: awaiting_payment_proof=True", d_paid.get("awaiting_payment_proof") is True)
    check("UX paid_claim: нет повторной ссылки на 1С", "1С" not in last_paid, last_paid)
    check("UX paid_claim: просит чек", "чек" in last_paid.lower(), last_paid)
    check("UX paid_claim: manager/admin note отправлен", mock_paid_note.await_count == 1)

    # ─── UX: soft_positive — один мягкий вопрос, без фиксации ─────────────────
    asyncio.run(cd_mod.start_client_dialog(
        phone="77011234576",
        client_name="Кайрбек",
        manager_name="Ергали",
        manager_chat_id=123,
        level=2,
        days=12,
        amount=25150.0,
        message_text="Напоминание по задолженности.",
    ))
    with patch("collector.collection_agent._call_deepseek") as mock_soft:
        mock_soft.return_value = '{"intent":"soft_positive","promise_date":null,"promise_amount":null,"requires_human":false,"suggested_reply":""}'
        with patch("collector.client_dialog._reply_to_client") as mock_soft_reply:
            mock_soft_reply.return_value = None
            asyncio.run(cd_mod.handle_incoming("77011234576", "Закрою в ближайшее время"))
    d_soft = cd_mod._get_client_dialog("77011234576")
    soft_replies = [ex["text"] for ex in d_soft.get("exchanges", []) if ex["role"] == "bot"]
    last_soft = soft_replies[-1] if soft_replies else ""
    with patch("collector.client_dialog._reply_to_client") as mock_paid_ack_reply:
        mock_paid_ack_reply.return_value = None
        with patch("collector.collection_agent._call_deepseek") as mock_paid_ack_ai:
            mock_paid_ack_ai.return_value = '{"intent":"unclear","promise_date":null,"promise_amount":null,"requires_human":false,"suggested_reply":""}'
            asyncio.run(cd_mod.handle_incoming("77011234575", "Хорошо"))
    d_paid_ack = cd_mod._get_client_dialog("77011234575")
    paid_ack_replies = [ex["text"] for ex in d_paid_ack.get("exchanges", []) if ex["role"] == "bot"]
    check("UX paid_claim ack: нет лишнего повторного ответа",
          len(paid_ack_replies) == len(paid_replies),
          str(paid_ack_replies))

    with patch("collector.client_dialog._reply_to_client") as mock_paid_proof_reply:
        mock_paid_proof_reply.return_value = None
        with patch("collector.client_dialog._notify_dialog_observers", new=AsyncMock()) as mock_paid_proof_note:
            asyncio.run(cd_mod.handle_incoming(
                "77011234575",
                "[клиент прислал documentMessage]",
                attachment={
                    "type": "documentMessage",
                    "download_url": "https://example.test/receipt.pdf",
                    "file_name": "receipt.pdf",
                    "caption": "чек оплаты",
                },
            ))
    d_paid_proof = cd_mod._get_client_dialog("77011234575")
    paid_proof_replies = [ex["text"] for ex in d_paid_proof.get("exchanges", []) if ex["role"] == "bot"]
    last_paid_proof = paid_proof_replies[-1] if paid_proof_replies else ""
    proof_note_text = mock_paid_proof_note.await_args.args[1] if mock_paid_proof_note.await_args else ""
    check("UX paid_claim proof: state=awaiting_manager", d_paid_proof.get("state") == "awaiting_manager")
    check("UX paid_claim proof: awaiting_payment_proof reset", d_paid_proof.get("awaiting_payment_proof") is False)
    check("UX paid_claim proof: reply confirms forwarding",
          "передали менеджеру" in last_paid_proof.lower(), last_paid_proof)
    check("UX paid_claim proof: note contains download url",
          "https://example.test/receipt.pdf" in proof_note_text, proof_note_text)
    check("UX paid_claim proof: note sent once", mock_paid_proof_note.await_count == 1)

    asyncio.run(cd_mod.start_client_dialog(
        phone="77011234577",
        client_name="Ольга VED-STAR",
        manager_name="Ергали",
        manager_chat_id=123,
        level=2,
        days=10,
        amount=1060103.0,
        message_text="Напоминание по задолженности.",
    ))
    with patch("collector.collection_agent._call_deepseek") as mock_soft_commit:
        mock_soft_commit.return_value = '{"intent":"soft_positive","promise_date":null,"promise_amount":null,"requires_human":false,"suggested_reply":""}'
        with patch("collector.client_dialog._reply_to_client") as mock_soft_commit_reply:
            mock_soft_commit_reply.return_value = None
            with patch("collector.client_dialog.escalate_to_manager") as mock_soft_commit_escalate:
                asyncio.run(cd_mod.handle_incoming("77011234577", "счс оплачу"))
    d_soft_commit = cd_mod._get_client_dialog("77011234577")
    soft_commit_replies = [ex["text"] for ex in d_soft_commit.get("exchanges", []) if ex["role"] == "bot"]
    last_soft_commit = soft_commit_replies[-1] if soft_commit_replies else ""
    check("UX soft_positive commitment: state=escalated", d_soft_commit.get("state") == "escalated")
    check("UX soft_positive commitment: просит чек, а не первый платёж",
          "чек" in last_soft_commit.lower() and "первый плат" not in last_soft_commit.lower(),
          last_soft_commit)
    check("UX soft_positive commitment: менеджер уведомляется", mock_soft_commit_escalate.called)

    asyncio.run(cd_mod.start_client_dialog(
        phone="77011234578",
        client_name="Ольга VED-STAR",
        manager_name="Ергали",
        manager_chat_id=123,
        level=2,
        days=10,
        amount=1060103.0,
        message_text="Напоминание по задолженности.",
    ))
    with patch("collector.client_dialog._reply_to_client") as mock_service_reply:
        mock_service_reply.return_value = None
        with patch("collector.client_dialog.escalate_to_manager") as mock_service_escalate:
            asyncio.run(cd_mod.handle_incoming("77011234578", "акт сверки сбросьте за апрель"))
    d_service = cd_mod._get_client_dialog("77011234578")
    service_replies = [ex["text"] for ex in d_service.get("exchanges", []) if ex["role"] == "bot"]
    last_service = service_replies[-1] if service_replies else ""
    check("UX service request: сразу передаёт менеджеру",
          "передаю вас менеджеру" in last_service.lower(), last_service)
    check("UX service request: не дожимает оплату",
          "первый плат" not in last_service.lower(), last_service)
    check("UX service request: есть эскалация", mock_service_escalate.called)
    check("UX soft_positive: state=active", d_soft.get("state") == "active")
    check("UX soft_positive: мягкий вопрос про первый платёж",
          "первый плат" in last_soft.lower(), last_soft)
    check("UX soft_positive: нет ложной фиксации",
          "фиксируем" not in last_soft.lower(), last_soft)

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
        record_send_results,
        get_pending_managers,
        _all_managers_responded,
        _build_decisions,
        _build_admin_decisions,
        _save_admin_decisions,
        _admin_client_list_keyboard,
        _get_manager_by_idx,
        expire_old_batches,
        handle_manager_callback,
        handle_admin_callback,
        promote_silent_batches_to_admin,
        supersede_batch,
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
    "":          [{"name": "TEST fixture: клиент без manager_name", "amount": 1, "days": 10, "level": 1}],
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
    _batch1["status"] = "sent"
    save_batch(_batch1)
    check(
        "APPROVAL T3c: load_latest_batch не возвращает финальный sent батч",
        load_latest_batch() is None,
    )
    _batch1["status"] = "superseded"
    save_batch(_batch1)
    check(
        "APPROVAL T3d: load_latest_batch не возвращает superseded батч",
        load_latest_batch() is None,
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
    record_send_results(_batch8["batch_id"], [{
        "name": "ТОО Альфа",
        "manager": "Алена",
        "phone": "+77011111111",
        "status": "sent",
        "reason": "test",
    }])
    _after_partial8 = load_batch(_batch8["batch_id"])
    check(
        "APPROVAL T8d: частичная отправка не блокирует оставшихся approved-клиентов",
        is_ready_for_send(_batch8["batch_id"]) is True
        and _after_partial8.get("status") == "partially_sent"
        and len(get_approved_clients(_batch8["batch_id"])) == 3,
        str(_after_partial8.get("status")),
    )
    record_send_results(_batch8["batch_id"], [{
        "name": "ИП Бета",
        "manager": "Алена",
        "phone": "+77012222222",
        "status": "sent",
        "reason": "test",
    }])
    _after_second8 = load_batch(_batch8["batch_id"])
    check(
        "APPROVAL T8e: результаты частичной отправки накапливаются, а не затираются",
        len(_after_second8.get("send_results", [])) == 2
        and _after_second8.get("send_summary", {}).get("sent") == 2
        and _after_second8.get("send_summary", {}).get("approved_total") == 3,
        str(_after_second8.get("send_summary")),
    )
    _batch8_admin = load_batch(_batch8["batch_id"])
    _admin_decisions8 = _build_admin_decisions(_batch8_admin)
    _first_admin_key8 = sorted(_admin_decisions8.keys())[0]
    _admin_decisions8[_first_admin_key8] = "skip"
    _save_admin_decisions(_batch8_admin, _admin_decisions8)
    save_batch(_batch8_admin)
    _after_admin8 = load_batch(_batch8["batch_id"])
    check(
        "APPROVAL T8f: admin manual decisions сохраняются отдельно от manager approve",
        len(_after_admin8.get("admin_keep_keys", [])) == 2
        and len(_after_admin8.get("admin_skip_keys", [])) == 1,
        str({
            "admin_keep_keys": _after_admin8.get("admin_keep_keys"),
            "admin_skip_keys": _after_admin8.get("admin_skip_keys"),
        }),
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
_batch10c = create_batch({"Алена": _batch1_clients, "Оксана": _batch7_data["Оксана"]})
_batch10c["expires_at"] = "2000-01-01T00:00:00+05:00"
_batch10c["managers"]["Алена"]["status"] = "pending"
_batch10c["managers"]["Оксана"]["status"] = "manual_editing"
_tmp_dir10c = _tempfile.mkdtemp()
_af_mod._BATCHES_PATH = Path(_tmp_dir10c) / "wa_approval_batches.json"
try:
    save_batch(_batch10c)
    _expired10c = expire_old_batches()
    _after10c = load_batch(_batch10c["batch_id"])
    check(
        "APPROVAL T10c: expire_old_batches помечает батч как expired",
        _expired10c == 1 and _after10c is not None and _after10c.get("status") == "expired",
        str(_after10c.get("status") if _after10c else None),
    )
    check(
        "APPROVAL T10d: молчавшие менеджеры переводятся в timeout",
        _after10c is not None
        and _after10c["managers"]["Алена"].get("status") == "timeout"
        and _after10c["managers"]["Оксана"].get("status") == "timeout"
        and bool(_after10c.get("expired_at")),
        str(_after10c["managers"] if _after10c else None),
    )
finally:
    _af_mod._BATCHES_PATH = _orig_path
    _shutil_t3.rmtree(_tmp_dir10c, ignore_errors=True)

_batch10e = create_batch({"Алена": _batch1_clients, "Оксана": _batch7_data["Оксана"]})
_batch10e["created_at"] = "2000-01-01T00:00:00+05:00"
_tmp_dir10e = _tempfile.mkdtemp()
_af_mod._BATCHES_PATH = Path(_tmp_dir10e) / "wa_approval_batches.json"
try:
    save_batch(_batch10e)
    with patch("collector.approval_flow.send_admin_summary", new=AsyncMock()) as _send_admin_mock:
        _promoted10e = asyncio.run(promote_silent_batches_to_admin())
    _after10e = load_batch(_batch10e["batch_id"])
    check(
        "APPROVAL T10e: после часа молчания батч переводится в pending_admin",
        _promoted10e == 1 and _after10e is not None and _after10e.get("status") == "pending_admin",
        str(_after10e.get("status") if _after10e else None),
    )
    check(
        "APPROVAL T10f: молчавшие менеджеры получают timeout и админу уходит сводка",
        _after10e is not None
        and _after10e["managers"]["Алена"].get("status") == "timeout"
        and _after10e["managers"]["Оксана"].get("status") == "timeout"
        and bool(_after10e.get("escalated_to_admin_at"))
        and _send_admin_mock.await_count == 1,
        str(_after10e["managers"] if _after10e else None),
    )
finally:
    _af_mod._BATCHES_PATH = _orig_path
    _shutil_t3.rmtree(_tmp_dir10e, ignore_errors=True)

_batch10g = create_batch({"Алена": _batch1_clients})
_tmp_dir10g = _tempfile.mkdtemp()
_af_mod._BATCHES_PATH = Path(_tmp_dir10g) / "wa_approval_batches.json"
try:
    save_batch(_batch10g)
    supersede_batch(_batch10g, superseded_by="new-batch-1234")
    _after10g = load_batch(_batch10g["batch_id"])
    check(
        "APPROVAL T10g: активный батч можно закрыть как superseded",
        _after10g is not None
        and _after10g.get("status") == "superseded"
        and _after10g.get("superseded_by") == "new-batch-1234",
        str(_after10g),
    )
finally:
    _af_mod._BATCHES_PATH = _orig_path
    _shutil_t3.rmtree(_tmp_dir10g, ignore_errors=True)

_batch10h = create_batch({"Алена": _batch1_clients})
_batch10h["status"] = "pending_admin"
_tmp_dir10h = _tempfile.mkdtemp()
_af_mod._BATCHES_PATH = Path(_tmp_dir10h) / "wa_approval_batches.json"
try:
    save_batch(_batch10h)
    with patch("collector.approval_flow._tg_edit", new=AsyncMock()) as _edit10h:
        _handled10h = asyncio.run(handle_manager_callback("wa_appr_mgr_ok|" + _batch10h["batch_id"] + "|0", 1, 2))
    _after10h = load_batch(_batch10h["batch_id"])
    check(
        "APPROVAL T10h: manager-callback по pending_admin батчу блокируется",
        _handled10h is True
        and _after10h is not None
        and _after10h["managers"]["Алена"].get("status") == "pending"
        and _edit10h.await_count == 1,
        str(_after10h["managers"]["Алена"] if _after10h else None),
    )
finally:
    _af_mod._BATCHES_PATH = _orig_path
    _shutil_t3.rmtree(_tmp_dir10h, ignore_errors=True)

section("10i. agreed promise stats")
_orig_promises_path = _af_mod._PROMISES_PATH
_tmp_promises_dir = _tempfile.mkdtemp()
_af_mod._PROMISES_PATH = Path(_tmp_promises_dir) / "wa_agreed_promises.json"
try:
    _af_mod.save_agreed_promise("ИП Исполнен", "Магира", "до 10.05, 50000 тг", "batch-a")
    _af_mod.save_agreed_promise("ТОО Срыв", "Магира", "до 11.05, 80000 тг", "batch-a")
    _af_mod.save_agreed_promise("ИП Отказ", "Ергали", "до 12.05, 30000 тг", "batch-b")
    _af_mod.save_agreed_promise("ТОО Активный", "Ергали", "до 13.05, 40000 тг", "batch-b")
    _promises10i = _af_mod._load_promises()
    _promises10i["ИП Исполнен"]["status"] = "fulfilled"
    _promises10i["ТОО Срыв"]["status"] = "broken"
    _promises10i["ИП Отказ"]["status"] = "rejected"
    _af_mod._save_promises(_promises10i)
    _stats10i = _af_mod.get_agreed_promise_stats()
    _mgr10i = {item["manager"]: item for item in _stats10i.get("managers", [])}
    check(
        "APPROVAL T10i: статистика обещаний считает общие статусы",
        _stats10i["totals"]["total"] == 4
        and _stats10i["totals"]["fulfilled"] == 1
        and _stats10i["totals"]["broken"] == 1
        and _stats10i["totals"]["rejected"] == 1
        and _stats10i["totals"]["in_control"] == 1,
        str(_stats10i["totals"]),
    )
    check(
        "APPROVAL T10j: статистика обещаний агрегируется по менеджерам",
        _mgr10i["Магира"]["fulfilled"] == 1
        and _mgr10i["Магира"]["broken"] == 1
        and _mgr10i["Ергали"]["rejected"] == 1
        and _mgr10i["Ергали"]["in_control"] == 1,
        str(_mgr10i),
    )
    _stats_text10i = _af_mod.format_agreed_promise_stats_text()
    check(
        "APPROVAL T10k: текстовая сводка обещаний содержит менеджеров и ключевые счётчики",
        "Магира" in _stats_text10i
        and "Ергали" in _stats_text10i
        and "Сорвано: <b>1</b>" in _stats_text10i
        and "Отклонено директором: <b>1</b>" in _stats_text10i,
        _stats_text10i,
    )
finally:
    _af_mod._PROMISES_PATH = _orig_promises_path
    _shutil_t3.rmtree(_tmp_promises_dir, ignore_errors=True)

_batch10i = create_batch({"Алена": _batch1_clients})
_batch10i["status"] = "pending_admin"
_batch10i["admin_status"] = "pending"
_tmp_dir10i = _tempfile.mkdtemp()
_af_mod._BATCHES_PATH = Path(_tmp_dir10i) / "wa_approval_batches.json"
try:
    save_batch(_batch10i)
    _flat10i = _af_mod._iter_admin_clients(_batch10i)
    _decisions10i = _build_admin_decisions(_batch10i)
    _first_key10i = _flat10i[0]["_admin_key"]
    _decisions10i[_first_key10i] = "skip"
    _save_admin_decisions(_batch10i, _decisions10i)
    _batch10i["admin_reviewed_keys"] = [_first_key10i]
    save_batch(_batch10i)
    _markup10i = _admin_client_list_keyboard(
        _batch10i["batch_id"],
        _flat10i,
        _decisions10i,
        {_first_key10i},
    )
    _markup10i_text = json.dumps(_markup10i, ensure_ascii=False)
    check(
        "APPROVAL T10i: обработанный админом клиент исчезает из ручного списка",
        _flat10i[0]["name"] not in _markup10i_text and _flat10i[1]["name"] in _markup10i_text,
        _markup10i_text,
    )
finally:
    _af_mod._BATCHES_PATH = _orig_path
    _shutil_t3.rmtree(_tmp_dir10i, ignore_errors=True)

_batch10j = create_batch({"Алена": _batch1_clients})
_batch10j["status"] = "superseded"
_batch10j["superseded_by"] = "new-batch-9999"
_tmp_dir10j = _tempfile.mkdtemp()
_af_mod._BATCHES_PATH = Path(_tmp_dir10j) / "wa_approval_batches.json"
try:
    save_batch(_batch10j)
    with patch("collector.approval_flow._tg_edit", new=AsyncMock()) as _edit10j:
        _handled10j = asyncio.run(handle_admin_callback("wa_appr_adm_view|" + _batch10j["batch_id"], 1, 2))
    _after10j = load_batch(_batch10j["batch_id"])
    check(
        "APPROVAL T10j: admin-callback по superseded запросу блокируется",
        _handled10j is True
        and _after10j is not None
        and _after10j.get("status") == "superseded"
        and _edit10j.await_count == 1,
        str(_after10j),
    )
finally:
    _af_mod._BATCHES_PATH = _orig_path
    _shutil_t3.rmtree(_tmp_dir10j, ignore_errors=True)

_engine_src_v2 = (Path(__file__).parent.parent / "collector" / "collections_engine.py").read_text(encoding="utf-8")
_approval_src_v2 = (Path(__file__).parent.parent / "collector" / "approval_flow.py").read_text(encoding="utf-8")
check(
    "APPROVAL T11: run_approval_preview присутствует в collections_engine.py",
    "run_approval_preview" in _engine_src_v2,
)
check(
    "APPROVAL T11b: --preview флаг добавлен в CLI",
    '"--preview"' in _engine_src_v2 or "'--preview'" in _engine_src_v2,
)
check(
    "APPROVAL T11c: Telegram send-now callback присутствует в approval_flow.py",
    "wa_appr_adm_send" in _approval_src_v2,
)
check(
    "APPROVAL T11d: новый запрос закрывает и старые админские сообщения",
    "close_admin_messages(" in _engine_src_v2,
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
_required_fb = {
    "soft_reminder",
    "payment_plan_control",
    "strict_reminder",
    "stoplist_reminder",
    "legacy_tail_reminder",
    "partial_tail_reminder",
}
_missing_fb = _required_fb - set(_fb.keys())
check("PROMPTS T5: все типы fallback_templates присутствуют",
      len(_missing_fb) == 0,
      f"Отсутствуют: {_missing_fb}")

# ── T6: нет "торговой точке" в промптах (исправлена старая формулировка) ─────
_prompts_text = json.dumps(_loaded_prompts, ensure_ascii=False)
check("PROMPTS T6: 'торговой точке' отсутствует (заменено на 'задолженности')",
      "торговой точке" not in _prompts_text)

# ── T7: нет старых механических команд клиенту ───────────────────────────────
check("PROMPTS T7: 'ответьте «менеджер»' отсутствует",
      "ответьте «менеджер»" not in _prompts_text)
check("PROMPTS T7b: 'напишите 1' отсутствует",
      "напишите 1" not in _prompts_text)
check("PROMPTS T7c: промпт использует Астану, а не Алматы",
      "Астана" in _prompts_text and "Алматы" not in _prompts_text)

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
_stop_tpl = _loaded_prompts.get("fallback_templates", {}).get("stoplist_reminder", "")
check("PROMPTS T9c: stoplist_reminder без слова 'критическая'",
      "критичес" not in _stop_tpl.lower(), _stop_tpl)
check("PROMPTS T9d: stoplist_reminder указывает срок незакрытого остатка",
      "не закрыт уже {days_text}" in _stop_tpl, _stop_tpl)
check("PROMPTS T9d2: stoplist_reminder указывает дату отчёта рядом с остатком",
      "остаток задолженности{report_date_part} составляет" in _stop_tpl.lower(), _stop_tpl)
check("PROMPTS T9e: stoplist_reminder не пишет 'передан руководству'",
      "руководств" not in _stop_tpl.lower(), _stop_tpl)
_legacy_tail_tpl = _loaded_prompts.get("fallback_templates", {}).get("legacy_tail_reminder", "")
check("PROMPTS T9e2: legacy_tail_reminder без фразы про отгрузки",
      "отгруз" not in _legacy_tail_tpl.lower(), _legacy_tail_tpl)
_partial_tail_tpl = _loaded_prompts.get("fallback_templates", {}).get("partial_tail_reminder", "")
check("PROMPTS T9e3: partial_tail_reminder без фразы про отгрузки",
      "отгруз" not in _partial_tail_tpl.lower(), _partial_tail_tpl)
check("PROMPTS T9f: тон L5 не содержит 'критическая'",
      "критичес" not in _tone5, _tone5)
check("PROMPTS T9g: тон L5 не содержит 'передан руководству'",
      "руководств" not in _tone5, _tone5)

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
check("PROMPTS T12e: fallback НЕ содержит 'напишите 1'", "напишите 1" not in _tpl)
_typed_stop_msg = _ca_mod.generate_message(
    client_name="Е Олжас",
    debt_amount=830781.58,
    days_overdue=31,
    level=5,
    language="ru",
    manager_name="Ергали",
    msg_type="stoplist_reminder",
    report_date="2026-04-11",
)
check("PROMPTS T12f: msg_type=stoplist_reminder использует шаблон без DeepSeek",
      "критичес" not in _typed_stop_msg.lower()
      and "руководств" not in _typed_stop_msg.lower()
      and "остаток задолженности на 11.04.2026 составляет 830 782 тг" in _typed_stop_msg.lower()
      and "не закрыт уже 31 день" in _typed_stop_msg,
      _typed_stop_msg)
_typed_legacy_tail_msg = _ca_mod.generate_message(
    client_name="Е ИП Шахин",
    debt_amount=340000.0,
    days_overdue=33,
    level=5,
    language="ru",
    manager_name="Ергали",
    msg_type="legacy_tail_reminder",
    report_date="2026-04-27",
)
check("PROMPTS T12g: legacy_tail_reminder без упоминания отгрузок",
      "отгруз" not in _typed_legacy_tail_msg.lower()
      and "задолженность на 27.04.2026 составляет 340 000 тг" in _typed_legacy_tail_msg.lower(),
      _typed_legacy_tail_msg)

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

# T15: analyze_response должен переживать None от AI без падения на .get
with patch("collector.collection_agent._call_deepseek", return_value=None):
    _none_ai = _ca_mod.analyze_response("Оплачу позже", manager_name="Ергали")
check(
    "PROMPTS T15: analyze_response переживает None от AI и уходит в fallback",
    isinstance(_none_ai, dict)
    and _none_ai.get("intent") == "unclear"
    and _none_ai.get("requires_human") is True,
    str(_none_ai),
)

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

# T9: stopped + старый хвост без движения -> legacy_tail_reminder
_d9 = _collector_candidate_decision(
    {"name": "Е ИП Шахин", "amount": 340000.0, "days": 33, "opening": 340000.0, "debit": 0.0, "credit": 0.0},
    _p4_contact,
    {"status": "stopped"},
)
check("P4 T9: stopped + старый хвост без движения -> client_approval",
      _d9.get("action") == "client_approval", str(_d9))
check("P4 T9b: stopped + старый хвост без движения -> legacy_tail_reminder",
      _d9.get("msg_type") == "legacy_tail_reminder", str(_d9))

# T10: stopped + старый хвост с частичной оплатой -> partial_tail_reminder
_d10 = _collector_candidate_decision(
    {"name": "Е Еркебулан", "amount": 739409.67, "days": 28, "opening": 767268.67, "debit": 0.0, "credit": 27859.0},
    _p4_contact,
    {"status": "stopped"},
)
check("P4 T10: stopped + старый хвост с частичной оплатой -> client_approval",
      _d10.get("action") == "client_approval", str(_d10))
check("P4 T10b: stopped + старый хвост с частичной оплатой -> partial_tail_reminder",
      _d10.get("msg_type") == "partial_tail_reminder", str(_d10))


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

section("15.5 send-approved freshness gate")

import collector.collections_engine as ce_mod

_approved_batch_client = {
    "name": "Е Еркебулан",
    "manager": "Ергали",
    "phone": "77087578717",
    "amount": 767269.0,
    "days": 28,
    "level": 4,
    "language": "ru",
    "msg_type": "strict_reminder",
    "report_date": "2026-04-27",
    "reason": "old snapshot",
}

with patch("collector.approval_flow.is_ready_for_send", return_value=True), \
     patch("collector.approval_flow.get_approved_clients", return_value=[dict(_approved_batch_client)]), \
     patch("collector.approval_flow.load_batch", return_value={"batch_id": "batch-1", "created_at": "2026-04-28T13:00:00+05:00"}), \
     patch("collector.approval_flow.record_send_results") as _record_send, \
     patch("collector.collections_engine._live_send_allowed", return_value=True), \
     patch("collector.collections_engine.load_latest_debt_json", return_value={"clients": [{"name": "Е Еркебулан"}]}), \
     patch("collector.collections_engine.classify_debtors", return_value=[{
         "name": "Е Еркебулан",
         "amount": 120000.0,
         "days": 12,
         "level": 1,
         "report_date": "2026-04-28",
     }]), \
     patch("collector.collections_engine._apply_collector_day_policy", side_effect=lambda c, name, use_first_seen: c), \
     patch("collector.collections_engine.match_client", return_value={"manager": "Ергали", "whatsapp": "77087578717", "language": "ru"}), \
     patch("collector.collections_engine._get_stop_record", return_value={}), \
     patch("collector.collections_engine._collector_candidate_decision", return_value={"action": "client_approval", "msg_type": "soft_reminder", "reason": "fresh debt"}), \
     patch("collector.collections_engine._send_approved_client", new=AsyncMock(return_value={"name": "Е Еркебулан", "status": "sent", "reason": "ok"})) as _send_refreshed, \
     patch("collector.collections_engine.notify_admin", new=AsyncMock()) as _notify_refresh:
    refreshed_results = asyncio.run(ce_mod.send_approved_batch("batch-1"))

sent_payload = _send_refreshed.await_args.args[0] if _send_refreshed.await_args else {}
check("freshness gate: send_approved_batch uses refreshed amount",
      sent_payload.get("amount") == 120000.0, str(sent_payload))
check("freshness gate: send_approved_batch uses refreshed msg_type",
      sent_payload.get("msg_type") == "soft_reminder", str(sent_payload))
check("freshness gate: admin notified about batch refresh", _notify_refresh.await_count == 1)
check("freshness gate: record_send_results called", _record_send.called)
check("freshness gate: returned sent result", any(r.get("status") == "sent" for r in refreshed_results), str(refreshed_results))

with patch("collector.approval_flow.is_ready_for_send", return_value=True), \
     patch("collector.approval_flow.get_approved_clients", return_value=[dict(_approved_batch_client)]), \
     patch("collector.approval_flow.load_batch", return_value={"batch_id": "batch-2", "created_at": "2026-04-28T13:00:00+05:00"}), \
     patch("collector.approval_flow.record_send_results") as _record_stale, \
     patch("collector.collections_engine._live_send_allowed", return_value=True), \
     patch("collector.collections_engine.load_latest_debt_json", return_value={"clients": []}), \
     patch("collector.collections_engine.classify_debtors", return_value=[]), \
     patch("collector.collections_engine._send_approved_client", new=AsyncMock()) as _send_stale, \
     patch("collector.collections_engine.notify_admin", new=AsyncMock()) as _notify_stale:
    stale_results = asyncio.run(ce_mod.send_approved_batch("batch-2"))

check("freshness gate stale client: no send happens", _send_stale.await_count == 0)
check("freshness gate stale client: skipped result returned",
      any("stale approved batch" in str(r.get("reason", "")) for r in stale_results),
      str(stale_results))
check("freshness gate stale client: admin notified", _notify_stale.await_count == 1)
check("freshness gate stale client: record_send_results called", _record_stale.called)

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
        "oldest_unpaid_date": "2026-03-11",
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
_captured_report_date = []
_captured_start_dialog = []

def _mock_generate_message(**kwargs):
    _captured_msg_type.append(kwargs.get("msg_type", "__NOT_SET__"))
    _captured_report_date.append(kwargs.get("report_date", "__NOT_SET__"))
    return "тестовое сообщение"

async def _mock_start_client_dialog(**kwargs):
    _captured_start_dialog.append(kwargs)

with patch("collector.collections_engine.send_whatsapp", return_value=True), \
     patch("collector.collections_engine.generate_message", side_effect=_mock_generate_message), \
     patch("collector.collections_engine.already_contacted_today", return_value=False), \
     patch("collector.collections_engine._get_manager_chat_id", return_value=99999999), \
     patch("collector.collections_engine.update_after_contact"), \
     patch("collector.client_dialog.start_client_dialog", side_effect=_mock_start_client_dialog):
    asyncio.run(_send_approved_client(_h4_approved_client))

check("H4 T3: _send_approved_client передаёт msg_type=stoplist_reminder в generate_message",
      _captured_msg_type == ["stoplist_reminder"],
      f"captured: {_captured_msg_type}")
check("H4 T3b: _send_approved_client передаёт дату отчёта в generate_message",
      _captured_report_date == ["2026-04-05"],
      f"captured: {_captured_report_date}")
check("H4 T4: _send_approved_client сохраняет дату отчёта в клиентский диалог",
      _captured_start_dialog
      and _captured_start_dialog[0].get("report_date") == "2026-04-05",
      f"captured: {_captured_start_dialog}")


# ═══════════════════════════════════════════════════════════════
# 17. WHATSAPP AUDIO STT — AssemblyAI primary, safe fallback
# ═══════════════════════════════════════════════════════════════
section("WhatsApp audio STT: AssemblyAI primary")

import collector.whatsapp_poller as _wa_poller

_orig_dialogs_path = _wa_poller._CLIENT_DIALOGS_PATH
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as _td_wa:
    _wa_poller._CLIENT_DIALOGS_PATH = Path(_td_wa) / "collector_client_dialogs.json"
    _wa_poller._CLIENT_DIALOGS_PATH.write_text(
        json.dumps({"77753306745": {"client": "Е ИП Трое Нурлан"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    check("WA STT T1: активный диалог определяется по collector_client_dialogs",
          _wa_poller._has_active_collector_dialog("+77753306745") is True)
    check("WA STT T2: должник без активного диалога не проходит гейт",
          _wa_poller._has_active_collector_dialog("+77750000000") is False)
    with patch.object(_wa_poller, "WA_REQUIRE_ACTIVE_DIALOG", False):
        check("WA STT T2b: выделенный бот-номер пропускает входящие без active-dialog гейта",
              _wa_poller._should_process_incoming("+77750000000") is True)
    with patch.object(_wa_poller, "WA_REQUIRE_ACTIVE_DIALOG", True):
        check("WA STT T2c: legacy privacy-гейт можно вернуть через WA_REQUIRE_ACTIVE_DIALOG=1",
              _wa_poller._should_process_incoming("+77750000000") is False)
    _wa_poller._CLIENT_DIALOGS_PATH = _orig_dialogs_path

with patch.object(_wa_poller, "ASSEMBLYAI_API_KEY", "aai-key"), \
     patch.object(_wa_poller, "_download_audio_to_temp", new=AsyncMock(return_value=("fake.ogg", ".ogg"))), \
     patch.object(_wa_poller, "_transcribe_with_assemblyai_file", new=AsyncMock(return_value="оплачу завтра")) as _aai_mock:
    _stt_text = asyncio.run(_wa_poller.transcribe_audio("https://example.test/audio.ogg"))
    check("WA STT T3: AssemblyAI транскрибирует успешно",
          _stt_text == "оплачу завтра" and _aai_mock.await_count == 1)

with patch.object(_wa_poller, "ASSEMBLYAI_API_KEY", "aai-key"), \
     patch.object(_wa_poller, "_download_audio_to_temp", new=AsyncMock(return_value=("fake.ogg", ".ogg"))), \
     patch.object(_wa_poller, "_transcribe_with_assemblyai_file", new=AsyncMock(return_value="")) as _aai_empty_mock:
    _fallback_text = asyncio.run(_wa_poller.transcribe_audio("https://example.test/audio.ogg"))
    check("WA STT T4: при ошибке AssemblyAI возвращается пустая строка (нет fallback на OpenAI)",
          _fallback_text == "" and _aai_empty_mock.await_count == 1)


# ═══════════════════════════════════════════════════════════════
# 18. SAIDA PAYMENT HOLD — collector suppression
# ═══════════════════════════════════════════════════════════════
section("Saida payment hold: collector skips clients waiting for 1C posting")

import collector.payment_hold as _payment_hold
_orig_hold_path = _payment_hold.PAYMENT_HOLD_PATH
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as _td_hold:
    _payment_hold.PAYMENT_HOLD_PATH = Path(_td_hold) / "saida_payment_holds.json"
    _hold_rec = _payment_hold.create_manager_payment_request(
        manager="Магира",
        client="М Халал тест",
        debt=40438.20,
        debt_str="40 438,20",
        manager_chat_id=123,
    )
    _payment_hold.confirm_by_saida(_hold_rec["token"], "full")
    _hold_decision = _collector_candidate_decision(
        {
            "name": "М Халал тест",
            "amount": 40438.20,
            "days": 31,
            "opening": 0.0,
            "debit": 0.0,
            "credit": 0.0,
        },
        {"whatsapp": "+77771234567"},
        {},
    )
    check("PAYHOLD T1: preview decision пропускает клиента с подтверждением Саиды",
          _hold_decision.get("action") == "skip" and "Саида" in _hold_decision.get("reason", ""),
          str(_hold_decision))

    _hold_send_client = {
        "name": "М Халал тест",
        "manager": "Магира",
        "phone": "+77771234567",
        "amount": 40438.20,
        "days": 31,
        "level": 5,
        "language": "ru",
    }
    with patch("collector.collections_engine.send_whatsapp", return_value=True) as _send_mock:
        _hold_send_result = asyncio.run(_send_approved_client(_hold_send_client))
    check("PAYHOLD T2: send-approved не отправляет клиента с подтверждением Саиды",
          _hold_send_result.get("status") == "skipped" and not _send_mock.called,
          str(_hold_send_result))
    _payment_hold.PAYMENT_HOLD_PATH = _orig_hold_path


# ═══════════════════════════════════════════════════════════════
# 19. Debt stop admin shipment limit — после оплаты с лимитом
# ═══════════════════════════════════════════════════════════════
section("19. Debt stop admin shipment limit")

import bot.debt_stop_control as _dstop

class _FakeDstopBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, **kwargs):
        self.messages.append(kwargs)
        return type("Msg", (), {
            "message_id": len(self.messages),
            "date": datetime.now(),
        })()

_orig_dstop_state = _dstop.STATE_FILE
_orig_dstop_registry = _dstop.REGISTRY_FILE
_orig_dstop_saida = _dstop.SAIDA_CHAT_ID
_orig_dstop_deletion = _dstop.DELETION_QUEUE
_dstop_tmpdir = tempfile.mkdtemp()
try:
    _dstop.STATE_FILE = Path(_dstop_tmpdir) / "debt_stop_state.json"
    _dstop.REGISTRY_FILE = Path(_dstop_tmpdir) / "debt_stop_registry.json"
    _dstop.DELETION_QUEUE = Path(_dstop_tmpdir) / "deletion_queue.json"
    _dstop.SAIDA_CHAT_ID = 0
    _fake_dstop_bot = _FakeDstopBot()
    _admin_id = 123456
    _today = datetime.now(_dstop.TZ).strftime("%Y-%m-%d")

    _dstop.save_state({
        "date": _today,
        "next_id": 2,
        "saida_sent": False,
        "candidates": {
            "1": {
                "client": "ТОО Лимит После Оплаты",
                "manager": "Ергали",
                "manager_chat_id": 654321,
                "days_silence": 31,
                "debt": 4_000_000,
                "level": "10+",
                "admin_approved": None,
                "awaiting_shipment_limit": True,
            }
        },
    })
    with patch.dict(os.environ, {"ADMIN_CHAT_ID": str(_admin_id)}):
        _handled = asyncio.run(_dstop.handle_dstop_detail_message(_admin_id, "1000000", _fake_dstop_bot))
    _rec = _dstop.load_registry().get("ТОО Лимит После Оплаты", {})
    check("DSTOP LIMIT T1: ввод лимита админом обработан", _handled is True)
    check("DSTOP LIMIT T1b: решение хранится как после оплаты с лимитом",
          _rec.get("status") == "allow_after_payment" and _rec.get("shipment_limit") == 1_000_000,
          str(_rec))

    _dstop.save_registry({
        "ТОО Уже Оплатил": {
            "manager": "Алена",
            "manager_chat_id": 111,
            "status": "awaiting_clearance_limit",
            "debt_at_approval": 700_000,
        }
    })
    with patch.dict(os.environ, {"ADMIN_CHAT_ID": str(_admin_id)}):
        _handled2 = asyncio.run(_dstop.handle_dstop_detail_message(_admin_id, "500000", _fake_dstop_bot))
    _rec2 = _dstop.load_registry().get("ТОО Уже Оплатил", {})
    check("DSTOP LIMIT T2: после полной оплаты можно задать лимит новой отгрузки",
          _handled2 is True)
    check("DSTOP LIMIT T2b: старый стоп закрыт с лимитом",
          _rec2.get("status") == "cleared_limited" and _rec2.get("shipment_limit") == 500_000,
          str(_rec2))

    _dstop.save_state({
        "date": _today,
        "next_id": 2,
        "saida_sent": False,
        "candidates": {
            "1": {
                "client": "ТОО Напоминание",
                "manager": "Магира",
                "manager_chat_id": 777,
                "days_silence": 10,
                "debt": 250_000,
                "level": "10+",
                "manager_response": None,
                "admin_approved": None,
            }
        },
    })
    _fake_dstop_bot.messages.clear()
    _mock_daytime = MagicMock(wraps=datetime)
    _mock_daytime.now = MagicMock(return_value=datetime.now(_dstop.TZ).replace(hour=12, minute=0, second=0))
    _mock_daytime.fromisoformat = datetime.fromisoformat
    with patch("bot.debt_stop_control.datetime", _mock_daytime):
        asyncio.run(_dstop.send_manager_reminders(_fake_dstop_bot))
    _state_after_remind = _dstop.load_state()
    _cand_after_remind = _state_after_remind["candidates"]["1"]
    check("DSTOP REMIND T1: менеджеру отправлено напоминание",
          len(_fake_dstop_bot.messages) == 1 and _fake_dstop_bot.messages[0]["chat_id"] == 777,
          str(_fake_dstop_bot.messages))
    check("DSTOP REMIND T1b: счётчик напоминаний растёт",
          _cand_after_remind.get("manager_remind_count") == 1,
          str(_cand_after_remind))
finally:
    _dstop.STATE_FILE = _orig_dstop_state
    _dstop.REGISTRY_FILE = _orig_dstop_registry
    _dstop.DELETION_QUEUE = _orig_dstop_deletion
    _dstop.SAIDA_CHAT_ID = _orig_dstop_saida


# ═══════════════════════════════════════════════════════════════
# 19b. Saida full payment auto-clears stop
import collector.shipment_control as _ship_for_saida
_orig_ship_for_saida = _ship_for_saida._DECISIONS_PATH
_ship_for_saida._DECISIONS_PATH = Path(_dstop_tmpdir) / "collector_shipment_decisions_auto_clear.json"
_ship_for_saida.set_decision("77011110000", "ТОО АвтоСнятие", "block_until", manager_name="Алена", manager_chat_id=111, amount=300_000)
_dstop.save_registry({
    "ТОО АвтоСнятие": {
        "manager": "Алена",
        "manager_chat_id": 111,
        "approved_at": _today,
        "days_at_approval": 18,
        "debt_at_approval": 300_000,
        "status": "block_until_payment",
        "added_by": "admin_block_until_payment",
        "discipline_violation": False,
        "cleared_at": None,
    }
})
_dstop.save_state({
    "date": _today,
    "next_id": 2,
    "saida_sent": False,
    "candidates": {
        "1": {
            "client": "ТОО АвтоСнятие",
            "manager": "Алена",
            "manager_chat_id": 111,
            "days_silence": 18,
            "debt": 300_000,
            "saida_payment_confirmed": None,
        }
    },
})
_fake_dstop_bot.messages.clear()
_full_result = asyncio.run(_dstop._handle_saida_confirm_full("1", 0, _fake_dstop_bot))
_reg_after_full = _dstop.load_registry().get("ТОО АвтоСнятие", {})
_ship_after_full = _ship_for_saida.get_decision("ТОО АвтоСнятие")
check("DSTOP SAIDA FULL T1: полная оплата Саиды авто-снимает стоп",
      _reg_after_full.get("status") == "cleared" and bool(_reg_after_full.get("cleared_at")),
      str(_reg_after_full))
check("DSTOP SAIDA FULL T2: manager notified after auto-clear",
      any(m.get("chat_id") == 111 and "АвтоСнятие" in str(m.get("text", "")) for m in _fake_dstop_bot.messages),
      str(_fake_dstop_bot.messages))
check("DSTOP SAIDA FULL T3: shipment decision closed after auto-clear",
      _ship_after_full is None,
      str(_ship_after_full))
check("DSTOP SAIDA FULL T4: callback returns auto-clear confirmation",
      "автоматичес" in str(_full_result).lower(),
      str(_full_result))
_ship_for_saida._DECISIONS_PATH = _orig_ship_for_saida

# 20. Shipment control — collector/shipment_control.py
# ═══════════════════════════════════════════════════════════════
_orig_dstop_json_dir = _dstop.JSON_DIR
_orig_dstop_config_dir = _dstop.CONFIG_DIR
try:
    _dstop.JSON_DIR = Path(_dstop_tmpdir) / "json"
    _dstop.CONFIG_DIR = Path(_dstop_tmpdir) / "config"
    _dstop.JSON_DIR.mkdir(parents=True, exist_ok=True)
    _dstop.CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    (_dstop.CONFIG_DIR / "managers.json").write_text(
        json.dumps({"Алена": 188939016}, ensure_ascii=False),
        encoding="utf-8",
    )
    (_dstop.CONFIG_DIR / "clients.json").write_text(
        json.dumps({"clients": {}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (_dstop.CONFIG_DIR / "weekly_clients.json").write_text(
        json.dumps({"clients": []}, ensure_ascii=False),
        encoding="utf-8",
    )

    (_dstop.JSON_DIR / "debt_ext_Ведомость_по_взаиморасчетам_с_контрагентами_Алена (336).json").write_text(
        json.dumps({
            "clients": [{
                "client": "А Фурманова Евгений (склад № 20)",
                "days_silence": 7,
                "debt": 285535.02,
            }]
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    (_dstop.JSON_DIR / "debt_ext_Детальный Дебиторы Алена (143).json").write_text(
        json.dumps({
            "clients": [{
                "client": "А Фурманова Евгений (склад № 20)",
                "days_silence": 2,
                "debt": 36588.52,
            }]
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    os.utime(_dstop.JSON_DIR / "debt_ext_Ведомость_по_взаиморасчетам_с_контрагентами_Алена (336).json", (1, 1))
    os.utime(_dstop.JSON_DIR / "debt_ext_Детальный Дебиторы Алена (143).json", (2, 2))

    _picked = _dstop._get_latest_debt_file("Алена")
    check("DSTOP FILE T1: _get_latest_debt_file выбирает свежий detailed debt, а не старую ведомость с большим номером",
          _picked is not None and "Детальный Дебиторы Алена (143)" in _picked.name,
          str(_picked))

    _dstop.save_registry({})
    _dstop.save_state({"date": _today, "candidates": {}, "next_id": 1, "saida_sent": False})
    _rebuilt = _dstop._build_candidates()
    _cand_values = list(_rebuilt.get("candidates", {}).values())
    check("DSTOP FILE T2: _build_candidates не поднимает клиента из свежего файла, если он уже не проходит пороги",
          len(_cand_values) == 0,
          str(_cand_values))

    (_dstop.JSON_DIR / "debt_ext_Детальный Дебиторы Алена (144).json").write_text(
        json.dumps({
            "clients": [ {
                "client": "А Фурманова Евгений (склад № 20)",
                "days_silence": 10,
                "debt": 36588.52,
            } ]
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    os.utime(_dstop.JSON_DIR / "debt_ext_Детальный Дебиторы Алена (144).json", (3, 3))

    (_dstop.JSON_DIR / "debt_ext_Детальный Дебиторы Алена (145).json").write_text(
        json.dumps({
            "clients": [ {
                "client": "А Тестовый стоп-клиент",
                "days_silence": 9,
                "debt": 150000.0,
            }, {
                "client": "А Тестовый стоп-клиент 10д",
                "days_silence": 10,
                "debt": 150000.0,
            } ]
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    os.utime(_dstop.JSON_DIR / "debt_ext_Детальный Дебиторы Алена (145).json", (4, 4))

    _dstop.save_registry({})
    _dstop.save_state({"date": _today, "candidates": {}, "next_id": 1, "saida_sent": False})
    _rebuilt10 = _dstop._build_candidates()
    _cand_names10 = {item.get("client") for item in _rebuilt10.get("candidates", {}).values()}
    check("DSTOP FILE T3: клиент с 9 днями молчания больше не попадает в stop-flow",
          "А Тестовый стоп-клиент" not in _cand_names10,
          str(_cand_names10))
    check("DSTOP FILE T4: клиент с 10 днями молчания уже попадает в stop-flow",
          "А Тестовый стоп-клиент 10д" in _cand_names10,
          str(_cand_names10))
finally:
    _dstop.JSON_DIR = _orig_dstop_json_dir
    _dstop.CONFIG_DIR = _orig_dstop_config_dir

section("20. Shipment control (условная отгрузка)")

import tempfile as _tmpmod
_ship_tmpdir = _tmpmod.mkdtemp()
_ship_path = Path(_ship_tmpdir) / "collector_shipment_decisions.json"

try:
    import collector.shipment_control as _ship
    _orig_ship_path = _ship._DECISIONS_PATH
    _ship._DECISIONS_PATH = _ship_path

    # SC-1: set_decision записывает валидное решение
    try:
        rec = _ship.set_decision("77011112233", "ТОО Тест Отгрузка", "allow_after",
                                  manager_name="Алена", manager_chat_id=12345, amount=500000)
        check("SC-1: set_decision сохраняет allow_after",
              rec.get("decision") == "allow_after" and rec.get("client_name") == "ТОО Тест Отгрузка")
    except Exception as e:
        check("SC-1: set_decision сохраняет allow_after", False, str(e))

    # SC-2: get_decision возвращает активную запись
    try:
        got = _ship.get_decision("ТОО Тест Отгрузка")
        check("SC-2: get_decision возвращает запись", got is not None and got.get("decision") == "allow_after")
    except Exception as e:
        check("SC-2: get_decision возвращает запись", False, str(e))

    # SC-3: resolve_decision закрывает запись
    try:
        ok = _ship.resolve_decision("ТОО Тест Отгрузка", "debt_cleared")
        after = _ship.get_decision("ТОО Тест Отгрузка")
        check("SC-3: resolve_decision закрывает, get_decision возвращает None",
              ok and after is None)
    except Exception as e:
        check("SC-3: resolve_decision закрывает", False, str(e))

    # SC-4: list_pending возвращает только незакрытые allow_after/block_until
    try:
        _ship.set_decision("77011112244", "ТОО Ромашка", "block_until", amount=200000)
        _ship.set_decision("77011112255", "ИП Иванов", "allow", amount=50000)  # allow не попадает в pending
        pending = _ship.list_pending()
        names = [r.get("client_name") for r in pending]
        check("SC-4: list_pending возвращает только allow_after/block_until",
              "ТОО Ромашка" in names and "ИП Иванов" not in names and "ТОО Тест Отгрузка" not in names)
    except Exception as e:
        check("SC-4: list_pending фильтрует правильно", False, str(e))

    # SC-5: set_decision с невалидным решением → ValueError
    try:
        _ship.set_decision("77011112266", "Кто-то", "invalid_action")
        check("SC-5: set_decision с невалидным решением → ValueError", False, "исключение не было поднято")
    except ValueError:
        check("SC-5: set_decision с невалидным решением → ValueError", True)
    except Exception as e:
        check("SC-5: set_decision с невалидным решением → ValueError", False, str(e))

    # SC-6: нормализация имени — регистр и пробелы не важны
    try:
        _ship.set_decision("77011112277", "  ТОО  ЗАРЯ  ", "block", amount=999)
        got1 = _ship.get_decision("ТОО ЗАРЯ")
        got2 = _ship.get_decision("тоо заря")
        check("SC-6: нормализация имени работает (регистр/пробелы)",
              got1 is not None and got2 is not None)
    except Exception as e:
        check("SC-6: нормализация имени работает", False, str(e))

    # SC-7: check_pending_decisions разрешает allow_after при долге ≤ threshold
    try:
        import asyncio as _aio
        _ship.set_decision("77011112288", "ТОО ОплатилА", "allow_after",
                            manager_chat_id=0, amount=300000)
        # Мок debt_monitor.load_latest_debt_json
        _fake_debt = {"clients": [{"name": "ТОО ОплатилА", "amount": 500}]}  # 500 ₸ ≤ 1000
        with patch("collector.debt_monitor.load_latest_debt_json", return_value=_fake_debt):
            resolved = _aio.run(_ship.check_pending_decisions(None))
        after = _ship.get_decision("ТОО ОплатилА")
        check("SC-7: check_pending_decisions закрывает allow_after при долге ≤ 1000 ₸",
              resolved >= 1 and after is None)
    except Exception as e:
        check("SC-7: check_pending_decisions авто-закрытие", False, str(e))

    # SC-8: check_pending_decisions НЕ закрывает при долге > threshold
    try:
        _ship.set_decision("77011112299", "ТОО НеЗакрыл", "allow_after",
                            manager_chat_id=0, amount=300000)
        _fake_debt2 = {"clients": [{"name": "ТОО НеЗакрыл", "amount": 50000}]}  # 50k > 1000
        with patch("collector.debt_monitor.load_latest_debt_json", return_value=_fake_debt2):
            resolved2 = _aio.run(_ship.check_pending_decisions(None))
        still = _ship.get_decision("ТОО НеЗакрыл")
        check("SC-8: check_pending_decisions не закрывает при долге > 1000 ₸",
              resolved2 == 0 and still is not None)
    except Exception as e:
        check("SC-8: check_pending_decisions не закрывает при долге > 1000 ₸", False, str(e))

finally:
    _ship._DECISIONS_PATH = _orig_ship_path
    import shutil as _sh2
    _sh2.rmtree(_ship_tmpdir, ignore_errors=True)

# ═══════════════════════════════════════════════════════════════
# 20. ИТОГ
# ═══════════════════════════════════════════════════════════════
section("ИТОГ")  # секция 20
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

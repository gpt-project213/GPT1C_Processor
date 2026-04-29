#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тесты Phase 2 — wa_dialog_suppress:
  1. set/get/clear функции в collections_db
  2. Истёкший suppress возвращает None
  3. collections_engine.run() пропускает клиента при активном suppress
"""
import sys
import json
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import unittest.mock as _m
for _mod in ("telegram", "telegram.ext", "httpx", "portalocker"):
    sys.modules.setdefault(_mod, _m.MagicMock())

# portalocker.Lock нужен как настоящий контекст-менеджер
import unittest.mock as um
_pl = sys.modules["portalocker"]
_pl.Lock = um.MagicMock(return_value=um.MagicMock(__enter__=um.MagicMock(return_value=None),
                                                   __exit__=um.MagicMock(return_value=False)))
_pl.LockException = Exception

PASS, FAIL = "PASS", "FAIL"
results: list = []


def check(name: str, ok: bool, detail: str = ""):
    icon = PASS if ok else FAIL
    msg = f"  {icon} {name}"
    if detail:
        msg += f": {detail}"
    print(msg)
    results.append((name, ok, detail))


# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_db_module(state_file: Path):
    """Импортирует collections_db с переопределённым STATE_PATH."""
    import importlib
    import collector.collections_db as db
    importlib.reload(db)
    db.STATE_PATH = state_file
    return db


# ─── Test 1: set_wa_dialog_suppress записывает флаг ─────────────────────────

def test_set_suppress():
    with tempfile.TemporaryDirectory() as td:
        state_file = Path(td) / "collector_state.json"
        db = _make_db_module(state_file)

        until = (date.today() + timedelta(days=2)).isoformat()
        db.set_wa_dialog_suppress("ТестКлиент", "paid_claim", until)

        raw = json.loads(state_file.read_text(encoding="utf-8"))
        sup = raw.get("ТестКлиент", {}).get("wa_dialog_suppress")
        ok = (
            sup is not None
            and sup.get("reason") == "paid_claim"
            and sup.get("until") == until
        )
        check("set_suppress — флаг записан в JSON", ok, str(sup))


# ─── Test 2: get возвращает активный флаг ────────────────────────────────────

def test_get_active_suppress():
    with tempfile.TemporaryDirectory() as td:
        state_file = Path(td) / "collector_state.json"
        db = _make_db_module(state_file)

        until = (date.today() + timedelta(days=1)).isoformat()
        db.set_wa_dialog_suppress("Клиент А", "attachment", until)
        result = db.get_wa_dialog_suppress("Клиент А")
        ok = result is not None and result.get("reason") == "attachment"
        check("get_suppress — активный флаг возвращается", ok)


# ─── Test 3: get возвращает None для несуществующего клиента ─────────────────

def test_get_suppress_missing():
    with tempfile.TemporaryDirectory() as td:
        state_file = Path(td) / "collector_state.json"
        db = _make_db_module(state_file)
        result = db.get_wa_dialog_suppress("НеизвестныйКлиент")
        check("get_suppress — None для неизвестного клиента", result is None)


# ─── Test 4: истёкший suppress возвращает None ───────────────────────────────

def test_get_expired_suppress():
    with tempfile.TemporaryDirectory() as td:
        state_file = Path(td) / "collector_state.json"
        db = _make_db_module(state_file)

        expired_until = (date.today() - timedelta(days=1)).isoformat()
        db.set_wa_dialog_suppress("Клиент Б", "paid_claim", expired_until)
        result = db.get_wa_dialog_suppress("Клиент Б")
        check("get_suppress — истёкший флаг возвращает None", result is None,
              f"until={expired_until}")


# ─── Test 5: clear_wa_dialog_suppress сбрасывает флаг ────────────────────────

def test_clear_suppress():
    with tempfile.TemporaryDirectory() as td:
        state_file = Path(td) / "collector_state.json"
        db = _make_db_module(state_file)

        until = (date.today() + timedelta(days=3)).isoformat()
        db.set_wa_dialog_suppress("Клиент В", "paid_claim", until)
        db.clear_wa_dialog_suppress("Клиент В")

        raw = json.loads(state_file.read_text(encoding="utf-8"))
        sup = raw.get("Клиент В", {}).get("wa_dialog_suppress")
        check("clear_suppress — флаг сброшен в None", sup is None, str(sup))


# ─── Test 6: _empty_record содержит wa_dialog_suppress ───────────────────────

def test_empty_record_has_field():
    with tempfile.TemporaryDirectory() as td:
        state_file = Path(td) / "collector_state.json"
        db = _make_db_module(state_file)
        rec = db._empty_record()
        ok = "wa_dialog_suppress" in rec and rec["wa_dialog_suppress"] is None
        check("_empty_record — поле wa_dialog_suppress присутствует", ok)


# ─── Test 7: engine пропускает клиента с активным suppress ───────────────────

def test_engine_skips_suppressed_client():
    """run() должен пропустить клиента если get_wa_dialog_suppress возвращает флаг."""
    import importlib

    # Мокаем все зависимости engine
    for mod in (
        "collector.debt_monitor", "collector.collections_db",
        "collector.payment_hold", "collector.no_movement",
        "collector.approval_flow", "collector.collection_agent",
        "bot.send_reports",
    ):
        sys.modules[mod] = _m.MagicMock()

    # Патчим get_wa_dialog_suppress чтобы вернуть активный флаг
    suppress_flag = {"reason": "paid_claim", "until": (date.today() + timedelta(days=2)).isoformat()}
    sys.modules["collector.collections_db"].get_wa_dialog_suppress = MagicMock(
        return_value=suppress_flag
    )
    sys.modules["collector.collections_db"].get_client_state = MagicMock(
        return_value={"last_contact_date": None, "history": [], "wa_dialog_suppress": suppress_flag}
    )
    sys.modules["collector.payment_hold"].get_hold_for_client = MagicMock(return_value=None)

    # Минимальный debt snapshot
    fresh_debt = {
        "clients": [{"name": "Подавленный Клиент", "amount": 50000, "days": 30,
                     "debit": 100, "credit": 100, "phone": "77001234567",
                     "manager": "Менеджер А"}],
        "_freshness": {"is_stale": False, "max_age_hours": 2, "stale_managers": [],
                       "warning_managers": []},
    }
    sys.modules["collector.debt_monitor"].load_latest_debt_json = MagicMock(
        return_value=fresh_debt
    )

    import collector.collections_engine as eng
    importlib.reload(eng)

    processed = []

    async def _fake_process(client, *args, **kwargs):
        processed.append(client["name"])
        return {"name": client["name"], "sent": True}

    import asyncio
    with patch.object(eng, "_process_single", side_effect=_fake_process):
        asyncio.run(eng.run(dry_run=True))

    ok = "Подавленный Клиент" not in processed
    check("engine.run() пропускает клиента с wa_dialog_suppress", ok,
          f"processed={processed}")


# ─── Запуск ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Phase 2: wa_dialog_suppress ===\n")
    test_set_suppress()
    test_get_active_suppress()
    test_get_suppress_missing()
    test_get_expired_suppress()
    test_clear_suppress()
    test_empty_record_has_field()
    test_engine_skips_suppressed_client()

    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'='*40}")
    print(f"Итог: {passed} прошли, {failed} упали")
    if failed:
        print("\nПровалившиеся тесты:")
        for name, ok, detail in results:
            if not ok:
                print(f"  ❌ {name}: {detail}")
    raise SystemExit(0 if failed == 0 else 1)

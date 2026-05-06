#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тесты collector/audit_log.py:
  1. audit() записывает JSONL-строку
  2. Запись содержит ts, event и kwargs
  3. Несколько записей — несколько строк
  4. read_recent() возвращает последние N записей
  5. read_for_client() фильтрует по имени
  6. Ошибка записи не пробрасывается
  7. Вызовы из collections_db пишут suppress_set / suppress_cleared
"""
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import unittest.mock as _m
for _mod in ("portalocker",):
    sys.modules.setdefault(_mod, _m.MagicMock())
_pl = sys.modules["portalocker"]
_pl.Lock = _m.MagicMock(return_value=_m.MagicMock(
    __enter__=_m.MagicMock(return_value=None),
    __exit__=_m.MagicMock(return_value=False)))
_pl.LockException = Exception

PASS, FAIL = "PASS", "FAIL"
results: list = []


def check(name, ok, detail=""):
    print(f"  {PASS if ok else FAIL} {name}" + (f": {detail}" if detail else ""))
    results.append((name, ok))


import importlib
import collector.audit_log as al


def _with_tmp_path(fn):
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / "collector_audit.jsonl"
        with patch.object(al, "_AUDIT_PATH", tmp_path):
            fn(tmp_path)


# ── Test 1: пишет одну строку ──────────────────────────────────────────────

def test_writes_single_line():
    def run(path):
        al.audit("wa_sent", name="ТОО Ромашка", amount=100000)
        lines = path.read_text(encoding="utf-8").splitlines()
        check("audit() — одна строка записана", len(lines) == 1, str(lines))
    _with_tmp_path(run)


# ── Test 2: содержит ts, event, kwargs ────────────────────────────────────

def test_record_structure():
    def run(path):
        al.audit("wa_skipped", name="Клиент Б", reason="payment_hold")
        rec = json.loads(path.read_text(encoding="utf-8").strip())
        ok = (
            "ts" in rec
            and rec["event"] == "wa_skipped"
            and rec["name"] == "Клиент Б"
            and rec["reason"] == "payment_hold"
        )
        check("audit() — запись содержит ts, event, kwargs", ok, str(rec))
    _with_tmp_path(run)


# ── Test 3: несколько вызовов — несколько строк ───────────────────────────

def test_multiple_lines():
    def run(path):
        al.audit("wa_sent", name="А")
        al.audit("wa_sent", name="Б")
        al.audit("wa_sent", name="В")
        lines = path.read_text(encoding="utf-8").splitlines()
        check("audit() — три вызова дают три строки", len(lines) == 3)
    _with_tmp_path(run)


# ── Test 4: read_recent возвращает записи ──────────────────────────────────

def test_read_recent():
    def run(path):
        for i in range(5):
            al.audit("batch_created", batch_id=f"b_{i:03d}", clients=i)
        recs = al.read_recent(n=3)
        ok = len(recs) == 3 and recs[-1]["batch_id"] == "b_004"
        check("read_recent(3) — возвращает 3 последних", ok, str([r["batch_id"] for r in recs]))
    _with_tmp_path(run)


# ── Test 5: read_for_client фильтрует по имени ────────────────────────────

def test_read_for_client():
    def run(path):
        al.audit("wa_sent", name="Ромашка", amount=1)
        al.audit("wa_sent", name="Лютик", amount=2)
        al.audit("wa_skipped", name="Ромашка", reason="suppress")
        recs = al.read_for_client("Ромашка")
        ok = len(recs) == 2 and all(r["name"] == "Ромашка" for r in recs)
        check("read_for_client() — только события нужного клиента", ok,
              str([r["event"] for r in recs]))
    _with_tmp_path(run)


# ── Test 6: ошибка записи не пробрасывается ───────────────────────────────

def test_write_error_silent():
    try:
        with patch.object(al, "_AUDIT_PATH", Path("/nonexistent_dir/audit.jsonl")):
            al.audit("wa_sent", name="Тест")
        check("audit() — ошибка записи не пробрасывается", True)
    except Exception as e:
        check("audit() — ошибка записи не пробрасывается", False, str(e))


# ── Test 7: suppress_set / suppress_cleared через collections_db ──────────

def test_suppress_writes_audit():
    import collector.collections_db as db
    importlib.reload(db)

    captured = []

    def fake_audit(event, **kw):
        captured.append({"event": event, **kw})

    with (
        tempfile.TemporaryDirectory() as td,
        patch.object(db, "STATE_PATH", Path(td) / "state.json"),
        patch.object(al, "_AUDIT_PATH", Path(td) / "audit.jsonl"),
        patch("collector.collections_db.audit_log" if hasattr(db, "audit_log") else "collector.audit_log.audit", fake_audit, create=True),
    ):
        with patch("collector.audit_log.audit", fake_audit):
            from datetime import date, timedelta
            until = (date.today() + timedelta(days=2)).isoformat()
            db.set_wa_dialog_suppress("Клиент X", "paid_claim", until)
            db.clear_wa_dialog_suppress("Клиент X")

    set_ev = [e for e in captured if e.get("event") == "suppress_set"]
    clear_ev = [e for e in captured if e.get("event") == "suppress_cleared"]
    ok = len(set_ev) >= 1 and len(clear_ev) >= 1
    check("collections_db — suppress_set и suppress_cleared в audit", ok,
          f"set={len(set_ev)} clear={len(clear_ev)}")


# ── Запуск ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Фаза 3А: audit_log ===\n")
    test_writes_single_line()
    test_record_structure()
    test_multiple_lines()
    test_read_recent()
    test_read_for_client()
    test_write_error_silent()
    test_suppress_writes_audit()

    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    print(f"\n{'='*40}")
    print(f"Итог: {passed} прошли, {failed} упали")
    if failed:
        for name, ok in results:
            if not ok:
                print(f"  FAIL {name}")
    raise SystemExit(0 if failed == 0 else 1)

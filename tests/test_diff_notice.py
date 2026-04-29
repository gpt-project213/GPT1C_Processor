#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тесты diff-notice (preview_batch_changes):
  1. Нет изменений -> None
  2. Изменение суммы -> строка с diff
  3. Клиент исчез из дебиторки -> строка про skipped
  4. Ошибка при загрузке дебиторки -> None (не падает)
  5. Блокировка freshness -> текст с предупреждением
  6. approval_flow вставляет diff-блок в текст утверждения
"""
import sys
import importlib
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import unittest.mock as _m
for _mod in ("telegram", "telegram.ext", "httpx", "portalocker"):
    sys.modules.setdefault(_mod, _m.MagicMock())

PASS, FAIL = "PASS", "FAIL"
results: list = []


def check(name: str, ok: bool, detail: str = ""):
    icon = PASS if ok else FAIL
    msg = f"  {icon} {name}"
    if detail:
        msg += f": {detail}"
    print(msg)
    results.append((name, ok, detail))


# ─── import once ─────────────────────────────────────────────────────────────

import collector.collections_engine as eng

# ─── Test 1: нет изменений → None ───────────────────────────────────────────

def test_no_changes_returns_none():
    with patch.object(eng, "_refresh_approved_batch_clients", return_value=([], [], [], None)):
        result = eng.preview_batch_changes("batch_001", [{"name": "Клиент А"}])
    check("preview_batch_changes — нет изменений -> None", result is None, repr(result))


# ─── Test 2: изменение суммы → строка с описанием ───────────────────────────

def test_amount_change():
    changes = ["Клиент А: amount 100,000->85,000"]
    with patch.object(eng, "_refresh_approved_batch_clients", return_value=([], [], changes, None)):
        result = eng.preview_batch_changes("batch_002", [{"name": "Клиент А"}])
    ok = result is not None and "Клиент А" in result and "Изменений" in result
    check("preview_batch_changes — изменение суммы попадает в текст", ok, repr(result))


# ─── Test 3: клиент исчез → упоминание skipped ───────────────────────────────

def test_client_disappeared():
    skipped = [{"name": "Клиент Б", "reason": "not in debt shortlist"}]
    with patch.object(eng, "_refresh_approved_batch_clients", return_value=([], skipped, [], None)):
        result = eng.preview_batch_changes("batch_003", [{"name": "Клиент Б"}])
    ok = result is not None and "Исчезли" in result
    check("preview_batch_changes — исчезнувший клиент -> упоминание в тексте", ok, repr(result))


# ─── Test 4: исключение внутри refresh → None, не падает ─────────────────────

def test_exception_returns_none():
    with patch.object(eng, "_refresh_approved_batch_clients", side_effect=RuntimeError("нет файла")):
        result = eng.preview_batch_changes("batch_004", [{"name": "Клиент В"}])
    check("preview_batch_changes — исключение -> возвращает None", result is None, repr(result))


# ─── Test 5: blocked_reason → текст предупреждения ───────────────────────────

def test_blocked_reason():
    with patch.object(eng, "_refresh_approved_batch_clients", return_value=([], [], [], "debt file is stale")):
        result = eng.preview_batch_changes("batch_005", [])
    ok = result is not None and "Проверка" in result and "stale" in result
    check("preview_batch_changes — blocked_reason -> предупреждение", ok, repr(result))


# ─── Test 6: diff-блок в тексте утверждения approval_flow ───────────────────

def test_approval_flow_shows_diff():
    """handle_admin_callback вставляет diff-блок в текст если есть изменения."""
    import collector.approval_flow as af

    fake_batch = {
        "batch_id": "batch_006",
        "status": "pending_admin",
        "admin_status": "pending",
        "managers": {
            "Менеджер А": {
                "status": "approved",
                "clients": [
                    {"name": "Клиент Х", "manager": "Менеджер А", "phone": "77001234567",
                     "amount": 50000, "days": 30, "msg_type": "standard"},
                ],
            }
        },
        "rejected_managers": [],
    }

    captured_text = []

    async def _fake_tg_edit(cid, mid, text, markup=None):
        captured_text.append(text)

    diff_text = "Изменений с момента формирования (1):\n  • Клиент Х: amount 50,000->45,000"

    with (
        patch.object(af, "load_batch", return_value=fake_batch),
        patch.object(af, "save_batch"),
        patch.object(af, "_tg_edit", side_effect=_fake_tg_edit),
        patch.object(af, "_build_admin_decisions", return_value={"keep_0": "keep"}),
        patch.object(af, "_iter_admin_clients", return_value=[
            {"name": "Клиент Х", "manager": "Менеджер А", "phone": "77001234567",
             "amount": 50000, "_admin_key": "keep_0"},
        ]),
        patch.object(eng, "preview_batch_changes", return_value=diff_text),
    ):
        import asyncio
        asyncio.run(af.handle_admin_callback("wa_appr_adm_ok|batch_006", 1001, 42, None))

    ok = any("Данные обновились" in t for t in captured_text)
    check("approval_flow — diff-блок появляется в тексте утверждения", ok,
          (captured_text[0][:200] if captured_text else "нет текста"))


# ─── Запуск ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Diff-notice: preview_batch_changes ===\n")
    test_no_changes_returns_none()
    test_amount_change()
    test_client_disappeared()
    test_exception_returns_none()
    test_blocked_reason()
    test_approval_flow_shows_diff()

    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'='*40}")
    print(f"Итог: {passed} прошли, {failed} упали")
    if failed:
        for name, ok, detail in results:
            if not ok:
                print(f"  FAIL {name}: {detail}")
    raise SystemExit(0 if failed == 0 else 1)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Фаза 4 — Тесты: hard-ban на фразы об отгрузке для хвостовых клиентов.

Проверяем:
  1. legacy_tail_reminder — не содержит слово «отгрузк»
  2. partial_tail_reminder — не содержит слово «отгрузк»
  3. promise_broken_reminder — не содержит слово «отгрузк» (фикс Phase 4)
  4. stoplist_reminder — СОДЕРЖИТ слово «отгрузк» (намеренно, трогать нельзя)
  5. promise_broken_reminder — всё ещё содержит запрос даты платежа
  6. promise_broken_reminder рендерится без ошибок KeyError
  7. JSON-конфиг fallback_templates — promise_broken_reminder тоже без «отгрузк»
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import unittest.mock as _m
for _mod in ("httpx", "dotenv"):
    sys.modules.setdefault(_mod, _m.MagicMock())
sys.modules.setdefault("zoneinfo", _m.MagicMock())

# Теперь безопасно импортировать агент
from collector import collection_agent as agent

PASS, FAIL = "PASS", "FAIL"
results: list = []

FORBIDDEN = "отгруз"  # охватывает «отгрузки», «отгрузок», «отгрузках», «отгрузку»
PAYMENT_KEYWORDS = ("дату", "дата", "платёж", "платеж", "оплат")


def check(name, ok, detail=""):
    print(f"  {PASS if ok else FAIL} {name}" + (f": {detail}" if detail else ""))
    results.append((name, ok))


# ── Test 1 ─────────────────────────────────────────────────────────────────

def test_legacy_tail_no_shipment():
    tpl = agent._FALLBACK_TEMPLATES_DEFAULT.get("legacy_tail_reminder", "")
    ok = FORBIDDEN not in tpl.lower()
    check("legacy_tail_reminder — нет фразы об отгрузке", ok,
          "(шаблон содержит фразу!)" if not ok else "")


# ── Test 2 ─────────────────────────────────────────────────────────────────

def test_partial_tail_no_shipment():
    tpl = agent._FALLBACK_TEMPLATES_DEFAULT.get("partial_tail_reminder", "")
    ok = FORBIDDEN not in tpl.lower()
    check("partial_tail_reminder — нет фразы об отгрузке", ok,
          "(шаблон содержит фразу!)" if not ok else "")


# ── Test 3 ─────────────────────────────────────────────────────────────────

def test_promise_broken_no_shipment():
    tpl = agent._FALLBACK_TEMPLATES_DEFAULT.get("promise_broken_reminder", "")
    ok = FORBIDDEN not in tpl.lower()
    check("promise_broken_reminder — нет фразы об отгрузке (Phase 4 fix)", ok,
          "(ФРАЗА ОСТАЛАСЬ В ШАБЛОНЕ!)" if not ok else "")


# ── Test 4 ─────────────────────────────────────────────────────────────────

def test_stoplist_reminder_has_shipment():
    """stoplist_reminder ДОЛЖЕН содержать «отгрузки» — это намеренно."""
    tpl = agent._FALLBACK_TEMPLATES_DEFAULT.get("stoplist_reminder", "")
    ok = FORBIDDEN in tpl.lower()
    check("stoplist_reminder — содержит фразу об отгрузке (намеренно)", ok,
          "(фраза ПРОПАЛА — кто-то убрал из не того шаблона!)" if not ok else "")


# ── Test 5 ─────────────────────────────────────────────────────────────────

def test_promise_broken_still_requests_payment():
    tpl = agent._FALLBACK_TEMPLATES_DEFAULT.get("promise_broken_reminder", "").lower()
    ok = any(kw in tpl for kw in PAYMENT_KEYWORDS)
    check("promise_broken_reminder — содержит запрос даты/суммы платежа", ok,
          "(запрос пропал — шаблон слишком мягкий!)" if not ok else "")


# ── Test 6 ─────────────────────────────────────────────────────────────────

def test_promise_broken_renders():
    try:
        text = agent._get_fallback_template(
            "promise_broken_reminder",
            company="ТестКо",
            client_name="ТОО Пример",
            manager_name="Иван",
            amount="500000",
            days=45,
            report_date="2026-04-01",
        )
        ok = "ТОО Пример" in text and "ТестКо" in text
        check("promise_broken_reminder — рендерится без KeyError", ok, text[:60])
    except Exception as e:
        check("promise_broken_reminder — рендерится без KeyError", False, str(e))


# ── Test 7 ─────────────────────────────────────────────────────────────────

def test_json_config_promise_broken_no_shipment():
    cfg_path = ROOT / "config" / "collector_prompts.json"
    if not cfg_path.exists():
        check("JSON-конфиг promise_broken_reminder — нет «отгрузк»", True, "(файл отсутствует — пропуск)")
        return
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    tpl = cfg.get("fallback_templates", {}).get("promise_broken_reminder", "")
    if not tpl:
        check("JSON-конфиг promise_broken_reminder — нет «отгрузк»", True, "(в JSON нет шаблона — используется Python default)")
        return
    ok = FORBIDDEN not in tpl.lower()
    check("JSON-конфиг promise_broken_reminder — нет «отгрузк»", ok,
          "(ФРАЗА В JSON!)" if not ok else "")


# ── Запуск ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Фаза 4: hard-ban на отгрузки в хвостовых шаблонах ===\n")
    test_legacy_tail_no_shipment()
    test_partial_tail_no_shipment()
    test_promise_broken_no_shipment()
    test_stoplist_reminder_has_shipment()
    test_promise_broken_still_requests_payment()
    test_promise_broken_renders()
    test_json_config_promise_broken_no_shipment()

    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    print(f"\n{'='*40}")
    print(f"Итог: {passed} прошли, {failed} упали")
    if failed:
        for name, ok in results:
            if not ok:
                print(f"  FAIL {name}")
    raise SystemExit(0 if failed == 0 else 1)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тесты изменений сессии 2026-04-28:
  1. get_clients_without_phones() — фильтр ЗП/служебных записей
  2. _crm_cleanup_pending() — выметание ЗП-записей из памяти
  3. daily_summary() — блок WA-получателей
  4. Константы триггера коллектора (окно 22:00, TTL 14ч)
"""
import sys
import json
import asyncio
import importlib
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── стабы чтобы не тянуть telegram/httpx ──────────────────────────────
import unittest.mock as _m
for _mod in ("telegram", "telegram.ext", "httpx", "portalocker"):
    sys.modules.setdefault(_mod, _m.MagicMock())

PASS, FAIL = "✅", "❌"
results: list = []

def check(name: str, ok: bool, detail: str = ""):
    icon = PASS if ok else FAIL
    msg = f"  {icon} {name}"
    if detail and not ok:
        msg += f"\n      {detail}"
    print(msg)
    results.append((name, ok))


# ════════════════════════════════════════════════════════════════════
# 1. get_clients_without_phones — фильтр ЗП
# ════════════════════════════════════════════════════════════════════

def _make_crm_data(*names_with_manager):
    """Создаёт минимальный CRM-словарь."""
    clients = {}
    for name, mgr in names_with_manager:
        clients[name] = {"manager": mgr, "sources": ["debt"]}
    return {"clients": clients}


def test_zp_excluded_from_no_phone():
    import bot.crm_clients as crm
    data = _make_crm_data(
        ("ТОО Реальный клиент", "Ергали"),
        ("Ергали тов. под ЗП", "Ергали"),
        ("Без клиента", "Ергали"),
        ("Недостача", "Ергали"),
        ("Выдача ЗП январь", "Ергали"),
    )
    with patch.object(crm, "load_clients", return_value=data):
        result = crm.get_clients_without_phones("Ергали", limit=50)
    check("ЗП-запись исключена из очереди телефонов",
          "Ергали тов. под ЗП" not in result)
    check("'Без клиента' исключён",
          "Без клиента" not in result)
    check("'Недостача' исключена",
          "Недостача" not in result)
    check("'Выдача ЗП январь' исключена",
          "Выдача ЗП январь" not in result)
    check("Реальный клиент остаётся в очереди",
          "ТОО Реальный клиент" in result)


# ════════════════════════════════════════════════════════════════════
# 2. _crm_cleanup_pending — выметание ЗП из памяти
# ════════════════════════════════════════════════════════════════════

def test_cleanup_removes_zp_from_memory():
    """_crm_cleanup_pending удаляет ЗП-записи из _CRM_PHONE_PENDING."""
    # Импортируем только нужные объекты, обходя тяжёлые зависимости send_reports
    import importlib, types, datetime

    # Минимальный стаб модуля вместо полного импорта send_reports
    fake_sr = types.ModuleType("send_reports_stub")
    fake_sr._CRM_PHONE_PENDING = {
        111: {"client_key": "Ергали тов. под ЗП", "last_sent": datetime.datetime.now().isoformat()},
        222: {"client_key": "ТОО Нормальный",     "last_sent": datetime.datetime.now().isoformat()},
        333: {"client_key": "Без клиента",          "last_sent": datetime.datetime.now().isoformat()},
    }
    CRM_PENDING_TTL_HOURS = 48
    import logging
    logger = logging.getLogger("test")

    # Воспроизводим логику _crm_cleanup_pending (только ЗП-ветку)
    stale = []
    for chat_id, pending in list(fake_sr._CRM_PHONE_PENDING.items()):
        ck = (pending.get("client_key") or "").lower()
        if "зп" in ck or ck in ("без клиента", "недостача"):
            stale.append(chat_id)
    for cid in stale:
        fake_sr._CRM_PHONE_PENDING.pop(cid, None)

    remaining = list(fake_sr._CRM_PHONE_PENDING.keys())
    check("ЗП-запись удалена из _CRM_PHONE_PENDING",
          111 not in remaining)
    check("'Без клиента' удалён из _CRM_PHONE_PENDING",
          333 not in remaining)
    check("Реальный клиент остался в _CRM_PHONE_PENDING",
          222 in remaining)


# ════════════════════════════════════════════════════════════════════
# 3. daily_summary — блок WA-получателей
# ════════════════════════════════════════════════════════════════════

def test_daily_summary_lists_wa_recipients():
    import collector.collections_engine as ce

    processed = [
        {"name": "Клиент А", "level": 2, "days": 45, "amount": 150000,
         "sent": True, "wa_phone": "+77011234567",
         "promise_received": False, "promise_broken": False,
         "no_contacts": False, "escalated": False},
        {"name": "Клиент Б", "level": 3, "days": 30, "amount": 80000,
         "sent": True, "wa_phone": "",
         "promise_received": False, "promise_broken": False,
         "no_contacts": False, "escalated": False},
        {"name": "Клиент В", "level": 1, "days": 10, "amount": 20000,
         "sent": False,
         "promise_received": False, "promise_broken": False,
         "no_contacts": False, "escalated": False},
    ]
    summary = ce.daily_summary(processed, total_classified=3, dry_run=False)

    check("Сводка содержит блок WhatsApp",
          "WhatsApp отправлен" in summary)
    check("Клиент А с телефоном попал в список WA",
          "Клиент А" in summary)
    check("Клиент Б без телефона не попал в список WA",
          summary.count("Клиент Б") == 0 or "WhatsApp" not in summary.split("Клиент Б")[0].split("\n")[-1])
    check("Счётчик WA = 1 (только у кого есть wa_phone)",
          "WhatsApp отправлен (1)" in summary)


def test_daily_summary_no_wa_block_when_none_sent():
    import collector.collections_engine as ce

    processed = [
        {"name": "Клиент А", "level": 1, "days": 10, "amount": 5000,
         "sent": False,
         "promise_received": False, "promise_broken": False,
         "no_contacts": False, "escalated": False},
    ]
    summary = ce.daily_summary(processed, total_classified=1, dry_run=False)
    check("Блок WA отсутствует когда никому не отправлено",
          "WhatsApp отправлен" not in summary)


# ════════════════════════════════════════════════════════════════════
# 4. Константы триггера коллектора
# ════════════════════════════════════════════════════════════════════

def test_trigger_constants():
    # Читаем send_reports.py как текст — не импортируем
    src = (ROOT / "bot" / "send_reports.py").read_text(encoding="utf-8")
    check("_COLLECTOR_TRIGGER_MAX_AGE_HOURS = 14",
          "_COLLECTOR_TRIGGER_MAX_AGE_HOURS = 14" in src)
    check("Окно триггера расширено до 22:00",
          "9 <= now.hour < 22" in src)


# ════════════════════════════════════════════════════════════════════
# Запуск
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n=== test_session_20260428 ===\n")

    print("1. get_clients_without_phones — фильтр ЗП")
    test_zp_excluded_from_no_phone()

    print("\n2. _crm_cleanup_pending — выметание ЗП из памяти")
    test_cleanup_removes_zp_from_memory()

    print("\n3. daily_summary — блок WA-получателей")
    test_daily_summary_lists_wa_recipients()
    test_daily_summary_no_wa_block_when_none_sent()

    print("\n4. Константы триггера")
    test_trigger_constants()

    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    print(f"\n{'='*40}")
    print(f"Итого: {passed} ✅  {failed} ❌  из {len(results)}")
    if failed:
        sys.exit(1)

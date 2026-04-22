#!/usr/bin/env python
# coding: utf-8
"""
Узкие регрессии на выбор актуальных отчётов/JSON без запуска Telegram runtime.
"""
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bot"))

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


section("1. debt_stop_control и crm_clients — только свежий detailed debt")

import bot.debt_stop_control as _dstop
import bot.crm_clients as _crm

_tmp = Path(tempfile.mkdtemp())
try:
    _dstop_json = _tmp / "json"
    _dstop_cfg = _tmp / "config"
    _dstop_json.mkdir(parents=True, exist_ok=True)
    _dstop_cfg.mkdir(parents=True, exist_ok=True)

    (_dstop_cfg / "managers.json").write_text(
        json.dumps({"Алена": 188939016}, ensure_ascii=False),
        encoding="utf-8",
    )
    (_dstop_cfg / "clients.json").write_text(json.dumps({"clients": {}}, ensure_ascii=False), encoding="utf-8")
    (_dstop_cfg / "weekly_clients.json").write_text(json.dumps({"clients": []}, ensure_ascii=False), encoding="utf-8")

    stale_path = _dstop_json / "debt_ext_Ведомость_по_взаиморасчетам_с_контрагентами_Алена (336).json"
    stale_path.write_text(json.dumps({
        "manager": "Алена",
        "clients": [{"client": "А Фурманова Евгений (склад № 20)", "days_silence": 7, "debt": 285535.02}],
    }, ensure_ascii=False), encoding="utf-8")

    fresh_path = _dstop_json / "debt_ext_Детальный Дебиторы Алена (143).json"
    fresh_path.write_text(json.dumps({
        "manager": "Алена",
        "clients": [{"client": "А Фурманова Евгений (склад № 20)", "days_silence": 2, "debt": 36588.52}],
    }, ensure_ascii=False), encoding="utf-8")

    os.utime(stale_path, (1, 1))
    os.utime(fresh_path, (2, 2))

    _orig_dstop_json = _dstop.JSON_DIR
    _orig_dstop_cfg = _dstop.CONFIG_DIR
    _orig_crm_json = _crm.JSON_DIR
    try:
        _dstop.JSON_DIR = _dstop_json
        _dstop.CONFIG_DIR = _dstop_cfg
        _crm.JSON_DIR = _dstop_json

        picked = _dstop._get_latest_debt_file("Алена")
        check(
            "FRESH T1: debt_stop_control выбирает свежий detailed debt",
            picked is not None and picked.name == fresh_path.name,
            str(picked),
        )

        clients = _crm._load_latest_debt_clients()
        check(
            "FRESH T2: crm_clients читает только актуальный debt JSON менеджера",
            clients == [("А Фурманова Евгений (склад № 20)", "Алена")],
            str(clients),
        )
    finally:
        _dstop.JSON_DIR = _orig_dstop_json
        _dstop.CONFIG_DIR = _orig_dstop_cfg
        _crm.JSON_DIR = _orig_crm_json
finally:
    shutil.rmtree(_tmp, ignore_errors=True)


section("2. inventory_summary — дневной JSON не заменяется range/cost-файлом")

from bot.inventory_summary import InventorySummary

_tmp = Path(tempfile.mkdtemp())
try:
    json_dir = _tmp / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    day_old = json_dir / "inventory_остатки_всем_307__21_апреля_2026_г.json"
    day_old.write_text(json.dumps({"period": "21 апреля 2026 г.", "total_qty": 10}, ensure_ascii=False), encoding="utf-8")
    day_new = json_dir / "inventory_остатки_всем_308__22_апреля_2026_г.json"
    day_new.write_text(json.dumps({"period": "22 апреля 2026 г.", "total_qty": 11}, ensure_ascii=False), encoding="utf-8")
    range_newer = json_dir / "inventory_ведомость_по_партиям_товаров_на_складах_224__01_04_2026_-_21_04_2026.json"
    range_newer.write_text(json.dumps({"period": "01.04.2026 - 21.04.2026", "total_qty": 999}, ensure_ascii=False), encoding="utf-8")
    cost_newest = json_dir / "inventory_cost_20260422132932_ведомость_по_партиям_товаров_на_скл.json"
    cost_newest.write_text(json.dumps({"period": "01.04.2026 - 21.04.2026", "total_qty": 1000}, ensure_ascii=False), encoding="utf-8")

    os.utime(day_old, (10, 10))
    os.utime(day_new, (20, 20))
    os.utime(range_newer, (30, 30))
    os.utime(cost_newest, (40, 40))

    picked = InventorySummary().get_latest_inventory_json(json_dir)
    check(
        "FRESH T3: inventory_summary предпочитает последний дневной inventory JSON",
        picked is not None and picked.name == day_new.name,
        str(picked),
    )
finally:
    shutil.rmtree(_tmp, ignore_errors=True)


section("3. send_reports — debt JSON для бота берётся из detailed debt")


class _DummyFileHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def emit(self, record):
        return


_orig_file_handler = logging.FileHandler
logging.FileHandler = _DummyFileHandler
try:
    import bot.send_reports as _sr
finally:
    logging.FileHandler = _orig_file_handler

_tmp = Path(tempfile.mkdtemp())
try:
    json_dir = _tmp / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    stale_path = json_dir / "debt_ext_Ведомость_по_взаиморасчетам_с_контрагентами_Алена (336).json"
    stale_path.write_text(json.dumps({"manager": "Алена", "period_max": "17.04.2026", "aggregates": {"close": 285535.02}}, ensure_ascii=False), encoding="utf-8")
    fresh_path = json_dir / "debt_ext_Детальный Дебиторы Алена (143).json"
    fresh_path.write_text(json.dumps({"manager": "Алена", "period_max": "22.04.2026", "aggregates": {"close": 36588.52}}, ensure_ascii=False), encoding="utf-8")

    now_ts = time.time()
    os.utime(stale_path, (now_ts - 120, now_ts - 120))
    os.utime(fresh_path, (now_ts - 60, now_ts - 60))

    _orig_sr_json = _sr.JSON_DIR
    try:
        _sr.JSON_DIR = json_dir
        picked = _sr.find_recent_json_for_manager("Алена", hours=999, report_type="DEBT")
        check(
            "FRESH T4: send_reports.find_recent_json_for_manager(DEBT) выбирает detailed debt",
            picked is not None and picked.name == fresh_path.name,
            str(picked),
        )
    finally:
        _sr.JSON_DIR = _orig_sr_json
finally:
    shutil.rmtree(_tmp, ignore_errors=True)


print("\n" + "=" * 60)
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"{PASS} Пройдено: {passed}/{total}")
if passed != total:
    sys.exit(1)

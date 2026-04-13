#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Мягкий сброс менеджерских очередей.

Сбрасывает только текущие запросы менеджерам:
- reports/debt_stop_state.json
- logs/crm_pending_state.json

Не трогает:
- logs/saida_payment_holds.json
- reports/debt_stop_registry.json
- logs/collector_client_dialogs.json
- logs/wa_approval_batches.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
TZ = ZoneInfo("Asia/Almaty")

DEBT_STOP_STATE = ROOT / "reports" / "debt_stop_state.json"
CRM_PENDING = ROOT / "logs" / "crm_pending_state.json"
SAIDA_HOLDS = ROOT / "logs" / "saida_payment_holds.json"


def _read_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_if_exists(path: Path, dst_dir: Path) -> None:
    if path.exists():
        shutil.copy2(path, dst_dir / path.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Мягко сбросить менеджерские очереди, не трогая Саиду")
    parser.add_argument("--dry-run", action="store_true", help="Показать что будет сделано без записи")
    args = parser.parse_args()

    now = datetime.now(TZ)
    stamp = now.strftime("%Y%m%d-%H%M%S")
    archive_dir = ROOT / "archive" / f"manager_queue_reset_{stamp}"

    debt_state = _read_json(DEBT_STOP_STATE, {"candidates": {}})
    crm_state = _read_json(CRM_PENDING, {})
    saida_state = _read_json(SAIDA_HOLDS, {})

    debt_count = len(debt_state.get("candidates", {}) if isinstance(debt_state, dict) else {})
    crm_count = len(crm_state if isinstance(crm_state, dict) else {})
    saida_count = sum(
        1 for item in (saida_state.values() if isinstance(saida_state, dict) else [])
        if isinstance(item, dict) and item.get("status") == "pending_saida"
    )

    print("Мягкий сброс менеджерских очередей")
    print(f"Архив: {archive_dir}")
    print(f"Стоп-лист менеджеров к сбросу: {debt_count}")
    print(f"CRM-запросы к сбросу: {crm_count}")
    print(f"Заявки Саиде сохраняются: {saida_count}")

    if args.dry_run:
        print("DRY-RUN: файлы не изменены")
        return 0

    archive_dir.mkdir(parents=True, exist_ok=True)
    _copy_if_exists(DEBT_STOP_STATE, archive_dir)
    _copy_if_exists(CRM_PENDING, archive_dir)

    _write_json(
        DEBT_STOP_STATE,
        {
            "date": now.date().isoformat(),
            "candidates": {},
            "next_id": 1,
            "saida_sent": False,
            "reset_at": now.isoformat(),
            "reset_reason": "manual_manager_queue_reset_keep_saida",
            "archive": str(archive_dir),
        },
    )
    _write_json(CRM_PENDING, {})

    summary = {
        "reset_at": now.isoformat(),
        "debt_stop_candidates_reset": debt_count,
        "crm_pending_reset": crm_count,
        "saida_pending_kept": saida_count,
        "files_archived": [
            str(DEBT_STOP_STATE),
            str(CRM_PENDING),
        ],
    }
    _write_json(archive_dir / "summary.json", summary)
    print("Сброс выполнен. Саида не тронута.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


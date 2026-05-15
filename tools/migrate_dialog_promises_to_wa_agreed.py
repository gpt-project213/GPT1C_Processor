"""One-shot миграция: client-promise из collector_client_dialogs.json
→ wa_agreed_promises.json через record_client_promise().

Нужна один раз для существующих overdue dialog-promises которые
до фикса F-B1 (commit с record_client_promise) висели только в dialog-state
и не подбирались handler'ом check_broken_agreed_deadlines (10:30).

Запуск: python tools/migrate_dialog_promises_to_wa_agreed.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collector.approval_flow import record_client_promise, _load_promises  # noqa: E402

DIALOGS_PATH = ROOT / "logs" / "collector_client_dialogs.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="не записывать, только показать")
    args = ap.parse_args()

    if not DIALOGS_PATH.exists():
        print(f"[skip] {DIALOGS_PATH} не существует")
        return 0

    dialogs = json.loads(DIALOGS_PATH.read_text(encoding="utf-8"))
    existing_promises = _load_promises()

    candidates = []
    for phone, d in dialogs.items():
        if not isinstance(d, dict):
            continue
        if d.get("state") not in ("active", "escalated"):
            continue
        promise_date = d.get("promise_date")
        if not promise_date:
            continue
        client_name = (d.get("client_name") or "").strip()
        manager_name = (d.get("manager_name") or "").strip()
        if not client_name or not manager_name:
            continue
        # Skip если уже есть active manager-promise (не наш source)
        existing = existing_promises.get(client_name)
        if existing and existing.get("status") in ("active", "accepted") and existing.get("source") != "client_dialog":
            print(f"[skip] {client_name!r} — уже есть manager-promise (status={existing.get('status')})")
            continue
        candidates.append({
            "client_name":   client_name,
            "manager_name":  manager_name,
            "promise_date":  promise_date,
            "amount":        d.get("promise_amount") or "",
            "details":       f"WA dialog (migration): обещал оплатить до {promise_date}",
        })

    if not candidates:
        print("[done] кандидатов на миграцию нет")
        return 0

    print(f"[plan] {len(candidates)} dialog-promise(s) для миграции:")
    for c in candidates:
        print(f"  - {c['client_name']!r} -> {c['promise_date']} (mgr={c['manager_name']!r})")

    if args.dry_run:
        print("\n[dry-run] записи НЕ внесены. Уберите --dry-run для применения.")
        return 0

    written = 0
    for c in candidates:
        ok = record_client_promise(
            client_name=c["client_name"],
            manager_name=c["manager_name"],
            promise_date=c["promise_date"],
            details=c["details"],
        )
        if ok:
            written += 1
            print(f"[write] {c['client_name']!r} -> wa_agreed_promises.json")
        else:
            print(f"[fail]  {c['client_name']!r} — record_client_promise вернул False")

    print(f"\n[done] записано: {written}/{len(candidates)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

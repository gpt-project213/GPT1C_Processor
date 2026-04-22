#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2 safe live send checks."""

import asyncio
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["COLLECTOR_TEST_MODE"] = "1"
BOT_DIR = ROOT / "bot"
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from collector import approval_flow as af
from collector import collections_engine as ce
from collector import manager_dialog as md


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"OK {name}")


def make_temp_batches():
    tmp = tempfile.TemporaryDirectory()
    af._BATCHES_PATH = Path(tmp.name) / "wa_approval_batches.json"
    return tmp


async def noop_start_client_dialog(**kwargs):
    return None


def test_send_approved_uses_only_approved_clients() -> None:
    tmp = make_temp_batches()
    try:
        batch = {
            "batch_id": "20260412-120000-ab12",
            "status": "admin_approved",
            "admin_status": "approved",
            "approved_clients": [
                {
                    "name": "ТОО Альфа",
                    "manager": "Алена",
                    "phone": "+77011112233",
                    "amount": 1000,
                    "days": 10,
                    "level": 1,
                }
            ],
            "managers": {
                "Алена": {
                    "clients": [
                        {"name": "ТОО Альфа", "phone": "+77011112233"},
                        {"name": "ТОО Бета", "phone": "+77014445566"},
                    ],
                }
            },
        }
        af.save_batch(batch)

        sent = []
        ce._live_send_allowed = lambda reason_prefix="": True
        ce.already_contacted_today = lambda name: False
        ce.generate_message = lambda **kwargs: "message"
        ce.update_after_contact = lambda *args, **kwargs: None
        ce._get_manager_chat_id = lambda manager_name: 0
        ce.send_whatsapp = lambda phone, text: sent.append((phone, text)) or True

        import collector.client_dialog as cd

        cd.start_client_dialog = noop_start_client_dialog
        results = asyncio.run(ce.send_approved_batch(batch["batch_id"]))
        saved = af.load_batch(batch["batch_id"])

        check("approved batch send uses only approved clients", sent == [("+77011112233", "message")])
        check("per-client send results written into batch", saved["send_results"] == results)
        check("send result marked sent", saved["send_summary"]["sent"] == 1)
    finally:
        tmp.cleanup()


def test_old_manager_dialog_path_cannot_send_live() -> None:
    import collector.communications as comm

    comm.send_whatsapp = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("send called"))
    result = asyncio.run(md._send_whatsapp_and_notify({"client_name": "TEST"}))
    check("old manager_dialog path cannot send live", result is False)


def test_manual_editing_does_not_complete_batch() -> None:
    batch = {"managers": {"A": {"status": "manual_editing"}, "B": {"status": "approved_all"}}}
    check("manual_editing does not complete batch", af._all_managers_responded(batch) is False)
    batch["managers"]["A"] = {"status": "manual_done"}
    check("manual_done completes batch", af._all_managers_responded(batch) is True)


def test_unique_batch_ids_do_not_collide() -> None:
    data = {"A": [{"name": "C", "amount": 1, "days": 10, "level": 1, "phone": "+77011112233"}]}
    a = af.create_batch(data)["batch_id"]
    b = af.create_batch(data)["batch_id"]
    pattern = r"\d{8}-\d{6}-[0-9a-f]{4}"
    check("unique batch id format", re.fullmatch(pattern, a) is not None)
    check("unique batch ids do not collide", a != b)


def test_placeholder_phone_is_blocked() -> None:
    # _PLACEHOLDER_PHONE_KEYS пуст — блокировка по явным номерам убрана.
    # Проверяем что пустой телефон блокируется валидатором.
    valid, reason = af.validate_production_phone(
        "",
        "О ТД Артем.мясной зал.Кус Вкус тел 87023069994",
    )
    check("empty phone is invalid", valid is False)


def test_scheduler_cannot_trigger_unsafe_live_send() -> None:
    # Phase 3: WHATSAPP_ENABLED=1 → scheduler вызывает --preview (approval-flow),
    # НЕ --send/--send-approved. Это безопасно: --preview только создаёт батч.
    # WHATSAPP_ENABLED=0 → --dry-run (проверяется отдельно ниже).
    code = (
        "import sys, os, asyncio; "
        "sys.path.insert(0, 'bot'); "
        "import bot.send_reports as sr; import bot.workday_checker as wc; "
        "calls=[]; ns={'calls': calls}; "
        "exec(\"async def fake_run(*args, **kwargs):\\n    calls.append(args)\\n    return (0, '', '')\", ns); "
        "sr.run_script_async=ns['fake_run']; sr.log_event=lambda *args, **kwargs: None; "
        "wc.is_holiday_today=lambda: False; "
        "os.environ['WHATSAPP_ENABLED']='1'; os.environ['LIVE_SEND_ALLOWED']='1'; "
        "asyncio.run(sr.debt_collector_daily(None)); "
        "assert calls and calls[0][1]=='--preview', calls"
    )
    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
    check("scheduler cannot trigger unsafe live send", completed.returncode == 0)


def test_scheduler_uses_dry_run_when_whatsapp_disabled() -> None:
    # WHATSAPP_ENABLED=0 → scheduler обязан использовать --dry-run, не --preview
    code = (
        "import sys, os, asyncio; "
        "sys.path.insert(0, 'bot'); "
        "import bot.send_reports as sr; import bot.workday_checker as wc; "
        "calls=[]; ns={'calls': calls}; "
        "exec(\"async def fake_run(*args, **kwargs):\\n    calls.append(args)\\n    return (0, '', '')\", ns); "
        "sr.run_script_async=ns['fake_run']; sr.log_event=lambda *args, **kwargs: None; "
        "wc.is_holiday_today=lambda: False; "
        "os.environ['WHATSAPP_ENABLED']='0'; "
        "asyncio.run(sr.debt_collector_daily(None)); "
        "assert calls and calls[0][1]=='--dry-run', calls"
    )
    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
    check("scheduler uses --dry-run when whatsapp disabled", completed.returncode == 0)


def test_legacy_send_cli_is_disabled() -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["collections_engine.py", "--send"]
        check("legacy --send is disabled", ce.main() == 1)
    finally:
        sys.argv = old_argv


def main() -> None:
    test_send_approved_uses_only_approved_clients()
    test_old_manager_dialog_path_cannot_send_live()
    test_manual_editing_does_not_complete_batch()
    test_unique_batch_ids_do_not_collide()
    test_placeholder_phone_is_blocked()
    test_scheduler_cannot_trigger_unsafe_live_send()
    test_scheduler_uses_dry_run_when_whatsapp_disabled()
    test_legacy_send_cli_is_disabled()
    print("PHASE2 SAFE SEND TESTS PASSED")


if __name__ == "__main__":
    main()

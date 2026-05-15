#!/usr/bin/env python
# coding: utf-8
"""
Hermetic collector regressions.

Запуск:
    python -X utf8 tests/test_collector_regression_hermetic.py -v

Цель:
  - никакой реальной отправки в WhatsApp / Telegram;
  - никакой зависимости от сети;
  - быстрый regression-suite только для спорных collector-сценариев.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["COLLECTOR_TEST_MODE"] = "1"

import collector.client_dialog as client_dialog
import collector.debt_monitor as debt_monitor
import collector.collections_engine as collections_engine
import collector.approval_flow as approval_flow
import collector.whatsapp_poller as whatsapp_poller


class _DummyResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class _DummyAsyncClient:
    def __init__(self, response: _DummyResponse):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url):
        return self._response


class ClientDialogHermeticTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._orig_dialogs_path = client_dialog._DIALOGS_PATH
        client_dialog._DIALOGS_PATH = Path(self._tmpdir) / "collector_client_dialogs.json"
        # Изолируем wa_approval_batches.json — иначе send_approved_batch
        # пишет тестовые ключи (`batch-1`/`batch-2`) в боевой logs/wa_approval_batches.json.
        self._orig_batches_path = approval_flow._BATCHES_PATH
        approval_flow._BATCHES_PATH = Path(self._tmpdir) / "wa_approval_batches.json"

    def tearDown(self):
        client_dialog._DIALOGS_PATH = self._orig_dialogs_path
        approval_flow._BATCHES_PATH = self._orig_batches_path
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    async def _start_dialog(self, phone: str = "77010000001") -> None:
        await client_dialog.start_client_dialog(
            phone=phone,
            client_name="Тест Клиент",
            manager_name="Ергали",
            manager_chat_id=123,
            level=2,
            days=12,
            amount=50000.0,
            message_text="Напоминание по задолженности.",
        )

    async def test_paid_claim_enters_awaiting_payment_proof(self):
        await self._start_dialog()
        with patch("collector.collection_agent._call_deepseek") as ai_mock, \
             patch("collector.client_dialog._reply_to_client") as reply_mock, \
             patch("collector.client_dialog._notify_dialog_observers", new=AsyncMock()) as note_mock:
            ai_mock.return_value = (
                '{"intent":"paid_claim","promise_date":null,"promise_amount":null,'
                '"requires_human":false,"suggested_reply":""}'
            )
            reply_mock.return_value = None
            await client_dialog.handle_incoming("77010000001", "Я уже оплатил")

        dialog = client_dialog._get_client_dialog("77010000001")
        self.assertEqual(dialog.get("state"), "awaiting_payment_proof")
        self.assertTrue(dialog.get("awaiting_payment_proof"))
        self.assertEqual(note_mock.await_count, 1)
        last_bot = [ex["text"] for ex in dialog["exchanges"] if ex["role"] == "bot"][-1].lower()
        self.assertIn("чек", last_bot)

    async def test_short_ack_after_paid_claim_does_not_trigger_second_reply(self):
        await self._start_dialog()
        with patch("collector.collection_agent._call_deepseek") as ai_mock, \
             patch("collector.client_dialog._reply_to_client") as reply_mock, \
             patch("collector.client_dialog._notify_dialog_observers", new=AsyncMock()):
            ai_mock.return_value = (
                '{"intent":"paid_claim","promise_date":null,"promise_amount":null,'
                '"requires_human":false,"suggested_reply":""}'
            )
            reply_mock.return_value = None
            await client_dialog.handle_incoming("77010000001", "Оплатили вчера")

        before = client_dialog._get_client_dialog("77010000001")
        bot_count_before = len([ex for ex in before["exchanges"] if ex["role"] == "bot"])

        with patch("collector.client_dialog._reply_to_client") as reply_mock, \
             patch("collector.client_dialog._notify_dialog_observers", new=AsyncMock()):
            reply_mock.return_value = None
            await client_dialog.handle_incoming("77010000001", "Хорошо")

        after = client_dialog._get_client_dialog("77010000001")
        bot_count_after = len([ex for ex in after["exchanges"] if ex["role"] == "bot"])
        self.assertEqual(bot_count_before, bot_count_after)

    async def test_payment_proof_attachment_goes_to_manager_without_reasking_client(self):
        await self._start_dialog()
        with patch("collector.collection_agent._call_deepseek") as ai_mock, \
             patch("collector.client_dialog._reply_to_client") as reply_mock, \
             patch("collector.client_dialog._notify_dialog_observers", new=AsyncMock()):
            ai_mock.return_value = (
                '{"intent":"paid_claim","promise_date":null,"promise_amount":null,'
                '"requires_human":false,"suggested_reply":""}'
            )
            reply_mock.return_value = None
            await client_dialog.handle_incoming("77010000001", "Давно оплатили")

        with patch("collector.client_dialog._reply_to_client") as reply_mock, \
             patch("collector.client_dialog._notify_dialog_observers", new=AsyncMock()) as note_mock:
            reply_mock.return_value = None
            await client_dialog.handle_incoming(
                "77010000001",
                "[клиент прислал documentMessage]",
                attachment={
                    "type": "documentMessage",
                    "download_url": "https://example.test/receipt.pdf",
                    "file_name": "receipt.pdf",
                    "caption": "чек оплаты",
                },
            )

        dialog = client_dialog._get_client_dialog("77010000001")
        self.assertEqual(dialog.get("state"), "awaiting_manager")
        self.assertFalse(dialog.get("awaiting_payment_proof"))
        self.assertEqual(note_mock.await_count, 1)
        note_text = note_mock.await_args.args[1]
        self.assertIn("https://example.test/receipt.pdf", note_text)
        last_bot = [ex["text"] for ex in dialog["exchanges"] if ex["role"] == "bot"][-1].lower()
        self.assertIn("передали менеджеру", last_bot)

    async def test_service_request_escalates_without_debt_push(self):
        await self._start_dialog(phone="77010000002")
        with patch("collector.client_dialog._reply_to_client") as reply_mock, \
             patch("collector.client_dialog.escalate_to_manager", new=AsyncMock()) as escalate_mock:
            reply_mock.return_value = None
            await client_dialog.handle_incoming("77010000002", "Акт сверки сбросьте за апрель")

        dialog = client_dialog._get_client_dialog("77010000002")
        self.assertEqual(escalate_mock.await_count, 1)
        last_bot = [ex["text"] for ex in dialog["exchanges"] if ex["role"] == "bot"][-1].lower()
        self.assertIn("передаю вас менеджеру", last_bot)
        self.assertNotIn("первый плат", last_bot)


class FreshnessGateHermeticTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Изолируем wa_approval_batches.json от продакшна — иначе
        # send_approved_batch пишет тестовые ключи (`batch-1`/`batch-2`/`batch-stale`)
        # в боевой logs/wa_approval_batches.json.
        self._tmpdir = tempfile.mkdtemp()
        self._orig_batches_path = approval_flow._BATCHES_PATH
        approval_flow._BATCHES_PATH = Path(self._tmpdir) / "wa_approval_batches.json"

    def tearDown(self):
        approval_flow._BATCHES_PATH = self._orig_batches_path
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_load_latest_debt_json_exposes_stale_freshness_metadata(self):
        tmpdir = tempfile.mkdtemp()
        original_json_dir = debt_monitor.JSON_DIR
        debt_monitor.JSON_DIR = Path(tmpdir)
        try:
            payload = {
                "manager": "Ергали",
                "period_max": "26.04.2026",
                "clients": [
                    {"name": "Е ИП Шахин", "debt": 340000, "days": 33}
                ],
            }
            (debt_monitor.JSON_DIR / "debt_ext_test.json").write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            loaded = debt_monitor.load_latest_debt_json()
        finally:
            debt_monitor.JSON_DIR = original_json_dir
            shutil.rmtree(tmpdir, ignore_errors=True)

        freshness = loaded.get("_freshness") or {}
        self.assertTrue(freshness.get("is_stale"))
        self.assertIn("Ергали", freshness.get("stale_managers") or [])
        self.assertEqual((freshness.get("managers") or {}).get("Ергали", {}).get("period_max_ru"), "26.04.2026")

    async def test_send_approved_batch_refreshes_client_before_send(self):
        approved_client = {
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
             patch("collector.approval_flow.get_approved_clients", return_value=[dict(approved_client)]), \
             patch("collector.approval_flow.load_batch", return_value={"batch_id": "batch-1", "created_at": "2026-04-28T13:00:00+05:00"}), \
             patch("collector.approval_flow.save_batch") as _save_batch_mock1, \
             patch("collector.approval_flow.record_send_results") as record_mock, \
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
             patch("collector.collections_engine._send_approved_client", new=AsyncMock(return_value={"name": "Е Еркебулан", "status": "sent", "reason": "ok"})) as send_mock, \
             patch("collector.collections_engine.notify_admin", new=AsyncMock()) as admin_mock:
            results = await collections_engine.send_approved_batch("batch-1")

        payload = send_mock.await_args.args[0]
        self.assertEqual(payload.get("amount"), 120000.0)
        self.assertEqual(payload.get("msg_type"), "soft_reminder")
        self.assertEqual(admin_mock.await_count, 1)
        self.assertTrue(record_mock.called)
        self.assertTrue(any(r.get("status") == "sent" for r in results))

    async def test_send_approved_batch_skips_removed_stale_client(self):
        approved_client = {
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
             patch("collector.approval_flow.get_approved_clients", return_value=[dict(approved_client)]), \
             patch("collector.approval_flow.load_batch", return_value={"batch_id": "batch-2", "created_at": "2026-04-28T13:00:00+05:00"}), \
             patch("collector.approval_flow.save_batch") as _save_batch_mock2, \
             patch("collector.approval_flow.record_send_results") as record_mock, \
             patch("collector.collections_engine._live_send_allowed", return_value=True), \
             patch("collector.collections_engine.load_latest_debt_json", return_value={"clients": []}), \
             patch("collector.collections_engine.classify_debtors", return_value=[]), \
             patch("collector.collections_engine._send_approved_client", new=AsyncMock()) as send_mock, \
             patch("collector.collections_engine.notify_admin", new=AsyncMock()) as admin_mock:
            results = await collections_engine.send_approved_batch("batch-2")

        self.assertEqual(send_mock.await_count, 0)
        self.assertEqual(admin_mock.await_count, 1)
        self.assertTrue(record_mock.called)
        self.assertTrue(any("stale approved batch" in str(r.get("reason", "")) for r in results))

    async def test_send_approved_batch_blocks_when_selected_manager_data_is_stale(self):
        approved_client = {
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
        stale_debt = {
            "clients": [{"name": "Е Еркебулан"}],
            "_freshness": {
                "warn_threshold_days": 1,
                "block_threshold_days": 2,
                "managers": {
                    "Ергали": {
                        "period_max": "2026-04-26",
                        "period_max_ru": "26.04.2026",
                        "age_days": 3,
                        "warn": True,
                        "stale": True,
                        "file": "debt_ext_test.json",
                    }
                },
            },
        }
        with patch("collector.approval_flow.is_ready_for_send", return_value=True), \
             patch("collector.approval_flow.get_approved_clients", return_value=[dict(approved_client)]), \
             patch("collector.approval_flow.load_batch", return_value={"batch_id": "batch-stale", "created_at": "2026-04-28T13:00:00+05:00"}), \
             patch("collector.approval_flow.save_batch") as _save_batch_mock5, \
             patch("collector.approval_flow.record_send_results") as record_mock, \
             patch("collector.collections_engine._live_send_allowed", return_value=True), \
             patch("collector.collections_engine.load_latest_debt_json", return_value=stale_debt), \
             patch("collector.collections_engine.classify_debtors") as classify_mock, \
             patch("collector.collections_engine._send_approved_client", new=AsyncMock()) as send_mock, \
             patch("collector.collections_engine.notify_admin", new=AsyncMock()) as admin_mock:
            results = await collections_engine.send_approved_batch("batch-stale")

        self.assertEqual(classify_mock.call_count, 0)
        self.assertEqual(send_mock.await_count, 0)
        self.assertEqual(record_mock.call_count, 0)
        self.assertEqual(results, [])
        self.assertEqual(admin_mock.await_count, 1)
        self.assertIn("26.04.2026", admin_mock.await_args.args[0])


class LegacyTailClassificationHermeticTests(unittest.TestCase):
    def test_stopped_legacy_tail_without_payments_uses_legacy_tail_msg_type(self):
        decision = collections_engine._collector_candidate_decision(
            {
                "name": "Е ИП Шахин",
                "amount": 340000.0,
                "days": 33,
                "opening": 340000.0,
                "debit": 0.0,
                "credit": 0.0,
            },
            {"whatsapp": "77025626272", "manager": "Ергали"},
            {"status": "stopped"},
        )
        self.assertEqual(decision.get("action"), "client_approval")
        self.assertEqual(decision.get("msg_type"), "legacy_tail_reminder")
        self.assertNotIn("отгруз", decision.get("reason", "").lower())

    def test_stopped_legacy_tail_with_partial_payments_uses_partial_tail_msg_type(self):
        decision = collections_engine._collector_candidate_decision(
            {
                "name": "Е Еркебулан",
                "amount": 739409.67,
                "days": 28,
                "opening": 767268.67,
                "debit": 0.0,
                "credit": 27859.0,
            },
            {"whatsapp": "77087578717", "manager": "Ергали"},
            {"status": "stopped"},
        )
        self.assertEqual(decision.get("action"), "client_approval")
        self.assertEqual(decision.get("msg_type"), "partial_tail_reminder")
        self.assertIn("оплата", decision.get("reason", "").lower())

    def test_live_stop_case_keeps_stoplist_reminder(self):
        decision = collections_engine._collector_candidate_decision(
            {
                "name": "Активный клиент",
                "amount": 250000.0,
                "days": 8,
                "opening": 250000.0,
                "debit": 125000.0,
                "credit": 0.0,
            },
            {"whatsapp": "77010001122", "manager": "Ергали"},
            {"status": "stopped"},
        )
        self.assertEqual(decision.get("msg_type"), "stoplist_reminder")

    def test_generate_message_for_legacy_tail_has_no_shipments_phrase(self):
        text = collections_engine.generate_message(
            client_name="Е ИП Шахин",
            debt_amount=340000.0,
            days_overdue=33,
            level=5,
            language="ru",
            manager_name="Ергали",
            msg_type="legacy_tail_reminder",
            report_date="2026-04-27",
        )
        self.assertIn("задолженность", text.lower())
        self.assertNotIn("отгруз", text.lower())


class PreviewFreshnessFormattingHermeticTests(unittest.TestCase):
    def test_manager_preview_shows_snapshot_date_and_warning(self):
        batch = {
            "batch_id": "batch-1",
            "debt_snapshot": {
                "snapshot_label_ru": "28.04.2026",
                "max_age_days": 1,
                "has_warning": True,
                "warning_managers": ["Ергали"],
            },
        }
        text = approval_flow._format_manager_preview_text(
            "Ергали",
            [{
                "name": "Е ИП Шахин",
                "amount": 340000.0,
                "days": 33,
                "debit": 0.0,
                "credit": 0.0,
                "msg_type": "legacy_tail_reminder",
                "reason": "старый хвост",
            }],
            "batch-1",
            batch=batch,
        )
        self.assertIn("Данные дебиторки", text)
        self.assertIn("28.04.2026", text)
        self.assertIn("Ергали", text)

    def test_admin_summary_shows_snapshot_date_and_warning(self):
        batch = approval_flow.create_batch({
            "Ергали": [{
                "name": "Е ИП Шахин",
                "amount": 340000.0,
                "days": 33,
                "level": 5,
                "debit": 0.0,
                "credit": 0.0,
                "phone": "77025626272",
                "msg_type": "legacy_tail_reminder",
                "reason": "старый хвост",
            }]
        })
        batch["debt_snapshot"] = {
            "snapshot_label_ru": "28.04.2026",
            "max_age_days": 1,
            "has_warning": True,
            "warning_managers": ["Ергали"],
        }
        text = approval_flow._format_admin_summary_text(batch)
        self.assertIn("Данные дебиторки", text)
        self.assertIn("28.04.2026", text)
        self.assertIn("Ергали", text)


class TimeoutLabelHermeticTests(unittest.TestCase):
    """Regression: timeout не должен молча показывать «не ответил»,
    если менеджер успел нажать кнопку и оставил bot в waiting_for_*.
    """

    def _make_batch(self, mgr_name: str = "Магира"):
        return approval_flow.create_batch({
            mgr_name: [{
                "name": "Е Тестовый Клиент",
                "amount": 100000.0,
                "days": 12,
                "level": 3,
                "debit": 0.0,
                "credit": 0.0,
                "phone": "77001112233",
                "msg_type": "reminder",
                "reason": "просрочка",
            }]
        })

    def test_timeout_with_waiting_for_agreed_shows_started_label(self):
        batch = self._make_batch("Магира")
        mgr_state = batch["managers"]["Магира"]
        mgr_state["status"] = "timeout"
        mgr_state["waiting_for_agreed"] = {
            "client_name": "Е Тестовый Клиент",
            "batch_id":    batch["batch_id"],
            "cli_idx":     0,
        }
        text = approval_flow._format_admin_summary_text(batch)
        self.assertIn("начал — не написал детали", text)
        self.assertIn("Е Тестовый Клиент", text)
        self.assertNotIn("🔇 не ответил", text)

    def test_timeout_with_waiting_for_proof_shows_started_label(self):
        batch = self._make_batch("Оксана")
        mgr_state = batch["managers"]["Оксана"]
        mgr_state["status"] = "timeout"
        mgr_state["waiting_for_proof"] = {
            "client_name": "Е Тестовый Клиент",
            "batch_id":    batch["batch_id"],
            "cli_idx":     0,
        }
        text = approval_flow._format_admin_summary_text(batch)
        self.assertIn("начал — не прислал документ", text)
        self.assertIn("Е Тестовый Клиент", text)
        self.assertNotIn("🔇 не ответил", text)

    def test_timeout_without_waiting_keeps_silent_label(self):
        batch = self._make_batch("Ергали")
        mgr_state = batch["managers"]["Ергали"]
        mgr_state["status"] = "timeout"
        text = approval_flow._format_admin_summary_text(batch)
        self.assertIn("🔇 не ответил", text)
        self.assertNotIn("начал — не написал", text)
        self.assertNotIn("начал — не прислал", text)


class ClientPromiseRecordHermeticTests(unittest.TestCase):
    """Regressions для F-B1: client-promise из WA-диалога должен попадать
    в wa_agreed_promises.json через record_client_promise(), чтобы handler
    check_broken_agreed_deadlines (10:30) его подхватил.
    """

    def setUp(self):
        # Изолируем _PROMISES_PATH в tempdir
        self._tmp = tempfile.TemporaryDirectory()
        self._tmp_path = Path(self._tmp.name) / "wa_agreed_promises.json"
        self._patcher = patch.object(approval_flow, "_PROMISES_PATH", self._tmp_path)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmp.cleanup()

    def _load(self) -> dict:
        if not self._tmp_path.exists():
            return {}
        return json.loads(self._tmp_path.read_text(encoding="utf-8"))

    def test_record_creates_new_entry(self):
        ok = approval_flow.record_client_promise(
            client_name="М Гриль Косши ул Республика 18 б",
            manager_name="Магира",
            promise_date="2026-05-20",
            details="WA dialog: обещал до 20 мая",
        )
        self.assertTrue(ok)
        data = self._load()
        entry = data.get("М Гриль Косши ул Республика 18 б")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["deadline"], "2026-05-20")
        self.assertEqual(entry["manager"], "Магира")
        self.assertEqual(entry["status"], "active")
        self.assertEqual(entry["source"], "client_dialog")
        self.assertEqual(entry["batch_id"], "client_dialog")

    def test_record_does_not_override_manager_promise(self):
        # Сначала manager создаёт promise через "Договорились"
        approval_flow.save_agreed_promise(
            client_name="М Тестклиент",
            manager_name="Магира",
            details="до 15 мая",
            batch_id="20260513-170000-aaaa",
        )
        # Затем клиент в WA-диалоге обещает другую дату — не должно перезаписать
        ok = approval_flow.record_client_promise(
            client_name="М Тестклиент",
            manager_name="Магира",
            promise_date="2026-05-25",
            details="WA dialog: до 25 мая",
        )
        self.assertFalse(ok)
        entry = self._load()["М Тестклиент"]
        self.assertNotEqual(entry["deadline"], "2026-05-25")
        self.assertNotEqual(entry.get("source"), "client_dialog")
        self.assertEqual(entry.get("batch_id"), "20260513-170000-aaaa")

    def test_record_updates_existing_client_promise(self):
        # Клиент уже зафиксирован через WA — потом уточнил дату
        approval_flow.record_client_promise(
            client_name="О Тестклиент2",
            manager_name="Оксана",
            promise_date="2026-05-20",
            details="WA dialog: до 20 мая",
        )
        approval_flow.record_client_promise(
            client_name="О Тестклиент2",
            manager_name="Оксана",
            promise_date="2026-05-22",
            details="WA dialog: уточнил до 22 мая",
        )
        entry = self._load()["О Тестклиент2"]
        self.assertEqual(entry["deadline"], "2026-05-22")
        self.assertEqual(entry["source"], "client_dialog")

    def test_record_reopens_after_broken(self):
        # Был manager-promise, нарушен (status=broken) — клиент даёт новое обещание в WA
        approval_flow.save_agreed_promise(
            client_name="Е Тестклиент3",
            manager_name="Ергали",
            details="до 10 мая",
            batch_id="20260510-170000-bbbb",
        )
        promises = approval_flow._load_promises()
        promises["Е Тестклиент3"]["status"] = "broken"
        approval_flow._save_promises(promises)
        # Теперь client_dialog должен записать новый promise
        ok = approval_flow.record_client_promise(
            client_name="Е Тестклиент3",
            manager_name="Ергали",
            promise_date="2026-05-25",
            details="WA dialog: новая дата",
        )
        self.assertTrue(ok)
        entry = self._load()["Е Тестклиент3"]
        self.assertEqual(entry["deadline"], "2026-05-25")
        self.assertEqual(entry["status"], "active")
        self.assertEqual(entry["source"], "client_dialog")

    def test_record_rejects_invalid_date(self):
        ok = approval_flow.record_client_promise(
            client_name="М Клиент",
            manager_name="Магира",
            promise_date="not-a-date",
            details="WA dialog",
        )
        self.assertFalse(ok)
        self.assertEqual(self._load(), {})

    def test_record_rejects_empty_fields(self):
        self.assertFalse(approval_flow.record_client_promise("", "Магира", "2026-05-20"))
        self.assertFalse(approval_flow.record_client_promise("Клиент", "", "2026-05-20"))
        self.assertFalse(approval_flow.record_client_promise("Клиент", "Магира", ""))


class WhatsAppPollerHermeticTests(unittest.IsolatedAsyncioTestCase):
    async def test_poll_once_passes_attachment_metadata_to_client_dialog(self):
        payload = {
            "receiptId": 101,
            "body": {
                "typeWebhook": "incomingMessageReceived",
                "senderData": {"sender": "77015554433@c.us"},
                "messageData": {
                    "typeMessage": "documentMessage",
                    "fileMessageData": {
                        "downloadUrl": "https://example.test/proof.jpg",
                        "fileName": "proof.jpg",
                        "caption": "чек",
                    },
                },
            },
        }
        with patch.object(whatsapp_poller, "GREENAPI_ID", "gid"), \
             patch.object(whatsapp_poller, "GREENAPI_TOKEN", "gtoken"), \
             patch.object(whatsapp_poller, "_should_process_incoming", return_value=True), \
             patch.object(whatsapp_poller, "_delete_notification", new=AsyncMock()) as delete_mock, \
             patch("collector.client_dialog.handle_incoming", new=AsyncMock()) as handle_mock, \
             patch.object(
                 whatsapp_poller.httpx,
                 "AsyncClient",
                 return_value=_DummyAsyncClient(_DummyResponse(payload)),
             ):
            await whatsapp_poller.poll_once()

        self.assertEqual(handle_mock.await_count, 1)
        args = handle_mock.await_args.args
        kwargs = handle_mock.await_args.kwargs
        self.assertEqual(args[0], "77015554433")
        self.assertEqual(args[1], "чек")
        self.assertEqual(kwargs["attachment"]["download_url"], "https://example.test/proof.jpg")
        self.assertEqual(kwargs["attachment"]["file_name"], "proof.jpg")
        self.assertEqual(kwargs["attachment"]["type"], "documentMessage")
        self.assertEqual(delete_mock.await_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

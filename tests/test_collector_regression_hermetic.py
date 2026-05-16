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

    def test_sync_helper_skipped_in_test_mode(self):
        """_TEST_MODE guard в _sync_promise_to_agreed защищает prod state.

        Критично: без guard юнит-тесты, прогоняющие handle_incoming с
        intent=promise, контаминировали бы боевой wa_agreed_promises.json.
        Прецедент 2026-05-15: запись "Кайрбек" с deadline=2026-04-22 (overdue)
        могла бы спровоцировать ложное «обещание нарушено» уведомление в проде.
        """
        from collector import client_dialog as cd
        # _TEST_MODE должен быть True (COLLECTOR_TEST_MODE=1 установлен на уровне файла)
        self.assertTrue(cd._TEST_MODE, "тестовая среда должна иметь _TEST_MODE=True")

        dialog = {"client_name": "Тестклиент", "manager_name": "Магира"}
        result = cd._sync_promise_to_agreed(dialog, "2026-05-20", "WA dialog: тест")

        # В test mode — никакая запись не должна быть сохранена
        self.assertFalse(result)
        self.assertEqual(self._load(), {})

    def test_sync_helper_writes_when_not_test_mode(self):
        """Без _TEST_MODE helper должен записать через record_client_promise.

        Защита: явно проверяем что _PROMISES_PATH замокан в tempdir
        перед тем как трогать запись.
        """
        from collector import client_dialog as cd
        # Защита от случайной записи в боевой файл
        self.assertEqual(
            approval_flow._PROMISES_PATH, self._tmp_path,
            "_PROMISES_PATH должен указывать на tempfile, а не на боевой logs/",
        )
        dialog = {"client_name": "Тестклиент", "manager_name": "Магира"}
        with patch.object(cd, "_TEST_MODE", False):
            result = cd._sync_promise_to_agreed(dialog, "2026-05-20", "WA dialog: real")
        self.assertTrue(result)
        data = self._load()
        self.assertIn("Тестклиент", data)
        self.assertEqual(data["Тестклиент"]["deadline"], "2026-05-20")
        self.assertEqual(data["Тестклиент"]["source"], "client_dialog")


class SilenceFullPaymentHermeticTests(unittest.TestCase):
    """Regressions для BUG-saida + BUG-1a (2026-05-16):

    По бизнес-правилу пользователя: молчание = молчание В ОПЛАТЕ.
    Только confirmed_full Саидой снимает клиента из silence-отчёта.
    Частичная (confirmed_partial), pending_saida, rejected — НЕ снимают.
    TTL подтверждения: 7 дней (достаточно для разноски в 1С).
    """

    def setUp(self):
        # Изолируем PAYMENT_HOLD_PATH в tempdir чтобы не трогать боевой файл
        import collector.payment_hold as ph
        self._tmp = tempfile.TemporaryDirectory()
        self._hold_path = Path(self._tmp.name) / "saida_payment_holds.json"
        self._patcher = patch.object(ph, "PAYMENT_HOLD_PATH", self._hold_path)
        self._patcher.start()
        from bot.silence_alerts import SilenceAlert
        self._alerter = SilenceAlert()

    def tearDown(self):
        self._patcher.stop()
        self._tmp.cleanup()

    def _write_holds(self, holds: dict):
        self._hold_path.parent.mkdir(parents=True, exist_ok=True)
        self._hold_path.write_text(json.dumps(holds, ensure_ascii=False), encoding="utf-8")

    def _now_iso(self, delta_days: float = 0):
        from datetime import datetime as _dt, timedelta as _td
        from zoneinfo import ZoneInfo as _ZI
        tz = _ZI("Asia/Almaty")
        return (_dt.now(tz) - _td(days=delta_days)).isoformat()

    def test_full_payment_within_grace_marks_payment_hold(self):
        self._write_holds({
            "tok1": {
                "client": "М Тестклиент",
                "status": "confirmed_full",
                "saida_confirmed_at": self._now_iso(delta_days=1),
            }
        })
        data = [{"client": "М Тестклиент", "debt": 100000, "silence_days": 14}]
        result = self._alerter.apply_payment_holds(data)
        self.assertTrue(result[0].get("payment_hold"))
        self.assertEqual(result[0].get("payment_hold_status"), "confirmed_full")

    def test_partial_payment_does_NOT_mark_payment_hold(self):
        # БИЗНЕС-ПРАВИЛО: только полная оплата снимает молчание.
        self._write_holds({
            "tok1": {
                "client": "М Тестклиент",
                "status": "confirmed_partial",
                "saida_confirmed_at": self._now_iso(delta_days=1),
            }
        })
        data = [{"client": "М Тестклиент", "debt": 100000, "silence_days": 14}]
        result = self._alerter.apply_payment_holds(data)
        self.assertFalse(result[0].get("payment_hold"))

    def test_pending_saida_does_NOT_mark_payment_hold(self):
        # Пока Саида не подтвердила — клиент в молчании.
        self._write_holds({
            "tok1": {
                "client": "М Тестклиент",
                "status": "pending_saida",
                "created_at": self._now_iso(delta_days=1),
            }
        })
        data = [{"client": "М Тестклиент", "debt": 100000, "silence_days": 14}]
        result = self._alerter.apply_payment_holds(data)
        self.assertFalse(result[0].get("payment_hold"))

    def test_rejected_does_NOT_mark_payment_hold(self):
        self._write_holds({
            "tok1": {
                "client": "М Тестклиент",
                "status": "rejected",
                "saida_confirmed_at": self._now_iso(delta_days=1),
            }
        })
        data = [{"client": "М Тестклиент", "debt": 100000, "silence_days": 14}]
        result = self._alerter.apply_payment_holds(data)
        self.assertFalse(result[0].get("payment_hold"))

    def test_full_payment_older_than_grace_does_NOT_mark(self):
        # Подтверждение старше 7 дней — должно было быть разнесено в 1С.
        # Если нет — клиент возвращается в молчание (что-то пошло не так).
        self._write_holds({
            "tok1": {
                "client": "М Тестклиент",
                "status": "confirmed_full",
                "saida_confirmed_at": self._now_iso(delta_days=10),
            }
        })
        data = [{"client": "М Тестклиент", "debt": 100000, "silence_days": 14}]
        result = self._alerter.apply_payment_holds(data)
        self.assertFalse(result[0].get("payment_hold"))

    def test_legacy_short_name_compat(self):
        # BUG-2 enabler: legacy holds могли быть записаны с обрезанным именем.
        # apply_payment_holds должен матчить полное имя с обрезанным holding-именем.
        full = "М Гриль Косши ул Республика 18 б тел 87751827070"
        short = full[:26]
        self._write_holds({
            "tok1": {
                "client": short,
                "status": "confirmed_full",
                "saida_confirmed_at": self._now_iso(delta_days=1),
            }
        })
        data = [{"client": full, "debt": 62385, "silence_days": 14}]
        result = self._alerter.apply_payment_holds(data)
        self.assertTrue(result[0].get("payment_hold"),
                        "Legacy обрезанное имя должно матчиться с полным")

    def test_categorize_skips_full_payment_client(self):
        """End-to-end: confirmed_full клиент не попадает в silence."""
        self._write_holds({
            "tok1": {
                "client": "М Полностью Оплатил",
                "status": "confirmed_full",
                "saida_confirmed_at": self._now_iso(delta_days=2),
            }
        })
        data = [
            {"client": "М Полностью Оплатил", "debt": 100000, "silence_days": 14,
             "debit_amount": 0, "paid_amount": 100000},
            {"client": "М Молчит", "debt": 200000, "silence_days": 14,
             "debit_amount": 0, "paid_amount": 0},
        ]
        data = self._alerter.apply_payment_holds(data)
        categorized = self._alerter.categorize_by_silence(data)
        all_in_silence = [
            c["client"]
            for cat in ("critical", "alarm", "silence", "overdue")
            for c in categorized.get(cat, [])
        ]
        self.assertNotIn("М Полностью Оплатил", all_in_silence)
        self.assertIn("М Молчит", all_in_silence)


class DebtAgeHistoryHermeticTests(unittest.TestCase):
    """Regressions для BUG-5 (2026-05-16): reset окна Саиды 15-го числа.

    Проблема: 14.05 Еркебулан показывался как "30 дн КРИТИЧНО с 14.04",
    15.05 — как "14 дн МОЛЧАНИЕ" потому что окно Саиды сжалось до 01.05-15.05.
    Долг тот же, возраст потерян. Решение: persistence min(saved, current).
    """

    def setUp(self):
        import bot.silence_alerts as sa
        self._tmp = tempfile.TemporaryDirectory()
        self._history_path = Path(self._tmp.name) / "debt_age_history.json"
        self._patcher = patch.object(sa.SilenceAlert, "DEBT_AGE_HISTORY_PATH", self._history_path)
        self._patcher.start()
        self._alerter = sa.SilenceAlert()

    def tearDown(self):
        self._patcher.stop()
        self._tmp.cleanup()

    def _load_history(self):
        if not self._history_path.exists():
            return {}
        return json.loads(self._history_path.read_text(encoding="utf-8"))

    def test_load_history_empty_returns_dict(self):
        self.assertEqual(self._alerter._load_debt_age_history(), {})

    def test_save_and_load_roundtrip(self):
        h = {"Е Еркебулан": {"oldest_unpaid_date": "2026-04-14", "last_updated": "2026-05-16T09:00:00+05:00"}}
        self._alerter._save_debt_age_history(h)
        self.assertEqual(self._alerter._load_debt_age_history(), h)

    def test_min_logic_preserves_earlier_date(self):
        """Если в истории 2026-04-14, а текущий отчёт даёт 2026-05-01 — используем 04-14."""
        # Подготовим историю с ранней датой
        self._alerter._save_debt_age_history({
            "Е Еркебулан": {"oldest_unpaid_date": "2026-04-14",
                            "last_updated": "2026-05-14T09:00:00+05:00"}
        })

        # Мокаем classify_debtors / load_latest_debt_json чтобы вернуть current=2026-05-01
        from bot import silence_alerts as sa_mod
        with patch("collector.debt_monitor.load_latest_debt_json", return_value={}), \
             patch("collector.debt_monitor.classify_debtors", return_value=[
                 {"name": "Е Еркебулан", "debt": 739409.67,
                  "oldest_unpaid_date": "2026-05-01",  # reset окна
                  "residual_debt_age_days": 14,
                  "days": 14}
             ]):
            data = [{"client": "Е Еркебулан", "debt": 739409.67, "silence_days": 14}]
            result = self._alerter.apply_residual_debt_age(data)

        # Должны взять сохранённую более раннюю дату
        self.assertEqual(result[0]["oldest_unpaid_date"], "2026-04-14")
        # И возраст пересчитан от 14.04 (сегодня минус 14.04 = ~32 дня)
        self.assertGreaterEqual(result[0]["residual_debt_age_days"], 30)

    def test_new_client_writes_to_history(self):
        # Клиент впервые увиден — должен быть записан в историю
        with patch("collector.debt_monitor.load_latest_debt_json", return_value={}), \
             patch("collector.debt_monitor.classify_debtors", return_value=[
                 {"name": "М Новый клиент", "debt": 100000,
                  "oldest_unpaid_date": "2026-05-10",
                  "days": 6}
             ]):
            data = [{"client": "М Новый клиент", "debt": 100000, "silence_days": 6}]
            self._alerter.apply_residual_debt_age(data)
        h = self._load_history()
        self.assertIn("М Новый клиент", h)
        self.assertEqual(h["М Новый клиент"]["oldest_unpaid_date"], "2026-05-10")

    def test_paid_client_removed_from_history(self):
        # Клиент оплатил (debt=0 в новой выгрузке) — должен быть удалён из истории
        self._alerter._save_debt_age_history({
            "М Оплатил": {"oldest_unpaid_date": "2026-04-01"}
        })
        with patch("collector.debt_monitor.load_latest_debt_json", return_value={}), \
             patch("collector.debt_monitor.classify_debtors", return_value=[
                 {"name": "М Оплатил", "debt": 0,
                  "oldest_unpaid_date": "", "days": 0}
             ]):
            data = [{"client": "М Оплатил", "debt": 0, "silence_days": 0}]
            self._alerter.apply_residual_debt_age(data)
        h = self._load_history()
        self.assertNotIn("М Оплатил", h)

    def test_current_earlier_than_history_updates_history(self):
        # Если current дата раньше сохранённой — обновляем (взяли более раннюю)
        self._alerter._save_debt_age_history({
            "Е Клиент": {"oldest_unpaid_date": "2026-04-20"}
        })
        with patch("collector.debt_monitor.load_latest_debt_json", return_value={}), \
             patch("collector.debt_monitor.classify_debtors", return_value=[
                 {"name": "Е Клиент", "debt": 500000,
                  "oldest_unpaid_date": "2026-04-10", "days": 36}
             ]):
            data = [{"client": "Е Клиент", "debt": 500000}]
            self._alerter.apply_residual_debt_age(data)
        h = self._load_history()
        self.assertEqual(h["Е Клиент"]["oldest_unpaid_date"], "2026-04-10")


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

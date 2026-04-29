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
import collector.collections_engine as collections_engine
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

    def tearDown(self):
        client_dialog._DIALOGS_PATH = self._orig_dialogs_path
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

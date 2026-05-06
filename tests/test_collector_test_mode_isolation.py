#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class CollectorTestModeIsolationTests(unittest.TestCase):
    def test_audit_log_uses_test_file_in_collector_test_mode(self):
        original_test_mode = os.environ.get("COLLECTOR_TEST_MODE")
        original_override = os.environ.get("COLLECTOR_AUDIT_PATH")
        try:
            os.environ["COLLECTOR_TEST_MODE"] = "1"
            os.environ.pop("COLLECTOR_AUDIT_PATH", None)

            import collector.audit_log as audit_log

            audit_log = importlib.reload(audit_log)
            self.assertEqual(audit_log._AUDIT_PATH.name, "collector_audit_test.jsonl")

            with tempfile.TemporaryDirectory() as td:
                test_path = Path(td) / "collector_audit_test.jsonl"
                with patch.object(audit_log, "_AUDIT_PATH", test_path):
                    audit_log.audit("dialog_started", name="Тест Клиент", phone_masked="7701***01")
                    self.assertTrue(test_path.exists())
                    self.assertIn("Тест Клиент", test_path.read_text(encoding="utf-8"))
        finally:
            if original_test_mode is None:
                os.environ.pop("COLLECTOR_TEST_MODE", None)
            else:
                os.environ["COLLECTOR_TEST_MODE"] = original_test_mode
            if original_override is None:
                os.environ.pop("COLLECTOR_AUDIT_PATH", None)
            else:
                os.environ["COLLECTOR_AUDIT_PATH"] = original_override

    def test_client_dialog_suppresses_observer_notifications_in_test_mode(self):
        original_test_mode = os.environ.get("COLLECTOR_TEST_MODE")
        try:
            os.environ["COLLECTOR_TEST_MODE"] = "1"

            import collector.client_dialog as client_dialog

            client_dialog = importlib.reload(client_dialog)
            dialog = {
                "client_name": "Тест Клиент",
                "manager_name": "Ергали",
                "manager_chat_id": 123456,
            }

            with patch("collector.communications.send_telegram", new=AsyncMock()) as send_mock:
                import asyncio

                asyncio.run(client_dialog._notify_dialog_observers(dialog, "test"))
                self.assertEqual(send_mock.await_count, 0)
        finally:
            if original_test_mode is None:
                os.environ.pop("COLLECTOR_TEST_MODE", None)
            else:
                os.environ["COLLECTOR_TEST_MODE"] = original_test_mode


if __name__ == "__main__":
    unittest.main()

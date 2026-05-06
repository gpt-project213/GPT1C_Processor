#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class SendApprovedLockTests(unittest.TestCase):
    def test_recent_send_lock_blocks_parallel_launch(self) -> None:
        import collector.collections_engine as ce_mod

        batch = {
            "batch_id": "batch-lock-1",
            "created_at": "2026-05-06T19:00:00+05:00",
            "send_in_progress": True,
            "send_started_at": "2099-05-06T19:00:00+05:00",
        }
        with patch("collector.approval_flow.load_batch", return_value=batch), \
             patch("collector.approval_flow.save_batch") as _save_batch, \
             patch("collector.approval_flow.is_ready_for_send", return_value=True) as _ready, \
             patch("collector.approval_flow.record_send_results") as _record, \
             patch("collector.collections_engine._live_send_allowed", return_value=True), \
             patch("collector.collections_engine._send_approved_client", new=AsyncMock()) as _send:
            results = asyncio.run(ce_mod.send_approved_batch("batch-lock-1"))

        self.assertEqual(_send.await_count, 0)
        self.assertFalse(_record.called)
        self.assertFalse(_ready.called)
        self.assertEqual(_save_batch.call_count, 0)
        self.assertTrue(any("send already in progress" in str(r.get("reason", "")) for r in results))


if __name__ == "__main__":
    unittest.main()

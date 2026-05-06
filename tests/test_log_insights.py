#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot.log_insights import (
    format_client_timeline,
    format_error_digest,
    read_client_timeline,
    summarize_errors_by_system,
)


class LogInsightsTests(unittest.TestCase):
    def test_summarize_errors_by_system_counts_domains(self):
        with tempfile.TemporaryDirectory() as td:
            logs_dir = Path(td)
            (logs_dir / "send_reports.log").write_text(
                "\n".join(
                    [
                        "2026-04-30 00:30:00, ERROR [CRM][FLOW] claim failed",
                        "2026-04-30 01:00:00, WARNING [COLLECTOR][FLOW] reminder delayed",
                        "2026-04-30 01:30:00, CRITICAL [BOT][CORE] bot down",
                        "2026-04-29 10:00:00, ERROR [CRM][FLOW] too old",
                    ]
                ),
                encoding="utf-8",
            )
            now = datetime(2026, 4, 30, 9, 5, 0)
            counts = summarize_errors_by_system(logs_dir, now=now, hours=12)
            self.assertEqual(counts["CRM"]["ERROR"], 1)
            self.assertEqual(counts["COLLECTOR"]["WARNING"], 1)
            self.assertEqual(counts["BOT"]["CRITICAL"], 1)
            text = format_error_digest(counts, now=now, hours=12)
            self.assertIn("CRM: WARNING 0 | ERROR 1 | CRITICAL 0", text)
            self.assertIn("COLLECTOR: WARNING 1 | ERROR 0 | CRITICAL 0", text)

    def test_read_client_timeline_merges_crm_and_collector(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            crm = root / "crm_audit.jsonl"
            collector = root / "collector_audit.jsonl"
            crm.write_text(
                "\n".join(
                    [
                        '{"ts":"2026-04-30T09:00:00+05:00","event":"claim_broadcast","client_key":"ИП Тест","notified_count":2}',
                        '{"ts":"2026-04-30T09:10:00+05:00","event":"claim_taken","client_key":"ИП Тест","claimer":"Магира"}',
                    ]
                ),
                encoding="utf-8",
            )
            collector.write_text(
                "\n".join(
                    [
                        '{"ts":"2026-04-30T09:20:00+05:00","event":"wa_sent","name":"ИП Тест","amount":150000}',
                        '{"ts":"2026-04-30T09:25:00+05:00","event":"payment_claim_reported","name":"ИП Тест","text":"Передам на оплату"}',
                    ]
                ),
                encoding="utf-8",
            )
            rows = read_client_timeline("ИП Тест", crm_path=crm, collector_path=collector, limit=10)
            self.assertEqual([row["source"] for row in rows], ["CRM", "CRM", "COLLECTOR", "COLLECTOR"])
            text = format_client_timeline("ИП Тест", rows)
            self.assertIn("claim_taken", text)
            self.assertIn("payment_claim_reported", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)

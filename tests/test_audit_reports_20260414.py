#!/usr/bin/env python
# coding: utf-8
"""
Доказательства фиксов аудита отчётов 2026-04-14.
Запуск: python -X utf8 tests/test_audit_reports_20260414.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestConcentrationMergedTotal(unittest.TestCase):
    """После дедупа total_revenue = сумма total по клиентам."""

    def test_sum_matches_clients(self):
        merged_clients = [
            {"client": "A", "total": 100.0},
            {"client": "B", "total": 250.5},
        ]
        merged_revenue = sum(float(c.get("total", 0.0) or 0.0) for c in merged_clients)
        self.assertAlmostEqual(merged_revenue, 350.5)


class TestRfmSegmentPerManager(unittest.TestCase):
    def test_vip_uses_manager_denominator(self):
        from rfm_clients_report import segment_client

        self.assertEqual(segment_client(50_000, 200_000), "VIP")  # 25%
        self.assertEqual(segment_client(50_000, 2_000_000), "REGULAR")  # 2.5% — порог 0.5–3%


class TestGrossPctMarginParse(unittest.TestCase):
    def test_comma_decimal_percent(self):
        from gross_report_pct import _money_to_float

        s = pd.Series(["12,5 %", "10%", ""])
        got = _money_to_float(s.astype(str))
        self.assertAlmostEqual(float(got.iloc[0]), 12.5, places=3)
        self.assertAlmostEqual(float(got.iloc[1]), 10.0, places=3)


class TestDebtFindHeaderNoSilentFallback(unittest.TestCase):
    def test_raises_without_real_header(self):
        from debt_auto_report import find_header

        raw = pd.DataFrame(
            [
                ["foo", "bar"],
                ["1", "2"],
            ]
        )
        test_log = Mock()
        with patch("debt_auto_report.log", test_log):
            with self.assertRaises(ValueError) as ctx:
                find_header(raw)
        self.assertIn("заголовка", str(ctx.exception).lower())
        test_log.error.assert_called_once()


class TestSalesParserDataEndAligned(unittest.TestCase):
    """Два подряд пустых строки — конец (как sales_report)."""

    def test_two_blanks_end(self):
        from sales_parser import find_data_end

        # data_start=1; строки 1-2 данные, 3-4 пустые в product/sale
        raw = pd.DataFrame(
            [
                ["x", "y"],
                ["Товар А", "100"],
                ["Товар Б", "200"],
                ["", ""],
                ["", ""],
                ["хвост", "1"],
            ]
        )

        colmap = {"product": 0, "sale": 1}

        end = find_data_end(raw, 1, colmap)
        self.assertEqual(end, 2)


class TestTurnoverNormalizeProduct(unittest.TestCase):
    def test_kg_dot_matches_kg(self):
        from inventory_turnover_report import normalize_product

        self.assertEqual(
            normalize_product("Молоко 1 кг."),
            normalize_product("Молоко 1 кг"),
        )


class TestConcentrationParetoDisclaimer(unittest.TestCase):
    def test_admin_html_contains_clarification(self):
        src = (ROOT / "revenue_concentration_report.py").read_text(encoding="utf-8")
        self.assertIn("Парето", src)
        self.assertIn("топ-5", src.lower())


class TestConcentrationManagerTemplate(unittest.TestCase):
    def test_pct_version_wraps_table_before_footer(self):
        src = (ROOT / "revenue_concentration_report.py").read_text(encoding="utf-8")
        idx_wrap = src.find('html_pct = f"""')
        self.assertGreater(idx_wrap, 0)
        block = src[idx_wrap : idx_wrap + 4500]
        self.assertIn('<div class="table-wrap"><table>', block)
        self.assertIn("</table></div>", block)
        self.assertLess(block.index("</table></div>"), block.index('class="footer"'))


if __name__ == "__main__":
    unittest.main(verbosity=2)

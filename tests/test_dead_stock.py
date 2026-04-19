#!/usr/bin/env python
# coding: utf-8
"""
tests/test_dead_stock.py - unit tests for dead_stock_report.py.
Run: python -X utf8 tests/test_dead_stock.py
"""
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dead_stock_report import (  # noqa: E402
    build_dead_stock_rows,
    build_report_data,
    normalize_product,
    render_html,
)


def _inventory(*items):
    return {"products": list(items)}


def test_never_sold_included():
    rows = build_dead_stock_rows(
        _inventory({"product": "Never Sold", "qty": 3, "total_cost": 9000}),
        {},
        as_of=date(2026, 4, 15),
    )
    assert len(rows) == 1
    assert rows[0]["last_sale_date"] is None
    assert rows[0]["days_since_sale"] is None


def test_within_threshold_excluded():
    rows = build_dead_stock_rows(
        _inventory({"product": "Fresh", "qty": 1, "total_cost": 1000}),
        {normalize_product("Fresh"): date(2026, 4, 5)},
        as_of=date(2026, 4, 15),
    )
    assert rows == []


def test_over_threshold_included():
    rows = build_dead_stock_rows(
        _inventory({"product": "Old", "qty": 2, "total_cost": 2000}),
        {normalize_product("Old"): date(2026, 3, 21)},
        as_of=date(2026, 4, 15),
    )
    assert len(rows) == 1
    assert rows[0]["days_since_sale"] == 25


def test_zero_stock_excluded():
    rows = build_dead_stock_rows(
        _inventory({"product": "No Stock", "qty": 0, "total_cost": 5000}),
        {},
        as_of=date(2026, 4, 15),
    )
    assert rows == []


def test_empty_result_no_crash():
    report = build_report_data(
        _inventory({"product": "Fresh", "qty": 1, "total_cost": 1000}),
        {normalize_product("Fresh"): date(2026, 4, 10)},
        as_of=date(2026, 4, 15),
    )
    html = render_html(report)
    assert "<html" in html
    assert "Нет мертвого запаса" in html


def test_normalize_join():
    assert normalize_product("Молоко 1 кг.") == normalize_product("Молоко 1 кг")


def _run():
    tests = [
        test_never_sold_included,
        test_within_threshold_excluded,
        test_over_threshold_included,
        test_zero_stock_excluded,
        test_empty_result_no_crash,
        test_normalize_join,
    ]
    failed = []
    for test in tests:
        try:
            test()
            print(f"  OK {test.__name__}")
        except Exception as exc:
            failed.append((test.__name__, exc))
            print(f"  FAIL {test.__name__}: {exc}")
    print(f"\n  RESULT: {len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _run()

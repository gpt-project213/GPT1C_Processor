#!/usr/bin/env python
# coding: utf-8
"""
dead_stock_report.py - dead stock by last sale date.

Input:
- reports/json/inventory_*.json or inventory_cost_*.json
- reports/json/sales_*.json with period_end from sales_parser.py

Output:
- reports/analytics/dead_stock_<YYYYMMDD>.html
"""
from __future__ import annotations

import html
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from inventory_turnover_report import normalize_product

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", encoding="utf-8-sig", override=True)

ROOT = Path(__file__).resolve().parent
JSON_DIR = ROOT / "reports" / "json"
ANALYTICS_DIR = ROOT / "reports" / "analytics"
LOGS = ROOT / "logs"
TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))

ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("dead_stock")

__VERSION__ = "1.0.0"
DEFAULT_THRESHOLD_DAYS = 20
NBSP = "\u202f"


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except (FileNotFoundError, OSError):
        return 0.0


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning("dead_stock: cannot read %s: %s", path.name, exc)
        return None


def load_latest_inventory(json_dir: Path = JSON_DIR) -> Optional[Dict[str, Any]]:
    for pattern in ("inventory_cost_*.json", "inventory_*.json"):
        files = sorted(json_dir.glob(pattern), key=_mtime, reverse=True)
        for path in files:
            data = _read_json(path)
            if isinstance(data, dict):
                LOG.info("dead_stock: inventory source %s", path.name)
                return data
    LOG.error("dead_stock: inventory JSON not found")
    return None


def flatten_inventory_products(inventory_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    products: List[Dict[str, Any]] = []
    if isinstance(inventory_data.get("categories"), list):
        for category in inventory_data.get("categories", []):
            if not isinstance(category, dict):
                continue
            for item in category.get("item_list", []) or []:
                if isinstance(item, dict):
                    products.append(item)
    for item in inventory_data.get("products", []) or []:
        if isinstance(item, dict):
            products.append(item)
    return products


def _iter_sales_products(sales_data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for client in sales_data.get("clients", []) or []:
        if not isinstance(client, dict):
            continue
        for key in ("products", "items"):
            for product in client.get(key, []) or []:
                if isinstance(product, dict):
                    yield product


def build_sales_history(json_dir: Path = JSON_DIR) -> Dict[str, date]:
    history: Dict[str, date] = {}
    for path in sorted(json_dir.glob("sales_*.json"), key=_mtime, reverse=True):
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        period_end = data.get("period_end")
        if not period_end:
            LOG.warning("dead_stock: skip %s without period_end", path.name)
            continue
        try:
            period_date = date.fromisoformat(str(period_end))
        except ValueError:
            LOG.warning("dead_stock: skip %s with bad period_end=%r", path.name, period_end)
            continue
        for product in _iter_sales_products(data):
            key = normalize_product(str(product.get("product") or ""))
            if not key:
                continue
            if period_date > history.get(key, date.min):
                history[key] = period_date
    LOG.info("dead_stock: sales history products=%d", len(history))
    return history


def _to_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _stock_cost(product: Dict[str, Any], qty: float) -> float:
    total_cost = _to_float(product.get("total_cost"))
    if total_cost:
        return total_cost
    cost = _to_float(product.get("cost"))
    if cost:
        return cost
    return _to_float(product.get("unit_cost")) * qty


def build_dead_stock_rows(
    inventory_data: Dict[str, Any],
    sales_history: Dict[str, date],
    *,
    as_of: Optional[date] = None,
    threshold_days: int = DEFAULT_THRESHOLD_DAYS,
) -> List[Dict[str, Any]]:
    today = as_of or datetime.now(TZ).date()
    rows: List[Dict[str, Any]] = []
    for product in flatten_inventory_products(inventory_data):
        name = str(product.get("product") or "").strip()
        if not name:
            continue
        qty = _to_float(product.get("qty", product.get("quantity")))
        if qty <= 0:
            continue
        key = normalize_product(name)
        last_sale = sales_history.get(key)
        days_since_sale: Optional[int] = None
        if last_sale is not None:
            days_since_sale = max(0, (today - last_sale).days)
            if days_since_sale < threshold_days:
                continue
        rows.append({
            "product": name,
            "qty": qty,
            "cost": _stock_cost(product, qty),
            "last_sale_date": last_sale.isoformat() if last_sale else None,
            "days_since_sale": days_since_sale,
        })
    rows.sort(key=lambda item: (
        item["days_since_sale"] is None,
        item["days_since_sale"] or 0,
        item["cost"],
    ), reverse=True)
    return rows


def build_report_data(
    inventory_data: Dict[str, Any],
    sales_history: Dict[str, date],
    *,
    as_of: Optional[date] = None,
    threshold_days: int = DEFAULT_THRESHOLD_DAYS,
) -> Dict[str, Any]:
    rows = build_dead_stock_rows(
        inventory_data,
        sales_history,
        as_of=as_of,
        threshold_days=threshold_days,
    )
    return {
        "as_of": (as_of or datetime.now(TZ).date()).isoformat(),
        "threshold_days": threshold_days,
        "rows": rows,
        "total_cost": sum(_to_float(row.get("cost")) for row in rows),
        "count": len(rows),
    }


def fmt_money(value: Any) -> str:
    return f"{_to_float(value):,.0f}".replace(",", NBSP) + " ₸"


def _age_class(row: Dict[str, Any]) -> str:
    days = row.get("days_since_sale")
    if days is None or days >= 60:
        return "danger"
    if days >= 20:
        return "warning"
    return ""


def render_html(report: Dict[str, Any]) -> str:
    row_html = ""
    for index, row in enumerate(report.get("rows", []), 1):
        days = row.get("days_since_sale")
        days_label = "Никогда" if days is None else str(days)
        last_sale = row.get("last_sale_date") or "Никогда"
        row_html += (
            f'<tr class="{_age_class(row)}">'
            f"<td>{index}</td>"
            f"<td>{html.escape(str(row.get('product') or ''))}</td>"
            f"<td class=\"num\">{_to_float(row.get('qty')):.1f}</td>"
            f"<td class=\"num\">{fmt_money(row.get('cost'))}</td>"
            f"<td>{html.escape(str(last_sale))}</td>"
            f"<td class=\"num\">{html.escape(days_label)}</td>"
            "</tr>\n"
        )
    if not row_html:
        row_html = '<tr><td colspan="6" class="empty">Нет мертвого запаса</td></tr>'

    generated_at = datetime.now(TZ).strftime("%d.%m.%Y %H:%M")
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Мертвый запас по последней продаже</title>
<style>
body{{font-family:Arial,sans-serif;margin:0;padding:18px;background:#f4f6f8;color:#1f2933}}
.wrap{{max-width:1180px;margin:0 auto}}
h1{{font-size:24px;margin:0 0 8px}}
.meta{{color:#52606d;margin-bottom:18px}}
.summary{{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}}
.metric{{background:#fff;border:1px solid #d9e2ec;border-radius:8px;padding:12px 16px}}
.metric b{{display:block;font-size:22px;margin-top:4px}}
.table-wrap{{overflow:auto;background:#fff;border:1px solid #d9e2ec;border-radius:8px}}
table{{width:100%;border-collapse:collapse}}
th,td{{padding:9px 10px;border-bottom:1px solid #e4e7eb;text-align:left}}
th{{background:#edf2f7;font-weight:700}}
.num{{text-align:right;white-space:nowrap}}
tr.warning td{{background:#fff8db}}
tr.danger td{{background:#ffe7e7}}
.empty{{text-align:center;color:#52606d;padding:20px}}
.footer{{margin-top:16px;color:#52606d;font-size:12px}}
</style>
</head>
<body>
<div class="wrap">
<h1>Мертвый запас по последней продаже</h1>
<div class="meta">Дата расчета: {html.escape(str(report.get("as_of") or ""))} | Сгенерировано: {generated_at}</div>
<div class="summary">
<div class="metric">Позиций<b>{int(report.get("count") or 0)}</b></div>
<div class="metric">Заморожено<b>{fmt_money(report.get("total_cost"))}</b></div>
<div class="metric">Порог<b>{int(report.get("threshold_days") or 0)} дней</b></div>
</div>
<div class="table-wrap">
<table>
<thead>
<tr>
<th>#</th>
<th>Товар</th>
<th class="num">Остаток</th>
<th class="num">Стоимость</th>
<th>Последняя продажа</th>
<th class="num">Дней</th>
</tr>
</thead>
<tbody>
{row_html}</tbody>
</table>
</div>
<div class="footer">dead_stock_report.py v{__VERSION__}</div>
</div>
</body>
</html>"""


def generate_report(
    *,
    json_dir: Path = JSON_DIR,
    analytics_dir: Path = ANALYTICS_DIR,
    as_of: Optional[date] = None,
    threshold_days: int = DEFAULT_THRESHOLD_DAYS,
) -> Optional[Path]:
    inventory_data = load_latest_inventory(json_dir)
    if not inventory_data:
        return None
    sales_history = build_sales_history(json_dir)
    report = build_report_data(
        inventory_data,
        sales_history,
        as_of=as_of,
        threshold_days=threshold_days,
    )
    analytics_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(TZ).strftime("%Y%m%d")
    out_path = analytics_dir / f"dead_stock_{ts}.html"
    out_path.write_text(render_html(report), encoding="utf-8")
    LOG.info("dead_stock: saved %s rows=%d", out_path.name, report["count"])
    return out_path


if __name__ == "__main__":
    try:
        result = generate_report()
        raise SystemExit(0 if result else 1)
    except Exception as exc:
        LOG.exception("dead_stock: failed: %s", exc)
        raise SystemExit(1)

#!/usr/bin/env python
# coding: utf-8
"""
inventory_turnover_report.py · v1.1.4 (2026-03-10)
────────────────────────────────────────────────────────────────────
Отчёт "Мертвый запас + Оборачиваемость"

Источники:
- reports/json/inventory_*.json  (остатки на складе)
- reports/json/sales_*.json      (продажи)

Выход:
- reports/analytics/turnover_<date>.html

Показывает:
- Товары без продаж >30 дней (мертвый запас)
- Сколько денег заморожено
- Оборачиваемость товаров

Доступ: ТОЛЬКО Admin

v1.1.4: Fix P-008: добавлен load_dotenv() — TZ теперь читается из .env
v1.1.3: TZ timezone(timedelta(hours=5)) → ZoneInfo("Asia/Almaty") (Bug TZ)
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", encoding="utf-8-sig", override=True)

TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))
ROOT = Path(__file__).resolve().parent
JSON_DIR = ROOT / "reports" / "json"
ANALYTICS_DIR = ROOT / "reports" / "analytics"
LOGS = ROOT / "logs"
ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("turnover")

__VERSION__ = "1.1.4"
NBSP = "\u202f"


def load_latest_json(pattern: str, skip_keywords: list = None) -> Optional[Dict[str, Any]]:
    files = sorted(JSON_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        LOG.error(f"Не найдены файлы {pattern}")
        return None
    for path in files:
        if skip_keywords and any(kw.lower() in path.name.lower() for kw in skip_keywords):
            LOG.warning(f"Пропускаю: {path.name}")
            continue
        LOG.info(f"Загружаю: {path.name}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            LOG.warning(f"Ошибка чтения {path.name}: {e}")
    return None


def fmt_money(x: float) -> str:
    return f"{float(x):,.0f}".replace(",", NBSP) + " ₸"


def normalize_product(name: str) -> str:
    """Нормализация названия товара для сравнения"""
    s = name.lower().strip()
    s = s.replace("/", " ").replace("-", " ")
    s = s.replace("(", " ").replace(")", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def generate_report() -> None:
    LOG.info("=" * 60)
    LOG.info("ГЕНЕРАЦИЯ ОТЧЁТА: Мертвый запас + Оборачиваемость")

    # Загрузка
    inventory_data = load_latest_json("inventory_*.json")
    sales_data = load_latest_json("sales_*.json", skip_keywords=["товару"])

    if not inventory_data or not sales_data:
        LOG.error("Недостаточно данных")
        return

    # Создать справочник продаж
    sales_dict = {}
    for client in sales_data.get("clients", []):
        for product in client.get("products", []):
            prod_name = normalize_product(product["product"])
            sales_dict[prod_name] = sales_dict.get(prod_name, 0.0) + product["sum"]

    # Собрать плоский список товаров из возможных форматов JSON:
    # - inventory.py сохраняет: {"categories": [{"item_list": [{product, qty, cost}, ...]}]}
    # - inventory_cost_parser.py сохраняет: {"products": [{product, qty, total_cost}, ...]}
    all_products = []
    if "categories" in inventory_data:
        for cat in inventory_data.get("categories", []):
            for item in cat.get("item_list", []):
                all_products.append(item)
        LOG.info("Формат: categories/item_list, товаров: %d", len(all_products))
    else:
        all_products = inventory_data.get("products", [])
        LOG.info("Формат: products[], товаров: %d", len(all_products))

    # Анализ остатков
    dead_stock = []
    fast_movers = []
    total_frozen = 0.0

    for product in all_products:
        prod_name = product["product"]
        prod_normalized = normalize_product(prod_name)

        qty = product.get("qty", product.get("quantity", 0.0))
        # inventory.py: поле "cost"; inventory_cost_parser: поле "total_cost"
        total_cost = product.get("total_cost", product.get("cost", 0.0))
        if (not total_cost) and ("unit_cost" in product):
            try:
                total_cost = float(product.get("unit_cost") or 0.0) * float(qty or 0.0)
            except Exception:
                total_cost = total_cost

        has_sales = prod_normalized in sales_dict
        sales_amount = sales_dict.get(prod_normalized, 0.0)

        # Мертвый запас = нет продаж и стоимость > 1000 тг
        if not has_sales and total_cost > 1000:
            dead_stock.append({
                "product": prod_name,
                "qty": qty,
                "cost": total_cost
            })
            total_frozen += total_cost

        # Быстрооборачиваемые (продажи > 2× остаток)
        elif sales_amount > total_cost * 2:
            fast_movers.append({
                "product": prod_name,
                "sales": sales_amount,
                "stock": total_cost
            })

    # Сортировка
    dead_stock.sort(key=lambda x: x["cost"], reverse=True)
    fast_movers.sort(key=lambda x: x["sales"], reverse=True)

    # Построение HTML
    dead_rows = ""
    for i, item in enumerate(dead_stock[:20], 1):
        dead_rows += f"""
        <tr>
            <td>{i}</td>
            <td>{item['product']}</td>
            <td style="text-align:right">{item['qty']:.1f}</td>
            <td style="text-align:right">{fmt_money(item['cost'])}</td>
        </tr>"""

    fast_rows = ""
    for i, item in enumerate(fast_movers[:10], 1):
        ratio = item['sales'] / item['stock'] if item['stock'] > 0 else 0
        fast_rows += f"""
        <tr>
            <td>{i}</td>
            <td>{item['product']}</td>
            <td style="text-align:right">{fmt_money(item['sales'])}</td>
            <td style="text-align:right">{ratio:.1f}×</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Мертвый запас + Оборачиваемость</title>
<style>
body{{font-family:Arial,sans-serif;background:#f5f5f5;margin:20px}}
.container{{max-width:1200px;margin:0 auto;background:#fff;padding:30px;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.1)}}
h1{{color:#2c3e50;margin-bottom:10px}}
.meta{{color:#666;font-size:14px;margin-bottom:30px}}
.alert{{background:#fff3cd;border:1px solid #ffc107;padding:20px;border-radius:8px;margin:20px 0}}
.alert.danger{{background:#f8d7da;border-color:#dc3545}}
.alert-value{{font-size:32px;font-weight:700;color:#dc3545;margin:10px 0}}
table{{width:100%;border-collapse:collapse;margin:20px 0}}
th,td{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}
th{{background:#f8f9fa;font-weight:600}}
tr:hover{{background:#f8f9fa}}
h2{{color:#2c3e50;margin-top:40px;padding-bottom:10px;border-bottom:2px solid #007bff}}
.footer{{margin-top:30px;padding-top:20px;border-top:1px solid #eee;text-align:center;color:#999;font-size:12px}}
</style>
</head>
<body>
<div class="container">
<h1>📦 Мертвый запас + Оборачиваемость</h1>
<div class="meta">Отчёт ТОЛЬКО для Admin | {datetime.now(TZ).strftime("%d.%m.%Y %H:%M")}</div>

<div class="alert danger">
<div style="font-weight:600;font-size:18px">⚠️ ЗАМОРОЖЕНО КАПИТАЛА</div>
<div class="alert-value">{fmt_money(total_frozen)}</div>
<div style="color:#666">В товарах без продаж (мертвый запас)</div>
</div>

<h2>🚫 Мертвый запас (топ-20)</h2>
<p style="color:#666">Товары БЕЗ продаж в последнем отчёте:</p>
<table>
<thead>
<tr>
<th style="width:40px">#</th>
<th>Товар</th>
<th style="width:100px;text-align:right">Остаток</th>
<th style="width:150px;text-align:right">Заморожено</th>
</tr>
</thead>
<tbody>
{dead_rows if dead_rows else '<tr><td colspan="4" style="text-align:center;color:#666">Нет мертвого запаса ✅</td></tr>'}
</tbody>
</table>

<h2>🚀 Быстрооборачиваемые (топ-10)</h2>
<p style="color:#666">Товары с высокими продажами относительно остатков:</p>
<table>
<thead>
<tr>
<th style="width:40px">#</th>
<th>Товар</th>
<th style="width:150px;text-align:right">Продажи</th>
<th style="width:120px;text-align:right">Оборот</th>
</tr>
</thead>
<tbody>
{fast_rows if fast_rows else '<tr><td colspan="4" style="text-align:center;color:#666">Нет данных</td></tr>'}
</tbody>
</table>

<div class="footer">
inventory_turnover_report.py v{__VERSION__} | {datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")}
</div>
</div>
</body>
</html>"""

    ts = datetime.now(TZ).strftime("%Y%m%d")
    html_path = ANALYTICS_DIR / f"turnover_{ts}.html"
    html_path.write_text(html, encoding="utf-8")
    LOG.info(f"✅ Сохранено: {html_path.name}")
    LOG.info(f"Мертвый запас: {len(dead_stock)} товаров, {fmt_money(total_frozen)}")


if __name__ == "__main__":
    try:
        generate_report()
    except Exception as e:
        LOG.error(f"Ошибка: {e}", exc_info=True)
        exit(1)
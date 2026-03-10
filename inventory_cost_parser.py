#!/usr/bin/env python
# coding: utf-8
"""
inventory_cost_parser.py — Парсер отчёта "Ведомость по партиям товаров на складах" (с себестоимостью)
Версия 1.6.6 (2026-03-10) — TZ timezone(timedelta(hours=5)) → ZoneInfo("Asia/Almaty") (Bug TZ)
Версия 1.6.5 — исправлен пропуск строки "Оптовый" по колонке товара.
"""

from __future__ import annotations

import json
import math
import re
import logging
import argparse
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Any, Optional

import pandas as pd

try:
    from utils_excel import ensure_clean_xlsx
except ImportError:
    ensure_clean_xlsx = None

TZ = ZoneInfo("Asia/Almaty")
ROOT = Path(__file__).resolve().parent
JSON_OUT = ROOT / "reports" / "json"
HTML_OUT = ROOT / "reports" / "html"
LOGS = ROOT / "logs"
JSON_OUT.mkdir(parents=True, exist_ok=True)
HTML_OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
LOG = logging.getLogger("inventory_cost_parser")

__VERSION__ = "1.6.6"
NBSP = "\u202f"

# Регулярные выражения
TOTAL_RE = re.compile(r"\b(итог(?:о)?|всего|итоги|общий итог|total)\b", re.I)
PERIOD_RE = re.compile(r"период[:\s]*([^\n|;]+)", re.I)

# Фиксированные индексы колонок (на основе анализа файла)
COL_PRODUCT = 1      # B - наименование товара
COL_QTY_BASE = 4     # E - количество в базовых единицах
COL_UNIT_COST = 5    # F - себестоимость единицы
COL_TOTAL_COST = 6   # G - общая стоимость

def clean(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x).replace("\xa0", " ").replace(NBSP, " ").strip()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    if s.lower() in {"nan", "nat", "none", "null"}:
        return ""
    if s in {"-", "–", "—"}:
        return ""
    return s

def to_float(x: Any) -> float:
    s = clean(x)
    if not s:
        return float("nan")
    s = s.replace(",", ".")
    neg = s.startswith("-") or s.endswith("-")
    s = s.strip("+-")
    s = re.sub(r"[^\d.]", "", s)
    if not s:
        return float("nan")
    try:
        v = float(s)
        return -v if neg else v
    except Exception:
        return float("nan")

def slugify(text: str) -> str:
    s = clean(text).lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s)
    return s[:50]

def _ensure_file_logging():
    ts = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
    p = LOGS / f"inventory_cost_parser_{ts}.log"
    fh = logging.FileHandler(p, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s, %(levelname)s %(message)s"))
    fh.setLevel(logging.INFO)
    LOG.addHandler(fh)
    LOG.info("Лог-файл: %s", p)

def read_excel_raw(path: Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {p}")

    if ensure_clean_xlsx and not p.name.endswith(".__clean.xlsx"):
        p = ensure_clean_xlsx(p, force_fix=True)
        LOG.info("Создан очищенный файл: %s", p.name)
    else:
        LOG.info("Используем исходный (уже очищенный): %s", p.name)

    if not p.exists():
        raise FileNotFoundError(f"Файл после очистки не найден: {p}")

    return pd.read_excel(p, header=None, dtype=str, engine="openpyxl")

def find_header_row(df: pd.DataFrame, max_scan: int = 40) -> int:
    keywords_product = ["номенклатура", "товар"]
    keywords_qty_base = ["количество (в базовых ед.)", "количество в базовых единицах"]

    for i in range(min(max_scan, len(df))):
        row = df.iloc[i].astype(str).str.lower()
        row_str = " ".join(row)
        if any(pat in row_str for pat in ["период", "показатели", "группировки", "отборы"]):
            continue
        has_product = any(any(kw in cell for kw in keywords_product) for cell in row)
        has_qty_base = any(any(kw in cell for kw in keywords_qty_base) for cell in row)
        if has_product and has_qty_base:
            LOG.info(f"Найдена строка заголовков на индексе {i}")
            return i
    raise ValueError("Не удалось найти строку заголовков таблицы")

def find_first_data_row(df: pd.DataFrame, header_row: int) -> int:
    start = header_row + 1
    # Пропускаем строку с единицами измерения
    if start < len(df):
        row_str = " ".join(clean(x) for x in df.iloc[start].tolist())
        if "ед." in row_str.lower():
            LOG.info("Пропускаем строку с единицами измерения")
            start += 1
    # Пропускаем строку "Оптовый" (итоги по складу) – проверяем колонку товара (B)
    while start < len(df):
        if COL_PRODUCT < df.shape[1] and clean(df.iloc[start, COL_PRODUCT]).lower() == "оптовый":
            LOG.info("Пропускаем строку с итогами по складу (Оптовый)")
            start += 1
        else:
            break
    return start

def find_last_data_row(df: pd.DataFrame, data_start: int) -> int:
    last = data_start - 1
    empty_seq = 0
    for i in range(data_start, len(df)):
        row = df.iloc[i]
        row_str = " ".join(clean(x) for x in row.tolist())
        if TOTAL_RE.search(row_str):
            LOG.info(f"Найдена строка итогов на индексе {i}")
            break
        if COL_PRODUCT < len(row) and clean(row[COL_PRODUCT]):
            empty_seq = 0
            last = i
        else:
            empty_seq += 1
            if empty_seq >= 3:
                break
    return last

def find_total_row(df: pd.DataFrame, data_start: int, data_end: int) -> Optional[int]:
    for i in range(data_end + 1, min(data_end + 10, len(df))):
        row_str = " ".join(clean(x) for x in df.iloc[i].tolist())
        if TOTAL_RE.search(row_str):
            return i
    return None

def parse_inventory_with_cost(df: pd.DataFrame, data_start: int, data_end: int) -> Dict[str, Any]:
    raw_products = []

    for i in range(data_start, data_end + 1):
        row = df.iloc[i].tolist()

        # Проверяем, не является ли строка итоговой по складу (колонка B)
        if COL_PRODUCT < len(row) and clean(row[COL_PRODUCT]).lower() == "оптовый":
            continue

        name = clean(row[COL_PRODUCT]) if COL_PRODUCT < len(row) else ""
        if not name or TOTAL_RE.search(name):
            continue

        raw_qty = row[COL_QTY_BASE] if COL_QTY_BASE < len(row) else ""
        qty = to_float(raw_qty)
        unit_cost = to_float(row[COL_UNIT_COST]) if COL_UNIT_COST < len(row) else 0.0
        cost = to_float(row[COL_TOTAL_COST]) if COL_TOTAL_COST < len(row) else 0.0

        # Пропускаем, если нет базового количества (пусто)
        if not raw_qty or str(raw_qty).strip() == "":
            continue

        if math.isnan(qty):
            qty = 0.0
        if math.isnan(unit_cost):
            unit_cost = 0.0
        if math.isnan(cost):
            cost = 0.0

        # Пропускаем полностью нулевые позиции
        if math.isclose(qty, 0.0) and math.isclose(cost, 0.0):
            continue

        raw_products.append({
            "name": name,
            "qty": qty,
            "unit_cost": unit_cost,
            "cost": cost
        })

    LOG.info(f"Всего собрано строк: {len(raw_products)}")

    # Агрегируем по (name, unit_cost), суммируя количество и стоимость
    aggregated = {}
    for p in raw_products:
        key = (p["name"], p["unit_cost"])
        if key not in aggregated:
            aggregated[key] = {"qty": 0.0, "cost": 0.0}
        aggregated[key]["qty"] += p["qty"]
        aggregated[key]["cost"] += p["cost"]

    # Формируем итоговый список товаров
    products = []
    total_qty = 0.0
    total_cost = 0.0
    for (name, unit_cost), values in aggregated.items():
        product = {
            "product": name,
            "qty": values["qty"],
            "unit_cost": unit_cost,
            "total_cost": values["cost"]
        }
        products.append(product)
        total_qty += values["qty"]
        total_cost += values["cost"]

    products.sort(key=lambda x: (-x["total_cost"], x["product"]))

    return {
        "total_qty": total_qty,
        "total_cost": total_cost,
        "products": products
    }

def validate_totals(df: pd.DataFrame, total_row: int, parsed_qty: float, parsed_cost: float) -> None:
    if total_row is None:
        LOG.warning("Не найдена строка с итогами в Excel")
        return

    row = df.iloc[total_row].tolist()
    excel_qty = to_float(row[COL_QTY_BASE]) if COL_QTY_BASE < len(row) else 0.0
    excel_cost = to_float(row[COL_TOTAL_COST]) if COL_TOTAL_COST < len(row) else 0.0

    if math.isnan(excel_qty):
        excel_qty = 0.0
    if math.isnan(excel_cost):
        excel_cost = 0.0

    if not math.isclose(parsed_qty, excel_qty, rel_tol=1e-5, abs_tol=0.01):
        LOG.warning(f"Расхождение в количестве: рассчитано {parsed_qty:.3f}, в Excel {excel_qty:.3f}")
    else:
        LOG.info(f"Количество совпадает: {parsed_qty:.3f}")

    if not math.isclose(parsed_cost, excel_cost, rel_tol=1e-5, abs_tol=0.01):
        LOG.warning(f"Расхождение в стоимости: рассчитано {parsed_cost:.2f}, в Excel {excel_cost:.2f}")
    else:
        LOG.info(f"Стоимость совпадает: {parsed_cost:.2f}")

def extract_period(df: pd.DataFrame, up_to_row: int) -> str:
    for i in range(min(20, up_to_row)):
        line = " | ".join(clean(x) for x in df.iloc[i].tolist() if clean(x))
        m = PERIOD_RE.search(line)
        if m:
            return re.split(r"[|;]", m.group(1).strip())[0].strip()
    return "Не указан"

def extract_warehouse() -> str:
    return "Оптовый"

def save_json(data: Dict[str, Any], slug: str) -> Path:
    json_path = JSON_OUT / f"inventory_cost_{slug}.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("JSON сохранён: %s", json_path)
    return json_path

def save_html(data: Dict[str, Any], slug: str) -> Path:
    html_path = HTML_OUT / f"inventory_cost_{slug}.html"

    def fmt_money(v: float) -> str:
        return f"{v:,.0f}".replace(",", NBSP) + " ₸"

    rows_html = ""
    for i, p in enumerate(data["products"], 1):
        rows_html += f"""
        <tr>
            <td>{i}</td>
            <td>{p['product']}</td>
            <td style="text-align:right">{p['qty']:.3f}</td>
            <td style="text-align:right">{fmt_money(p['unit_cost'])}</td>
            <td style="text-align:right">{fmt_money(p['total_cost'])}</td>
        </tr>"""

    html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ведомость по партиям — {slug}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: #f5f5f5;
            margin: 20px;
            color: #111;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: #fff;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .meta {{
            color: #666;
            font-size: 14px;
            margin-bottom: 30px;
        }}
        .kpi {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-bottom: 30px;
        }}
        .card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        .card-label {{
            font-size: 13px;
            color: #666;
            text-transform: uppercase;
        }}
        .card-value {{
            font-size: 24px;
            font-weight: 700;
            margin-top: 5px;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            padding: 10px;
            border-bottom: 1px solid #eee;
            text-align: left;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            text-align: center;
            color: #999;
            font-size: 12px;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>📦 Ведомость по партиям товаров (с себестоимостью)</h1>
    <div class="meta">
        <div><b>Склад:</b> {data['warehouse']}</div>
        <div><b>Период:</b> {data['period']}</div>
        <div><b>Источник:</b> {data['source_file']}</div>
    </div>

    <div class="kpi">
        <div class="card">
            <div class="card-label">Всего позиций</div>
            <div class="card-value">{len(data['products'])}</div>
        </div>
        <div class="card">
            <div class="card-label">Общее количество (базовые ед.)</div>
            <div class="card-value">{data['total_qty']:.3f}</div>
        </div>
        <div class="card">
            <div class="card-label">Общая стоимость</div>
            <div class="card-value">{fmt_money(data['total_cost'])}</div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th style="width:40px">#</th>
                <th>Товар</th>
                <th style="width:100px; text-align:right">Количество (баз.)</th>
                <th style="width:130px; text-align:right">Себест. ед.</th>
                <th style="width:130px; text-align:right">Общая стоимость</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>

    <div class="footer">
        Сформировано: {datetime.now(TZ).strftime("%d.%m.%Y %H:%M")} | inventory_cost_parser.py v{__VERSION__}
    </div>
</div>
</body>
</html>"""

    html_path.write_text(html_content, encoding="utf-8")
    LOG.info("HTML сохранён: %s", html_path)
    return html_path

def build_inventory_cost_report(xlsx: Path) -> Dict[str, Path]:
    raw = read_excel_raw(xlsx)
    header_row = find_header_row(raw)
    LOG.info(f"Заголовок таблицы на строке {header_row}")

    period = extract_period(raw, header_row)
    data_start = find_first_data_row(raw, header_row)
    data_end = find_last_data_row(raw, data_start)
    LOG.info(f"Диапазон данных: строки {data_start}–{data_end}")

    total_row = find_total_row(raw, data_start, data_end)

    data = parse_inventory_with_cost(raw, data_start, data_end)
    data["source_file"] = xlsx.name
    data["period"] = period
    data["warehouse"] = extract_warehouse()

    validate_totals(raw, total_row, data["total_qty"], data["total_cost"])

    slug = slugify(f"{xlsx.stem}_{period}")
    json_path = save_json(data, slug)
    html_path = save_html(data, slug)
    return {"json": json_path, "html": html_path}

def main(argv: Optional[List[str]] = None) -> int:
    _ensure_file_logging()
    ap = argparse.ArgumentParser(description="Парсер остатков с себестоимостью → JSON и HTML")
    ap.add_argument("xlsx", help="Excel-файл ведомости по партиям")
    args = ap.parse_args(argv)

    try:
        p = Path(args.xlsx)
        if not p.exists():
            LOG.error("Файл не существует: %s", p)
            return 1
        LOG.info("Файл: %s", p)
        paths = build_inventory_cost_report(p)
        LOG.info("✅ JSON: %s", paths["json"])
        LOG.info("✅ HTML: %s", paths["html"])
        return 0
    except Exception as e:
        LOG.error("❌ Ошибка", exc_info=e)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
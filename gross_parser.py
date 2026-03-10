#!/usr/bin/env python
# coding: utf-8
"""
gross_parser.py · v1.0.2 (2026-03-10)
────────────────────────────────────────────────────────────────────
Парсер отчётов "Валовая прибыль" из 1С в JSON формат.

ИСПРАВЛЕНИЯ v1.0.2:
- TZ timezone(timedelta(hours=5)) → ZoneInfo("Asia/Almaty") (Bug TZ)

ИСПРАВЛЕНИЯ v1.0.1:
- Фильтрация служебной строки "Номенклатура"

Вход:  Excel файл "Валовая прибыль"
Выход: reports/json/gross_<slug>.json

Структура JSON:
{
  "source_file": str,
  "report_type": "GROSS_PROFIT",
  "period": str,
  "total_revenue": float,
  "total_cost": float,
  "gross_profit": float,
  "margin_pct": float,
  "products": [
    {
      "product": str,
      "revenue": float,
      "cost": float,
      "profit": float,
      "margin_pct": float
    }
  ],
  "metadata": {...}
}
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

# ──────────────────────────────────────────────────────────────────
# Настройки
TZ = ZoneInfo("Asia/Almaty")
ROOT = Path(__file__).resolve().parent
JSON_OUT = ROOT / "reports" / "json"
LOGS = ROOT / "logs"
JSON_OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
LOG = logging.getLogger("gross_parser")

__VERSION__ = "1.0.2"

NBSP = "\u202f"

# ──────────────────────────────────────────────────────────────────
# Регулярные выражения
TOTAL_RE = re.compile(r"^(итог|итого|всего|total)\b", re.I)
SERVICE_RE = re.compile(r"(группиров|показател|дополнит)", re.I)
PERIOD_RE = re.compile(r"период[:\s]*([^\n|;]+)", re.I)

# Колонки
COLS = {
    "product": ("номенк", "товар", "наимен", "наименование", "продукт"),
    "sale":    ("стоим", "прод", "выруч", "реализ", "продаж", "выручка", "оборот"),
    "cost":    ("себест", "себестоим", "затрат", "расход"),
    "gp":      ("валов", "прибыл", "доход", "прибыль", "gross profit"),
    "margin":  ("рентаб", "рентабель", "маржа", "марж", "%"),
}

# ──────────────────────────────────────────────────────────────────
# Утилиты
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
    """Создать безопасное имя файла"""
    s = clean(text).lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s)
    return s[:50]

def _ensure_file_logging():
    ts = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
    p = LOGS / f"gross_parser_{ts}.log"
    fh = logging.FileHandler(p, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s, %(levelname)s %(message)s"))
    fh.setLevel(logging.INFO)
    LOG.addHandler(fh)
    LOG.info("Лог-файл: %s", p)

# ──────────────────────────────────────────────────────────────────
# Парсинг Excel
def read_excel_raw(path: Path) -> pd.DataFrame:
    p = Path(path)
    if ensure_clean_xlsx:
        p = ensure_clean_xlsx(p, force_fix=True)
    return pd.read_excel(p, header=None, dtype=str)

def _probe(row) -> Dict[str, int]:
    res = {}
    for j, cell in enumerate(row):
        t = clean(cell).lower()
        for k, alts in COLS.items():
            if k in res:
                continue
            if any(a in t for a in alts):
                res[k] = j
                break
    return res

def find_header_block(df: pd.DataFrame, max_scan: int = 60):
    width = df.shape[1]
    best = None
    score_best = -1
    
    for i in range(min(max_scan, len(df))):
        # Пропустить служебные строки
        line = " ".join(clean(x) for x in df.iloc[i].tolist())
        if SERVICE_RE.search(line):
            continue
        
        base = _probe(df.iloc[i])
        if i + 1 < len(df):
            extra = _probe(df.iloc[i + 1])
            for k, v in extra.items():
                base.setdefault(k, v)
        
        score = 0
        if "product" in base:
            score += 3
        if "sale" in base:
            score += 2
        if "cost" in base:
            score += 2
        
        if score >= 5 and score > score_best:
            head = [clean(df.iat[i, j]) for j in range(width)]
            best = (base, i + 1, head)
            score_best = score
    
    if not best:
        raise ValueError("Не найдена строка заголовков")
    return best

def extract_period(df: pd.DataFrame, data_start: int) -> str:
    for i in range(min(20, data_start)):
        line = " | ".join(clean(x) for x in df.iloc[i].tolist() if clean(x))
        m = PERIOD_RE.search(line)
        if m:
            return re.split(r"[|;]", m.group(1).strip())[0].strip()
    return "Не указан"

def find_data_end(raw: pd.DataFrame, data_start: int, colmap: Dict[str, int]) -> int:
    key_cols = [j for j in (colmap.get("product"), colmap.get("sale")) if j is not None]
    last = data_start - 1
    empty_seq = 0
    
    for i in range(data_start, len(raw)):
        row = raw.iloc[i]
        empty = True
        for j in key_cols:
            if j is None or j >= len(row):
                continue
            if clean(row[j]):
                empty = False
                break
        
        if empty:
            empty_seq += 1
            if empty_seq >= 3:
                return last
        else:
            empty_seq = 0
            last = i
    
    return last

def is_total_line(text: str) -> bool:
    return bool(TOTAL_RE.search(text))

def _money_to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d.,-]", "", regex=True).str.replace(",", "."),
        errors="coerce"
    ).fillna(0.0)

# ──────────────────────────────────────────────────────────────────
# Основной парсер
def parse_gross_profit(df: pd.DataFrame, colmap: Dict[str, int]) -> Dict[str, Any]:
    prod_j = colmap.get("product", -1)
    sale_j = colmap.get("sale", -1)
    cost_j = colmap.get("cost", -1)
    gp_j = colmap.get("gp", -1)
    margin_j = colmap.get("margin", -1)
    
    if prod_j == -1 or sale_j == -1:
        raise ValueError("Не найдены обязательные колонки: product, sale")
    
    # Конвертировать числовые колонки
    if sale_j < len(df.columns):
        df.iloc[:, sale_j] = _money_to_float(df.iloc[:, sale_j])
    
    if cost_j != -1 and cost_j < len(df.columns):
        df.iloc[:, cost_j] = _money_to_float(df.iloc[:, cost_j])
    
    if gp_j != -1 and gp_j < len(df.columns):
        df.iloc[:, gp_j] = _money_to_float(df.iloc[:, gp_j])
    
    products = []
    total_revenue = 0.0
    total_cost = 0.0
    total_profit = 0.0
    
    for _, r in df.iterrows():
        row = r.tolist()
        
        name = clean(row[prod_j]) if prod_j < len(row) else ""
        if not name or is_total_line(name) or name.lower() in ("номенклатура", "контрагент"):
            continue
        
        sale = to_float(row[sale_j]) if sale_j < len(row) else 0.0
        cost = to_float(row[cost_j]) if cost_j != -1 and cost_j < len(row) else 0.0
        
        # Валовая прибыль
        if gp_j != -1 and gp_j < len(row):
            profit = to_float(row[gp_j])
        else:
            profit = sale - cost
        
        # Маржа
        if margin_j != -1 and margin_j < len(row):
            margin_str = clean(row[margin_j])
            margin = to_float(margin_str.replace("%", ""))
        else:
            margin = (profit / sale * 100) if sale > 0 else 0.0
        
        if math.isnan(sale):
            sale = 0.0
        if math.isnan(cost):
            cost = 0.0
        if math.isnan(profit):
            profit = 0.0
        if math.isnan(margin):
            margin = 0.0
        
        product = {
            "product": name,
            "revenue": float(sale),
            "cost": float(cost),
            "profit": float(profit),
            "margin_pct": float(margin)
        }
        products.append(product)
        
        total_revenue += sale
        total_cost += cost
        total_profit += profit
    
    # Сортировка по прибыли (убыв)
    products.sort(key=lambda x: (-x["profit"], x["product"]))
    
    overall_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0.0
    
    return {
        "total_revenue": total_revenue,
        "total_cost": total_cost,
        "gross_profit": total_profit,
        "margin_pct": overall_margin,
        "products": products
    }

# ──────────────────────────────────────────────────────────────────
# Главная функция
def build_gross_json(xlsx: Path) -> Path:
    raw = read_excel_raw(xlsx)
    colmap, data_start, _ = find_header_block(raw)
    period = extract_period(raw, data_start)
    data_end = find_data_end(raw, data_start, colmap)
    df = raw.iloc[data_start:data_end + 1].reset_index(drop=True)
    
    data = parse_gross_profit(df, colmap)
    
    result = {
        "source_file": xlsx.name,
        "report_type": "GROSS_PROFIT",
        "period": period,
        "total_revenue": data["total_revenue"],
        "total_cost": data["total_cost"],
        "gross_profit": data["gross_profit"],
        "margin_pct": data["margin_pct"],
        "product_count": len(data["products"]),
        "products": data["products"],
        "metadata": {
            "version": __VERSION__,
            "parsed_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": "Asia/Qyzylorda"
        }
    }
    
    # Создать имя файла
    slug = slugify(f"{xlsx.stem}_{period}")
    out = JSON_OUT / f"gross_{slug}.json"
    
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    LOG.info("JSON создан: %s", out)
    LOG.info("Товаров: %d, Выручка: %.2f ₸, Прибыль: %.2f ₸, Маржа: %.2f%%",
             len(data["products"]), data["total_revenue"], 
             data["gross_profit"], data["margin_pct"])
    
    return out

# ──────────────────────────────────────────────────────────────────
# CLI
def main(argv: Optional[List[str]] = None) -> int:
    _ensure_file_logging()
    ap = argparse.ArgumentParser(description="Парсер валовой прибыли → JSON")
    ap.add_argument("xlsx", help="Excel-файл валовой прибыли")
    args = ap.parse_args(argv)
    
    try:
        p = Path(args.xlsx)
        LOG.info("Файл: %s", p)
        out = build_gross_json(p)
        LOG.info("OK: %s", out)
        return 0
    except Exception as e:
        LOG.error("FAILED", exc_info=e)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
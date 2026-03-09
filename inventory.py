#!/usr/bin/env python
# coding: utf-8
"""
inventory.py v1.1.2 (2026-03-09) — Остатки товаров на складах (с группировкой по категориям)
Генерирует HTML и JSON отчёты.
Сортировка: категории по убыванию общего количества, товары по убыванию количества.
"""

from __future__ import annotations
import math
import re
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

try:
    from utils_excel import ensure_clean_xlsx
except ImportError:
    ensure_clean_xlsx = None

TZ = timezone(timedelta(hours=5))
ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / "templates"
OUT_HTML = ROOT / "reports" / "html"
OUT_JSON = ROOT / "reports" / "json"
LOGS = ROOT / "logs"

for d in (TEMPLATES, OUT_HTML, OUT_JSON, LOGS):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="(%(asctime)s) %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
LOG = logging.getLogger("inventory")

NBSP = "\u202f"

# Основные категории (верхнего уровня)
MAIN_CATEGORIES = {
    "утка", "ряба", "укпф", "айсер", "ардагер", "рыба", "китай",
    "продукция кз", "продукция россия", "полуфабрикаты",
    "ягоды, овощи", "ягоды и овощи", "морепродукты",
    "бразилия", "говядина", "без категории"
}

COLS = {
    "warehouse": ("склад",),
    "product":   ("товар", "номенклатура", "наимен"),
    "qty_end":   ("количество", "остаток", "конечный остаток", "кон. остаток"),
    "cost":      ("стоимость",),
    "unit_cost": ("себестоимость", "ед.товара", "единица"),
}
TOTAL_RE  = re.compile(r"\b(итог(?:о)?|всего|итоги|общий итог|total)\b", re.I)
PERIOD_RE = re.compile(r"период[:\s]*([^\n|;]+)", re.I)

def _ensure_file_logging():
    ts = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
    p = LOGS / f"inventory_{ts}.log"
    fh = logging.FileHandler(p, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s, %(levelname)s %(message)s"))
    fh.setLevel(logging.INFO)
    LOG.addHandler(fh)
    LOG.info("Лог-файл: %s", p)

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

def fmt_qty(v: float | int) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "0"
    f = float(v)
    s = f"{f:.3f}".rstrip("0").rstrip(".")
    if "." in s:
        i, d = s.split(".", 1)
    else:
        i, d = s, ""
    try:
        i_fmt = f"{int(i):,}".replace(",", NBSP)
        return f"{i_fmt}.{d}" if d else i_fmt
    except Exception:
        return s

def fmt_int(v: float | int) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        v = 0
    n = int(round(float(v)))
    s = f"{abs(n):,}".replace(",", NBSP)
    return f"-{s}" if n < 0 else s

def read_excel_raw(path: Path) -> pd.DataFrame:
    p = Path(path)
    if ensure_clean_xlsx and not p.name.endswith(".__clean.xlsx"):
        p = ensure_clean_xlsx(p, force_fix=True)
        LOG.info("Очищенный файл: %s", p.name)
    else:
        LOG.info("Используем исходный (уже очищенный): %s", p.name)
    return pd.read_excel(p, header=None, dtype=str, engine="openpyxl")

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
        base = _probe(df.iloc[i])
        if i + 1 < len(df):
            extra = _probe(df.iloc[i + 1])
            for k, v in extra.items():
                base.setdefault(k, v)
        score = 0
        if "product" in base:
            score += 3
        if "qty_end" in base:
            score += 2
        if score >= 4 and score > score_best:
            head = [clean(df.iat[i, j]) for j in range(width)]
            best = (base, i + 1, head)
            score_best = score
    if not best:
        for i in range(min(max_scan, len(df))):
            base = _probe(df.iloc[i])
            if "product" in base and "qty_end" in base:
                return (base, i + 1, [clean(x) for x in df.iloc[i].tolist()])
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
    key_cols = [j for j in (colmap.get("product"), colmap.get("qty_end")) if j is not None]
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

def is_main_category(name: str) -> bool:
    name_clean = clean(name).lower().strip()
    if name_clean in MAIN_CATEGORIES:
        return True
    words = name_clean.split()
    if len(words) <= 3:
        has_digits = any(char.isdigit() for char in name_clean)
        has_brackets = '(' in name_clean or ')' in name_clean
        has_slashes = '/' in name_clean
        has_dates = re.search(r'\d{2,}\.\d{2,}\.\d{2,}', name_clean)
        if not (has_digits or has_brackets or has_slashes or has_dates):
            return True
    return False

def is_product_detail(name: str) -> bool:
    name_clean = clean(name).lower()
    has_digits = any(char.isdigit() for char in name_clean)
    has_brackets = '(' in name_clean or ')' in name_clean
    has_slashes = '/' in name_clean
    has_dates = re.search(r'\d{2,}\.\d{2,}\.\d{2,}', name_clean)
    has_kg = 'кг' in name_clean or 'kg' in name_clean
    return any([has_digits, has_brackets, has_slashes, has_dates, has_kg])

def parse_grouped(df: pd.DataFrame, colmap: Dict[str, int]) -> Dict[str, Any]:
    pj = colmap.get("product", -1)
    qj = colmap.get("qty_end", -1)
    cj = colmap.get("cost", -1)
    uj = colmap.get("unit_cost", -1)

    warehouse_name = "Оптовый"
    groups: Dict[str, Dict[str, Any]] = {}
    current_category: Optional[str] = None

    LOG.info("=== ПАРСИНГ: определение категорий и товаров ===")

    for _, r in df.iterrows():
        row = r.tolist()
        name = clean(row[pj]) if 0 <= pj < len(row) else ""
        qty = to_float(row[qj]) if 0 <= qj < len(row) else float("nan")

        if not name:
            continue
        if is_total_line(name):
            continue
        if name.lower() == "оптовый":
            warehouse_name = "Оптовый"
            continue

        if is_main_category(name) and not is_product_detail(name):
            current_category = clean(name)
            groups.setdefault(current_category, {
                "category": current_category,
                "item_list": []
            })
            continue

        if current_category and is_product_detail(name):
            item = {
                "product": name,
                "qty": 0.0 if math.isnan(qty) else float(qty),
            }
            cost = to_float(row[cj]) if 0 <= cj < len(row) else float("nan")
            if not math.isnan(cost):
                item["cost"] = float(cost)
            ucost = to_float(row[uj]) if 0 <= uj < len(row) else float("nan")
            if not math.isnan(ucost):
                item["unit_cost"] = float(ucost)
            groups[current_category]["item_list"].append(item)
            continue

        if not current_category and is_product_detail(name):
            if "Без категории" not in groups:
                groups["Без категории"] = {
                    "category": "Без категории",
                    "item_list": []
                }
            item = {
                "product": name,
                "qty": 0.0 if math.isnan(qty) else float(qty),
            }
            groups["Без категории"]["item_list"].append(item)

    # Удаляем пустые категории
    groups = {cat: data for cat, data in groups.items() if data["item_list"]}

    # Считаем общее количество по категориям
    for cat_data in groups.values():
        cat_data["total_qty"] = sum(it.get("qty", 0.0) for it in cat_data["item_list"])
        cat_data["total_cost"] = sum(it.get("cost", 0.0) for it in cat_data["item_list"])

    # Сортируем категории по убыванию общего количества
    sorted_categories = sorted(
        groups.values(),
        key=lambda x: (-x["total_qty"], x["category"].lower())
    )

    # Сортируем товары внутри категории по убыванию количества
    for cat in sorted_categories:
        cat["item_list"].sort(key=lambda x: (-float(x.get("qty", 0)), x["product"].lower()))

    total_qty = sum(cat["total_qty"] for cat in sorted_categories)
    total_cost = sum(cat.get("total_cost", 0.0) for cat in sorted_categories)

    LOG.info("=== РЕЗУЛЬТАТЫ ПАРСИНГА ===")
    LOG.info("Найдено категорий: %d", len(sorted_categories))
    for cat in sorted_categories:
        LOG.info("  %s: %d товаров, всего %.2f", cat["category"], len(cat["item_list"]), cat["total_qty"])
    LOG.info("Общий итог: %.2f", total_qty)

    return {
        "warehouse": warehouse_name,
        "categories": sorted_categories,
        "total_qty": total_qty,
        "total_cost": total_cost if total_cost else None,
    }

def save_json(data: Dict[str, Any], slug: str) -> Path:
    json_path = OUT_JSON / f"inventory_{slug}.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("JSON сохранён: %s", json_path)
    return json_path

def build_report(xlsx: Path) -> Path:
    raw = read_excel_raw(xlsx)
    colmap, data_start, _ = find_header_block(raw)
    period = extract_period(raw, data_start)
    data_end = find_data_end(raw, data_start, colmap)
    df = raw.iloc[data_start:data_end + 1].reset_index(drop=True)

    data = parse_grouped(df, colmap)

    # Slug для имён файлов
    slug = re.sub(r"[^\w\-]+", "_", f"{xlsx.stem}_{period}").strip("_").lower()

    # Сохраняем JSON
    save_json(data, slug)

    # Генерируем HTML
    env = Environment(
        loader=FileSystemLoader([str(TEMPLATES)]),
        autoescape=select_autoescape(["html"])
    )
    env.filters["qty"] = fmt_qty
    env.filters["money"] = fmt_int
    env.filters["price"] = fmt_int

    # Fix v1.1.2: шаблон ждёт products[] (плоский список), а не categories[] (вложенная структура)
    products = []
    for cat in data.get("categories", []):
        cat_name = cat.get("category", "")
        for item in cat.get("item_list", []):
            products.append({
                "category": cat_name,
                "product":  item.get("product", ""),
                "qty_raw":  float(item.get("qty", 0.0)),
            })

    tpl = env.get_template("inventory_simple.html")
    html = tpl.render(
        title=f"Остатки — {data['warehouse']}",
        period=period,
        generated_at=datetime.now(TZ).strftime("%d.%m.%Y %H:%M"),
        products=products,
        grand_total_qty=fmt_qty(data["total_qty"]),
        grand_total_cost=fmt_int(data["total_cost"]) if data["total_cost"] else "—"
    )
    html_path = OUT_HTML / f"inventory_{slug}.html"
    html_path.write_text(html, encoding="utf-8")
    LOG.info("HTML сохранён: %s", html_path)
    return html_path

def main(argv: Optional[List[str]] = None) -> int:
    _ensure_file_logging()
    ap = argparse.ArgumentParser(description="Остатки товаров на складах (группировка по категориям)")
    ap.add_argument("xlsx", help="Excel-файл ведомости остатков")
    args = ap.parse_args(argv)

    try:
        p = Path(args.xlsx)
        if not p.exists():
            LOG.error("Файл не существует: %s", p)
            return 1
        LOG.info("Файл: %s", p)
        out = build_report(p)
        LOG.info("OK: %s", out)
        return 0
    except Exception as e:
        LOG.error("Ошибка", exc_info=e)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
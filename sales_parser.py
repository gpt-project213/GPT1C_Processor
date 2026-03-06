#!/usr/bin/env python
# coding: utf-8
"""
sales_parser.py · v1.0.3 (2026-03-01)
────────────────────────────────────────────────────────────────────
Парсер отчётов "Продажи" из 1С в JSON формат.

ИСПРАВЛЕНИЯ v1.0.3:
- БАГ #9: Устранён двойной счёт выручки (+30-70%).
  Строка клиента определяется по ОТСУТСТВИЮ qty+price (а не sale).
  Раньше итоговая строка клиента (с суммой) засчитывалась как товар.

ИСПРАВЛЕНИЯ v1.0.1:
- Добавлено распознавание клиентов по префиксу "О " (организация)
- Фильтрация служебной строки "Номенклатура"
- Товары распознаются по паттерну (ДДММГГ) ЦЕНА

Вход:  Excel файл "Продажи по менеджерам"
Выход: reports/json/sales_<slug>.json

Структура JSON:
{
  "source_file": str,
  "report_type": "CLIENT_GROUPED",
  "period": str,
  "manager": str,
  "total_revenue": float,
  "client_count": int,
  "clients": [
    {
      "client": str,
      "total": float,
      "products": [
        {"product": str, "qty": float, "price": float, "sum": float}
      ]
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
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional

import pandas as pd

try:
    from utils_excel import ensure_clean_xlsx
except ImportError:
    ensure_clean_xlsx = None

# ──────────────────────────────────────────────────────────────────
# Настройки
TZ = timezone(timedelta(hours=5))
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
LOG = logging.getLogger("sales_parser")

__VERSION__ = "1.0.3"

NBSP = "\u202f"

# ──────────────────────────────────────────────────────────────────
# Регулярные выражения
TOTAL_RE = re.compile(r"\b(итог(?:о)?|всего|итоги|общий итог|total)\b", re.I)
PERIOD_RE = re.compile(r"период[:\s]*([^\n|;]+)", re.I)
MANAGER_RE = re.compile(r"(менеджер|manager)[:\s]*([^\n|;]+)", re.I)

# ──────────────────────────────────────────────────────────────────
# Fallback: менеджер из имени файла (если в Excel нет строки "Менеджер:")
def _load_manager_aliases() -> list[tuple[str, str]]:
    """Возвращает список (alias_lower, canonical_name). Без внешних зависимостей."""
    # Пытаемся читать config/managers.json (если есть)
    try:
        cfg = (Path(__file__).resolve().parent / "config" / "managers.json")
        if cfg.exists():
            data = json.loads(cfg.read_text(encoding="utf-8"))
            out = []
            for m in data.get("managers", []):
                name = str(m.get("name", "")).strip()
                if not name:
                    continue
                aliases = m.get("aliases") or []
                # Само имя тоже алиас
                aliases = list(aliases) + [name]
                for a in aliases:
                    a = str(a).strip()
                    if a:
                        out.append((a.lower(), name))
            if out:
                return out
    except Exception:
        pass

    # Жёсткий fallback (канонические менеджеры проекта)
    return [
        ("алена", "Алена"),
        ("оксана", "Оксана"),
        ("магира", "Магира"),
        ("ергали", "Ергали"),
        ("арман", "Арман"),
    ]

def _manager_from_filename(xlsx_path: Path | None) -> Optional[str]:
    if not xlsx_path:
        return None
    name = xlsx_path.name.lower()
    for alias, canonical in _load_manager_aliases():
        if alias and alias in name:
            return canonical
    return None

# Колонки
COLS = {
    "client":  ("контраг", "клиент", "покупат", "организ"),
    "product": ("товар", "номенк", "наимен", "продукт"),
    "qty":     ("колич", "кол-во", "кол во", "кол", "ед"),
    "price":   ("цена", "price", "стоимость единицы"),
    "sale":    ("сумма", "продаж", "выруч", "реализ", "оборот"),
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
    p = LOGS / f"sales_parser_{ts}.log"
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
        
        if score >= 4 and score > score_best:
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

def extract_manager(df: pd.DataFrame, data_start: int, xlsx_path: Optional[Path] = None) -> Optional[str]:
    for i in range(min(20, data_start)):
        line = " | ".join(clean(x) for x in df.iloc[i].tolist() if clean(x))
        m = MANAGER_RE.search(line)
        if m:
            return m.group(2).strip()
    # fallback: из имени файла
    return _manager_from_filename(xlsx_path)

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

def is_client_header(row: List[str], colmap: Dict[str, int]) -> bool:
    """
    Определить строку заголовка клиента.
    
    v1.0.3: Клиент = есть имя, НЕТ количества и НЕТ цены.
    В 1С отчёте "Продажи":
      - Строка клиента: имя, пусто qty, пусто price, есть итоговая sum
      - Строка товара:  имя, есть qty, есть price, есть sum
    Старая логика (отсутствие sale) была неверной — у клиентов тоже есть сумма.
    """
    client_j = colmap.get("client", colmap.get("product", -1))
    if client_j == -1 or client_j >= len(row):
        return False
    
    name = clean(row[client_j])
    if not name:
        return False
    
    # Служебные строки — НЕ клиенты
    if name.lower() in ("номенклатура", "контрагент"):
        return False
    
    # Проверяем qty и price — если оба пусты → строка клиента (итоговая)
    qty_j = colmap.get("qty", -1)
    price_j = colmap.get("price", -1)
    
    qty_empty = True
    if qty_j != -1 and qty_j < len(row):
        qty_val = to_float(row[qty_j])
        if not math.isnan(qty_val) and abs(qty_val) > 0.0001:
            qty_empty = False
    
    price_empty = True
    if price_j != -1 and price_j < len(row):
        price_val = to_float(row[price_j])
        if not math.isnan(price_val) and abs(price_val) > 0.0001:
            price_empty = False
    
    # Клиент = нет количества И нет цены (это итоговая строка клиента)
    if qty_empty and price_empty:
        return True
    
    # Дополнительный паттерн: явные маркеры клиента в названии
    if name.startswith("О ") or " ТОО " in name or " ИП " in name:
        return True
    
    return False

# ──────────────────────────────────────────────────────────────────
# Основной парсер
def parse_sales_grouped(df: pd.DataFrame, colmap: Dict[str, int]) -> Dict[str, Any]:
    client_j = colmap.get("client", colmap.get("product", -1))
    prod_j = colmap.get("product", -1)
    qty_j = colmap.get("qty", -1)
    price_j = colmap.get("price", -1)
    sale_j = colmap.get("sale", -1)
    
    clients: List[Dict[str, Any]] = []
    current_client: Optional[Dict[str, Any]] = None
    total_revenue = 0.0
    
    for _, r in df.iterrows():
        row = r.tolist()
        
        # Пропустить итоговые строки
        name = clean(row[prod_j]) if prod_j != -1 and prod_j < len(row) else ""
        if is_total_line(name):
            continue
        
        # Пропустить служебные строки
        if name.lower() in ("номенклатура", "контрагент"):
            continue
        
        # Проверить заголовок клиента
        if is_client_header(row, colmap):
            client_name = clean(row[client_j])
            current_client = {
                "client": client_name,
                "total": 0.0,
                "products": []
            }
            clients.append(current_client)
            continue
        
        # Строка товара
        if not name:
            continue
        
        qty = to_float(row[qty_j]) if qty_j != -1 and qty_j < len(row) else float("nan")
        price = to_float(row[price_j]) if price_j != -1 and price_j < len(row) else float("nan")
        sale = to_float(row[sale_j]) if sale_j != -1 and sale_j < len(row) else float("nan")
        
        if math.isnan(sale):
            sale = 0.0
        if math.isnan(qty):
            qty = 0.0
        if math.isnan(price):
            price = 0.0
        
        # Если нет текущего клиента - создать "(Без клиента)"
        if current_client is None:
            current_client = {
                "client": "(Без клиента)",
                "total": 0.0,
                "products": []
            }
            clients.append(current_client)
        
        product = {
            "product": name,
            "qty": float(qty),
            "price": float(price),
            "sum": float(sale)
        }
        current_client["products"].append(product)
        current_client["total"] += sale
        total_revenue += sale
    
    # Сортировка
    for client in clients:
        client["products"].sort(key=lambda x: (-x["sum"], x["product"]))
    
    clients.sort(key=lambda c: (-c["total"], c["client"]))
    
    return {
        "clients": clients,
        "total_revenue": total_revenue,
        "client_count": len(clients)
    }

# ──────────────────────────────────────────────────────────────────
# Главная функция
def build_sales_json(xlsx: Path) -> Path:
    raw = read_excel_raw(xlsx)
    colmap, data_start, _ = find_header_block(raw)
    period = extract_period(raw, data_start)
    manager = extract_manager(raw, data_start, xlsx)
    data_end = find_data_end(raw, data_start, colmap)
    df = raw.iloc[data_start:data_end + 1].reset_index(drop=True)
    
    data = parse_sales_grouped(df, colmap)
    
    result = {
        "source_file": xlsx.name,
        "report_type": "CLIENT_GROUPED",
        "period": period,
        "manager": manager,
        "total_revenue": data["total_revenue"],
        "client_count": data["client_count"],
        "clients": data["clients"],
        "metadata": {
            "version": __VERSION__,
            "parsed_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": "Asia/Qyzylorda"
        }
    }
    
    # Создать имя файла
    slug = slugify(f"{xlsx.stem}_{period}")
    if manager:
        slug = slugify(f"{manager}_{period}")
    
    out = JSON_OUT / f"sales_{slug}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    LOG.info("JSON создан: %s", out)
    LOG.info("Клиентов: %d, Выручка: %.2f ₸", 
             data["client_count"], data["total_revenue"])
    
    return out

# ──────────────────────────────────────────────────────────────────
# CLI
def main(argv: Optional[List[str]] = None) -> int:
    _ensure_file_logging()
    ap = argparse.ArgumentParser(description="Парсер продаж → JSON")
    ap.add_argument("xlsx", help="Excel-файл продаж")
    args = ap.parse_args(argv)
    
    try:
        p = Path(args.xlsx)
        LOG.info("Файл: %s", p)
        out = build_sales_json(p)
        LOG.info("OK: %s", out)
        return 0
    except Exception as e:
        LOG.error("FAILED", exc_info=e)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
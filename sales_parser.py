#!/usr/bin/env python
# coding: utf-8
"""
sales_parser.py · v1.0.5 (2026-03-10)
────────────────────────────────────────────────────────────────────
Парсер отчётов "Продажи" из 1С в JSON формат.

ИСПРАВЛЕНИЯ v1.0.5:
- БАГ TZ: TZ = timezone(timedelta(hours=5)) заменён на ZoneInfo("Asia/Almaty").
  Импорт timezone+timedelta удалён, добавлен ZoneInfo.

ИСПРАВЛЕНИЯ v1.0.4:
- БАГ B2: _load_manager_aliases() читала managers.json в неправильном формате.
  Ожидала: {"managers": [{"name": "Оксана", "aliases": [...]}]}
  Фактический формат: {"Оксана": 1446255940, "Магира": 735574334, ...}
  Теперь читает корректно: ключ = имя менеджера, значение = chat_id (игнорируем).
  Fallback-список очищен от уволенного "Арман".

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
LOG = logging.getLogger("sales_parser")

__VERSION__ = "1.0.5"

NBSP = "\u202f"

# ──────────────────────────────────────────────────────────────────
# Регулярные выражения
TOTAL_RE = re.compile(r"\b(итог(?:о)?|всего|итоги|общий итог|total)\b", re.I)
PERIOD_RE = re.compile(r"период[:\s]*([^\n|;]+)", re.I)
MANAGER_RE = re.compile(r"(менеджер|manager)[:\s]*([^\n|;]+)", re.I)

# Системный менеджер — исключаем из всех списков
_SYSTEM_MANAGERS = {"Минай"}

# ──────────────────────────────────────────────────────────────────
# Fallback: менеджер из имени файла (если в Excel нет строки "Менеджер:")
def _load_manager_aliases() -> list[tuple[str, str]]:
    """
    Возвращает список (alias_lower, canonical_name).

    v1.0.4: Исправлен формат чтения managers.json.
    Реальный формат файла: {"Оксана": 1446255940, "Магира": 735574334, ...}
    (ключ = имя менеджера, значение = chat_id — его игнорируем здесь)
    """
    try:
        cfg = Path(__file__).resolve().parent / "config" / "managers.json"
        if cfg.exists():
            data = json.loads(cfg.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                out = []
                for name, _chat_id in data.items():
                    name = str(name).strip()
                    if not name or name in _SYSTEM_MANAGERS:
                        continue
                    out.append((name.lower(), name))
                if out:
                    LOG.debug("Загружено %d менеджеров из managers.json", len(out))
                    return out
    except Exception as e:
        LOG.warning("Не удалось прочитать managers.json: %s — используется fallback", e)

    LOG.warning("managers.json недоступен — определение менеджера по имени файла отключено")
    return []

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

        if math.isnan(sale) or sale == 0:
            continue

        product = {
            "product": name,
            "qty": 0.0 if math.isnan(qty) else round(qty, 3),
            "price": 0.0 if math.isnan(price) else round(price, 2),
            "sum": round(sale, 2)
        }

        if current_client is None:
            current_client = {"client": "Без клиента", "total": 0.0, "products": []}
            clients.append(current_client)

        current_client["products"].append(product)
        current_client["total"] = round(current_client["total"] + sale, 2)
        total_revenue = round(total_revenue + sale, 2)

    return {
        "clients": clients,
        "total_revenue": total_revenue
    }

# ──────────────────────────────────────────────────────────────────
# Публичный API
def parse_file(xlsx_path: str | Path, out_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Основная точка входа: парсит Excel-файл продаж → JSON.
    Возвращает путь к JSON или None при ошибке.
    """
    _ensure_file_logging()
    path = Path(xlsx_path)

    if not path.exists():
        LOG.error("Файл не найден: %s", path)
        return None

    LOG.info("Парсинг: %s", path.name)

    try:
        df = read_excel_raw(path)
        colmap, data_start, headers = find_header_block(df)

        period = extract_period(df, data_start)
        manager = extract_manager(df, data_start, path)

        data_end = find_data_end(df, data_start, colmap)
        if data_end < data_start:
            LOG.warning("Нет строк данных в файле")
            return None

        data_df = df.iloc[data_start:data_end + 1].reset_index(drop=True)
        result = parse_sales_grouped(data_df, colmap)

        slug = slugify(path.stem)
        ts = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
        out_name = f"sales_{slug}_{ts}.json"

        out_path = (out_dir or JSON_OUT) / out_name
        payload = {
            "source_file": path.name,
            "report_type": "CLIENT_GROUPED",
            "period": period,
            "manager": manager or "Не определён",
            "total_revenue": result["total_revenue"],
            "client_count": len(result["clients"]),
            "clients": result["clients"],
            "metadata": {
                "version": __VERSION__,
                "generated_at": datetime.now(TZ).isoformat(),
                "headers": headers,
                "colmap": colmap,
            }
        }

        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        LOG.info("JSON сохранён: %s (менеджер=%s, клиентов=%d, выручка=%.2f)",
                 out_path.name, manager, len(result["clients"]), result["total_revenue"])
        return out_path

    except Exception as e:
        LOG.exception("Ошибка парсинга %s: %s", path.name, e)
        return None


# ──────────────────────────────────────────────────────────────────
# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Парсер отчётов продаж 1С")
    parser.add_argument("file", help="Путь к Excel файлу")
    parser.add_argument("--out", help="Папка для JSON (по умолчанию reports/json/)")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else None
    result = parse_file(args.file, out_dir)
    if result:
        print(f"OK: {result}")
    else:
        print("ERROR: парсинг не удался, см. логи")
        exit(1)
#!/usr/bin/env python
# coding: utf-8
r"""
sales_report.py · v9.3.7 · 2025-09-18 (Asia/Almaty)
Улучшенная обработка метаданных и классификации строк
Создано DeepSeek AI • Дипсик Аналитика
"""

from __future__ import annotations

import math, re, logging, argparse, html as _html
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Any, Optional
from utils_excel import ensure_clean_xlsx

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

# ──────────────────────────────────────────────────────────────────────────────
# Пути / окружение
TZ = timezone(timedelta(hours=5))  # Asia/Almaty
ROOT = Path(__file__).resolve().parent

def _pick_dir(options: List[Path]) -> Path:
    for p in options:
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            continue
    return options[-1]

TPL = next((p for p in [Path(r"E:\templates"), ROOT / "templates"] if p.exists()), ROOT / "templates")
OUT = _pick_dir([Path(r"E:\reports\html"), ROOT / "reports" / "html"])
(OUT / "suspects").mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s, %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
LOG = logging.getLogger("sales_report")

# ──────────────────────────────────────────────────────────────────────────────
# Лог-файл
def _ensure_file_logging():
    log_dir = Path(r"E:\logs")
    if not log_dir.exists():
        log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"sales_{ts}.log"
    if not any(isinstance(h, logging.FileHandler) for h in LOG.handlers):
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s, %(levelname)s %(message)s"))
        fh.setLevel(logging.INFO)
        LOG.addHandler(fh)
        LOG.info("Лог-файл: %s", log_path)

# ──────────────────────────────────────────────────────────────────────────────
# Утилиты/форматирование
NBSP = "\u202f"
EPS = 1e-9
DELTA_TOL = 25
BIG_NO_PRODUCT = 20000  # порог «крупной суммы без товарных признаков»

def clean(x: Any) -> str:
    if pd.isna(x) or x is None or str(x).lower() in ['nan', 'none', '']:
        return ""
    s = str(x).replace("\u00a0"," ").replace(NBSP," ").strip()
    return re.sub(r"\s+"," ", s)

def to_float(x: Any) -> float:
    s = clean(x)
    if not s or s.lower() in ['nan', 'none', '']:
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

def round_half_up_to_int(v: float | int) -> int:
    if not isinstance(v,(int,float)) or (isinstance(v,float) and math.isnan(v)):
        return 0
    neg = v < 0
    v = abs(float(v))
    i = int(v)
    if v - i >= 0.5 - 1e-12:
        i += 1
    return -i if neg else i

def fmt_int(v: float | int) -> str:
    n = round_half_up_to_int(v)
    s = f"{abs(n):,}".replace(",", NBSP)
    return f"-{s}" if n < 0 else s

def fmt_qty3(v: float | int) -> str:
    if v is None or (isinstance(v,float) and math.isnan(v)):
        return "0"
    f = float(v)
    s = f"{f:.3f}".rstrip("0").rstrip(".")
    if "." in s:
        i, d = s.split(".",1)
    else:
        i, d = s, ""
    try:
        i_fmt = f"{int(i):,}".replace(",", NBSP)
        return f"{i_fmt}.{d}" if d else i_fmt
    except Exception:
        return s

# ──────────────────────────────────────────────────────────────────────────────
# Регэкспы/синонимы колонок и метаданные
META_RE   = re.compile(r"(показатели|группиров|отбор|сортиров|формат|ресурс|поля)", re.I)
TOTAL_RE  = re.compile(r"\b(итог(?:о)?(?:\s+клиента)?|всего|итоги|общий итог|итог по|всего по|к оплате|total)\b", re.I)
NUM_RE    = re.compile(r"[^\d,.\-]")
EXCL_RE   = re.compile(r"тов[\w.]*\s*под\s*з\s*[\/\\.]?\s*п", re.I)

CLIENT_LEGAL_KZ = re.compile(r"\b(ИП|ТОО|ЖШС|АО)\b", re.I)
CLIENT_BIZ_KW   = re.compile(r"\b(кафе|ресторан|столовая|бар|маркет|магазин|бутик|рынок|склад|цех|трц|университет|колледж|школa|садик|акимат|кгу|кгп)\b", re.I)
CLIENT_TEL      = re.compile(r"(?:\+?7|8)\D?\d{3}\D?\d{3}\D?\d{2}\D?\d{2}")
CLIENT_ADDR     = re.compile(r"\b(ул|улица|просп|пр-кт|пр-т|мкр|микрорайон|дом|д\.|корп|кв|№|г\.)\b", re.I)
CLIENT_LETTER_PREF = re.compile(r"^[А-ЯЁA-Z]\s")
AR_PREFIX       = re.compile(r"^\s*ар\s", re.I)
CHL_MARKER      = re.compile(r"\bч/л\b", re.I)

SALARY_HEADER_RE = re.compile(
    r"\b(зарплат|оклад|преми|бонус|начислени|удержани)\b",
    re.IGNORECASE | re.UNICODE,
)

PHONE_LOOSE_RE   = re.compile(r'(?<!\d)(?:\+?7|8)\d{7,10}(?!\d)')
ADDRESS_EXTRA_RE = re.compile(r'\b(рынок|павильон|трц|ряд|место|секция|этаж|киоск|витрина|жк|мкр)\b', re.I)

UNIT = r"(?:кг|г|гр|л|мл|мм|см|м|шт|уп|пак|кор|бут|бан|ящ|пал|roll|pcs|box)"
PROD_NUM_UNIT = re.compile(rf"\b\d+(?:[.,]\d+)?\s*{UNIT}\b", re.I)
PROD_SIZE     = re.compile(r"\b\d{2,4}\s*[xх×]\s*\d{2,4}\b", re.I)
PROD_PCT      = re.compile(r"\b\d{1,3}\s?%\b", re.I)

COLS = {
    "client":  ("контрагент","клиент","покупател"),
    "product": ("номенклатура","товар","наимен"),
    "qty":     ("колич",),
    "price":   ("цена",),
    "sale":    ("сумма продажи","сумма","оборот","стоим","выруч","итог"),
}

# Улучшенные регулярные выражения для метаданных
PERIOD_RE = re.compile(r"период[:\s]*([^\n|;]+)", re.I)
MANAGER_RE = re.compile(r"(?:менеджер|ответственный|торгпред)[\s:\-–—]*([^\n|;]+)", re.I)

def has_product_signals(text: str) -> bool:
    t = text or ""
    return bool(
        PROD_NUM_UNIT.search(t) or
        PROD_SIZE.search(t)     or
        PROD_PCT.search(t)      or
        re.search(r"\(\d{6}\)", t) or
        re.search(r"/\d{1,3}\b", t)
    )

# ──────────────────────────────────────────────────────────────────────────────
# Поиск шапки/мета/границ таблицы
def _probe(row) -> Dict[str,int]:
    res={}
    for i,cell in enumerate(row):
        t = clean(cell).lower()
        for k,alts in COLS.items():
            if k in res: continue
            if any(a in t for a in alts):
                if k == "qty" and (("ед" in t and "отчет" in t) or ("ед" in t and "отч" in t)):
                    continue
                res[k]=i; break
    return res

def _prefer_qty_index(candidates_row: List[str], mapping: Dict[str,int]) -> None:
    qi = mapping.get("qty")
    if qi is None: return
    title = clean(candidates_row[qi]).lower()
    if "ед" in title and "отчет" in title:
        best = None
        for j, val in enumerate(candidates_row):
            t = clean(val).lower()
            if "колич" in t and "(" not in t and "отчет" not in t and "ед" not in t:
                best = j; break
        if best is not None:
            mapping["qty"] = best

def find_header_block(df: pd.DataFrame, max_scan: int = 80) -> Tuple[Dict[str,int], int, List[str]]:
    width = df.shape[1]
    for i in range(min(max_scan, len(df))):
        base = _probe(df.iloc[i])
        if len(base) < 2 or "sale" not in base:
            continue
        comb = [clean(df.iat[i, j]) for j in range(width)]
        for j in (1,2):
            if i+j < len(df):
                rowj = df.iloc[i+j]
                got  = _probe(rowj)
                for k,v in got.items():
                    base[k]=v
                for c in range(width):
                    v = clean(rowj[c])
                    if v:
                        comb[c] = v
        if base.get("client") == base.get("product") and i+3 < len(df):
            extra = _probe(df.iloc[i+3])
            for k in ("product","qty","price"):
                if extra.get(k) is not None:
                    base[k] = extra[k]
            data_start = i+3
        else:
            data_start = i + (1 if "product" in base else 2)
        _prefer_qty_index(comb, base)
        return base, data_start, comb
    raise ValueError("Не найдена строка шапки")

def extract_meta(df: pd.DataFrame, data_start: int) -> Tuple[str,str]:
    period = ""
    manager = ""
    
    for i in range(min(20, data_start)):
        row_text = " | ".join(clean(x) for x in df.iloc[i] if clean(x))
        
        if not period:
            period_match = PERIOD_RE.search(row_text)
            if period_match:
                period = period_match.group(1).strip()
                
        if not manager:
            manager_match = MANAGER_RE.search(row_text)
            if manager_match:
                manager = manager_match.group(1).strip()
                manager = re.sub(r'^(?:равно|[=:–—-])\s*', '', manager, flags=re.I)
        
        if period and manager:
            break
            
    # Дополнительная попытка извлечь менеджера
    if not manager:
        for i in range(min(20, data_start)):
            for j in range(df.shape[1]):
                cell_value = clean(df.iat[i, j])
                if "ответственный" in cell_value.lower() and "равно" in cell_value.lower():
                    parts = cell_value.split("равно")
                    if len(parts) > 1:
                        manager = parts[-1].strip().split(';')[0]
                        break
            if manager:
                break
                
    # Очистка от лишних данных
    period = re.split(r'[|;]', period)[0].strip() if period else "Не указан"
    manager = re.split(r'[|;]', manager)[0].strip() if manager else "Не указан"
    
    return period, manager

def read_excel_raw(path: Path) -> pd.DataFrame:
    clean_path = ensure_clean_xlsx(path, force_fix=True)
    return pd.read_excel(clean_path, header=None, dtype=str)

def find_data_end(raw: pd.DataFrame, data_start: int, colmap: Dict[str,int]) -> int:
    table_cols = [j for j in (colmap.get("product"), colmap.get("sale"), colmap.get("qty"), colmap.get("price")) if j is not None]
    consec_empty = 0
    last_non_empty = data_start - 1
    for i in range(data_start, len(raw)):
        row = raw.iloc[i]
        empty = True
        for j in table_cols:
            if j is None or j >= len(row): continue
            if clean(row[j]):
                empty = False; break
        if empty:
            consec_empty += 1
            if consec_empty >= 2:
                return last_non_empty
        else:
            consec_empty = 0
            last_non_empty = i
    return last_non_empty
# ──────────────────────────────────────────────────────────────────────────────
# Классификация строк
def classify_row(row: List[Any], colmap: Dict[str,int]) -> str:
    def get_txt(key: str) -> str:
        j = colmap.get(key)
        if j is None or j >= len(row): return ""
        return clean(row[j])

    prod   = get_txt("product")
    client = get_txt("client")
    row_text = " | ".join(clean(x) for x in row if clean(x))
    name = prod or client
    t = (name or "").lower()

    if not prod and not client and not row_text:
        return "EMPTY"

    # TOTAL - улучшенная проверка
    row_lower = row_text.lower()
    if (TOTAL_RE.search((prod or "").lower()) or 
        TOTAL_RE.search((client or "").lower()) or 
        TOTAL_RE.search(row_lower)):
        # Дополнительная проверка, чтобы не спутать с клиентом
        if not any(x in row_lower for x in ["ар ", "ч/л", "маг ", "кафе ", "ресторан "]):
            return "TOTAL"

    # КРИТИЧЕСКОЕ: крупная сумма БЕЗ товарных признаков → CLIENT_HEADER
    sale_j = colmap.get("sale", -1)
    if sale_j != -1 and sale_j < len(row):
        sale_val = to_float(row[sale_j])
        if isinstance(sale_val, float) and not math.isnan(sale_val) and abs(sale_val) >= BIG_NO_PRODUCT:
            if not has_product_signals(prod):
                client_indicators = (
                    AR_PREFIX.search(name) or 
                    CHL_MARKER.search(name) or
                    CLIENT_LEGAL_KZ.search(t) or 
                    CLIENT_BIZ_KW.search(t) or
                    CLIENT_TEL.search(row_text) or 
                    CLIENT_ADDR.search(row_text) or
                    CLIENT_LETTER_PREF.search(name) or 
                    ADDRESS_EXTRA_RE.search(row_text) or
                    SALARY_HEADER_RE.search(name)
                )
                
                if client_indicators:
                    return "CLIENT_HEADER"

    # META: телефоны/адреса и т.п.
    if PHONE_LOOSE_RE.search(row_text) or ADDRESS_EXTRA_RE.search(row_text):
        return "META"

    # «товар под/по ЗП» — это шапка клиента
    if SALARY_HEADER_RE.search(name):
        return "CLIENT_HEADER"

    # Обычные клиентские признаки
    client_indicators = (
        CLIENT_LEGAL_KZ.search(t) or 
        CLIENT_BIZ_KW.search(t) or
        CLIENT_TEL.search(row_text) or 
        CLIENT_ADDR.search(row_text) or
        CLIENT_LETTER_PREF.search(name) or 
        AR_PREFIX.search(name) or 
        CHL_MARKER.search(name)
    )
    
    if client_indicators:
        return "CLIENT_HEADER"

    # Остальное — товар
    return "PRODUCT"

# ──────────────────────────────────────────────────────────────────────────────
# Итоги из тела и из мета-зон
def table_grand_total_from_body(df: pd.DataFrame, colmap: Dict[str,int]) -> Optional[int]:
    if "sale" not in colmap: return None
    sale_j = colmap["sale"]
    for i in range(len(df) - 1, -1, -1):
        row = df.iloc[i].tolist()
        tag = classify_row(row, colmap)
        if tag != "TOTAL":
            continue
        sale_val = to_float(row[sale_j]) if sale_j < len(row) else float("nan")
        if not (isinstance(sale_val, float) and math.isnan(sale_val)):
            return round_half_up_to_int(sale_val)
    return None

def _ints_in_text(s: str) -> List[int]:
    if not s: return []
    s = s.replace("\u202f"," ").replace("\xa0"," ")
    raw = re.findall(r"-?\d[\d\s]{3,}", s)
    out=[]
    for n in raw:
        try: out.append(int(re.sub(r"[^\d-]", "", n)))
        except Exception: pass
    return out

def extract_excel_total_from_meta(raw: pd.DataFrame, data_start: int, data_end: int, calc_total_i: int) -> Tuple[Optional[int], str]:
    n = len(raw)
    top_idx = list(range(0, min(40, data_start)))
    bottom_from = max(data_end + 1, n - 40)
    bottom_idx = list(range(bottom_from, n)) if bottom_from < n else []

    scale_lo = int(0.1 * abs(calc_total_i)) if calc_total_i else 0
    scale_hi = int(5 * abs(calc_total_i)) if calc_total_i else 0

    candidates: List[Tuple[int,int,int,str,str]] = []

    def push(indices: List[int], zone: str):
        for i in indices:
            vals = [clean(x) for x in raw.iloc[i].tolist()]
            line = " ".join(v for v in vals if v)
            if not line or not re.search(TOTAL_RE, line):
                continue
            nums = _ints_in_text(line)
            if not nums:
                continue
            nums2 = []
            for v in nums:
                if abs(v) < 1000:
                    continue
                if calc_total_i:
                    if abs(v) < scale_lo or abs(v) > scale_hi:
                        continue
                nums2.append(v)
            if not nums2:
                continue
            best = max(nums2, key=lambda x: abs(x))
            distance = i if zone == "top" else (n - 1 - i)
            line_short = line if len(line) <= 120 else (line[:117] + "…")
            candidates.append((distance, int(best), i, line_short, zone))

    push(top_idx, "top")
    push(bottom_idx, "bottom")

    if not candidates:
        return None, ""

    candidates.sort(key=lambda t: (t[0], 0 if t[4]=="top" else 1))
    dist, val, idx, line_short, zone = candidates[0]
    source = f"{zone}: row {idx+1}, '{line_short}'"
    return int(val), source

# ──────────────────────────────────────────────────────────────────────────────
# Режимы сборки
def build_product_mode(df: pd.DataFrame, colmap: Dict[str,int]) -> Tuple[List[Dict[str,Any]], float, List[Dict[str,Any]]]:
    sale_j = colmap.get("sale", -1)
    prod_j = colmap.get("product", -1)
    qty_j = colmap.get("qty", -1)
    price_j = colmap.get("price", -1)
    
    if sale_j == -1 or prod_j == -1:
        return [], 0.0, []
    
    items: Dict[str, Dict[str,Any]] = {}
    suspects: List[Dict[str,Any]] = []
    turnover = 0.0
    
    for idx in range(len(df)):
        row = df.iloc[idx].tolist()
        tag = classify_row(row, colmap)
        if tag != "PRODUCT":
            continue
        
        name = clean(row[prod_j])
        
        # Извлекаем все значения: sale, qty, price
        sale = to_float(row[sale_j]) if sale_j != -1 and sale_j < len(row) else float("nan")
        qty = to_float(row[qty_j]) if qty_j != -1 and qty_j < len(row) else float("nan")
        price_val = to_float(row[price_j]) if price_j != -1 and price_j < len(row) else float("nan")
        
        if isinstance(sale, float) and math.isnan(sale):
            sale = 0.0
        if isinstance(qty, float) and math.isnan(qty):
            qty = 0.0
        if isinstance(price_val, float) and math.isnan(price_val):
            price_val = 0.0
        
        key = name.lower()
        it = items.setdefault(key, {
            "product": name, 
            "qty_raw": 0.0, 
            "price_raw": 0.0, 
            "sale_raw": 0.0
        })
        
        # Аккумулируем все значения
        it["qty_raw"] += qty
        it["sale_raw"] += sale
        
        # Вычисляем средневзвешенную цену
        if it["qty_raw"] > 0:
            it["price_raw"] = it["sale_raw"] / it["qty_raw"]
        
        turnover += sale
        
        t = name.lower()
        if not has_product_signals(t) and abs(sale) >= 20000:
            suspects.append({
                "row": int(idx), 
                "class": "PRODUCT", 
                "text": name, 
                "sale": float(sale), 
                "reason": "prod-no-signals"
            })
    
    products = sorted(items.values(), key=lambda x: (-x["sale_raw"], x["product"]))
    return products, turnover, suspects

def build_client_mode(df: pd.DataFrame, colmap: Dict[str,int]) -> Tuple[List[Dict[str,Any]], float, List[Dict[str,Any]]]:
    clients: List[Dict[str,Any]] = []
    current = None
    sum_sale = 0.0
    suspects: List[Dict[str,Any]] = []
    dup_guard: Dict[str, set] = {}

    sale_j = colmap.get("sale", -1)
    prod_j = colmap.get("product", -1)
    qty_j  = colmap.get("qty", -1)
    price_j= colmap.get("price", -1)

    for idx, r in df.iterrows():
        row = r.tolist()
        tag = classify_row(row, colmap)

        if tag == "CLIENT_HEADER":
            cname = clean(row[colmap.get("client", colmap.get("product"))]) if ("client" in colmap or "product" in colmap) else "(Без клиента)"
            current = {"client": cname, "products": [], "computed": 0.0, "qty_sum": 0.0}
            dup_guard[cname] = set()
            clients.append(current)
            if sale_j != -1 and sale_j < len(row):
                v = to_float(row[sale_j])
                if isinstance(v,float) and not math.isnan(v) and abs(v) >= 20000:
                    suspects.append({"row": int(idx), "class":"CLIENT_HEADER", "text": cname, "sale": float(v), "reason":"client-header-with-sum"})
            continue  # ← КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: пропускаем строки CLIENT_HEADER

        if tag != "PRODUCT":
            continue

        name = clean(row[prod_j]) if prod_j != -1 and prod_j < len(row) else ""
        if EXCL_RE.search(name):
            continue

        sale = to_float(row[sale_j]) if sale_j != -1 and sale_j < len(row) else float("nan")
        qty  = to_float(row[qty_j])  if qty_j  != -1 and qty_j  < len(row) else float("nan")
        price= to_float(row[price_j])if price_j!= -1 and price_j< len(row) else float("nan")
        if isinstance(sale, float) and math.isnan(sale):
            sale = 0.0

        if current is None:
            cname = "(Без клиента)"
            current = {"client": cname, "products": [], "computed": 0.0, "qty_sum": 0.0}
            dup_guard.setdefault(cname, set())
            clients.append(current)

        key = (name or "").lower()
        if key in dup_guard[current["client"]]:
            suspects.append({"row": int(idx), "class": "PRODUCT", "client": current["client"], "text": name, "sale": float(sale), "reason":"duplicate-product"})
        else:
            dup_guard[current["client"]].add(key)

        current["products"].append({
            "product": name or "—",
            "qty_raw":   0.0 if (isinstance(qty,float)  and math.isnan(qty))  else qty,
            "price_raw": 0.0 if (isinstance(price,float)and math.isnan(price)) else price,
            "sale_raw": sale
        })
        current["computed"] += sale
        if isinstance(qty, (int, float)) and not (isinstance(qty, float) and math.isnan(qty)):
            current["qty_sum"] += float(qty)
        sum_sale += sale  # ← Только для PRODUCT строк!

        t = (name or "").lower()
        if not has_product_signals(t) and abs(sale) >= 20000:
            suspects.append({"row": int(idx), "class":"PRODUCT", "text": name, "sale": float(sale), "reason":"prod-no-signals"})

    for c in clients:
        c["computed_i"] = round_half_up_to_int(c["computed"] or 0)
        c["qty_sum_i"] = float(c.get("qty_sum", 0.0))
        c["products"].sort(key=lambda x: (-x["sale_raw"], x["product"]))
    clients.sort(key=lambda c: (-c["computed"], c["client"]))
    return clients, sum_sale, suspects

# ──────────────────────────────────────────────────────────────────────────────
# Δ-форензика → лог и HTML
def log_delta_forensic(df: pd.DataFrame, colmap: Dict[str,int], turnover_i: int, table_total_i: int):
    sale_j = colmap.get("sale", -1)
    prod_j = colmap.get("product", -1)
    client_j = colmap.get("client", -1)
    if sale_j == -1: return
    delta = turnover_i - table_total_i
    LOG.warning("Δ-ФОРЕНЗИКА: расчёт=%s ; итог_таблицы=%s ; Δ=%s", fmt_int(turnover_i), fmt_int(table_total_i), fmt_int(delta))

    client_header_with_sale = []
    product_no_prod_like    = []
    for i in range(len(df)):
        row = df.iloc[i].tolist()
        v = to_float(row[sale_j])
        if not (isinstance(v,float) and not math.isnan(v)):
            continue
        name = clean(row[prod_j]) if prod_j != -1 and prod_j < len(row) else ""
        cname= clean(row[client_j]) if client_j != -1 and client_j < len(row) else ""
        nm = name or cname
        tag = classify_row(row, colmap)
        if tag == "CLIENT_HEADER":
            client_header_with_sale.append((i, round_half_up_to_int(v), nm))
        has_prod = has_product_signals(name)
        if tag == "PRODUCT" and not has_prod and abs(v) >= 20000:
            product_no_prod_like.append((i, round_half_up_to_int(v), nm))

    if client_header_with_sale:
        client_header_with_sale.sort(key=lambda t: (-t[1], t[0]))
        LOG.warning("CLIENT_HEADER с суммой (первые 30):")
        for k,(i,val,name) in enumerate(client_header_with_sale[:30],1):
            LOG.warning("  #%02d row=%d  sale=%s  %s", k, i, fmt_int(val), name)
        LOG.warning("subtotal(CLIENT_HEADER с суммой): %s", fmt_int(sum(v for _,v,_ in client_header_with_sale[:30])))

    if product_no_prod_like:
        product_no_prod_like.sort(key=lambda t: (-t[1], t[0]))
        LOG.warning("PRODUCT без признаков товара (первые 30):")
        for k,(i,val,name) in enumerate(product_no_prod_like[:30],1):
            LOG.warning("  #%02d row=%d  sale=%s  %s", k, i, fmt_int(val), name)
        LOG.warning("subtotal(PRODUCT без признаков товара): %s", fmt_int(sum(v for _,v,_ in product_no_prod_like[:30])))

    target = abs(delta)
    pool = [(abs(val), rn, txt) for rn, val, txt in client_header_with_sale] + [(abs(val), rn, txt) for rn, val, txt in product_no_prod_like]
    pool.sort(reverse=True)
    picked = []
    acc = 0
    for val, rn, txt in pool:
        if acc + val <= target + DELTA_TOL:
            picked.append((val, rn, txt))
            acc += val
    if picked:
        LOG.warning("Кандидаты, покрывающие Δ (жадно): subtotal=%s (из %d строк)", fmt_int(acc), len(picked))
        for k,(val, rn, txt) in enumerate(picked, 1):
            LOG.warning("  pick#%02d row=%d sale=%s %s", k, rn, fmt_int(val), txt)

def build_delta_forensic_html(df: pd.DataFrame, colmap: Dict[str,int], turnover_i: int, table_total_i: int) -> str:
    sale_j = colmap.get("sale", -1)
    prod_j = colmap.get("product", -1)
    client_j = colmap.get("client", -1)
    if sale_j == -1: return ""
    delta = turnover_i - table_total_i
    if abs(delta) <= DELTA_TOL:
        return ""

    client_header_with_sale = []
    product_no_prod_like    = []

    for i in range(len(df)):
        row = df.iloc[i].tolist()
        v = to_float(row[sale_j])
        if not (isinstance(v,float) and not math.isnan(v)):
            continue
        name = clean(row[prod_j]) if prod_j != -1 and prod_j < len(row) else ""
        cname= clean(row[client_j]) if client_j != -1 and client_j < len(row) else ""
        nm = name or cname
        tag = classify_row(row, colmap)
        if tag == "CLIENT_HEADER":
            client_header_with_sale.append((int(i), int(round_half_up_to_int(v)), nm))
        has_prod = has_product_signals(name)
        if tag == "PRODUCT" and not has_prod and abs(v) >= 20000:
            product_no_prod_like.append((int(i), int(round_half_up_to_int(v)), nm))

    client_header_with_sale.sort(key=lambda t: (-t[1], t[0]))
    product_no_prod_like.sort(key=lambda t: (-t[1], t[0]))

    target = abs(delta)
    pool = [
        *[(abs(val), rn, txt) for rn, val, txt in client_header_with_sale],
        *[(abs(val), rn, txt) for rn, val, txt in product_no_prod_like],
    ]
    pool.sort(reverse=True)
    picked = []
    acc = 0
    for val, rn, txt in pool:
        if acc + val <= target + DELTA_TOL:
            picked.append((val, rn, txt)); acc += val
        if acc >= target - DELTA_TOL: break

    def _li(rows, limit):
        out = []
        for k,(rn, val, txt) in enumerate(rows[:limit], start=1):
            out.append(f'<li>#%02d row=%d sale=%s %s</li>' % (k, rn, fmt_int(val), _html.escape(txt)))
        return "\n".join(out)

    block = []
    block.append('<details class="tech"><summary><strong>Δ-форензика:</strong> расчёт=%s; итог_таблицы=%s; Δ=%s</summary>' % (fmt_int(turnover_i), fmt_int(table_total_i), fmt_int(delta)))
    if client_header_with_sale:
        subtotal = sum(v for _,v,_ in client_header_with_sale[:30])
        block.append('<p><strong>CLIENT_HEADER с суммой (первые 30):</strong> subtotal=%s</p>' % fmt_int(subtotal))
        block.append('<ol>' + _li([(rn,val,txt) for rn,val,txt in client_header_with_sale], 30) + '</ol>')
    if product_no_prod_like:
        subtotal = sum(v for _,v,_ in product_no_prod_like[:30])
        block.append('<p><strong>PRODUCT без признаков товара (первые 30):</strong> subtotal=%s</p>' % fmt_int(subtotal))
        block.append('<ol>' + _li([(rn,val,txt) for rn,val,txt in product_no_prod_like], 30) + '</ol>')
    if picked:
        block.append('<p><strong>Кандидаты, покрывающие Δ ( жадно):</strong> subtotal=%s (из %d строк)</p>' % (fmt_int(sum(v for v,_,_ in picked)), len(picked)))
        block.append('<ol>' + "\n".join(f'<li>pick#%02d row=%d sale=%s %s</li>' % (k, rn, fmt_int(val), _html.escape(txt)) for k,(val, rn, txt) in enumerate(picked, 1)) + '</ol>')
    block.append('</details>')
    return "\n".join(block)
# ──────────────────────────────────────────────────────────────────────────────
# Jinja2
def build_env() -> Environment:
    env = Environment(loader=FileSystemLoader([str(TPL)]), autoescape=select_autoescape(['html']))
    env.filters["money"] = fmt_int
    env.filters["qty"]   = fmt_qty3
    env.filters["price"] = fmt_int
    return env

def render_products(env: Environment, ctx: Dict[str,Any], stem: str) -> Path:
    tpl = env.get_template("sales_by_product.html")
    html = tpl.render(**ctx)
    out = OUT / f"sales_products_{stem}.html"
    out.write_text(html, encoding="utf-8")
    return out

def render_grouped(env: Environment, ctx: Dict[str,Any], stem: str, forensic_html: Optional[str]=None) -> Path:
    tpl = env.get_template("sales_report_grouped.html")
    html = tpl.render(**ctx)
    if forensic_html:
        m = re.search(r'(<p class="warn">[^<]*Расхождение \([^)]*\):[^<]*</p>)', html, flags=re.I)
        if m:
            insert_at = m.end()
            html = html[:insert_at] + "\n" + forensic_html + html[insert_at:]
        else:
            html = html.replace("</body>", forensic_html + "\n</body>")
    out = OUT / f"sales_grouped_{stem}.html"
    out.write_text(html, encoding="utf-8")
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Контексты и сборка
def build_product_mode_ctx(products: List[Dict[str,Any]], turnover: float, table_total: Optional[int], excel_total: Optional[int], period: str, manager: str) -> Dict[str,Any]:
    # Очистка периода и менеджера от лишних данных
    period = re.split(r'[|;]', period)[0].strip() if period else "Не указан"
    manager = re.split(r'[|;]', manager)[0].strip() if manager else "Не указан"
    
    grand_i = round_half_up_to_int(turnover)
    ctx = {
        "title": f"Отчёт по продажам — {period}",
        "period": period,
        "manager": manager,
        "client_count": None,
        "grand_total": fmt_int(grand_i),
        "excel_total": fmt_int(excel_total) if excel_total is not None else None,
        "delta": None if table_total is None else fmt_int(grand_i - round_half_up_to_int(table_total)),
        "generated_at": datetime.now(TZ).strftime("%d.%m.%Y %H:%M"),
        "products": [
            {
                "product": p["product"],
                "qty_raw": p.get("qty_raw", 0.0),
                "price_raw": p.get("price_raw", 0.0),
                "sale_raw": p["sale_raw"],
            } for p in products
        ],
        "table_delta_line": "" if table_total is None else (f"Расхождение (табл): {fmt_int(round_half_up_to_int(table_total) - grand_i)}" if abs(grand_i - round_half_up_to_int(table_total)) > DELTA_TOL else "")
    }
    return ctx

def build_grouped_mode_ctx(clients: List[Dict[str,Any]], turnover: float, table_total: Optional[int], excel_total: Optional[int], period: str, manager: str) -> Dict[str,Any]:
    # Очистка периода и менеджера от лишних данных
    period = re.split(r'[|;]', period)[0].strip() if period else "Не указан"
    manager = re.split(r'[|;]', manager)[0].strip() if manager else "Не указан"
    
    grand_i = round_half_up_to_int(turnover)
    ctx = {
        "title": f"Отчёт по продажам — {period}",
        "period": period,
        "manager": manager,
        "client_count": len(clients),
        "grand_total": fmt_int(grand_i),
        "excel_total": fmt_int(excel_total) if excel_total is not None else None,
        "delta": None if table_total is None else fmt_int(grand_i - round_half_up_to_int(table_total)),
        "generated_at": datetime.now(TZ).strftime("%d.%m.%Y %H:%M"),
        "clients": [
            {
                "client": c["client"],
                "computed": fmt_int(c["computed_i"]),
                "computed_i": c["computed_i"],
                "qty_sum": c.get("qty_sum_i", 0.0),
                "products": [
                    {
                        "product": p["product"],
                        "qty_raw": p["qty_raw"],
                        "price_raw": p["price_raw"],
                        "sale_raw": p["sale_raw"],
                    } for p in c["products"]
                ]
            } for c in clients
        ],
        "table_delta_line": "" if table_total is None else (f"Расхождение (табл): {fmt_int(round_half_up_to_int(table_total) - grand_i)}" if abs(grand_i - round_half_up_to_int(table_total)) > DELTA_TOL else "")
    }
    return ctx

def make_stem(xlsx: Path) -> str:
    return xlsx.stem  # как в ваших логах

def build_report(xlsx: Path, write_suspect_html: bool=False) -> Tuple[Path, Optional[Path]]:
    env = build_env()
    raw = read_excel_raw(xlsx)
    colmap, data_start, _comb = find_header_block(raw)
    period, manager = extract_meta(raw, data_start)
    if "product" not in colmap or "sale" not in colmap:
        missing = [k for k in ("product","sale") if k not in colmap]
        raise ValueError(f"Не найдены обязательные колонки: {', '.join(missing)}")
    data_end = find_data_end(raw, data_start, colmap)
    df = raw.iloc[data_start:data_end+1].reset_index(drop=True).copy()

    # режим: grouped если явно распознаны клиентские шапки
    ch = 0
    sample = df.head(400).fillna("")
    for _, r in sample.iterrows():
        if classify_row(r.tolist(), colmap) == "CLIENT_HEADER":
            ch += 1
            if ch >= 3: break
    mode = "grouped" if ch >= 3 or ("client" in colmap) else "product"

    stem = make_stem(xlsx)

    if mode == "product":
        products, turnover, suspects = build_product_mode(df, colmap)
        table_total = table_grand_total_from_body(df, colmap)
        excel_total, _excel_src = extract_excel_total_from_meta(raw, data_start, data_end, round_half_up_to_int(turnover))
        ctx = build_product_mode_ctx(products, turnover, table_total, excel_total, period, manager)
        out = render_products(env, ctx, stem)
        suspect_html = None
        if write_suspect_html and suspects:
            suspect_html = write_suspects_html(env, suspects, stem)
        return out, suspect_html

    clients, turnover, suspects = build_client_mode(df, colmap)
    table_total = table_grand_total_from_body(df, colmap)
    excel_total, _excel_src = extract_excel_total_from_meta(raw, data_start, data_end, round_half_up_to_int(turnover))

    grand_i = round_half_up_to_int(turnover)
    forensic_html = ""
    if table_total is not None and abs(grand_i - round_half_up_to_int(table_total)) > DELTA_TOL:
        log_delta_forensic(df, colmap, grand_i, round_half_up_to_int(table_total))
        forensic_html = build_delta_forensic_html(df, colmap, grand_i, round_half_up_to_int(table_total))

    ctx = build_grouped_mode_ctx(clients, turnover, table_total, excel_total, period, manager)
    out = render_grouped(env, ctx, stem, forensic_html)
    suspect_html = None
    if write_suspect_html and suspects:
        suspect_html = write_suspects_html(env, suspects, stem)
    return out, suspect_html

# ──────────────────────────────────────────────────────────────────────────────
# Suspects HTML
def write_suspects_html(env: Environment, suspects: List[Dict[str,Any]], stem: str) -> Path:
    html = [
        "<html><head><meta charset='utf-8'><title>Suspects</title>",
        "<style>body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif} table{border-collapse:collapse;width:100%} td,th{border:1px solid #ddd;padding:6px} tr:nth-child(even){background:#fafafa} .num{text-align:right}</style>",
        "</head><body>",
        f"<h2>Подозрительные строки — {stem}.xlsx</h2>",
        "<table><tr><th>#</th><th>Строка</th><th>Товар/Клиент</th><th class='num'>Сумма</th><th>Причина</th></tr>"
    ]
    for k, s in enumerate(suspects, 1):
        html.append(f"<tr><td>{k}</td><td>{s.get('row','')}</td><td>{_html.escape(str(s.get('text','')))}</td><td class='num'>{fmt_int(s.get('sale',0))}</td><td>{s.get('reason','')}</td></tr>")
    html.append("</table></body></html>")
    out = OUT / "suspects" / f"suspects_{stem}.html"
    out.write_text("\n".join(html), encoding="utf-8")
    return out

# ──────────────────────────────────────────────────────────────────────────────
# CLI
def main(argv: Optional[List[str]] = None) -> int:
    _ensure_file_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", nargs="+", help="Excel-файлы отчётов продаж")
    ap.add_argument("--suspect", action="store_true", help="Сохранить HTML со списком подозрительных строк")
    args = ap.parse_args(argv)

    rc = 0
    for f in args.xlsx:
        try:
            path = Path(f)
            LOG.info("Файл: %s", path)
            out, suspect = build_report(path, write_suspect_html=args.suspect)
            LOG.info("HTML: %s", out)
            if suspect:
                LOG.info("Suspects: %s", suspect)
        except Exception as e:
            LOG.error("FAILED on %s", f, exc_info=e)
            rc = 2
    LOG.info("done.")
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
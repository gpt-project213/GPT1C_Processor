#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extended_debt_report.py — расширенный отчёт по дебиторской задолженности (1С)
version: v1.4 (2025-09-02)

СТРОГО ПО ТЗ:
• Таймзона: Asia/Almaty; логи: logs/<module>_YYYYMMDD_HHMMSS.log
  формат: "%(asctime)s, %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
  → через config.setup_logging(), лог в файл и дублирование в консоль.
• Выходные пути:
  HTML → reports/html/debt_ext_{stem}.html
  JSON → reports/json/debt_ext_{stem}.json
• Футер HTML: "Сформировано: DD.MM.YYYY HH:MM (Asia/Almaty) | Версии: …"
• Бизнес-логика парсинга/агрегации НЕ меняется (шапка, фильтры, нормализация).

CLI:
    python extended_debt_report.py <xlsx> [--force-fix]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

# ── проектные инварианты и каталоги
import config
from config import HTML_DIR, JSON_DIR, setup_logging, generated_at_tz, register_version, get_versions_line

# ── utils: чистка xlsx (НЕ меняем бизнес-логику)
from utils_excel import ensure_clean_xlsx  # сигнатура как в твоём коде (поддерживает force_fix)

__VERSION__ = "extended_debt_report.py v1.4 — 2025-09-02"
log = setup_logging("extended_debt_report", level=logging.INFO)
register_version("extended_debt_report", "v1.4")

NBSP = "\u202f"  # тонкий пробел (для форматирования денег)

def df_map(df: pd.DataFrame, func):
    """
    Элементное преобразование DataFrame без FutureWarning.
    На новых pandas использует DataFrame.map, на старых — applymap.
    (Зарезервировано для возможных мест, где была applymap — сейчас не требуется.)
    """
    if hasattr(pd.DataFrame, "map"):   # pandas 2.2+
        return df.map(func)
    return df.applymap(func)           # pandas <=2.1

# ──────────────────────────────── Regex (как в твоём исходнике)
RE_CLIENT_EQ  = re.compile(r"^\s*контрагент\s*$", re.I)
RE_CLIENT_ANY = re.compile(r"(контрагент|покупатель|клиент)", re.I)

RE_OPEN   = re.compile(r"((^|\b)нач(\.|альный)?\s*остат(ок)?\b|сальдо\s*на\s*начало)", re.I)
RE_END    = re.compile(r"((^|\b)кон(\.|ечный)?\s*остат(ок)?\b|сальдо\s*на\s*конец)", re.I)
RE_DEBIT  = re.compile(r"(приход|отгр\w*|зачисл\w*|дебет|поставка|поступило)", re.I)
RE_CREDIT = re.compile(r"(расход|оплат\w*|списан\w*|кредит|оплата)", re.I)
RE_DOC    = re.compile(r"^(документ|основание|коммент|комментарий|описание)\b", re.I)

RE_DATE_CELL  = re.compile(r"^\s*\d{2}\.\d{2}\.\d{4}\s*$")
RE_META_CELL  = re.compile(r"(показатели|группировк|отбор|дополнительные\s*поля|сортировка)", re.I)
RE_TOTAL_CELL = re.compile(r"(итоговая\s*по\s*всем\s*покуп|^покупатели$)", re.I)
RE_LEVEL_CELL = re.compile(r"\bитог\b", re.I)
RE_ONLY_NUM   = re.compile(r"^[\d\s,\u202f]+$")
RE_DOT_SUFFIX = re.compile(r"\.\d+$")
META_SUBGROUPS_RE = re.compile(r"в\s*групп[еы][^()]*\(([^)]*)\)", re.I)

# Доп. регэкспы для детекта «простого/расширенного»
DATE_PAIR_RGX = re.compile(r"(\d{2}[./]\d{2}[./]\d{4}).{0,15}(\d{2}[./]\d{2}[./]\d{4})")
DATE_RGX      = re.compile(r"\b(\d{2}\.\d{2}\.\d{4})\b")

# ──────────────────────────────── helpers (как у тебя, без изменений)
def _row_vals(raw: pd.DataFrame, i: int) -> list[str]:
    return [str(x).strip() if pd.notna(x) else "" for x in raw.iloc[i].tolist()]

def _row_text(raw: pd.DataFrame, i: int) -> str:
    return " | ".join(_row_vals(raw, i)).lower()

def find_header(raw: pd.DataFrame) -> list[int]:
    keys = ["нач. остаток", "приход", "расход", "кон. остаток"]
    limit = min(120, len(raw) - 1)
    for i in range(limit):
        if any(RE_CLIENT_EQ.match(c) for c in _row_vals(raw, i)):
            r1 = [c.lower() for c in _row_vals(raw, i + 1)]
            if sum(1 for k in keys if any(k in c for c in r1)) >= 2:
                return [i, i + 1]
    for i in range(limit):
        cells_i = _row_vals(raw, i)
        if RE_CLIENT_ANY.search(" | ".join(cells_i)):
            r1 = [c.lower() for c in _row_vals(raw, i + 1)]
            if sum(1 for k in keys if any(k in c for c in r1)) >= 2:
                return [i, i + 1]
    raise ValueError("Не найдена строка шапки")

def clean_header_cell(s: str) -> str:
    s0 = str(s).replace("\r", "\n")
    s0 = re.sub(r"(Отборы:|Дополнительные\s*поля:|Сортировка:|Показатели:).*",
                "", s0, flags=re.I | re.S).strip()
    s0 = re.sub(r"[\n\t]+", " ", s0)
    s0 = re.sub(r"\s{2,}", " ", s0).strip()
    s0 = RE_DOT_SUFFIX.sub("", s0)
    return s0

def canon_from_joined(joined: str) -> str:
    low = joined.lower()
    if RE_CLIENT_EQ.search(low) or RE_CLIENT_ANY.search(low):
        return "Контрагент"
    if RE_OPEN.search(low):   return "нач. остаток"
    if RE_DEBIT.search(low):  return "приход"
    if RE_CREDIT.search(low): return "расход"
    if RE_END.search(low):    return "кон. остаток"
    if RE_DOC.search(low):    return "Документ"
    return ""

def build_names_from_header(raw: pd.DataFrame, header_rows: list[int]) -> list[str]:
    cells = raw.iloc[header_rows].astype(str).fillna("")
    names = []
    for col in range(cells.shape[1]):
        parts = [clean_header_cell(cells.iat[r, col]) for r in range(len(header_rows))]
        parts = [p for p in parts if p and not RE_META_CELL.search(p)]
        joined = " ".join(parts).strip()
        name = canon_from_joined(joined)
        if not name:
            for p in reversed(parts):
                lab = canon_from_joined(p)
                if lab:
                    name = lab; break
        if not name:
            short = [p for p in parts if len(p) <= 40]
            name = short[-1] if short else (parts[-1] if parts else "")
        names.append(name if name else f"col{col}")
    seen, uniq = {}, []
    for nm in names:
        if nm not in seen:
            seen[nm] = 0; uniq.append(nm)
        else:
            seen[nm] += 1; uniq.append(f"{nm}.{seen[nm]}")
    return uniq

def extract_subgroups_from_meta(raw: pd.DataFrame, header_rows: list[int]) -> List[str]:
    if not header_rows: return []
    hi = min(header_rows)
    found: List[str] = []
    for i in range(hi):
        txt = " ".join(_row_vals(raw, i))
        m = META_SUBGROUPS_RE.search(txt)
        if m:
            found.extend(p.strip() for p in re.split(r"[;,]", m.group(1)) if p.strip())
    seen, out = set(), []
    for x in found:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def parse_date_cell(v: Any) -> pd.Timestamp:
    if v is None or (isinstance(v, float) and np.isnan(v)): return pd.NaT
    s = str(v).strip()
    if not re.match(r"^\d{2}\.\d{2}\.\d{4}$", s): return pd.NaT
    return pd.to_datetime(s, format="%d.%m.%Y", errors="coerce")

def money_to_float(v: Any) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)): return 0.0
    s = str(v).strip().replace("\u00A0", "").replace("\u202f", "").replace(" ", "")
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in ("", "-", "."): return 0.0
    try: return float(s)
    except Exception: return 0.0

def fmt_money(v: Optional[float]) -> str:
    if v is None: v = 0.0
    return f"{int(round(v)):,}".replace(",", NBSP) + " тг"

def fmt_date(ts: Optional[pd.Timestamp]) -> str:
    return "" if ts is None or pd.isna(ts) else ts.strftime("%d.%m.%Y")

def escape_html(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ──────────────────────────────── dataclasses (без изменений)
@dataclass
class Movement:
    date: pd.Timestamp
    debit: float = 0.0
    credit: float = 0.0

@dataclass
class ClientBlock:
    subgroup: str
    client: str
    opening: Optional[float] = None
    closing: Optional[float] = None
    movements: List[Movement] = field(default_factory=list)
    last_date: Optional[pd.Timestamp] = None

    @property
    def sum_debit(self) -> float:
        return float(sum(m.debit for m in self.movements))

    @property
    def sum_credit(self) -> float:
        return float(sum(m.credit for m in self.movements))

    def compute(self) -> Tuple[float, float]:
        if self.opening is None: self.opening = 0.0
        calc_closing = (self.opening or 0.0) + self.sum_debit - self.sum_credit
        if self.closing is None: self.closing = calc_closing
        delta = (self.closing or 0.0) - calc_closing
        return calc_closing, delta

    def adjustments(self) -> float:
        op = self.opening or 0.0
        cl = self.closing or 0.0
        return cl - op - self.sum_debit + self.sum_credit

# ──────────────────────────────── агрегаты (без изменений)
def aggregate(blocks: List[ClientBlock]) -> Dict[str, Any]:
    total_open = total_debit = total_credit = total_adj = total_close = total_delta = 0.0
    n_clients = 0
    for b in blocks:
        n_clients += 1
        op = b.opening or 0.0
        db = b.sum_debit
        cr = b.sum_credit
        cl_calc, delta = b.compute()
        cl = b.closing if b.closing is not None else cl_calc
        adj = b.adjustments()
        total_open += op
        total_debit += db
        total_credit += cr
        total_adj += adj
        total_close += cl
        total_delta += delta
    trend = "Без изменений"
    if total_close < total_open: trend = "Прогресс"
    elif total_close > total_open: trend = "Регресс"
    return dict(
        n_clients=n_clients,
        open=total_open,
        debit=total_debit,
        credit=total_credit,
        adj=total_adj,
        close=total_close,
        delta=total_delta,
        trend=trend,
        delta_allow=n_clients,
    )

# ──────────────────────────────── основной парсер (без изменений)
def parse_excel(path: Path) -> Tuple[List[ClientBlock], Optional[pd.Timestamp], Optional[pd.Timestamp], Dict]:
    raw = pd.read_excel(path, header=None, dtype=str, keep_default_na=False)
    header_rows = find_header(raw)
    subgroups = extract_subgroups_from_meta(raw, header_rows)

    df = pd.read_excel(path, header=header_rows, dtype=str, keep_default_na=False)
    df.columns = build_names_from_header(raw, header_rows)[: df.shape[1]]

    name_client  = next(c for c in df.columns if RE_CLIENT_ANY.search(c.lower()))
    name_opening = next(c for c in df.columns if RE_OPEN.search(c.lower()))
    name_debit   = next(c for c in df.columns if RE_DEBIT.search(c.lower()))
    name_credit  = next(c for c in df.columns if RE_CREDIT.search(c.lower()))
    name_closing = next(c for c in df.columns if RE_END.search(c.lower()))

    dates_ser  = df[name_client].apply(parse_date_cell)
    period_min = dates_ser.dropna().min() if not dates_ser.empty else None
    period_max = dates_ser.dropna().max() if not dates_ser.empty else None

    df["_open"]   = df[name_opening].map(money_to_float)
    df["_debit"]  = df[name_debit].map(money_to_float)
    df["_credit"] = df[name_credit].map(money_to_float)
    df["_close"]  = df[name_closing].map(money_to_float)

    subs_set = set(subgroups)
    types = []
    for s in df[name_client].astype(str).fillna(""):
        ss = s.strip()
        if ss == "" or ss.lower() == "nan":
            types.append("EMPTY")
        elif ss in subs_set:
            types.append("SUBGR")
        elif RE_DATE_CELL.match(ss):
            types.append("DATE")
        elif RE_TOTAL_CELL.search(ss):
            types.append("TOTAL")
        elif RE_META_CELL.search(ss):
            types.append("META")
        elif RE_LEVEL_CELL.search(ss):
            types.append("LEVEL")
        else:
            types.append("CLIENT")
    df["_type"] = types
    df["_date"] = df[name_client].apply(parse_date_cell)

    blocks: List[ClientBlock] = []
    cur_subgroup = subgroups[0] if subgroups else ""
    cur_client: Optional[ClientBlock] = None

    for _, row in df.iterrows():
        t = row["_type"]
        if t == "SUBGR":
            cur_subgroup = str(row[name_client]).strip()
            cur_client = None
            continue
        if t == "CLIENT":
            title = str(row[name_client]).strip()
            if title.lower() == "покупатели":
                cur_client = None
                continue
            cur_client = ClientBlock(subgroup=cur_subgroup, client=title)
            op = money_to_float(row["_open"]); cl = money_to_float(row["_close"])
            if op != 0.0: cur_client.opening = op
            if cl != 0.0: cur_client.closing = cl
            blocks.append(cur_client); continue
        if t == "DATE":
            if cur_client is None: continue
            dts = row["_date"]; db = money_to_float(row["_debit"]); cr = money_to_float(row["_credit"])
            cur_client.movements.append(Movement(date=dts, debit=db, credit=cr))
            cur_client.last_date = dts; continue

    agg = aggregate(blocks)
    return blocks, period_min, period_max, agg

# ──────────────────────────────── HTML-рендер (ВИД не меняем, добавлен футер по ТЗ)
def build_html(blocks: List[ClientBlock],
               period_min: Optional[pd.Timestamp],
               period_max: Optional[pd.Timestamp],
               agg: Dict) -> str:
    by_sub: Dict[str, List[ClientBlock]] = {}
    for b in blocks:
        by_sub.setdefault(b.subgroup or "Без подгруппы", []).append(b)

    css = """
    <style>
      body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;color:#111;margin:16px;}
      h1{font-size:20px;margin:0;}
      .muted{color:#666;font-size:12px;}
      .money{text-align:right;font-variant-numeric:tabular-nums;}
      .pill{display:inline-block;padding:2px 6px;border-radius:9999px;background:#f2f2f2;font-size:11px;}
      .sub{margin:18px 0 12px;font-weight:600;font-size:16px;}
      .card{border:1px solid #e6e6e6;border-radius:12px;padding:12px;margin:10px 0;}
      .grid{display:grid;grid-template-columns:1fr auto auto auto auto auto;gap:6px;align-items:center;}
      table{width:100%;border-collapse:collapse;margin-top:6px;}
      th,td{border-top:1px dashed #ccc;padding:4px 6px;font-size:12px;}
      .delta-ok{color:#0a7;} .delta-bad{color:#c33;font-weight:700;}
      footer{margin-top:16px;color:#666;font-size:12px;}
    </style>
    """
    period = " — ".join(filter(None, (fmt_date(period_min), fmt_date(period_max))))
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Дебиторка</title>",
        css,
        "</head><body>",
        "<h1>Расширенный отчёт по дебиторской задолженности</h1>",
        f"<div class='muted'>Период: {period}</div>",
        f"<div class='muted'>Версия: {__VERSION__}</div><hr>",
        "<h3>Сводка</h3>",
        "<table>",
        f"<tr><td class='muted'>Клиентов</td><td class='money'>{agg['n_clients']}</td>"
        f"<td class='muted'>Нач. остаток</td><td class='money'>{fmt_money(agg['open'])}</td>"
        f"<td class='muted'>Кон. остаток</td><td class='money'>{fmt_money(agg['close'])}</td></tr>",
        "</table>",
    ]
    for sub, cl_list in by_sub.items():
        parts.append(f"<div class='sub'>{escape_html(sub)}</div>")
        for b in cl_list:
            calc_close, δ = b.compute()
            δok = abs(δ) <= 1
            parts.append("<div class='card'>")
            parts.append(
                f"<div class='grid'><div class='client-name'>{escape_html(b.client)}</div>"
                f"<div class='muted'>Отгр.</div><div class='money'>{fmt_money(b.sum_debit)}</div>"
                f"<div class='muted'>Оплата</div><div class='money'>{fmt_money(b.sum_credit)}</div></div>"
            )
            parts.append(
                f"<div class='grid' style='margin-top:4px;'>"
                f"<div class='muted'>Нач. остаток</div><div class='money'>{fmt_money(b.opening)}</div>"
                f"<div class='muted'>Кон. остаток</div><div class='money'>{fmt_money(b.closing)}</div>"
                f"<div class='muted'>Δ</div><div class='money {'delta-ok' if δok else 'delta-bad'}'>{fmt_money(δ)}</div></div>"
            )
            if b.movements:
                parts.append("<table><tr><th>Дата</th><th class='money'>Отгр.</th><th class='money'>Оплата</th></tr>")
                for m in sorted(b.movements, key=lambda x: x.date):
                    parts.append(
                        f"<tr><td>{fmt_date(m.date)}</td><td class='money'>{fmt_money(m.debit)}</td>"
                        f"<td class='money'>{fmt_money(m.credit)}</td></tr>"
                    )
                parts.append("</table>")
            parts.append("</div>")  # card
    # Футер по ТЗ
    parts.append("<footer>")
    parts.append(escape_html(generated_at_tz()) + escape_html(get_versions_line()))
    parts.append("</footer>")
    parts.append("</body></html>")
    return "\n".join(parts)

# ──────────────────────────────── JSON helper (без изменений)
def blocks_to_json(blocks: List[ClientBlock]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for b in blocks:
        out.append(
            dict(
                subgroup=b.subgroup,
                client=b.client,
                opening=b.opening or 0.0,
                closing=b.closing or 0.0,
                debit=b.sum_debit,
                credit=b.sum_credit,
                last_date=fmt_date(b.last_date),
            )
        )
    return out

# ──────────────────────────────── детект «простой» (как у тебя)
def _is_simple_by_rules(clean_xlsx: Path) -> bool:
    try:
        head = pd.read_excel(clean_xlsx, header=None, dtype=str, nrows=30, keep_default_na=False)
        lines = [" ".join([str(x).strip() for x in row if str(x).strip()]) for row in head.values]
        if any(DATE_PAIR_RGX.search(line) for line in lines):
            return False  # нашли период → расширенный
        body = pd.read_excel(clean_xlsx, header=None, dtype=str, nrows=150, keep_default_na=False)
        flat = " ".join(" ".join(map(str, row)) for row in body.values)
        total_dates = len(DATE_RGX.findall(flat))
        return total_dates < 15
    except Exception:
        return False

# ──────────────────────────────── main pipeline
def build_report(src_xlsx: Path, force_fix: bool = False) -> Path:
    log.info("SRC %s", src_xlsx.name)
    clean = ensure_clean_xlsx(src_xlsx, force_fix=force_fix)  # сигнатуру НЕ меняю
    log.info("Clean copy → %s", clean.name)

    # как у тебя: если это «простой», не читаем его расширенным
    if _is_simple_by_rules(clean):
        msg = ("Обнаружен ПРОСТОЙ отчёт (нет детализации по датам / период на одну дату). "
               "Запустите вручную: python debt_report.py <файл>")
        log.error(msg)
        raise SystemExit(4)

    blocks, pmin, pmax, agg = parse_excel(clean)

    # HTML → reports/html, имя по ТЗ
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    out_html = HTML_DIR / f"debt_ext_{src_xlsx.stem}.html"
    html_text = build_html(blocks, pmin, pmax, agg)
    out_html.write_text(html_text, encoding="utf-8")
    log.info("HTML → %s", out_html.name)
    log.info("✔ HTML: %s", str(out_html.resolve()))

    # JSON → reports/json, имя по ТЗ
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    json_obj = dict(
        period_min=fmt_date(pmin),
        period_max=fmt_date(pmax),
        agg=agg,
        blocks=blocks_to_json(blocks),
        generated_at=generated_at_tz(),
        version=__VERSION__,
    )
    out_json = JSON_DIR / f"debt_ext_{src_xlsx.stem}.json"
    out_json.write_text(json.dumps(json_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("JSON → %s", out_json.name)
    log.info("✔ JSON: %s", str(out_json.resolve()))

    print("✅ HTML и JSON сохранены")
    return out_html

def main() -> None:
    ap = argparse.ArgumentParser(description="Расширенный отчёт по дебиторке 1С (строго по ТЗ)")
    ap.add_argument("xlsx", help="Путь к исходному XLSX-файлу отчёта 1С")
    ap.add_argument("--force-fix", action="store_true", help="Пересоздать .__clean.xlsx принудительно")
    args = ap.parse_args()

    src = Path(args.xlsx).expanduser().resolve()
    if not src.is_file():
        print(f"❌ Файл не найден: {src}")
        raise SystemExit(1)

    try:
        build_report(src, force_fix=args.force_fix)
    except SystemExit:
        raise
    except Exception as e:
        log.exception("UNCAUGHT EXCEPTION")
        print(f"❌ ERROR: {e}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()

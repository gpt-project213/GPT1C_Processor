#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
debt_auto_report.py · v2.7.3 · 2026-03-10
Правки: simple → убраны «Отгрузка/Оплата» во «Все клиенты»; extended → агрегаты в шапку,
Δ (увеличение/уменьшение), сортировка «Движения» по убыванию closing, техданные без «Клиентов».

Совместимость: Python 3.11+/3.12

v2.7.3: Fix P-005: удалены дублированные RE_DATE_CELL и RE_MANAGER_FILTER (строки 55-56)
v2.7.2: Удалён "Арман" из INTERNAL_UNITS_MAP (уволен)
        Удалён захардкоженный fallback-список менеджеров (line 403)
        Деdup money(): удалена локальная функция, импорт из utils.py
"""

from __future__ import annotations

import sys, re, json, logging, argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from zoneinfo import ZoneInfo

# Внешние модули проекта
import config
from utils_excel import ensure_clean_xlsx
from utils import money
from analyze_debt_excel import parse_debt_report

__VERSION__ = "debt_auto=v2.7.3"
NBSP = "\u202f"

log = getattr(config, "setup_logging", lambda name: logging.getLogger(name))("debt_auto_report")

# ───── Регулярки/константы
DATE_RGX = re.compile(r"\b\d{2}[./]\d{2}[./]\d{4}\b")
DATEPAIR_RGX = re.compile(r"(\d{2}[./]\d{2}[./]\d{4}).{0,15}(\d{2}[./]\d{2}[./]\d{4})")
RE_CLIENT_ANY = re.compile(r"(контрагент|покупатель|клиент)", re.I)
RE_CLIENT_EQ = re.compile(r"^\s*контрагент\s*$", re.I)
RE_OPEN   = re.compile(r"((^|\b)нач(\.|альный)?\s*остат(ок)?\b|сальдо\s*на\s*начало)", re.I)
RE_END    = re.compile(r"((^|\b)кон(\.|ечный)?\s*остат(ок)?\b|сальдо\s*на\s*конец)", re.I)
RE_DEBIT  = re.compile(r"(приход|отгр\w*|зачисл\w*|дебет|поставка|поступило)", re.I)
RE_CREDIT = re.compile(r"(расход|оплат\w*|списан\w*|кредит|оплата)", re.I)
RE_META_CELL  = re.compile(r"(показатели|группировк|отбор|дополнительные\s*поля|сортировка)", re.I)
RE_TOTAL_CELL = re.compile(r"(итогова|итог|^покупатели$|покупатели\s*-\s*работники)", re.I)
RE_LEVEL_CELL = re.compile(r"\bитог\b", re.I)
RE_DOT_SUFFIX = re.compile(r"\.\d+$")
RE_DATE_CELL  = re.compile(r"^\s*\d{2}\.\d{2}\.\d{4}\s*$")
RE_MANAGER_FILTER = re.compile(r"контрагент\s+в\s+группе\s+из\s+списка\s*\(([^)]+)\)", re.I)

# ВНУТРЕННИЕ ПОДРАЗДЕЛЕНИЯ ПО МЕНЕДЖЕРАМ (дополняйте при необходимости)
INTERNAL_UNITS_MAP: dict[str, list[str]] = {
    "Магира": ["Магира ОПТ"],
    "Оксана": ["Оксана ОПТ"],
    "Алена":  ["Алена ОПТ"],
    "Ергали": ["Ергали ОПТ"],
}

def slugify(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"[\s/]+", "_", s).replace("ё", "e")
    s = re.sub(r"[^0-9a-zA-Z_\-]+", "_", s)
    return re.sub(r"__+", "_", s).strip("_") or "row"

# ───── Детект типа
def detect_report_type(xlsx_path: Path) -> str:
    try:
        head = pd.read_excel(xlsx_path, nrows=60, dtype=str).fillna("")
        for line in head.astype(str).agg(" ".join, axis=1).tolist()[:25]:
            if DATEPAIR_RGX.search(line):
                return "extended"
        col0 = head.iloc[:, 0].astype(str).tolist()
        if sum(1 for v in col0 if DATE_RGX.fullmatch(v.strip())) >= 8:
            return "extended"
        body = pd.read_excel(xlsx_path, header=None, dtype=str, nrows=150, keep_default_na=False)
        if len(DATE_RGX.findall(" ".join(" ".join(map(str, r)) for r in body.values))) >= 15:
            return "extended"
    except Exception as e:
        log.warning("Ошибка детекта типа: %s", e)
    return "simple"

# ───── Simple
def _unwrap_parse_result(res: Any) -> Tuple[pd.DataFrame, List[dict]]:
    if isinstance(res, dict):
        df = res.get("df") or res.get("data") or res.get("table")
        return df, (res.get("errors") or [])
    if isinstance(res, (list, tuple)) and len(res) >= 1:
        return res[0], (res[1] if len(res) >= 2 else [])
    raise ValueError("Неизвестный формат результата parse_debt_report")

def _find_col(cols: list[str], *cands: str) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for k in cands:
        if k.lower() in low: return low[k.lower()]
    for c in cols:
        for k in cands:
            if k.lower() in c.lower(): return c
    return None

def process_simple_report(clean_xlsx: Path, src_name: str) -> Dict[str, Any]:
    df_raw = parse_debt_report(clean_xlsx)
    df, errors = _unwrap_parse_result(df_raw)

    period, manager, managers_from_filter = _extract_header_info(clean_xlsx)
    df_filtered, dropped = _filter_out_managers(
        df, manager_hint=(manager if manager and manager != "—" else None),
        additional_managers=managers_from_filter or [],
    )

    cols = list(df_filtered.columns)
    c_client = _find_col(cols, "Клиент", "Контрагент", "client", "контрагент/клиент") or cols[0]
    c_final  = _find_col(cols, "кон", "кон. остаток", "сальдо кон", "задолженность", "конечный")

    client_count = int(df_filtered[c_client].astype(str).str.strip().nunique()) if c_client else int(len(df_filtered))

    total_debt = 0.0
    if c_final:
        total_debt = float(pd.to_numeric(df_filtered[c_final], errors="coerce").fillna(0).clip(lower=0).sum())

    top, silent, closed, overpay = _build_rows(df_filtered, total_debt=total_debt)

    # Все клиенты (только Клиент / Долг / Доля, %)
    all_rows: List[dict] = []
    if c_final and c_client:
        tmp = df_filtered.copy()
        tmp[c_final] = pd.to_numeric(tmp[c_final], errors="coerce").fillna(0.0)
        tmp = tmp.sort_values(c_final, ascending=False)
        for r in tmp.to_dict("records"):
            debt = float(r.get(c_final, 0.0) or 0.0)
            pct  = (debt / total_debt * 100.0) if (total_debt > 0 and debt > 0) else None
            all_rows.append({
                "client": r.get(c_client, ""),
                "client_slug": slugify(r.get(c_client, "")),
                "debt": debt,
                "pct": pct,
            })

    return {
        "title": "ОТЧЁТ ПО ДЕБИТОРСКОЙ ЗАДОЛЖЕННОСТИ",
        "period": period,
        "manager": manager,
        "client_count": client_count,
        "total_debt": total_debt,
        "top_debtors": top,
        "silent_rows": silent,   # по факту для simple обычно пусто — вкладка спрячется
        "closed_rows": closed,
        "overpay_rows": overpay,
        "all_rows": all_rows,
        "tech_info": {
            "Источник": src_name,
            "Clean": str(clean_xlsx.name),
            "Сброшено (менеджеры)": dropped,
            "Ошибки математики": len(errors),
            "Модуль": __VERSION__,
        },
        "report_type": "simple",
    }

# ───── Extended
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

def parse_extended_excel(path: Path, managers_list: List[str] | None = None
    ) -> Tuple[List[ClientBlock], Optional[pd.Timestamp], Optional[pd.Timestamp], Dict[str, Any]]:
    if managers_list is None: managers_list = []
    raw = pd.read_excel(path, header=None, dtype=str, keep_default_na=False)
    header_rows = find_header(raw)
    df = pd.read_excel(path, header=header_rows, dtype=str, keep_default_na=False)
    df.columns = build_names_from_header(raw, header_rows)[: df.shape[1]]

    def _reqcol(regex: re.Pattern) -> str:
        for c in df.columns:
            if regex.search(str(c).lower()):
                return c
        raise ValueError(f"Не найдена колонка {regex.pattern}")

    name_client = _reqcol(RE_CLIENT_ANY)
    name_opening = _reqcol(RE_OPEN)
    name_debit = _reqcol(RE_DEBIT)
    name_credit = _reqcol(RE_CREDIT)
    name_closing = _reqcol(RE_END)

    df["_open"]  = df[name_opening].map(money_to_float)
    df["_debit"] = df[name_debit].map(money_to_float)
    df["_credit"]= df[name_credit].map(money_to_float)
    df["_close"] = df[name_closing].map(money_to_float)
    df["_date"]  = df[name_client].apply(parse_date_cell)

    types: List[str] = []
    manager_patterns = _expand_variants(managers_list) if managers_list else []
    for s in df[name_client].astype(str).fillna(""):
        ss = s.strip()
        if not ss or ss.lower() == "nan": types.append("EMPTY")
        elif RE_DATE_CELL.match(ss):      types.append("DATE")
        elif RE_TOTAL_CELL.search(ss):    types.append("TOTAL")
        elif RE_LEVEL_CELL.search(ss):    types.append("LEVEL")
        elif any(p.fullmatch(ss) for p in manager_patterns): types.append("MANAGER")
        else:                              types.append("CLIENT")
    df["_type"] = types

    blocks: List[ClientBlock] = []
    cur_subgroup = "Общий"
    cur_client: Optional[ClientBlock] = None

    for _, row in df.iterrows():
        t = row["_type"]
        if t in ("MANAGER","EMPTY"): continue
        if t == "LEVEL":
            title = str(row[name_client]).strip()
            if title: cur_subgroup = title
            continue
        if t == "CLIENT":
            title = str(row[name_client]).strip()
            if any(p.fullmatch(title) for p in manager_patterns): continue
            cur_client = ClientBlock(subgroup=cur_subgroup, client=title)
            op = float(row["_open"]) if pd.notna(row["_open"]) else 0.0
            cl = float(row["_close"]) if pd.notna(row["_close"]) else 0.0
            if op != 0.0: cur_client.opening = op
            if cl != 0.0: cur_client.closing = cl
            blocks.append(cur_client)
            continue
        if t == "DATE" and cur_client:
            dts = row["_date"]
            db = float(row["_debit"]) if pd.notna(row["_debit"]) else 0.0
            cr = float(row["_credit"]) if pd.notna(row["_credit"]) else 0.0
            if not pd.isna(dts) and (db != 0.0 or cr != 0.0):
                cur_client.movements.append(Movement(date=dts, debit=db, credit=cr))
                cur_client.last_date = dts

    agg = aggregate_blocks(blocks)
    dates = [m.date for b in blocks for m in b.movements if not pd.isna(m.date)]
    period_min = min(dates) if dates else None
    period_max = max(dates) if dates else None
    return blocks, period_min, period_max, agg

def aggregate_blocks(blocks: List[ClientBlock]) -> Dict[str, Any]:
    total_open = total_debit = total_credit = total_close = 0.0
    for b in blocks:
        total_open  += b.opening or 0.0
        total_debit += b.sum_debit
        total_credit+= b.sum_credit
        total_close += b.closing or 0.0
    trend = "Без изменений"
    if total_close < total_open: trend = "Уменьшение"
    elif total_close > total_open: trend = "Увеличение"
    return {
        "n_clients": len(blocks),
        "open": total_open, "debit": total_debit, "credit": total_credit, "close": total_close,
        "trend": trend
    }

# ───── Шапки/утилиты
def find_header(raw: pd.DataFrame) -> list[int]:
    keys = ["нач. остаток", "приход", "расход", "кон. остаток"]
    limit = max(0, min(120, len(raw)-2))
    for i in range(limit):
        if any(RE_CLIENT_EQ.match(c) for c in _row_vals(raw, i)):
            r1 = [c.lower() for c in _row_vals(raw, i+1)]
            if sum(1 for k in keys if any(k in c for c in r1)) >= 2: return [i, i+1]
    for i in range(limit):
        cells_i = _row_vals(raw, i)
        if RE_CLIENT_ANY.search(" | ".join(cells_i)):
            r1 = [c.lower() for c in _row_vals(raw, i+1)]
            if sum(1 for k in keys if any(k in c for c in r1)) >= 2: return [i, i+1]
    return [0, 1]

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
                if lab: name = lab; break
        if not name:
            short = [p for p in parts if len(p) <= 40]
            name = short[-1] if short else (parts[-1] if parts else f"col{col}")
        names.append(name)
    seen, uniq = {}, []
    for nm in names:
        if nm not in seen:
            seen[nm] = 0; uniq.append(nm)
        else:
            seen[nm] += 1; uniq.append(f"{nm}.{seen[nm]}")
    return uniq

def money_to_float(v: Any) -> float:
    if v is None or (isinstance(v, float) and pd.isna(v)): return 0.0
    s = str(v).strip().replace("\u00A0","").replace("\u202f","").replace(" ","").replace(",",".")
    s = re.sub(r"[^0-9.\-]","", s)
    if s in ("","-","."): return 0.0
    try: return float(s)
    except Exception: return 0.0

def parse_date_cell(v: Any) -> pd.Timestamp:
    if v is None or (isinstance(v, float) and pd.isna(v)): return pd.NaT
    s = str(v).strip()
    if not re.match(r"^\d{2}\.\d{2}\.\d{4}$", s): return pd.NaT
    return pd.to_datetime(s, format="%d.%m.%Y", errors="coerce")

def fmt_date(ts: Optional[pd.Timestamp]) -> str:
    return "" if ts is None or pd.isna(ts) else ts.strftime("%d.%m.%Y")

def _row_vals(raw: pd.DataFrame, i: int) -> list[str]:
    return [str(x).strip() if pd.notna(x) else "" for x in raw.iloc[i].tolist()]

def clean_header_cell(s: str) -> str:
    s0 = str(s).replace("\r","\n")
    s0 = re.sub(r"(Отборы:|Дополнительные\s*поля:|Сортировка:|Показатели:).*","", s0, flags=re.I|re.S).strip()
    s0 = re.sub(r"[\n\t]+"," ", s0); s0 = re.sub(r"\s{2,}"," ", s0).strip()
    return RE_DOT_SUFFIX.sub("", s0)

def canon_from_joined(joined: str) -> str:
    low = joined.lower()
    if RE_CLIENT_EQ.search(low) or RE_CLIENT_ANY.search(low): return "Контрагент"
    if RE_OPEN.search(low):   return "нач. остаток"
    if RE_DEBIT.search(low):  return "приход"
    if RE_CREDIT.search(low): return "расход"
    if RE_END.search(low):    return "кон. остаток"
    return ""

# =================================================================
# НАЧАЛО БЛОКА ПОД ЗАМЕНУ (функция _extract_header_info)
# =================================================================

def _extract_header_info(xlsx: Path) -> Tuple[str, str, List[str]]:
    """
    Исправлено:
    1) Берём ВСЕ имена из фильтра «Контрагент В группе из списка (...)».
    2) Первый элемент списка = основной manager.
    3) Не расширяем «варианты» — храним как есть.
    4) Если внутренний юнит забыли в фильтре, добавляем его из INTERNAL_UNITS_MAP только для базового менеджера.
    """
    period, manager, managers_from_filter = "", "—", []

    try:
        head = pd.read_excel(xlsx, header=None, nrows=25, dtype=str).fillna("")
        lines = [" ".join(map(str, r)).strip() for _, r in head.iterrows()]

        for line in lines:
            lo = line.lower().replace("\xa0", " ").replace("\u202f", " ").strip()

            if lo.startswith("период:"):
                period = line.split(":", 1)[-1].strip()

            m_filter = RE_MANAGER_FILTER.search(line)  # ищем по исходной строке
            if m_filter:
                raw_inside = m_filter.group(1)
                managers_from_filter = [mm.strip() for mm in re.split(r"[;,\u061B]", raw_inside) if mm.strip()]
                if managers_from_filter:
                    manager = managers_from_filter[0]
                    base = manager.strip()
                    for unit in INTERNAL_UNITS_MAP.get(base, []):
                        if unit not in managers_from_filter:
                            managers_from_filter.append(unit)
                break

        if not period:
            for line in lines:
                m2 = DATEPAIR_RGX.search(line)
                if m2:
                    period = f"{m2.group(1)} – {m2.group(2)}"
                    break
    except Exception as e:
        log.warning("Ошибка чтения шапки Excel: %s", e)

    if manager == "—":
        all_mgrs = _manager_names_from_config() or []
        name_low = xlsx.name.lower()
        for nm in sorted(all_mgrs, key=len, reverse=True):
            if nm.lower() in name_low:
                manager = nm
                break

    return (period or "—", manager or "—", managers_from_filter or [])

# =================================================================
# КОНЕЦ БЛОКА ПОД ЗАМЕНУ
# =================================================================


def _manager_names_from_config() -> List[str]:
    cfg = getattr(config, "MANAGERS_CFG", {}) or {}
    names: List[str] = []
    mgrs = cfg.get("managers") or {}
    if isinstance(mgrs, dict): names.extend(list(mgrs.keys()))
    syns = cfg.get("synonyms") or {}
    if isinstance(syns, dict):
        for _, v in syns.items():
            if isinstance(v, list): names.extend([x for x in v if isinstance(x, str)])
    seen, uniq = set(), []
    for n in names:
        k = n.strip().lower()
        if k not in seen: uniq.append(n.strip()); seen.add(k)
    return uniq

def _expand_variants(names: List[str]) -> List[re.Pattern]:
    return [re.compile(fr"^(?:{re.escape(nm.strip())})(?:\s*[-–—]?[0-9]+)?$", re.I) for nm in names]

def _filter_out_managers(
    df: pd.DataFrame, *, manager_hint: str | None,
    additional_managers: List[str] | None = None
) -> Tuple[pd.DataFrame, int]:
    """
    Правило:
    - Имена из Excel-фильтра (additional_managers) исключаем ТОЛЬКО по точному совпадению
      без «расширений». Это сохраняет внешних клиентов вроде «Магира товар под зарплату».
    - Параллельно продолжаем отсекать заголовки/строки менеджеров из конфигурации по шаблонам.
    """
    col = None
    for c in df.columns:
        if str(c).lower() in {"клиент", "контрагент", "client", "контрагент/клиент"}:
            col = c
            break
    if not col:
        return df, 0

    s = df[col].astype(str).str.strip()

    # 1) Точное сравнение по именам из фильтра/подсказки
    exact_set = set()
    if manager_hint:
        exact_set.add(manager_hint.strip().lower())
    if additional_managers:
        for m in additional_managers:
            if m:
                exact_set.add(str(m).strip().lower())

    mask_exact = s.str.lower().isin(exact_set)

    # 2) Старые шаблоны для «заголовков менеджеров» из конфигурации
    mgr_names = _manager_names_from_config()
    pats_cfg = _expand_variants(mgr_names) if mgr_names else []
    mask_cfg = pd.Series(False, index=df.index)
    for p in pats_cfg:
        mask_cfg = mask_cfg | s.str.fullmatch(p)

    mask = mask_exact | mask_cfg
    return df.loc[~mask].copy(), int(mask.sum())

def _build_rows(df: pd.DataFrame, *, total_debt: float | None = None
    ) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    cols = list(df.columns)
    c_client = _find_col(cols, "Клиент","Контрагент","client","контрагент/клиент") or cols[0]
    c_final  = _find_col(cols, "кон","сальдо кон","задолженность","кон. остаток","конечный")
    c_days   = _find_col(cols, "дни","days","тишина","days_silence")
    c_ship   = _find_col(cols, "отгрузка","ship","приход")
    c_pay    = _find_col(cols, "оплата","pay","расход")

    def _num(col: Optional[str]) -> pd.Series:
        if not col or col not in df.columns: return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    final = _num(c_final)
    days  = pd.to_numeric(df[c_days], errors="coerce") if c_days and c_days in df.columns else pd.Series(np.nan, index=df.index)
    ship  = _num(c_ship); pay = _num(c_pay)

    # ТОП-15
    top_df = df[final > 0].copy(); top_df["_final"] = final[top_df.index]
    top_df = top_df.sort_values("_final", ascending=False).head(15)
    top: List[dict] = []
    for idx, r in top_df.iterrows():
        debt = float(r["_final"])
        pct  = (debt / total_debt * 100.0) if (total_debt and total_debt > 0) else None
        top.append({"client": r[c_client], "client_slug": slugify(r[c_client]), "debt": debt, "pct": pct})

    # Молчуны, Закрывшие, Переплата — без изменений
    silent_df = df[(final > 20000) & (days.fillna(0) > 0)].copy()
    silent_df["_final"] = final[silent_df.index]
    silent_df = silent_df.sort_values("_final", ascending=False)
    silent = [{"client": r[c_client], "client_slug": slugify(r[c_client]), "debt": float(r["_final"]),
               "days": int(days.loc[idx]) if pd.notna(days.loc[idx]) else 0}
              for idx, r in silent_df.iterrows()]
    closed_df = df[(ship > 0) & (pay > 0) & ((ship - pay).abs() <= 1)].copy()
    closed = [{"client": r[c_client], "client_slug": slugify(r[c_client]), "ship": float(ship.loc[idx]), "pay": float(pay.loc[idx])}
              for idx, r in closed_df.iterrows()]
    over_df = df[final < 0].copy()
    overpay = [{"client": r[c_client], "client_slug": slugify(r[c_client]), "overpay": float(final.loc[idx])}
               for idx, r in over_df.iterrows()]
    return top, silent, closed, overpay

def process_extended_report(clean_xlsx: Path, src_name: str) -> Dict[str, Any]:
    period_str, manager, managers_from_filter = _extract_header_info(clean_xlsx)

    managers_list = _manager_names_from_config()
    if manager and manager not in managers_list and manager != "—":
        managers_list.append(manager)
    if managers_from_filter:
        for m in managers_from_filter:
            if m not in managers_list:
                managers_list.append(m)

    blocks, period_min, period_max, agg = parse_extended_excel(clean_xlsx, managers_list)
    for b in blocks:
        if b.opening is None: b.opening = 0.0
        if b.closing is None: b.closing = 0.0

    # Δ по сравнению с нач. остатком
    delta = (agg["close"] - agg["open"])
    delta_label = "Увеличение" if delta > 0 else ("Уменьшение" if delta < 0 else "Без изменений")
    delta_abs = abs(delta)

    # Сортировки
    blocks_sorted = sorted(blocks, key=lambda x: (x.closing or 0.0), reverse=True)

    # ТОП-15
    top_debtors = [{"client": b.client, "client_slug": slugify(b.client), "debt": b.closing or 0.0}
                   for b in blocks_sorted[:15] if (b.closing or 0.0) > 0]

    # Все клиенты (+ days_silence)
    all_rows = [{
        "client": b.client, "client_slug": slugify(b.client),
        "debt": b.closing or 0.0, "opening": b.opening or 0.0,
        "debit": b.sum_debit, "credit": b.sum_credit, "movements": len(b.movements),
        "days_silence": (max(0, (period_max - (b.last_date or period_min)).days)
                         if (period_max is not None and (b.last_date or period_min) is not None) else None),
    } for b in blocks_sorted]

    # Движения: отсортировать клиентов в каждой подгруппе по убыванию closing
    by_subgroup: dict[str, list[ClientBlock]] = defaultdict(list)
    for b in blocks: by_subgroup[b.subgroup].append(b)
    by_subgroup_sorted = {k: sorted(v, key=lambda x: (x.closing or 0.0), reverse=True) for k, v in by_subgroup.items()}

    return {
        "title": "РАСШИРЕННЫЙ ОТЧЁТ ПО ДЕБИТОРСКОЙ ЗАДОЛЖЕННОСТИ",
        "period": f"{fmt_date(period_min)} — {fmt_date(period_max)}" if period_min and period_max else (period_str or "—"),
        "period_min": fmt_date(period_min), "period_max": fmt_date(period_max),
        "manager": manager,
        "client_count": agg["n_clients"],
        "total_debt": agg["close"],               # для карточки «Общий долг»
        "open_total": agg["open"], "debit_total": agg["debit"], "credit_total": agg["credit"], "close_total": agg["close"],
        "delta_label": delta_label, "delta_abs": delta_abs,
        "top_debtors": top_debtors,
        "all_rows": all_rows,
        "by_subgroup": by_subgroup_sorted,        # отсортировано по closing desc
        "aggregates": agg,
        "blocks": blocks,
        "tech_info": {
            "Источник": src_name,
            "Clean": str(clean_xlsx.name),
            "Модуль": __VERSION__,
        },
        "report_type": "extended",
    }

# ───── Рендер/сохранение
def _env() -> Environment:
    tpl_dir = getattr(config, "TEMPLATES_DIR", Path(__file__).parent / "templates")
    env = Environment(loader=FileSystemLoader(str(tpl_dir)),
                      autoescape=select_autoescape(["html","xml"]),
                      trim_blocks=True, lstrip_blocks=True)
    env.filters["money"] = money
    env.globals["abs"] = abs
    return env

def render_html(ctx: dict) -> str:
    return _env().get_template("debt_auto.html").render(**ctx)

def _generated_footer() -> str:
    if hasattr(config, "generated_at_tz"):
        return f"{config.generated_at_tz()} | Версия: {__VERSION__}"
    return f"Сформировано: {datetime.now(ZoneInfo('Asia/Almaty')):%d.%m.%Y %H:%M} (Asia/Almaty) | Версия: {__VERSION__}"

def _save_extended_json(ctx: dict, stem: str) -> Path:
    out_json_dir = getattr(config, "JSON_DIR", Path(__file__).parent / "reports" / "json")
    out_json_dir.mkdir(parents=True, exist_ok=True)

    def _mov(m: Movement) -> dict:
        return {
            "date": m.date.strftime("%Y-%m-%d") if (m.date is not None and not pd.isna(m.date)) else None,
            "debit": m.debit, "credit": m.credit,
        }

    blocks_json = []
    for b in ctx["blocks"]:
        blocks_json.append({
            "subgroup": b.subgroup, "client": b.client,
            "opening": b.opening or 0.0, "closing": b.closing or 0.0,
            "last_date": b.last_date.strftime("%Y-%m-%d") if (b.last_date is not None and not pd.isna(b.last_date)) else None,
            "movements": [_mov(m) for m in b.movements],
        })

    data = {
        "version": __VERSION__,
        "period_min": ctx.get("period_min"), "period_max": ctx.get("period_max"),
        "manager": ctx.get("manager"),
        "aggregates": ctx.get("aggregates"),
        "delta": {"label": ctx.get("delta_label"), "abs": ctx.get("delta_abs")},
        "clients": ctx.get("all_rows"),
        "blocks": blocks_json,
    }

    out_json = out_json_dir / f"debt_ext_{stem}.json"
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("JSON сохранён: %s", out_json)

    # ── Variant 2: создаём "simple-compatible" JSON для аналитики (DSO и др.)
    # Формат: reports/json/debt_<stem>.json
    try:
        simple = {
            "version": __VERSION__,
            "period": ctx.get("period"),
            "period_min": ctx.get("period_min"),
            "period_max": ctx.get("period_max"),
            "manager": ctx.get("manager"),
            "total_debt": (ctx.get("aggregates") or {}).get("close"),
            "clients": [],
            "__source_ext_json__": out_json.name,
        }
        for row in (ctx.get("all_rows") or []):
            # all_rows уже нормализован (client, debt/opening/debit/credit/movements/days_silence)
            if not isinstance(row, dict):
                continue
            simple["clients"].append({
                "client": row.get("client"),
                "debt": row.get("debt"),
                "opening": row.get("opening"),
                "debit": row.get("debit"),
                "credit": row.get("credit"),
                "days_silence": row.get("days_silence"),
            })
        out_simple = out_json_dir / f"debt_{stem}.json"
        out_simple.write_text(json.dumps(simple, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Simple JSON сохранён: %s", out_simple)
    except Exception as e:
        log.warning("Не удалось сохранить simple JSON (debt_%s.json): %s", stem, e)

    return out_json

def build_report(xlsx_path: str | Path, *, force: Optional[str] = None) -> Path:
    src = Path(xlsx_path).resolve()
    log.info("Обработка файла: %s", src.name)

    clean = ensure_clean_xlsx(src, force_fix=True)
    log.info("Создана чистая копия: %s", clean.name)

    report_type = force if force in {"simple","extended"} else detect_report_type(clean)
    log.info("Определён тип отчёта: %s", report_type)

    if report_type == "extended":
        ctx = process_extended_report(clean, src.name)
    else:
        ctx = process_simple_report(clean, src.name)

    ctx["generated"] = _generated_footer()
    html = render_html(ctx)

    out_html_dir = getattr(config, "OUT_DIR", Path(__file__).parent / "reports" / "html")
    out_html_dir.mkdir(parents=True, exist_ok=True)

    stem = src.stem
    if stem.endswith(".__clean"): stem = stem[:-8]

    if report_type == "extended":
        out_html = out_html_dir / f"debt_ext_{stem}.html"
        _save_extended_json(ctx, stem)
    else:
        out_html = out_html_dir / f"{stem}_debt.html"

    out_html.write_text(html, encoding="utf-8")
    log.info("Отчёт сохранён: %s", out_html)
    print(f"✅ Отчёт сохранён: {out_html}")
    return out_html

# ───── CLI
def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="Отчёт по дебиторке с автоопределением типа")
    p.add_argument("xlsx", help="Путь к XLSX файлу (исходник 1С)")
    p.add_argument("--force-simple", action="store_true", help="Принудительно собрать простой отчёт")
    p.add_argument("--force-extended", action="store_true", help="Принудительно собрать расширенный отчёт")
    args = p.parse_args(argv[1:])
    if args.force_simple and args.force_extended:
        print("Нельзя указывать одновременно --force-simple и --force-extended", file=sys.stderr); return 2
    try:
        force = "simple" if args.force_simple else ("extended" if args.force_extended else None)
        build_report(args.xlsx, force=force); return 0
    except SystemExit as e:
        return int(e.code)
    except Exception as e:
        log.exception("Ошибка при обработке: %s", e); return 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

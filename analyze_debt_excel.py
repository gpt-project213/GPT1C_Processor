#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_debt_excel.py
version: v2.1 (2025-09-02)

Назначение: парсер ПРОСТОГО отчёта 1С «Ведомость по взаиморасчётам с контрагентами».
Выход: DataFrame с колонками ['клиент','нач','приход','расход','кон'] и список ошибок математики.

Строго по ТЗ (инварианты и бизнес-логика):
• Распознавание «двухстрочной» шапки 1С: строка «Контрагент», ниже — «Нач. остаток / Приход / Расход / Кон. остаток».
• Фильтры служебных блоков: «Отборы/Показатели/Сортировка/Итог/Покупатели/Покупатели-работники».
• Нормализация чисел: удаление пробелов/неразрывных пробелов, запятая→точка, посторонние символы — в ноль.
• Δ-сверка: допуск ±1 тг трактуется как «Округление» (ТЗ) — EPS = 1.0 (раньше было 0.5).

Публичный API:
    parse_debt_report(xlsx) -> (df, errors)
    detect_has_movements(clean_xlsx) -> (has_movements, uniq_dates_count, total_dates_found)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Tuple, Dict, Optional
import re

import numpy as np
import pandas as pd

__VERSION__ = "analyze_debt_excel.py v2.1 — 2025-09-02"

# ── Регулярные выражения (синхронизированы с расширенным отчётом) ────────────
RE_CLIENT_EQ  = re.compile(r"^\s*контрагент\s*$", re.I)
RE_CLIENT_ANY = re.compile(r"(контрагент|покупатель|клиент)", re.I)

RE_OPEN   = re.compile(r"((^|\b)нач(\.|альный)?\s*остат(ок)?\b|сальдо\s*на\s*начало)", re.I)
RE_END    = re.compile(r"((^|\b)кон(\.|ечный)?\s*остат(ок)?\b|сальдо\s*на\s*конец)", re.I)
RE_DEBIT  = re.compile(r"(приход|отгр\w*|зачисл\w*|дебет|поставка|поступило)", re.I)
RE_CREDIT = re.compile(r"(расход|оплат\w*|списан\w*|кредит|оплата)", re.I)

RE_META_CELL  = re.compile(r"(показатели|группировк|отбор|дополнительные\s*поля|сортировка)", re.I)
RE_TOTAL_CELL = re.compile(r"(итогова|итог|^покупатели$|покупатели\s*-\s*работники)", re.I)
RE_LEVEL_CELL = re.compile(r"\bитог\b", re.I)

RE_DOT_SUFFIX = re.compile(r"\.\d+$")

# Допуск Δ-сверки по ТЗ (±1 тг = «округление»)
EPS = 1.0

# ── Утилиты -------------------------------------------------------------------
def _row_vals(raw: pd.DataFrame, i: int) -> list[str]:
    return [str(x).strip() if pd.notna(x) else "" for x in raw.iloc[i].tolist()]

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
    if RE_CLIENT_EQ.search(low) or RE_CLIENT_ANY.search(low): return "Контрагент"
    if RE_OPEN.search(low):   return "нач. остаток"
    if RE_DEBIT.search(low):  return "приход"
    if RE_CREDIT.search(low): return "расход"
    if RE_END.search(low):    return "кон. остаток"
    return ""

def find_header(raw: pd.DataFrame) -> list[int]:
    keys = ["нач. остаток", "приход", "расход", "кон. остаток"]
    limit = min(120, len(raw) - 1)

    # Вариант 1: явная строка «Контрагент»
    for i in range(limit):
        if any(RE_CLIENT_EQ.match(c) for c in _row_vals(raw, i)):
            r1 = [c.lower() for c in _row_vals(raw, i + 1)]
            if sum(1 for k in keys if any(k in c for c in r1)) >= 2:
                return [i, i + 1]

    # Вариант 2: строка с «контрагент/клиент/покупатель» + следующая со сводными полями
    for i in range(limit):
        cells_i = _row_vals(raw, i)
        if RE_CLIENT_ANY.search(" | ".join(cells_i)):
            r1 = [c.lower() for c in _row_vals(raw, i + 1)]
            if sum(1 for k in keys if any(k in c for c in r1)) >= 2:
                return [i, i + 1]

    raise ValueError("Не найдена строка шапки")

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

    # уникализация
    seen, uniq = {}, []
    for nm in names:
        if nm not in seen:
            seen[nm] = 0; uniq.append(nm)
        else:
            seen[nm] += 1; uniq.append(f"{nm}.{seen[nm]}")
    return uniq

def money_to_float(v: Any) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)): return 0.0
    s = str(v).strip().replace("\u00A0", "").replace("\u202f", "").replace(" ", "")
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in ("", "-", "."): return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0

# ── Публичный API -------------------------------------------------------------
def parse_debt_report(xlsx: str | Path) -> tuple[pd.DataFrame, list[tuple[str, float]]]:
    """
    Возвращает:
      df: колонки ['клиент','нач','приход','расход','кон']
      errors: список (клиент, delta), где delta = кон - (нач + приход - расход),
              и |delta| > EPS (EPS = 1.0 тг по ТЗ).
    """
    path = Path(xlsx)
    raw = pd.read_excel(path, header=None, dtype=str, keep_default_na=False)

    header_rows = find_header(raw)
    df = pd.read_excel(path, header=header_rows, dtype=str, keep_default_na=False)
    df.columns = build_names_from_header(raw, header_rows)[: df.shape[1]]

    # Имена нужных колонок (после канонизации)
    name_client  = next((c for c in df.columns if RE_CLIENT_ANY.search(c.lower())), None)
    name_opening = next((c for c in df.columns if RE_OPEN.search(c.lower())), None)
    name_debit   = next((c for c in df.columns if RE_DEBIT.search(c.lower())), None)
    name_credit  = next((c for c in df.columns if RE_CREDIT.search(c.lower())), None)
    name_closing = next((c for c in df.columns if RE_END.search(c.lower())), None)

    needed = dict(client=name_client, opening=name_opening, debit=name_debit,
                  credit=name_credit, closing=name_closing)
    missing = [k for k, v in needed.items() if v is None]
    if missing:
        miss = ", ".join(missing)
        raise ValueError(f"нет нужных колонок: {miss}")

    # Только полезные колонки
    df2 = df[[name_client, name_opening, name_debit, name_credit, name_closing]].copy()

    # Фильтрация служебных/пустых строк
    def _is_noise(name: str) -> bool:
        s = str(name or "").strip()
        if s == "": return True
        if RE_META_CELL.search(s): return True
        if RE_TOTAL_CELL.search(s): return True
        if RE_LEVEL_CELL.search(s): return True
        if s.lower() in {"покупатели", "покупатели-работники"}: return True
        return False

    df2[name_client] = df2[name_client].astype(str).str.strip()
    df2 = df2[~df2[name_client].map(_is_noise)].copy()

    # Нормализация денег
    df2["_нач"]    = df2[name_opening].map(money_to_float)
    df2["_приход"] = df2[name_debit].map(money_to_float)
    df2["_расход"] = df2[name_credit].map(money_to_float)
    df2["_кон"]    = df2[name_closing].map(money_to_float)

    # Итоговый вид (как ждёт простой отчёт)
    out = pd.DataFrame({
        "клиент": df2[name_client].astype(str).str.strip(),
        "нач":    df2["_нач"].astype(float),
        "приход": df2["_приход"].astype(float),
        "расход": df2["_расход"].astype(float),
        "кон":    df2["_кон"].astype(float),
    })

    # Диагностика математики (Δ по строкам; итоговую Δ в HTML НЕ показываем по вашему требованию)
    deltas = out["кон"] - (out["нач"] + out["приход"] - out["расход"])
    errors = [(row["клиент"], float(round(d, 2)))
              for row, d in zip(out.to_dict("records"), deltas)
              if abs(d) > EPS]

    out = out.fillna(0.0).reset_index(drop=True)
    return out, errors

# ── Лёгкий детект «есть движения по датам?» для оркестрации простого/расширенного ──
_DATE = re.compile(r"\b(?:\d{1,2}[./-]){2}\d{2,4}\b")

def detect_has_movements(clean_xlsx: str | Path) -> tuple[bool, int, int]:
    """
    Возвращает (has_movements, uniq_dates_count, total_dates_found).
    Критерии: наличие множества дат в первых ~150 строках и/или пара дат в шапке.
    """
    clean_xlsx = Path(clean_xlsx)
    head = pd.read_excel(clean_xlsx, header=None, dtype=str, nrows=40, keep_default_na=False)
    lines = [" ".join(str(x).strip() for x in row if str(x).strip()) for row in head.values]
    if any(len(_DATE.findall(line)) >= 2 for line in lines):
        return True, 2, 2
    body = pd.read_excel(clean_xlsx, header=None, dtype=str, nrows=150, keep_default_na=False)
    tokens = " ".join(" ".join(map(str, r)) for r in body.values)
    dates = _DATE.findall(tokens)
    uniq = len(set(dates))
    return (uniq >= 3 or len(dates) >= 6), uniq, len(dates)

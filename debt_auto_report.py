#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
debt_auto_report.py · v2.1 · 2025-09-03
Умный отчёт по дебиторке с автоопределением типа (простой/расширенный).
"""

from __future__ import annotations

import os
import sys
import re
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from zoneinfo import ZoneInfo

# Конфигурация и утилиты
import config
from utils_excel import ensure_clean_xlsx
from analyze_debt_excel import parse_debt_report

__VERSION__ = "debt_auto=v2.1"
NBSP = "\u202f"  # узкий пробел

# Настройка логирования
log = config.setup_logging("debt_auto_report")

# Регулярные выражения для детекта типа отчета
DATE_RGX = re.compile(r"\b\d{2}[./]\d{2}[./]\d{4}\b")
DATEPAIR_RGX = re.compile(r"(\d{2}[./]\d{2}[./]\d{4}).{0,10}(\d{2}[./]\d{2}[./]\d{4})")
RE_CLIENT_ANY = re.compile(r"(контрагент|покупатель|клиент)", re.I)

# Регулярные выражения для парсинга (добавлены недостающие)
RE_CLIENT_EQ = re.compile(r"^\s*контрагент\s*$", re.I)
RE_OPEN = re.compile(r"((^|\b)нач(\.|альный)?\s*остат(ок)?\b|сальдо\s*на\s*начало)", re.I)
RE_END = re.compile(r"((^|\b)кон(\.|ечный)?\s*остат(ок)?\b|сальдо\s*на\s*конец)", re.I)
RE_DEBIT = re.compile(r"(приход|отгр\w*|зачисл\w*|дебет|поставка|поступило)", re.I)
RE_CREDIT = re.compile(r"(расход|оплат\w*|списан\w*|кредит|оплата)", re.I)
RE_META_CELL = re.compile(r"(показатели|группировк|отбор|дополнительные\s*поля|сортировка)", re.I)
RE_TOTAL_CELL = re.compile(r"(итогова|итог|^покупатели$|покупатели\s*-\s*работники)", re.I)
RE_LEVEL_CELL = re.compile(r"\bитог\b", re.I)
RE_DOT_SUFFIX = re.compile(r"\.\d+$")
RE_DATE_CELL = re.compile(r"^\s*\d{2}\.\d{2}\.\d{4}\s*$")

# Утилиты форматирования
def money(x: float | int | None) -> str:
    try:
        s = f"{float(x):,.2f}"
    except Exception:
        return "0,00" if x in (None, "", float("nan")) else str(x)
    return s.replace(",", NBSP).replace(".", ",")

def slugify(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"[\s/]+", "_", s)
    s = s.replace("ё", "e")
    s = re.sub(r"[^0-9a-zA-Z_\-]+", "_", s)
    s = re.sub(r"__+", "_", s).strip("_")
    return s or "row"

# Детект типа отчета
def detect_report_type(xlsx_path: Path) -> str:
    """Определяет тип отчета: 'simple' или 'extended'"""
    try:
        # Проверка на расширенный отчет по шапке
        head = pd.read_excel(xlsx_path, nrows=60, dtype=str).fillna("")
        for line in head.astype(str).agg(" ".join, axis=1).tolist()[:25]:
            if DATEPAIR_RGX.search(line):
                return "extended"
        
        # Проверка по количеству дат в первом столбце
        col0 = head.iloc[:,0].astype(str).tolist()
        hits = sum(1 for v in col0 if DATE_RGX.fullmatch(v.strip()))
        if hits >= 8:
            return "extended"
            
        # Дополнительная проверка по содержимому
        body_check = pd.read_excel(xlsx_path, header=None, dtype=str, nrows=150, keep_default_na=False)
        flat_text = " ".join(" ".join(map(str, row)) for row in body_check.values)
        total_dates = len(DATE_RGX.findall(flat_text))
        
        if total_dates >= 15:
            return "extended"
            
    except Exception as e:
        log.warning(f"Ошибка при детекте типа отчета: {e}")
    
    return "simple"

# Обработка простого отчета
def process_simple_report(clean_xlsx: Path, src_name: str) -> Dict[str, Any]:
    """Обрабатывает простой отчет и возвращает данные для шаблона"""
    df, errors = parse_debt_report(clean_xlsx)
    
    # Извлечение информации из шапки
    period, manager, managers_from_filter = _extract_header_info(clean_xlsx)
    
    # Фильтрация менеджеров
    df_filtered, dropped = _filter_out_managers(df, manager_hint=manager if manager != "—" else None, additional_managers=managers_from_filter)
    
    # Основные показатели
    cols = {c.lower(): c for c in df_filtered.columns}
    c_final = cols.get("кон") or cols.get("сальдо кон") or cols.get("задолженность")
    total_debt = float(df_filtered.loc[df_filtered[c_final] > 0, c_final].sum() if c_final else 0.0)
    client_count = int(len(df_filtered))
    
    # Построение секций отчета
    top, silent, closed, overpay = _build_rows(df_filtered)
    
    # Все клиенты
    all_rows = []
    if c_final:
        all_rows = [{
            "client": r.get(cols.get("клиент", "клиент"), r.get("клиент")),
            "client_slug": slugify(r.get(cols.get("клиент", "клиент"), r.get("клиент"))),
            "debt": float(r.get(c_final, 0.0)) or 0.0,
            "days": (int(r.get(cols.get("дни",""), 0)) if cols.get("дни") and pd.notna(r.get(cols.get("дни"), None)) else None),
            "ship": float(r.get(cols.get("отгрузка",""), 0)) if cols.get("отгрузка") else 0.0,
            "pay":  float(r.get(cols.get("оплата",""), 0))  if cols.get("оплата")  else 0.0,
        } for r in df_filtered.sort_values(c_final, ascending=False).to_dict("records")]
    
    return {
        "title": "ОТЧЁТ ПО ДЕБИТОРСКОЙ ЗАДОЛЖЕННОСТИ",
        "period": period,
        "manager": manager,
        "client_count": client_count,
        "total_debt": total_debt,
        "top_debtors": top,
        "silent_rows": silent,
        "closed_rows": closed,
        "overpay_rows": overpay,
        "all_rows": all_rows,
        "tech_info": {
            "Источник": src_name,
            "Clean": str(clean_xlsx.name),
            "Строк (после фильтра)": client_count,
            "Сброшено (менеджеры)": dropped,
            "Ошибки математики": len(errors),
            "Модуль": __VERSION__,
        },
        "report_type": "simple"
    }

# Обработка расширенного отчета
def process_extended_report(clean_xlsx: Path, src_name: str) -> Dict[str, Any]:
    """Обрабатывает расширенный отчет и возвращает данные для шаблона"""
    blocks, period_min, period_max, agg = parse_extended_excel(clean_xlsx)
    
    # Добавьте эту проверку для обработки None значений
    for block in blocks:
        if block.opening is None:
            block.opening = 0.0
        if block.closing is None:
            block.closing = 0.0
        if block.sum_debit is None:
            block.sum_debit = 0.0
        if block.sum_credit is None:
            block.sum_credit = 0.0
    
    # Сортировка блоков по убыванию задолженности
    blocks_sorted = sorted(blocks, key=lambda x: x.closing or 0, reverse=True)
    
    # TOP-10 должников
    top_debtors = [{
        "client": b.client,
        "client_slug": slugify(b.client),
        "debt": b.closing or 0
    } for b in blocks_sorted[:10] if (b.closing or 0) > 0]
    
    # Все клиенты
    all_rows = [{
        "client": b.client,
        "client_slug": slugify(b.client),
        "debt": b.closing or 0,
        "opening": b.opening or 0,
        "debit": b.sum_debit,
        "credit": b.sum_credit,
        "movements": len(b.movements)
    } for b in blocks_sorted]
    
    # Группировка по подгруппам
    by_subgroup = {}
    for b in blocks:
        by_subgroup.setdefault(b.subgroup, []).append(b)
    
    return {
        "title": "РАСШИРЕННЫЙ ОТЧЁТ ПО ДЕБИТОРСКОЙ ЗАДОЛЖЕННОСТИ",
        "period": f"{fmt_date(period_min)} — {fmt_date(period_max)}",
        "period_min": fmt_date(period_min),
        "period_max": fmt_date(period_max),
        "client_count": agg['n_clients'],
        "total_debt": agg['close'],
        "top_debtors": top_debtors,
        "all_rows": all_rows,
        "by_subgroup": by_subgroup,
        "aggregates": agg,
        "blocks": blocks,
        "tech_info": {
            "Источник": src_name,
            "Clean": str(clean_xlsx.name),
            "Клиентов": agg['n_clients'],
            "Нач. остаток": money(agg['open']),
            "Кон. остаток": money(agg['close']),
            "Отгрузка": money(agg['debit']),
            "Оплата": money(agg['credit']),
            "Модуль": __VERSION__,
        },
        "report_type": "extended"
    }

# Парсер расширенного отчета
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

def parse_extended_excel(path: Path) -> Tuple[List[ClientBlock], Optional[pd.Timestamp], Optional[pd.Timestamp], Dict]:
    """Парсит расширенный отчет Excel"""
    raw = pd.read_excel(path, header=None, dtype=str, keep_default_na=False)
    
    # Поиск заголовка
    header_rows = find_header(raw)
    
    # Чтение данных
    df = pd.read_excel(path, header=header_rows, dtype=str, keep_default_na=False)
    df.columns = build_names_from_header(raw, header_rows)[: df.shape[1]]
    
    # Определение колонок
    name_client = next(c for c in df.columns if RE_CLIENT_ANY.search(c.lower()))
    name_opening = next(c for c in df.columns if RE_OPEN.search(c.lower()))
    name_debit = next(c for c in df.columns if RE_DEBIT.search(c.lower()))
    name_credit = next(c for c in df.columns if RE_CREDIT.search(c.lower()))
    name_closing = next(c for c in df.columns if RE_END.search(c.lower()))
    
    # Конвертация денежных значений
    df["_open"] = df[name_opening].map(money_to_float)
    df["_debit"] = df[name_debit].map(money_to_float)
    df["_credit"] = df[name_credit].map(money_to_float)
    df["_close"] = df[name_closing].map(money_to_float)
    df["_date"] = df[name_client].apply(parse_date_cell)
    
    # Классификация строк
    types = []
    for s in df[name_client].astype(str).fillna(""):
        ss = s.strip()
        if not ss or ss.lower() == "nan":
            types.append("EMPTY")
        elif RE_DATE_CELL.match(ss):
            types.append("DATE")
        elif RE_TOTAL_CELL.search(ss):
            types.append("TOTAL")
        elif RE_LEVEL_CELL.search(ss):
            types.append("LEVEL")
        else:
            types.append("CLIENT")
    
    df["_type"] = types
    
    # Парсинг блоков клиентов
    blocks = []
    cur_subgroup = "Общий"
    cur_client = None
    
    for _, row in df.iterrows():
        t = row["_type"]
        if t == "CLIENT":
            title = str(row[name_client]).strip()
            cur_client = ClientBlock(subgroup=cur_subgroup, client=title)
            op = money_to_float(row["_open"])
            cl = money_to_float(row["_close"])
            if op != 0.0: cur_client.opening = op
            if cl != 0.0: cur_client.closing = cl
            blocks.append(cur_client)
        elif t == "DATE" and cur_client:
            dts = row["_date"]
            db = money_to_float(row["_debit"])
            cr = money_to_float(row["_credit"])
            if not pd.isna(dts) and (db != 0 or cr != 0):
                cur_client.movements.append(Movement(date=dts, debit=db, credit=cr))
                cur_client.last_date = dts
    
    # Агрегация
    agg = aggregate_blocks(blocks)
    
    # Определение периода
    dates = [m.date for b in blocks for m in b.movements if not pd.isna(m.date)]
    period_min = min(dates) if dates else None
    period_max = max(dates) if dates else None
    
    return blocks, period_min, period_max, agg

def aggregate_blocks(blocks: List[ClientBlock]) -> Dict[str, Any]:
    """Агрегирует данные по всем клиентам"""
    total_open = total_debit = total_credit = total_close = 0.0
    n_clients = len(blocks)
    
    for b in blocks:
        total_open += b.opening or 0.0
        total_debit += b.sum_debit
        total_credit += b.sum_credit
        total_close += b.closing or 0.0
    
    trend = "Без изменений"
    if total_close < total_open: 
        trend = "Прогресс"
    elif total_close > total_open: 
        trend = "Регресс"
    
    return {
        "n_clients": n_clients,
        "open": total_open,
        "debit": total_debit,
        "credit": total_credit,
        "close": total_close,
        "trend": trend
    }

# Утилиты для парсинга
def find_header(raw: pd.DataFrame) -> list[int]:
    """Находит строки заголовка в DataFrame"""
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
    
    return [0, 1]  # fallback

def build_names_from_header(raw: pd.DataFrame, header_rows: list[int]) -> list[str]:
    """Строит имена колонок на основе заголовка"""
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
                    name = lab
                    break
        
        if not name:
            short = [p for p in parts if len(p) <= 40]
            name = short[-1] if short else (parts[-1] if parts else f"col{col}")
        
        names.append(name)
    
    # Уникализация имен колонок
    seen, uniq = {}, []
    for nm in names:
        if nm not in seen:
            seen[nm] = 0
            uniq.append(nm)
        else:
            seen[nm] += 1
            uniq.append(f"{nm}.{seen[nm]}")
    
    return uniq

def money_to_float(v: Any) -> float:
    """Конвертирует денежное значение в float"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    
    s = str(v).strip().replace("\u00A0", "").replace("\u202f", "").replace(" ", "")
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9.\-]", "", s)
    
    if s in ("", "-", "."):
        return 0.0
    
    try:
        return float(s)
    except Exception:
        return 0.0

def parse_date_cell(v: Any) -> pd.Timestamp:
    """Парсит дату из ячейки"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return pd.NaT
    
    s = str(v).strip()
    if not re.match(r"^\d{2}\.\d{2}\.\d{4}$", s):
        return pd.NaT
    
    return pd.to_datetime(s, format="%d.%m.%Y", errors="coerce")

def fmt_date(ts: Optional[pd.Timestamp]) -> str:
    """Форматирует дату в строку"""
    return "" if ts is None or pd.isna(ts) else ts.strftime("%d.%m.%Y")

# Вспомогательные функции
def _row_vals(raw: pd.DataFrame, i: int) -> list[str]:
    return [str(x).strip() if pd.notna(x) else "" for x in raw.iloc[i].tolist()]

def clean_header_cell(s: str) -> str:
    s0 = str(s).replace("\r", "\n")
    s0 = re.sub(r"(Отборы:|Дополнительные\s*поля:|Сортировка:|Показатели:).*", "", s0, flags=re.I | re.S).strip()
    s0 = re.sub(r"[\n\t]+", " ", s0)
    s0 = re.sub(r"\s{2,}", " ", s0).strip()
    s0 = RE_DOT_SUFFIX.sub("", s0)
    return s0

def canon_from_joined(joined: str) -> str:
    low = joined.lower()
    if RE_CLIENT_EQ.search(low) or RE_CLIENT_ANY.search(low):
        return "Контрагент"
    if RE_OPEN.search(low):
        return "нач. остаток"
    if RE_DEBIT.search(low):
        return "приход"
    if RE_CREDIT.search(low):
        return "расход"
    if RE_END.search(low):
        return "кон. остаток"
    return ""

# Функции из debt_report.py
def _extract_header_info(xlsx: Path) -> Tuple[str, str, List[str]]:
    """Извлекает период, менеджера и список менеджеров из шапки файла"""
    period = ""
    manager = "—"
    managers_from_filter = []  # Список менеджеров из отбора

    try:
        head = pd.read_excel(xlsx, header=None, nrows=25, dtype=str).fillna("")
        lines = [" ".join(row).strip() for _, row in head.iterrows()]
        
        for line in lines:
            lo = line.lower().replace("\xa0", " ").replace("\u202f", " ").strip()
            if lo.startswith("период:"):
                period = line.split(":",1)[-1].strip()
            # Ищем строку с отбором по контрагентам
            if "контрагент в группе из списка" in lo:
                # Пример строки: "Контрагент В группе из списка (Алена; Алена 1);"
                # Извлечем часть в скобках
                match = re.search(r"\(([^)]+)\)", line)
                if match:
                    managers_str = match.group(1)
                    # Разделяем по запятой или точке с запятой
                    managers_from_filter = [m.strip() for m in re.split(r"[;,]", managers_str) if m.strip()]
        
        if not period:
            for line in lines:
                m = DATEPAIR_RGX.search(line)
                if m:
                    period = f"{m.group(1)} – {m.group(2)}"
                    break
    except Exception:
        pass
    
    # Извлечение имени менеджера из имени файла
    stem = xlsx.stem
    m1 = re.search(r"(?i)^дебитор[а-яё]*\s+([A-Za-zА-ЯЁа-я]+)", stem)
    if m1: 
        manager = m1.group(1)
    
    m2 = re.search(r"(?i)_(?:[A-Za-zА-ЯЁа-я]+\s)?([A-Za-zА-ЯЁа-я]+)\s*\(\d+\)$", stem)
    if m2: 
        manager = m2.group(1)
    
    return period or "—", manager or "—", managers_from_filter

def _filter_out_managers(df: pd.DataFrame, *, manager_hint: str | None, additional_managers: List[str] = None) -> Tuple[pd.DataFrame, int]:
    """Фильтрует строки менеджеров из данных"""
    # Поиск колонки с клиентами
    col = None
    for c in df.columns:
        if c.lower() in {"клиент","контрагент","client","контрагент/клиент"}:
            col = c
            break
    
    if not col:
        return df, 0
    
    # Получение списка менеджеров из конфига
    managers = _manager_names_from_config()
    if manager_hint and manager_hint not in managers:
        managers.append(manager_hint)
    
    # Добавляем дополнительных менеджеров
    if additional_managers:
        for m in additional_managers:
            if m not in managers:
                managers.append(m)
    
    patterns = _expand_variants(managers) if managers else []
    if not patterns:
        return df, 0
    
    # Фильтрация
    s = df[col].astype(str).str.strip()
    mask = pd.Series([False]*len(df))
    
    for p in patterns:
        mask = mask | s.str.fullmatch(p)
    
    return df.loc[~mask].copy(), int(mask.sum())

def _manager_names_from_config() -> List[str]:
    """Получает список менеджеров из конфигурации"""
    cfg = getattr(config, "MANAGERS_CFG", {}) or {}
    names = []
    mgrs = cfg.get("managers") or {}
    
    if isinstance(mgrs, dict):
        names.extend(list(mgrs.keys()))
    
    syns = cfg.get("synonyms") or {}
    if isinstance(syns, dict):
        for base, variants in syns.items():
            if isinstance(variants, list):
                names.extend([v for v in variants if isinstance(v, str)])
    
    # Уникализация
    seen = set()
    uniq = []
    
    for n in names:
        k = n.strip().lower()
        if k not in seen:
            uniq.append(n.strip())
            seen.add(k)
    
    return uniq

def _expand_variants(names: List[str]) -> List[re.Pattern]:
    """Создает regex patterns для вариантов имен менеджеров"""
    pats = []
    for nm in names:
        nmq = re.escape(nm.strip())
        pats.append(re.compile(fr"^(?:{nmq})(?:\s*[-–—]?[0-9]+)?$", re.IGNORECASE))
    return pats

def _build_rows(df: pd.DataFrame) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    """Строит секции отчета из DataFrame"""
    cols = {c.lower(): c for c in df.columns}
    c_client = cols.get("клиент") or cols.get("контрагент") or cols.get("client") or cols.get("контрагент/клиент")
    c_final = cols.get("кон") or cols.get("сальдо кон") or cols.get("задолженность")
    c_days = cols.get("дни") or cols.get("days") or cols.get("тишина") or cols.get("days_silence")
    c_ship = cols.get("отгрузка") or cols.get("ship") or cols.get("отгрузка (приход)")
    c_pay = cols.get("оплата") or cols.get("pay") or cols.get("оплата (расход)")

    # ТОП должников
    top_df = df[df[c_final] > 0].sort_values(c_final, ascending=False).head(10) if c_final else df.head(0)
    top = [{
        "client": r[c_client],
        "client_slug": slugify(r[c_client]),
        "debt": float(r[c_final]),
        "days": (int(r.get(c_days, 0)) if c_days and pd.notna(r.get(c_days, None)) else None),
    } for r in top_df.to_dict("records")]

    # Молчуны
    silent = []
    if c_days and c_final:
        sd = df[(df[c_final] > 20000) & (df[c_days].fillna(0).astype(float) > 0)].sort_values(c_final, ascending=False)
        silent = [{
            "client": r[c_client], "client_slug": slugify(r[c_client]),
            "debt": float(r[c_final]), "days": int(r.get(c_days, 0))
        } for r in sd.to_dict("records")]

    # Закрывшие долг
    closed = []
    if c_ship and c_pay:
        cd = df[(df[c_ship].fillna(0).astype(float) > 0) &
                (df[c_pay].fillna(0).astype(float) > 0) &
                ((df[c_ship] - df[c_pay]).abs() <= 1)]
        closed = [{
            "client": r[c_client], "client_slug": slugify(r[c_client]),
            "ship": float(r.get(c_ship, 0)), "pay": float(r.get(c_pay, 0))
        } for r in cd.to_dict("records")]

    # Переплата
    overpay_df = df[df[c_final] < 0] if c_final else df.head(0)
    overpay = [{
        "client": r[c_client], "client_slug": slugify(r[c_client]),
        "overpay": float(r[c_final])  # отрицательное
    } for r in overpay_df.to_dict("records")]

    return top, silent, closed, overpay

# Рендеринг HTML
def render_html(ctx: dict) -> str:
    """Рендерит HTML из шаблона с использованием переданного контекста"""
    tpl_dir = getattr(config, "TEMPLATES_DIR", Path(__file__).parent / "templates")
    env = Environment(
        loader=FileSystemLoader(str(tpl_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True, 
        lstrip_blocks=True
    )
    env.filters["money"] = money
    tpl = env.get_template("debt_auto.html")
    return tpl.render(**ctx)

def build_report(xlsx_path: str | Path) -> Path:
    """Основная функция построения отчета"""
    src = Path(xlsx_path).resolve()
    log.info("Обработка файла: %s", src.name)
    
    # Создание чистой копии
    clean = ensure_clean_xlsx(src, force_fix=True)
    log.info("Создана чистая копия: %s", clean.name)
    
    # Определение типа отчета
    report_type = detect_report_type(clean)
    log.info("Определен тип отчета: %s", report_type)
    
    # Обработка в зависимости от типа
    if report_type == "extended":
        log.info("Обработка как расширенный отчет")
        ctx = process_extended_report(clean, src.name)
    else:
        log.info("Обработка как простой отчет")
        ctx = process_simple_report(clean, src.name)
    
    # Добавление общей информации
    ctx["generated"] = (f"{config.generated_at_tz()} | Версия: {__VERSION__}"
                       if hasattr(config, "generated_at_tz")
                       else f"Сформировано: {datetime.now(ZoneInfo('Asia/Almaty')):%d.%m.%Y %H:%M} (Asia/Almaty) | Версия: {__VERSION__}")
    
    # Рендеринг HTML
    html_content = render_html(ctx)
    
    # Сохранение результата
    out_dir = getattr(config, "OUT_DIR", Path(__file__).parent / "reports" / "html")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    stem = src.stem
    if stem.endswith(".__clean"):
        stem = stem[:-8]
    
    out_path = out_dir / f"{stem}_debt.html"
    out_path.write_text(html_content, encoding="utf-8")
    
    log.info("Отчет сохранен: %s", out_path)
    print(f"✅ Отчет сохранен: {out_path}")
    
    return out_path

def main(argv: List[str]) -> int:
    """Точка входа CLI"""
    parser = argparse.ArgumentParser(description="Умный отчёт по дебиторке с автоопределением типа")
    parser.add_argument("xlsx", help="Путь к XLSX файлу (исходник 1С)")
    args = parser.parse_args(argv[1:])
    
    try:
        build_report(args.xlsx)
        return 0
    except SystemExit as e:
        return int(e.code)
    except Exception as e:
        log.exception("Ошибка при обработке: %s", e)
        return 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
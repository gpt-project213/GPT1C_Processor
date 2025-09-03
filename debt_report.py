#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
debt_report.py · v1.0.2 · 2025-09-03

«Дебиторка (простой)» — кликабельный отчёт (вкладки):
ТОП должников / Молчуны / Закрывшие долг / Переплата / Все клиенты / Техданные.

Инварианты (ТЗ):
• HTML → reports/html, имя: {stem}_debt.html
• Логи: logs/debt_report_YYYYMMDD_HHMMSS.log, формат: '%(asctime)s, %(levelname)s %(message)s', TZ=Asia/Almaty
• Футер: 'Сформировано: DD.MM.YYYY HH:MM (Asia/Almaty) | Версия: <__VERSION__>'
• Кодировка: UTF-8; стиль «как в простой дебиторке»

Публичный API: build_simple_report(xlsx) → Path(HTML)

Бизнес-логика парсинга НЕ меняется:
• utils_excel.ensure_clean_xlsx(src, force_fix=...) — без out_dir
• analyze_debt_excel.parse_debt_report(xlsx) → (DataFrame, errors)

НОВОЕ:
• Режим детекта расширенного отчёта настраивается:
  - CLI: --ext-detect exit|redirect|ignore
  - ENV: DEBT_EXT_DETECT_MODE
  - CONFIG: EXT_DETECT_MODE = "exit"|"redirect"|"ignore"
  Приоритет: CLI > ENV > CONFIG > default("exit")
"""
from __future__ import annotations

import os, sys, re, logging, argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from zoneinfo import ZoneInfo

import config
from utils_excel import ensure_clean_xlsx
from analyze_debt_excel import parse_debt_report

__VERSION__ = "debt_simple=v1.0.2"
NBSP = "\u202f"  # узкий пробел

# ─────────────────────────── ЛОГИ (через config) ─────────────────────────────
log = config.setup_logging("debt_report")

# ─────────────────────────── Формат денег ────────────────────────────────────
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

def _stem_for_out(src_path: Path) -> str:
    name = src_path.name
    if name.endswith(".xlsx.__clean.xlsx"):
        name = name[:-len(".xlsx.__clean.xlsx")]
    elif name.endswith(".xlsx"):
        name = name[:-5]
    return name

DATE_RGX      = re.compile(r"\b\d{2}[./]\d{2}[./]\d{4}\b")
DATEPAIR_RGX  = re.compile(r"(\d{2}[./]\d{2}[./]\d{4}).{0,10}(\d{2}[./]\d{2}[./]\d{4})")

def _extract_header_info(xlsx: Path) -> Tuple[str, str]:
    """(period, manager) из шапки/имени файла (без вмешательства в парсер)."""
    period = ""
    manager = "—"
    try:
        head = pd.read_excel(xlsx, header=None, nrows=25, dtype=str).fillna("")
        lines = [" ".join(row).strip() for _, row in head.iterrows()]
        for line in lines:
            lo = line.lower().replace("\xa0", " ").replace("\u202f", " ").strip()
            if lo.startswith("период:"):
                period = line.split(":",1)[-1].strip(); break
        if not period:
            for line in lines:
                m = DATEPAIR_RGX.search(line)
                if m:
                    period = f"{m.group(1)} – {m.group(2)}"; break
    except Exception:
        pass
    stem = xlsx.stem
    m1 = re.search(r"(?i)^дебитор[а-яё]*\s+([A-Za-zА-ЯЁа-я]+)", stem)
    if m1: manager = m1.group(1)
    m2 = re.search(r"(?i)_(?:[A-Za-zА-ЯЁа-я]+\s)?([A-Za-zА-ЯЁа-я]+)\s*\(\d+\)$", stem)
    if m2: manager = m2.group(1)
    return period or "—", manager or "—"

def _is_extended(xlsx: Path) -> bool:
    """Эвристика: период (дата–дата) в шапке ИЛИ много дат в первом столбце."""
    try:
        head = pd.read_excel(xlsx, nrows=60, dtype=str).fillna("")
    except Exception:
        return False
    for line in head.astype(str).agg(" ".join, axis=1).tolist()[:25]:
        if DATEPAIR_RGX.search(line):
            return True
    col0 = head.iloc[:,0].astype(str).tolist()
    hits = sum(1 for v in col0 if DATE_RGX.fullmatch(v.strip()))
    return hits >= 8

def _manager_names_from_config() -> List[str]:
    """Берём config.managers.json (если есть). Формат: {"managers": {...}, "synonyms": {...}}"""
    cfg = getattr(config, "MANAGERS_CFG", {}) or {}
    names: List[str] = []
    mgrs = cfg.get("managers") or {}
    if isinstance(mgrs, dict):
        names.extend(list(mgrs.keys()))
    syns = cfg.get("synonyms") or {}
    if isinstance(syns, dict):
        for base, variants in syns.items():
            if isinstance(variants, list):
                names.extend([v for v in variants if isinstance(v, str)])
    # уникализация по lower
    seen = set(); uniq = []
    for n in names:
        k = n.strip().lower()
        if k not in seen:
            uniq.append(n.strip()); seen.add(k)
    return uniq

def _expand_variants(names: List[str]) -> List[re.Pattern]:
    pats: List[re.Pattern] = []
    for nm in names:
        nmq = re.escape(nm.strip())
        pats.append(re.compile(fr"^(?:{nmq})(?:\s*[-–—]?[0-9]+)?$", re.IGNORECASE))
    return pats

def _filter_out_managers(df: pd.DataFrame, *, manager_hint: str | None) -> Tuple[pd.DataFrame, int]:
    # ищем колонку клиента
    col = None
    for c in df.columns:
        if c.lower() in {"клиент","контрагент","client","контрагент/клиент"}:
            col = c; break
    if not col:
        return df, 0
    base = _manager_names_from_config()
    if manager_hint and manager_hint not in base:
        base.append(manager_hint)
    patterns = _expand_variants(base) if base else []
    if not patterns:
        return df, 0
    s = df[col].astype(str).str.strip()
    mask = pd.Series([False]*len(df))
    for p in patterns:
        mask = mask | s.str.fullmatch(p)
    return df.loc[~mask].copy(), int(mask.sum())

def _build_rows(df: pd.DataFrame) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    cols = {c.lower(): c for c in df.columns}
    c_client = cols.get("клиент") or cols.get("контрагент") or cols.get("client") or cols.get("контрагент/клиент")
    c_final  = cols.get("кон") or cols.get("сальдо кон") or cols.get("задолженность")
    c_days   = cols.get("дни") or cols.get("days") or cols.get("тишина") or cols.get("days_silence")
    c_ship   = cols.get("отгрузка") or cols.get("ship") or cols.get("отгрузка (приход)")
    c_pay    = cols.get("оплата") or cols.get("pay") or cols.get("оплата (расход)")

    # ТОП (только положительная задолженность)
    top_df = df[df[c_final] > 0].sort_values(c_final, ascending=False).head(10) if c_final else df.head(0)
    top = [{
        "client": r[c_client],
        "client_slug": slugify(r[c_client]),
        "debt": float(r[c_final]),
        "days": (int(r.get(c_days, 0)) if c_days and pd.notna(r.get(c_days, None)) else None),
    } for r in top_df.to_dict("records")]

    # молчуны (есть дни и долг > 20000)
    silent: List[dict] = []
    if c_days and c_final:
        sd = df[(df[c_final] > 20000) & (df[c_days].fillna(0).astype(float) > 0)].sort_values(c_final, ascending=False)
        silent = [{
            "client": r[c_client], "client_slug": slugify(r[c_client]),
            "debt": float(r[c_final]), "days": int(r.get(c_days, 0))
        } for r in sd.to_dict("records")]

    # закрывшие |отгрузка-оплата| <= 1
    closed: List[dict] = []
    if c_ship and c_pay:
        cd = df[(df[c_ship].fillna(0).astype(float) > 0) &
                (df[c_pay].fillna(0).astype(float) > 0) &
                ((df[c_ship] - df[c_pay]).abs() <= 1)]
        closed = [{
            "client": r[c_client], "client_slug": slugify(r[c_client]),
            "ship": float(r.get(c_ship, 0)), "pay": float(r.get(c_pay, 0))
        } for r in cd.to_dict("records")]

    # переплата
    overpay_df = df[df[c_final] < 0] if c_final else df.head(0)
    overpay = [{
        "client": r[c_client], "client_slug": slugify(r[c_client]),
        "overpay": float(r[c_final])  # отрицательное
    } for r in overpay_df.to_dict("records")]

    return top, silent, closed, overpay

def _render_html(ctx: dict) -> str:
    tpl_dir = getattr(config, "TEMPLATES_DIR", Path(__file__).parent / "templates")
    env = Environment(loader=FileSystemLoader(str(tpl_dir)),
                      autoescape=select_autoescape(["html","xml"]),
                      trim_blocks=True, lstrip_blocks=True)
    env.filters["money"] = money
    tpl = env.get_template("debt.html")
    return tpl.render(**ctx)

def build_debt_report(clean_xlsx: Path, *, out_stem: str, src_name: str) -> Path:
    df, errors = parse_debt_report(clean_xlsx)

    period, manager = _extract_header_info(clean_xlsx)
    df_filtered, dropped = _filter_out_managers(df, manager_hint=manager if manager != "—" else None)

    cols = {c.lower(): c for c in df_filtered.columns}
    c_final = cols.get("кон") or cols.get("сальдо кон") or cols.get("задолженность")
    total_debt = float(df_filtered.loc[df_filtered[c_final] > 0, c_final].sum() if c_final else 0.0)
    client_count = int(len(df_filtered))

    top, silent, closed, overpay = _build_rows(df_filtered)

    ctx = {
        "title": "ОТЧЁТ ПО ДЕБИТОРСКОЙ ЗАДОЛЖЕННОСТИ",
        "period": period,
        "manager": manager,
        "client_count": client_count,
        "total_debt": total_debt,
        "trend": "—",
        "top_debtors": top,
        "silent_rows": silent,
        "closed_rows": closed,
        "overpay_rows": overpay,
        "all_rows": [{
            "client": r.get(cols.get("клиент", "клиент"), r.get("клиент")),
            "client_slug": slugify(r.get(cols.get("клиент", "клиент"), r.get("клиент"))),
            "debt": float(r.get(c_final, 0.0)),
            "days": (int(r.get(cols.get("дни",""), 0)) if cols.get("дни") and pd.notna(r.get(cols.get("дни"), None)) else None),
            "ship": float(r.get(cols.get("отгрузка",""), 0)) if cols.get("отгрузка") else None,
            "pay":  float(r.get(cols.get("оплата",""), 0))  if cols.get("оплата")  else None,
        } for r in df_filtered.sort_values(c_final, ascending=False).to_dict("records")] if c_final else [],
        "generated": (f"{config.generated_at_tz()} | Версия: {__VERSION__}"
                      if hasattr(config, "generated_at_tz")
                      else f"Сформировано: {datetime.now(ZoneInfo('Asia/Almaty')):%d.%m.%Y %H:%M} (Asia/Almaty) | Версия: {__VERSION__}"),
        "tech_info": {
            "Источник": src_name,
            "Clean": str(clean_xlsx.name),
            "Строк (после фильтра)": client_count,
            "Сброшено (менеджеры)": dropped,
            "Ошибки математики": len(errors),
            "Модуль": __VERSION__,
        },
    }

    html = _render_html(ctx)
    out_dir = getattr(config, "OUT_DIR", Path(__file__).parent / "reports" / "html")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_stem}_debt.html"
    out_path.write_text(html, encoding="utf-8")
    log.info("✔ debt HTML %s", out_path)
    print(f"✔ HTML: {out_path}")
    return out_path

def _resolve_detect_mode(cli_mode: str | None) -> str:
    if cli_mode:
        return cli_mode.lower()
    env_mode = os.getenv("DEBT_EXT_DETECT_MODE", "").lower().strip()
    if env_mode in {"exit","redirect","ignore"}:
        return env_mode
    cfg_mode = getattr(config, "EXT_DETECT_MODE", "exit")
    return (cfg_mode or "exit").lower()

def build_simple_report(xlsx: str | Path, *, detect_mode: str | None = None) -> Path:
    src = Path(xlsx).resolve()
    log.info("SRC %s", src.name)
    clean = ensure_clean_xlsx(src, force_fix=True)
    log.info("Clean copy → %s", clean.name)

    mode = _resolve_detect_mode(detect_mode)

    if _is_extended(clean):
        if mode == "ignore":
            log.warning("Детект расширенного проигнорирован (mode=ignore). Продолжаю строить ПРОСТОЙ.")
        elif mode == "redirect":
            log.info("Обнаружен расширенный → redirect в extended_debt_report.build_extended_report()")
            from extended_debt_report import build_extended_report
            # build_extended_report сам должен сохранить HTML/JSON по ТЗ
            return build_extended_report(clean)
        else:  # exit
            log.error("Обнаружена детализация по датам/период: это РАСШИРЕННЫЙ отчёт. Запустите extended_debt_report.py на этом файле.")
            sys.exit(3)

    stem_for_out = _stem_for_out(src)
    return build_debt_report(clean, out_stem=stem_for_out, src_name=src.name)

def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Дебиторка (простой) — HTML-отчёт")
    parser.add_argument("xlsx", help="Путь к XLSX (исходник 1С)")
    parser.add_argument("--ext-detect", choices=["exit","redirect","ignore"],
                        help="Поведение при обнаружении расширенного отчёта (default=exit)")
    args = parser.parse_args(argv[1:])
    try:
        build_simple_report(args.xlsx, detect_mode=args.ext_detect); return 0
    except SystemExit as e:
        return int(e.code)
    except Exception as e:
        log.exception("FAILED: %s", e); return 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

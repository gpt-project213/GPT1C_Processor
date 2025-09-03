#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
debt_unified.py · v1.0.0 · 2025-09-03

Единый генератор отчётов по дебиторке с автодетектом:
- Простой (одна дата в шапке, нет детализации по датам) → reports/html/{stem}_debt.html
- Расширенный (период в шапке + детализация по датам)   → reports/html/debt_ext_{stem}.html + reports/json/debt_ext_{stem}.json
- Сводный (не распознано имя менеджера) помечается в шапке как «Сводный отчёт»

Инварианты ТЗ:
• Таймзона: Asia/Almaty; футер: "Сформировано: DD.MM.YYYY HH:MM (Asia/Almaty) | Версия: <__VERSION__>"
• Логи: logs/debt_unified_YYYYMMDD_HHMMSS.log, формат "%(asctime)s, %(levelname)s %(message)s"
• Кодировка: UTF-8 без BOM
• Стиль HTML: "как в простой дебиторке", адаптив, кликабельные вкладки
• Публичный API: build_simple_report(xlsx), build_extended_report(xlsx) — совместимость для бота/CLI сохранена
• Бизнес-логика парсеров не меняется: берём parse_debt_report() и parse_excel() из твоих модулей

Версии шаблонов:
• templates/debt.html       — v1.0.0 (2025-09-03)
• templates/debt_ext.html   — v1.0.0 (2025-09-03)
"""
from __future__ import annotations

import sys, re, json, logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from zoneinfo import ZoneInfo

import config
from utils_excel import ensure_clean_xlsx
from analyze_debt_excel import parse_debt_report      # простой
import extended_debt_report as X                      # расширенный (parse_excel, ClientBlock,...)

__VERSION__ = "debt_unified=v1.0.0 (2025-09-03)"
NBSP = "\u202f"

log = config.setup_logging("debt_unified")
TZ  = ZoneInfo("Asia/Almaty")

# ──────────────────────────────────────────────────────────────────────────────
# УТИЛИТЫ
def money(x: Any) -> str:
    try:
        f = float(x)
    except Exception:
        return "0,00" if x in (None, "", float("nan")) else str(x)
    s = f"{f:,.2f}".replace(",", NBSP).replace(".", ",")
    return s

def slugify(s: str) -> str:
    s0 = str(s or "").strip().lower()
    s0 = re.sub(r"[\s/]+", "_", s0)
    s0 = s0.replace("ё", "e")
    s0 = re.sub(r"[^0-9a-zA-Z_\-]+", "_", s0)
    s0 = re.sub(r"__+", "_", s0).strip("_")
    return s0 or "row"

def env() -> Environment:
    tpl_dir = getattr(config, "TEMPLATES_DIR", Path(__file__).parent / "templates")
    e = Environment(
        loader=FileSystemLoader(str(tpl_dir)),
        autoescape=select_autoescape(["html","xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    e.filters["money"] = money
    return e

def generated_line() -> str:
    ts = config.generated_at_tz() if hasattr(config, "generated_at_tz") else datetime.now(tz=TZ).strftime("%d.%m.%Y %H:%M")
    return f"Сформировано: {ts} (Asia/Almaty) | Версия: {__VERSION__}"

def stem_for_out(src: Path) -> str:
    name = src.name
    if name.endswith(".xlsx.__clean.xlsx"):
        return name[:-len(".xlsx.__clean.xlsx")]
    if name.endswith(".xlsx"):
        return name[:-5]
    return src.stem

# ──────────────────────────────────────────────────────────────────────────────
# ДЕТЕКТ ТИПА ОТЧЁТА
DATE_RGX      = re.compile(r"\b\d{2}[./]\d{2}[./]\d{4}\b")
DATEPAIR_RGX  = re.compile(r"(\d{2}[./]\d{2}[./]\d{4}).{0,15}(\d{2}[./]\d{2}[./]\d{4})")
RE_DATE_CELL  = re.compile(r"^\s*\d{2}[./]\d{2}[./]\d{4}\s*$")

def read_header_lines(xlsx: Path, nrows: int = 30) -> List[str]:
    try:
        df = pd.read_excel(xlsx, header=None, nrows=nrows, dtype=str).fillna("")
        return [" ".join([str(v) for v in row if pd.notna(v)]).strip() for _, row in df.iterrows()]
    except Exception:
        return []

def header_period_kind(xlsx: Path) -> str:
    """
    Возвращает: "period" (есть диапазон дат), "single" (одна дата), "" (не нашли).
    """
    for line in read_header_lines(xlsx):
        if DATEPAIR_RGX.search(line):
            return "period"
    for line in read_header_lines(xlsx):
        if DATE_RGX.search(line):
            return "single"
    return ""

def has_date_detailing(xlsx: Path, scan_rows: int = 120) -> bool:
    """Есть ли в теле таблицы колонка дат (много ячеек вида dd.mm.yyyy)."""
    try:
        df = pd.read_excel(xlsx, nrows=scan_rows, dtype=str).fillna("")
        col0 = df.iloc[:, 0].astype(str).tolist()
        hits = sum(1 for v in col0 if RE_DATE_CELL.fullmatch(v.strip()))
        return hits >= 8
    except Exception:
        return False

def detect_manager(xlsx: Path) -> str:
    stem = xlsx.stem
    m1 = re.search(r"(?i)^дебитор[а-яё]*\s+([A-Za-zА-ЯЁа-я]+)", stem)
    if m1: return m1.group(1)
    m2 = re.search(r"(?i)_(?:[A-Za-zА-ЯЁа-я]+\s)?([A-Za-zА-ЯЁа-я]+)\s*\(\d+\)$", stem)
    if m2: return m2.group(1)
    # из шапки
    for line in read_header_lines(xlsx):
        m = re.search(r"(?i)менеджер[:\s]+([A-Za-zА-ЯЁа-я]+)", line)
        if m: return m.group(1)
    return ""

def detect_kind(xlsx: Path) -> Tuple[str, str]:
    """
    Возвращает (kind, manager):
      kind ∈ {"simple","extended","unknown"}
    """
    hp = header_period_kind(xlsx)
    dd = has_date_detailing(xlsx)
    manager = detect_manager(xlsx)

    if hp == "period" and dd:
        return "extended", manager
    if hp == "single" and not dd:
        return "simple", manager
    return "unknown", manager

# ──────────────────────────────────────────────────────────────────────────────
# КОНФИГЫ: менеджеры и ABC
def load_rules() -> Dict[str, Any]:
    path = Path(__file__).parent / "debt_rules.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"abc": {"A": 0.75, "B": 0.90}}

def manager_names() -> List[str]:
    cfg = getattr(config, "MANAGERS_CFG", {}) or {}
    names: List[str] = []
    mgrs = cfg.get("managers") or {}
    if isinstance(mgrs, dict):
        names += list(mgrs.keys())
    syns = cfg.get("synonyms") or {}
    if isinstance(syns, dict):
        for base, variants in syns.items():
            if isinstance(variants, list):
                names += [v for v in variants if isinstance(v, str)]
    # уникализация (lower)
    seen, out = set(), []
    for n in names:
        k = n.strip().lower()
        if k not in seen:
            out.append(n.strip()); seen.add(k)
    return out

def exclude_managers(df: pd.DataFrame, *, col_client: str, manager_hint: str | None) -> Tuple[pd.DataFrame, int]:
    base = manager_names()
    if manager_hint and manager_hint not in base:
        base.append(manager_hint)
    if not base:
        return df, 0
    pats = [re.compile(fr"^(?:{re.escape(n)})\s*\d*$", re.IGNORECASE) for n in base]
    s = df[col_client].astype(str).str.strip()
    mask = pd.Series(False, index=df.index)
    for p in pats:
        mask = mask | s.str.fullmatch(p)
    return df.loc[~mask].copy(), int(mask.sum())

# ──────────────────────────────────────────────────────────────────────────────
# ПРОСТОЙ: TOP-15, ABC, доли
def abc_mark(rows: List[Dict[str, Any]], *, abc_cfg: Dict[str, float]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    A = float(abc_cfg.get("A", 0.75))
    B = float(abc_cfg.get("B", 0.90))
    total = sum(max(0.0, float(r["debt"])) for r in rows)
    if total <= 0:
        for r in rows: r["share_pct"] = 0.0; r["abc"] = "C"
        return rows, {"A":0.0,"B":0.0,"C":0.0}

    rows_sorted = sorted(rows, key=lambda r: r["debt"], reverse=True)
    cum = 0.0
    for r in rows_sorted:
        share = max(0.0, float(r["debt"])) / total
        cum += share
        r["share_pct"] = round(share * 100, 2)
        r["abc"] = "A" if cum <= A else ("B" if cum <= B else "C")
    abc_sum = {"A":0.0,"B":0.0,"C":0.0}
    for r in rows_sorted:
        abc_sum[r["abc"]] += max(0.0, float(r["debt"]))
    return rows_sorted, abc_sum

def build_simple(core_xlsx: Path, *, src_name: str, manager_hint: str) -> Path:
    df, errors = parse_debt_report(core_xlsx)
    cols = {c.lower(): c for c in df.columns}
    c_client = cols.get("клиент") or cols.get("контрагент") or cols.get("client") or cols.get("контрагент/клиент")
    c_final  = cols.get("кон") or cols.get("сальдо кон") or cols.get("задолженность")
    if not c_client or not c_final:
        raise SystemExit("Обязательные колонки не найдены (клиент/кон)")

    df2, dropped = exclude_managers(df, col_client=c_client, manager_hint=manager_hint or None)
    df2 = df2.copy()
    df2[c_final] = pd.to_numeric(df2[c_final], errors="coerce").fillna(0.0)

    pos = df2[df2[c_final] > 0].copy()
    pos["__client"] = pos[c_client].astype(str)
    pos["__debt"]   = pos[c_final].astype(float)

    rows = [{"client": r["__client"], "client_slug": slugify(r["__client"]), "debt": float(r["__debt"])}
            for r in pos.to_dict("records")]

    # ABC + TOP-15
    rules = load_rules()
    rows_marked, abc_sum = abc_mark(rows, abc_cfg=rules.get("abc", {}))
    top15 = rows_marked[:15]
    total_debt = sum(r["debt"] for r in rows_marked)

    # контекст
    period = "—"  # в простом — одна дата в шапке (не детализируем)
    manager = manager_hint or ("Сводный отчёт" if not manager_hint else manager_hint)

    ctx = {
        "title": "ОТЧЁТ ПО ДЕБИТОРСКОЙ ЗАДОЛЖЕННОСТИ",
        "period": period,
        "manager": (manager_hint if manager_hint else "Сводный отчёт"),
        "client_count": int(len(df2)),
        "total_debt": float(total_debt),
        "trend": "—",
        "top_debtors": top15,
        "silent_rows": [],      # в простом «дни» не считаем/не показываем
        "closed_rows": [],
        "overpay_rows": [{"client": r[c_client], "client_slug": slugify(r[c_client]), "overpay": float(r[c_final])}
                         for r in df2[df2[c_final] < 0].to_dict("records")],
        "all_rows": [{"client": r[c_client], "client_slug": slugify(r[c_client]), "debt": float(r[c_final])}
                     for r in df2.sort_values(c_final, ascending=False).to_dict("records")],
        "abc_summary": abc_sum,
        "generated": generated_line(),
        "tech_info": {
            "Источник": src_name,
            "Clean": core_xlsx.name,
            "Строк (после фильтра)": int(len(df2)),
            "Сброшено (менеджеры)": dropped,
            "Ошибки математики": len(errors),
            "Модуль": __VERSION__,
        },
    }

    # рендер
    out_dir = getattr(config, "OUT_DIR", Path(__file__).parent / "reports" / "html")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem_for_out(Path(src_name))}_debt.html"

    log.info("HTML → %s", out_path.name)
    html = env().get_template("debt.html").render(**ctx)
    out_path.write_text(html, encoding="utf-8")
    log.info("✔ HTML: %s", out_path)
    print(f"✔ HTML: {out_path}")
    return out_path

# ──────────────────────────────────────────────────────────────────────────────
# РАСШИРЕННЫЙ: блоки + «дни молчания» + JSON
def _last_activity_date(block: X.ClientBlock) -> Optional[pd.Timestamp]:
    if getattr(block, "last_date", None) is not None:
        return block.last_date
    # fallback: по последнему движению
    if block.movements:
        return max((m.date for m in block.movements if m.date is not None), default=None)
    return None

def build_extended(core_xlsx: Path, *, src_name: str) -> Path:
    # парсинг расширенного из твоего модуля
    blocks, pmin, pmax, agg = X.parse_excel(core_xlsx)
    period_end = pmax or pmin

    # формирую JSON-структуру с «днями молчания»
    jrows: List[Dict[str, Any]] = []
    for b in blocks:
        lad = _last_activity_date(b)
        days_silence = None
        if period_end and lad:
            try:
                days_silence = int((period_end.date() - lad.date()).days)
                if days_silence < 0:
                    days_silence = 0
            except Exception:
                days_silence = None
        calc_close, delta = b.compute()
        jrows.append({
            "subgroup": b.subgroup,
            "client": b.client,
            "opening": b.opening or 0.0,
            "debit": b.sum_debit,
            "credit": b.sum_credit,
            "closing": b.closing if b.closing is not None else calc_close,
            "delta": float(delta),
            "adjustments": b.adjustments(),
            "days_silence": days_silence,
        })

    # агрегаты
    summary = X.aggregate(blocks)
    total = float(summary.get("close", 0.0))

    # HTML-контекст
    ctx = {
        "title": "РАСШИРЕННЫЙ ОТЧЁТ ПО ДЕБИТОРСКОЙ ЗАДОЛЖЕННОСТИ",
        "period": " — ".join(filter(None, (X.fmt_date(pmin), X.fmt_date(pmax)))),
        "manager": detect_manager(core_xlsx) or "—",
        "client_count": int(summary.get("n_clients", len(blocks))),
        "total_debt": total,
        "trend": summary.get("trend", "—"),
        "top_debtors": [],   # в расширенном отдаём всё по клиентам с «днями»
        "silent_rows": [{"client": r["client"], "client_slug": slugify(r["client"]),
                         "debt": float(r["closing"]), "days": r["days_silence"]}
                        for r in jrows if r["days_silence"] is not None and r["closing"] > 0 and r["days_silence"] > 0],
        "closed_rows": [],   # при необходимости можно сформировать по |отгрузка-оплата|<=1 на блоках
        "overpay_rows": [{"client": r["client"], "client_slug": slugify(r["client"]),
                          "overpay": float(r["closing"])} for r in jrows if r["closing"] < 0],
        "all_rows": [{"client": r["client"], "client_slug": slugify(r["client"]),
                      "debt": float(r["closing"]), "days": r["days_silence"],
                      "ship": float(r["debit"]), "pay": float(r["credit"])} for r in jrows],
        "abc_summary": {},  # ABC делаем в простом
        "generated": generated_line(),
        "tech_info": {
            "Источник": src_name,
            "Clean": core_xlsx.name,
            "Клиентов": int(summary.get("n_clients", len(blocks))),
            "Δ-аггр": float(summary.get("delta", 0.0)),
            "Модуль": __VERSION__,
        },
    }

    # пути
    out_html_dir = getattr(config, "OUT_DIR",  Path(__file__).parent / "reports" / "html")
    out_json_dir = Path(__file__).parent / "reports" / "json"
    out_html_dir.mkdir(parents=True, exist_ok=True)
    out_json_dir.mkdir(parents=True, exist_ok=True)

    stem = stem_for_out(Path(src_name))
    html_path = out_html_dir / f"debt_ext_{stem}.html"
    json_path = out_json_dir / f"debt_ext_{stem}.json"

    # вывод
    log.info("HTML → %s", html_path.name)
    html = env().get_template("debt_ext.html").render(**ctx)
    html_path.write_text(html, encoding="utf-8")
    log.info("JSON → %s", json_path.name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": jrows, "generated": ctx["generated"], "version": __VERSION__}, f, ensure_ascii=False, indent=2)

    log.info("✔ HTML: %s", html_path)
    log.info("✔ JSON: %s", json_path)
    print(f"✔ HTML: {html_path}")
    print(f"✔ JSON: {json_path}")
    return html_path

# ──────────────────────────────────────────────────────────────────────────────
# ПУБЛИЧНЫЕ API (для совместимости с ботом/CLI)
def build_simple_report(xlsx: str | Path) -> Path:
    src = Path(xlsx).resolve()
    log.info("SRC %s", src.name)
    clean = ensure_clean_xlsx(src, force_fix=True)
    log.info("Clean copy → %s", clean.name)
    _, manager = detect_kind(clean)
    return build_simple(clean, src_name=src.name, manager_hint=manager)

def build_extended_report(xlsx: str | Path) -> Path:
    src = Path(xlsx).resolve()
    log.info("SRC %s", src.name)
    clean = ensure_clean_xlsx(src, force_fix=True)
    log.info("Clean copy → %s", clean.name)
    return build_extended(clean, src_name=src.name)

def build_auto(xlsx: str | Path) -> Path:
    src = Path(xlsx).resolve()
    log.info("SRC %s", src.name)
    clean = ensure_clean_xlsx(src, force_fix=True)
    log.info("Clean copy → %s", clean.name)
    kind, manager = detect_kind(clean)
    if kind == "extended":
        return build_extended(clean, src_name=src.name)
    elif kind == "simple":
        return build_simple(clean, src_name=src.name, manager_hint=manager)
    else:
        # по умолчанию — простой без «дней»
        log.warning("Тип не распознан, строю ПРОСТОЙ.")
        return build_simple(clean, src_name=src.name, manager_hint=manager)

# ──────────────────────────────────────────────────────────────────────────────
# CLI
def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage:\n  python debt_unified.py <file.xlsx>\n  python debt_unified.py --simple <file.xlsx>\n  python debt_unified.py --extended <file.xlsx>")
        return 2
    if argv[1] == "--simple" and len(argv) >= 3:
        build_simple_report(argv[2]); return 0
    if argv[1] == "--extended" and len(argv) >= 3:
        build_extended_report(argv[2]); return 0
    build_auto(argv[1]); return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

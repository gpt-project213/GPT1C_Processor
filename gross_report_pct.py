#!/usr/bin/env python
# coding: utf-8
r"""
gross_report_pct.py · v1.5.7 · 2025-09-27 (Asia/Almaty)

ИЗМЕНЕНО: Логика извлечения метаданных полностью унифицирована с gross_report.py.
Удалены старые функции, исправлен вызов в основной функции.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging, re
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from zoneinfo import ZoneInfo
from utils_excel import ensure_clean_xlsx

ROOT    = Path(__file__).resolve().parent
OUT_DIR = ROOT / "reports" / "html"
TPL_DIR = ROOT / "templates"
LOG_DIR = ROOT / "logs"
for d in (OUT_DIR, TPL_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

LOG = logging.getLogger("gross_pct")
if not LOG.handlers:
    LOG.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s, %(levelname)s %(message)s")
    fh = logging.FileHandler(LOG_DIR / "gross_report_pct.log", encoding="utf-8", mode="a"); fh.setFormatter(fmt); LOG.addHandler(fh)
    sh = logging.StreamHandler(); sh.setFormatter(fmt); LOG.addHandler(sh)

ENV = Environment(loader=FileSystemLoader(str(TPL_DIR)), autoescape=select_autoescape(["html","xml"]))
TPL = ENV.get_template("gross_percent.html")

NBSP = "\u202f"
TOTAL_RE   = re.compile(r"^(итог|итого|всего|total)\b", re.I)
SERVICE_RE = re.compile(r"(группиров|показател|дополнит)", re.I)

TOP_N = 15
DELTA_RATE_TOL = 0.10

COLS = {
    "product": ("номенк","товар","наимен","наименование","продукт","артикул"),
    "sale":    ("стоим","прод","выруч","реализ","продаж","выручка","сумма продаж","сумма реализации","оборот"),
    "cost":    ("себест","себестоим","затрат","расход","издержк","себестоимость"),
    "gp":      ("валов","прибыл","доход","прибыль","валовая прибыль","gross profit"),
    "margin":  ("рентаб","рентабель","маржа","марж","рентабельность","%"),
}

def _clean(s: str) -> str:
    return str(s or "").replace("\u00a0"," ").replace("\u202f"," ").strip().lower()

def _money_to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(r"[^\d.-]", "", regex=True).str.replace(",", "."),
        errors="coerce"
    ).fillna(0.0)

def _fmt_pct(x: float | int) -> str:
    return f"{float(x):,.2f}".replace(",", NBSP).replace(".", ",") + " %"

def _make_unique(headers: List[str]) -> List[str]:
    used: Dict[str, int] = {}
    out: List[str] = []
    for h in headers:
        key = _clean(h) or "col"
        used[key] = used.get(key, 0) + 1
        out.append(key if used[key] == 1 else f"{key}_{used[key]}")
    return out

def _find_hdr_row(df: pd.DataFrame, max_scan: int = 40) -> int:
    best_i, best_cnt = 0, -1
    for i in range(min(max_scan, len(df))):
        row = df.iloc[i].astype(str)
        if SERVICE_RE.search(" ".join(row)):
            continue
        cnt = int(row.replace("", pd.NA).notna().sum())
        if cnt > best_cnt:
            best_cnt, best_i = cnt, i
    if best_cnt < 0:
        raise ValueError("Не найдена строка заголовка")
    return best_i

def _map_cols(headers: List[str]) -> Dict[str, str]:
    used: set[str] = set()
    m: Dict[str, str] = {}
    h_clean = {h: _clean(h) for h in headers}
    for k, syns in COLS.items():
        for h, hc in h_clean.items():
            if h in used:
                continue
            if any(s in hc for s in syns):
                m[k] = h; used.add(h); break
    if "product" not in m:
        raise ValueError("Нет обязательной колонки: product")
    return m

# =================================================================
# НАЧАЛО БЛОКА: ЕДИНАЯ ЛОГИКА ИЗВЛЕЧЕНИЯ МЕТАДАННЫХ (ЭТАЛОН)
# =================================================================
def _try_extract_meta(xlsx: Path) -> Tuple[str, str]:
    """ИСПРАВЛЕННАЯ финальная версия парсинга периода и менеджера."""
    period, manager = "—", "—"
    
    # Словарь нормализации месяцев в именительный падеж
    MONTHS_NORM = {
        "январ": "Январь", "января": "Январь", "январь": "Январь",
        "феврал": "Февраль", "февраля": "Февраль", "февраль": "Февраль",
        "март": "Март", "марта": "Март",
        "апрел": "Апрель", "апреля": "Апрель", "апрель": "Апрель",
        "май": "Май", "мая": "Май",
        "июн": "Июнь", "июня": "Июнь", "июнь": "Июнь",
        "июл": "Июль", "июля": "Июль", "июль": "Июль",
        "август": "Август", "августа": "Август",
        "сентябр": "Сентябрь", "сентября": "Сентябрь", "сентябрь": "Сентябрь",
        "октябр": "Октябрь", "октября": "Октябрь", "октябрь": "Октябрь",
        "ноябр": "Ноябрь", "ноября": "Ноябрь", "ноябрь": "Ноябрь",
        "декабр": "Декабрь", "декабря": "Декабрь", "декабрь": "Декабрь",
    }
    
    # Словарь месяцев в родительном падеже (для дат с числом)
    MONTHS_GENITIVE = {
        "январ": "января", "января": "января", "январь": "января",
        "феврал": "февраля", "февраля": "февраля", "февраль": "февраля",
        "март": "марта", "марта": "марта",
        "апрел": "апреля", "апреля": "апреля", "апрель": "апреля",
        "май": "мая", "мая": "мая",
        "июн": "июня", "июня": "июня", "июнь": "июня",
        "июл": "июля", "июля": "июля", "июль": "июля",
        "август": "августа", "августа": "августа",
        "сентябр": "сентября", "сентября": "сентября", "сентябрь": "сентября",
        "октябр": "октября", "октября": "октября", "октябрь": "октября",
        "ноябр": "ноября", "ноября": "ноября", "ноябрь": "ноября",
        "декабр": "декабря", "декабря": "декабря", "декабрь": "декабря",
    }
    
    try:
        head = pd.read_excel(xlsx, header=None, nrows=25, dtype=str).fillna("")

        # Собираем первые ~25 строк в один плоский текст
        rows = []
        for _, row in head.iterrows():
            s = " ".join(map(str, row)).strip()
            if s and s.lower() != "nan":
                rows.append(s)
        all_text = " ".join(rows)

        LOG.info(f"Полный текст для парсинга (первые 500 символов): {all_text[:500]}...")

        # --- ПЕРИОД ---
        period_patterns = [
            # 1) "Период: Сентябрь 2025 г." или "Период: 11 октября 2025 г."
            re.compile(
                r"период:\s*"
                r"(\d+\s+)?"  # ЗАХВАТЫВАЕМ число дня (группа 1)
                r"(январ[ья]?|феврал[ья]?|марта?|апрел[ья]?|мая?|июн[я]?|июл[я]?|августа?|сентябр[ья]?|октябр[ья]?|ноябр[ья]?|декабр[ья]?)"  # группа 2
                r"\s+(\d{4})\s*г\.?",  # группа 3
                re.IGNORECASE,
            ),
            # 2) "Период: 01.09.2025 - 10.09.2025"
            re.compile(
                r"период:\s*(\d{2}\.\d{2}\.\d{4})\s*[-–—]\s*(\d{2}\.\d{2}\.\d{4})",
                re.IGNORECASE,
            ),
            # 3) Диапазон без слова "Период": "01.09.2025 - 10.09.2025"
            re.compile(
                r"(\d{2}\.\d{2}\.\d{4})\s*[-–—]\s*(\d{2}\.\d{2}\.\d{4})"
            ),
        ]

        for i, pattern in enumerate(period_patterns, start=1):
            m = pattern.search(all_text)
            if m:
                LOG.info(f"Паттерн периода #{i} сработал: {m.groups()}")
                if i == 1:
                    day_part = m.group(1)  # может быть "11 " или None
                    month_raw = m.group(2).lower()
                    year = m.group(3)
                    
                    if day_part:
                        # Дата с числом: "11 октября 2025 г."
                        day_num = day_part.strip()
                        month_gen = MONTHS_GENITIVE.get(month_raw, month_raw.lower())
                        period = f"{day_num} {month_gen} {year} г."
                    else:
                        # Только месяц: "Октябрь 2025 г."
                        month_clean = MONTHS_NORM.get(month_raw, month_raw.capitalize())
                        period = f"{month_clean} {year} г."
                elif i in (2, 3):
                    period = f"{m.group(1)} — {m.group(2)}"
                LOG.info(f"Найден период: '{period}'")
                break

        # --- МЕНЕДЖЕР ---
        # Бизнес-кейс: "Покупатель В группе из списка (Арман;"
        m = re.search(
            r"покупатель\s+в\s+группе\s+из\s+списка\s*\(([^;)]+)",
            all_text,
            re.IGNORECASE,
        )
        if m:
            LOG.info(f"Паттерн менеджера сработал: {m.groups()}")
            name = m.group(1).strip()
            if name:
                manager = name.title()
                LOG.info(f"Найден менеджер: '{manager}'")

    except Exception as e:
        LOG.error(f"Ошибка в парсинге метаданных: {e}", exc_info=True)

    LOG.info(f"ИТОГ ПАРСИНГА: период='{period}', менеджер='{manager}'")
    return period, manager

# =================================================================
# КОНЕЦ БЛОКА
# =================================================================

def _find_excel_totals_anywhere(df: pd.DataFrame, cols: Dict[str, str]) -> Tuple[Optional[float], Optional[float]]:
    sale_t = gp_t = None
    try:
        for i in range(len(df)):
            row = df.iloc[i].astype(str)
            row_clean = [_clean(x) for x in row]
            if any(TOTAL_RE.match(x or "") for x in row_clean):
                sale_t = gp_t = None
                if "sale" in cols:
                    sale_t = _money_to_float(pd.Series([row[cols["sale"]]])).iloc[0]
                if "gp" in cols:
                    gp_t = _money_to_float(pd.Series([row[cols["gp"]]])).iloc[0]
        return sale_t, gp_t
    except Exception as e:
        LOG.info("Excel total parse fail (pct): %r", e)
        return None, None

# ── Билдер ────────────────────────────────────────────────────────────────
def build_gross_report_percent(xlsx: str | Path) -> Optional[Path]:
    xlsx = Path(xlsx)
    if not xlsx.exists():
        LOG.error("Файл не найден: %s", xlsx); return None

    clean_xlsx = ensure_clean_xlsx(xlsx, force_fix=True)  # <-- СТРОКА 188: СНАЧАЛА очищаем
    period, manager = _try_extract_meta(clean_xlsx)  # <-- СТРОКА 189: ПОТОМ читаем метаданные
    
    # БЛОКИРОВКА СВОДНЫХ ОТЧЕТОВ СНЯТА (логика удаления)
    # if manager == "—":
    #    LOG.info("Summary report detected → percent-html skipped")

    raw = pd.read_excel(clean_xlsx, header=None, dtype=str).fillna("")
    hdr = _find_hdr_row(raw)
    headers = _make_unique([str(c) for c in raw.iloc[hdr]])
    df = raw.copy(); df.columns = headers

    skip_line = _clean(" ".join(df.iloc[hdr + 1].astype(str)))
    body = df.iloc[hdr + 2:] if ("ед." in skip_line or "ндс" in skip_line) else df.iloc[hdr + 1:]
    body = body.dropna(how="all")
    cols = _map_cols(list(body.columns))

    pcol = cols["product"]
    tmp = body.assign(_p=body[pcol].astype(str).map(_clean))
    tmp = tmp[~tmp["_p"].str.match(TOTAL_RE, na=False)]
    tmp = tmp[tmp["_p"] != ""].drop(columns="_p").reset_index(drop=True)

    sale = gp = None
    if "sale" in cols:
        tmp[cols["sale"]] = sale = _money_to_float(tmp[cols["sale"]])
    if "gp" in cols:
        tmp[cols["gp"]] = gp = _money_to_float(tmp[cols["gp"]])

    if "margin" in cols:
        margin_s = (tmp[cols["margin"]].astype(str)
                    .str.replace(r"[^\d.-]", "", regex=True)
                    .str.replace(",", "."))
        tmp["__margin_row"] = pd.to_numeric(margin_s, errors="coerce")
    elif (sale is not None) and (gp is not None):
        tmp["__margin_row"] = (tmp[cols["gp"]] / tmp[cols["sale"]].replace(0, pd.NA) * 100)
    else:
        tmp["__margin_row"] = pd.NA

    overall_margin = None
    if (sale is not None) and (gp is not None):
        s_sum = float(sale.sum()); gp_sum = float(gp.sum())
        overall_margin = (gp_sum / s_sum * 100.0) if s_sum else 0.0

    sale_t, gp_t = _find_excel_totals_anywhere(df, cols)
    rate_note = "OK"
    if sale_t and gp_t and sale_t != 0:
        rate_excel = (float(gp_t) / float(sale_t) * 100.0)
        delta_pp = abs(rate_excel - (overall_margin or 0.0))
        if delta_pp > DELTA_RATE_TOL:
            rate_note = f"Δ={delta_pp:.2f} п.п."

    base = tmp.copy()
    base["product"] = tmp[pcol].astype(str)
    base["margin"]  = base["__margin_row"].astype(float)

    neg_show = pd.DataFrame([])
    if "gp" in cols:
        neg_idx = tmp[tmp[cols["gp"]] < 0].index
        neg_show = base.loc[neg_idx].sort_values("margin").head(TOP_N)[["product", "margin"]]

    low_cand = base.dropna(subset=["margin"]).sort_values("margin")
    if not neg_show.empty:
        low_cand = low_cand.loc[~low_cand.index.isin(neg_show.index)]
    low_sorted = low_cand.head(TOP_N)[["product", "margin"]]

    top_sorted = base.dropna(subset=["margin"]).sort_values("margin", ascending=False).head(TOP_N)[["product", "margin"]]

    def _mk_tbl(dfv: pd.DataFrame) -> str:
        if dfv.empty:
            return '<div class="table-wrap"><table><tbody></tbody></table></div>'
        out = ['<div class="table-wrap"><table>',
               "<thead><tr><th>#</th><th>Товар</th><th>%</th></tr></thead><tbody>"]
        for i, r in dfv.reset_index(drop=True).iterrows():
            out.append(f"<tr><td>{i+1}</td><td>{str(r['product'])}</td><td>{_fmt_pct(float(r['margin'])) if pd.notna(r['margin']) else ''}</td></tr>")
        out.append("</tbody></table></div>")
        return "".join(out)

    low_parts: List[str] = []
    if not neg_show.empty:
        low_parts.append(f'<div class="section">Отрицательная рентабельность — {len(neg_show)} позиций</div>')
        low_parts.append(_mk_tbl(neg_show))
    low_parts.append(_mk_tbl(low_sorted))
    low_html = "".join(low_parts)

    top_html = _mk_tbl(top_sorted)

    total_rows = int(len(body))
    filtered_rows = int(len(body) - len(tmp))
    nan_margin = int(pd.isna(base["margin"]).sum())
    neg_gp = int((tmp[cols["gp"]] < 0).sum()) if "gp" in cols else 0
    tech_lines = [
        f"Сверка рентабельности: {rate_note}",
        f"Статистика: всего строк={total_rows}; отфильтровано={filtered_rows}; строк без процента={nan_margin}; строк с отрицательной валовой прибылью={neg_gp}",
        f"Источник файла: {xlsx.name}",
    ]
    tech_html = "<br>".join(tech_lines)
    
    manager_label = manager if (manager and manager != "—") else "Сводный отчёт"

    ctx = {
        "title": "Рентабельность (проценты)",
        "generated": datetime.now(ZoneInfo('Asia/Almaty')).strftime("%d.%m.%Y %H:%M"),
        "period": period,
        "manager": manager_label,
        "overall_margin": overall_margin,
        "top_table": top_html,
        "low_table": low_html,
        "top_title": f"ТОП-{TOP_N} рентабельности",
        "low_title": f"Низкая рентабельность — ТОП-{TOP_N}",
        "tech_block": tech_html,
    }

# --- ИМЯ ВЫХОДНОГО ФАЙЛА: без .xlsx.__clean и с менеджером ---
    orig_name = Path(xlsx).name
    # убираем .xlsx и .xlsx.__clean из конца
    base_stem = re.sub(r"\.xlsx(?:\.__clean)?$", "", orig_name, flags=re.IGNORECASE)

    # Если менеджер распознан — пишем менеджерский файл, иначе — сводный по исходному stem
    if manager and manager != "—":
        # базовая унификация, чтобы имена были как у gross_sum
        # оставляем канонику "Валовая прибыль {Менеджер}_gross_pct.html"
        out_name = f"Валовая прибыль {manager}_gross_pct.html"
    else:
        out_name = f"{base_stem}_gross_pct.html"

    out = OUT_DIR / out_name

    html = TPL.render(**ctx)
    out.write_text(html, encoding="utf-8")
    LOG.info("Процентный отчёт сохранён: %s", out)
    return out

__all__ = ["build_gross_report_percent"]

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit("Usage: python gross_report_pct.py <file.xlsx>")
    build_gross_report_percent(sys.argv[1])

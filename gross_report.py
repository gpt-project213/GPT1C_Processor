#!/usr/bin/env python
# coding: utf-8
"""
gross_report.py · v27.7 · 2025-09-27 (Asia/Almaty)

ИЗМЕНЕНО: Функция _try_extract_meta заменена на финальную, унифицированную версию.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import html as _html
import logging, re
from zoneinfo import ZoneInfo
from utils_excel import ensure_clean_xlsx

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

# ── Пути/логирование ──────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent
OUT_DIR = ROOT / "reports" / "html"
TPL_DIR = ROOT / "templates"
LOG_DIR = ROOT / "logs"
for d in (OUT_DIR, TPL_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

LOG = logging.getLogger("gross_full")
if not LOG.handlers:
    LOG.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s, %(levelname)s %(message)s")
    fh = logging.FileHandler(LOG_DIR / "gross_report.log", encoding="utf-8", mode="a"); fh.setFormatter(fmt); LOG.addHandler(fh)
    sh = logging.StreamHandler(); sh.setFormatter(fmt); LOG.addHandler(sh)

ENV = Environment(loader=FileSystemLoader(str(TPL_DIR)), autoescape=select_autoescape(["html","xml"]))
_TPL = None

def _get_tpl():
    global _TPL
    if _TPL is None:
        _TPL = ENV.get_template("gross.html")
    return _TPL

NBSP = "\u202f"
TOTAL_RE    = re.compile(r"^(итог|итого|всего|total)\b", re.I)
SERVICE_RE  = re.compile(r"(группиров|показател|дополнит)", re.I)
DATE_RNG_RE = re.compile(r"(\d{2}[./]\d{2}[./]\d{4}).{0,12}(\d{2}[./]\d{2}[./]\d{4})")

# Пороги и константы
DELTA_TENGE_TOL = 50.0
DELTA_RATE_TOL  = 0.10
TOP_N = 15

COLS = {
    "product": ("номенк","товар","наимен","наименование","продукт","артикул"),
    "qty":     ("колич","количество","кол-во","кол во","кол","ед"),
    "sale":    ("стоим","прод","выруч","реализ","продаж","выручка","сумма продаж","сумма реализации","оборот"),
    "cost":    ("себест","себестоим","затрат","расход","издержк","себестоимость"),
    "gp":      ("валов","прибыл","доход","прибыль","валовая прибыль","gross profit"),
    "margin":  ("рентаб","рентабель","маржа","марж","рентабельность","%"),
}

# ── Утилиты ────────────────────────────────────────────────────────────
def _clean(s: str) -> str:
    return str(s or "").replace("\u00a0", " ").replace("\u202f", " ").strip().lower()

def _money_to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(r"[^\d.,-]", "", regex=True).str.replace(",", "."),
        errors="coerce"
    ).fillna(0.0)

def _fmt_money(x: float | int) -> str:
    return f"{float(x):,.2f}".replace(",", NBSP).replace(".", ",")

def _fmt_pct(x: float | int) -> str:
    return f"{float(x):,.2f}".replace(",", NBSP).replace(".", ",") + " %"

def _make_unique(headers: List[str]) -> List[str]:
    used: Dict[str, int] = {}
    out: List[str] = []
    for h in headers:
        key = _clean(h)
        used[key] = used.get(key, 0) + 1
        out.append(key if used[key] == 1 else f"{key}_{used[key]}")
    return out

def _find_hdr_row(df: pd.DataFrame) -> int:
    for i in range(min(40, len(df))):
        row = df.iloc[i]
        if sum(bool(c) for c in row) < 3:
            continue
        if any(SERVICE_RE.search(str(c)) for c in row):
            continue
        row_clean = [_clean(x) for x in row]
        if any(any(s in c for s in COLS["product"]) for c in row_clean):
            return i
    raise ValueError("Не найдена строка заголовка")

def _map_cols(cols: List[str]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    used = set()
    for k, syn in COLS.items():
        for c in cols:
            if c in used:
                continue
            if any(s in c for s in syn):
                m[k] = c
                used.add(c)
                break
    if "product" not in m:
        raise ValueError("Нет обязательной колонки: product")
    if not (("sale" in m) and ("cost" in m) and ("gp" in m)):
        LOG.warning("Не найдены все денежные колонки: sale/cost/gp — вычисления могут быть частичны")
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
        LOG.info("Excel total parse fail: %r", e)
        return None, None

# ── Основной билдер ────────────────────────────────────────────────────
def build_gross_report(xlsx: str | Path) -> Optional[Path]:
    xlsx = Path(xlsx)
    if not xlsx.exists():
        LOG.error("Файл не найден: %s", xlsx)
        return None

    clean_xlsx = ensure_clean_xlsx(xlsx, force_fix=True)  # <-- СТРОКА 202: СНАЧАЛА очищаем
    period, manager = _try_extract_meta(clean_xlsx)  # <-- СТРОКА 203: ПОТОМ читаем метаданные
    raw = pd.read_excel(clean_xlsx, header=None, dtype=str).fillna("")
    hdr = _find_hdr_row(raw)
    headers = _make_unique([str(c) for c in raw.iloc[hdr]])
    df = raw.copy()
    df.columns = headers

    skip_line = _clean(" ".join(df.iloc[hdr + 1].astype(str)))
    body = df.iloc[hdr + 2:] if ("ед." in skip_line or "ндс" in skip_line) else df.iloc[hdr + 1:]
    body = body.dropna(how="all")
    cols = _map_cols(list(body.columns))

    pcol = cols["product"]
    tmp = body.assign(_p=body[pcol].astype(str).map(_clean))
    tmp = tmp[~tmp["_p"].str.match(TOTAL_RE, na=False)]
    tmp = tmp[tmp["_p"] != ""].drop(columns="_p").reset_index(drop=True)

    if "sale" in cols:
        tmp[cols["sale"]] = _money_to_float(tmp[cols["sale"]])
    else:
        tmp["__sale_f"] = 0.0; cols["sale"] = "__sale_f"
    if "cost" in cols:
        tmp[cols["cost"]] = _money_to_float(tmp[cols["cost"]])
    else:
        tmp["__cost_f"] = 0.0; cols["cost"] = "__cost_f"
    if "gp" in cols:
        tmp[cols["gp"]] = _money_to_float(tmp[cols["gp"]])
    else:
        tmp["__gp_f"] = tmp[cols["sale"]] - tmp[cols["cost"]]; cols["gp"] = "__gp_f"

    if "margin" in cols:
        margin_s = (tmp[cols["margin"]].astype(str)
                    .str.replace(r"[^\d.,-]", "", regex=True)
                    .str.replace(",", "."))
        tmp["__margin_row"] = pd.to_numeric(margin_s, errors="coerce")
    else:
        tmp["__margin_row"] = (tmp[cols["gp"]] / tmp[cols["sale"]].replace(0, pd.NA) * 100)

    # fail-safe dedup
    before_sale = float(tmp[cols["sale"]].sum())
    before_gp   = float(tmp[cols["gp"]].sum())
    removed_dups = 0
    dedup_decision = "kept(0)"
    try:
        key_cols = [pcol, cols["sale"], cols["cost"], cols["gp"]]
        dedup = tmp.drop_duplicates(subset=key_cols, keep="first")
        removed_dups = int(len(tmp) - len(dedup))
        if removed_dups > 0:
            after_sale = float(dedup[cols["sale"]].sum())
            after_gp   = float(dedup[cols["gp"]].sum())
            sale_excel, gp_excel = _find_excel_totals_anywhere(df, cols)
            revert = False
            if sale_excel is not None and abs(after_sale - sale_excel) > abs(before_sale - sale_excel) + DELTA_TENGE_TOL:
                revert = True
            if gp_excel is not None and abs(after_gp - gp_excel) > abs(before_gp - gp_excel) + DELTA_TENGE_TOL:
                revert = True
            if not revert:
                tmp = dedup.reset_index(drop=True); dedup_decision = "kept"
            else:
                dedup_decision = "reverted"
    except Exception as e:
        dedup_decision = f"error:{e!r}"
        LOG.info("Dedup error: %r", e)

    sale_sum = float(tmp[cols["sale"]].sum())
    cost_sum = float(tmp[cols["cost"]].sum())
    gp_sum   = float(tmp[cols["gp"]].sum())
    overall_margin = (gp_sum / sale_sum * 100) if sale_sum else 0.0

    sale_excel, gp_excel = _find_excel_totals_anywhere(df, cols)
    delta_line = ""
    delta_rate_line = ""
    if sale_excel is not None:
        d_sale = sale_sum - sale_excel; delta_line += f"Δsale={_fmt_money(d_sale)}"
    if gp_excel is not None:
        d_gp = gp_sum - gp_excel; delta_line += (", " if delta_line else "") + f"Δgp={_fmt_money(d_gp)}"
    if sale_excel and gp_excel:
        excel_rate = (gp_excel / sale_excel * 100) if sale_excel else 0.0
        d_rate = overall_margin - excel_rate
        delta_rate_line = f"Δрентаб={_fmt_pct(d_rate).replace(' ', '')}"
        if abs(d_rate) > DELTA_RATE_TOL:
            LOG.info("Rate delta exceeds tol: calc=%s excel=%s d=%+.2f п.п.",
                     _fmt_pct(overall_margin), _fmt_pct(excel_rate), d_rate)
    if delta_line or delta_rate_line:
        LOG.info("Сверка Excel: sale_calc=%s ; gp_calc=%s ; %s %s",
                 _fmt_money(sale_sum), _fmt_money(gp_sum),
                 delta_line, f"({delta_rate_line})" if delta_rate_line else "")

    top = tmp.sort_values(cols["gp"], ascending=False).head(TOP_N)
    top_tbl = [
        {"#": i+1,
         "product": _html.escape(str(r[pcol])),
         "sale": r[cols["sale"]],
         "cost": r[cols["cost"]],
         "gp": r[cols["gp"]],
         "margin": float(r["__margin_row"]) if pd.notna(r["__margin_row"]) else None}
        for i, r in top.reset_index(drop=True).iterrows()
    ]
    def _fmt_money_td(v: float) -> str: return _fmt_money(v)
    def _fmt_pct_td(v: Optional[float]) -> str: return "" if v is None else _fmt_pct(v)
    top10_html = (
        '<div class="table-wrap"><table>'
        '<thead><tr><th>#</th><th>Товар</th><th>Выручка</th><th>Себестоимость</th><th>Валовая прибыль</th><th>%</th></tr></thead><tbody>'
        + "".join(
            f"<tr><td>{r['#']}</td><td>{r['product']}</td>"
            f"<td>{_fmt_money_td(r['sale'])}</td><td>{_fmt_money_td(r['cost'])}</td>"
            f"<td>{_fmt_money_td(r['gp'])}</td><td>{_fmt_pct_td(r.get('margin'))}</td></tr>"
            for r in top_tbl
        )
        + "</tbody></table></div>"
    )

    neg = tmp[tmp[cols["gp"]] < 0].sort_values(cols["gp"])
    neg_show = neg.head(TOP_N)

    low = tmp.sort_values("__margin_row")
    if not neg_show.empty:
        low = low.loc[~low.index.isin(neg_show.index)]
    low = low.head(TOP_N)

    def _tbl(rows: pd.DataFrame) -> str:
        if rows.empty:
            return '<div class="table-wrap"><table><tbody></tbody></table></div>'
        rows = rows.reset_index(drop=True)
        parts = ['<div class="table-wrap"><table>',
                 '<thead><tr><th>#</th><th>Товар</th><th>Выручка</th><th>Себестоимость</th><th>Валовая прибыль</th><th>%</th></tr></thead><tbody>']
        for i, r in rows.iterrows():
            parts.append(
                f"<tr><td>{i+1}</td><td>{_html.escape(str(r[pcol]))}</td>"
                f"<td>{_fmt_money(float(r[cols['sale']]))}</td>"
                f"<td>{_fmt_money(float(r[cols['cost']]))}</td>"
                f"<td>{_fmt_money(float(r[cols['gp']]))}</td>"
                f"<td>{'' if pd.isna(r['__margin_row']) else _fmt_pct(float(r['__margin_row']))}</td></tr>"
            )
        parts.append("</tbody></table></div>")
        return "".join(parts)

    loss_parts: List[str] = []
    if not neg_show.empty:
        loss_parts.append(f'<div class="section">Отрицательная рентабельность — {len(neg_show)} позиций</div>')
        loss_parts.append(_tbl(neg_show))
    loss_parts.append(_tbl(low))
    loss_html = "".join(loss_parts)

    abc = tmp[[pcol, cols["gp"]]].copy()
    abc.columns = ["product", "gp"]
    abc = abc.sort_values("gp", ascending=False).reset_index(drop=True)
    abc["cum"] = abc["gp"].cumsum()
    total_gp = abc["gp"].sum() or 1.0
    abc["share"] = abc["cum"] / total_gp * 100.0
    def _bucket(x: float) -> str:
        if x <= 80.0: return "A"
        if x <= 95.0: return "B"
        return "C"
    abc["ABC"] = abc["share"].map(_bucket)
    abc_count = abc.groupby("ABC").size().reindex(["A","B","C"], fill_value=0)
    abc_sum   = abc.groupby("ABC")["gp"].sum().reindex(["A","B","C"], fill_value=0.0)
    abc_html = (
        '<div class="table-wrap"><table>'
        '<thead><tr><th>Группа</th><th>Кол-во позиций</th><th>Сумма прибыли</th><th>Доля прибыли</th></tr></thead><tbody>'
        f'<tr><td>A</td><td>{int(abc_count.get("A",0))}</td><td>{_fmt_money(float(abc_sum.get("A",0.0)))}</td><td>до 80%</td></tr>'
        f'<tr><td>B</td><td>{int(abc_count.get("B",0))}</td><td>{_fmt_money(float(abc_sum.get("B",0.0)))}</td><td>80–95%</td></tr>'
        f'<tr><td>C</td><td>{int(abc_count.get("C",0))}</td><td>{_fmt_money(float(abc_sum.get("C",0.0)))}</td><td>95–100%</td></tr>'
        "</tbody></table></div>"
    )

    summary_html = (
        f'<div class="kv"><div><span class="k">Выручка</span><span class="v">{_fmt_money(sale_sum)}</span></div>'
        f'<div><span class="k">Себестоимость</span><span class="v">{_fmt_money(cost_sum)}</span></div>'
        f'<div><span class="k">Валовая прибыль</span><span class="v">{_fmt_money(gp_sum)}</span></div>'
        f'<div><span class="k">Рентабельность</span><span class="v">{_fmt_pct(overall_margin)}</span></div></div>'
    )

    total_rows = int(len(body))
    filtered_rows = int(len(body) - len(tmp))
    nan_margin = int(pd.isna(tmp["__margin_row"]).sum())
    neg_gp_cnt = int((tmp[cols["gp"]] < 0).sum())
    tech_lines = []
    if sale_excel is not None or gp_excel is not None:
        excel_total = f"S={_fmt_money(sale_excel) if sale_excel is not None else '—'}; GP={_fmt_money(gp_excel) if gp_excel is not None else '—'}"
        tech_lines.append(f"Итог Excel: {excel_total}")
    if delta_line:
        tech_lines.append(f"Сверка: {delta_line}" + (f" ({delta_rate_line})" if delta_rate_line else ""))
    tech_lines.append(
        f"Статистика: всего строк={total_rows}; отфильтровано={filtered_rows}; "
        f"строк без процента={nan_margin}; строк с отрицательной валовой прибылью={neg_gp_cnt}; "
        f"dedup_removed={removed_dups}; dedup={dedup_decision}"
    )
    generated_local = datetime.now(ZoneInfo("Asia/Almaty")).strftime("%d.%m.%Y %H:%M")
    tech_lines.append(f"Источник файла: {xlsx.name}")
    tech_lines.append(f"Сформировано: {generated_local} (Asia/Almaty)")
    tech_html = "<br>".join(tech_lines)

    manager_label = manager if (manager and manager != "—") else "Сводный отчёт"

    ctx = {
        "title": "Отчёт по валовой прибыли",
        "period": period,
        "manager": manager_label,
        "generated": generated_local,
        "file_name": "Отчёт по валовой прибыли",
        "summary": summary_html,
        "abc_table": abc_html,
        "top10_profit": top10_html,
        "loss_title": f"Низкая рентабельность — ТОП-{TOP_N}",
        "loss_table": loss_html,
        "tech_block": tech_html,
    }

    out = OUT_DIR / f"{xlsx.stem}_gross_sum.html"
    html = _get_tpl().render(**ctx)
    out.write_text(html, encoding="utf-8")
    LOG.info("✔ Отчёт сохранён: %s", out)
    return out

__all__ = ["build_gross_report"]

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit("Usage: python gross_report.py <file.xlsx>")
    build_gross_report(sys.argv[1])

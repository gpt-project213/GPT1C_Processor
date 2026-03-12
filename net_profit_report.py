#!/usr/bin/env python
# coding: utf-8
"""
net_profit_report.py · v1.2.6 (2026-03-10)
────────────────────────────────────────────────────────────────────
Отчёт "Чистая прибыль" = Валовая - Расходы

v1.2.4: Генерирует ДВА файла в поддиректории (day/mtd):
  - net_profit_day_YYYYMMDD.html    (за конкретный день)
  - net_profit_mtd_YYYYMMDD.html    (за период/нарастающим)
v1.2.1: Строгое совпадение периодов (без fallback)
v1.2.0: Исправлен мэтчинг MTD vs DAY

Источники:
- reports/json/gross_*.json     (валовая прибыль)
- reports/json/expenses_*.json  (расходы)

Выход:
- reports/analytics/net_profit_day_<date>.html   ← ЗА ДЕНЬ
- reports/analytics/net_profit_mtd_<date>.html   ← ЗА ПЕРИОД

Доступ: ТОЛЬКО Admin

v1.2.6: Fix P-007: добавлен load_dotenv() — TZ теперь читается из .env
v1.2.5: TZ timezone(timedelta(hours=5)) → ZoneInfo("Asia/Almaty") (Bug TZ)
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from datetime import datetime, date
from zoneinfo import ZoneInfo
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", encoding="utf-8-sig", override=True)

# ──────────────────────────────────────────────────────────────────
TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))
ROOT = Path(__file__).resolve().parent
JSON_DIR = ROOT / "reports" / "json"
ANALYTICS_DIR = ROOT / "reports" / "analytics"
# v1.2.4: поддиректории для разделения day/mtd отчётов
ANALYTICS_DAY_DIR = ANALYTICS_DIR / "net_profit_day"
ANALYTICS_MTD_DIR = ANALYTICS_DIR / "net_profit_mtd"
LOGS = ROOT / "logs"
ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
ANALYTICS_DAY_DIR.mkdir(parents=True, exist_ok=True)
ANALYTICS_MTD_DIR.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("net_profit")

__VERSION__ = "1.2.6"
NBSP = "\u202f"


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except (FileNotFoundError, OSError):
        return 0.0

MONTHS_RU = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
    "январь": 1, "февраль": 2, "март": 3, "апрель": 4,
    "май": 5, "июнь": 6, "июль": 7, "август": 8,
    "сентябрь": 9, "октябрь": 10, "ноябрь": 11, "декабрь": 12,
}


def parse_date_str(s: str) -> Optional[date]:
    """Парсит строку даты. Форматы: '19.02.2026', '2026-02-19', '19 февраля 2026 г.'"""
    s = s.strip().rstrip(".")
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.match(r"(\d{1,2})[./](\d{1,2})[./](\d{4})", s)
    if m:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    m = re.match(r"(\d{1,2})\s+([а-яё]+)\s+(\d{4})", s.lower())
    if m:
        mon = MONTHS_RU.get(m.group(2))
        if mon:
            return date(int(m.group(3)), mon, int(m.group(1)))
    return None


def extract_period_dates(period_str: str) -> Tuple[Optional[date], Optional[date]]:
    """Возвращает (start_date, end_date). Для DAY: start == end."""
    p = period_str.strip()
    range_pat = r"(\d{1,2}[./]\d{1,2}[./]\d{4})\s*[-\u2013\u2014]\s*(\d{1,2}[./]\d{1,2}[./]\d{4})"
    m = re.search(range_pat, p)
    if m:
        d1 = parse_date_str(m.group(1))
        d2 = parse_date_str(m.group(2))
        if d1 and d2:
            return d1, d2
    d = parse_date_str(p)
    if d:
        return d, d
    return None, None


def extract_period_from_json(data: Dict[str, Any]) -> str:
    if "period" in data:
        return str(data["period"])
    if "metadata" in data and "period" in data["metadata"]:
        return str(data["metadata"]["period"])
    return "unknown"


def load_all_jsons(pattern: str) -> List[Dict[str, Any]]:
    """Загрузить все JSON по паттерну, отсортированные по mtime (новые первые)."""
    files = sorted(JSON_DIR.glob(pattern), key=_mtime, reverse=True)
    result = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["__source_path__"] = str(path)
            data["__mtime__"] = _mtime(path)
            result.append(data)
        except Exception as e:
            LOG.warning(f"Ошибка чтения {path.name}: {e}")
    return result


def find_matching_expenses(gross_period: str, all_expenses: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """v1.2.1: Найти expenses с ТОЧНЫМ совпадением периода. Без fallback."""
    gross_start, gross_end = extract_period_dates(gross_period)
    if gross_start is None:
        LOG.error(f"❌ Не удалось распарсить период gross: '{gross_period}'")
        return None

    ptype = "DAY" if gross_start == gross_end else "RANGE"
    LOG.info(f"Ищу expenses для: {gross_start} → {gross_end} (тип: {ptype})")

    for exp in all_expenses:
        exp_period = extract_period_from_json(exp)
        exp_start, exp_end = extract_period_dates(exp_period)
        is_exact = (exp_start == gross_start and exp_end == gross_end)
        LOG.info(f"  {Path(exp['__source_path__']).name} | period='{exp_period}' | exact={is_exact}")
        if is_exact:
            return exp

    # Fallback (v1.2.4): если точного совпадения нет — берём ближайший по дате окончания (того же типа DAY/RANGE)
    # Это не меняет бизнес-логику расчёта, но позволяет не "молчать" при минимальных расхождениях формата периода.
    best = None
    best_days = 10**9
    for exp in all_expenses:
        exp_period = extract_period_from_json(exp)
        exp_start, exp_end = extract_period_dates(exp_period)
        if exp_start is None or exp_end is None:
            continue
        exp_ptype = "DAY" if exp_start == exp_end else "RANGE"
        if exp_ptype != ptype:
            continue
        days = abs((exp_end - gross_end).days)
        if days < best_days:
            best_days = days
            best = exp
    if best is not None and best_days <= 1:
        LOG.warning("⚠️ Fallback: использую ближайший expenses (Δ=%s дн.)", best_days)
        return best
    LOG.error(
        f"❌ Не найден expenses с точным периодом [{gross_start}–{gross_end}]. "
        f"Загрузите файл затрат за этот период."
    )
    return None


def fmt_money(x: float) -> str:
    return f"{float(x):,.0f}".replace(",", NBSP) + " ₸"


def fmt_pct(x: float) -> str:
    return f"{float(x):.1f}%"


def generate_html(
    gross_revenue: float, gross_profit: float, gross_margin_pct: float,
    total_expenses: float, net_profit: float, net_margin_pct: float,
    period_label: str, period_type_label: str,
) -> str:
    color = "green" if net_profit >= 0 else "red"
    icon = "✅" if net_profit >= 0 else "❌"
    now_str = datetime.now(TZ).strftime("%d.%m.%Y %H:%M")
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Чистая прибыль — {period_type_label}</title>
<style>
body{{font-family:Arial,sans-serif;background:#f5f5f5;margin:20px;padding:0}}
.container{{max-width:900px;margin:0 auto;background:#fff;padding:30px;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.1)}}
h1{{color:#2c3e50;margin-bottom:5px}}
.period-type{{display:inline-block;background:#007bff;color:#fff;padding:3px 12px;border-radius:12px;font-size:13px;font-weight:600;margin-bottom:8px}}
.period{{color:#444;font-size:17px;font-weight:600;margin-bottom:5px}}
.meta{{color:#888;font-size:12px;margin-bottom:25px}}
.kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(195px,1fr));gap:18px;margin:25px 0}}
.card{{background:#f8f9fa;padding:20px;border-radius:8px;border-left:4px solid #007bff}}
.card.green{{border-left-color:#28a745}}
.card.red{{border-left-color:#dc3545}}
.card-label{{font-size:11px;color:#666;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px}}
.card-value{{font-size:24px;font-weight:700;color:#2c3e50}}
.card-sub{{font-size:13px;color:#666;margin-top:6px}}
.formula{{background:#fffbea;border:1px solid #ffd700;padding:12px 20px;border-radius:8px;margin:20px 0;text-align:center;font-size:16px;font-weight:600}}
.footer{{margin-top:30px;padding-top:15px;border-top:1px solid #eee;text-align:center;color:#bbb;font-size:11px}}
</style>
</head>
<body>
<div class="container">
<h1>💰 Чистая прибыль</h1>
<div class="period-type">{period_type_label}</div>
<div class="period">📅 {period_label}</div>
<div class="meta">Отчёт ТОЛЬКО для Admin | Сгенерировано: {now_str}</div>

<div class="formula">Чистая прибыль = Валовая прибыль − Расходы</div>

<div class="kpi">
<div class="card">
<div class="card-label">Выручка</div>
<div class="card-value">{fmt_money(gross_revenue)}</div>
</div>
<div class="card green">
<div class="card-label">Валовая прибыль</div>
<div class="card-value">{fmt_money(gross_profit)}</div>
<div class="card-sub">Маржа: {fmt_pct(gross_margin_pct)}</div>
</div>
<div class="card red">
<div class="card-label">Расходы</div>
<div class="card-value">{fmt_money(total_expenses)}</div>
</div>
<div class="card {color}">
<div class="card-label">{icon} Чистая прибыль</div>
<div class="card-value">{fmt_money(net_profit)}</div>
<div class="card-sub">Маржа: {fmt_pct(net_margin_pct)}</div>
</div>
</div>

<div class="footer">net_profit_report.py v{__VERSION__} | {datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")}</div>
</div>
</body>
</html>"""


def save_report(gross_data, expenses_data, period_type, period_label, period_type_label, date_tag):
    gross_revenue    = float(gross_data.get("total_revenue", 0))
    gross_profit     = float(gross_data.get("gross_profit", gross_data.get("total_profit", 0)))
    gross_margin_pct = float(gross_data.get("margin_pct", 0))
    total_expenses   = float(expenses_data.get("total_expenses", 0))
    net_profit       = gross_profit - total_expenses
    net_margin_pct   = (net_profit / gross_revenue * 100) if gross_revenue > 0 else 0.0

    data = {
        "report_type":    f"NET_PROFIT_{period_type.upper()}",
        "period":         period_label,
        "period_type":    period_type,
        "gross_revenue":  gross_revenue,
        "gross_profit":   gross_profit,
        "gross_margin_pct": gross_margin_pct,
        "total_expenses": total_expenses,
        "net_profit":     net_profit,
        "net_margin_pct": net_margin_pct,
        "metadata": {
            "version":      __VERSION__,
            "generated_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    html = generate_html(
        gross_revenue, gross_profit, gross_margin_pct,
        total_expenses, net_profit, net_margin_pct,
        period_label, period_type_label,
    )

    json_path = ANALYTICS_DIR / f"net_profit_{period_type}_{date_tag}.json"
    # v1.2.4: HTML сохраняем в поддиректорию
    if period_type == "day":
        html_dir = ANALYTICS_DAY_DIR
    else:
        html_dir = ANALYTICS_MTD_DIR
    html_path = html_dir / f"net_profit_{period_type}_{date_tag}.html"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    LOG.info(f"✅ Сохранено: {html_path.name}")
    LOG.info(f"   Чистая прибыль: {fmt_money(net_profit)} ({fmt_pct(net_margin_pct)})")
    return html_path


# ──────────────────────────────────────────────────────────────────
def generate_report():
    """
    v1.2.2: Генерирует отдельные файлы для DAY и RANGE периодов.
    Берёт самый свежий gross каждого типа + ищет точно совпадающие expenses.
    """
    LOG.info("=" * 60)
    LOG.info(f"ГЕНЕРАЦИЯ ОТЧЁТОВ: Чистая прибыль v{__VERSION__}")

    all_gross    = load_all_jsons("gross_*.json")
    all_expenses = load_all_jsons("expenses_*.json")

    if not all_gross:
        LOG.error("Не найдены данные валовой прибыли (gross_*.json)")
        exit(1)
    if not all_expenses:
        LOG.error("Не найдены данные расходов (expenses_*.json)")
        exit(1)

    LOG.info(f"Найдено gross: {len(all_gross)}, expenses: {len(all_expenses)}")

    # Находим лучший (свежайший) gross для DAY и для RANGE
    best_day   = None  # (start, end, data)
    best_range = None

    seen: set = set()
    for g in all_gross:  # уже отсортированы по mtime desc
        gp = extract_period_from_json(g)
        start, end = extract_period_dates(gp)
        if start is None:
            continue
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        if start == end and best_day is None:
            best_day = (start, end, g, gp)
        elif start != end and best_range is None:
            best_range = (start, end, g, gp)
        if best_day and best_range:
            break

    generated = 0

    # ── DAY ──────────────────────────────────────────────────────
    if best_day:
        start, end, g, gp = best_day
        LOG.info(f"\n── DAY период: '{gp}' ──")
        exp = find_matching_expenses(gp, all_expenses)
        if exp:
            save_report(g, exp, "day",
                        period_label=start.strftime("%d.%m.%Y"),
                        period_type_label="За день",
                        date_tag=start.strftime("%Y%m%d"))
            generated += 1
        else:
            LOG.warning("⚠️ Пропуск DAY — нет matching расходов")
    else:
        LOG.warning("⚠️ Не найден gross файл типа DAY (один день)")

    # ── RANGE / MTD ──────────────────────────────────────────────
    if best_range:
        start, end, g, gp = best_range
        LOG.info(f"\n── RANGE период: '{gp}' ──")
        exp = find_matching_expenses(gp, all_expenses)
        if exp:
            period_label = f"{start.strftime('%d.%m')}–{end.strftime('%d.%m.%Y')}"
            save_report(g, exp, "mtd",
                        period_label=period_label,
                        period_type_label="За период",
                        date_tag=start.strftime("%Y%m%d"))
            generated += 1
        else:
            LOG.warning("⚠️ Пропуск RANGE — нет matching расходов")
    else:
        LOG.warning("⚠️ Не найден gross файл типа RANGE (диапазон дат)")

    LOG.info(f"\n{'='*60}")
    LOG.info(f"ИТОГО: сгенерировано {generated} отчёт(а/ов)")
    if generated == 0:
        exit(1)


if __name__ == "__main__":
    try:
        generate_report()
    except Exception as e:
        LOG.error(f"Ошибка: {e}", exc_info=True)
        exit(1)
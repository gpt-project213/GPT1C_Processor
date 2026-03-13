#!/usr/bin/env python
# coding: utf-8
"""
dso_aging_report.py · v1.1.1 (2026-03-10)
FIX #DSO-1: период из JSON, не /30
FIX #DSO-2: aging из days_silence, не синтетика
FIX #DSO-3: sales по периоду, не по max revenue
Fix P-016: _mtime() helper — p.stat().st_mtime обёрнут в try/except (FileNotFoundError, OSError)
────────────────────────────────────────────────────────────────────
Отчёт "DSO + Aging дебиторки"

DSO = Days Sales Outstanding (средний срок оплаты)
Aging = структура долгов по срокам

Источники:
- reports/json/debt_*.json   (дебиторка по клиентам)
- reports/json/sales_*.json  (продажи для расчёта DSO)

Выход:
- reports/analytics/dso_<manager>_<date>.html

Показывает:
- DSO по каждому менеджеру
- Структура долгов: 0-7, 8-14, 15-30, >30 дней
- Проблемные клиенты (>30 дней)

Доступ: Admin + Subadmin (для помощи с взысканием)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Any, Optional
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", encoding="utf-8-sig", override=True)

# ──────────────────────────────────────────────────────────────────
TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))
ROOT = Path(__file__).resolve().parent
JSON_DIR = ROOT / "reports" / "json"
ANALYTICS_DIR = ROOT / "reports" / "analytics"
LOGS = ROOT / "logs"
ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("dso")

__VERSION__ = "1.1.1"
NBSP = "\u202f"

def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except (FileNotFoundError, OSError):
        return 0.0

# ──────────────────────────────────────────────────────────────────
def load_latest_json(pattern: str, skip_keywords: list = None, min_clients: int = 0) -> Optional[Dict[str, Any]]:
    files = sorted(JSON_DIR.glob(pattern), key=_mtime, reverse=True)
    if not files:
        return None
    for path in files:
        if skip_keywords and any(kw.lower() in path.name.lower() for kw in skip_keywords):
            LOG.warning(f"Пропускаю: {path.name}")
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if min_clients > 0 and len(data.get("clients", [])) < min_clients:
                LOG.warning(f"Пропускаю {path.name}: клиентов < {min_clients}")
                continue
            LOG.info(f"Загружаю: {path.name}")
            return data
        except Exception as e:
            LOG.warning(f"Ошибка чтения {path.name}: {e}")
    return None


def _parse_period_date(s: str):
    """Парсит дату 'dd.mm.yyyy'. Возвращает datetime или None."""
    from datetime import datetime as _dt
    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            return _dt.strptime(s.strip(), fmt)
        except ValueError:
            pass
    return None


def _period_days(period_min: str, period_max: str) -> int:
    """
    FIX Bug #DSO-1: реальный период в днях вместо хардкодного 30.
    Считает кол-во дней между period_min и period_max включительно.
    Fallback = 30 если даты не распознаны.
    """
    d1 = _parse_period_date(period_min or "")
    d2 = _parse_period_date(period_max or "")
    if d1 and d2 and d2 >= d1:
        return (d2 - d1).days + 1
    LOG.warning("Не удалось распознать период ('%s' - '%s'), fallback=30", period_min, period_max)
    return 30


def load_best_sales_json(min_clients: int = 3,
                          debt_period_min: str = None,
                          debt_period_max: str = None) -> Optional[Dict[str, Any]]:
    """
    FIX Bug #DSO-3: выбирает sales по совпадению периода с долгом, не по max(revenue).

    Приоритет:
    1. Файл с периодом, пересекающимся с периодом дебиторки
    2. Fallback: файл с max(revenue) — логируем предупреждение
    """
    import re as _re
    files = sorted(JSON_DIR.glob("sales_*.json"), key=_mtime, reverse=True)
    d_min = _parse_period_date(debt_period_min or "")
    d_max = _parse_period_date(debt_period_max or "")

    candidates_match = []   # совпадение периода
    candidates_all   = []   # все пригодные файлы

    for p in files:
        name_lower = p.name.lower()
        if "товару" in name_lower or "profitability" in name_lower:
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            clients = data.get("clients", []) or []
            if len(clients) < min_clients:
                continue
            rev = float(data.get("total_revenue", 0.0) or 0.0)
            candidates_all.append((rev, p, data))

            # Проверяем пересечение периодов
            if d_min and d_max:
                period_str = data.get("period", "") or ""
                m = _re.search(r"(\d{2}\.\d{2}\.\d{4})[^\d]+(\d{2}\.\d{2}\.\d{4})", period_str)
                if m:
                    s1 = _parse_period_date(m.group(1))
                    s2 = _parse_period_date(m.group(2))
                    if s1 and s2 and s2 >= d_min and s1 <= d_max:
                        candidates_match.append((rev, p, data))
        except Exception as e:
            LOG.warning("Ошибка чтения %s: %s", p.name, e)

    for pool, label in [(candidates_match, "период"), (candidates_all, "max revenue (fallback)")]:
        if pool:
            rev, p, data = max(pool, key=lambda x: x[0])
            if "fallback" in label:
                LOG.warning("DSO-3 fallback: period не совпал, берём %s (revenue=%.0f)", p.name, rev)
            else:
                LOG.info("DSO-3: выбран sales по '%s': %s", label, p.name)
            return data
    return None

def _ext_to_simple_clients(ext: dict) -> list:
    """Конвертирует clients из debt_ext формата в упрощённый."""
    result = []
    for c in (ext.get("clients") or []):
        if not isinstance(c, dict):
            continue
        result.append({
            "client":  c.get("client"),
            "debt":    c.get("debt", c.get("closing")),
            "opening": c.get("opening"),
            "debit":   c.get("debit"),
            "credit":  c.get("credit"),
            "days_silence": c.get("days_silence"),
        })
    return result


def load_best_debt_json() -> Optional[Dict[str, Any]]:
    """
    Мёржит ВСЕ debt_ext_*.json файлы с наиболее свежим period_max.

    Проблема: при одинаковом mtime load_latest_json возвращал только ОДИН
    файл (напр. только Оксану), и DSO генерировался только для одного менеджера.
    Теперь берём все файлы, у которых period_max = макс. period_max по архиву,
    и объединяем клиентов из всех — DSO покрывает всех менеджеров.
    """
    files = sorted(JSON_DIR.glob("debt_ext_*.json"), key=_mtime, reverse=True)
    if not files:
        LOG.error("Нет debt_ext_*.json файлов")
        return None

    # Шаг 1: определяем самый свежий period_max (сравниваем как даты, не строки)
    best_period_max      = None
    best_period_max_date = None
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            pmax = data.get("period_max") or ""
            if not pmax:
                continue
            pmax_date = _parse_period_date(pmax)
            if pmax_date is None:
                continue
            if best_period_max_date is None or pmax_date > best_period_max_date:
                best_period_max      = pmax
                best_period_max_date = pmax_date
        except Exception as e:
            LOG.warning("load_best_debt_json: ошибка чтения %s: %s", path.name, e)

    if not best_period_max:
        LOG.warning("load_best_debt_json: не удалось определить period_max, беру первый файл")
        try:
            with open(files[0], "r", encoding="utf-8") as fh:
                ext = json.load(fh)
            return {"clients": _ext_to_simple_clients(ext),
                    "total_debt": (ext.get("aggregates") or {}).get("close"),
                    "period_min": ext.get("period_min"),
                    "period_max": ext.get("period_max"),
                    "manager": ext.get("manager")}
        except Exception:
            return None

    # Шаг 2: мёржим всех клиентов из файлов с этим period_max
    merged_clients: list = []
    merged_total   = 0.0
    ref_period_min = None
    loaded = 0
    seen_clients: set = set()

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("period_max") != best_period_max:
                continue
            clients = _ext_to_simple_clients(data)
            for c in clients:
                key = (c.get("client") or "").strip().lower()
                if key and key not in seen_clients:
                    seen_clients.add(key)
                    merged_clients.append(c)
            total = (data.get("aggregates") or {}).get("close") or 0.0
            merged_total += float(total)
            if ref_period_min is None:
                ref_period_min = data.get("period_min")
            loaded += 1
            LOG.info("load_best_debt_json: добавляю %s (period %s - %s, %d клиентов)",
                     path.name, data.get("period_min"), data.get("period_max"), len(clients))
        except Exception as e:
            LOG.warning("load_best_debt_json: ошибка %s: %s", path.name, e)

    LOG.info("load_best_debt_json: итого файлов=%d клиентов=%d period_max=%s",
             loaded, len(merged_clients), best_period_max)
    return {
        "clients":    merged_clients,
        "total_debt": merged_total,
        "period_min": ref_period_min,
        "period_max": best_period_max,
    }

def fmt_money(x: float) -> str:
    return f"{float(x):,.0f}".replace(",", NBSP) + " ₸"

def get_manager_from_client(client_name: str) -> str:
    from config import get_manager_by_client_prefix
    return get_manager_by_client_prefix(client_name)

# ──────────────────────────────────────────────────────────────────
def generate_report():
    LOG.info("="*60)
    LOG.info("ГЕНЕРАЦИЯ ОТЧЁТА: DSO + Aging")
    
    # Загрузка дебиторки
    debt_data = load_best_debt_json()
    if not debt_data:
        LOG.error("Нет данных о дебиторке")
        return

    # FIX Bug #DSO-1: реальный период из debt JSON
    period_min  = debt_data.get("period_min") or ""
    period_max  = debt_data.get("period_max") or ""
    period_days = _period_days(period_min, period_max)
    LOG.info("DSO-1 fix: period_min=%s period_max=%s -> %d дней", period_min, period_max, period_days)

    # FIX Bug #DSO-3: подбираем sales по совпадению периода
    sales_data = load_best_sales_json(min_clients=3,
                                      debt_period_min=period_min,
                                      debt_period_max=period_max)

    # Дневная выручка по реальному периоду (не 30)
    daily_revenue = 0.0
    if sales_data:
        total_revenue = float(sales_data.get("total_revenue", 0.0) or 0.0)
        daily_revenue = total_revenue / period_days
        LOG.info("Дневная выручка: %.0f / %d = %.0f тг/день", total_revenue, period_days, daily_revenue)
    
    # Группировка по менеджерам
    managers_data = defaultdict(lambda: {
        "total_debt": 0.0,
        "dso": 0.0,
        "aging": {"0-7": 0.0, "8-14": 0.0, "15-30": 0.0, ">30": 0.0},
        "problem_clients": []
    })
    
    # Обработка долгов
    for client_data in debt_data.get("clients", []):
        client_name = (client_data.get("client") or client_data.get("name") or "")
        closing_debt = client_data.get("debt", client_data.get("closing", 0.0))  # debt_auto_report → "debt"
        
        if closing_debt <= 0:
            continue
        
        manager = get_manager_from_client(client_name)
        if manager == "Неизвестно":
            continue
        
        managers_data[manager]["total_debt"] += closing_debt

        # FIX Bug #DSO-2: реальная aging через поле days_silence из debt_ext JSON
        # Убраны синтетические коэффициенты 0.1 / 0.2 / 0.3 / 0.4 / 0.6
        days_silence = int(client_data.get("days_silence") or 0)

        if days_silence > 30:
            managers_data[manager]["aging"][">30"] += closing_debt
            managers_data[manager]["problem_clients"].append({
                "name": client_name,
                "debt": closing_debt,
                "days": days_silence,
            })
        elif days_silence > 14:
            managers_data[manager]["aging"]["15-30"] += closing_debt
        elif days_silence > 7:
            managers_data[manager]["aging"]["8-14"] += closing_debt
        else:
            managers_data[manager]["aging"]["0-7"] += closing_debt
    
    # Расчёт DSO
    for manager, data in managers_data.items():
        if daily_revenue > 0:
            data["dso"] = data["total_debt"] / daily_revenue
    
    # Генерация отчётов
    for manager, data in managers_data.items():
        dso = data["dso"]
        
        # Оценка DSO
        if dso >= 20:
            dso_status = "🚨 МЕДЛЕННО"
            dso_color = "#dc3545"
        elif dso >= 15:
            dso_status = "⚠️ СРЕДНЕ"
            dso_color = "#ffc107"
        else:
            dso_status = "✅ БЫСТРО"
            dso_color = "#28a745"
        
        # Aging таблица
        aging_rows = ""
        total_debt = data["total_debt"]
        for period, amount in data["aging"].items():
            pct = (amount / total_debt * 100) if total_debt > 0 else 0
            color = "#dc3545" if period == ">30" else "#ffc107" if period == "15-30" else "#28a745"
            aging_rows += f"""
            <tr>
                <td><span style="color:{color};font-weight:600">{period} дней</span></td>
                <td style="text-align:right">{fmt_money(amount)}</td>
                <td style="text-align:right">{pct:.1f}%</td>
            </tr>"""
        
        # Проблемные клиенты
        problem_rows = ""
        for i, client in enumerate(sorted(data["problem_clients"], key=lambda x: x["debt"], reverse=True)[:10], 1):
            problem_rows += f"""
            <tr>
                <td>{i}</td>
                <td>{client['name']}</td>
                <td style="text-align:right;color:#dc3545;font-weight:600">{fmt_money(client['debt'])}</td>
                <td style="text-align:right">{client.get('days', 0)}</td>
            </tr>"""
        
        html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DSO + Aging: {manager}</title>
<style>
body{{font-family:Arial,sans-serif;background:#f0f4f8;margin:0;padding:15px;color:#1a2332;font-size:14px;line-height:1.5}}
.container{{max-width:1000px;margin:0 auto;background:#fff;padding:20px 26px 26px;border-radius:10px;box-shadow:0 2px 10px rgba(26,58,92,.10)}}
.brand-bar{{display:flex;align-items:center;border-bottom:3px solid #1a3a5c;padding-bottom:10px;margin-bottom:18px}}
.brand-name{{font-size:14px;font-weight:800;color:#1a3a5c;letter-spacing:.5px;text-transform:uppercase}}
.brand-name::before{{content:"▲ ";color:#0070c0}}
h1{{color:#1a2332;font-size:21px;margin:0 0 6px}}
h2{{color:#1a2332;font-size:16px;margin:22px 0 8px;padding-bottom:7px;border-bottom:2px solid #0070c0}}
.meta{{color:#64748b;font-size:13px;margin-bottom:16px}}
.dso-card{{padding:24px;border-radius:8px;margin:18px 0;text-align:center;border:2px solid {dso_color};background:rgba(0,0,0,.03)}}
.dso-status{{font-size:16px;font-weight:700;color:{dso_color};margin-bottom:8px}}
.dso-value{{font-size:44px;font-weight:700;color:{dso_color};margin:12px 0}}
.dso-label{{font-size:13px;color:#64748b}}
table{{width:100%;border-collapse:collapse;margin:10px 0}}
th,td{{padding:9px 11px;border-bottom:1px solid #d0d9e8}}
th{{background:#eef2f8;font-weight:600;text-align:left;border-bottom:2px solid #d0d9e8}}
tr:hover{{background:#f5f8fc}}
.alert{{background:#fffbeb;border:1px solid #e09000;padding:12px 14px;border-radius:6px;margin:14px 0}}
a,button{{touch-action:manipulation;-webkit-tap-highlight-color:rgba(0,0,0,.04)}}
.table-wrap{{overflow:auto;border:1px solid #d0d9e8;border-radius:8px;margin:10px 0}}
.footer{{margin-top:20px;padding-top:12px;border-top:1px solid #d0d9e8;text-align:center;color:#64748b;font-size:11px}}
@media(max-width:768px){{body{{padding:8px}}.container{{padding:12px 14px 18px}}h1{{font-size:17px}}h2{{font-size:14px}}table{{min-width:auto!important;table-layout:auto}}th,td{{padding:6px 7px;font-size:12px}}}}
</style>
</head>
<body>
<div class="container">
<div class="brand-bar"><span class="brand-name">AI 1C PRO</span></div>
<h1>💳 DSO + Aging: {manager}</h1>
<div class="meta">Для admin + subadmin | {datetime.now(TZ).strftime("%d.%m.%Y %H:%M")}</div>

<div class="dso-card">
<div class="dso-status">{dso_status}</div>
<div style="font-size:14px;color:#666">DSO (средний срок оплаты):</div>
<div class="dso-value">{dso:.0f} дней</div>
<div class="dso-label">Дебиторка: {fmt_money(total_debt)}</div>
</div>

<div class="alert">
<div style="font-weight:600;margin-bottom:8px">📞 ДЕЙСТВИЕ:</div>
<div style="color:#666">
{"Срок оплаты высокий! Нужна помощь с взысканием долгов. Обратить внимание на проблемных клиентов." if dso >= 20 else
 "Срок оплаты средний. Контролировать ситуацию с долгами >30 дней." if dso >= 15 else
 "Срок оплаты нормальный. Клиенты платят вовремя."}
</div>
</div>

<h2>📊 Aging дебиторки</h2>
<p style="color:#666">Структура долгов по срокам:</p>
<div class="table-wrap"><table>
<thead>
<tr>
<th>Период</th>
<th style="width:150px;text-align:right">Сумма</th>
<th style="width:100px;text-align:right">Доля</th>
</tr>
</thead>
<tbody>
{aging_rows}
</tbody>
</table></div>

<h2>⚠️ Проблемные клиенты (>30 дней)</h2>
<p style="color:#666">Клиенты требующие внимания:</p>
<div class="table-wrap"><table>
<thead>
<tr>
<th style="width:40px">#</th>
<th>Клиент</th>
<th style="width:150px;text-align:right">Долг</th>
<th style="width:70px;text-align:right">Дней</th>
</tr>
</thead>
<tbody>
{problem_rows if problem_rows else '<tr><td colspan="4" style="text-align:center;color:#28a745">Нет проблемных клиентов ✅</td></tr>'}
</tbody>
</table></div>

<div class="footer"><strong>AI 1C PRO</strong> | dso_aging_report.py v{__VERSION__} | {datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")}</div>
</div>
</body>
</html>"""
        
        # Сохранение
        ts = datetime.now(TZ).strftime("%Y%m%d")
        html_path = ANALYTICS_DIR / f"dso_{manager}_{ts}.html"
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        LOG.info(f"✅ Создан отчёт для {manager}")
        LOG.info(f"  DSO: {dso:.0f} дней, Проблемных клиентов: {len(data['problem_clients'])}")

if __name__ == "__main__":
    try:
        generate_report()
    except Exception as e:
        LOG.error(f"Ошибка: {e}", exc_info=True)
        exit(1)
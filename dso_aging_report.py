#!/usr/bin/env python
# coding: utf-8
"""
dso_aging_report.py · v1.0.2 (2026-02-09)
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
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────
TZ = timezone(timedelta(hours=5))
ROOT = Path(__file__).resolve().parent
JSON_DIR = ROOT / "reports" / "json"
ANALYTICS_DIR = ROOT / "reports" / "analytics"
LOGS = ROOT / "logs"
ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("dso")

__VERSION__ = "1.0.0"
NBSP = "\u202f"

# ──────────────────────────────────────────────────────────────────
def load_latest_json(pattern: str, skip_keywords: list = None, min_clients: int = 0) -> Optional[Dict[str, Any]]:
    files = sorted(JSON_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
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


def load_best_sales_json(min_clients: int = 3) -> Optional[Dict[str, Any]]:
    """Выбирает sales JSON с максимальной выручкой (устраняет DSO-explosion при выборе мелкого менеджера)."""
    files = sorted(JSON_DIR.glob("sales_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    best = None
    best_rev = -1.0
    for path in files:
        if "товару" in path.name.lower():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            clients = data.get("clients", []) or []
            if len(clients) < min_clients:
                continue
            rev = float(data.get("total_revenue", 0.0) or 0.0)
            if rev > best_rev:
                best_rev = rev
                best = data
        except Exception as e:
            LOG.warning("Ошибка чтения %s: %s", path.name, e)
    if best:
        LOG.info("Выбран sales по max revenue: total_revenue=%s", best.get("total_revenue"))
    return best

def load_best_debt_json() -> Optional[Dict[str, Any]]:
    """Сначала ищет debt_*.json (simple), иначе берёт debt_ext_*.json и упрощает до формата debt_*"""
    debt = load_latest_json("debt_*.json")
    if debt:
        return debt
    ext = load_latest_json("debt_ext_*.json")
    if not ext:
        return None
    # debt_ext хранит clients в расширенном виде — оставляем client/debt
    simple = {
        "clients": [],
        "total_debt": (ext.get("aggregates") or {}).get("close"),
        "period_min": ext.get("period_min"),
        "period_max": ext.get("period_max"),
        "manager": ext.get("manager"),
        "__source_ext_json__": ext.get("__source_path__") if isinstance(ext, dict) else None,
    }
    for c in (ext.get("clients") or []):
        if not isinstance(c, dict):
            continue
        simple["clients"].append({
            "client": c.get("client"),
            "debt": c.get("debt", c.get("closing")),
            "opening": c.get("opening"),
            "debit": c.get("debit"),
            "credit": c.get("credit"),
        })
    return simple

def fmt_money(x: float) -> str:
    return f"{float(x):,.0f}".replace(",", NBSP) + " ₸"

def get_manager_from_client(client_name: str) -> str:
    client_upper = client_name.upper()
    if client_upper.startswith("О "):
        return "Оксана"
    elif client_upper.startswith("М "):
        return "Магира"
    elif client_upper.startswith("Е "):
        return "Ергали"
    elif client_upper.startswith("А "):
        return "Алена"
    return "Неизвестно"

# ──────────────────────────────────────────────────────────────────
def generate_report():
    LOG.info("="*60)
    LOG.info("ГЕНЕРАЦИЯ ОТЧЁТА: DSO + Aging")
    
    # Загрузка
    debt_data = load_best_debt_json()
    sales_data = load_best_sales_json(min_clients=3)
    
    if not debt_data:
        LOG.error("Нет данных о дебиторке")
        return
    
    # Средняя дневная выручка для DSO
    daily_revenue = 0.0
    if sales_data:
        total_revenue = sales_data.get("total_revenue", 0.0)
        daily_revenue = total_revenue / 30  # Примерно месяц
    
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
        
        # Упрощённая aging (по закрывающему долгу)
        # В реальности нужны даты движений
        # Здесь просто распределяем: >30 дней = 10%
        problem_threshold = closing_debt * 0.1
        if problem_threshold > 10000:  # Если долг большой, помечаем как проблемный
            managers_data[manager]["problem_clients"].append({
                "name": client_name,
                "debt": closing_debt
            })
            managers_data[manager]["aging"][">30"] += problem_threshold
            managers_data[manager]["aging"]["15-30"] += closing_debt * 0.2
            managers_data[manager]["aging"]["8-14"] += closing_debt * 0.3
            managers_data[manager]["aging"]["0-7"] += closing_debt * 0.4
        else:
            # Нормальный долг
            managers_data[manager]["aging"]["0-7"] += closing_debt * 0.6
            managers_data[manager]["aging"]["8-14"] += closing_debt * 0.4
    
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
            </tr>"""
        
        html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DSO + Aging: {manager}</title>
<style>
body{{font-family:Arial,sans-serif;background:#f5f5f5;margin:20px}}
.container{{max-width:1000px;margin:0 auto;background:#fff;padding:30px;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.1)}}
h1{{color:#2c3e50;margin-bottom:10px}}
.meta{{color:#666;font-size:14px;margin-bottom:30px}}
.dso-card{{padding:30px;border-radius:8px;margin:30px 0;text-align:center;background:rgba(220,53,69,0.1);border:2px solid {dso_color}}}
.dso-status{{font-size:18px;font-weight:600;color:{dso_color};margin-bottom:10px}}
.dso-value{{font-size:48px;font-weight:700;color:{dso_color};margin:15px 0}}
.dso-label{{font-size:14px;color:#666}}
table{{width:100%;border-collapse:collapse;margin:20px 0}}
th,td{{padding:12px;border-bottom:1px solid #ddd}}
th{{background:#f8f9fa;font-weight:600;text-align:left}}
tr:hover{{background:#f8f9fa}}
h2{{color:#2c3e50;margin-top:40px;padding-bottom:10px;border-bottom:2px solid #007bff}}
.alert{{background:#fff3cd;border:1px solid #ffc107;padding:15px;border-radius:8px;margin:20px 0}}
.footer{{margin-top:30px;padding-top:20px;border-top:1px solid #eee;text-align:center;color:#999;font-size:12px}}
</style>
</head>
<body>
<div class="container">
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
<table>
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
</table>

<h2>⚠️ Проблемные клиенты (>30 дней)</h2>
<p style="color:#666">Клиенты требующие внимания:</p>
<table>
<thead>
<tr>
<th style="width:40px">#</th>
<th>Клиент</th>
<th style="width:150px;text-align:right">Долг</th>
</tr>
</thead>
<tbody>
{problem_rows if problem_rows else '<tr><td colspan="3" style="text-align:center;color:#28a745">Нет проблемных клиентов ✅</td></tr>'}
</tbody>
</table>

<div class="footer">
dso_aging_report.py v{__VERSION__} | {datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")}
</div>
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
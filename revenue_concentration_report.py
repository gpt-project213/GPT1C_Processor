#!/usr/bin/env python
# coding: utf-8
"""
revenue_concentration_report.py · v1.1.4 (2026-03-10)
FIX: Bug #RC1 - добавлена normalize_client_name (NameError при каждом запуске)
Fix P-016: _mtime() helper — p.stat().st_mtime обёрнут в try/except (FileNotFoundError, OSError)
────────────────────────────────────────────────────────────────────
Отчёт "Концентрация выручки"

Показывает: насколько менеджер зависит от крупных клиентов

Источники:
- reports/json/sales_*.json

Выход:
- reports/analytics/concentration_<manager>_<date>.html (с суммами)
- reports/analytics/concentration_<manager>_pct_<date>.html (только %)

Доступ:
- Admin: видит суммы
- Managers: видят только %
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
LOG = logging.getLogger("concentration")

__VERSION__ = "1.1.4"
NBSP = "\u202f"

def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except (FileNotFoundError, OSError):
        return 0.0

# ──────────────────────────────────────────────────────────────────
from utils_common import normalize_client_name  # Fix #RC-1: вынесено в utils_common.py
def load_all_jsons_merged(pattern: str, min_clients: int = 0, skip_keywords: list = None) -> Optional[Dict[str, Any]]:
    """
    v1.1.1: Загружает ВСЕ sales JSON за один период и мёржит клиентов.

    Исправляет два бага:
    1. Раньше брался только один JSON → только один менеджер в отчёте
    2. Каждый client помечается "_manager" из поля JSON → без prefix-guessing
    """
    files = sorted(JSON_DIR.glob(pattern), key=_mtime, reverse=True)
    if not files:
        LOG.error(f"Нет JSON файлов по паттерну: {pattern}")
        return None

    # Сводные файлы (без реального менеджера) пропускаем.
    # В сводном Excel менеджер отсутствует — это норма, не баг.
    _SKIP_MANAGERS = {"", "не определён", "неизвестно"}

    reference_period = None
    reference_data   = None
    reference_path   = None
    for path in files:
        if skip_keywords and any(kw.lower() in path.name.lower() for kw in skip_keywords):
            LOG.warning(f"Пропускаю (skip_keywords): {path.name}")
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            mgr = (data.get("manager") or "").strip().lower()
            if mgr in _SKIP_MANAGERS:
                LOG.debug(f"Пропускаю сводный файл (нет менеджера): {path.name}")
                continue
            if len(data.get("clients", [])) < min_clients:
                LOG.warning(f"Пропускаю {path.name}: клиентов={len(data.get('clients',[]))} < {min_clients}")
                continue
            reference_period = data.get("period", "")
            reference_data   = data
            reference_path   = path
            LOG.info(f"Эталонный файл: {path.name} | period='{reference_period}' | менеджер='{data.get('manager','?')}'")
            break
        except Exception as e:
            LOG.warning(f"Ошибка чтения {path.name}: {e}")

    if reference_data is None:
        LOG.error(f"Нет подходящего JSON по паттерну: {pattern}")
        return None

    ref_manager    = reference_data.get("manager", "")
    merged_clients = []
    for c in reference_data.get("clients", []):
        c["_manager"] = ref_manager
        merged_clients.append(c)
    merged_revenue = reference_data.get("total_revenue", 0.0)
    loaded_files   = 1

    for path in files:
        if path == reference_path:
            continue
        if skip_keywords and any(kw.lower() in path.name.lower() for kw in skip_keywords):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            mgr = (data.get("manager") or "").strip()
            if mgr.lower() in _SKIP_MANAGERS:
                LOG.debug(f"Пропускаю сводный при добавлении: {path.name}")
                continue
            if data.get("period", "") != reference_period:
                LOG.info(f"Пропускаю {path.name}: period '{data.get('period','')}' ≠ '{reference_period}'")
                continue
            clients = data.get("clients", [])
            if len(clients) < 1:   # порог мёржа = 1, не min_clients
                continue
            for c in clients:
                c["_manager"] = mgr
            merged_clients.extend(clients)
            merged_revenue += data.get("total_revenue", 0.0)
            loaded_files   += 1
            LOG.info(f"Добавляю: {path.name} | менеджер='{mgr}' | +{len(clients)} клиентов")
        except Exception as e:
            LOG.warning(f"Ошибка чтения {path.name}: {e}")

    # ДЕДУП: один и тот же клиент может встретиться в нескольких JSON (сводный + менеджерские, а также дубль-имена файлов).
    # Правило: по нормализованному имени клиента берём запись с МАКСИМАЛЬНОЙ выручкой.
    by_key = {}
    for c in merged_clients:
        if not isinstance(c, dict):
            continue
        key = normalize_client_name(str(c.get("client", "")))
        if not key:
            continue
        rev = float(c.get("revenue", 0.0) or 0.0)
        prev = by_key.get(key)
        if (prev is None) or (rev > float(prev.get("revenue", 0.0) or 0.0)):
            by_key[key] = c
    merged_clients = list(by_key.values())
    LOG.info(f"Итого: файлов={loaded_files}, клиентов={len(merged_clients)}")
    merged = dict(reference_data)
    merged["clients"]       = merged_clients
    merged["total_revenue"] = merged_revenue
    merged["client_count"]  = len(merged_clients)
    return merged

def fmt_money(x: float) -> str:
    return f"{float(x):,.0f}".replace(",", NBSP) + " ₸"

def get_manager_from_client(client_name: str) -> str:
    from config import get_manager_by_client_prefix
    return get_manager_by_client_prefix(client_name)

# ──────────────────────────────────────────────────────────────────
def generate_report():
    LOG.info("="*60)
    LOG.info("ГЕНЕРАЦИЯ ОТЧЁТА: Концентрация выручки")
    
    # v1.1.1: загружаем ВСЕ sales JSON одного периода (все менеджеры)
    sales_data = load_all_jsons_merged("sales_*.json", min_clients=3, skip_keywords=["товару"])
    if not sales_data:
        LOG.error("Нет данных о продажах")
        return
    
    # Группировка по менеджерам
    managers_data = defaultdict(lambda: {"clients": [], "total": 0.0})
    
    for client_data in sales_data.get("clients", []):
        client_name    = client_data.get("client", "")
        client_revenue = client_data.get("total", 0)
        # v1.1.1: берём _manager из JSON — надёжнее prefix-guessing
        manager = client_data.get("_manager") or get_manager_from_client(client_name)
        if not manager:
            manager = "Неизвестно"
        
        if manager != "Неизвестно":
            managers_data[manager]["clients"].append({
                "name": client_name,
                "revenue": client_revenue
            })
            managers_data[manager]["total"] += client_revenue
    
    # Генерация отчётов
    if not managers_data:
        total_clients = sum(1 for _ in sales_data.get("clients", []))
        LOG.error(f"Нет клиентов с распознанным менеджером! Всего в JSON: {total_clients}")
        sample = [c.get("client", "?") for c in sales_data.get("clients", [])[:5]]
        LOG.error(f"Примеры имён клиентов: {sample}")
        LOG.error("Ожидаются имена вида 'О Иванов', 'М Петрова', 'Е Сидоров', 'А Козлова'")
        return
    
    for manager, data in managers_data.items():
        # Сортировка
        data["clients"].sort(key=lambda x: x["revenue"], reverse=True)
        
        total = data["total"]
        top5 = data["clients"][:5]
        top5_sum = sum(c["revenue"] for c in top5)
        top5_pct = (top5_sum / total * 100) if total > 0 else 0
        
        # Определение риска
        if top5_pct >= 70:
            risk = "🚨 ВЫСОКИЙ"
            risk_color = "#dc3545"
        elif top5_pct >= 50:
            risk = "⚠️ СРЕДНИЙ"
            risk_color = "#ffc107"
        else:
            risk = "✅ НИЗКИЙ"
            risk_color = "#28a745"
        
        # Таблица топ-5
        top5_rows = ""
        cumulative_pct = 0.0
        for i, client in enumerate(top5, 1):
            pct = (client["revenue"] / total * 100) if total > 0 else 0
            cumulative_pct += pct
            top5_rows += f"""
            <tr>
                <td>{i}</td>
                <td>{client['name']}</td>
                <td style="text-align:right">{fmt_money(client['revenue'])}</td>
                <td style="text-align:right">{pct:.1f}%</td>
            </tr>"""
        
        # Версия с суммами (для admin)
        html_full = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Концентрация выручки: {manager}</title>
<style>
body{{font-family:Arial,sans-serif;background:#f0f4f8;margin:0;padding:15px;color:#1a2332;font-size:14px;line-height:1.5}}
.container{{max-width:1000px;margin:0 auto;background:#fff;padding:20px 26px 26px;border-radius:10px;box-shadow:0 2px 10px rgba(26,58,92,.10)}}
.brand-bar{{display:flex;align-items:center;border-bottom:3px solid #1a3a5c;padding-bottom:10px;margin-bottom:18px}}
.brand-name{{font-size:14px;font-weight:800;color:#1a3a5c;letter-spacing:.5px;text-transform:uppercase}}
.brand-name::before{{content:"▲ ";color:#0070c0}}
h1{{color:#1a2332;font-size:21px;margin:0 0 8px}}
.meta{{color:#64748b;font-size:14px;margin-bottom:20px;line-height:1.6}}
.alert{{padding:20px;border-radius:8px;margin:20px 0;text-align:center}}
.alert-risk{{font-size:18px;font-weight:600;margin-bottom:10px}}
.alert-value{{font-size:38px;font-weight:700;margin:12px 0}}
table{{width:100%;border-collapse:collapse;margin:16px 0}}
th,td{{padding:8px 10px;border-bottom:1px solid #d0d9e8}}
th{{background:#eef2f8;font-weight:600;text-align:left;border-bottom:2px solid #d0d9e8}}
tr:hover td{{background:#f5f8fc}}
h2{{color:#1a3a5c;margin-top:30px;font-size:17px}}
.footer{{margin-top:20px;padding-top:12px;border-top:1px solid #d0d9e8;text-align:center;color:#64748b;font-size:11px}}
.footer strong{{color:#1a3a5c}}
</style>
</head>
<body>
<div class="container">
<div class="brand-bar"><span class="brand-name">AI 1C PRO</span></div>
<h1>🎯 Концентрация выручки: {manager}</h1>
<div class="meta">ADMIN версия (с суммами) | {datetime.now(TZ).strftime("%d.%m.%Y %H:%M")}</div>

<div class="alert" style="background:rgba(220,53,69,0.1);border:2px solid {risk_color}">
<div class="alert-risk" style="color:{risk_color}">{risk}</div>
<div style="font-size:14px;color:#666;margin-bottom:10px">Топ-5 клиентов дают:</div>
<div class="alert-value" style="color:{risk_color}">{top5_pct:.1f}%</div>
<div style="font-size:14px;color:#666">от общей выручки ({fmt_money(top5_sum)} из {fmt_money(total)})</div>
</div>

<h2>📊 Топ-5 клиентов</h2>
<table>
<thead>
<tr>
<th style="width:40px">#</th>
<th>Клиент</th>
<th style="width:150px;text-align:right">Выручка</th>
<th style="width:100px;text-align:right">Доля</th>
</tr>
</thead>
<tbody>
{top5_rows}
</tbody>
</table>

<div style="background:#f8f9fa;padding:20px;border-radius:8px;margin:30px 0">
<div style="font-weight:600;margin-bottom:10px">💡 Рекомендация:</div>
<div style="color:#666">
{"Концентрация высокая! Один крупный клиент уйдёт = серьёзная потеря выручки. Нужно развивать средних клиентов." if top5_pct >= 70 else 
 "Концентрация умеренная. Есть зависимость от крупных клиентов, но риск управляем." if top5_pct >= 50 else
 "Концентрация низкая. Выручка распределена равномерно — хорошая диверсификация!"}
</div>
</div>

<div class="footer"><strong>AI 1C PRO</strong> | revenue_concentration_report.py v{__VERSION__} | {datetime.now(TZ).strftime("%d.%m.%Y %H:%M")} (Asia/Almaty)</div>
</div>
</body>
</html>"""

        # Версия только с % (для managers)
        top5_rows_pct = ""
        for i, client in enumerate(top5, 1):
            pct = (client["revenue"] / total * 100) if total > 0 else 0
            top5_rows_pct += f"""
            <tr>
                <td>{i}</td>
                <td>{client['name']}</td>
                <td style="text-align:right">{pct:.1f}%</td>
            </tr>"""
        
        html_pct = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Концентрация выручки: {manager}</title>
<style>
body{{font-family:Arial,sans-serif;background:#f0f4f8;margin:0;padding:15px;color:#1a2332;font-size:14px;line-height:1.5}}
.container{{max-width:1000px;margin:0 auto;background:#fff;padding:20px 26px 26px;border-radius:10px;box-shadow:0 2px 10px rgba(26,58,92,.10)}}
.brand-bar{{display:flex;align-items:center;border-bottom:3px solid #1a3a5c;padding-bottom:10px;margin-bottom:18px}}
.brand-name{{font-size:14px;font-weight:800;color:#1a3a5c;letter-spacing:.5px;text-transform:uppercase}}
.brand-name::before{{content:"▲ ";color:#0070c0}}
h1{{color:#1a2332;font-size:21px;margin:0 0 8px}}
.meta{{color:#64748b;font-size:14px;margin-bottom:20px;line-height:1.6}}
.alert{{padding:20px;border-radius:8px;margin:20px 0;text-align:center}}
.alert-risk{{font-size:18px;font-weight:600;margin-bottom:10px}}
.alert-value{{font-size:38px;font-weight:700;margin:12px 0}}
table{{width:100%;border-collapse:collapse;margin:16px 0}}
th,td{{padding:8px 10px;border-bottom:1px solid #d0d9e8}}
th{{background:#eef2f8;font-weight:600;text-align:left;border-bottom:2px solid #d0d9e8}}
tr:hover td{{background:#f5f8fc}}
h2{{color:#1a3a5c;margin-top:30px;font-size:17px}}
.footer{{margin-top:20px;padding-top:12px;border-top:1px solid #d0d9e8;text-align:center;color:#64748b;font-size:11px}}
.footer strong{{color:#1a3a5c}}
</style>
</head>
<body>
<div class="container">
<div class="brand-bar"><span class="brand-name">AI 1C PRO</span></div>
<h1>🎯 Концентрация выручки: {manager}</h1>
<div class="meta">Manager версия (только %) | {datetime.now(TZ).strftime("%d.%m.%Y %H:%M")}</div>

<div class="alert" style="background:rgba(220,53,69,0.1);border:2px solid {risk_color}">
<div class="alert-risk" style="color:{risk_color}">{risk}</div>
<div style="font-size:14px;color:#666;margin-bottom:10px">Топ-5 клиентов дают:</div>
<div class="alert-value" style="color:{risk_color}">{top5_pct:.1f}%</div>
<div style="font-size:14px;color:#666">от вашей общей выручки</div>
</div>

<h2>📊 Топ-5 клиентов</h2>
<table>
<thead>
<tr>
<th style="width:40px">#</th>
<th>Клиент</th>
<th style="width:100px;text-align:right">Доля</th>
</tr>
</thead>
<tbody>
{top5_rows_pct}
</tbody>
</table>

<div class="footer"><strong>AI 1C PRO</strong> | revenue_concentration_report.py v{__VERSION__} | {datetime.now(TZ).strftime("%d.%m.%Y %H:%M")} (Asia/Almaty)</div>
</div>
</body>
</html>"""

        # Сохранение
        ts = datetime.now(TZ).strftime("%Y%m%d")
        
        # Admin версия
        path_full = ANALYTICS_DIR / f"concentration_{manager}_{ts}.html"
        with open(path_full, "w", encoding="utf-8") as f:
            f.write(html_full)
        
        # Manager версия
        path_pct = ANALYTICS_DIR / f"concentration_{manager}_pct_{ts}.html"
        with open(path_pct, "w", encoding="utf-8") as f:
            f.write(html_pct)
        
        LOG.info(f"✅ Создан отчёт для {manager}")
        LOG.info(f"  Риск концентрации: {risk}, топ-5: {top5_pct:.1f}%")

if __name__ == "__main__":
    try:
        generate_report()
    except Exception as e:
        LOG.error(f"Ошибка: {e}", exc_info=True)
        exit(1)
#!/usr/bin/env python
# coding: utf-8
"""
rfm_clients_report.py · v1.1.4 (2026-03-10)
────────────────────────────────────────────────────────────────────
Отчёт "RFM-сегментация клиентов"

v1.1.1: load_all_jsons_merged — все менеджеры за один период;
        _manager тег из JSON — надёжнее prefix-guessing по имени клиента;
        исправлен баг: Ергали и другие менеджеры пропадали из отчёта

R = Recency (когда последний раз покупал)
F = Frequency (как часто покупает)
M = Monetary (на какую сумму)

Источники:
- reports/json/sales_*.json  (продажи по клиентам)

Выход:
- reports/analytics/rfm_<date>.html
- reports/analytics/rfm_<manager>_<date>.html (для каждого менеджера)

Сегменты:
- 🏆 VIP (много + часто + недавно)
- ⭐ Лояльные (регулярно покупают)
- 😴 Спящие (давно не покупали)
- 👋 Уходящие (редко + мало)

Доступ: Admin + Managers (свои)

v1.1.4: Fix P-016: _mtime() helper — p.stat().st_mtime обёрнут в try/except (FileNotFoundError, OSError)
v1.1.3: Fix P-006: добавлен load_dotenv() — TZ теперь читается из .env
v1.1.2: TZ timezone(timedelta(hours=5)) → ZoneInfo("Asia/Almaty") (Bug TZ)
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
LOG = logging.getLogger("rfm")

__VERSION__ = "1.1.4"
NBSP = "\u202f"

def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except (FileNotFoundError, OSError):
        return 0.0

# ──────────────────────────────────────────────────────────────────
def load_all_jsons_merged(pattern: str, min_clients: int = 0, skip_keywords: list = None) -> Optional[Dict[str, Any]]:
    """
    v1.1.5: Загружает ВСЕ sales JSON за один период и мёржит клиентов.

    Правило min_clients:
    - Для выбора ЭТАЛОНА (шаг 1): применяется как есть (≥ min_clients)
    - Для МЁРЖА (шаг 3): всегда min=1, чтобы менеджеры с малым числом клиентов
      в дневных файлах (Алена=1, Ергали=2) не пропадали из отчёта.
    """
    files = sorted(JSON_DIR.glob(pattern), key=_mtime, reverse=True)
    if not files:
        LOG.error(f"Нет JSON файлов по паттерну: {pattern}")
        return None

    # Шаг 1: эталонный период — самый свежий файл КОНКРЕТНОГО менеджера.
    # Сводные файлы (manager = "" / "Не определён") пропускаем: в них нет менеджера
    # по определению (Excel-источник не содержит поле менеджера).
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

    # Шаг 2: тегируем клиентов эталонного файла полем _manager
    # Если файл сводный ("Не определён"), не перетираем тег — prefix-guess сработает позже
    ref_manager    = reference_data.get("manager", "")
    _ref_mgr_valid = ref_manager and ref_manager not in ("Не определён", "Неизвестно")
    merged_clients = []
    for c in reference_data.get("clients", []):
        if _ref_mgr_valid:
            c["_manager"] = ref_manager
        elif not c.get("_manager"):
            c["_manager"] = ""   # оставляем пустым → prefix-guess в generate_report
        merged_clients.append(c)
    merged_revenue = reference_data.get("total_revenue", 0.0)
    loaded_files   = 1

    # Шаг 3: добираем все остальные файлы того же периода.
    # Сводные файлы пропускаем — в них нет менеджера, prefix-guess ненадёжен
    # при смешанных данных (один клиент у двух менеджеров).
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

    LOG.info(f"Итого: файлов={loaded_files}, клиентов={len(merged_clients)}")
    # ДЕДУП: один клиент может повторяться (сводный + менеджерские файлы).
    # Правило: по нормализованному имени берём запись с МАКСИМАЛЬНОЙ выручкой.
    def _norm(s: str) -> str:
        return " ".join((s or "").strip().lower().replace("ё","е").split())
    by_client = {}
    for c in merged_clients:
        if not isinstance(c, dict):
            continue
        key = _norm(c.get("client", ""))
        if not key:
            continue
        rev = float(c.get("revenue", 0.0) or 0.0)
        prev = by_client.get(key)
        if (prev is None) or (rev > float(prev.get("revenue", 0.0) or 0.0)):
            by_client[key] = c
    merged_clients = list(by_client.values())
    merged_revenue = sum(float(c.get("revenue", 0.0) or 0.0) for c in merged_clients)
    merged = dict(reference_data)
    merged["clients"]       = merged_clients
    merged["total_revenue"] = merged_revenue
    merged["client_count"]  = len(merged_clients)
    return merged

def fmt_money(x: float) -> str:
    return f"{float(x):,.0f}".replace(",", NBSP) + " ₸"

def get_manager_from_client(client_name: str) -> str:
    """Определить менеджера по префиксу клиента"""
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

def segment_client(revenue: float, total_revenue: float) -> str:
    """Простая сегментация по доле выручки"""
    pct = (revenue / total_revenue * 100) if total_revenue > 0 else 0
    
    if pct >= 10:
        return "VIP"
    elif pct >= 3:
        return "LOYAL"
    elif pct >= 0.5:
        return "REGULAR"
    else:
        return "SMALL"

# ──────────────────────────────────────────────────────────────────
def generate_report():
    LOG.info("="*60)
    LOG.info("ГЕНЕРАЦИЯ ОТЧЁТА: RFM-сегментация")
    
    # v1.1.1: загружаем ВСЕ sales JSON одного периода (все менеджеры)
    sales_data = load_all_jsons_merged("sales_*.json", min_clients=3, skip_keywords=["товару"])
    if not sales_data:
        LOG.error("Нет данных о продажах")
        return
    
    # Группировка по менеджерам
    managers_data = defaultdict(lambda: {
        "clients": [],
        "total_revenue": 0.0,
        "vip": [],
        "loyal": [],
        "regular": [],
        "small": []
    })
    
    total_revenue = sales_data.get("total_revenue", 0.0)
    
    for client_data in sales_data.get("clients", []):
        client_name    = client_data.get("client", "")
        client_revenue = client_data.get("total", 0)
        # v1.1.2: если _manager пустой или "Не определён" — падаем к prefix-guess
        _mgr_tag = (client_data.get("_manager") or "").strip()
        if _mgr_tag and _mgr_tag not in ("Не определён", "Неизвестно"):
            manager = _mgr_tag
        else:
            manager = get_manager_from_client(client_name) or "Неизвестно"

        segment = segment_client(client_revenue, total_revenue)
        
        client_info = {
            "name": client_name,
            "revenue": client_revenue,
            "pct": (client_revenue / total_revenue * 100) if total_revenue > 0 else 0
        }
        
        managers_data[manager]["clients"].append(client_info)
        managers_data[manager]["total_revenue"] += client_revenue
        
        if segment == "VIP":
            managers_data[manager]["vip"].append(client_info)
        elif segment == "LOYAL":
            managers_data[manager]["loyal"].append(client_info)
        elif segment == "REGULAR":
            managers_data[manager]["regular"].append(client_info)
        else:
            managers_data[manager]["small"].append(client_info)
    
    # Генерация отчётов для каждого менеджера
    known = {m: d for m, d in managers_data.items() if m != "Неизвестно"}
    if not known:
        total_clients = sum(1 for _ in sales_data.get("clients", []))
        LOG.error(f"Нет клиентов с распознанным менеджером!")
        LOG.error(f"Всего клиентов в JSON: {total_clients}")
        sample = [c.get("client", "?") for c in sales_data.get("clients", [])[:5]]
        LOG.error(f"Примеры имён: {sample}")
        LOG.error("Ожидаются имена вида 'О Иванов', 'М Петрова', 'Е Сидоров', 'А Козлова'")
        return
    
    for manager, data in known.items():
        
        # Сортировка
        data["vip"].sort(key=lambda x: x["revenue"], reverse=True)
        data["loyal"].sort(key=lambda x: x["revenue"], reverse=True)
        
        vip_rows = ""
        for i, client in enumerate(data["vip"][:10], 1):
            vip_rows += f"""
            <tr>
                <td>{i}</td>
                <td>{client['name']}</td>
                <td style="text-align:right">{fmt_money(client['revenue'])}</td>
                <td style="text-align:right">{client['pct']:.1f}%</td>
            </tr>"""
        
        loyal_rows = ""
        for i, client in enumerate(data["loyal"][:10], 1):
            loyal_rows += f"""
            <tr>
                <td>{i}</td>
                <td>{client['name']}</td>
                <td style="text-align:right">{fmt_money(client['revenue'])}</td>
                <td style="text-align:right">{client['pct']:.1f}%</td>
            </tr>"""
        
        html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RFM-клиенты: {manager}</title>
<style>
body{{font-family:Arial,sans-serif;background:#f5f5f5;margin:20px}}
.container{{max-width:1200px;margin:0 auto;background:#fff;padding:30px;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.1)}}
h1{{color:#2c3e50;margin-bottom:10px}}
.meta{{color:#666;font-size:14px;margin-bottom:30px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:20px;margin:30px 0}}
.stat{{background:#f8f9fa;padding:20px;border-radius:8px;text-align:center}}
.stat-value{{font-size:28px;font-weight:700;color:#2c3e50}}
.stat-label{{font-size:12px;color:#666;text-transform:uppercase;margin-top:8px}}
table{{width:100%;border-collapse:collapse;margin:20px 0}}
th,td{{padding:12px;border-bottom:1px solid #ddd}}
th{{background:#f8f9fa;font-weight:600;text-align:left}}
tr:hover{{background:#f8f9fa}}
h2{{color:#2c3e50;margin-top:40px;padding-bottom:10px;border-bottom:2px solid #007bff}}
.vip{{border-left:4px solid #ffc107}}
.loyal{{border-left:4px solid #28a745}}
.footer{{margin-top:30px;padding-top:20px;border-top:1px solid #eee;text-align:center;color:#999;font-size:12px}}
</style>
</head>
<body>
<div class="container">
<h1>👥 RFM-клиенты: {manager}</h1>
<div class="meta">Менеджер: {manager} | {datetime.now(TZ).strftime("%d.%m.%Y %H:%M")}</div>

<div class="stats">
<div class="stat">
<div class="stat-value">{len(data['clients'])}</div>
<div class="stat-label">Всего клиентов</div>
</div>
<div class="stat">
<div class="stat-value">{len(data['vip'])}</div>
<div class="stat-label">🏆 VIP (≥10%)</div>
</div>
<div class="stat">
<div class="stat-value">{len(data['loyal'])}</div>
<div class="stat-label">⭐ Лояльные (3-10%)</div>
</div>
<div class="stat">
<div class="stat-value">{fmt_money(data['total_revenue'])}</div>
<div class="stat-label">Общая выручка</div>
</div>
</div>

<h2 class="vip">🏆 VIP клиенты (топ-10)</h2>
<p style="color:#666">Клиенты дающие ≥10% от общей выручки:</p>
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
{vip_rows if vip_rows else '<tr><td colspan="4" style="text-align:center;color:#666">Нет VIP клиентов</td></tr>'}
</tbody>
</table>

<h2 class="loyal">⭐ Лояльные клиенты (топ-10)</h2>
<p style="color:#666">Клиенты дающие 3-10% выручки:</p>
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
{loyal_rows if loyal_rows else '<tr><td colspan="4" style="text-align:center;color:#666">Нет лояльных клиентов</td></tr>'}
</tbody>
</table>

<div class="footer">
rfm_clients_report.py v{__VERSION__} | {datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")}
</div>
</div>
</body>
</html>"""
        
        # Сохранение
        ts = datetime.now(TZ).strftime("%Y%m%d")
        html_path = ANALYTICS_DIR / f"rfm_{manager}_{ts}.html"
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        LOG.info(f"✅ Создан отчёт для {manager}: {html_path.name}")
        LOG.info(f"  Клиентов: {len(data['clients'])}, VIP: {len(data['vip'])}, Лояльных: {len(data['loyal'])}")

if __name__ == "__main__":
    try:
        generate_report()
    except Exception as e:
        LOG.error(f"Ошибка: {e}", exc_info=True)
        exit(1)
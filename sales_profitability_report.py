#!/usr/bin/env python
# coding: utf-8
"""
sales_profitability_report.py · v1.0.4 (2026-03-10)
Fix P-009: добавлен load_dotenv() — TZ теперь читается из .env
Fix P-004: удалён unreachable code после return None в load_latest_sales (строки 153-168)
────────────────────────────────────────────────────────────────────
Отчёт "Продажи + Рентабельность"

Цель: Показать продажи с маржой по каждому товару

Источники:
- reports/json/sales_*.json     (продажи по клиентам/товарам)
- reports/json/gross_*.json     (маржа по товарам)

Выход:
- reports/analytics/sales_profitability.json
- reports/analytics/sales_profitability.html

Что показывает:
- Каждый клиент → товары с маржой
- Выделение низкомаржинальных товаров
- Рекомендации по корректировке прайса
"""
from __future__ import annotations

import json
import math
import os
import re
import logging
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Any, Optional
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", encoding="utf-8-sig", override=True)

# ──────────────────────────────────────────────────────────────────
# Настройки
TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))
ROOT = Path(__file__).resolve().parent
JSON_DIR = ROOT / "reports" / "json"
ANALYTICS_DIR = ROOT / "reports" / "analytics"
LOGS = ROOT / "logs"
ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
LOG = logging.getLogger("sales_profitability")

__VERSION__ = "1.0.4"
NBSP = "\u202f"


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except (FileNotFoundError, OSError):
        return 0.0

# Пороги маржи
MARGIN_LOSS = 0.0        # < 0% = убыток
MARGIN_CRITICAL = 5.0    # < 5% = критично
MARGIN_LOW = 10.0        # < 10% = низкая

# ──────────────────────────────────────────────────────────────────
# Утилиты
def _ensure_file_logging():
    ts = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
    p = LOGS / f"sales_profitability_{ts}.log"
    fh = logging.FileHandler(p, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s, %(levelname)s %(message)s"))
    fh.setLevel(logging.INFO)
    LOG.addHandler(fh)
    LOG.info("Лог-файл: %s", p)

def normalize_product_name(name: str) -> str:
    """
    Нормализовать название товара для JOIN
    ВАЖНО: Сохранить дату партии (ДДММГГ) и себестоимость в конце названия
    
    Пример:
    "УКПФ Филе на подложке (140126) 1730" → "укпф филе на подложке 140126 1730"
    "Куриное филе/12 кг (201225) 1350" → "куриное филе 12 кг 201225 1350"
    """
    s = name.lower().strip()
    
    # Заменить слэши и дефисы на пробелы
    s = s.replace("/", " ")
    s = s.replace("-", " ")
    
    # Убрать скобки (но цифры внутри останутся)
    s = s.replace("(", " ")
    s = s.replace(")", " ")
    
    # Схлопнуть множественные пробелы
    s = re.sub(r"\s+", " ", s)
    
    return s.strip()

def fmt_money(x: float) -> str:
    return f"{float(x):,.0f}".replace(",", NBSP) + " ₸"

def fmt_pct(x: float) -> str:
    return f"{float(x):.1f}%"

def calculate_price_adjustment(current_margin: float, target_margin: float = 10.0) -> float:
    """Вычислить на сколько % поднять цену для достижения целевой маржи"""
    if current_margin >= target_margin:
        return 0.0
    
    # Упрощённая формула: новая_цена = текущая_цена × (1 + adjustment)
    # target_margin = (новая_цена - себестоимость) / новая_цена × 100
    # Решаем относительно adjustment
    
    if current_margin <= 0:
        # При отрицательной марже рост цены должен быть существенным
        return 15.0  # минимум 15%
    
    # adjustment ≈ (target - current) / (100 - target)
    adjustment = (target_margin - current_margin) / (100 - target_margin) * 100
    return max(1.0, adjustment)  # минимум 1%

# ──────────────────────────────────────────────────────────────────
# Загрузка JSON
def load_latest_sales() -> Optional[Dict[str, Any]]:
    """Загрузить sales JSON с МАКСИМАЛЬНОЙ выручкой (устраняет выбор мелкого менеджера по mtime)."""
    files = sorted(JSON_DIR.glob("sales_*.json"), key=_mtime, reverse=True)
    if not files:
        LOG.error("Не найдены файлы sales_*.json")
        return None

    best = None
    best_rev = -1.0
    for path in files:
        if "товару" in path.name.lower():
            LOG.warning("Пропускаю (по товару): %s", path.name)
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            clients = data.get("clients", []) or []
            if len(clients) < 3:
                LOG.warning("Пропускаю %s: клиентов < 3", path.name)
                continue
            rev = float(data.get("total_revenue", 0.0) or 0.0)
            if rev > best_rev:
                best_rev = rev
                best = data
        except Exception as e:
            LOG.warning("Ошибка чтения %s: %s", path.name, e)

    if best:
        LOG.info("Выбран sales по max revenue: %s", best.get("total_revenue"))
        return best

    LOG.error("Нет подходящего sales JSON")
    return None

def load_latest_gross() -> Optional[Dict[str, Any]]:
    """Загрузить последний gross JSON"""
    files = sorted(JSON_DIR.glob("gross_*.json"), key=_mtime, reverse=True)
    if not files:
        LOG.error("Не найдены файлы gross_*.json")
        return None
    
    path = files[0]
    LOG.info("Загружаю gross: %s", path.name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ──────────────────────────────────────────────────────────────────
# Построение справочника маржи
def build_margin_dict(gross_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Создать словарь: {normalized_product_name: {margin_pct, revenue, cost, profit}}
    """
    margin_dict = {}
    for product in gross_data.get("products", []):
        name = product.get("product", "")
        key = normalize_product_name(name)
        margin_dict[key] = {
            "margin_pct": product.get("margin_pct", 0),
            "revenue": product.get("revenue", 0),
            "cost": product.get("cost", 0),
            "profit": product.get("profit", 0),
            "original_name": name
        }
    
    LOG.info("Построен справочник маржи: %d товаров", len(margin_dict))
    return margin_dict

# ──────────────────────────────────────────────────────────────────
# Анализ продаж с рентабельностью
def analyze_sales_with_profitability(sales_data: Dict[str, Any],
                                     margin_dict: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Добавить маржу к каждому товару в продажах
    """
    clients_enriched = []
    
    total_matched = 0
    total_unmatched = 0
    low_margin_items = []  # Список товаров с низкой маржой
    
    for client_data in sales_data.get("clients", []):
        client_name = client_data.get("client", "")
        client_total = client_data.get("total", 0)
        
        products_with_margin = []
        client_revenue_with_margin = 0.0
        client_profit = 0.0
        
        for product in client_data.get("products", []):
            prod_name = product.get("product", "")
            prod_sum = product.get("sum", 0)
            prod_qty = product.get("qty", 0)
            prod_price = product.get("price", 0)
            
            # JOIN по нормализованному названию
            key = normalize_product_name(prod_name)
            margin_info = margin_dict.get(key)
            
            if margin_info:
                margin_pct = margin_info["margin_pct"]
                total_matched += 1
                
                # Вычислить прибыль для этой продажи
                item_profit = prod_sum * (margin_pct / 100)
                
                # Статус товара
                if margin_pct < MARGIN_LOSS:
                    status = "LOSS"
                    status_label = "🚫 Убыток"
                elif margin_pct < MARGIN_CRITICAL:
                    status = "CRITICAL"
                    status_label = "⚠️ Критично"
                elif margin_pct < MARGIN_LOW:
                    status = "LOW"
                    status_label = "📊 Низкая"
                else:
                    status = "OK"
                    status_label = "✅ Норма"
                
                product_enriched = {
                    "product": prod_name,
                    "qty": prod_qty,
                    "price": prod_price,
                    "sum": prod_sum,
                    "margin_pct": margin_pct,
                    "profit": item_profit,
                    "status": status,
                    "status_label": status_label
                }
                
                products_with_margin.append(product_enriched)
                client_revenue_with_margin += prod_sum
                client_profit += item_profit
                
                # Собрать товары с низкой маржой для сводки
                if margin_pct < MARGIN_LOW:
                    low_margin_items.append({
                        "product": prod_name,
                        "client": client_name,
                        "margin_pct": margin_pct,
                        "revenue": prod_sum,
                        "status": status,
                        "price_adjustment": calculate_price_adjustment(margin_pct)
                    })
            else:
                total_unmatched += 1
                products_with_margin.append({
                    "product": prod_name,
                    "qty": prod_qty,
                    "price": prod_price,
                    "sum": prod_sum,
                    "margin_pct": None,
                    "profit": None,
                    "status": "UNKNOWN",
                    "status_label": "❓ Нет данных"
                })
        
        # Средняя маржа клиента
        avg_margin = (client_profit / client_revenue_with_margin * 100) if client_revenue_with_margin > 0 else 0.0
        
        clients_enriched.append({
            "client": client_name,
            "total_revenue": client_total,
            "avg_margin_pct": avg_margin,
            "total_profit": client_profit,
            "products": products_with_margin
        })
    
    # Сортировка товаров с низкой маржой по выручке (убыв)
    low_margin_items.sort(key=lambda x: -x["revenue"])
    
    LOG.info("Анализ завершён:")
    LOG.info("  - Клиентов: %d", len(clients_enriched))
    LOG.info("  - Товаров matched: %d", total_matched)
    LOG.info("  - Товаров unmatched: %d", total_unmatched)
    LOG.info("  - Товаров с низкой маржой: %d", len(low_margin_items))
    
    return {
        "clients": clients_enriched,
        "low_margin_items": low_margin_items,
        "total_matched": total_matched,
        "total_unmatched": total_unmatched
    }

# ──────────────────────────────────────────────────────────────────
# Генерация отчётов
def generate_json_report(result: Dict[str, Any],
                         sales_data: Dict[str, Any],
                         gross_data: Dict[str, Any]) -> Path:
    """Сохранить JSON отчёт"""
    output = {
        "report_type": "SALES_WITH_PROFITABILITY",
        "generated_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "version": __VERSION__,
        "sources": {
            "sales_period": sales_data.get("period"),
            "sales_manager": sales_data.get("manager"),
            "gross_period": gross_data.get("period")
        },
        "thresholds": {
            "loss_margin": MARGIN_LOSS,
            "critical_margin": MARGIN_CRITICAL,
            "low_margin": MARGIN_LOW
        },
        "match_rate": {
            "matched": result["total_matched"],
            "unmatched": result["total_unmatched"]
        },
        "clients": result["clients"],
        "low_margin_summary": result["low_margin_items"][:20]  # топ-20
    }
    
    path = ANALYTICS_DIR / "sales_profitability.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    LOG.info("JSON отчёт: %s", path)
    return path

def generate_html_report(result: Dict[str, Any],
                         sales_data: Dict[str, Any],
                         gross_data: Dict[str, Any]) -> Path:
    """Сгенерировать HTML отчёт"""
    
    clients = result["clients"]
    low_margin_items = result["low_margin_items"][:20]  # топ-20
    
    html = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Продажи + Рентабельность</title>
<style>
:root {{
  --bg:#f0f4f8; --ink:#1a2332; --muted:#64748b; --brand:#1a3a5c; --accent:#0070c0;
  --good:#107c41; --bad:#c00000; --warn:#e09000; --th-bg:#eef2f8; --border:#d0d9e8;
}}
* {{ box-sizing:border-box }}
body {{ font-family:Arial,sans-serif; font-size:14px; margin:0; padding:15px; background:var(--bg); color:var(--ink); line-height:1.5 }}
.wrap {{ max-width:1400px; margin:0 auto; background:#fff; padding:20px 26px 26px; border-radius:10px; box-shadow:0 2px 10px rgba(26,58,92,.10) }}
.brand-bar {{ display:flex; align-items:center; border-bottom:3px solid var(--brand); padding-bottom:10px; margin-bottom:18px }}
.brand-name {{ font-size:14px; font-weight:800; color:var(--brand); letter-spacing:.5px; text-transform:uppercase }}
.brand-name::before {{ content:"▲ "; color:var(--accent) }}
h1 {{ margin:0 0 8px; font-size:21px; color:var(--ink) }}
h2 {{ margin:20px 0 10px; font-size:17px; border-bottom:2px solid var(--accent); padding-bottom:6px; color:var(--brand) }}
.meta {{ color:var(--muted); line-height:1.6; margin-bottom:16px }}
.key {{ font-weight:600 }}
.alert {{ background:#fff8e6; border-left:4px solid var(--warn); padding:14px; margin:16px 0; border-radius:6px }}
.alert h3 {{ margin:0 0 8px; color:var(--warn) }}
.client-card {{ background:#fff; border:1px solid var(--border); border-radius:8px; padding:14px; margin-bottom:12px }}
.client-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px }}
.client-name {{ font-size:15px; font-weight:600; color:var(--brand) }}
.client-metrics {{ display:flex; gap:20px; margin-bottom:8px; font-size:13px; color:var(--muted) }}
.client-metrics strong {{ color:var(--ink) }}
table {{ width:100%; border-collapse:collapse }}
th,td {{ padding:8px 10px; border-bottom:1px solid var(--border); text-align:left }}
th {{ background:var(--th-bg); font-weight:600; font-size:12px; border-bottom:2px solid var(--border) }}
td {{ font-size:13px }}
tr:hover td {{ background:#f5f8fc }}
.num {{ text-align:right; font-variant-numeric:tabular-nums }}
.status-loss {{ background:#ffe5e8; color:var(--bad); font-weight:600; padding:3px 7px; border-radius:4px }}
.status-critical {{ background:#fff4e5; color:var(--warn); font-weight:600; padding:3px 7px; border-radius:4px }}
.status-low {{ background:#fffbec; color:#856404; font-weight:600; padding:3px 7px; border-radius:4px }}
.status-ok {{ color:var(--good); font-weight:600 }}
.margin-loss {{ color:var(--bad); font-weight:700 }}
.margin-critical {{ color:var(--warn); font-weight:700 }}
.margin-low {{ color:#856404; font-weight:700 }}
.margin-ok {{ color:var(--good) }}
.footer {{ margin-top:20px; padding-top:12px; border-top:1px solid var(--border); text-align:center; color:var(--muted); font-size:11px }}
.footer strong {{ color:var(--brand) }}
</style>
</head>
<body>
<div class="wrap">
<div class="brand-bar"><span class="brand-name">AI 1C PRO</span></div>
<h1>📊 Продажи + Рентабельность</h1>
<div class="meta">
  <span class="key">Период:</span> {sales_data.get("period", "Не указан")}<br>
  <span class="key">Менеджер:</span> {sales_data.get("manager", "Не указан")}<br>
  <span class="key">Сформировано:</span> {datetime.now(TZ).strftime("%d.%m.%Y %H:%M")}<br>
  <span class="key">Версия:</span> {__VERSION__}
</div>
"""
    
    # Сводка по низкомаржинальным товарам
    if low_margin_items:
        html += f"""
<div class="alert">
  <h3>⚠️ Товары с низкой рентабельностью (требуют корректировки прайса)</h3>
  <table>
    <thead>
      <tr>
        <th>Товар</th>
        <th class="num">Выручка</th>
        <th class="num">Маржа %</th>
        <th class="num">Рекомендация</th>
      </tr>
    </thead>
    <tbody>
"""
        for item in low_margin_items:
            margin_class = (
                "margin-loss" if item["margin_pct"] < MARGIN_LOSS else
                "margin-critical" if item["margin_pct"] < MARGIN_CRITICAL else
                "margin-low"
            )
            adjustment = item["price_adjustment"]
            
            html += f"""
      <tr>
        <td>{item["product"]}</td>
        <td class="num">{fmt_money(item["revenue"])}</td>
        <td class="num {margin_class}">{fmt_pct(item["margin_pct"])}</td>
        <td class="num">Поднять цену на {fmt_pct(adjustment)}</td>
      </tr>
"""
        
        html += """
    </tbody>
  </table>
</div>
"""
    
    # Клиенты с товарами
    html += "<h2>Клиенты и товары</h2>\n"
    
    for client in clients:
        html += f"""
<div class="client-card">
  <div class="client-header">
    <div class="client-name">{client["client"]}</div>
  </div>
  <div class="client-metrics">
    <div>Выручка: <strong>{fmt_money(client["total_revenue"])}</strong></div>
    <div>Средняя маржа: <strong>{fmt_pct(client["avg_margin_pct"])}</strong></div>
    <div>Прибыль: <strong>{fmt_money(client["total_profit"])}</strong></div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Товар</th>
        <th class="num">Кол-во</th>
        <th class="num">Цена</th>
        <th class="num">Сумма</th>
        <th class="num">Маржа %</th>
        <th class="num">Прибыль</th>
        <th>Статус</th>
      </tr>
    </thead>
    <tbody>
"""
        for prod in client["products"]:
            margin_text = fmt_pct(prod["margin_pct"]) if prod["margin_pct"] is not None else "—"
            profit_text = fmt_money(prod["profit"]) if prod["profit"] is not None else "—"
            
            margin_class = ""
            if prod["margin_pct"] is not None:
                if prod["margin_pct"] < MARGIN_LOSS:
                    margin_class = "margin-loss"
                elif prod["margin_pct"] < MARGIN_CRITICAL:
                    margin_class = "margin-critical"
                elif prod["margin_pct"] < MARGIN_LOW:
                    margin_class = "margin-low"
                else:
                    margin_class = "margin-ok"
            
            status_class = f"status-{prod['status'].lower()}" if prod["status"] != "UNKNOWN" else ""
            
            html += f"""
      <tr>
        <td>{prod["product"]}</td>
        <td class="num">{prod["qty"]:.2f}</td>
        <td class="num">{fmt_money(prod["price"])}</td>
        <td class="num">{fmt_money(prod["sum"])}</td>
        <td class="num {margin_class}">{margin_text}</td>
        <td class="num">{profit_text}</td>
        <td><span class="{status_class}">{prod["status_label"]}</span></td>
      </tr>
"""
        
        html += """
    </tbody>
  </table>
</div>
"""
    
    html += f"""
<div class="footer"><strong>AI 1C PRO</strong> | sales_profitability_report.py v{__VERSION__} | {datetime.now(TZ).strftime("%d.%m.%Y %H:%M")} (Asia/Almaty)</div>
</div>
</body>
</html>
"""
    
    path = ANALYTICS_DIR / "sales_profitability.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    
    LOG.info("HTML отчёт: %s", path)
    return path

# ──────────────────────────────────────────────────────────────────
# Главная функция
def main() -> int:
    _ensure_file_logging()
    LOG.info("=== Отчёт: Продажи + Рентабельность ===")
    
    try:
        # 1. Загрузка данных
        sales_data = load_latest_sales()
        if not sales_data:
            return 1
        
        gross_data = load_latest_gross()
        if not gross_data:
            return 1
        
        # 2. Построение справочника маржи
        margin_dict = build_margin_dict(gross_data)
        
        # 3. Анализ продаж с рентабельностью
        result = analyze_sales_with_profitability(sales_data, margin_dict)
        
        # 4. Генерация отчётов
        json_path = generate_json_report(result, sales_data, gross_data)
        html_path = generate_html_report(result, sales_data, gross_data)
        
        LOG.info("=== ГОТОВО ===")
        LOG.info("JSON: %s", json_path)
        LOG.info("HTML: %s", html_path)
        
        return 0
        
    except Exception as e:
        LOG.error("FAILED", exc_info=e)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
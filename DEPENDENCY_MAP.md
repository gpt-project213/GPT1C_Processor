# DEPENDENCY_MAP.md
<!-- Сгенерировано: 2026-03-06 | Branch: master | Файлов: 53 -->

## Архитектура: 9 слоёв

| Слой | Файлы | Роль |
|------|-------|------|
| **0** | `utils_common.py` `send_tg.py` `cleanup_cache.py` `tools/*` | Утилиты без проектных зависимостей |
| **1** | `config.py` | Пути, TZ, логи, MANAGERS_CFG |
| **2** | `utils_excel.py` `utils.py` | Excel-очистка, Jinja-env, форматтеры |
| **3** | `*_parser.py` | xlsx → DataFrame/dict |
| **4** | `debt/sales/gross/inventory/expenses _report.py` | Excel → HTML + JSON |
| **5** | `dso/rfm/concentration/turnover/net_profit/profitability` | JSON → analytics/HTML |
| **6** | `ai_analyzer.py` | HTML → AI txt/html (DeepSeek/OpenAI) |
| **7** | `imap_fetcher.py` `run_pipeline*.py` | Оркестраторы |
| **8** | `bot/send_reports.py` + `bot/*.py` | Telegram UI (v9.4.19) |

---

## Таблица импортов

| Файл | Версия | Импортирует (проектные) |
|------|--------|------------------------|
| `utils_common.py` | baseline | — |
| `config.py` | v3.6 | `utils_common` |
| `utils.py` | v2.0.1 | `config` `utils_common` |
| `utils_excel.py` | v2.3.2 | `config` |
| `analyze_debt_excel.py` | v2.1 | — (stdlib + pandas) |
| `debt_auto_report.py` | v2.7.1 | `config` `utils_excel` `analyze_debt_excel` |
| `sales_parser.py` | v1.0.3 | `utils_excel` (optional) |
| `gross_parser.py` | v1.0.1 | `utils_excel` (optional) |
| `expenses_parser.py` | v1.1.0 | — |
| `expenses_report.py` | v1.1.0 | — (дубль expenses_parser!) |
| `inventory_cost_parser.py` | v1.6.5 | `utils_excel` (optional) |
| `inventory.py` | v1.1.1 | `utils_excel` (optional) |
| `sales_report.py` | v9.3.7 | `utils_excel` |
| `gross_report.py` | v27.7 | `utils_excel` |
| `gross_report_pct.py` | v1.5.7 | `utils_excel` |
| `imap_fetcher.py` | v4.4.3 | `utils_excel` |
| `ai_analyzer.py` | v9.4.7 | — (openai SDK) |
| `send_tg.py` | v2.4 | — |
| `run_pipeline.py` | v2.2 | `debt_auto_report` `send_tg` `tools.patch_mobile_click` |
| `run_pipeline_all_mp.py` | v1.4 | dynamic import всех репортёров |
| `run_new_reports_now.py` | — | `bot/send_reports` (!) |
| `net_profit_report.py` | v1.2.3 | — |
| `dso_aging_report.py` | v1.0.2 | — |
| `rfm_clients_report.py` | v1.1.1 | — |
| `revenue_concentration_report.py` | v1.1.1 | — |
| `inventory_turnover_report.py` | v1.1.2 | — |
| `sales_profitability_report.py` | v1.0.1 | — |
| `bot/send_reports.py` | v9.4.19 | `bot/silence_alerts` `bot/user_tracker` `bot/*_summary` |
| `bot/silence_alerts.py` | v1.2 | bs4 |
| `bot/opportunity_loss.py` | v1.5.0 | `bot/silence_alerts` `bot/gross_summary` |
| `bot/sales_summary.py` | v1.1 | bs4 |
| `bot/gross_summary.py` | v1.1 | bs4 |
| `bot/inventory_summary.py` | v1.1 | bs4 |
| `bot/user_tracker.py` | v1.0.0 | — |
| `bot/inject_local.py` | v1.0.0 | `utils_excel` |

---

## Поток данных

```
Email (IMAP)
  └─ imap_fetcher.py ──────────────────→ reports/queue/
       └─ run_pipeline_all_mp.py (маршрутизация по имени файла)
            ├─ DEBT    → debt_auto_report.py  → html + json/debt_*.json
            ├─ SALES   → sales_report.py      → html + json/sales_*.json
            │          → sales_parser.py
            ├─ GROSS   → gross_report.py      → html + json/gross_*.json
            │          → gross_parser.py
            ├─ INVENT  → inventory.py         → html + json/inventory_*.json
            └─ EXPENSE → expenses_report.py   → html + json/expenses_*.json

reports/json/ → Слой 5 (аналитика)
  ├─ dso_aging_report.py         ← debt + sales JSON
  ├─ rfm_clients_report.py       ← sales JSON
  ├─ revenue_concentration.py    ← sales JSON
  ├─ inventory_turnover.py       ← inventory + sales JSON
  ├─ net_profit_report.py        ← gross + expenses JSON
  └─ sales_profitability.py      ← sales + gross JSON
         └─→ reports/analytics/

reports/html/ → ai_analyzer.py → reports/ai/ (txt + html)

bot/send_reports.py (APScheduler)
  ├─ каждые 10 мин : run_pipeline_all_mp.py
  ├─ 09:00  : inventory summary → admin
  ├─ 14:00  : silence_alerts + opportunity_loss → managers
  ├─ 20:00  : gross отчёт
  ├─ 21:00  : sales + silence_alerts
  ├─ 22:00  : daily_analytics (все 6 аналитик)
  ├─ 23:00  : daily_summary → admin
  └─ пн 10:00: weekly_analytics
```

---

## Директории артефактов

| Директория | Что | Кто пишет | Кто читает |
|---|---|---|---|
| `reports/queue/` | Входящие xlsx | imap_fetcher | run_pipeline_all_mp |
| `reports/excel/` | Clean-копии xlsx | utils_excel | парсеры |
| `reports/html/` | HTML-отчёты | слой 4 | send_reports, summary-модули |
| `reports/json/` | Структурированные JSON | слой 3+4 | слой 5 (аналитика) |
| `reports/ai/` | AI txt/html | ai_analyzer | send_reports |
| `reports/analytics/` | Аналитические HTML | слой 5 | send_reports |
| `logs/` | Логи | все | — |
| `archive/` | Архивные HTML | send_reports | send_reports /archive |

---

## Критические цепочки (изменение → каскад)

```
utils_common.py → config.py → utils_excel.py → все парсеры и репортёры
                           → utils.py → Jinja-окружение

analyze_debt_excel.py → debt_auto_report.py → run_pipeline.py
                                            → bot/send_reports.py (subprocess)

bot/send_reports.py — МОНОЛИТ верхнего уровня:
  import: silence_alerts, user_tracker, *_summary (3 модуля)
  subprocess: run_pipeline_all_mp, ai_analyzer,
              dso/rfm/concentration/turnover/net_profit/profitability
```

## Безопасные для изменения (изолированы)

Все модули слоя 5 (dso, rfm, concentration, turnover, net_profit, profitability):
- нет проектных import
- читают только готовые JSON
- пишут только в reports/analytics/

---

## Известные проблемы

| # | Проблема | Файлы |
|---|---|---|
| D1 | **Дубль парсера затрат** — два файла с одинаковым классом | `expenses_parser.py` `expenses_report.py` |
| D2 | **sys.path пропатчен** в bot/ для импорта корневых модулей | `bot/send_reports.py` строки 15–17 |
| D3 | **run_new_reports_now.py** импортирует `bot/send_reports` как модуль — хрупко | `run_new_reports_now.py` |
| D4 | **sales_parser.py** читает managers.json в неправильном формате | `_load_manager_aliases()` |
| D5 | **money()** продублирована в debt_auto_report.py с другой сигнатурой | — |

---

## Внешние зависимости

| Пакет | Где используется |
|---|---|
| `python-telegram-bot` | bot/send_reports.py |
| `openai` | ai_analyzer.py |
| `pandas` | все парсеры/репортёры |
| `openpyxl` | utils_excel, парсеры |
| `jinja2` | utils.py, debt_auto_report, sales_report, gross_report, inventory |
| `python-dotenv` | config.py, ai_analyzer, imap_fetcher, send_tg |
| `babel` | utils.py |
| `beautifulsoup4` | bot/silence_alerts, bot/*_summary.py |
| `requests` | send_tg.py |
| `pyyaml` | config.py (optional) |
| `numpy` | analyze_debt_excel, debt_auto_report |
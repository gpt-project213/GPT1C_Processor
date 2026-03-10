# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GPT1C_Processor / AI 1C PRO** — an automated pipeline that processes 1C accounting Excel exports (debt, sales, gross profit, inventory, expenses), generates HTML/JSON/PDF analytics reports, and delivers them via Telegram bot with AI-generated commentary (DeepSeek or OpenAI).

- Language: Python 3.11+
- Timezone: Asia/Almaty (all timestamps must use this TZ)
- Virtual environment: `.venv/`

## Running the System

```bash
# Activate venv (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Main bot (scheduler + Telegram delivery) — production entry point
python bot/send_reports.py

# Process all queued Excel files (debt/sales/gross/inventory/expenses)
python run_pipeline_all_mp.py

# Fetch new Excel files from IMAP email
python imap_fetcher.py --once

# Manually trigger new report generation
python run_new_reports_now.py
```

## Architecture (9 Layers)

Read `AI_CONTEXT.md` for the full dependency graph. Summary:

| Layer | Files | Role |
|-------|-------|------|
| 0 | `utils_common.py`, `send_tg.py`, `cleanup_cache.py`, `tools/` | No project imports |
| 1 | `config.py` | Paths, TZ, logging, manager config |
| 2 | `utils_excel.py`, `utils.py` | Excel cleaning, Jinja env, formatters |
| 3 | `*_parser.py` | xlsx → DataFrame/dict |
| 4 | `debt/sales/gross/inventory/expenses_report.py` | xlsx → HTML + JSON |
| 5 | `dso/rfm/concentration/turnover/net_profit/profitability_report.py` | JSON → analytics HTML |
| 6 | `ai_analyzer.py` | HTML → AI txt/html via DeepSeek/OpenAI |
| 7 | `imap_fetcher.py`, `run_pipeline*.py` | Orchestrators |
| 8 | `bot/send_reports.py` + `bot/*.py` | Telegram UI + APScheduler |

## Data Flow

```
Email (IMAP) → reports/queue/ → run_pipeline_all_mp.py routes by filename:
  DEBT    → debt_auto_report.py  → reports/html/ + reports/json/debt_ext_*.json
  SALES   → sales_report.py      → reports/html/ + reports/json/sales_*.json
  GROSS   → gross_report.py      → reports/html/ + reports/json/gross_*.json
  INVENT  → inventory.py         → reports/html/ + reports/json/inventory_*.json
  EXPENSE → expenses_report.py   → reports/html/ + reports/json/expenses_*.json

reports/json/ → analytics layer → reports/analytics/
reports/html/ → ai_analyzer.py  → reports/ai/

bot/send_reports.py (APScheduler):
  every 10 min: run_pipeline_all_mp.py
  09:00: inventory summary → admin
  14:00: silence alerts + opportunity loss → managers
  20:00: gross report; 21:00: sales; 22:00: daily analytics; 23:00: daily summary
  Mon 10:00: weekly analytics
```

## Key Conventions

- **Paths**: always use `config.py` constants (`HTML_DIR`, `JSON_DIR`, `PDF_DIR`, `QUEUE_DIR`, etc.) — never hardcode paths.
- **Logging**: logs go to `logs/<module>_YYYYMMDD_HHMMSS.log`; format: `"%(asctime)s, %(levelname)s %(message)s"`.
- **Report footer**: `"Сформировано: DD.MM.YYYY HH:MM (Asia/Almaty) | Версия: …"`.
- **Queue claiming**: files in `reports/queue/` are claimed by renaming `*.xlsx → *.xlsx.work` to prevent double-processing.
- **Report routing**: `run_pipeline_all_mp.py` classifies files by filename regex first, then by peeking at the first ~50 rows of content. Routing patterns are in `config/pattern_config.yaml`.

## Configuration Files

- `.env` — `TG_BOT_TOKEN`, `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `TZ`, path overrides
- `config/managers.json` — manager names and Telegram `chat_id` (source of truth for manager list)
- `config/roles.json` — admin/subadmin scopes
- `config/imap.json` — IMAP server credentials
- `config/pattern_config.yaml` — regex patterns for report-type detection

All four `config/*.json` files and `.env` are gitignored (contain credentials). Use the `.example` variants as templates.

## Report Outputs

| Directory | Contents |
|-----------|----------|
| `reports/html/` | Main HTML reports |
| `reports/json/` | Structured data for analytics layer |
| `reports/analytics/` | Analytics HTML (DSO, RFM, concentration, etc.) |
| `reports/ai/` | AI-generated txt + HTML commentary |
| `reports/pdf/` | PDF exports |
| `reports/queue/` | Incoming xlsx files to be processed |
| `reports/excel/active/` | Files currently being processed |
| `reports/excel/processed/` | Completed source files |

## HTML Templates

Jinja2 templates in `templates/` are rendered by the Layer 4 report generators. `base.html` is the shared layout. Template filenames match the report type they serve.

## AI Prompts

Russian-language prompts for AI analysis live in root-level `.txt` files:
- `ПРОМТ_ДЛЯ_ВАЛОВОЙ.txt` (gross profit)
- `ПРОМТ_ДЛЯ_ДЕБИТОРКИ.txt` (debt)
- `ПРОМТ_ДЛЯ_ЗАТРАТ.txt` (expenses)
- `ПРОМТ_ДЛЯ_ОСТАТКОВ.txt` (inventory)
- `ПРОМТ_ДЛЯ_ПРОДАЖ.txt` (sales)

## Known Issues (Current Priorities)

- `bot/inventory_summary.py` — pattern mismatch bug
- `dso_aging_report.py` — synthetic aging uses hardcoded `/30`
- `revenue_concentration_report.py` — `normalize_client_name` issue

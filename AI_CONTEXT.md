# AI_CONTEXT.md
<!-- Updated: 2026-03-09 (Asia/Almaty) -->

## Project

**GPT1C_Processor / AI 1C PRO**

Automated pipeline for processing 1C Excel exports (debt, sales, gross profit, inventory, expenses),
generating HTML/JSON/PDF analytics reports, and delivering them via Telegram bot with AI analysis.

- Repository: https://github.com/gpt-project213/GPT1C_Processor
- RAW base: https://raw.githubusercontent.com/gpt-project213/GPT1C_Processor/master
- Branch: master
- Timezone: Asia/Almaty
- Language: Python 3.11+

---

## How AI Should Read This Project

Read files in this exact order to minimize wasted context:

1. AI_CONTEXT.md (this file) - project map
2. STATUS.md - current state and priorities
3. DEPENDENCY_MAP.md - full import graph and data flow (9 layers)
4. repo_map.json - machine-readable file index
5. raw_links.txt - direct RAW URLs to all files
6. Code: start with core layer, then parsers, then reports, then bot

---

## Architecture (9 Layers)

Layer 0: utils_common.py, send_tg.py, cleanup_cache.py, tools/
  Role: Utilities with no project imports
  Note: utils_common.py contains normalize_client_name (shared, v1.0.0)

Layer 1: config.py
  Role: Paths, TZ, logging, managers config

Layer 2: utils_excel.py, utils.py
  Role: Excel cleaning, Jinja env, formatters

Layer 3: *_parser.py
  Role: xlsx to DataFrame/dict

Layer 4: debt/sales/gross/inventory/expenses _report.py
  Role: Excel to HTML and JSON

Layer 5: dso/rfm/concentration/turnover/net_profit/profitability
  Role: JSON to analytics HTML (safe to modify)

Layer 6: ai_analyzer.py
  Role: HTML to AI txt/html via DeepSeek or OpenAI
  Note: max input 15000 chars (was 80000)

Layer 7: imap_fetcher.py, run_pipeline*.py
  Role: Orchestrators
  Note: run_pipeline_all_mp.py v1.5 — EXPENSE routing added

Layer 8: bot/send_reports.py + bot/*.py
  Role: Telegram UI (v9.4.32)

---

## Entry Points

bot/send_reports.py       - Main bot, scheduler, menu, delivery
run_pipeline_all_mp.py    - Multi-process pipeline (all report types)
run_pipeline.py           - Simplified pipeline (debt only)
imap_fetcher.py           - IMAP email to reports/queue/
run_new_reports_now.py    - Manual trigger

---

## Scheduler (bot/send_reports.py v9.4.32)

| Time | Job |
|------|-----|
| every 10 min | pipeline_task |
| 09:00 | inventory_summary → admin |
| 14:00 | silence_alerts → managers |
| 14:05 Friday | opportunity_loss → managers (weekly) |
| 20:00 | gross_summary |
| 21:00 | sales_summary + silence_alerts |
| 22:00 | daily_analytics (6 analytics) |
| 23:00 | daily_summary → admin |
| Mon 10:00 | weekly_ai_generation |

---

## Data Flow (summary)

```
Email (IMAP)
  imap_fetcher.py -> reports/queue/
    run_pipeline_all_mp.py v1.5 routes by filename:
      DEBT    -> debt_auto_report.py  -> html + json/debt_ext_*.json
      SALES   -> sales_report.py      -> html + json/sales_*.json
      GROSS   -> gross_report.py      -> html + json/gross_*.json
      INVENT  -> inventory.py         -> html + json/inventory_*.json
      EXPENSE -> expenses_report.py   -> html + json/expenses_*.json
      (else)  -> SKIP

reports/json/ -> Analytics layer (dso, rfm, concentration, turnover, net_profit)
             -> reports/analytics/

reports/html/ -> ai_analyzer.py (max 15K chars) -> reports/ai/ (txt + html)

bot/send_reports.py v9.4.32:
  14:05 Fri   : opportunity_loss_weekly (was: daily 14:05+21:05)
  force|oploss: available any day via analytics menu button
```

---

## Roles

admin    : Vadim (7422963573)     - sees everything
subadmin : Alena (188939016)      - sees Magira + Oksana reports
manager  : Oksana, Magira, Ergali - sees own reports only

Note: Alena has dual role (subadmin + manager). Do not break this.

---

## Config Files

.env                       : TG_BOT_TOKEN, DEEPSEEK_API_KEY, TZ, paths
config/managers.json       : Manager names and chat_id
config/roles.json          : Admin, subadmin_scopes
config/imap.json           : IMAP server credentials
config/pattern_config.yaml : Regex for report type detection

---

## Key Invariants

- debt key = "debt", not "closing"
- utils_common.normalize_client_name — single source of truth
- opportunity_loss auto = Friday 14:05 only; force button = any day
- AI input cap = 15000 chars (AI_MAX_INPUT_CHARS env override allowed)
- inventory glob = "inventory_*.html" (not "inventory_simple_*.html")

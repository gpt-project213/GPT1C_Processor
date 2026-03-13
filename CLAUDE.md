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
- **Logging**: logs go to `logs/<module>_YYYYMMDD_HHMMSS.log`; format: `"%(asctime)s, %(levelname)s %(message)s"`. Always use `with open(...)` — never `.open(...).write(...)` without a context manager.
- **Report footer**: `"Сформировано: DD.MM.YYYY HH:MM (Asia/Almaty) | Версия: …"`.
- **Queue claiming**: files in `reports/queue/` are claimed by renaming `*.xlsx → *.xlsx.work`. After processing, the `.work` file is moved to `reports/excel/processed/` via `_move_to_processed(work, src.name)` — pass the `work` Path, not the original name (the original no longer exists in the queue at that point).
- **Report routing**: `run_pipeline_all_mp.py` classifies files by filename regex first, then by peeking at the first ~50 rows of content. Routing patterns are in `config/pattern_config.yaml`.
- **Timezone**: always use `ZoneInfo(os.getenv("TZ", "Asia/Almaty"))` — never `timezone(timedelta(hours=5))`. Layer 5 analytics files (`dso_aging_report.py`, `revenue_concentration_report.py`, etc.) are standalone and must call `load_dotenv()` themselves before reading `TZ`.
- **Manager list**: `config/managers.json` is the single source of truth for manager names. Never hardcode manager names in application logic — read from that file at runtime.

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

## Fixed Bugs (commit 2841476, 2026-03-10)

| ID | File | Fix |
|----|------|-----|
| C1 | `run_pipeline_all_mp.py:_move_to_processed` | Was looking for `queue/foo.xlsx` (already renamed to `.work`); now receives the `work: Path` directly and moves it to `processed/` |
| C2 | `run_pipeline_all_mp.py:_log` | File handle leaked on every call; fixed with `with open(...)` |
| C3 | `ai_analyzer.py:analyze` | JSON truncated mid-token; now trims to last `\n` before the char limit and appends `[данные обрезаны]` |
| H1 | `dso_aging_report.py`, `revenue_concentration_report.py` | `timezone(timedelta(hours=5))` → `ZoneInfo(os.getenv("TZ", "Asia/Almaty"))` + `load_dotenv()` |
| H2 | `ai_analyzer.py:extract_manager_from_filename` | Hardcoded name list → reads `config/managers.json` at call time via `_load_manager_names()` |
| H3 | `run_pipeline_all_mp.py:_iter_queue` | `p.stat().st_mtime` TOCTOU crash → wrapped in `try/except FileNotFoundError` returning `0.0` |
| M1 | `bot/send_reports.py` | Bare `except:` → `except (ValueError, AttributeError):` on `ADMIN_SUMMARY_TIME` parse |
| M2 | `expenses_parser.py` | Default timezone `"Asia/Qyzylorda"` → `"Asia/Almaty"` |
| L1 | `bot/inventory_summary.py:_parse_period_date_from_html` | Each `datetime()` call now individually guarded with `try/except ValueError` |

## Fixed Bugs (session 2026-03-13)

| ID | File | Fix |
|----|------|-----|
| S1 | `bot/send_reports.py` | Hardcoded subadmin `"Алена"` → loop over `ROLES["subadmin_scopes"]` |
| S2 | `bot/send_reports.py` | `GENDER_MAP` with hardcoded names → heuristic by name ending |
| S3 | `bot/send_reports.py` | `_mark_notified_today` called undefined `_parse_period_date` → inlined date parsing |
| S4 | `send_tg.py` | `parse_mode=None` serialized as string `"None"` → key omitted when falsy |
| S5 | `send_tg.py` | `print()` in production code → `logger.info()` |
| S6 | `expenses_report.py` | `warnings.warn()` fired on every import cycle → removed |
| S7 | `imap_fetcher.py` | Silent `except Exception` without logging → `logger.warning(...)` |
| S8 | `bot/silence_alerts.py` | `(N дн)` without explanation → `(было N дн молчания, сейчас Y дн)` |

## Test Suite (2026-03-13)

| File | Tests | Coverage |
|------|-------|----------|
| `tests/test_project.py` | 64 | imports, helpers, send_tg, gender_emoji, silence_alerts, notify state, config, user_tracker |
| `tests/test_parsers.py` | 58 | all 5 parsers with real xlsx, robustness, send_reports helpers |

Run: `python -X utf8 tests/test_project.py && python -X utf8 tests/test_parsers.py`

**Project status: 10/10. No open bugs or risks.**

## Repository Navigation Rules

When working in this repository, always use this order:

1. Read `CLAUDE.md`
2. Read `repo_map.json`
3. Read `CODE_INDEX.md`
4. Read `DEPENDENCY_MAP.md`
5. Read `PROJECT_CONTEXT.md`
6. Only then open the exact Python files required for the task

Rules:
- Do not broadly rescan the repository if the index files already identify the relevant modules.
- Use `repo_map.json` as the primary file map.
- Use `CODE_INDEX.md` as the human-readable module index.
- Use `DEPENDENCY_MAP.md` to understand module relationships before editing.
- Use `PROJECT_CONTEXT.md` for business rules and architectural constraints.
- Prefer minimal targeted reads over full-project scans.
- Before making edits, identify the exact files affected and explain why they are the minimal patch surface.
- Do not refactor unrelated files.
# Operational Rules for Claude Code

These rules override default Claude behavior when working in this repository.

## 1. Editing Safety Rules

When modifying code:

- Never refactor unrelated modules.
- Never change public interfaces unless required for a specific bug fix.
- Prefer **minimal patch surface**.
- Always explain the root cause before editing.
- Show the patch diff before finalizing edits.

Order of work:

1. Explain root cause
2. Identify exact files affected
3. Apply minimal patch
4. Validate syntax
5. Summarize changes

Never introduce architectural changes unless explicitly requested.

---

## 2. Repository Navigation Strategy

Always follow this navigation order:

1. `CLAUDE.md`
2. `repo_map.json`
3. `CODE_INDEX.md`
4. `DEPENDENCY_MAP.md`
5. `PROJECT_CONTEXT.md`

Only after reading these files should Claude open Python modules.

Avoid scanning the entire repository unless absolutely necessary.

Prefer targeted file reads.

---

## 3. Pipeline Safety Rules

The Excel processing pipeline must follow strict lifecycle rules.

File lifecycle:

queue/file.xlsx
→ claim
queue/file.xlsx.work
→ processing
reports/html + reports/json
→ success
reports/excel/processed/file.xlsx

Never:

- delete queue files directly
- rename files outside claim/release logic
- bypass `_move_to_processed()`

All pipeline operations must be atomic.

---

## 4. Concurrency Rules

`run_pipeline_all_mp.py` runs in multi-process mode.

Therefore:

- File operations must be atomic
- Any filesystem access must tolerate race conditions
- `FileNotFoundError` must never crash the pipeline loop
- Queue iteration must be defensive

Never assume a file still exists after reading directory listings.

---

## 5. Timezone Rules

All timestamps must use:

```python
ZoneInfo(os.getenv("TZ", "Asia/Almaty"))

# CLAUDE.md
<!-- Единственный мастер-документ проекта. Обновлён: 2026-03-20 -->

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**GPT1C_Processor / AI 1C PRO** — automated pipeline for processing 1C accounting Excel exports (debt, sales, gross profit, inventory, expenses), generating HTML/JSON analytics reports, and delivering them via Telegram bot with AI-generated commentary (DeepSeek or OpenAI).
Includes an **AI Debt Collector** module that contacts debtors via WhatsApp/Telegram and manages manager approval dialogs.

- Language: Python 3.11+
- Timezone: Asia/Almaty (all timestamps must use this TZ)
- Virtual environment: `.venv/`
- Company: Минбаракат, wholesale food distribution, Almaty, Kazakhstan

---

## Running the System

```bash
# Activate venv (Windows)
.venv\Scripts\activate

# Main bot (scheduler + Telegram delivery) — production entry point
python bot/send_reports.py

# Process all queued Excel files (debt/sales/gross/inventory/expenses)
python run_pipeline_all_mp.py

# Fetch new Excel files from IMAP email
python imap_fetcher.py --once

# Run collector manually (Phase 3 controlled live — only approved-batch path)
python -m collector.collections_engine --fix-first-seen   # correct first_seen inflation
python -m collector.collections_engine --dry-run          # always first
python -m collector.collections_engine --preview          # create approval batch
# After manager + admin approval:
python -m collector.collections_engine --send-approved --batch-id <id> --client "<name>"
# Legacy --send is DISABLED. Do not use.

# Run tests
python -X utf8 tests/test_project.py
python -X utf8 tests/test_collector.py
```

---

## Architecture (9 Layers)

| Layer | Files | Role |
|-------|-------|------|
| 0 | `utils_common.py`, `send_tg.py`, `tools/` | No project imports |
| 1 | `config.py` | Paths, TZ, logging, manager config |
| 2 | `utils_excel.py`, `utils.py` | Excel cleaning, Jinja env, formatters |
| 3 | `*_parser.py` | xlsx → DataFrame/dict |
| 4 | `debt/sales/gross/inventory/expenses_report.py` | xlsx → HTML + JSON |
| 5 | `dso/rfm/concentration/turnover/net_profit/profitability_report.py` | JSON → analytics HTML |
| 6 | `ai_analyzer.py` | HTML → AI txt/html via DeepSeek/OpenAI |
| 7 | `imap_fetcher.py`, `run_pipeline*.py` | Orchestrators |
| 8 | `bot/send_reports.py` + `bot/*.py` | Telegram UI + APScheduler |
| 9 | `collector/` | AI Debt Collector (standalone module) |

---

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
  every 10 min : run_pipeline_all_mp.py
  09:00        : collector --send (AI debt contacts)
  09:00        : inventory summary → admin
  10:00        : collector --check-promises
  14:00        : silence alerts + opportunity loss → managers
  20:00        : gross report
  21:00        : sales + silence alerts
  22:00        : daily analytics (all 6 analytics scripts)
  23:00        : daily summary → admin
  Mon 10:00    : weekly analytics
  03:00        : auto cleanup
```

---

## Collector Module (`collector/`)

AI Debt Collector — contacts overdue clients via WhatsApp/Telegram with manager approval flow.

```
collector/
    __init__.py
    debt_monitor.py         # debt JSON analysis, debtor classification (levels 0–5)
    collections_db.py       # state store → logs/collector_state.json (atomic JSON)
    communications.py       # WhatsApp (Green API) + Telegram gateway
    collection_agent.py     # DeepSeek message generation + response analysis
    voice_calls.py          # Retell AI voice calls (level >= 4)
    collections_engine.py   # main orchestrator + CLI
    manager_dialog.py       # manager approval dialog FSM
```

**Debt levels:**
```
0–9 days   → level 0 (skip)
10–14 days → level 1 (soft reminder)
15–19 days → level 2 (medium pressure)
20–24 days → level 3 (assertive)
25–29 days → level 4 (strict + call)
30+ days   → level 5 (hard + admin escalation)
```

**State files:**
- `logs/collector_state.json` — client contact history, promises, escalation flags
- `logs/collector_dialogs.json` — active manager approval dialogs
- `config/debtors_contacts.json` — client contact directory (WhatsApp/Telegram/phone)

**Key constraints:**
- Max 1 message per client per day
- Send only 09:00–18:00 Asia/Almaty, not weekends
- Voice calls only 09:00–17:00, level >= 4, `do_not_call=false`
- Violation threshold: `opening >= 100 AND debit > 0` (not `> 0` — avoids 1C rounding artifacts)
- `--dry-run` before every `--send`

**Manager dialog states:**
`AWAITING_CONFIRM` → `AWAITING_DATA` / `AWAITING_REJECTION_REASON` → `DEADLINE_SET` / `CONFIRMED` / `DONE`

---

## Roles and Access

| Role | Who | Access |
|------|-----|--------|
| admin | Вадим (7422963573) | Everything, all managers |
| subadmin | Алена (188939016) | Магира + Оксана reports + team mini-rating |
| manager | Оксана, Магира, Ергали, Алена | Own data only |

**Critical:** Алена has dual role — subadmin AND manager. She participates in sales logic as manager but gets team mini-rating as subadmin. Do not break this.

---

## Key Conventions

- **Paths**: always use `config.py` constants (`HTML_DIR`, `JSON_DIR`, `QUEUE_DIR`, `LOGS_DIR`, etc.) — never hardcode paths.
- **Logging**: logs go to `logs/<module>_YYYYMMDD_HHMMSS.log`; format: `"%(asctime)s, %(levelname)s %(message)s"`. Use `setup_logging()` from `config.py`. Never `logging.basicConfig()` at module level.
- **Report footer**: `"Сформировано: DD.MM.YYYY HH:MM (Asia/Almaty) | Версия: …"`.
- **Queue claiming**: files in `reports/queue/` are claimed by renaming `*.xlsx → *.xlsx.work`. After processing, moved to `reports/excel/processed/` via `_move_to_processed(work, src.name)`.
- **Report routing**: `run_pipeline_all_mp.py` classifies files by filename regex first, then by peeking at first ~50 rows. Patterns in `config/pattern_config.yaml`.
- **Timezone**: always `ZoneInfo(os.getenv("TZ", "Asia/Almaty"))` — never `timezone(timedelta(hours=5))` and never hardcoded `ZoneInfo("Asia/Almaty")`. Load `.env` before reading TZ.
- **Manager list**: `config/managers.json` is the single source of truth. Never hardcode manager names.
- **Debt key**: always `debt`, never `closing` — invariant, source of past analytics bugs.
- **No PDF**: no PDF output in this project, do not add.
- **Арман**: уволен, нигде не упоминать.
- **Version bump**: +0.0.1 to any modified file's `__VERSION__`. Run `python -m py_compile <file>` after each change.

---

## Configuration Files

- `.env` — `TG_BOT_TOKEN`, `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `TZ`, `GREENAPI_ID`, `GREENAPI_TOKEN`, `RETELL_API_KEY`, `RETELL_AGENT_ID`, `COMPANY_PHONE`, `OPENCLAW_ENABLED`, `COLLECTOR_HOUR_START`, `COLLECTOR_HOUR_END`, `COLLECTOR_CALL_LEVEL`, `WHATSAPP_ENABLED`, `RETELL_ENABLED`
- `config/managers.json` — manager names and Telegram `chat_id`
- `config/roles.json` — admin/subadmin scopes
- `config/imap.json` — IMAP server credentials
- `config/pattern_config.yaml` — regex patterns for report-type detection
- `config/debtors_contacts.json` — collector client contact directory

All `config/*.json` files and `.env` are gitignored (contain credentials).

---

## Report Outputs

| Directory | Contents |
|-----------|----------|
| `reports/html/` | Main HTML reports |
| `reports/json/` | Structured data for analytics layer |
| `reports/analytics/` | Analytics HTML (DSO, RFM, concentration, etc.) |
| `reports/ai/` | AI-generated txt + HTML commentary |
| `reports/queue/` | Incoming xlsx files to be processed |
| `reports/excel/active/` | Files currently being processed |
| `reports/excel/processed/` | Completed source files |

---

## HTML Templates

Jinja2 templates in `templates/` are rendered by Layer 4 report generators. `base.html` is the shared layout.
**CSS brand palette:** `--brand:#1a3a5c` (navy), `--accent:#0070c0` (blue), `--good:#107c41`, `--bad:#c00000`, `--warn:#e09000`, `--bg:#f0f4f8`
Layer 5 files use Python f-strings — CSS braces must be escaped as `{{`/`}}`.

---

## AI Prompts

Russian-language prompts for AI analysis in root-level `.txt` files:
- `ПРОМТ_ДЛЯ_ВАЛОВОЙ.txt` (gross profit)
- `ПРОМТ_ДЛЯ_ДЕБИТОРКИ.txt` (debt)
- `ПРОМТ_ДЛЯ_ЗАТРАТ.txt` (expenses)
- `ПРОМТ_ДЛЯ_ОСТАТКОВ.txt` (inventory)
- `ПРОМТ_ДЛЯ_ПРОДАЖ.txt` (sales)

---

## Test Suite (current: 2026-03-20)

| File | Tests | Coverage |
|------|-------|----------|
| `tests/test_project.py` | 62 | imports, helpers, send_tg, gender_emoji, silence_alerts, notify state, config, user_tracker |
| `tests/test_collector.py` | 106 | collector imports, classify, contacts, daily_summary, WhatsApp, manager dialog, violations |

Run: `python -X utf8 tests/test_project.py && python -X utf8 tests/test_collector.py`

---

## Known Open Issues (Audit 2026-03-20)

> **Полная история исправлений:** см. `SESSION_CONTEXT.md` (18 коммитов, сессии 1–3, 2026-03-19).
> После сессий 1–3 на F:/external: **0 критических / 0 высоких / 2 архитектурных**.

### OPEN — Архитектурные (не критично)
| ID | File | Issue |
|----|------|-------|
| ARCH-1 | `tools/txt_to_html.py` + `bot/send_reports.py` | `txt_to_html` в двух местах — разные интерфейсы, унификация ломает call sites |
| ARCH-3 | `expenses_parser.py` | Inline HTML в парсере — изолированный модуль, работает корректно |

### OPEN — LOW
- `bot/send_reports.py:388` — `logging.Formatter.formatTime` monkey-patch глобальный (BSR-01)

### FIXED — 2026-03-20 (C:/sync session)
| ID | Fix |
|----|-----|
| LOG-01 | httpx/httpcore silenced → `logging.WARNING` в `bot/send_reports.py:main()` |
| REQ-01 | `numpy>=1.26.0` добавлен в `requirements.txt` |
| RNR-01 | `run_new_reports_now.py` — sys.path + bot/ dir перед импортом |

---

## Fixed Bugs History

### Session 2026-03-19, part 3 (F:/external — коммиты 5c522ac–ac234ab)
| ID | Коммит | Fix |
|----|--------|-----|
| BUG-M2 | `5c522ac` | `gross/debt_auto` — regex `(?:покупатель\|контрагент)` |
| BUG-M17 | `e6dea07` | `inventory_summary` — regex вместо хрупкого split |
| BUG-L3,L4 | `488f739` | `inventory.py` категории из config; `inventory_cost_parser` — динамические индексы |
| ARCH-2,M11,M12 | `ac234ab` | `ai_analyzer` — убран mutable global; версии обновлены |

### Session 2026-03-19, part 2 (F:/external — коммиты 237bc6e–e48ba5a)
| ID | Коммит | Fix |
|----|--------|-----|
| BUG-H2 | `237bc6e` | `silence_alerts` — индексы колонок из `<thead>` |
| BUG-H4,L7 | `8712810` | `user_tracker.py` — `threading.Lock` + narrow except |
| BUG-H5 | `67768a7` | `debt_auto_report` — `config.HTML_DIR` вместо `getattr(OUT_DIR)` |
| BUG-M14 | `58da0fa` | `ai_analyzer.py` — 30+ `print()` → `logger` |
| BUG-M7,L5,M9 | `cfb5d22` | `send_reports` fd leak + PID print→logger + failed counter |
| BUG-M15,M16 | `2741ae9` | `collector/` — bool env + CALL_HOUR_END из env |
| BUG-L1,L2,L8 | `e48ba5a` | `inject_local`, `txt_to_html`, `opportunity_loss` |

### Session 2026-03-19, part 1 (F:/external — коммиты 6383aa5–7877a87)
| ID | Коммит | Fix |
|----|--------|-----|
| BUG-C2 | `6383aa5` | `gross_report_pct._money_to_float` — запятая в regex |
| BUG-M1 | `1413f9a` | дубликат `_try_extract_meta` → импорт из `gross_report.py` |
| BUG-H1 | `27c1d35` | `opportunity_loss.py` — мёртвый ключ `"dead"` |
| BUG-H3 | `adc61bc` | `imap_fetcher.py` — `load_dotenv()` перед TZ |
| BUG-M3,M5 | `91d0d57` | `sales_parser/report.py`, `config.py` — except + TZ env |
| BUG-M6,M10 | `651bb36` | `send_reports` TZ; `silence_alerts` dead import |
| BUG-M8 | `7877a87` | хардкод "Минай" → `_SYSTEM_ACCOUNTS` из roles.json |

### Session 2026-03-20 (C: collector audit — до объединения)
| ID | File | Fix |
|----|------|-----|
| B1 | `collector/collections_engine.py` | `result["sent"]`/`result["dialog_started"]` not set in dialog path |
| B2 | `collector/collections_engine.py` | `generate_message()` called before manager-busy check → 48 wasted API calls |
| B4 | `collector/debt_monitor.py` | Violation threshold `opening > 0` → `opening >= 100` |

### Session 2026-03-13 (main bot audit)
| ID | File | Fix |
|----|------|-----|
| S1 | `bot/send_reports.py` | Hardcoded subadmin `"Алена"` → loop over `ROLES["subadmin_scopes"]` |
| S2 | `bot/send_reports.py` | `GENDER_MAP` hardcoded names → heuristic by name ending |
| S3 | `bot/send_reports.py` | `_mark_notified_today` called undefined `_parse_period_date` → inlined |
| S4 | `send_tg.py` | `parse_mode=None` serialized as `"None"` → key omitted when falsy |
| S5 | `send_tg.py` | `print()` → `logger.info()` |
| A1 | `bot/sales_summary.py` | 16× `except Exception` → specific types |
| A2 | `bot/user_tracker.py` | TZ hardcode → `ZoneInfo(os.getenv("TZ","Asia/Almaty"))` |
| A3 | `bot/send_reports.py` | 4 missing `schedule_message_deletion` calls added |
| A4 | `bot/gross_summary.py` | 2× `except Exception` → specific types |

### Session 2026-03-10 (pipeline audit)
| ID | File | Fix |
|----|------|-----|
| C1 | `run_pipeline_all_mp.py:_move_to_processed` | Was looking for original file (already renamed to `.work`) |
| C2 | `run_pipeline_all_mp.py:_log` | File handle leaked on every call |
| C3 | `ai_analyzer.py:analyze` | JSON truncated mid-token |
| H1 | `dso_aging_report.py`, `revenue_concentration_report.py` | `timezone(timedelta(hours=5))` → `ZoneInfo` |
| H2 | `ai_analyzer.py` | Hardcoded manager name list → reads `managers.json` |
| H3 | `run_pipeline_all_mp.py` | `p.stat().st_mtime` TOCTOU crash → `try/except FileNotFoundError` |
| M1 | `bot/send_reports.py` | Bare `except:` → specific types |
| M2 | `expenses_parser.py` | Default TZ `"Asia/Qyzylorda"` → `"Asia/Almaty"` |
| L1 | `bot/inventory_summary.py` | `datetime()` calls individually guarded with `try/except ValueError` |

---

## Repository Navigation Rules

When working in this repository:

1. Read `CLAUDE.md` (this file) — full project context
2. Read `repo_map.json` — machine-readable file index
3. Only then open the exact Python files required for the task

Rules:
- Do not broadly rescan the repository. Use `repo_map.json` as the primary file map.
- Prefer targeted file reads over full-project scans.
- Before making edits: identify exact files affected, explain root cause, apply minimal patch, validate syntax (`python -m py_compile`), bump version.
- Do not refactor unrelated files.
- Never introduce architectural changes unless explicitly requested.

---

## Operational Rules for Claude Code

### 1. Editing Safety Rules

- Never refactor unrelated modules.
- Never change public interfaces unless required for a specific bug fix.
- Prefer **minimal patch surface**.
- Always explain the root cause before editing.

Order of work:
1. Explain root cause
2. Identify exact files affected
3. Apply minimal patch
4. Validate syntax (`python -m py_compile <file>`)
5. Bump `__VERSION__` +0.0.1
6. Summarize changes

### 2. Pipeline Safety Rules

File lifecycle:
```
queue/file.xlsx → (claim) → queue/file.xlsx.work → (process) → reports/excel/processed/file.xlsx
```

Never:
- delete queue files directly
- rename files outside claim/release logic
- bypass `_move_to_processed()`

All pipeline operations must be atomic.

### 3. Concurrency Rules

`run_pipeline_all_mp.py` runs multi-process. File operations must be atomic. `FileNotFoundError` must never crash the pipeline loop. Queue iteration must be defensive.

### 4. Timezone Rules

```python
# CORRECT
ZoneInfo(os.getenv("TZ", "Asia/Almaty"))

# WRONG — do not use
timezone(timedelta(hours=5))
ZoneInfo("Asia/Almaty")  # hardcode without env
datetime.now()           # naive datetime
```

Layer 5 analytics files are standalone — they must call `load_dotenv()` themselves before reading `TZ`.

### 5. Security Rules

- Never log BOT_TOKEN or API keys. Add to `bot/send_reports.py` startup:
  ```python
  logging.getLogger("httpx").setLevel(logging.WARNING)
  logging.getLogger("httpcore").setLevel(logging.WARNING)
  ```
- All secrets in `.env`, `.env` in `.gitignore`.
- Client names/amounts must NOT appear in log messages — only statuses.

---

## Dependency Map (key relationships)

```
utils_common.py → config.py → utils_excel.py → all parsers/reporters
                           → utils.py → Jinja environment

analyze_debt_excel.py → debt_auto_report.py → run_pipeline_all_mp.py

bot/send_reports.py (MONOLITH, Layer 8):
  imports: silence_alerts, user_tracker, *_summary (3), opportunity_loss
  subprocess: run_pipeline_all_mp, ai_analyzer,
              dso/rfm/concentration/turnover/net_profit/profitability

collector/ → send_tg, collections_db, communications, collection_agent,
             voice_calls, manager_dialog → bot/send_reports.py (scheduler)
```

**Safe to modify (isolated):** All Layer 5 modules (dso, rfm, concentration, turnover, net_profit, profitability) — no project imports except concentration→utils_common.

**Known architectural issues:**
- D1: `expenses_parser.py` + `expenses_report.py` — intentional shim (compatibility)
- D2: `sys.path` hack in `bot/send_reports.py` for bot/ directory imports
- D3: `run_new_reports_now.py` imports `bot/send_reports` as module — fragile
- D5: `money()` function duplicated in `debt_auto_report.py` with different signature

---

## Sales Logic (do not revert)

- Manager sales files accumulate in `_pipeline_managers_by_period`
- Summary goes to admin
- Each manager gets their own result
- Алена as subadmin gets separate team mini-rating
- Manager identified by **filename**, not by JSON `manager` field

# STATUS.md
Version: v2026-03-09
Timezone: Asia/Almaty
Status: auto-updated by sync_project_to_github.ps1

# GPT1C_Processor / AI 1C PRO - STATUS

## 1. Project

Local production project: E:\GPT1C_Processor_analitic

GitHub repository: https://github.com/gpt-project213/GPT1C_Processor

Branch: master

RAW base: https://raw.githubusercontent.com/gpt-project213/GPT1C_Processor/master

---

## 2. Confirmed

- production code is published to GitHub
- local GitHub sync works
- raw_links.txt generation is built in
- repo_map.json generation is built in
- STATUS.md generation is built in
- AI_CONTEXT.md generation is built in

---

## 3. Current versions (production)

| File | Version | Notes |
|------|---------|-------|
| bot/send_reports.py | v9.4.32 | #INV-1 #MENU-SILENCE #AI-MENU fixed; oploss weekly |
| bot/inventory_summary.py | v1.2 | glob pattern fix |
| bot/opportunity_loss.py | v1.5.1 | manager-specific gross fix |
| ai_analyzer.py | v9.4.18 | AI input 80K→15K chars |
| run_pipeline_all_mp.py | v1.5 | EXPENSE routing added |
| revenue_concentration_report.py | v1.1.3 | normalize_client_name → utils_common |
| utils_common.py | v1.0.0 | NEW — shared utilities |
| dso_aging_report.py | v1.1.0 | period_days, aging coefficients |

---

## 4. Scheduler (bot/send_reports.py v9.4.32)

| Time | Job |
|------|-----|
| every 10 min | pipeline_task |
| 09:00 daily | inventory_summary → admin |
| 14:00 daily | silence_alerts → managers |
| 14:05 **Friday** | opportunity_loss → managers (weekly) |
| 20:00 daily | gross_summary |
| 21:00 daily | sales_summary + silence_alerts |
| 22:00 daily | daily_analytics (6 analytics) |
| 23:00 daily | daily_summary → admin |
| Mon 10:00 | weekly_ai_generation |

---

## 5. Completed bug fixes (audit 09.03.2026)

| # | File | Fix |
|---|------|-----|
| INV-1 | inventory_summary.py | inventory_simple_*.html → inventory_*.html |
| MENU-SILENCE | send_reports.py | send_main_menu() after silence/oploss alerts |
| AI-MENU | send_reports.py | AI button moved from debt menu → analytics |
| PIPE-1 | run_pipeline_all_mp.py | EXPENSE branch added, else→SKIP |
| OPLOSS-1 | opportunity_loss.py | fallback to wrong manager's gross removed |
| AI-1 | ai_analyzer.py | AI_MAX_INPUT_CHARS 80000→15000 |
| RC-1 | revenue_concentration_report.py | normalize_client_name → utils_common.py |

---

## 6. Audit layer gaps (still open)

- README.md
- SECURITY.md
- .env.example
- config/imap.example.json
- config/managers.example.json
- config/roles.example.json

---

## 7. Purpose

- short project status
- recovery point
- GitHub audit checkpoint
- current audit layer reference

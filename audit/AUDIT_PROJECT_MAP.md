# AUDIT_PROJECT_MAP

## 1) Точки входа

- `bot/send_reports.py` — основной runtime (Telegram polling + scheduler + pipeline orchestration).
- `imap_fetcher.py --once` — забор вложений из IMAP в `reports/queue` + clean-копии.
- `run_pipeline_all_mp.py` — отдельный оркестратор queue-пайплайна (DEBT/SALES/GROSS/INVENTORY/EXPENSE).
- `collector/collections_engine.py` — standalone AI Collector (`--dry-run`, `--send`, `--check-promises`).

## 2) Карта модулей

### Входящий поток
- `imap_fetcher.py`: IMAP connect/login/search/fetch, whitelist, manager-in-filename filter, queue save, clean copy, delete/expunge.
- `run_pipeline_all_mp.py`: claim `.xlsx -> .work`, route by filename/content, запуск нужного билдера, move to processed.

### Парсеры/генераторы
- `debt_auto_report.py` -> debt html/json.
- `sales_report.py` + `sales_parser.py` -> sales html/json.
- `gross_report.py` + `gross_report_pct.py` + `gross_parser.py` -> gross html/json.
- `inventory.py` + `inventory_cost_parser.py` -> inventory html/json.
- `expenses_report.py` + `expenses_parser.py` -> expenses html/json.

### Telegram runtime
- `bot/send_reports.py`: меню, ACL, отправка отчётов, архив, daily/weekly jobs, janitor, workflow callbacks.
- `send_tg.py`: low-level Telegram API helper.

### CRM / Collector / Debt Stop
- `bot/crm_clients.py`: автосинхронизация клиентов из JSON-отчётов, запрос телефонов.
- `collector/collections_engine.py`: оркестратор контактов должников.
- `collector/collections_db.py`: state (`logs/collector_state.json`) и антидубликатные флаги.
- `collector/dialog_store.py` + `collector/manager_dialog.py`: FSM согласования с менеджером.
- `bot/debt_stop_control.py`: контур stop-list (`debt_stop_state`, `debt_stop_registry`) и эскалации.

## 3) Карта конфигов

Найдены в snapshot:
- `config/pattern_config.yaml`
- `config/weekly_clients.json`

Ожидаются кодом, но отсутствуют в snapshot репозитория:
- `.env`
- `config/managers.json`
- `config/roles.json`
- `config/imap.json`
- `config/clients.json`
- `config/debtors_contacts.json`

## 4) Карта state/памяти между перезапусками

- `logs/collector_state.json` — история контактов, promise, escalation, daily dedupe.
- `logs/collector_dialogs.json` — активные manager dialogs.
- `logs/deletion_queue.json` — очередь автоудаления сообщений Telegram.
- `logs/notify_state.json` — state дедупликации нотификаций/доставок.
- `reports/debt_stop_state.json` — суточные кандидаты стоп-листа.
- `reports/debt_stop_registry.json` — постоянный реестр исключений/нарушителей.
- `config/weekly_clients.json` — weekly исключения по stop/silence логике.

## 5) Карта вход/выход каталогов

Вход:
- `reports/queue/` (xlsx/xls)
- IMAP inbox

Промежуточные:
- `reports/excel/clean/`
- `reports/excel/active/`
- `reports/excel/processed/`

Выход:
- `reports/html/`
- `reports/json/`
- `reports/analytics/`
- `reports/ai/`

Логи:
- `logs/*.log`
- snapshot: `logs_public/*.log`

## 6) Job-схема (по коду)

`bot/send_reports.py` регистрирует:
- repeating: pipeline/new_reports/janitor/crm_phone_reminders/collector_reminders/whatsapp_poller
- daily: silence alerts (14:00/21:00), opportunity loss (14:05), AI reset, daily analytics, summaries (inventory/gross/sales), cleanup, workday check, CRM daily, collector daily/promises
- daily debt-stop jobs (если модуль импортирован): monitor/managers/escalate/saida
- polling mode: `application.run_polling(drop_pending_updates=True)`

## 7) Карта потоков

### Поток A
email(IMAP) → `reports/queue` → route/build (reports) → HTML/JSON → bot index → user delivery.

Критические блокировки:
- whitelist reject
- manager-in-filename reject
- route=SKIP
- build without outputs
- ACL/chat-id delivery issues

### Поток B
JSON debt/sales → CRM update (`clients.json`) → manager phone request → response → state update.

Критические блокировки:
- отсутствующие config-файлы
- daily dedupe в collector/crm
- pending flags в dialog/state

### Поток C
debt JSON → debt_stop candidate generation → manager request → escalation → final Saida list + registry/state writes.

Критические блокировки:
- отсутствие managers/roles/admin IDs
- stale registry/state entries
- unanswered pending branches

### Поток D
state/log driven repeat cycle (janitor, reminders, promises, collector)

Критические блокировки:
- broken/inconsistent state json
- date-based blocks not reset
- Telegram delivery failures

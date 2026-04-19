# Комплексный аудит проекта

Дата: 2026-04-19  
Статус: актуализировано после коммитов до `25bee91`  
Проект: `E:\GPT1C_Processor_analitica`

## Цель

Собрать в одном файле актуальные результаты комплексного аудита, проверки коммитов,
тестов и текущих рисков проекта. Старые audit-выводы сверены с текущим кодом и
последними коммитами: устаревшие баги не считаются открытыми.

## Изученные материалы

Прочитаны:

- корневые Markdown-файлы: `CLAUDE.md`, `SESSION_CONTEXT.md`, `BUGFIX_PLAN_2026-04-15.md`, `COLLECTOR_CLASSIFICATION_RECONCILIATION_2026-04-12.md`, `gpt1c.md`;
- папки `audit/` и `аудит/`;
- session/context файлы: `audit/SESSION_CONTEXT.md`, `audit/CODEX_AUTOSAVE_CONTEXT_2026-04-12.md`, `audit/CODEX_SUPPORT_CONTEXT_2026-04-11.md`;
- `.claude/sessions/2026-04-11-wa-no-send-session.tmp`;
- `openclaw/SOUL.md`, `openclaw/skills/debt_collector.md`;
- последние коммиты git до HEAD `25bee91`.

Большие `audit/logs/*.log` не загружались полностью: для архитектурного аудита
использовались уже подготовленные incident/audit документы и точечные проверки кода.

## Git-состояние

Текущая ветка:

```text
master
```

Текущий HEAD:

```text
25bee9158e62709ae95746b9bef8e736a7c30318
```

Последние важные коммиты:

| Коммит | Статус / смысл |
|---|---|
| `25bee91` | `collector_state.json` защищён межпроцессной блокировкой `portalocker`. |
| `6582ed9` | Документирован production Data Flow: основной путь `pipeline_task()` в `bot/send_reports.py`; `run_pipeline_all_mp.py` — ручной инструмент. |
| `be1f7a8` | Удалены DEAD-3/DEAD-4 в `collector/manager_dialog.py`. |
| `306385a` | `_save_contact()` теперь пишет в CRM (`config/clients.json`) через `set_client_details()`. |
| `c30a2bb` | `load_contacts_compat()` теперь только CRM; legacy merge с `debtors_contacts.json` убран. |
| `28b73d8` | Удалены мёртвые импорты `analyze_response`, `get_call_result`. |
| `51f7892` | T4 в `tests/test_project.py` мокирует `_log`, ERROR больше не загрязняет production-log. |
| `8aede7e` | `imap_fetcher._save_bytes()` пишет атомарно через `NamedTemporaryFile` + `os.replace`. |
| `9e97a2b` | `ai_analyzer.py` получил retry 3x backoff через `tenacity`. |
| `0464c2f` | Weekly callback overflow исправлен: вместо `client_name` используется token 8 hex. |
| `76de59c` | `test_phase2_safe_send.py` актуализирован: `WA=1` → `--preview`, `WA=0` → `--dry-run`. |
| `87f07cc` | `ai_analyzer.py`: `timeout=120`. |
| `ea2e567` | `sales_parser.py`: убран fallback `client_j→product_j`, клиент и товар строго разделены. |
| `a3331a8` | CRM очищена от товарных/metadata записей; добавлен guard от мусора из sales JSON. |
| `5dc633b` | AssemblyAI исправлен: `speech_model="best"`. |
| `65d618d` | Закрыт `InlineKeyboardMarkup` / `UnboundLocalError`, удалён тестовый телефон. |

## Текущий рабочий статус

На момент актуализации:

```text
 M CLAUDE.md
?? =2.8.2
?? COMPREHENSIVE_AUDIT_2026-04-19.md
?? gpt1c.md
```

Примечания:

- `CLAUDE.md` изменён Claude: Issue-4 обновлён как реализованный.
- `COMPREHENSIVE_AUDIT_2026-04-19.md` и `gpt1c.md` — локальные контекстные артефакты.
- `=2.8.2` выглядит как случайный артефакт после команды установки/зависимости; в рамках аудита не трогался.

## Результаты тестов

Обычный sandbox запуск упирался в `PermissionError` на запись `logs/*.log`.
Ключевые тесты перезапущены с разрешением на запись логов.

```text
python -X utf8 tests/test_project.py
```

Результат:

```text
81/81 OK
```

```text
WHATSAPP_ENABLED=0 LIVE_SEND_ALLOWED=0 python -X utf8 tests/test_collector.py
```

Результат:

```text
256/256 OK
```

```text
WHATSAPP_ENABLED=0 LIVE_SEND_ALLOWED=0 python -X utf8 tests/test_phase2_safe_send.py
```

Результат:

```text
PHASE2 SAFE SEND TESTS PASSED
```

```text
python -X utf8 tests/test_audit_reports_20260414.py
```

Результат:

```text
8/8 OK
```

Ранее также проходил:

```text
python -X utf8 tests/test_parsers.py
```

Результат:

```text
61/61 OK
```

## Закрытые пункты, которые больше нельзя считать open

### Issue-4: условная отгрузка

Закрыто. В `bot/debt_stop_control.py` реализованы:

- 4 admin-кнопки в `escalate_unanswered()`:
  - `dstop_admin_remove|`
  - `dstop_admin_limit_after|`
  - `dstop_admin_block_until|`
  - `dstop_admin_ok|`
- 4 admin-кнопки в `monitor_exceptions()` при долге `<= FULL_PAYMENT_THRESHOLD`:
  - `dstop_clear|`
  - `dstop_limit_clear|`
  - `dstop_keep_until_paid|`
  - `dstop_keep|`
- callback routing для всех веток;
- `_handle_conditional_clearance()`, который ставит статус `conditional` и уведомляет Саиду/менеджера.

### Weekly callback overflow

Закрыто коммитом `0464c2f`. Полный `client_name` больше не кладётся в
`callback_data`; используется короткий token 8 hex.

### AI retry

Закрыто коммитом `9e97a2b`. В `ai_analyzer.py` есть:

- `tenacity.retry`;
- `stop_after_attempt(3)`;
- `wait_exponential`;
- retry на `APIConnectionError`, `APITimeoutError`, `RateLimitError`;
- `timeout=120`.

### IMAP attachment atomic write

Закрыто коммитом `8aede7e`. `_save_bytes()` пишет во временный файл в той же
директории и затем делает `os.replace`.

### Safe-send test conflict

Закрыто коммитом `76de59c`. Тестовый контракт теперь соответствует Phase 3:

- при `WHATSAPP_ENABLED=1` scheduler вызывает `--preview`;
- при `WHATSAPP_ENABLED=0` scheduler вызывает `--dry-run`;
- legacy `--send` отключён.

### Test-log pollution

Закрыто коммитом `51f7892`: проблемный тест мокирует `_log`.

### Dead collector code

Закрыто коммитами `28b73d8` и `be1f7a8`:

- удалены мёртвые импорты `analyze_response`, `get_call_result`;
- удалены `_send_control_reminder`, `_on_update`, `col_update_` ветка.

### `collector_state.json` race

Частично закрыто коммитом `25bee91`. Для `collector_state.json` добавлен
`portalocker` lock на `.lock` файл. Это закрывает race бот ↔ subprocess именно
для `collector_state.json`.

### SALES production pipeline

Старый вывод про `run_pipeline_all_mp.py` как production-баг устарел.
Коммит `6582ed9` уточняет: production-путь — `pipeline_task()` в
`bot/send_reports.py`, и он запускает оба контура SALES:

- `sales_report.py`;
- `sales_parser.py`.

`run_pipeline_all_mp.py` остаётся ручным инструментом.

### Товары в CRM

Закрыто для штатного pipeline. CRM пополняется из `sales JSON clients[]`, а
`sales_parser.py` после `ea2e567` строго различает client/product колонки и не
делает fallback клиента на товарную колонку.

Фактическая проверка 2026-04-19:

```text
config/clients.json: 475 клиентов, 0 product/metadata hits
последние 20 sales_*.json: 0 product/metadata hits в clients[]
```

Нюанс: текущий `bot/crm_clients.py` больше не содержит отдельный `_PRODUCT_NAME_RE`.
Защита находится в parser-слое. Если в CRM вручную или внешним скриптом попадёт
битый sales JSON, где товар уже записан как `clients[].client`, `crm_clients.py`
отфильтрует metadata строки, но не все товарные названия по тексту.

## Старые закрытые баги

| Тема | Статус |
|---|---|
| `InlineKeyboardMarkup` / `UnboundLocalError` | Закрыто `65d618d`. |
| AssemblyAI `speech_models` | Закрыто `5dc633b`: используется `speech_model="best"`. |
| `.work` recovery | Закрыто `1907556`: `.work` возвращается в `.xlsx` при ошибке. |
| DSO `closing` fallback | Закрыто `cb1d6c3`: enforced `debt`. |
| `chat_id` строкой | Закрыто `a831ee5`: `int(val)`. |
| Unknown user ACL | Закрыто `8bed42f` + `ef82c7b`: `_acl_gate()`. |
| RFM/Concentration dedup | Закрыто `0962b95`: дедуп по `total`. |
| Debt stop fixed `.tmp` | Закрыто `d2a6908`. |
| WhatsApp incident guards | Закрыто `18a8ea8` + `5d9a3fa` + последующие approval-flow фиксы. |

## Актуальные риски после последних коммитов

### 1. Остальные JSON-state файлы без подтверждённого lock

`collector_state.json` теперь защищён, но это не означает автоматическую защиту
всех state-файлов. Отдельной проверки требуют:

- `logs/wa_approval_batches.json`;
- `logs/collector_client_dialogs.json`;
- `logs/collector_dialogs.json`;
- `logs/saida_payment_holds.json`;
- `logs/collector_shipment_decisions.json`;
- `config/clients.json`.

Подход: не добавлять lock массово. Сначала проверить owners/read-modify-write
циклы и добавить lock только там, где есть реальный concurrent write.

### 2. Контакты и CRM после отказа от legacy merge

Последние коммиты перевели collector contact source в CRM:

- `load_contacts_compat()` теперь только `config/clients.json`;
- `_save_contact()` пишет в CRM через `set_client_details()`;
- legacy `debtors_contacts.json` больше не должен участвовать в merge.

Это правильное направление, но требует отдельной сверки данных:

- сколько должников не матчится с CRM по имени;
- сколько телефонов потеряно после отказа от legacy merge;
- какие записи в `config/debtors_contacts.json` и `collector/debtors_contacts.json` теперь являются архивным хвостом;
- нужен ли backup и migration note.

### 3. Live WhatsApp контур

Live-контур стал безопаснее, но остаётся зоной повышенного риска.

Перед любым live:

1. `git status --short`;
2. backup state/config файлов;
3. проверка workday/time window;
4. `--dry-run`;
5. `--preview`;
6. manager approval;
7. admin approval;
8. только затем `--send-approved --batch-id <id>`.

Legacy `--send` не использовать.

### 4. `bot/send_reports.py` остаётся монолитом

Файл около 7700 строк и содержит scheduler, callbacks, отчёты, ACL, collector
интеграции. Даже после закрытия конкретных багов это главный архитектурный риск.

Подход: только точечные изменения, без широкого рефакторинга рядом с live/send
логикой.

### 5. Данные и runtime-state

Из session context остаются важные эксплуатационные задачи:

- backup перед чисткой state;
- сверка старых runtime записей;
- проверка test-like хвостов;
- сверка CRM/contact ownership;
- финальная приёмка по живым сценариям.

Автоматически чистить данные нельзя без отдельного разрешения.

### 6. Автомониторинг логов

Добавлен встроенный мониторинг логов для удобной последующей диагностики:

- модуль: `bot/log_monitor.py`;
- scheduler job: `log_monitor` каждые 2 часа;
- state: `logs/log_monitor_state.json`;
- summary: `logs/log_monitor_summary.log`;
- Telegram alert отправляется admin только при новых проблемах;
- первый запуск делает baseline текущих логов и не рассылает старые ошибки;
- `[TEST]` строки игнорируются;
- собственный summary-файл не сканируется.

Проверка:

```text
python -X utf8 tests/test_log_monitor.py
8/8 OK
```

## Конфигурация и зависимости

Основные зависимости актуально включают:

```text
openai==2.8.0
tenacity==8.2.3
portalocker>=2.8.2
python-telegram-bot==20.7
APScheduler==3.11.0
pandas==2.2.2
openpyxl==3.1.2
```

Наблюдения:

- `tenacity` теперь реально используется в `ai_analyzer.py`.
- `portalocker` теперь реально используется в `collector/collections_db.py`.
- `.env` на машине содержит live-флаги `WHATSAPP_ENABLED=1` и `LIVE_SEND_ALLOWED=1`; тесты должны задавать безопасный env явно.
- `ASSEMBLYAI_SPEECH_MODELS` в `.env` остаётся legacy-настройкой; код использует `speech_model="best"`.

## Архитектурные выводы

1. Большинство ранее найденных P2/P3 технических багов уже закрыто последними коммитами.
2. Текущая ключевая test suite зелёная: 81/81, 256/256, phase2 safe-send passed, audit 8/8.
3. Production pipeline для SALES не равен `run_pipeline_all_mp.py`; основной путь — `bot/send_reports.py::pipeline_task()`.
4. Collector state стал безопаснее благодаря lock для `collector_state.json`, но остальные JSON-state требуют отдельной проверки.
5. Самый большой риск теперь не конкретный найденный баг, а работа с live WhatsApp, CRM/contact migration и монолитным scheduler/callback файлом.

## Безопасный следующий план

### Шаг 1. Не трогать live-send без runbook

Любая live-проверка только после backup и явного разрешения.

### Шаг 2. Проверить contact/CRM последствия последних коммитов

Read-only аудит:

- сравнить текущих debt candidates с `config/clients.json`;
- посчитать matched/unmatched;
- отдельно вывести телефоны, которые были только в legacy `debtors_contacts.json`;
- ничего не мигрировать без подтверждения.

### Шаг 3. Проверить locks для остальных state-файлов

Не добавлять lock массово. Сначала составить таблицу:

- файл;
- кто пишет;
- есть ли read-modify-write;
- есть ли параллельный writer;
- нужен ли lock.

### Шаг 4. После каждого изменения запускать минимальный набор

```text
python -X utf8 tests/test_project.py
WHATSAPP_ENABLED=0 LIVE_SEND_ALLOWED=0 python -X utf8 tests/test_collector.py
WHATSAPP_ENABLED=0 LIVE_SEND_ALLOWED=0 python -X utf8 tests/test_phase2_safe_send.py
python -X utf8 tests/test_audit_reports_20260414.py
```

### Шаг 5. Не считать старые audit-пункты актуальными без recheck

Перед любой правкой:

1. проверить session/context;
2. проверить последние коммиты;
3. проверить текущий код;
4. только потом делать вывод.

### Шаг 6. Правила тестов и доказательств

Для дальнейших правок зафиксирован обязательный порядок:

1. session/context;
2. последние коммиты;
3. текущий код;
4. root cause;
5. минимальный патч;
6. `py_compile`;
7. релевантные тесты;
8. доказательства в ответе.

Базовый набор для runtime-правок:

```powershell
python -m py_compile <изменённые .py файлы>
python -X utf8 tests/test_project.py
$env:WHATSAPP_ENABLED='0'; $env:LIVE_SEND_ALLOWED='0'; python -X utf8 tests/test_collector.py
python -X utf8 tests/test_log_monitor.py
```

Collector / WhatsApp / approval / scheduler:

```powershell
$env:WHATSAPP_ENABLED='0'; $env:LIVE_SEND_ALLOWED='0'; python -X utf8 tests/test_phase2_safe_send.py
```

Reports / parsers / analytics:

```powershell
python -X utf8 tests/test_parsers.py
python -X utf8 tests/test_audit_reports_20260414.py
```

Для docs-only изменений тесты не обязательны, но нужно явно указать, что код продукта не менялся.

## Итог

Проект после серии коммитов 2026-04-19 стал заметно актуальнее и безопаснее:
weekly callback, AI retry, IMAP atomic write, safe-send тесты, dead code и
`collector_state.json` race уже закрыты.

Готовность к точечной работе: высокая при соблюдении SAFE SURGERY.  
Готовность к массовому рефакторингу: низкая.  
Готовность к live WhatsApp действиям: только через backup/runbook/approval-flow.

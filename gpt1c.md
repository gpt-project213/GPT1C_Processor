# GPT1C / Минбаракат — Project Skill

Специализированный скилл для работы с `GPT1C_Processor_analitica` (Минбаракат, Алматы).
Активируй через `/gpt1c` когда нужно: исправить баг, добавить функцию, проверить логи, разобраться с коллектором.

---

## Контекст проекта

**Компания:** Минбаракат, оптовые продукты питания, Алматы, Казахстан  
**Стек:** Python 3.11+, python-telegram-bot, APScheduler, DeepSeek/OpenAI, Green API (WhatsApp), AssemblyAI, Retell AI  
**Точка входа:** `bot/send_reports.py` (монолит Layer 8)  
**TZ:** Asia/Almaty → `ZoneInfo(os.getenv("TZ", "Asia/Almaty"))` — всегда через env  
**Venv:** `.venv/` | **Tests:** `python -X utf8 tests/test_project.py && python -X utf8 tests/test_collector.py`

---

## Архитектура (9 слоёв)

```
Layer 0: utils_common.py, send_tg.py           — нет project-импортов
Layer 1: config.py                              — пути, TZ, логи, роли
Layer 2: utils_excel.py, utils.py              — Excel, Jinja2
Layer 3: *_parser.py                            — xlsx → DataFrame/dict
Layer 4: *_report.py (debt/sales/gross/inv/exp) — xlsx → HTML + JSON
Layer 5: dso/rfm/concentration/turnover/...     — JSON → аналитика HTML
Layer 6: ai_analyzer.py                         — HTML → AI комментарий
Layer 7: imap_fetcher.py, run_pipeline_all_mp.py — оркестраторы
Layer 8: bot/send_reports.py + bot/*.py         — Telegram UI + APScheduler
Layer 9: collector/                             — AI Debt Collector
```

**Безопасно трогать (изолированы):** Layer 5 модули, отдельные парсеры, collector/ — если не меняешь интерфейсы.  
**Монолит:** `bot/send_reports.py` — трогать только точечно, не рефакторить рядом.

---

## Правило SAFE SURGERY (обязательно соблюдать)

**Каждый баг — отдельный коммит. Порядок:**
```
0. Сначала прочитать session/context, последние коммиты и текущий код
1. Объяснить root cause и доказать, что баг актуален сейчас
2. Определить точные файлы (только нужные)
3. Применить минимальный патч
4. python -m py_compile <изменённые .py файлы>
5. Bump __VERSION__ +0.0.1 в изменённом runtime-файле
6. Прогнать релевантные тесты из матрицы ниже
7. В ответе указать доказательства: файлы/строки, тесты, коммиты
8. git commit -m "fix(<module>): <описание>" — только если пользователь просит коммит
```

### Обязательная последовательность перед выводами и правками

1. Прочитать актуальные session/context материалы:
   `SESSION_CONTEXT.md`, `audit/SESSION_CONTEXT.md`, `audit/CODEX_*`, `.claude/sessions/*` при наличии.
2. Проверить последние коммиты: `git log --oneline -20` и relevant `git show`.
3. Проверить текущий код и tests, а не доверять старому audit-документу.
4. Если старый баг уже закрыт коммитом или тестом — не чинить повторно.
5. Перед правкой написать, какие файлы будут изменены и почему.
6. После правки показать доказательства: команды тестов, результат, затронутые файлы.

### Матрица обязательных тестов

Базовый набор после любой runtime-правки:

```powershell
python -m py_compile <изменённые .py файлы>
python -X utf8 tests/test_project.py
$env:WHATSAPP_ENABLED='0'; $env:LIVE_SEND_ALLOWED='0'; python -X utf8 tests/test_collector.py
```

Если менялся collector / WhatsApp / approval / scheduler:

```powershell
$env:WHATSAPP_ENABLED='0'; $env:LIVE_SEND_ALLOWED='0'; python -X utf8 tests/test_phase2_safe_send.py
```

Если менялись отчёты, parser, analytics, Excel→HTML/JSON:

```powershell
python -X utf8 tests/test_parsers.py
python -X utf8 tests/test_audit_reports_20260414.py
```

Если менялся только `.md`/контекст:

```text
Тесты не обязательны, но нужно указать: "код продукта не менялся".
```

Если тест падает из-за sandbox permission на `logs/*.log`, повторить с разрешённой записью
в проектные logs и явно указать, что первичная ошибка была инфраструктурной.

### Правила доказательств

- Любой вывод "закрыто" должен иметь одно из доказательств: commit hash, grep/code location, тест или фактическую проверку данных.
- Любой вывод "открыто" должен иметь reproduction: строка кода, failing test, byte/format check, log evidence или data count.
- В финальном ответе всегда отделять "исправлено кодом" от "осталось как эксплуатационная приёмка".
- Не использовать старые audit-выводы как доказательство без recheck по текущему HEAD.

**НИКОГДА:**
- Не рефакторить соседние модули при исправлении бага
- Не менять публичные интерфейсы без явного запроса
- Не хардкодить пути — использовать `config.py`: `HTML_DIR`, `JSON_DIR`, `LOGS_DIR` и т.д.
- Не использовать `timezone(timedelta(hours=5))` или `ZoneInfo("Asia/Almaty")` без env
- Не использовать `logging.basicConfig()` на уровне модуля (использовать `setup_logging()`)
- Не добавлять PDF вывод
- Не использовать `--send` напрямую (отключён) — только `--send-approved`
- Не упоминать Армана (уволен)

---

## Критические инварианты

| Правило | Суть |
|---------|------|
| `debt` ключ | Всегда `debt`, никогда `closing` — архивный баг, источник аналитических ошибок |
| Сводные файлы | `manager="—"` (em-dash) — только для admin; subadmin/manager не получают |
| Алена | dual-role: subadmin + manager (Оксана+Магира) — не ломать ни одну ипостась |
| Violation threshold | `opening >= 100 AND debit > 0` (не `> 0` — артефакты округления 1С) |
| Queue lifecycle | `*.xlsx` → `.xlsx.work` → `processed/` через `_move_to_processed()` — атомарно |
| Атомарность JSON | Всегда `NamedTemporaryFile` + `os.replace` — не `open().write()` напрямую |
| chat_id в config | `int(val)` — если строка в JSON, конвертировать, иначе менеджер выпадет из маршрутизации |

---

## Актуальный статус задач (проверено 2026-04-19, HEAD `25bee91`)

Источник актуализации: текущий код, последние коммиты до `25bee91`, `CLAUDE.md`,
`SESSION_CONTEXT.md`, session-файлы, `BUGFIX_PLAN_2026-04-15.md`, папки `audit/` и `аудит/`.

### ✅ Уже закрыто — не чинить повторно
| Тема | Статус |
|------|--------|
| `InlineKeyboardMarkup` / `UnboundLocalError` в `cb_data()` | Закрыто коммитом `65d618d` от 2026-04-15: локальный импорт удалён |
| `WHATSAPP_ENABLED=0` | На текущей машине `.env`: `WHATSAPP_ENABLED=1`, `LIVE_SEND_ALLOWED=1` |
| AssemblyAI `speech_models` list → `speech_model` string | Закрыто коммитом `5dc633b` от 2026-04-19; используется `speech_model="best"` |
| `.work` файлы не возвращались в retry | Закрыто коммитом `1907556` |
| `debt` invariant / fallback на `closing` в DSO | Закрыто коммитом `cb1d6c3` |
| `chat_id` строкой в config | Закрыто коммитом `a831ee5`: `int(val)` |
| unknown user ACL | Закрыто коммитами `8bed42f` + `ef82c7b`: `_acl_gate()` |
| `revenue` vs `total` дедуп в RFM/Concentration | Закрыто коммитом `0962b95` |
| `debt_stop_control` фиксированный `.tmp` | Закрыто коммитом `d2a6908` |
| Отчёты 14.04 HIGH/MEDIUM | Закрыто, `tests/test_audit_reports_20260414.py`: 8/8 OK |
| Issue-4 условная отгрузка | Закрыто: 4 admin-кнопки в `escalate_unanswered()`, 4 кнопки в `monitor_exceptions()`, callback routing и `_handle_conditional_clearance()` |
| Weekly `callback_data` overflow | Закрыто коммитом `0464c2f`: token 8 hex вместо полного `client_name` |
| AI retry/backoff | Закрыто коммитом `9e97a2b`: `tenacity`, 3 попытки, exponential backoff |
| IMAP `_save_bytes` non-atomic | Закрыто коммитом `8aede7e`: `NamedTemporaryFile` + `os.replace` |
| `test_phase2_safe_send.py` dry-run/preview конфликт | Закрыто коммитом `76de59c`: `WA=1` → `--preview`, `WA=0` → `--dry-run` |
| Test-log pollution T4 | Закрыто коммитом `51f7892`: `_log` мокируется |
| DEAD imports/callbacks collector | Закрыто коммитами `28b73d8`, `be1f7a8` |
| `collector_state.json` race | Частично закрыто коммитом `25bee91`: `portalocker` lock для `collector_state.json` |
| Товары/metadata 1С попадали в CRM | Закрыто связкой `a3331a8` + `ea2e567`: CRM читает `sales JSON clients[]`, а `sales_parser.py` строго разделяет client/product колонки без fallback |
| Автомониторинг логов | Реализовано: `bot/log_monitor.py` + scheduler job `log_monitor` каждые 2 часа |

### Проверенные коммиты, влияющие на статус
| Коммит | Дата | Что учитывать |
|--------|------|---------------|
| `25bee91` | 2026-04-19 | `collector/collections_db.py`: `portalocker` lock для `collector_state.json` |
| `6582ed9` | 2026-04-19 | `CLAUDE.md`: production Data Flow = `bot/send_reports.py::pipeline_task()`, `run_pipeline_all_mp.py` ручной |
| `be1f7a8` | 2026-04-19 | `collector/manager_dialog.py`: удалены DEAD-3/DEAD-4 |
| `306385a` | 2026-04-19 | `_save_contact()` сохраняет контакты в CRM через `set_client_details()` |
| `c30a2bb` | 2026-04-19 | `load_contacts_compat()` теперь только CRM, legacy merge удалён |
| `51f7892` | 2026-04-19 | T4 в `tests/test_project.py` больше не загрязняет production-log |
| `8aede7e` | 2026-04-19 | `imap_fetcher.py`: atomic write для `_save_bytes()` |
| `9e97a2b` | 2026-04-19 | `ai_analyzer.py`: retry 3x backoff через `tenacity` |
| `0464c2f` | 2026-04-19 | `bot/send_reports.py`: weekly callback token вместо `client_name` |
| `76de59c` | 2026-04-19 | `tests/test_phase2_safe_send.py`: актуальный контракт Phase 3 |
| `87f07cc` | 2026-04-19 | `ai_analyzer.py`: добавлен `timeout=120` |
| `5dc633b` | 2026-04-19 | `collector/whatsapp_poller.py`: AssemblyAI исправлен на `speech_model="best"` |
| `ea2e567` | 2026-04-18 | `sales_parser.py`: убран fallback `client_j→product_j`, клиент и товар строго разные колонки |
| `a3331a8` | 2026-04-18 | `bot/crm_clients.py`: очищены товарные/metadata записи из CRM; позже текстовый product regex убран, потому что разделение перенесено в parser |
| `65d618d` | 2026-04-15 | Удалён тестовый телефон; закрыт `InlineKeyboardMarkup` UnboundLocalError; создан `BUGFIX_PLAN` |
| `3336383` | 2026-04-14 | `approval_flow.py`: retry Telegram send 3x; это не retry AI API |
| `2c1f832` | 2026-04-14 | `bot/send_reports.py`: защита от утечки данных в debt/sales ACL |
| `ef82c7b` | 2026-04-14 | `bot/send_reports.py`: unknown users полностью блокируются через `_acl_gate()` |
| `f1eca89` | 2026-04-14 | `bot/send_reports.py`: `_TzFormatter` вместо глобального monkey-patch |
| `569fe52` | 2026-04-14 | `bot/send_reports.py`: убран двойной `query.answer()` |
| `a831ee5` | 2026-04-13 | `config.py`: `chat_id` приводится через `int(val)` |
| `d2a6908` | 2026-04-13 | `bot/debt_stop_control.py`: атомарная запись через `NamedTemporaryFile` |
| `0962b95` | 2026-04-13 | RFM/Concentration: дедуп по `total`, не `revenue` |
| `bdaf933` | 2026-04-13 | `collector/manager_dialog.py`: удалён DEAD-1 после `return False` |
| `cb1d6c3` | 2026-04-13 | `dso_aging_report.py`: удалён fallback на `closing`, enforced `debt` |
| `1907556` | 2026-04-13 | `run_pipeline_all_mp.py`: `.work` возвращается в `.xlsx` при ошибке |
| `6449778` | 2026-04-13 | `collector/voice_calls.py`: logger init до try/except |
| `5d9a3fa` | 2026-04-11 | `collections_engine.py`: двойной live-send guard `WHATSAPP_ENABLED` + `LIVE_SEND_ALLOWED` |
| `18a8ea8` | 2026-04-11 | Emergency fix после WhatsApp incident; collector send-path менять только через dry-run/approval |

### 🔴 P1 — Реальных критичных production-багов сейчас не подтверждено

Не открывать P1 без свежего лога/репродукции. Старые P1 из `BUGFIX_PLAN_2026-04-15.md`
устарели после коммитов 15-19.04.

### 🟠 P2 — Актуальные высокие риски
| # | Файл/контур | Проблема | Безопасный подход |
|---|------|---------|-------------------|
| 1 | CRM/contact source | После `c30a2bb` collector берёт контакты только из CRM; нужно проверить, не потеряны ли телефоны из legacy `debtors_contacts.json` | Read-only сверка matched/unmatched debt candidates vs `config/clients.json`, без автозаписи |
| 2 | JSON-state кроме `collector_state.json` | Lock добавлен только для `collector_state.json`; остальные state-файлы требуют отдельной проверки writers | Таблица owners/read-modify-write, затем точечный lock только там, где есть concurrent write |
| 3 | Live WhatsApp | `.env` может иметь `WHATSAPP_ENABLED=1` и `LIVE_SEND_ALLOWED=1`; live path требует runbook | Backup → dry-run → preview → manager approval → admin approval → `--send-approved` |

### CRM и товары

Товарные строки сейчас не должны попадать в CRM при штатном pipeline:

- `sales_parser.py` строго берёт клиента из client/contragent колонки и товар из product/nomenclature колонки;
- fallback `client_j→product_j` удалён коммитом `ea2e567`;
- `bot/crm_clients.py::_load_latest_sales_clients()` берёт только `data["clients"][].client`;
- фактическая проверка 2026-04-19: `config/clients.json` — 475 клиентов, 0 product/metadata hits; последние 20 `sales_*.json` — 0 product/metadata hits в `clients[]`.

Нюанс: отдельного текстового `_PRODUCT_NAME_RE` в текущем `crm_clients.py` больше нет. Защита держится на корректном `sales_parser.py`; если вручную подложить битый sales JSON, где товар уже записан как `clients[].client`, CRM его может принять, кроме metadata строк.

### Автомониторинг логов

Бот сам проверяет новые строки в `logs/*.log` каждые 2 часа:

- scheduler job: `log_monitor` в `bot/send_reports.py`;
- реализация: `bot/log_monitor.py`;
- state: `logs/log_monitor_state.json`;
- краткий журнал: `logs/log_monitor_summary.log`;
- при новых `ERROR`/`CRITICAL`/`Traceback`/`PermissionError` отправляет Telegram-уведомление admin;
- первый запуск baselines текущие концы логов, чтобы не слать старые ошибки;
- `[TEST]` строки игнорируются;
- summary-файл самого монитора не сканируется, чтобы не повторять свои же alerts.

Для следующей диагностики сначала читать:

```powershell
Get-Content logs\log_monitor_state.json -Raw
Get-Content logs\log_monitor_summary.log -Tail 50
```

### 🟡 P3 — Технический долг / устойчивость
| # | Файл/контур | Проблема |
|---|------|---------|
| 4 | `bot/send_reports.py` | Монолит ~7700 строк: scheduler, callbacks, ACL, reports, collector; менять только точечно |
| 5 | `config/clients.json` | Данные контактов требуют отдельной сверки после перехода на CRM-only source |
| 6 | `pdfkit` / docs | PDF-зависимость есть в requirements, активный PDF-контур не найден; не добавлять PDF |

### 🟢 OPEN — Аудит отчётов 14.04 (не баги, открытые вопросы)
| # | Модуль | Описание |
|---|--------|---------|
| C1 | `inventory_turnover_report.py` | «Мёртвый запас» = отсутствие в JSON, а не N дней без продаж — алгоритм не реализован |
| ARCH | CSS/шаблоны | Дублирование CSS между шаблонами vs `base.html` |

### ⚠️ Collector / WhatsApp — особая зона риска

В `audit/` зафиксирован production incident 2026-04-10 по несанкционированным WhatsApp-отправкам.
Любые изменения в collector send-path, approval flow, contacts, state или live flags:

1. Сначала `git status --short`
2. Backup state-файлов (`logs/*.json`, `config/debtors_contacts.json`, `config/clients.json`)
3. Только `--dry-run` → `--preview`
4. Live только через `--send-approved --batch-id <id>` после manager + admin approval
5. Никогда не использовать legacy `--send`

---

## Статус аудитов

### Аудит 13.04 (collector/pipeline/bot)
- **Критические**: исправлены в сессии 13.04 и последующих коммитах.
- **Высокие/устойчивость**: retry AI, IMAP atomic write, test-log pollution, weekly callback overflow и часть locks закрыты коммитами 19.04.
- **Осталось**: проверка locks для state-файлов кроме `collector_state.json`, CRM/contact data audit.

### Аудит 14.04 (отчёты Excel→HTML→бот)
- **HIGH** (6/6): все исправлены ✅
- **MEDIUM** (8/8): все исправлены ✅
- **Tests** `test_audit_reports_20260414.py`: 8/8 зелёных ✅
- **Открыто**: C1 (алгоритм мёртвого запаса), ARCH CSS

---

## Collector — ключевые правила

```
Уровни: 0-9д→L0(skip), 10-14→L1, 15-19→L2, 20-24→L3, 25-29→L4, 30+→L5
Отправка: только 09:00-18:00 Asia/Almaty, НЕ выходные
Звонки: только 09:00-17:00, level>=4, do_not_call=false
Макс: 1 сообщение на клиента в день

CLI (правильный порядок):
  python -m collector.collections_engine --dry-run     ← всегда первым
  python -m collector.collections_engine --preview
  # После одобрения менеджером + admin:
  python -m collector.collections_engine --send-approved --batch-id <id> --client "<name>"
  # --send ОТКЛЮЧЁН
```

**FSM диалога менеджера:**
`AWAITING_CONFIRM` → `AWAITING_DATA` / `AWAITING_REJECTION_REASON` → `DEADLINE_SET` / `CONFIRMED` / `DONE`

**State-файлы:**
- `logs/collector_state.json` — история контактов, обещания, флаги
- `logs/collector_dialogs.json` — активные диалоги менеджеров
- `config/debtors_contacts.json` — справочник контактов клиентов

---

## Роли и доступ

| Роль | Кто | Chat ID | Доступ |
|------|-----|---------|--------|
| admin | Вадим | 7422963573 | Всё, все менеджеры |
| subadmin | Алена | 188939016 | Магира + Оксана + team mini-rating |
| manager | Оксана, Магира, Ергали, Алена | — | Только свои данные |

**ACL-защита:** `_acl_gate()` блокирует unknown полностью. Сводные файлы (manager="—") → только admin.

---

## Команды

```bash
# Активировать venv
.venv\Scripts\activate

# Основной бот
python bot/send_reports.py

# Pipeline вручную
python run_pipeline_all_mp.py

# Тесты
python -X utf8 tests/test_project.py
set WHATSAPP_ENABLED=0
set LIVE_SEND_ALLOWED=0
python -X utf8 tests/test_collector.py
python -X utf8 tests/test_phase2_safe_send.py
python -X utf8 tests/test_audit_reports_20260414.py
python -X utf8 tests/test_log_monitor.py

# Синтаксис-проверка (обязательно после каждого изменения)
python -m py_compile <файл>
```

---

## Алгоритм работы при получении задачи

1. Читай `CLAUDE.md` — полный контекст
2. Читай `repo_map.json` — карта файлов (не сканируй весь репо)
3. Определи layer затронутого файла — не трогай соседние layers
4. Проверь `BUGFIX_PLAN_2026-04-15.md` — задача уже может быть описана
5. Проверь `аудит/` — может быть уже проанализирован root cause
6. Применяй SAFE SURGERY — один патч, compile, version, test, commit

---

## Типичные диагностики

| Симптом | Где смотреть |
|---------|-------------|
| Бот не отвечает | `logs/send_reports_*.log`, `.env` → TG_BOT_TOKEN |
| Коллектор не шлёт | `WHATSAPP_ENABLED` в `.env`, `logs/collector_*.log` |
| AI не генерирует | `logs/ai_analyzer_*.log`, DEEPSEEK_API_KEY в `.env` |
| Pipeline завис | `reports/queue/*.work` — застрявшие файлы (переименовать в `.xlsx`) |
| Аналитика устарела | `reports/json/` — дата файлов → запустить pipeline |
| Голосовые не работают | AssemblyAI quota/API/network, активный код использует `speech_model="best"` |
| Кнопки в боте падают | Проверить конкретный callback и лимит 64 байта; weekly overflow уже закрыт token-ами |
| Менеджер не получает | `chat_id` в `config/managers.json` → должен быть int, не строка |

---

## Форматы коммитов (conventional commits)

```
fix(<module>): <что исправлено>
feat(<module>): <что добавлено>
refactor(<module>): <что улучшено>
test(<module>): <что покрыто>
docs: <что задокументировано>
perf(<module>): <оптимизация>
```

Версия: `__VERSION__ = "X.Y.Z"` → +0.0.1 при каждом изменении файла.

---

## Брендбук CSS (для HTML-отчётов)

```css
--brand: #1a3a5c  /* navy */
--accent: #0070c0 /* blue */
--good:   #107c41
--bad:    #c00000
--warn:   #e09000
--bg:     #f0f4f8
```

Jinja2 → `templates/base.html`. Layer 5 (f-strings) → фигурные скобки экранировать `{{`/`}}`.
Footer: `"Сформировано: DD.MM.YYYY HH:MM (Asia/Almaty) | Версия: …"`

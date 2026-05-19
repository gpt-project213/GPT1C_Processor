# PROJECT_ENCYCLOPEDIA

Актуальная единая база знаний по `GPT1C_Processor_analitica`.

Статус: актуально на `2026-05-19`  
Текущая базовая ветка: `master`  
Последний коммит: `0fbbaba`

---

## 1. Назначение

Этот файл заменяет разрозненные описания проекта, audit-сводки, handoff-справки, UX-заметки и исторические проектные `.md`, которые раньше были размазаны по корню и `audit/`.

Что оставлено отдельно:
- `AGENTS.md` — рабочие правила для агента и инженерный protocol.
- `SESSION_CONTEXT.md` — живой журнал последних изменений и handoff.
- prompt-активы, которые являются не справкой, а рабочими шаблонами:
  - `openclaw/SOUL.md`
  - `openclaw/skills/debt_collector.md`
  - `autoagent/.ai_reviews/REVIEW_TEMPLATE.md`

---

## 2. Что это за проект

`GPT1C_Processor / AI 1C PRO` — production-проект компании Минбаракат для обработки Excel-выгрузок из 1С, построения HTML/JSON-отчётов, AI-комментариев и доставки их через Telegram-бота.

Внутри проекта есть несколько больших контуров:
- pipeline обработки входящих Excel;
- генерация отчётов и аналитики;
- Telegram-бот с расписанием и ролями;
- CRM-контур уточнения телефонов и владения клиентами;
- AI Debt Collector для WhatsApp/Telegram-коммуникации по дебиторке;
- stop/payment-контур со стороны Саиды и директора;
- локальный autoagent/orchestrator слой для агентной разработки.

Стек:
- Python 3.11+
- `python-telegram-bot`
- APScheduler
- DeepSeek / OpenAI
- Green API
- Retell AI
- `.venv/`

TZ-политика:
- всегда через `ZoneInfo(os.getenv("TZ", "Asia/Almaty"))`
- все бизнес-времена и логи интерпретируются как `Asia/Almaty`

---

## 3. Главные точки входа

Основные команды:

```powershell
.venv\Scripts\activate
python bot/send_reports.py
python run_pipeline_all_mp.py
python imap_fetcher.py --once
python -m collector.collections_engine --dry-run
python -m collector.collections_engine --preview
python -m collector.collections_engine --send-approved --batch-id <id> --client "<name>"
python -X utf8 tests/test_project.py
python -X utf8 tests/test_collector.py
```

Прод-вход:
- `bot/send_reports.py`

Operational note:
- На Windows запуск через `.venv\Scripts\python.exe` может визуально давать два `python.exe` в Process List.
- Проверенный кейс на `2026-05-09`: лёгкий родительский процесс `.venv\Scripts\python.exe` стартует из `start_bot_watchdog.bat`, а рабочий интерпретатор живёт как дочерний `C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe`.
- Это само по себе не означает второй экземпляр scheduler-а. Признак реального дубля нужно искать не по двум `python.exe`, а по двум независимым `bot_starting`, конфликту `bot.pid`, дублирующим APScheduler job runs или отдельным родителям процесса.
- На той же проверке `ParentProcessId` у системного `Python311\python.exe` указывал на `.venv`-процесс watchdog, а `logs/bot.pid` принадлежал дочернему рабочему интерпретатору. Это соответствует launcher/redirector-поведению Windows venv, а не двум отдельным запускам бота.

Ручной CLI/pipeline:
- `run_pipeline_all_mp.py`
- `imap_fetcher.py`

Collector:
- только `--preview` и `--send-approved`
- legacy `--send` считать отключённым и неиспользуемым

---

## 3A. CRM Ownership And Same-Client Flow

Состояние на `2026-05-19`:

- Пакет `0fbbaba` закрыл ложные `payment_hold` / Минай / Саиду:
  - auto-create hold убран из показа кнопок
  - введён `silence_candidates.json`
  - реальный hold теперь отделён от чернового candidate
  - downstream завязан на `source` + `sent_to_saida_at`

- Следующий слой CRM-логики:
  - если в CRM появляется новый ключ клиента, похожий на уже закреплённую карточку,
    бот не должен сразу слать общий `Чей клиент?`
  - сначала он должен спросить владельца похожей карточки:
    - `Это один и тот же клиент?`
    - `Да, тот же клиент`
    - `Нет, это другой`

Правильная модель:

- `Да`:
  - новый ключ связывается с существующей карточкой как alias
  - ownership наследуется от уже известного клиента
  - при необходимости запускается обычная CRM-цепочка дозаполнения:
    - имя
    - телефон
    - адрес

- `Нет`:
  - пара явно помечается как разные клиенты
  - дальше уже идёт обычный `claim_broadcast` / идентификация

Принцип:

- не автосклеивать клиентов без подтверждения менеджера
- не заставлять менеджера повторно полностью идентифицировать уже известного клиента
- `claim_broadcast` должен быть fallback, а не первым действием

---

## 4. Архитектура проекта

Проект живёт в 9 слоях:

| Layer | Роль |
|---|---|
| 0 | low-level utils без project imports |
| 1 | `config.py`, пути, TZ, logging, роли |
| 2 | Excel/Jinja helpers |
| 3 | parsers `xlsx -> dict/DataFrame` |
| 4 | main reports `xlsx -> HTML + JSON` |
| 5 | analytics reports `JSON -> HTML` |
| 6 | `ai_analyzer.py` |
| 7 | orchestrators `imap_fetcher.py`, `run_pipeline*` |
| 8 | Telegram bot и scheduler |
| 9 | collector / stop / CRM-связки |

Ключевой монолит:
- `bot/send_reports.py`

Изолированные зоны:
- Layer 5 analytics
- отдельные parser/report-модули
- части `collector/`

---

## 5. Data Flow

Основной производственный путь:

```text
Email (IMAP)
  -> reports/queue/
  -> pipeline_task() в bot/send_reports.py
  -> report/parser modules
  -> reports/html/ + reports/json/
  -> analytics layer
  -> reports/analytics/
  -> ai_analyzer.py
  -> reports/ai/
  -> Telegram delivery
```

Маршрутизация pipeline:
- `DEBT` -> `debt_auto_report.py`
- `SALES` -> `sales_report.py` + `sales_parser.py`
- `GROSS` -> `gross_report.py`
- `INVENT` -> `inventory.py`
- `EXPENSE` -> `expenses_report.py`

Важно:
- ботовый pipeline — это `pipeline_task()` в `bot/send_reports.py`
- `run_pipeline_all_mp.py` — ручной CLI-инструмент, не сам бот
- при ручном `SALES` HTML строится, а sales JSON может не создаваться — это допустимо для ручного режима

Файловый lifecycle очереди:

```text
reports/queue/file.xlsx
-> claim
-> reports/queue/file.xlsx.work
-> processing
-> reports/excel/processed/file.xlsx
```

Нельзя:
- удалять queue-файлы вручную из бизнес-логики
- обходить `_move_to_processed()`
- ломать атомарность claim/release

---

## 6. Роли и доступы

| Role | Кто | Доступ |
|---|---|---|
| admin | Вадим (`7422963573`) | всё |
| subadmin | Алена (`188939016`) | Магира + Оксана + team mini-rating |
| manager | Оксана, Магира, Ергали, Алена | только свои данные |

Критично:
- Алена dual-role: и `subadmin`, и `manager`
- эту двойную семантику ломать нельзя

---

## 7. Collector и связанные контуры

### 7.1. Collector

Collector — отдельный AI-контур для касаний по дебиторке через WhatsApp/Telegram.

Основные части:
- `collector/collections_engine.py`
- `collector/debt_monitor.py`
- `collector/approval_flow.py`
- `collector/client_dialog.py`
- `collector/collections_db.py`
- `collector/audit_log.py`
- `collector/logging_utils.py`

Debt levels:
- `0-9` дней -> skip
- `10-14` -> soft
- `15-19` -> medium
- `20-24` -> assertive
- `25-29` -> strict + call
- `30+` -> hard + escalation

Основные state-файлы:
- `logs/collector_state.json`
- `logs/collector_dialogs.json`
- `logs/wa_approval_batches.json`
- `logs/wa_agreed_promises.json`
- `logs/collector_audit.jsonl`

Ограничения:
- не больше 1 сообщения клиенту в день
- send window только в рабочее разрешённое время
- voice calls только по отдельным правилам

### 7.2. Approval flow — текущее состояние

К маю 2026 approval flow доведён до управляемого состояния.

Что теперь считается нормой:
- просто `убрать` без причины больше не считается хорошей моделью
- `Оплатил` и `Договорились` — это отдельные бизнес-сценарии
- `Договорились` требует деталей; если нажал кнопку и не написал — бот ждёт
- обещание имеет дедлайн
- сорванное обещание автоматически возвращает клиента в WA
- повторная бесконечная отмена менеджером не считается корректной логикой
- директор может отдельно проверять договорённости менеджеров

Ключевые изменения, которые уже находятся в `master`:
- B-lite review договорённостей директором
- promise quality analytics
- Saida backlog analytics
- tight send window / cutoff / честный expires_at
- немедленная эскалация при позднем батче
- расписание без конфликтов stop-list vs collector

Admin summary — актуально на 2026-05-14:
- `_format_admin_summary_text` (`approval_flow.py`) — единственная живая функция admin summary
- `_format_admin_detail_text` удалена как dead-code (коммит `0f412ae`)
- timeout + `waiting_for_agreed` → `"⏳ начал — не написал детали по «X»"` (не «🔇 не ответил»)
- timeout + `waiting_for_proof` → `"⏳ начал — не прислал документ по «X»"` (аналогично)

Client dialog — актуально на 2026-05-15:
- `dialog_blocks_new_outreach(dialog)` — единый предикат блокировки: возвращает `(bool, reason)`.
  Два случая разблокировки: `awaiting_payment_proof` + 72h → `stale_payment_proof`;
  `active` + 0 ответов клиента + 24h+ → `stale_silent_active`.
- До этого клиенты типа Шахин/Тян/Еркебулан залипали в `active, exchange_count=0` бесконечно и
  никогда не попадали в следующую ежедневную рассылку.
- `start_client_dialog()` при supersede stale-диалога пробрасывает `phone_silent_cycles`.
- Env: `COLLECTOR_SILENT_ACTIVE_RESEND_HOURS` (default 24).
- `collections_engine.py` (preview + send-approved) полностью переведён на `dialog_blocks_new_outreach()`; единственный источник истины для TTL-логики диалогов.

Penalty reset — актуально на 2026-05-15:
- `build_reset_state()` / `reset_penalty_state()` в `collector/approval_penalty.py`.
- `reset_penalty_state()`: делает backup `logs/approval_penalty_state.json`, пишет новый state
  с `wa_reset_floor` и `crm_reset_floor` — батчи/CRM-записи старше floor не получают штраф
  при следующем backfill.
- Admin UI: кнопка `♻️ Сброс штрафов` в экране 🤖 Коллектор (confirm-step, admin only).
  После подтверждения показывает: месяц, WA floor, CRM floor, имя backup-файла.

Promise-tracking — три системы (актуально на 2026-05-15 вечер, F-B1):
1. **Manager-promise** (через «Договорились» в admin batch) → `logs/wa_agreed_promises.json` →
   handler `check_broken_agreed_deadlines` (`collector/approval_flow.py:2571`) ежедневно в 10:30
   (job `wa_agreed_deadline_check` в `bot/send_reports.py:10962`).
2. **Legacy state-promise** → `logs/collector_state.json` (`promise_date` поле) →
   handler `check_promises` (`collector/collections_engine.py:1463`) ежедневно в 10:00.
3. **Client-promise** (от ответа клиента в WA-диалоге) → `logs/collector_client_dialogs.json`
   (`promise_date` внутри dialog) → синхронизируется в **wa_agreed_promises.json** через
   `record_client_promise()` (`collector/approval_flow.py`) — handler 10:30 подхватывает.

Client-dialog promise-sync (`collector/client_dialog.py:1.1.11`):
- Helper `_sync_promise_to_agreed(dialog, promise_date, details)` — единая точка вызова
  `record_client_promise()` с критическим `_TEST_MODE` guard.
- Вызывается из двух intent-веток в `handle_incoming`: `intent="promise"` и `intent="promise_schedule"`.
- `record_client_promise()`: не перезаписывает активный manager-promise (manager главнее),
  обновляет существующий client-promise при уточнении, reopens после broken/fulfilled.
- `source="client_dialog"` + `batch_id="client_dialog"` для forensic-различия.

Защита от контаминации боевого state из тестов:
- `_TEST_MODE` guard в `_sync_promise_to_agreed` — при `COLLECTOR_TEST_MODE=1` запись пропускается.
- Прецедент 2026-05-15: без guard юнит-тесты записали `"Кайрбек"` с `deadline=2026-04-22` в
  боевой `wa_agreed_promises.json`. Запись была обнаружена и немедленно удалена.
- Регрессия: `test_sync_helper_skipped_in_test_mode` + проверка `_PROMISES_PATH` замокан перед записью.

### 7.3. Stop/payment-контур

Это отдельный контур, не равный collector.

Главный файл:
- `bot/debt_stop_control.py`

State-файлы:
- `logs/saida_payment_holds.json`
- `reports/debt_stop_registry.json`
- `reports/debt_stop_state.json`

Участники:
- менеджер
- Саида
- директор

Смысл:
- не “кому писать про долг”, а “можно ли снимать стоп / считать оплату подтвержденной”

К маю 2026 закрыто важное операционное отверстие:
- если Саида подтверждает полную оплату, стоп снимается автоматически
- менеджер и Саида получают уведомление
- связанное shipment-control решение закрывается без второго ручного шага

#### Silence-отчёт — бизнес-правило и закрытые баги (2026-05-16)

**Бизнес-правило:** Молчание = молчание В ОПЛАТЕ. Только `confirmed_full` от Саиды снимает клиента из silence-отчёта. Частичная оплата, диалог, обещание — не снимают.

Silence-отчёт живёт в `bot/silence_alerts.py`. Три P0-бага закрыты в коммите `f2ab785`.

| Баг | Файл | Суть фикса |
|---|---|---|
| BUG-2 | `bot/debt_stop_control.py` | token-based callback `sp0001` вместо `name[:26]`. Новые: `_saida_payment_token`, `_resolve_saida_payment_token`, `_kb_saida_stop_item`. `send_saida_full_stoplist` переведён на токены. |
| BUG-saida | `bot/silence_alerts.py` | `apply_payment_holds` теперь использует собственный `_load_recent_full_payments()` (TTL 7 дней). payment_hold=True только для `confirmed_full`. Partial/pending/rejected не снимают молчание. |
| BUG-5 | `bot/silence_alerts.py` | Новый файл `logs/debt_age_history.json` с persistence. `apply_residual_debt_age` берёт `min(saved, current)` — reset окна Саиды 15-го числа больше не обнуляет историю. TEST_MODE guard на запись. |

Косметика P2/P3 (одновременно закрыта):
- `"остаток N дн"` → `"долг N дн"`
- `"МОЛЧАНИЕ/ПРОСРОЧКА"` → `"Долг N–M дн"`
- `"ВСЕГО МОЛЧАЩИХ"` → `"ВСЕГО ДОЛГА"`

Тесты: +22 регрессии (9 BUG-2 в `test_collector.py`; 7 SilenceFullPayment + 6 DebtAgeHistory в hermetic).

### 7.4. Саида — текущее состояние процесса

У Саиды есть отдельный backlog-контур:
- `pending_saida`
- SLA-предупреждение
- байпас директору
- аналитика oldest age / count / per-manager split

На `2026-05-06` уже был зафиксирован и частично очищен исторический stale backlog:
- старый хвост был заархивирован
- из рабочей очереди удалены только stale pending старше `48h`
- живые кейсы оставлены

### 7.5. Напоминалка Минай (2026-05-19)

Изолированный WhatsApp-контур для Минай Рашидовой (+7 702 317 7888).

Главный файл:
- `bot/minai_reminders.py`

State-файлы:
- `logs/minai_reminder_state.json` — статусы + флаги диалога
- `logs/minai_custom_reminders.json` — кастомные напоминания от Минай

Настройка:
- `.env`: `MINAI_WA_PHONE=77023177888`
- Активируется автоматически при наличии `MINAI_WA_PHONE`

Встроенное расписание (все в 10:00, "Доброе утро"):

| День | Напоминание | Тип |
|---|---|---|
| 17-е | Завтра платить интернет | предупреждение |
| 18-е | Крайний срок — интернет | 🚨 critical, повторы 13/16/19 |
| 3-е | Через 2 дня — аренда | предупреждение |
| 4-е | Завтра — аренда | предупреждение |
| 5-е | Крайний срок — аренда | 🚨 critical, повторы 13/16/19 |
| 23-е | Через 2 дня — налоги | предупреждение |
| 24-е | Завтра — налоги | предупреждение |
| 25-е | Крайний срок — налоги | 🚨 critical, повторы 13/16/19 |

Механика взаимодействия:
- Кнопки: `✅ Сделала` / `⏰ Позже` / `➕ Добавить`
- Голосовые: AssemblyAI транскрибирует → Минай подтверждает транскрипцию
- Добавление напоминаний: DeepSeek разбирает свободный текст → confirm → сохраняется
- Рабочее / Личное: при добавлении Минай выбирает; рабочее = уведомляет Вадима при провале

Критичные дни (интернет/аренда/налоги):
- 21:00 без подтверждения → финальный алерт Минай («ПОСЛЕДНИЙ ШАНС»)
- 22:00 без подтверждения → Telegram Вадиму: «🚨 ПРОВАЛ»

Защиты от зависаний:
- Диалоговые состояния (`__awaiting_add__`, `__pending_confirm__` и др.) — таймаут 30 мин с уведомлением
- После 21:00 — никаких новых отправок
- Сноуз за пределами дня → `auto_closed`

Фидбек:
- Минай пишет «неудобно» / «хочу изменить» → бот принимает текст/голос → пересылает Вадиму в Telegram
- Catch-all: любое нераспознанное сообщение → подсказка с кнопками

---

## 8. CRM-контур

Главные файлы:
- `bot/crm_clients.py`
- части `bot/send_reports.py`
- `bot/crm_audit_log.py`

State-файлы (все 4 покрыты lock-discipline на 2026-05-14):
- `logs/crm_pending_state.json` — phone-pending очередь
- `logs/crm_claim_pending_state.json` — claim-рассылка
- `logs/crm_duplicate_review_state.json` — dup-review
- `logs/crm_ambiguous_conflicts.json` — ambiguous conflicts

Что стабилизировано к 2026-05-15:
- canonical client key + alias-expansion в load_contacts_compat
- restart-safe `crm_claim` + `_crm_write_ok` gate (F-07)
- `crm_audit.jsonl` audit trail
- portalocker lock-discipline (`_crm_state_lock`) — **hard-fail**: при недоступности lock поднимает `CrmStateLockError`; все 4 save → `bool`; callback-callers делают rollback + user-facing error; scheduler-callers логируют error и продолжают (2026-05-15)
- stale claim tokens: `_crm_claim_is_stale` — без `created_at` или просроченные удаляются при cleanup; callback `crm_claim|...` явно отклоняет expired (снимает кнопку)
- deterministic keep-key в dup-review custom phone: `sorted(client_keys)[0]` вместо `client_keys[0]`
- phone-aware ambiguous signature: `sorted(“client_key#normalized_phone”)` — если phone изменился, signature новый → reopen автоматически через новую pending-запись
- admin CRM backlog screen: кнопка `📋 CRM бэклог` в admin menu (callback `crm_backlog`), formatter `_format_crm_backlog_text` — 4 категории, top-5, age-label, stale-count
- **"Частное лицо" placeholder filter (2026-05-15 вечер)**: keywords `"частное лицо"`, `"физическое лицо"`, `"физлицо"` добавлены в `_VENDOR_NAME_KEYWORDS` (`bot/crm_clients.py:74-94`). Закрывает все 4 CRM-пути через единый `is_service_client_name()`: update_from_reports, get_clients_without_phones, _crm_collect_unowned_claim_clients, get_phone_conflict_groups. Root cause — 1С регулярно выгружает placeholder-имена ("Частное лицо 1" и т.п.), которые попадали в claim broadcast.

Audit-документы:
- `AUDIT_CRM_PRIVATE_PERSON_2026-05-15.md` — Block A, placeholder filter
- `AUDIT_COLLECTOR_RUNTIME_2026-05-15.md` — Block B, фактический runtime после рестарта v9.4.79

Что ещё не закрыто:
- stale dup-review callback: отвечает «запрос устарел», но без full cleanup-строгости как у claim
- три несогласованных хранилища: `clients.json` / `debtors_contacts.json` / batch snapshot
- backfill 6 существующих placeholder-записей ("Частное лицо*") как `is_vendor=True` — optional, не блокирует runtime

---

## 9. Логирование и наблюдаемость

К маю 2026 единое логирование collector/stop/bot фактически закрыто.

Что является нормой:
- `config.setup_logging()`
- без `logging.basicConfig()` на module level
- доменные префиксы в логах
- collector/stop через общий logging helper
- audit trail в JSONL

Основные наблюдательные файлы:
- `logs/send_reports.log`
- `logs/log_monitor_state.json`
- `logs/log_monitor_summary.log`
- `logs/collector_audit.jsonl`
- `logs/crm_audit.jsonl`

При вопросах “жив ли бот” сначала смотреть:
- `logs/log_monitor_state.json`
- `logs/log_monitor_summary.log`

---

## 10. Autoagent и OpenClaw

### 10.1. Autoagent / orchestrator

В проекте существует отдельный локальный orchestration-слой для агентной работы.

Смысл:
- очередь задач
- состояние агента
- локальный запуск Codex/Claude
- фиксация stdout/stderr/results

Это служебный слой разработки, а не боевой контур дебиторки.

Его основные рабочие артефакты:
- `orchestrator.py`
- `autoagent/*`
- `.ai_reviews/REVIEW_TEMPLATE.md`

Исторические длинные описания orchestrator были сведены сюда, чтобы не держать отдельный dated-context файл.

### 10.2. OpenClaw

`openclaw/*` — это не проектная энциклопедия, а prompt-активы/skill-поведение для debt-агента.

Поэтому их смысл кратко:
- `openclaw/SOUL.md` — persona и жёсткие рамки debt-agent поведения
- `openclaw/skills/debt_collector.md` — skill/алгоритм ответа на сообщения должника

Их не стоит смешивать с общей справкой проекта, поэтому они оставлены как отдельные рабочие prompt-файлы.

---

## 11. Тестовая матрица

Актуальный рабочий набор к `2026-05-19`:

| Тест | Результат | Примечание |
|---|---|---|
| `tests/test_project.py` | `110/110` | |
| `tests/test_collector.py` | `657/657` | +4 PendingSaidaStaleExpiry (lazy-expiry pending_saida) |
| `tests/test_collector_regression_hermetic.py` | `43/43` | |
| `tests/test_crm_regression.py` | `44/44` | |
| `tests/test_parsers.py` | `69/69` | |
| `tests/test_audit_reports_20260414.py` | `8/8` | |
| `tests/test_phase2_safe_send.py` | green | |
| `tests/test_log_monitor.py` | `9/9` | |

Все тесты зелёные. Prod state неизменён (SHA-256 верификация 10/10).

---

## 12. Что уже закрыто

Ниже не полный список каждого старого бага, а то, что нужно помнить как текущую правду проекта.

### 12.1. Закрыто и уже в `master`

- major approval-flow redesign
- SLA-контроль Саиды
- auto-clear stop после полного подтверждения оплаты Саидой
- manager promise analytics
- Saida backlog analytics
- unified collector logging
- CRM duplicate/canonicalization fixes
- stale approved batch refresh
- freshness-aware collector behavior
- proof-поток для оплат и suppress-логика
- diff-notice перед admin approve
- dead `_format_admin_detail_text` удалён (2026-05-14)
- timeout-label в admin summary различает waiting_for_agreed / waiting_for_proof (2026-05-14)
- portalocker lock-discipline для 4 CRM state-файлов — best-effort (2026-05-14)
- stale claim tokens cleanup + callback reject (F-16, 2026-05-14)
- deterministic keep-key в dup-review custom phone (F-12, 2026-05-14)
- phone-aware ambiguous signature для reopen (F-13, 2026-05-14)
- admin CRM backlog screen `📋 CRM бэклог` (2026-05-14)
- CRM lock hard-fail: `CrmStateLockError` + save `→ bool` + rollback в callback-callers (2026-05-15)
- stale silent dialogs разблокированы: `dialog_blocks_new_outreach()` + `stale_silent_active` (2026-05-15)
- admin penalty reset: `♻️ Сброс штрафов` + floor-marks + backup (2026-05-15)
- **"Частное лицо" placeholder filter** (Block A, 2026-05-15 вечер): keywords в `_VENDOR_NAME_KEYWORDS`, закрывает claim broadcast по placeholder из 1С
- **F-B1 client-promise sync**: `record_client_promise()` + `_sync_promise_to_agreed()` helper в `client_dialog.py` v1.1.11 — dialog-promise теперь попадает в `wa_agreed_promises.json` и handler 10:30 его подхватывает (2026-05-15 вечер)
- **`_TEST_MODE` guard для promise-sync**: защита боевого `wa_agreed_promises.json` от контаминации юнит-тестами (2026-05-15)
- **BUG-2 — token-based Saida callback**: `_saida_payment_token` / `_resolve_saida_payment_token`, `_kb_saida_stop_item(full_name, state)` — обрезка `name[:26]` удалена, `send_saida_full_stoplist` переведён на токены `sp0001` (2026-05-16)
- **BUG-saida — полная оплата снимает молчание**: `apply_payment_holds` использует `_load_recent_full_payments()` (TTL 7 дней), payment_hold=True только для `confirmed_full` — partial/pending не снимают (2026-05-16)
- **BUG-5 — persistence debt_age_history**: `logs/debt_age_history.json`, `min(saved, current)` для oldest_unpaid_date, TEST_MODE guard — reset окна Саиды 15-го числа больше не обнуляет историю долгов (2026-05-16)
- **Косметика silence-отчёта P2/P3**: формулировки "долг N дн", "Долг N–M дн", "ВСЕГО ДОЛГА" (2026-05-16)
- **`pending_saida` lazy-expiry**: `get_hold_for_client` теперь немедленно закрывает stale-записи (age ≥ `SAIDA_STALE_TTL_HOURS=12h`) — убирает race-window collector vs scheduler (2026-05-18)
- **`expire_stale_holds_on_startup()`**: при каждом старте бота чистит все просроченные holds — накопленные пока бот не работал (2026-05-18)
- **`strip_manager_prefix()`**: однобуквенный префикс менеджера ("О Гриль Косши" → "Гриль Косши") убран из всех внешних сообщений — WhatsApp-рассылка, Саида, Минай (2026-05-18)
- **Уведомление Минай при новом hold**: `create_manager_payment_request()` отправляет WhatsApp Минай (+7 702 317 7888 / `MINAI_WA_PHONE`) с просьбой предоставить выписку Саиде (2026-05-18)
- **`bot/minai_reminders.py`** — WhatsApp-напоминалка для Минай (2026-05-19): расписание интернет/аренда/налоги; кнопки ✅/⏰/➕; DeepSeek для кастомных напоминаний; голосовые через AssemblyAI; авто-закрытие в 21:00; финальный алерт + Telegram Вадиму в 22:00 для критичных невыполненных; рабочее vs личное; фидбек → Вадиму; catch-all для любых сообщений; приветствие при первом запуске

### 12.2. Закрыто частично / не считать идеальным

- CRM ownership ambiguity — значительно улучшено (signature, deterministic keep), но не исчерпано
- stale dup-review callback — базовый ответ есть, full cleanup как у claim ещё не сделан
- часть legacy/architectural duplication
- часть broad `except Exception` по проекту

---

## 13. Что остаётся открытым

Это уже не список старых аудитов, а реально полезный current backlog.

### Product / business
- partial payment логика остаётся не полностью автоматизированной
- качество менеджерских обещаний можно дальше усиливать политиками и репортингом

### CRM
- ambiguous multi-manager ownership: значительно улучшено (signature, deterministic keep), но не исчерпано
- три несогласованных хранилища: `clients.json` / `debtors_contacts.json` / batch snapshot

### Collector
- `debt_age_history.json` bootstrap из архива — опционально; без него бот накопит историю за 1-2 месяца сам
- `COLLECTOR_SILENT_ACTIVE_RESEND_HOURS` (default 24h) — проверить порог в бою после рестарта

### Architecture
- `txt_to_html` duplication
- inline HTML в `expenses_parser.py`
- ряд старых архитектурных дублей низкой срочности

### Operations
- дисциплина Саиды всё ещё управленческая проблема, даже при наличии SLA-аналитики
- stale backlog требует периодической эксплуатации, а не только кода

---

## 14. История аудитов — в сжатом виде

В проекте было много документов с разными срезами. Их смысл теперь сводится к таким блокам:

### 14.1. Базовые аудиты марта-апреля

Дали:
- стартовую карту багов

---

## 15. Approval flow — known issues и диагностика (актуально на 2026-05-07)

### 15.1. Как устроен approval flow

```
17:00 batch created → превью менеджерам
18:10 (если молчат) → эскалация директору (admin_message_id записывается в батч)
19:30 expires_at — после этого отправка заблокирована
Директор: Утвердить (wa_appr_adm_ok) → Отправить сейчас (wa_appr_adm_send)
```

Ключевые файлы:
- `collector/approval_flow.py` — весь approval state machine + Telegram callbacks
- `collector/collections_engine.py` — `send_approved_batch()`, `_live_send_allowed()`
- `logs/wa_approval_batches.json` — персистентное состояние батчей
- `logs/collector_audit.jsonl` — audit trail: `batch_created`, `batch_approved`, `partially_sent`

### 15.2. Root causes "кнопка не работает" — закрытые (07.05.2026)

**1. `configure_runtime_logging()` на import-time** (фикс `44df2a4`)

`approval_flow` делает `lazy import collections_engine` внутри callback.
До фикса импорт модуля вызывал `configure_runtime_logging()` → закрывал root handlers бота →
`ValueError: I/O operation on closed file` на любом `logger.*` → callback падал, UI не обновлялся.

Признак: `wa_appr callback error: I/O operation on closed file` в логах, повторяется при каждом нажатии.

Фикс: `_configure_cli_logging()` вызывается только из `main()`. Импорт как библиотека — нейтральный.

**2. `asyncio.CancelledError` не ловился в preview_batch_changes** (фикс `cd87c47`)

`preview_batch_changes` запускается через `asyncio.to_thread(timeout=10)`.
`CancelledError` — `BaseException`, не `Exception` → не ловился → `_tg_edit` с кнопкой "Отправить сейчас" пропускался.

Признак: два `batch_approved` в `collector_audit.jsonl` подряд, нет `TG edit ok` после первого.

Фикс: `except asyncio.CancelledError as _ce` → UI обновляется, потом `raise _ce`.

**3. Нет `q.answer()` перед тяжёлыми callbacks** (фикс `cd87c47`)

Без `q.answer()` Telegram держит spinner 30 сек → "Query is too old" визуально.

Фикс: `q.answer()` добавлен в `send_reports.py` перед `_wa_appr_cb` для всех `wa_appr_*`.

### 15.3. Диагностика: что смотреть когда Send не ушёл

1. `logs/collector_audit.jsonl` — есть `batch_approved`? есть `partially_sent`/`dialog_started`?
2. `logs/wa_approval_batches.json`:
   - `send_started_at` — если есть, `send_approved_batch` вызвался
   - `send_lock_release_reason` — причина блокировки (`live_send_not_allowed` = время/флаги)
   - `expires_at` — не просрочен ли
3. `send_reports.log` — `wa_appr callback error`, `I/O operation on closed file`, `SEND-APPROVED BLOCKED`

### 15.4. Entry-логи в approval flow (актуальны с db058dd)

```
[batch_id] admin approve button pressed by chat_id=...
[batch_id] admin approve: preview_batch_changes start/finish
[batch_id] Администратор УТВЕРДИЛ отправку: N клиентов
[batch_id] admin send button pressed ... status=... admin_status=...
[batch_id] Администратор запустил отправку из Telegram: N результатов
```

### 15.5. Time window

- `batch.expires_at` = 19:30 — source of truth для `wa_appr_adm_send`
- `is_allowed_time()` / `COLLECTOR_HOUR_END=21` — только для CLI `--send-approved`
- После 19:30: кнопка показывает `⛔ Окно отправки закрыто. Нужен новый батч.`
- архитектурные проблемные зоны
- TZ / pipeline / ACL / report-fixes

Их practical outcome уже отражён в коде, `AGENTS.md` и текущем статусе выше.

### 14.2. Collector / WhatsApp / approval аудиты апреля

Дали:
- controlled live protocol
- shortlist logic
- classification corrections
- UX approval flow
- phase 2-4 collector hardening

Их результат теперь уже находится в `master`.

### 14.3. CRM + logging аудиты конца апреля

Дали:
- canonical duplicate logic
- restart-safe claim state
- CRM audit log
- unified logging core

Это тоже уже вошло в основную ветку.

### 14.4. Draft-аудиты 21-22 апреля

Часть документов тех дат были именно draft/evidence-артефактами:
- не все пункты в них должны были считаться живыми на текущем HEAD
- они использовались как исторические доказательства и рабочие заметки
- после консолидации держать их отдельной пачкой больше не требуется

---

## 15. Политика документации после консолидации

С этого момента целевая модель такая:

### `AGENTS.md`
- инженерные правила
- безопасный workflow
- тестовая матрица
- операционные ограничения

### `PROJECT_ENCYCLOPEDIA.md`
- единое описание проекта
- архитектура
- подсистемы
- актуальный статус
- consolidated knowledge из audit/history/docs

### `SESSION_CONTEXT.md`
- только живой handoff
- последние изменения, проверки, грязные файлы, next steps

Все старые dated audit/doc `.md`, если их смысл уже перенесён сюда, считаются архивно-избыточными и могут жить только в git history.

---

## 16. Что удалено из активной документации

После консолидации больше не нужны как отдельные активные источники истины:
- старые audit-сводки в `audit/`
- старые manager shortlist docs
- dated orchestrator context
- дубль `CLAUDE.md`
- старый `gpt1c.md`

Если потребуется восстановить детали:
- брать из git history по дате или коммиту
- не возвращать обратно пачку параллельных `.md` в рабочую базу знаний

---

## 17. Внешние источники на диске C

Дополнительно вне репозитория на `C:` были найдены project-related артефакты:

### 17.1. Codex memory

- `C:\Users\user\.codex\memories\gpt1c-python-audit-context.md`

Роль:
- краткая persistent memory для следующей сессии Codex;
- хранит последние коммиты, проверки, dirty files, handoff и operational cleanup.

### 17.2. Claude project memory

Найдены project-specific memory-файлы:
- `C:\Users\user\.claude\projects\e--GPT1C-Processor-analitica\memory\project_overview.md`
- `C:\Users\user\.claude\projects\e--GPT1C-Processor-analitica\memory\recent_changes.md`
- `C:\Users\user\.claude\projects\e--GPT1C-Processor-analitica\memory\open_issues.md`
- и связанные вспомогательные `MEMORY.md`, `roles.md`, `recent_changes.md`

Смысл:
- это не authoritative docs проекта;
- это внешняя память другого агентного контура;
- после текущей консолидации их нужно считать вспомогательными, а не источником истины.

### 17.3. Миграционные и backup-копии проекта

Найдены архивные копии:
- `C:\_migration_to_C_drive_20260429\backup_GPT1C_Processor_analitica_20260429_203309\...`
- `C:\_migration_to_C_drive_20260429\collector_crm_fix_backup_20260429_220421\...`
- `C:\_migration_to_C_drive_20260429\collector_crm_system_backup_20260429_221939\...`

Внутри них лежат старые версии:
- `CLAUDE.md`
- `gpt1c.md`
- `SESSION_CONTEXT.md`
- `audit/*.md`
- `autoagent/ORCHESTRATOR_CONTEXT_*.md`

Вывод:
- это архивные снимки, а не активная документация текущего репозитория;
- для рабочей базы знаний их дубли не нужны;
- при forensic-разборе их можно использовать как внешний архивный слой, но не как current truth.

### 17.4. Соседний проект

На `C:` также найден отдельный соседний проект:
- `C:\minbarakat_site`

Там есть собственные:
- `AGENTS.md`
- `CLAUDE.md`
- `README.md`
- `docs/reference.md`

Это другая кодовая база, не часть `GPT1C_Processor_analitica`.

---

## 18. Быстрый practical summary

Если открыть только один файл кроме `AGENTS.md`, открывать нужно этот.

Если нужен статус “что уже работает сейчас”:
- `master` уже содержит merge ветки `fix/log-noise-by-design-markers`
- approval flow и stop/payment-контур существенно стабилизированы
- collector/logging/CRM стали заметно прозрачнее
- тестовый контур на текущем срезе зелёный

Если нужен статус “что проверять руками дальше”:
- дисциплина Саиды
- спорные CRM ownership кейсы
- partial payment policy
- эксплуатация реальных очередей и SLA, а не только код

---

## 19. Collector dialog routing status (2026-05-11)

Текущее состояние ветки `collector/client_dialog.py` после аудита и исправления:

- устранён дублирующийся `_is_greeting_only()`
- исправлен `_normalize_text()`: нормализация больше не разваливает слова на посимвольные токены
- greeting-guard работает на нормализованном тексте и корректно ловит:
  - `Здравствуйте?`
  - `Добрый  день`
  - `Добрый день)`
- в `soft_positive` нейтральные acknowledgements больше не трактуются как готовность платить:
  - `Ок`
  - `Окей`
  - `Хорошо`
  - `Понял`
  - `Ладно`
  - `Да`
  - `Ага`
  - `Угу`
- убрана мёртвая запись `ас-саляму алейкум` из `_PURE_GREETINGS`: после нормализации она всё равно схлопывалась в вариант с пробелом

Смысл изменения:
- короткое приветствие или нейтральное подтверждение больше не должно вести в ложное:
  - `ждём ближайшую оплату`
  - `пришлите чек`
- вместо этого бот остаётся в `active` и задаёт уточняющий вопрос без эскалации и без `awaiting_payment_proof`

Покрытие:
- в `tests/test_collector.py` есть runtime-регрессии для:
  - `soft_positive + greeting`
  - `soft_positive + acknowledgement`

Проверка на текущем HEAD:
- `python -m py_compile collector/client_dialog.py` -> OK
- `python -X utf8 tests/test_collector.py` -> `481/481`

## Collector update 2026-05-14

- Актуальный collector-path: только --preview и --send-approved; refresh перед отправкой обязан сохранять семантику preview и не выбрасывать manager_review-клиентов, если они остаются актуальны в свежем shortlist.
- Повторный заход клиента в новый WA-цикл должен блокироваться не только active-dialog, но и wa_dialog_suppress.
- stop_status="exception" нельзя сразу возвращать в следующий WA-цикл: нужен короткий grace-period.
- Для admin-навигации актуальный actionable экран теперь определяется кнопкой 🧭 Актуальный батч.

---

## 16. 2026-05-15 Architecture Stabilization Notes

- Contact truth for collector is now explicit:
  - primary source: `config/clients.json`
  - fallback only: `config/debtors_contacts.json`
  - batch snapshot is historical evidence, not an active source of contact truth
- Runtime path:
  - `bot/crm_clients.py` -> `load_contacts_for_collector()`
  - `collector/collections_engine.py` -> `_load_collector_contacts()`
- Legacy compatibility path:
  - `collector/registry_manager.py` no longer writes `debtors_contacts.json`
  - registry mutations now go to CRM (`clients.json`)
- Safety/rollback:
  - pre-change backup: `config/clients.json.bak-arch-stabilization-20260515-205838`
  - if rollback is needed, restore that backup and revert the architecture commit
- **CRM ownership stabilization** (2026-05-15 late): explicit manual ownership marker via `ownership_manager/ownership_decided_*`; claim-broadcast and admin ambiguous-assign now persist ownership decision instead of relying on bare `manager` only.
- **Claim candidate filter respects explicit ownership**: `_crm_collect_unowned_claim_clients()` skips groups with `ownership_manager`, reducing repeat claim-broadcast on already manually assigned clients.

## 2026-05-16 — Test Hygiene Note

- `tests/test_collector.py`:
  - `P4 T10/T10b` больше не зависят от live `wa_dialog_suppress` в боевом `collector_state.json`;
  - для этих двух flat-тестов suppress-state теперь явно обнуляется через `patch("collector.collections_db.get_wa_dialog_suppress", return_value=None)`.
- Практический смысл:
  - тест проверяет именно business-rule legacy tail;
  - не падает из-за реального боевого клиента с активным suppress.
- Проверено:
  - `python -X utf8 tests/test_collector.py` -> `652/652`

## 2026-05-16 — Silence Audit Closed

- Silence-контур по итогам 2026-05-16 закрыт полностью:
  - `39bb8f0` — закрыты 3 реальные silence-ошибки:
    - `BUG-2`: identity Саиды больше не завязана на `name[:26]`, используется короткий token + state;
    - `BUG-saida`: только `confirmed_full` у Саиды на 7 дней убирает клиента из silence-отчёта; partial не убирает;
    - `BUG-5`: возраст старого долга больше не омолаживается после переключения окна отчёта; введён `logs/debt_age_history.json`.
  - `fc4bab5` — добит последний legacy `name[:26]` в full stoplist path Саиды.
  - `30bbbf9` — тестовая гигиена: `P4 T10/T10b` изолированы от live `wa_dialog_suppress` через явный patch.
  - `f2ab785` — закрыта косметика формулировок:
    - `остаток 14 дн` -> `долг 14 дн`
    - `МОЛЧАНИЕ/ПРОСРОЧКА/...` -> единая шкала `Долг 7-9 / 10-14 / 15-29 / 30+ дн`
    - `ВСЕГО МОЛЧАЩИХ` -> `ВСЕГО ДОЛГА`
- Проверено на финальном пакете:
  - `python -X utf8 tests/test_collector.py` -> `652/652`
  - `python -X utf8 tests/test_collector_regression_hermetic.py` -> `39/39`
  - `python -X utf8 tests/test_crm_regression.py` -> `44/44`
  - `python -X utf8 tests/test_project.py` -> `110/110`
  - `python -X utf8 tests/test_parsers.py` -> `69/69`

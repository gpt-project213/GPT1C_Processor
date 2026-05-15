# PROJECT_ENCYCLOPEDIA

Актуальная единая база знаний по `GPT1C_Processor_analitica`.

Статус: актуально на `2026-05-15`  
Текущая базовая ветка: `master`  
Последний коммит: `785bc55`

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
- `collections_engine.py` (preview + send-approved) переведён на `dialog_blocks_new_outreach()`;
  ? preview-path ??? ???????? legacy TTL-????? ??? `awaiting_payment_proof` ?
  ????????????? ??????????, ?? ?????????? ???????????? helper-??????.

Penalty reset — актуально на 2026-05-15:
- `build_reset_state()` / `reset_penalty_state()` в `collector/approval_penalty.py`.
- `reset_penalty_state()`: делает backup `logs/approval_penalty_state.json`, пишет новый state
  с `wa_reset_floor` и `crm_reset_floor` — батчи/CRM-записи старше floor не получают штраф
  при следующем backfill.
- Admin UI: кнопка `♻️ Сброс штрафов` в экране 🤖 Коллектор (confirm-step, admin only).
  После подтверждения показывает: месяц, WA floor, CRM floor, имя backup-файла.

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

Что ещё не закрыто:
- stale dup-review callback: отвечает «запрос устарел», но без full cleanup-строгости как у claim
- три несогласованных хранилища: `clients.json` / `debtors_contacts.json` / batch snapshot

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

Актуальный рабочий набор к `2026-05-15`:

| Тест | Результат | Примечание |
|---|---|---|
| `tests/test_project.py` | `110/110` | |
| `tests/test_collector.py` | `643/643` | +13 (silent dialog + penalty reset, 2026-05-15) |
| `tests/test_collector_regression_hermetic.py` | `18/18` | +3 TimeoutLabelHermeticTests (2026-05-14) |
| `tests/test_crm_regression.py` | `28/28` | +14 StabilizationRegressionTests + lock contention (2026-05-15) |
| `tests/test_parsers.py` | `69/69` | |
| `tests/test_audit_reports_20260414.py` | `8/8` | |
| `tests/test_phase2_safe_send.py` | green | |
| `tests/test_log_monitor.py` | `9/9` | |

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
- stale dup-review callback: дать полную cleanup-строгость как у claim
- ambiguous multi-manager ownership: значительно улучшено, но не исчерпано
- три несогласованных хранилища контактов: clients.json / debtors_contacts.json / batch snapshot

### Collector
- `build_reset_state()` / `reset_penalty_state()` подключены к admin UI; при необходимости расширить до CLI-утилиты
- `COLLECTOR_SILENT_ACTIVE_RESEND_HOURS` env по умолчанию 24ч — проверить достаточность порога в бою

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

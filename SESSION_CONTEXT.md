# SESSION CONTEXT — АРХИВ

> **ВНИМАНИЕ:** Этот файл содержит исторические сессии (2026-04-09, 2026-04-13, 2026-04-14).
> Многие "OPEN" пункты уже закрыты коммитами. **Актуальный статус → `gpt1c.md`.**

---

## HANDOFF 2026-04-22 16:40 Asia/Almaty

### Что зафиксировано в репозитории после предыдущих handoff

- `65d2164` — `docs(audit): map audit corpus and fix path anomaly`
  - добавлена карта содержимого `audit/AUDIT_CONTENT_MAP_20260422.md`
  - исправлена git-анomaly по старому пути `аудит/` без потери содержимого
- `8171b4a` — `docs(audit): clarify 2026-04-21 draft status`
  - `audit/AUDIT_20260421.md` помечен как незавершённый audit draft, а не финальный вердикт
- `863ad80` — `chore(project): remove obsolete pdf traces`
  - удалён пустой каталог `reports/pdf`
  - удалён `pdfkit` из `requirements.txt`
  - убраны оставшиеся project-side PDF-следы вне `audit/`

### Что проверено

- `python -m py_compile config.py` — OK
- поиск по проекту вне `audit/`, `.venv`, `__pycache__` на:
  - `pdf`
  - `PDF`
  - `pdfkit`
  - `reports/pdf`
  - `*.pdf`
  дал `0` совпадений

### Что читать новому ИИ в первую очередь

Чтобы быстро и без фантазий восстановить реальную картину проекта, достаточно прочитать с начала до конца:

1. `AGENTS.md`
2. `CLAUDE.md`
3. `gpt1c.md`
4. `SESSION_CONTEXT.md`
5. `audit/AUDIT_CONTENT_MAP_20260422.md`
6. `audit/AUDIT_COLLECTOR_20260422.md`

А затем посмотреть ключевые коммиты этой ветки:

- `67f8e0f` — log-noise cleanup
- `3695405` — collector state hardening
- `6f7c6bf` — stale admin requests + voice STT repair
- `da5486d` — voice STT + guard empty AI analysis
- `48017d0` — silence alerts once daily
- `ea74457` — grouped 1C sales + manager top3
- `65d2164` — audit map + path anomaly fix
- `8171b4a` — audit draft clarification
- `863ad80` — PDF traces removed

### Текущее состояние дерева

- после этого handoff в рабочем дереве не должно оставаться незакоммиченного `SESSION_CONTEXT.md`
- если появятся новые локальные правки, сначала смотреть `git status --short`, затем читать этот файл сверху вниз

---

## HANDOFF 2026-04-22 11:22 Asia/Almaty

### Что доделано после предыдущего handoff

- В `collector/approval_flow.py` и `collector/collections_engine.py` добавлена защита от конфликта старого и нового approval-запроса:
  - новый актуальный preview-запрос вытесняет предыдущий активный;
  - старый запрос получает статус `superseded`;
  - старые manager-preview сообщения закрываются, кнопки снимаются;
  - старые manager-callback больше не принимаются сервером.
- В `collector/approval_flow.py` добавлена эскалация при молчании менеджеров:
  - через `1` час молчания запрос автоматически переводится на решение администратора;
  - молчавшие менеджеры получают `timeout`;
  - администратору отправляется итоговая сводка без ожидания всех ответов.
- В `bot/send_reports.py` hourly `collector_reminder_task()` теперь дополнительно запускает проверку эскалации молчавших approval-запросов.
- Для менеджеров тексты сделаны без техтерминов:
  - `Запрос устарел`
  - `Исходный список уже закрыт`
  - `Сформирован новый список`

### Проверки

- `python -m py_compile collector/approval_flow.py` — OK
- `python -m py_compile collector/collections_engine.py` — OK
- `python -m py_compile bot/send_reports.py` — OK
- `$env:WHATSAPP_ENABLED='0'; $env:LIVE_SEND_ALLOWED='0'; python -X utf8 tests\test_collector.py` — `266/266`
- `python -X utf8 tests\test_phase2_safe_send.py` — PASS

### Что изменилось в тестах

- Добавлены регрессии:
  - `APPROVAL T3d` — `superseded` не считается активным
  - `APPROVAL T10e` — после 1 часа молчания запрос переходит в `pending_admin`
  - `APPROVAL T10f` — молчавшие менеджеры получают `timeout`, админу уходит сводка
  - `APPROVAL T10g` — активный запрос можно закрыть как `superseded`
  - `APPROVAL T10h` — manager-callback по уже закрытому запросу блокируется

### Разбор LOG MONITOR по ошибке `--send disabled`

- Уведомление `collector_20260422.log: ERROR --send disabled for Phase 2 controlled live` не указывает на боевой scheduler.
- По самому `logs/collector_20260422.log` перед этой строкой идут тестовые записи:
  - `send-approved: batch=20260412-120000-ab12 ...`
  - `[TEST] legacy manager_dialog live send blocked ...`
- Вывод: это след тестового прогона в рабочем лог-файле коллектора, а не продовая попытка scheduler вызвать `--send`.
- Отдельный операционный хвост:
  - был закрыт в этой же итерации:
    - добавлен `COLLECTOR_TEST_MODE=1`
    - `collector/collections_engine.py` в тестовом режиме больше не пишет в боевой `collector_YYYYMMDD.log`
    - `tests/test_collector.py` и `tests/test_phase2_safe_send.py` выставляют этот флаг до импорта модуля
  - проверено фактом: после повторного тестового прогона в `06:23` хвост `logs/collector_20260422.log` не изменился

### Актуальные файлы этой итерации

- `collector/approval_flow.py`
- `collector/collections_engine.py`
- `bot/send_reports.py`
- `tests/test_collector.py`
- `tests/test_phase2_safe_send.py`
- `audit/AUDIT_COLLECTOR_20260422.md`

### Не смешивать с collector-коммитом

- `bot/sales_summary.py`
- `sales_parser.py`
- `tests/test_parsers.py`
- `SESSION_CONTEXT.md`
- `audit/AUDIT_20260421.md`

### Следующий безопасный шаг

1. Сделать изолированный collector-коммит без sales/parser-правок.
2. Затем пуш.

## HANDOFF 2026-04-22 10:58 Asia/Almaty

### Что сделано в этой сессии

- Проведён целевой аудит коллектора по цепочке:
  - `scheduler -> preview -> manager approvals -> admin approve -> send-approved -> batch state`
- Подтверждены и исправлены 2 state-багa в `collector/approval_flow.py`:
  1. `load_latest_batch()` больше не считает финальными "активными" батчи со статусами:
     - `sent`
     - `partially_sent`
     - `send_failed`
     - `send_empty`
  2. `expire_old_batches()` теперь:
     - ставит `expired_at`
     - переводит молчавших менеджеров из `pending` / `manual_editing` в `timeout`

### Что уже было в рабочем дереве и дополнительно верифицировано

- раннее уведомление администратору о создании approval-батча
- ручной выбор клиентов администратором перед отправкой
- кнопка `Отправить сейчас` после admin approve
- safe-send path отправляет только `approved_clients`

### Тесты и проверки

- `python -m py_compile collector\\approval_flow.py` — OK
- `python -m py_compile collector/collections_engine.py` — OK
- `$env:WHATSAPP_ENABLED='0'; $env:LIVE_SEND_ALLOWED='0'; python -X utf8 tests\\test_collector.py` — `261/261`
- `python -X utf8 tests\\test_phase2_safe_send.py` — PASS

Примечание по окружению:
- первый запуск `tests/test_phase2_safe_send.py` в песочнице упал на `PermissionError` по `logs/collector_20260422.log`
- повторный запуск вне песочницы прошёл успешно; это был lock лог-файла, не поломка бизнес-логики

### Новые/обновлённые файлы этой сессии

- `collector/approval_flow.py`
- `tests/test_collector.py`
- `audit/AUDIT_COLLECTOR_20260422.md`

### Состояние аудита

- старый файл `audit/AUDIT_20260421.md` остаётся как черновик/рабочий draft, не перезаписывался
- новый актуальный файл по этой сессии:
  - `audit/AUDIT_COLLECTOR_20260422.md`

### Состояние git на момент handoff

- Ветка: `fix/log-noise-by-design-markers`
- `HEAD`: `67f8e0f`
- Коммит по коллектору ЕЩЁ НЕ создан

Причина остановки:
- попытка выполнить `git add ... && git commit ...` через PowerShell сорвалась не по git-логике, а из-за синтаксиса:
  - `&&` не поддержан как разделитель в данной версии PowerShell

### Что готово к коммиту

Логически готово коммитить только эти файлы:
- `collector/approval_flow.py`
- `collector/collections_engine.py`
- `tests/test_collector.py`
- `audit/AUDIT_COLLECTOR_20260422.md`

Не брать в этот коммит:
- `bot/sales_summary.py`
- `sales_parser.py`
- `tests/test_parsers.py`
- `SESSION_CONTEXT.md`
- `audit/AUDIT_20260421.md`

### Следующий безопасный шаг

Выполнить по отдельности, без `&&`:

1. `git add collector/approval_flow.py collector/collections_engine.py tests/test_collector.py audit/AUDIT_COLLECTOR_20260422.md`
2. `git commit -m "fix(collector): harden approval batch states and save audit"`
3. `git push`

Дата последней фиксации: 2026-04-14
Проект: `GPT1C_Processor_analitica`

---

## Сессия 2026-04-14: фиксы утечки данных + верификация дебиторки

### Режим работы

Продолжение «SAFE SURGERY PROJECT MODE». Фокус — утечка данных в аналитических отчётах.

### Контекст проблемы

Субадмин (Алена) получала в DSO/аналитике данные внутренних клиентов (Минай, Алибек, Минбаракат), которые не должны быть ей видны. Причина — сводный debt JSON (manager="—", 517 клиентов) мёржился с именными файлами в `load_best_debt_json()`, а `_PREFIX_MAP` по первой букве назначал внутренних клиентов реальным менеджерам.

### Верификация дебиторки (твои правки из предыдущей сессии)

Проверены правки из stash (до моих изменений):
- `_is_manager_debt_extended_name()` — helper проверки именных файлов
- `_classify_type()` — сводные debt_ext → UNKNOWN (не индексируются)
- `send_with_acl()` — двойная защита DEBT_EXTENDED для не-admin
- `tests/test_parsers.py` — 3 новых кейса

**Вердикт: правки корректные, ничего лишнего не сделано.**

### 3 коммита утечки данных

| # | Коммит | Файл | Суть |
|---|--------|------|------|
| 1 | `e54da79` | `dso_aging_report.py` | `load_best_debt_json()` пропускает сводный файл (manager="—"/""/ None) |
| 2 | `9f75c6c` | `rfm_clients_report.py`, `revenue_concentration_report.py` | `"—"` добавлен в `_SKIP_MANAGERS` (превентивный) |
| 3 | `2c1f832` | `bot/send_reports.py`, `tests/test_parsers.py` | 1) Дебиторка: classify+ACL защита. 2) Sales fallback на сводный — только admin |

### Доказательства

- DSO: тест `_tmp_dso_leak_proof.py` — 411 клиентов из 4 именных файлов, 0 внутренних
- `test_parsers.py`: 61/61 тестов прошло, 0 упало
- Компиляция всех файлов — OK
- Worktree чистый после коммитов
- Все 25 коммитов запушены на GitHub (origin/master = HEAD)

### Текущие версии файлов

- `bot/send_reports.py` → v9.4.52/14.04.2026
- `dso_aging_report.py` → v1.1.3
- `rfm_clients_report.py` → v1.1.5
- `revenue_concentration_report.py` → v1.1.5

### Что ещё не сделано (из предыдущей сессии — OPEN)

**Функциональное:**
1. `debt_collector_daily` (17:00) всегда `--dry-run` — fallback не работает
2. `config/collector_prompts.json` не существует → WARNING при каждом запуске
3. `+77001234567` в 6 UI-подсказках — Вадим просил убрать
4. Условная отгрузка (4 кнопки) — не реализована полностью
5. 1 клиент без `manager_name` в `create_batch`

**Архитектурное:**
- ARCH-1: `txt_to_html` в двух местах — разные интерфейсы
- ARCH-3: inline HTML в `expenses_parser.py`
- BSR-01: `logging.Formatter.formatTime` monkey-patch (FIXED в send_reports, но может быть в других)

**Данные:**
- Сверка `config/debtors_contacts.json` vs `collector/debtors_contacts.json`
- Очистка test-like записей из runtime-state

### Ключевые знания о проекте

- Алена = субадмин + менеджер (двойная роль), подшефные: Магира, Оксана
- Сводные debt-файлы имеют `manager="—"` (em-dash), sales — `"Не определён"`
- `_PREFIX_MAP` по первой букве клиента — ненадёжен для внутренних клиентов
- unknown пользователи = полный 0 доступа (реализовано `_acl_gate`)
- DSO/RFM/Concentration используют `load_best_debt_json` / `load_all_jsons_merged` для мёржа
- Sales fallback на сводный — теперь только для admin

---

## Сессия 2026-04-13: полный аудит + безопасная хирургия

### Режим работы

«SAFE SURGERY PROJECT MODE» — только доказанные баги, по одному, с py_compile + тестами + отдельным коммитом.

### Что сделано

**Полный аудит проекта** → `аудит/ПОЛНЫЙ_СВОД_АУДИТА_13.04.2026.md` (21 секция).

**16 bug-fix коммитов** (все 335 тестов зелёные после каждого):

| # | ID | Коммит | Файл | Суть |
|---|-----|--------|------|------|
| 1 | CRIT-1 | `6449778` | `collector/voice_calls.py` | logger init перед try/except — NameError при bad env |
| 2 | CRIT-2 | `1907556` | `run_pipeline_all_mp.py` | .work → .xlsx при ошибке (файлы застревали навсегда) |
| 3 | CRIT-3 | `cb1d6c3` | `dso_aging_report.py` | Убран fallback на closing — enforced debt-only инвариант |
| 4 | DEAD-1 | `bdaf933` | `collector/manager_dialog.py` | 119 строк мёртвого кода после return False |
| 5 | TEST | `3d58a17` | `tests/test_collector.py` | Mock datetime в REMIND тестах (ломались ночью) |
| 6 | HIGH-1 | `0962b95` | `rfm_clients_report.py`, `revenue_concentration_report.py` | Дедупликация по total вместо несуществующего revenue |
| 7 | HIGH-2 | `d2a6908` | `bot/debt_stop_control.py` | NamedTemporaryFile вместо .tmp (race condition) |
| 8 | HIGH-3 | `a831ee5` | `config.py` | int(val) для chat_id из managers.json (строка → None) |
| 9 | R3 | `323d98e` | `debt_auto_report.py` | Warning при fallback find_header → [0,1] |
| 10 | R4 | `67df3f9` | `collector/debt_monitor.py` | date.today() → datetime.now(TZ).date() |
| 11 | R5 | `f88a91f` | `collector/registry_manager.py` | Маскировка телефонов в логах (PII) |
| 12 | SEC-3 | `472dcf3` | `bot/send_reports.py` | Generic error вместо f"Ошибка: {e}" пользователю |
| 13 | CB-2 | `569fe52` | `bot/send_reports.py` | Двойной query.answer() удалён |
| 14 | LOW-1 | `f1eca89` | `bot/send_reports.py` | _TzFormatter subclass вместо глобального monkey-patch |
| 15 | ACL-1 | `8bed42f` | `bot/send_reports.py` | unknown → минимальное меню вместо менеджерского |
| 16 | ACL-2 | `ef82c7b` | `bot/send_reports.py` | ПОЛНАЯ блокировка unknown: _acl_gate() во всех entry points |

**Также до аудита (начало сессии):**
- Issue-1: `debt_collector_daily` fallback `--dry-run` → `--preview`
- Issue-2: `collector_prompts.json` WARNING → DEBUG
- Issue-3: `+77001234567` → generic placeholders в 6 местах UI
- Issue-5: `create_batch` — client names в логе при skip без manager_name

### Текущие версии файлов после сессии

- `bot/send_reports.py` → v9.4.51
- `run_pipeline_all_mp.py` → v1.5.4
- `dso_aging_report.py` → v1.1.2
- `config.py` → v3.6.2
- `debt_auto_report.py` → v2.7.7
- `collector/voice_calls.py` — logger moved up
- `collector/manager_dialog.py` → v1.0.1
- `collector/debt_monitor.py` → v1.0.6
- `collector/registry_manager.py` → v1.0.1
- `bot/debt_stop_control.py` → v1.0.4
- `rfm_clients_report.py` → v1.1.5
- `revenue_concentration_report.py` → v1.1.5

### Оставшиеся OPEN (не баги — требуют решений)

| ID | Тип | Описание |
|----|-----|----------|
| REFACTOR | Массовый рефакторинг | ~50+ `except Exception` по всему проекту |
| ARCH-1 | Архитектурное | `txt_to_html` дублируется (tools/ и bot/send_reports.py) |
| ARCH-3 | Архитектурное | Inline HTML в `expenses_parser.py` |
| D5 | Архитектурное | `money()` дублируется с разной сигнатурой |
| HARDCODE | Бизнес-решения | ~25+ hardcoded порогов/процентов/лимитов |
| FEATURE | Feature request | Retry для AI API вызовов (ai_analyzer.py) |
| FEATURE | Feature request | Тесты для silence_alerts, opportunity_loss |
| DOCS | Документация | 7 расхождений CLAUDE.md ↔ код |
| Issue-4 | Не реализовано | Условная отгрузка (4 кнопки для админа в debt_stop_control) |

**Ни один из оставшихся — не точечный баг.** Каждый требует либо архитектурного решения, либо бизнес-решения, либо это feature request.

### Ключевые знания о проекте (для продолжения)

1. **Архитектура:** 9 слоёв, монолит `bot/send_reports.py` ~7660 строк — главный бот
2. **Тесты:** 81 + 254 = 335, запуск: `python -X utf8 tests/test_project.py && python -X utf8 tests/test_collector.py`
3. **Роли:** admin(Вадим 7422963573), subadmin(Алена 188939016 — dual role!), managers(Оксана, Магира, Ергали, Алена)
4. **Инвариант:** всегда `debt`, никогда `closing`
5. **TZ:** всегда `ZoneInfo(os.getenv("TZ", "Asia/Almaty"))`
6. **Менеджеры:** из `config/managers.json`, никогда hardcode
7. **ACL:** unknown теперь полностью заблокированы (_acl_gate)
8. **Collector:** уровни 0-5, approval flow, WhatsApp+Telegram, state в `logs/collector_state.json`
9. **Pipeline:** queue/ → .work claim → process → processed/; .work теперь возвращается в .xlsx при ошибке
10. **Протокол правок:** один баг = один коммит, py_compile, тесты, version bump +0.0.1

### Файлы контекста на флешке

- `CLAUDE.md` — мастер-документ (обновлён 2026-04-13)
- `SESSION_CONTEXT.md` — этот файл
- `аудит/ПОЛНЫЙ_СВОД_АУДИТА_13.04.2026.md` — полный свод аудита
- `repo_map.json` — карта файлов проекта

### Рекомендуемый следующий шаг

1. Если нужны фичи — Issue-4 (4 кнопки условной отгрузки)
2. Если нужна чистка — `except Exception` рефакторинг (начать с collector/)
3. Если нужна документация — синхронизировать CLAUDE.md ↔ код (7 расхождений)
4. Если нужна приёмка — backup state, очистка тестовых хвостов, боевые сценарии

---

## Сессия 2026-04-09 (предыдущая)

## Что сделано в этой сессии

- Выполнен полный локальный аудит проекта.
- Создан итоговый документ:
  - `AUDIT_FULL_PROJECT_2026-04-09.md`
- Сверено текущее состояние проекта с ТЗ из:
  - `ТЗ.txt`

## Ключевой вывод по ТЗ

Работа шла именно по ТЗ про стабилизацию коллектора, WhatsApp-оповещений и менеджерских диалогов.

Но ТЗ закрыто не полностью.

Статус:

- основные аварийные дефекты сняты;
- архитектурно опасные блокировки ослаблены;
- collector/CRM стали устойчивее;
- финальная production-доводка state и приёмка по живым сценариям ещё нужны.

## Что уже соответствует ТЗ

### Collector / manager flow

- жёсткий lock одного менеджера на весь поток ослаблен;
- direct send разрешается только при наличии подтверждённого телефона;
- добавлена защита от дубля по клиенту на уровне активных/недавно обработанных диалогов;
- молчание менеджера переводит кейс в контроль/эскалацию, а не в вечное зависание;
- повторная эскалация подавляется.

### WhatsApp / CONFIRMED

- `CONFIRMED` больше не ставится до успешного send-path;
- при неуспешной отправке кейс не считается завершённым;
- WhatsApp pipeline доведён до стадии реальной попытки отправки, а не только до кнопки менеджера.

### Contacts

- запись collector-контактов переведена на `config/debtors_contacts.json`;
- при записи сохраняются `whatsapp` как основное поле и `phone` как совместимое;
- `_needs_phone` снимается при корректном вводе телефона.

### CRM flow

- имя клиента больше не блокирует сбор телефона;
- поддержаны режимы:
  - ввести имя;
  - оставить как в системе;
  - позже;
- сохраняются:
  - `original_name`
  - `display_name`
  - `name_mode`
  - `name_review_needed`
- reminder уважает `paused_until` после действия “Позже”.

### Cleanup / TTL

- stale dialog cleanup больше не переводит живой кейс в `DONE`;
- вместо этого ставится `control_deadline`;
- legacy pending `__phone_pending__` / `__name_pending__` очищаются по TTL.

## Что не закрыто полностью

### 1. Полное соответствие ТЗ по collector-state

Корневой ключ manager-dialog всё ещё завязан на `manager_chat_id`, а не на независимый `client/session` key.

Это значит:

- проблема смягчена;
- но базовая архитектурная зависимость ещё не исчезла.

### 2. Контакты

В проекте всё ещё существует legacy-файл:

- `collector/debtors_contacts.json`

Даже если основной рабочий путь уже переведён на `config/debtors_contacts.json`, legacy-хвост остаётся источником риска и путаницы.

### 3. Runtime-state не стерилен

В текущем локальном runtime обнаружены test-like записи в:

- `logs/collector_dialogs.json`

Примеры:

- `OTHER CLIENT`
- `NO PHONE CLIENT`

Это означает, что перед финальной приёмкой нужен аккуратный state cleanup с backup.

### 4. Дедупликация клиента

Текущая anti-duplicate логика опирается в основном на `client_name`.

Это рабочее временное решение, но не идеальное production-решение.

### 5. Полная приёмка по боевым сценариям

Нужен отдельный финальный этап:

- backup state;
- сверка state ownership;
- чистка тестовых хвостов;
- повторный прогон сценариев на живом runtime-state.

Только после этого можно честно фиксировать “ТЗ закрыто”.

## Самые важные файлы для следующего этапа

- `AUDIT_FULL_PROJECT_2026-04-09.md`
- `ТЗ.txt`
- `bot/send_reports.py`
- `collector/manager_dialog.py`
- `collector/dialog_store.py`
- `collector/collections_engine.py`
- `collector/debt_monitor.py`
- `bot/crm_clients.py`
- `config/debtors_contacts.json`
- `collector/debtors_contacts.json`
- `logs/crm_pending_state.json`
- `logs/collector_dialogs.json`
- `logs/collector_state.json`

## Рекомендуемый следующий шаг

Не делать новый широкий рефакторинг.

Следующий правильный этап:

1. backup рабочих state-файлов;
2. разовый акт сверки `config/debtors_contacts.json` vs `collector/debtors_contacts.json`;
3. очистка test-like записей из runtime-state;
4. финальная приёмка по боевым сценариям;
5. только затем закрытие ТЗ.

---

## Session Handoff - 2026-04-22 08:53 +05:00

### What was done

- Checked unattended health logs on `2026-04-21`.
- Confirmed `balance Excel` attachments are by-design non-pipeline inputs and should be ignored.
- Confirmed repeated `create_batch ... without manager_name` log line came from a test fixture, not a production client.
- Implemented explicit log markers to prevent both cases from being misread as bugs:
  - `imap_fetcher.py`:
    - version `v4.4.5 -> v4.4.6`
    - added explicit `IGNORE by-design non-pipeline attachment (...)` for:
      - files containing `баланс`
      - files containing `ведомость денежных средств`
  - `collector/approval_flow.py`:
    - version `1.0.3 -> 1.0.4`
    - test fixtures without `manager_name` now log at `INFO`
    - real data without `manager_name` still logs at `WARNING`
  - `tests/test_collector.py`:
    - renamed fixture to `TEST fixture: клиент без manager_name`

### Commit / branch

- Branch created: `fix/log-noise-by-design-markers`
- Commit created: `67f8e0f fix(logs): mark by-design IMAP ignores and collector test-noise explicitly`

### Verification completed

- `python -m py_compile imap_fetcher.py collector\approval_flow.py` -> OK
- `python -X utf8 tests\test_project.py` -> `101/101`
- `WHATSAPP_ENABLED=0 LIVE_SEND_ALLOWED=0 python -X utf8 tests\test_collector.py` -> `256/256`
- Additional local check requested by user:
  - `python -m py_compile bot\sales_summary.py` -> OK

### Important working tree state left untouched

At the time of context save, working tree is NOT clean. These files were intentionally left alone:

- modified:
  - `bot/sales_summary.py`
  - `collector/approval_flow.py`
  - `collector/collections_engine.py`
- untracked:
  - `audit/AUDIT_20260421.md`

Notes:

- `audit/AUDIT_20260421.md` is Claude's unfinished audit draft from `2026-04-21`; user explicitly asked to leave it untouched for later continuation.
- Do not delete, stage, or commit that audit draft unless user explicitly asks.
- Current modified state of `bot/sales_summary.py` and `collector/collections_engine.py` was not touched in this handoff turn.

### Most recent user intent

- Keep the unfinished audit draft intact.
- Save context for later continuation.

### Recommended next step

Before any new edits:

1. run `git status --short`;
2. inspect whether `collector/approval_flow.py` local modification is only the committed `67f8e0f` patch or additional user edits on top;
3. keep `audit/AUDIT_20260421.md` out of commits until Claude's audit continuation resumes.
## Session Handoff - 2026-04-22 11:45 +05:00

### What was done

- Confirmed live collector path already worked in production on batch `20260422-105927-4ab7`:
  - preview -> manager replies -> admin summary -> admin approve -> send-approved -> WhatsApp
  - final state became `sent`
- Investigated why one incoming voice message was not recognized:
  - root cause was not client silence and not manager flow
  - AssemblyAI returned `400` because request still sent deprecated field `speech_model`
- Applied follow-up collector hardening:
  - `collector/whatsapp_poller.py`
    - version `1.1.2 -> 1.1.3`
    - removed deprecated `speech_model` from AssemblyAI transcript request
  - `collector/approval_flow.py`
    - version `1.0.8 -> 1.0.9`
    - old admin messages now close when a newer актуальный список replaces the current one
    - admin callbacks on stale/finalized requests are blocked
    - in manual admin selection, a client disappears from the list immediately after `Отправлять` / `Не отправлять`
  - `collector/collections_engine.py`
    - when a new preview supersedes an active one, closes not only manager previews but also old admin messages
  - `tests/test_collector.py`
    - added regressions for disappearing admin list item and stale admin callback blocking
  - `audit/AUDIT_COLLECTOR_20260422.md`
    - added findings for stale admin messages and AssemblyAI voice STT failure

### Verification completed

- `python -m py_compile collector/approval_flow.py` -> OK
- `python -m py_compile collector/collections_engine.py` -> OK
- `python -m py_compile collector/whatsapp_poller.py` -> OK
- `WHATSAPP_ENABLED=0 LIVE_SEND_ALLOWED=0 python -X utf8 tests\test_collector.py` -> `269/269`

### Important working tree state left untouched

The working tree is still intentionally dirty outside this collector follow-up:

- modified:
  - `bot/sales_summary.py`
  - `sales_parser.py`
  - `tests/test_parsers.py`
- untracked:
  - `audit/AUDIT_20260421.md`

Do not mix these sales/parser files or the old draft audit into the collector follow-up commit.

### Most recent user intent

- Old hanging messages must never stay actionable after a newer актуальный список appears.
- Manager/admin UI must stay in plain Russian without technical batch jargon.
- After collector stabilization, continue live monitoring rather than broad refactoring.

### Recommended next step

1. create a narrow collector follow-up commit with:
   - `collector/approval_flow.py`
   - `collector/collections_engine.py`
   - `collector/whatsapp_poller.py`
   - `tests/test_collector.py`
   - `audit/AUDIT_COLLECTOR_20260422.md`
2. push it to `origin/fix/log-noise-by-design-markers`
3. continue monitoring the next real collector cycle:
   - old messages close
   - stale callbacks do not revive old requests
   - next incoming voice message transcribes without the deprecated-parameter failure

## Handoff Update - 2026-04-22 15:15 +05:00

### Sales tail completed

- Separate sales/parser tail finished and pushed:
  - commit `ea74457` — `fix(sales): handle grouped 1c clients and manager top3`
- Files included in this commit:
  - `bot/sales_summary.py`
  - `sales_parser.py`
  - `tests/test_parsers.py`

### What was fixed

- `bot/sales_summary.py`
  - manager Top-3 clients now supports both JSON contracts:
    - new pipeline format `{client,total}`
    - legacy format `{name,amount}`
  - pseudo-client buckets such as `Без клиента` are excluded from Top-3
- `sales_parser.py`
  - fixed grouped 1C sales where `Контрагент` and `Номенклатура` share one column
  - client aggregate rows with sale amount now start a new `current_client`
  - orphan product rows no longer create artificial bucket `Без клиента`; they are logged and skipped
- `tests/test_parsers.py`
  - added regression on real file `Продажи Магира (302).xlsx`
  - asserts:
    - `client_count > 40`
    - no `Без клиента` bucket
    - `total_revenue > 10_000_000`

### Verification completed

- `python -m py_compile bot/sales_summary.py` -> OK
- `python -m py_compile sales_parser.py` -> OK
- `python -X utf8 tests/test_parsers.py` -> `69/69`
- Evidence from test run:
  - `Продажи Магира (302)` parsed with `client_count=54`
  - `total_revenue=10624154.86`

### Working tree intentionally left dirty

- modified:
  - `SESSION_CONTEXT.md`
- deleted/untracked anomaly left untouched:
  - old Russian-named files under `audit/` appear as both `D` and `??`
- untracked:
  - `audit/AUDIT_20260421.md`

Do not mix the audit-path anomaly into the sales or collector commits without separate inspection.

## Handoff Update - 2026-04-22 16:05 +05:00

### Audit folder triage completed

- Fully reviewed the current `audit/` corpus by content, not by filename only.
- Added:
  - `audit/AUDIT_CONTENT_MAP_20260422.md`
    - factual map of audit document roles and why they matter
- Confirmed that `audit/` is not a trash folder:
  - it contains architecture targets
  - incident reports
  - collector launch/readiness protocols
  - historical runtime evidence
  - director-facing shortlist explanations
  - Codex/Claude handoff context

### Audit path anomaly fixed without content loss

- The old git anomaly was real:
  - two Russian audit files were tracked under legacy path `аудит/`
  - actual files on disk lived under `audit/`
- Verified by blob hashes that content was identical.
- Fixed as a pure git path correction:
  - commit `65d2164` — `docs(audit): map audit corpus and fix path anomaly`
  - git recorded both files as `rename (100%)`, not delete/recreate

### AUDIT_20260421 clarified

- `audit/AUDIT_20260421.md` was reviewed.
- It is useful, but it is an unfinished audit draft, not a final full-project verdict.
- Added an explicit status note at the top of the file so future sessions do not misread it as a fully current completed audit.

### Current remaining dirty files

- modified:
  - `SESSION_CONTEXT.md`
- untracked no longer:
  - `audit/AUDIT_20260421.md` is now a tracked working file if the user decides to commit this clarification

### Recommended next step

1. make a small docs-only commit with:
   - `audit/AUDIT_20260421.md`
2. keep `SESSION_CONTEXT.md` local unless the user wants it committed too

## Handoff Update - 2026-04-22 16:45 +05:00

### Phase 4 quick scan closed

- Completed quick scan of remaining modules and repository artifacts after Phase 3.
- Main actionable finding was not a runtime bug but tracked secret exposure in `logs_public/`.
- Added:
  - `audit/AUDIT_PHASE4_QUICKSCAN_20260422.md`

### Tracked log secret exposure fixed

- Historical tracked logs in `logs_public/` contained full Telegram bot token URLs.
- Sanitized only the secret-bearing fragments in place:
  - `https://api.telegram.org/bot<real-token>/...`
  - became `https://api.telegram.org/bot<TG_TOKEN>/...`
- No log files were deleted.
- Operational content of the logs was preserved.

Affected files:
- `logs_public/send_reports_20260212.log`
- `logs_public/send_reports_20260216.log`
- `logs_public/send_reports_20260217.log`
- `logs_public/send_reports_20260218.log`
- `logs_public/send_reports_20260219.log`
- `logs_public/send_reports_20260220.log`
- `logs_public/send_reports_20260222.log`
- `logs_public/send_reports_20260223.log`
- `logs_public/send_reports_20260225.log`
- `logs_public/send_reports_20260226.log`

Verification:
- `rg -n "api\.telegram\.org/bot[0-9]{5,}:[A-Za-z0-9_-]+/|bot[0-9]{5,}:[A-Za-z0-9_-]+" logs_public`
  - no matches after redaction

### repo_map refreshed

- `repo_map.json` was stale:
  - old branch: `master`
  - old timestamp: `2026-03-10 23:40:59`
- Regenerated to current branch:
  - `fix/log-noise-by-design-markers`

### Current dirty files

- modified:
  - `logs_public/send_reports_20260212.log`
  - `logs_public/send_reports_20260216.log`
  - `logs_public/send_reports_20260217.log`
  - `logs_public/send_reports_20260218.log`
  - `logs_public/send_reports_20260219.log`
  - `logs_public/send_reports_20260220.log`
  - `logs_public/send_reports_20260222.log`
  - `logs_public/send_reports_20260223.log`
  - `logs_public/send_reports_20260225.log`
  - `logs_public/send_reports_20260226.log`
  - `repo_map.json`
  - `SESSION_CONTEXT.md`
- added:
  - `audit/AUDIT_PHASE4_QUICKSCAN_20260422.md`

### Recommended next step

1. commit the Phase 4 artifact cleanup separately from runtime code
2. push
3. optionally continue with deeper review of `bot/send_reports.py` only if a new concrete issue appears

## Handoff Update - 2026-04-22 19:20 +05:00

### Freshness fix for stale report selection

- Trigger: user reported that `А Фурманова Евгений (склад № 20)` was shown in stop-control with debt `285 535 ₸`, while the fresh Excel source already reflected payment and a much smaller остаток.
- Root cause confirmed against primary sources:
  - stale source previously selected:
    - `reports/excel/processed/20260418170613_Ведомость_по_взаиморасчетам_с_контрагентами_Алена (336).xlsx`
    - contained debt `285535.02`
  - fresh source that should win:
    - `reports/excel/processed/20260422155716_Детальный Дебиторы Алена (143).xlsx`
    - contained debt `36588.52`
- The bug was not in Excel and not in the client row. It was in selectors that still allowed older report families (`Ведомость ...`) to outrank fresh current ones.

### Runtime fixes applied

- `bot/debt_stop_control.py`
  - `_get_latest_debt_file()` no longer chooses by bracket number.
  - Now prefers fresh `Детальный Дебиторы <manager>` by `mtime`, then falls back to any manager-specific debt JSON.
- `bot/crm_clients.py`
  - `_load_latest_debt_clients()` now prefers the same fresh manager-specific detailed debt family instead of older grouped debt files.
- `bot/inventory_summary.py`
  - `get_latest_inventory_json()` now prefers daily inventory JSON by parsed report period.
  - Prevents newer `inventory_cost_*` or range JSON from masking the actual current day inventory snapshot.
- `bot/send_reports.py`
  - `find_recent_json_for_manager(..., report_type="DEBT")` now prefers fresh detailed debt JSON.
  - `_build_manager_ranking()` now builds debt totals from the latest detailed debt per manager instead of older ledger family files.
  - `__VERSION__` bumped to `v9.4.58/22.04.2026`.

### Evidence and tests

- New focused regression script:
  - `tests/test_report_freshness.py`
  - proves:
    - `FRESH T1` stop-control picks fresh detailed debt
    - `FRESH T2` CRM picks fresh manager debt JSON
    - `FRESH T3` inventory summary picks day JSON, not range/cost artifact
    - `FRESH T4` bot debt selector picks fresh detailed debt
- Existing collector regression additions:
  - `tests/test_collector.py`
  - `DSTOP FILE T1`
  - `DSTOP FILE T2`

Verification run:
- `python -X utf8 tests/test_report_freshness.py`
  - `4/4` passed
- `python -X utf8 tests/test_project.py`
  - `105/105` passed
- `python -X utf8 tests/test_collector.py`
  - long-running suite showed no failures in freshness/collector sections before sandbox timeout; earlier full baseline before this step was green
- `python -m py_compile bot/debt_stop_control.py`
- `python -m py_compile bot/crm_clients.py`
- `python -m py_compile bot/inventory_summary.py`
- `python -m py_compile bot/send_reports.py`
  - all four hit Windows `__pycache__` `PermissionError`, not syntax errors

### Git

- runtime fix commit:
  - `b5fb564` `fix(bot): prefer fresh report sources over stale snapshots`
- pushed to:
  - `origin/fix/log-noise-by-design-markers`

### Current status

- working tree should be clean after pushing this freshness fix and the next optional context commit
- no old Excel files were deleted
- logic now ignores stale families when fresher source-of-truth files exist
- if current `reports/debt_stop_state.json` was built before this fix, it may still contain stale snapshot data until rebuilt by the bot/jobs

### Recommended next step

1. if operators still see old debt-stop rows, rebuild the current daily stop snapshot instead of trusting the old `reports/debt_stop_state.json`
2. monitor the next live cycle and verify that stop-control, bot debt lookups, CRM, and inventory summary all use fresh sources only

## Handoff Update - 2026-04-22 19:35 +05:00

### Manual rebuild of current debt-stop snapshot completed

- User asked for an exact one-line PowerShell command to force rebuild `reports/debt_stop_state.json` without waiting for scheduler.
- Safe path used:
  - bot stopped first
  - backup of the previous state file created
  - `bot.debt_stop_control.save_state(...)` reset only the daily snapshot
  - `bot.debt_stop_control._build_candidates()` rebuilt candidates from current fresh `debt_ext_*.json`
- No Telegram sends were triggered by this rebuild.
- `debt_stop_registry.json` was not modified.

Observed rebuild result:
- `REBUILT candidates=13`
- candidates after rebuild:
  - `А ТД Асем (холодильник № 4)` | `Алена` | `959446.4` | `9`
  - `А ТД Евразия Мунарбек` | `Алена` | `309024.37` | `13`
  - `А ТД Сарыарка 1 ряд 12 место Жулдызбек` | `Алена` | `247031.6` | `11`
  - `А ТД Сарыарка 2 ряд 1 место Ляззат` | `Алена` | `112100.6` | `9`
  - `А Ресторан Tangirs ТОО GrandRest  Ак мешет 1` | `Алена` | `87155.5` | `8`
  - `М Ресторан Шама ИП Тян ул Мустафина 12` | `Магира` | `181297.6` | `21`
  - `Е Еркебулан` | `Ергали` | `767268.67` | `21`
  - `Е ТОО ГудФуд № 1 ул Досмухамедулы 48(Аида)` | `Ергали` | `527927.35` | `9`
  - `Е ИП Шахин` | `Ергали` | `340000.0` | `21`
  - `Е ТД Саянур Леонид` | `Ергали` | `239409.6` | `21`
  - `Е  ИП Алтын орда Косши` | `Ергали` | `199999.75` | `15`
  - `Е ТОО Социальная Столовая ул ул Бейбитшилик 9` | `Ергали` | `117556.65` | `14`
  - `Е ИП Трое Кайрат` | `Ергали` | `58425.0` | `18`

### Important confirmation

- `А Фурманова Евгений (склад № 20)` is not present in the rebuilt candidate list.
- This confirms:
  - stale daily snapshot was replaced
  - old debt `285535.02` is no longer driving the current stop-control list
  - fixed fresh-source selectors + manual rebuild together resolved the live symptom the user reported

### Current operational status

- Code fix already committed and pushed:
  - `b5fb564` `fix(bot): prefer fresh report sources over stale snapshots`
- Context handoff commit already pushed before this update:
  - `aa805c7` `docs(context): save current project handoff`
- After the manual rebuild, the next safe operational step is simply to restart:
  - `python bot/send_reports.py`

### Recommended next step

1. start the bot again on the fixed code
2. monitor the next live stop-control / manager / admin cycle
3. if another client is suspected, compare fresh Excel primary source vs current `debt_stop_state.json` first, not archived state

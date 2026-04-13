# SESSION CONTEXT

Дата последней фиксации: 2026-04-13
Проект: `GPT1C_Processor_analitica`

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

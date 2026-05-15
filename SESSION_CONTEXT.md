# SESSION CONTEXT — АРХИВ

> **ВНИМАНИЕ:** Этот файл содержит исторические сессии (2026-04-09…2026-05-11, 2026-05-12).
> Многие "OPEN" пункты уже закрыты коммитами. Актуальный рабочий набор документации: `AGENTS.md`, `PROJECT_ENCYCLOPEDIA.md`, `SESSION_CONTEXT.md`.

---

## HANDOFF 2026-05-14 — CRM/collector integration safety: clients.json lock + hard-fail propagation

### Что сделано

- `bot/crm_clients.py`
  - добавлен hard-fail lock для `config/clients.json` через `portalocker LOCK_EX|LOCK_NB`
  - `save_clients(data)` теперь возвращает `bool`
  - write-helpers больше не считают запись успешной, если `clients.json` не сохранился:
    - `set_client_phone(...)`
    - `set_client_details(...)`
    - `set_client_alias(...)`
    - `resolve_phone_conflict(...)`
    - `mark_phone_conflict_distinct(...)`
  - `update_from_reports()` при fail-save больше не делает вид, что CRM успешно обновлена

- `collector/manager_dialog.py`
  - `_save_contact(...)` теперь логирует и не считает CRM-update успешным, если `clients.json` не записался

- `bot/send_reports.py`
  - добиты callback/text paths, где `False` от CRM state-save раньше только логировался:
    - post-save phone flow
    - next-client chain
    - claim phone chain
    - address save / address next-client chain
  - при fail-save теперь:
    - rollback in-memory where needed
    - user получает `⚠️ Временная ошибка...`
    - flow не идёт дальше как при успехе

### Проверка

- `python -m py_compile bot\crm_clients.py` → OK
- `python -m py_compile bot\send_reports.py` → OK
- `python -m py_compile collector\manager_dialog.py` → OK
- `python -X utf8 tests\test_crm_regression.py` → `30/30`

### Новые регрессии

- `set_client_details()` возвращает `False`, если save `clients.json` не удался
- `resolve_phone_conflict()` возвращает `False`, если save `clients.json` не удался

### Что это закрывает

- основной межконтурный риск CRM ↔ collector по `clients.json` без lock
- ложные success-path после fail-save в ряде CRM callback/text flows

### Что ещё остаётся

- полный live smoke-test после рестарта бота
- при желании: довести те же строгие save-fail semantics до оставшихся background-only CRM save paths, где сейчас достаточно логирования

---

## HANDOFF 2026-05-14 — стабилизация Collector + CRM (P2/P3 + cleanup)

### Что сделано (master, коммиты `0f412ae`–`4da07ef`)

Закрыты все 5 направлений ТЗ на стабилизацию. Порядок выполнения: 1→3→2→4→5.

#### 1. Dead-code removal (`0f412ae`)

- Удалена `_format_admin_detail_text` (`approval_flow.py:1542`) — определена, никем не вызывалась.
- F-10 в `AUDIT_COLLECTOR_2026-05-14.md` помечен OBSOLETE.
- Тесты: 18/18 `test_collector_regression_hermetic`.

#### 2. Lock + atomic discipline для CRM state-файлов (`605edc0`)

Добавлен `_crm_state_lock(path)` — параметризованный context-manager по образцу `collector/payment_hold._hold_lock`. `LOCK_EX|LOCK_NB + retry-loop` (единственный режим с рабочим timeout на Windows).

Покрыты 4 пары save/load:
- `_crm_save_pending` / `_crm_load_pending` (phone-pending)
- `_crm_save_claim_pending` / `_crm_load_claim_pending` (claim queue)
- `_crmdup_save_pending` / `_crmdup_load_pending` (dup review)
- `_crmdup_save_ambiguous` / `_crmdup_load_ambiguous` (ambiguous conflicts)

⚠️ **Важный нюанс**: lock — best-effort. При недоступности (`LockException/PermissionError/OSError`) код логирует warning и продолжает без exclusive lock (`yield` в except-ветке, `bot/send_reports.py:6650`). Жёсткой гарантии отсутствия lost update при реальном contention нет — но дисциплина внедрена и покрывает нормальный путь.

#### 3. Stale CRM claim tokens (`72f4141`)

- F-16: `_crm_cleanup_claim_pending` — токены без `created_at` ранее `continue`-лись (никогда не удалялись). Исправлено: вынесен helper `_crm_claim_is_stale(claim)`, stale = без created_at | невалидная дата | TTL истёк.
- Callback `crm_claim|...`: явная TTL-проверка на конкретный токен до обработки → если stale, удалить, снять кнопку, ответить «запрос устарел».

#### 4. Dup-review / ambiguous хвосты (`7c6be03`)

**F-12** (deterministic keep-key):
- В custom-phone path: `client_keys[0]` → `sorted([k for k in client_keys if k])[0]`.
- Результат не зависит от порядка items в review при пересборке.

**F-13** (ambiguous reopen):
- `_ambiguous_signature` теперь чувствителен к телефонам: `sorted("client_key#normalized_phone")` вместо `sorted(client_keys)`.
- Если у тех же пар изменились phones → новый signature → новая pending-запись создаётся автоматически (reopen). Старая resolved остаётся как audit-trail.

#### 5. Admin CRM backlog (`579e4a5`)

Новый экран `📋 CRM бэклог` в admin-меню (callback `crm_backlog`):
- 4 категории с counts + top 5 кейсов и timestamp-возраст (м/ч/д).
- 📞 Phone pending, ✋ Claim pending (+N stale), 🔀 Duplicate review, ❓ Ambiguous (+N resolved в истории).
- Кнопки: Обновить, Ambiguous-очередь, Главное меню.
- `_format_crm_backlog_text()` + `_crm_backlog_keyboard()`.

#### 6. Регрессионные тесты (`4da07ef`)

Добавлен `StabilizationRegressionTests` (12 тестов) в `test_crm_regression.py`:
- stale claim: без created_at / expired / fresh / cleanup
- lock: context-manager создаёт lock-file; save+load round-trip
- F-12: deterministic keep-key order-independent
- F-13: signature reacts to phone-change; order-independent; reopen via new sig
- backlog: counts по 4 очередям; stale claims отдельной подписью

### Финальный прогон (по критериям ТЗ)

```
py_compile bot/send_reports.py, crm_clients.py, crm_audit_log.py  ✅
test_crm_regression:              26/26  ✅  (+12 новых)
test_collector_regression_hermetic: 18/18  ✅
test_project:                     110/110 ✅
```

### Что осталось открытым

| Finding | Суть |
|---|---|
| dead `_format_admin_detail_text` | ✅ удалён |
| F-11 (lock) | ✅ закрыт — best-effort, fallback без exclusive lock при LockException |
| F-12 (keep-key) | ✅ закрыт |
| F-13 (ambiguous reopen) | ✅ закрыт |
| F-14 (stale claim TTL) | ✅ закрыт |
| F-16 (claim без created_at) | ✅ закрыт |
| Admin CRM backlog | ✅ закрыт |
| Design | Три несогласованных хранилища: clients.json / debtors_contacts.json / batch snapshot — отложено |

---

## HANDOFF 2026-05-14 — live-инцидент: «Договорились» + потонувший превью

### Что сделано (master, коммит `7603db9`)

Разбор живого инцидента: Магира и Оксана жалуются, что бот помечает их «не ответил», хотя они отвечали.

#### Root cause

**Магира** — нажала кнопку «Договорились», бот перешёл в `waiting_for_agreed`. Магира не написала детали. Прошёл timeout → `status = "timeout"`, но `waiting_for_agreed` не был очищен. Бот показывал "🔇 не ответил" — некорректно, она _начала_ отвечать.

**Оксана** — ответила на вчерашний батч. Сегодняшний превью (msg_id=19128) был засыпан потоком сообщений: WA-уведомление (17:00), три "не видит оплату" от Саиды (17:30), CRM phone-запрос (18:00). Превью физически потерялось в истории чата.

#### Исправлено (`approval_flow.py`)

`_format_admin_summary_text` (`approval_flow.py:1416`): ветка `status == "timeout"` теперь проверяет `waiting_for_agreed` и `waiting_for_proof`:

- `waiting_for_agreed` → `"⏳ начал — не написал детали по «Клиент X» → авто (N кл.)"`
- `waiting_for_proof`  → `"⏳ начал — не прислал документ по «Клиент X» → авто (N кл.)"`
- иначе (полное молчание) → `"🔇 не ответил → авто (N кл.)"`

Обе ветки покрыты одной правкой — `waiting_for_proof` имел идентичный баг.

#### Обновлена инструкция (`docs/Инструкция по работе с ботом.html` → v9.4.67)

- После кнопки «🤝 Договорились»: красный блок 🚨 "ОБЯЗАТЕЛЬНО напишите ответ сразу после нажатия"
- После кнопки «💰 Оплатил»: красный блок 🚨 — либо фото, либо «Без документа»
- Новый раздел "Бот не понял": «Нажал кнопку — не написал детали» + «Не нашёл сообщение — засыпало другими»

#### Почему не поймал аудит

Оба случая — UX-проблемы, не code bugs:
- `waiting_for_agreed` → неверный label — аудит не проверял правильность _описания_ состояния в admin UI, только корректность state-machine
- Message flood — операционный сценарий, не код; виден только в live-данных с реальными пользователями

### Что остаётся открытым

Те же P2/P3 из CRM-аудита:

| Finding | Суть |
|---|---|
| F-11 | Нет portalocker на CRM state-файлах |
| F-12 | Custom phone в dup-review фиксирует keep_key=client_keys[0] произвольно |
| F-13 | Resolved ambiguous-конфликт не переоткрывается при повторном возникновении |
| F-14 | Stale claim-кнопки не убираются после TTL |
| F-16 | Claim-токен без created_at никогда не вычищается |
| Design | Три несогласованных хранилища контактов: clients.json / debtors_contacts.json / batch snapshot |

---

## HANDOFF 2026-05-14 — CRM-аудит + исправление P0/P1 багов

### Что сделано (master, коммиты `cc2004c`, `fe50b96`)

Проведён полный технический аудит CRM-контура согласно ТЗ.  
Полный отчёт: `AUDIT_CRM_BOT_2026-05-14.md` (в корне проекта).  
ТЗ: `AUDIT_CRM_BOT_TZ_2026-05-14.md`.

**Тесты: 631/631 + 14/14 CRM regression.**

#### Исправлено (коммиты `cc2004c`, `fe50b96`)

| Finding | Файл | Что исправлено |
|---|---|---|
| F-02 | `crm_clients.py:968` | `load_contacts_compat` раскрывает aliases → alias-клиенты видны коллектору |
| F-03 | `crm_clients.py:324` | `_merge_client_entries` переносит `whatsapp`/`phone`/`telegram_id` при merge |
| F-01 | `crm_clients.py:906` | `set_client_details` использует `_find_existing_client_key` вместо прямого get |
| F-04 | `send_reports.py:7048` | При `ok=False` pending остаётся в очереди, цепочка не движется |
| F-05 | `approval_penalty.py:345` | `paused_until` проверяется в `check_crm_ignores` → нет штрафа за "Позже" |
| F-06 | `send_reports.py:6866` | `_crm_collect_unowned_claim_clients` режет `is_vendor`/`do_not_call` |
| F-07 | `send_reports.py:8695` | `_crm_write_ok` gate: success-path (уведомления, phone-chain) только при успешном CRM write; rollback при ошибке |
| F-08 | `send_reports.py:6461` | TTL phone-pending от `created_at`, не `last_sent` |
| F-09 | `send_reports.py:6339` | `_crm_key_token` в callback_data CRM-кнопок — stale кнопка отклоняется |

### Верификация F-07

Первый коммит (`cc2004c`) сделал rollback state в except, но success-path продолжался безусловно — менеджеры получали ложное "✅ Взяли!" даже при ошибке CRM write. Второй коммит (`fe50b96`) добавил `_crm_write_ok` флаг — весь success-path выполняется только при подтверждённой записи в CRM.

### Ключевые ответы по аудиту

**Может ли бот записать телефон не в ту карточку?**  
→ Раньше да (F-01 тихий fail, F-09 stale callback). Оба исправлены.

**Теряется ли телефон при merge?**  
→ Раньше да (F-03). Исправлено.

**Alias-клиенты видны коллектору?**  
→ Раньше нет (F-02). Исправлено.

**Ложный штраф при "Позже"?**  
→ Раньше да (F-05). Исправлено.

### Что осталось открытым (не блокирует runtime)

| Finding | Суть |
|---|---|
| F-11 | Нет portalocker на CRM state-файлах (в отличие от collections_db) |
| F-12 | Custom phone в dup-review фиксирует keep_key=client_keys[0] произвольно |
| F-13 | Resolved ambiguous-конфликт не переоткрывается при повторном возникновении |
| F-14 | Stale claim-кнопки не убираются после TTL |
| F-16 | Claim-токен без created_at никогда не вычищается |
| UI | Admin не видит CRM backlog в реальном времени |
| Design | Три несогласованных хранилища контактов: clients.json / debtors_contacts.json / batch snapshot |

### Перезапуск бота нужен для активации всех фиксов в production

---

## HANDOFF 2026-05-14 — добавлено ТЗ на полный аудит CRM-бота

### Что сделано

- Создан новый документ: `AUDIT_CRM_BOT_TZ_2026-05-14.md`
- Это не результаты аудита, а именно подробное ТЗ на будущий комплексный CRM audit pass

### Что зафиксировано в ТЗ

- полный объект проверки:
  - `bot/crm_clients.py`
  - `bot/crm_audit_log.py`
  - CRM-части `bot/send_reports.py`
  - `config/clients.json`
  - `contacts.xlsx`
  - CRM state/log files
  - CRM↔collector integration
- обязательные направления:
  - phone pending flow
  - claim-flow
  - duplicate / ambiguous conflict review
  - reminder / escalation / penalty logic
  - ACL
  - state integrity / concurrency
  - UI / observability
  - test coverage gap analysis
- формат итогового CRM-аудита:
  - executive summary
  - architecture map
  - findings with evidence
  - separate issue registries
  - remediation plan P0/P1/P2/P3

### Важно

- Код продукта не менялся
- Тесты не запускались: правка только документационная
- Документ опирается на текущий HEAD `71ae45c` и текущее устройство CRM-контура, а не на старые аудиты как источник истины

---

## HANDOFF 2026-05-14 (финал) — технический долг закрыт, коллектор в боевом строю

### Что сделано (master, коммиты `e6307bb`, `1cb789e`)

Закрыты три оставшихся инженерных замечания после основного аудита.

#### portalocker timeout fix (коммит `e6307bb`)

- `LOCK_EX` (blocking mode) игнорирует `timeout=` на Windows → graceful fallback не работал
- Заменено на `LOCK_EX | LOCK_NB` + `check_interval=0.1, timeout=5` — retry-loop с реальным таймаутом
- `except` расширен: `(LockException, PermissionError, OSError)` — WinError 5 теперь перехватывается

#### WinError 5 + skip_summary UI (коммит `1cb789e`)

**WinError 5 root cause:**
- `_LOCK_FILE` был frozen module-level константой → при redirect `PAYMENT_HOLD_PATH` в тестах lock-файл создавался в реальном `logs/` → WinError 5 при cleanup temp-dir
- Исправлено: `lock_file = PAYMENT_HOLD_PATH.with_suffix(".lock")` вычисляется динамически в момент вызова → следует за redirect автоматически

**skip_summary UI:**
- `run_approval_preview()` собирает `_skipped[{name, reason}]` на каждом `continue`-пути
- `batch["skip_summary"] = _skipped[:20]` сохраняется в структуру батча
- `_format_collector_batch_text()` показывает блок "Пропущено (N):" с причинами — директор видит куда исчезли остальные клиенты

### Итоговое состояние коллектора

**Тесты: 631/631.**

- Все P0/P1 из аудита закрыты: `bb33f46`, `9af1127`
- portalocker timeout: `e6307bb`
- WinError 5 + skip_summary: `1cb789e`
- Формулировка: **коллектор в боевом строю по критерию P0/P1**
- Нет открытых блокирующих или high-severity замечаний

### Что остаётся (не блокирует runtime)

- `wa_appr_adm_later` supersede: warning в лог добавлен, TG-уведомление директору при вытеснении — не реализовано
- Zeropay flow не прогонялся в бою (тесты есть, боевого прогона нет)
- **Перезапуск бота** нужен для активации всех фиксов в production

---

## HANDOFF 2026-05-14 — Комплексный аудит collector + исправление всех найденных багов

### Что сделано (master, коммиты `bb33f46`, `9af1127`)

Проведён полный технический аудит контура `collector/` согласно ТЗ.
Полный отчёт: `AUDIT_COLLECTOR_2026-05-14.md` (в корне проекта).
Верификация findings: раздел 1.1 в том же файле.

**Тесты: 620 → 631/631.**

#### P0/P1 фиксы (коммит `bb33f46`)

| Finding | Файл:строка | Что исправлено |
|---|---|---|
| F-06 | `send_reports.py:3248` | `sent_ok` теперь считает `status=="sent"` — UI показывал `0/N` |
| F-05 | `payment_hold.py:307` | `pending_saida` блокирует collector shortlist; TTL только для confirmed |
| F-03 | `payment_hold.py` | `portalocker` (`_hold_lock`) на write-функции: race condition устранён |
| F-01 | `collections_engine.py:360` | exception без anchor date → `anchor = today`, grace period гарантирован |
| F-08 | `client_dialog.py:963` | `off_topic_count = 0` при любом продуктивном intent |
| F-02 | `approval_flow.py:2791` | merge order fix: `_load_batches() | {batch_id: batch}` |

#### P2/P3 фиксы (коммит `9af1127`)

| Finding | Файл | Что исправлено |
|---|---|---|
| F-10 | `approval_flow.py` | visual ghost: agreed/paid → иконки 🤝💰 вместо ◯ |
| F-11 | `approval_flow.py` | убран `"sent"` из `is_ready_for_send` (CLI re-send protection) |
| F-09 | `approval_flow.py` + `collections_engine.py` | warning при supersede отложенного батча; текст с предупреждением |
| F-12 | `collections_engine.py` | `contact = None` в начале каждой итерации loop |
| F-07 | `collections_engine.py` | `awaiting_payment_proof` TTL 3 дня — guard снимается автоматически |
| F-13 | `collections_engine.py` | warning при нет chat_id в send-approved refresh |
| F-15 | `client_dialog.py` | повторный greeting → escalate вместо бесконечного ответа |
| F-16 | `client_dialog.py` | `promise_without_date` + `exchange_count >= 2` → escalate |
| F-17 | `client_dialog.py` | мёртвые ё-паттерны в `_SERVICE_REQUEST_PATTERNS` удалены |
| F-14 | `payment_deferrals.py` | mtime-based cache invalidation |
| Low | `send_reports.py` | `_get_pending_admin_batch` читает JSON один раз вместо двух |
| F-04 | `tests/test_collector.py` | секция 29: 7 тестов zeropay + 4 regression для F-01/F-05 |

### Ключевые ответы на оперативные вопросы

**Почему клиенты повторно лезут в WA:**
- `pending_saida` не блокировал shortlist — исправлено (F-05)
- `exception` без anchor date — grace не считался — исправлено (F-01)
- `awaiting_payment_proof` зависал вечно — теперь TTL 3 дня (F-07)

**Куда исчезли "остальные клиенты" 13.05:**
- 7 — active dialog state (штатно)
- 2 — Саида подтвердила hold (штатно)
- 1 — малый остаток после оплаты (штатно)
- Причины skip в `collector.log`, в UI не отображаются (operational gap)

**Врёт ли UI:**
- Да, `0/N доставлено` — исправлено (F-06)

### Что осталось открытым

- Zeropay тесты написаны (F-04 закрыт), но zeropay flow не тестировался в бою
- `wa_appr_adm_later` supersede: предупреждение добавлено, но уведомление в TG директору при вытеснении — не реализовано (отложено)
- UI не показывает пропущенных клиентов в Telegram — operational gap, требует архитектурного решения (`skip_summary` в batch)
- Перезапуск бота нужен для активации всех фиксов в production

### Проверка

```
python -m py_compile collector/approval_flow.py   → OK
python -m py_compile collector/collections_engine.py → OK
python -m py_compile collector/client_dialog.py   → OK
python -m py_compile collector/payment_deferrals.py → OK
python -m py_compile collector/payment_hold.py    → OK
python -m py_compile bot/send_reports.py          → OK
python -X utf8 tests/test_collector.py            → 631/631
```

---

## HANDOFF 2026-05-13 — approval_penalty partial-timeout fix (Алена)

### Что закрыто

- Исправлена классификация partial WA-пропуска в `collector/approval_penalty.py`.
- Раньше `timeout` считался `partial` только если был `approved_names`.
- Из-за этого реальные частичные ответы менеджера через:
  - `waiting_for_agreed`
  - `paid_no_doc_names`
  - другие non-approve decision ветки
  ошибочно попадали в `full`, а не `partial`.

### Подтверждённый боевой симптом

- `logs/approval_penalty_state.json` по Алене содержал:
  - `20260511-170000-078e` → `type=full`, `penalty=0`
  - `20260512-170000-15b9` → `type=full`, `penalty=2000`
- Но в `logs/wa_approval_batches.json` у Алены были реальные следы ответа:
  - 11.05 → `waiting_for_agreed`
  - 12.05 → `paid_no_doc_names`

### Изменения

- `collector/approval_penalty.py`
  - `timeout` теперь считается `partial`, если есть любые признаки начатого manager-review:
    - decision lists
    - `waiting_for_proof`
    - `waiting_for_agreed`
    - `responded_at`
- `tests/test_collector.py`
  - добавлены регрессии:
    - `timeout + waiting_for_agreed -> partial`
    - `timeout + paid_no_doc_names -> partial`
    - `timeout без действий -> full`

### Проверка

- `python -m py_compile collector/approval_penalty.py` -> OK
- `python -X utf8 tests/test_collector.py` -> `587/587`

### Важно

- Это исправляет будущие начисления.
- Текущее содержимое `logs/approval_penalty_state.json` не пересчитывается автоматически и при необходимости требует отдельной операционной корректировки.

### Операционная корректировка

- По прямому решению пользователя state штрафов был сброшен вручную:
  - `logs/approval_penalty_state.json` -> `{ "month": "2026-05", "managers": {} }`
  - бэкап сохранён: `logs/approval_penalty_state.json.bak-20260513-reset`

### Follow-up code fix (same day)

- Закрыты сопутствующие баги approval/preview для всех менеджеров:
  - `collector/approval_flow.py`
    - manager preview показывает реальный дедлайн ответа: `min(created_at + 1h, expires_at)`
    - stale manager callback снимает inline-кнопки (`reply_markup=[]`)
    - `paid_no_doc` создаёт payment hold только внутри реально применённого callback
  - `bot/send_reports.py`
    - удалён побочный `payment_hold` после любого `handled=True` по `wa_appr_cli_paid_nodoc`
    - раньше это могло сработать даже после ответа `запрос уже неактуален`
  - `collector/collections_engine.py`
    - client with active/escalated `collector_client_dialogs.json` больше не попадает
      в новый manager preview на следующий день

### Проверка

- `python -m py_compile collector/approval_flow.py` -> OK
- `python -m py_compile collector/collections_engine.py` -> OK
- `python -m py_compile bot/send_reports.py` -> OK
- `python -m py_compile collector/approval_penalty.py` -> OK
- `python -X utf8 tests/test_collector.py` -> `588/588`

---

## HANDOFF 2026-05-12 — WA-диалог: платёжные слова, AI на unclear, _is_service_request

### Что сделано (master, коммит 8baa1ba)

**Тесты: 563/563.**

**Root cause инцидента Нурлан** («Ергали скинули / Расчет» → эскалация):
- «расчет» → `_is_service_request` ловил как подстроку «счет» → doc_request handler → эскалация без AI
- «скинули/расчет» — не в списке платёжных слов → AI возвращал `unclear` → эскалация

| Что | Фикс |
|-----|------|
| `_is_service_request` | regex `\b` вместо substring; «расчет»≠«счет», «факт»≠«акт» |
| `paid_claim` слова | скинул/закинул/перечислил/расчитался/оплатил/расчет/на каспи |
| `soft_positive` слова | скину/закину/перекину/отправлю/закроем |
| `unclear` + AI | `requires_human=False`; AI задаёт 1 вопрос; 2й unclear → ответ + эскалация |

### Что осталось открытым

- Живой тест WA-диалога с новыми платёжными словами
- Тесты `_handle_saida_zeropay_confirm/deny`
- exception stop_status: клиент снятый со стопа тут же появляется в WA-батче — возможно нужен grace period

---

## HANDOFF 2026-05-12 (поздний вечер) — отсрочки, дубли WA, баги

### Что сделано (master, коммиты 1a6622f…2d8516d)

| Коммит | Что |
|--------|-----|
| `1a6622f` | fix(dialog): дубли WA — guard по state диалога в start_client_dialog + collections_engine |
| `4b8b3fa` | invalid_phone:name_mismatch убран; show_stats HTML с html.escape |
| `676dfc1` | /batch + collector_resend_approval + show_stats fix |
| `657f47a` | feat(deferrals): payment_deferrals.py + config; 9 клиентов Оксаны |
| `50d6a7a` | добавлен Navat Азия Парк |
| `2d8516d` | deferral_level — отдельная шкала: eff=2→L1, eff=5→L3, eff=10→L5 |

### Логика отсрочки (для ВСЕХ клиентов в deferrals.json)

Для 10-дневной отсрочки: WA с 12-го дня (eff=2), настойчиво с 15-го (eff=5), критично с 20-го (eff=10).
Для 7-дневной (Румакс): аналогично, отсчёт от 7-го дня.

### Что осталось открытым

- Статистика нарушений фин. дисциплины по клиентам с отсрочкой
- Списки отсрочек от Алены / Ергали / Магиры (когда дадут)
- exception stop_status → grace period
- Перезапуск бота для активации всех фиксов

---

## HANDOFF 2026-05-12 (вечер) — /batch, collector_resend_approval, show_stats Markdown

### Что сделано (master, коммит 676dfc1)

**Тесты: 563/563.**

| Что | Фикс |
|-----|------|
| `/batch` команда | если pending_admin → сразу сводка с кнопками; иначе → экран коллектора |
| `collector_resend_approval` | кнопка «📋 Утвердить рассылку» в меню 🤖 Коллектор при pending_admin батче |
| `show_stats` Markdown | parse_mode Markdown → None; username с `_` ломал HTML-парсер |

### HANDOFF 2026-05-12 (ночь) — invalid_phone:name_mismatch + show_stats HTML + /batch + штрафы сброшены

| Коммит | Что |
|--------|-----|
| `676dfc1` | /batch + collector_resend_approval + show_stats Markdown→None |
| `4b8b3fa` | invalid_phone:name_mismatch убран; show_stats HTML с html.escape |

**approval_penalty_state.json сброшен вручную** — 12.05.2026 20:30.
Бэкап: `logs/approval_penalty_state.json.bak-20260512-reset`.
Учёт штрафов начинается заново с 13.05.2026 (следующий батч).

### Наблюдение (не баг)

А ТД Сарыарка СКЛАД с `stop_status="exception"` попал в батч корректно — collections_engine обрабатывает exception как обычного должника. Возможно нужен grace period.

---

## HANDOFF 2026-05-11 — forensic батчей, soft_positive, стоп-лист, штрафы, CRM-префикс, CRM-игноры

### Что сделано (master, коммиты cba6fdf…b1e3961)

**Тесты: 531/531.**

| Коммит | Что |
|--------|-----|
| `cba6fdf` | `close_reason` + `setdefault(escalation_reason)` в `approval_flow.py` — forensic observability |
| `eabb62d` | `_normalize_text` fix (`"".join`); `_is_greeting_only` без дубля; `_PURE_GREETINGS` 25+ вариантов |
| `...` | soft_positive: guard ветки игнорируют `suggested_reply`, hardcoded safe text |
| `...` | `client_dialog.py` v1.1.5: `_is_acknowledgement_only` в soft_positive |
| `...` | `collection_agent.py` v1.1.1: soft_positive требует платёжное слово |
| `...` | `debt_stop_control.py` v1.0.15: threshold 100 тг; нормализация; Саида→Админ цепочка |
| `...` | `approval_penalty.py` v1.0.2: штрафы WA; penalty_check 30мин; monthly 23:00 |
| `a6d0b52` | тесты 441→491: greeting/ack/E2E/penalty |
| `d66560a` | CRM-префикс: skip clarify_name, auto-assign unowned |
| `b7ba36b` | fix: display_name = client_key[2:] — префикс не в CRM |
| `b1e3961` | `approval_penalty.py` v1.1.0: `check_crm_ignores`; source="wa"\|"crm"; CRM-текст уведомлений |
| `4a5eac4` | fix(penalty): CRM ignore timer → `created_at` (не last_sent); workday guard 09–19 |
| `b205e1d` | fix(crm): `_normalize_system_display_name` в `set_client_details` — центральная защита от префикса |
| `6bf3412` | feat(collector): sticky approval v1.5.3 — unchanged no-movement clients пропускают manager preview |

### Ключевые решения

- `close_reason` + `setdefault(escalation_reason)` — оба поля для forensic
- Threshold стопа: 100 тг; Саида подтверждает ПЕРЕД admin-меню
- Формула штрафов: 1й=0, 2й=2000, N≥3: N×1000×2, partial -10%; WA и CRM в едином счётчике
- CRM-префикс: display_name = key[2:] в pending + `_normalize_system_display_name` в crm_clients.py как последний рубеж
- CRM_IGNORE_MIN_AGE_HOURS=22, возраст от `created_at` (не last_sent), guard 09–19 рабочие дни
- Sticky approval: если signature (amount/credit/debit/msg_type/stop_status) не изменилась → auto-send без переспрашивания менеджера
- config/clients.json: 151 системный display_name очищен от префикса (в .gitignore, не версионируется)

### Что осталось открытым

- Тесты для `_handle_saida_zeropay_confirm/deny`
- Боевой прогон: penalty_check, CRM-игноры, sticky approval
- Тесты: 533/533

---

## HANDOFF 2026-05-09 — аудит state-machine + batch expiry + notify_state cleanup

### Что сделано (master, коммиты 253c928…34462b4)

**Тесты: 441/441.**

| Коммит | Что |
|--------|-----|
| `253c928` | Текст Саиды использует f-string с `SAIDA_WARN_HOURS`/`SAIDA_BYPASS_HOURS`; `approval_flow >= expires_at` (граница cutoff) |
| `ebb0b80` | Тесты 8b/8c: минутная граница cutoff, cross-module; docstring `communications.py`; `.gitignore site/` |
| `4339d72` | autoagent: путь E:→C: |
| `f8b3777` | `get_latest_send_ready_batch`: expired `admin_approved` не возвращается; `expire_old_batches`: `pending_admin`/`admin_approved` → `too_late`; `too_late` в `final_statuses`; `log_monitor_task` вызывает очистку каждые 2ч включая выходные; тесты 23c (8 кейсов) |
| `34462b4` | `new_reports_notifier`: merge с pruning по `existing_paths` — мёртвые E:/F: пути удаляются; тесты 23d (9 кейсов); разовая очистка 6034→2995 записей |
| `(local)` | `payment_hold.py`: SAIDA_WARN_HOURS default 4→1, SAIDA_BYPASS_HOURS 8→2; `debt_stop_control.py` v1.0.14: `load_state()` логирует сброс с unresolved count |

### Аудит state-файлов 2026-05-09

- **Очищены оба зависших батча** вручную через `expire_old_batches()`: `20260507-170001-2709` и `20260508-170000-aea6` → `too_late`
- **notify_state.json**: 6034→2995 (E: 1959, F: 242, other 838 удалены)
- **Всё чисто**: saida_payment_holds все `rejected`/`cleared_by_1c`, crm_pending пусто, deletion_queue в будущем

### Живые бизнес-состояния (не мусор)

- `wa_agreed_promises.json`: 1 обещание deadline=2026-05-09, статус `accepted` → понедельник: коллектор проверит
- `debt_stop_registry.json`: 2 активных stopped → штатно
- `debt_stop_saida_known.json`: обновится в 22:00 ближайшего рабочего дня

### Следующий безопасный шаг

Понедельник 17:00 — новый коллекторский прогон. Проверить логи:
1. `collector_daily_start` → батч создан
2. Менеджеры ответили (или таймаут → `pending_admin`)
3. Батч утверждён и отправлен до 19:30
4. Обещание из 2026-05-09 помечено `promise_broken` если не оплачено

---

## OPERATIONAL NOTE 2026-05-09 — почему в Process List видно два `python.exe` при одном боте

Проверено локально на `C:\GPT1C_Processor_analitica`:

- в Process List одновременно видны:
  - `C:\GPT1C_Processor_analitica\.venv\Scripts\python.exe`
  - `C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe`
- оба процесса стартовали в одно время, но это не оказалось самостоятельным дублированным запуском scheduler-а
- `ParentProcessId` у системного `Python311\python.exe` указывал на `.venv`-процесс
- родитель `.venv` был запущен через `cmd.exe /c ""C:\GPT1C_Processor_analitica\start_bot_watchdog.bat""`
- `logs/bot.pid` принадлежал дочернему системному `Python311\python.exe`, то есть именно он был рабочим интерпретатором

Вывод:

- на этой Windows-машине `.venv\Scripts\python.exe` ведёт себя как launcher/redirector
- тяжёлый рабочий процесс бота живёт как дочерний `Python311\python.exe`
- поэтому **два `python.exe` в диспетчере задач не являются достаточным признаком второго экземпляра бота**

Как отличать норму от реального дубля:

- норма:
  - parent `.venv\Scripts\python.exe`
  - child `Python311\python.exe`
  - один `bot_starting` в логе на старт
  - один watchdog-parent
- реальный дубль:
  - два независимых `bot_starting`
  - два независимых родителя процесса
  - конфликты `bot.pid`
  - дублирующиеся APScheduler job runs / двойные уведомления

Связанное наблюдение:

- stale `notify_state.json` на 09.05.2026 объяснялся не «вторым ботом», а отдельным operational-state хвостом; файл был очищен вручную до 659 живых записей без temp-path мусора

---

## HANDOFF 2026-05-06 (финал) — WA approval полный цикл + SLA Саиды + аналитика

### Что сделано (ветка `fix/log-noise-by-design-markers`, HEAD `09f6531`)

**12 коммитов за день. Тесты: 330/330.**

| Коммит | Что |
|--------|-----|
| `6f64d4d` | Таймаут менеджеров + дедлайн в превью + cutoff 19:30 |
| `4d81ce7` | Немедленная эскалация при узком окне + честный expires_at |
| `aa552ba` | Немедленная проверка при создании батча |
| `0fbc8a4` | Минутная точность дедлайна директора |
| `2a0ce6a` | SLA Саиды: предупреждение 4ч + байпас 8ч |
| `7a5ec3b` | Расписание: 16:30 стоп-лист, 18:30 эскалация |
| `f83cb8e` | Кнопки: ✅/💰/🤝 — без причины убрать нельзя |
| `c6f01d5` | Авто-возврат при сорванном обещании |
| `285db0a` | Б-lite: директор проверяет договорённости |
| `202e677` | Auto-clear стопа после полной оплаты Саиды |
| `be6dac3` | Аналитика качества обещаний менеджеров |
| `09f6531` | Саида backlog analytics |

### Следующий безопасный шаг

- Боевой тест при поступлении дебиторки от Саиды (event-driven)
- Операционная задача: разобрать backlog Саиды (51 запрос, старейший 9 дней)
- Боевой тест новой approval flow при поступлении дебиторки
- Операционно: разобрать backlog Саиды (51 запрос)

---

## HANDOFF 2026-05-06 — Таймаут менеджеров + cutoff 19:00 для WA-рассылки

### Что сделано (ветка `fix/log-noise-by-design-markers`, коммит `6f64d4d`)

**Задача:** устранить структурный блокер: WhatsApp-уведомления не уходили должникам, потому что
батч создавался в 17:00, менеджеры не реагировали, а `collector_reminders` закрывался в 18:00 —
ровно в момент когда должен был сработать 1-часовой таймаут.

**Root cause:**
- `collector_reminders` окно: `9 <= hour < 18` — при батче созданном в 17:00 таймаут бил в 18:xx,
  но джоб уже не запускался
- При `timeout`-статусе менеджера `_build_admin_decisions` отдавал клиентов в `skip` (пустой
  `approved_names`), а не в авто-включение
- Нет жёсткого cutoff: батч висел до 02:00 ночи без пользы
- Менеджеры не видели дедлайн в превью — не понимали что молчание что-то означает

**Фикс в коде:**

- `collector/approval_flow.py`:
  - новая константа `SEND_WINDOW_CUTOFF_HOUR = 19` (env `WA_SEND_WINDOW_CUTOFF_HOUR`)
  - `_format_manager_preview_text`: добавлена строка дедлайна
    `"⏰ Ответьте до HH:MM. Если не успеете — уведомления уйдут автоматически."`
  - `_build_admin_decisions`: `status == "timeout"` → клиенты в `keep` (авто-включение),
    кроме явно отклонённых
  - `_format_admin_summary_text`: для timeout-менеджеров статус `🔇 не ответил → авто (N кл.)`,
    список авто-клиентов в сводке; `total_ok` учитывает авто-включённых
  - `promote_silent_batches_to_admin`: после 19:00 → статус `too_late`, уведомление админу
    "Сегодня не состоится. Следующий батч — при поступлении новой дебиторки."

- `bot/send_reports.py`:
  - `collector_reminders` окно расширено: `< 18` → `< 19`

**Доказательства:**
- `python -m py_compile collector/approval_flow.py bot/send_reports.py` → OK
- `python -X utf8 tests/test_collector.py` (с `WHATSAPP_ENABLED=0; LIVE_SEND_ALLOWED=0`) → 320/320 OK

### Что не трогать

- `autoagent/orchestrator_agents.json`, `autoagent/task_prompt.txt` — пользовательские/служебные
- `audit/` черновики
- `bot/debt_stop_control.py` — изменён в предыдущей сессии, не трогать

### Следующий безопасный шаг

Сегодня во второй половине дня Саида скинет свежую дебиторку.
Проверить в `logs/collector.log`:
1. создался батч при поступлении debt_ext файлов
2. менеджеры получили превью с дедлайном
3. через 1 час — сводка ушла директору с `🔇 Авто` статусами молчавших
4. после нажатия "Утвердить" директором — сообщения ушли в WhatsApp до 19:00

---

## HANDOFF 2026-04-30 — CRM duplicate phone review + cleanup

### Что сделано (ветка `fix/log-noise-by-design-markers`)

**Задача:** убрать CRM phone-queue шум от legacy-дублей, исключить служебные строки из phone queue, провести разовую сверку конфликтных телефонов через живого Telegram-бота и зафиксировать выбор менеджеров.

**Root cause:**
- `get_clients_without_phones()` поднимал legacy-дубли отдельно, даже если sibling-карточка уже имела телефон
- loose duplicate matching был недостаточно управляем для CRM cleanup
- служебные строки (`... под ЗП`, `Недостача`, `Без клиента`, `Водитель ...`) попадали в CRM-очереди как обычные клиенты
- в `config/clients.json` накопились runtime-дубли: часть с пустым sibling, часть с разными телефонами

**Фикс в коде:**
- `bot/crm_clients.py` v1.1.0
  - `canonicalize_client_key_loose()` ограничен legacy-паттерном хвостового дубля
  - `is_service_client_name()` централизует фильтр служебных строк
  - merge legacy duplicate -> preferred card с сохранением `aliases`
  - `get_clients_without_phones()` пропускает legacy-дубли, если sibling уже имеет контакт
  - добавлены `get_phone_conflict_groups()`, `resolve_phone_conflict()`, `mark_phone_conflict_distinct()`
- `bot/send_reports.py` v9.4.63/30.04.2026
  - добавлен разовый manager-review flow по конфликтным телефонам через Telegram inline-кнопки
  - persisted state: `logs/crm_duplicate_review_state.json`
  - admin command: `/crmdupsend`
  - обработка custom phone text и варианта `это разные клиенты`
- `tests/test_crm_regression.py`
  - регрессии на legacy sibling skip
  - регрессии на service-row filtering
  - защита от ложного merge реальных адресов
  - conflict-review API tests

**Live-операция:**
- очищены 6 safe data-дублей в локальном `config/clients.json` там, где был empty sibling при наличии карточки с телефоном
- создан backup: `config/clients.json.bak-20260430-crm-dedup`
- через живой бот разослано 13 review-case менеджерам по конфликтным дублям с разными телефонами
- все 13 кейсов закрыты менеджерами в тот же день
- выборы зафиксированы в CRM, состояние и audit сохранены в:
  - `logs/crm_duplicate_review_state.json`
  - `logs/crm_audit.jsonl`

**Итог после manager review и cleanup:**
- конфликтных duplicate groups с разными телефонами: `0`
- safe duplicate groups с одинаковым телефоном: `17` -> авто-схлопнуты локальным data cleanup
- ambiguous/no-phone duplicate groups: `1` -> закрыта вручную
  - оставлена карточка `М Плов центр ЕСБОЛОВА  Дукенулы 22 87055791444`
  - manager = `Магира`
  - номер подтверждён из имени клиента: `+77055791444`
  - sibling `manager=Не определён` схлопнут в `alias`
- итоговый локальный статус `config/clients.json`: `duplicate_groups=0`

**Доказательства / проверки:**
- `python -m py_compile bot/crm_clients.py bot/send_reports.py` -> OK
- `python -X utf8 tests/test_session_20260428.py` -> 15/15 OK
- `python -X utf8 tests/test_crm_regression.py` -> 13/13 OK
- runtime evidence:
  - `logs/send_reports.log` содержит `CRM duplicate review restored: 13 records`
  - `logs/crm_audit.jsonl` содержит 13 `duplicate_phone_conflict_sent` и 13 resolved events

### Что не трогать

- `autoagent/orchestrator_agents.json`, `autoagent/task_prompt.txt` — пользовательские/служебные
- audit-черновики в `audit/`
- `config/clients.json` не коммитится; cleanup остался локальным operational change
- backups локального cleanup:
  - `config/clients.json.bak-20260430-crm-dedup`
  - `config/clients.json.bak-20260430-053134-safe-merge`

### Следующий безопасный шаг

- если понадобится, вынести local data-cleanup в отдельный воспроизводимый admin-скрипт/команду
- держать `/crmdupsend` как разовый инструмент для будущих конфликтов телефонов

---

## HANDOFF 2026-04-30 — test log isolation fix

### Что сделано (ветка `fix/log-noise-by-design-markers`)

**Баг:** `test_collector_regression_hermetic.py` загрязнял боевой `logs/collector.log` тестовыми строками (`batch-stale`, `Task-31`, `[Тест Клиент]`).

**Root cause:** после logging-унификации 30.04 `configure_runtime_logging()` в `bot/logging_utils.py` всегда открывает `TimedRotatingFileHandler` на `logs/collector.log`. `COLLECTOR_TEST_MODE=1` читался в `_TEST_MODE`, но в вызов `configure_runtime_logging()` не передавался — файловый хэндлер открывался в любом случае.

**Фикс (коммит `522df85`):**
- `bot/logging_utils.py` v1.1.1 — параметр `test_mode: bool = False`; при `True` `TimedRotatingFileHandler` не создаётся
- `collector/collections_engine.py` v1.5.0 — передаёт `test_mode=_TEST_MODE`

**Доказательства:**
- `python -m py_compile bot/logging_utils.py collector/collections_engine.py` → OK
- `python -X utf8 tests/test_collector_regression_hermetic.py -v` → 15/15 OK
- хвост `logs/collector.log` остался на `03:37` после тестового прогона в `04:17`
- бот перезапущен в `04:18`, стартовал чисто, все 30+ джобов зарегистрированы, ошибок нет

### Состояние git

- Ветка: `fix/log-noise-by-design-markers`
- HEAD: `522df85`
- Незакоммиченное: `autoagent/orchestrator_agents.json`, `autoagent/task_prompt.txt` (не трогать)

### Что не трогать

- `audit/AUDIT_TZ_20260422_DATA_DISTORTION.md`, `audit/D_AUDIT_REPORT_20260422.md`, `audit/E_PATCH_PLAN_20260422.md`, `audit/run_20260422_data/` — untracked черновики
- `Новый текстовый документ.txt` — пользовательский файл

### Следующий безопасный шаг

- При следующем прогоне тестов убедиться, что `logs/collector.log` больше не получает тестовый шум
- Открытых P1/P2 на момент закрытия сессии нет

---

## HANDOFF 2026-04-29 (сессии 2–3) — Коллектор Фазы 2–4

### Что сделано (ветка `fix/log-noise-by-design-markers`)

**Фаза 2Б — wa_dialog_suppress** (коммит ~`wa_dialog_suppress`):
- `collector/collections_db.py`: поле `wa_dialog_suppress: {reason, set_at, until}` + set/get/clear API
- `collector/client_dialog.py`: paid_claim → suppress +3д, attachment → suppress +2д
- `collector/collections_engine.py` `run()`: проверка suppress после payment_hold, аудит `wa_skipped`
- Тест: `tests/test_wa_dialog_suppress.py` — 7/7

**Фаза 2А — diff-notice при admin approve** (коммит ~`diff-notice`):
- `collector/collections_engine.py`: публичная `preview_batch_changes(batch_id, approved_clients) → Optional[str]`
- `collector/approval_flow.py` `wa_appr_adm_ok`: вставляет diff-блок перед текстом кнопки "Отправить"
- Тест: `tests/test_diff_notice.py` — 6/6

**UI — кнопка 🤖 Коллектор в главном меню** (коммит ~`collector batch menu`):
- `bot/send_reports.py`: `kb_main()` admin получил кнопку → `_format_collector_batch_text()` → статус + список клиентов

**Фаза 3А — audit log** (коммит ~`audit_log Phase 3A`):
- `collector/audit_log.py`: append-only JSONL `logs/collector_audit.jsonl`, thread-safe через `threading.Lock`
- События: wa_sent, wa_skipped(4 причины), suppress_set/cleared, batch_created/approved/sent/failed
- Тест: `tests/test_audit_log.py` — 7/7

**Фаза 3Б — log prefixes** (коммит ~`log prefixes Phase 3B`):
- `collector/collections_engine.py`: `_PrefixAdapter` + `[COLLECTOR]`
- `bot/debt_stop_control.py`: `_PrefixAdapter` + `[STOP]`
- Разделяет контуры в grep: `grep "[COLLECTOR]"` vs `grep "[STOP]"`

**Фаза 4 — hard-ban отгрузок** (коммит `34ec994`):
- Найден реальный риск: `promise_broken_reminder` содержал "влечёт ограничение отгрузок"
- Этот шаблон назначается via `no_movement + promise_broken` — может достичь legacy_tail-клиентов
- Фикс: фраза удалена из `_FALLBACK_TEMPLATES_DEFAULT["promise_broken_reminder"]` в `collection_agent.py`
- Безопасный шаблон добавлен в `config/collector_prompts.json`
- Тест: `tests/test_phase4_hard_ban.py` — 7/7

### Что проверено

- `python tests/test_wa_dialog_suppress.py` → 7/7
- `python tests/test_diff_notice.py` → 6/6
- `python tests/test_audit_log.py` → 7/7
- `python tests/test_phase4_hard_ban.py` → 7/7
- Все тесты прогнаны вместе: 27/27

### Что осталось в очереди

- Фаза 3В — единое логирование `collector/*.py` через `get_collector_logger(__name__)`
- Фаза 5 — CRM аудит: баг повторного "Чей клиент?" (6 точек проверки: get_clients_without_phones, _crm_cleanup_pending, attribution persistence, crm_daily_task, дубли в crm_pending_state.json, TTL)

---

## HANDOFF 2026-04-30 Asia/Qyzylorda — Unified runtime logging

### Что сделано

- Добавлен общий runtime logging core:
  - `bot/logging_utils.py`
  - domain-aware formatter
  - rotating daily logs + retention
  - Telegram runtime alert handler с cooldown
- `bot/send_reports.py` переведен на доменные логгеры:
  - `BOT/CORE`
  - `BOT/SCHED`
  - `CRM/FLOW`
  - `PIPELINE/FLOW`
  - `STATE/STORE`
  - `INTEGRATION/API`
- `log_event()` теперь маршрутизирует события по доменам через `_logger_for_event()`
- `collector/logging_utils.py` больше не живет отдельной prefix-only схемой:
  - `get_collector_logger()` → `COLLECTOR/FLOW`
  - `get_stop_logger()` → `STOP_CONTROL/FLOW`
- `collector/collections_engine.py` переведен на тот же runtime bootstrap
- `config.setup_logging()` унифицирован через `configure_module_logger()`:
  - модули отчетов/парсеров теперь получают общий formatter + rotation + доменную классификацию
- `bot/crm_clients.py` переведен на `CRM/STORE`
- `bot/debt_stop_control.py` переведен на `STOP_CONTROL/FLOW`

### Доказательства

- `python -m py_compile bot/logging_utils.py bot/send_reports.py collector/logging_utils.py collector/collections_engine.py config.py` → OK
- `python -X utf8 tests/test_crm_regression.py` → 6/6 OK
- `python -X utf8 tests/test_collector_regression_hermetic.py` → 15/15 OK
- `python -X utf8 tests/test_logging_runtime.py` → OK

### Что это дало

- домен проблемы теперь виден сразу по строке лога:
  - `CRM`
  - `STATE`
  - `PIPELINE`
  - `INTEGRATION`
  - `COLLECTOR`
  - `STOP_CONTROL`
  - `BOT`
- unified runtime logging теперь покрывает:
  - bot orchestration
  - collector
  - standalone модульные логгеры через `config.setup_logging()`

### Что осталось

- не все исторические direct `logger.*(...)` внутри `bot/send_reports.py` доменно размечены вручную;
- основной routing уже закрыт через `log_event()`, но часть старого кода пока остается под `BOT/CORE`;
- если продолжать, следующий этап — точечная доменная разметка remaining direct logs в крупных старых ветках монолита.

---

## HANDOFF 2026-04-29 Asia/Qyzylorda

### Что исправлено

- `collector/collections_engine.py` v`1.4.5` -> v`1.4.6`
  - перед `send-approved` добавлена обязательная пересверка admin-approved batch по свежей дебиторке;
  - если клиент уже выпал из актуального shortlist, он не уходит в WhatsApp;
  - если по клиенту изменились сумма/дни/телефон/тип сообщения, в отправку идет уже обновленная версия;
  - если свежая дебиторка недоступна, отправка блокируется, а администратор получает notice.
- `collector/client_dialog.py` v`1.0.9` -> v`1.1.0`
  - `paid_claim` (`оплатили`, `вчера была оплата`, `давно оплатили`) больше не остается в обычной debt-ветке;
  - введен state `awaiting_payment_proof`: бот один раз просит чек/дату/сумму и дальше не дожимает клиента повторными debt-фразами;
  - при входящем доказательстве оплаты диалог переводится в `awaiting_manager`, а наблюдателям уходит note со ссылкой на вложение;
  - короткие реплики вроде `хорошо` после `paid_claim` больше не вызывают второй автоответ;
  - сервисные запросы вида `акт сверки` сразу эскалируются менеджеру.
- `collector/whatsapp_poller.py` v`1.1.5` -> v`1.1.6`
  - входящие `document/image/video` теперь пробрасывают в `handle_incoming()` метаданные вложения (`downloadUrl`, `fileName`, `caption`, `mimeType`);
  - это нужно, чтобы доказательство оплаты можно было сразу передать менеджеру/наблюдателям без повторного запроса к клиенту.
- добавлен герметичный regression-suite `tests/test_collector_regression_hermetic.py`
  - без реальных отправок в WhatsApp/Telegram;
  - без Green API/Telegram сети;
  - покрывает именно спорные collector-сценарии этой сессии.
- `collector/collections_engine.py` v`1.4.6` -> v`1.4.7`
  - stop-клиенты разделены на живой shipment-stop и старые хвостовые долги;
  - если клиент долго висит в долге, новых отгрузок нет и он не выглядит как живой торговый stop-case, используется отдельный `msg_type` без текста про ограничение отгрузок;
  - введены `legacy_tail_reminder` и `partial_tail_reminder`.
- `collector/approval_flow.py` v`1.0.9` -> v`1.1.0`
  - в согласовании WhatsApp теперь явно различаются:
    - `stoplist_reminder` — живой stop-кейс;
    - `legacy_tail_reminder` — старый хвост без движения;
    - `partial_tail_reminder` — старый хвост с частичным погашением.
- `collector/collection_agent.py` v`1.0.9` -> v`1.1.0`
  - добавлены fallback-шаблоны для старых хвостов без слова `отгрузки`.
- `config/collector_prompts.json`
  - добавлены `legacy_tail_reminder` и `partial_tail_reminder`;
  - фраза про ограничение отгрузок оставлена только для настоящего `stoplist_reminder`.
- `collector/debt_monitor.py` v`1.0.7` -> v`1.0.8`
  - loader теперь возвращает `_freshness` metadata по debt snapshot: `period_max`, `age_days`, `warn/stale`, `file` по каждому менеджеру;
  - это стало базой для блокировки live-send по старой дебиторке и для показа даты данных в preview.
- `collector/collections_engine.py` v`1.4.7` -> v`1.4.8`
  - live `run()` теперь блокируется, если debt snapshot устарел сверх SLA;
  - `send-approved` тоже блокируется по stale debt snapshot выбранных менеджеров;
  - `run_approval_preview()` сохраняет в batch `debt_snapshot` summary, чтобы preview/admin summary показывали дату и возраст данных.
- `collector/approval_flow.py` v`1.1.0` -> v`1.1.1`
  - manager preview, admin preview notice и admin summary теперь показывают `Данные дебиторки` и `Возраст данных`;
  - при stale-warning это видно ещё до финального утверждения отправки.
- операционный регламент collector уточнен:
  - trigger/check окно расширено до `09:00–22:00`;
  - TTL флага свежей debt-trigger логики расширен до `14ч`;
  - причина: Саида временно может разносить оплаты после `20:00`.
- усилена наблюдаемость live WhatsApp:
  - после каждого `send_whatsapp()` администратор получает мгновенный notice;
  - `daily_summary()` включает отдельный блок с перечнем фактических WA-получателей.
- CRM clarify-phone hygiene уточнена:
  - служебные/зарплатные записи должны фильтроваться и на входе `get_clients_without_phones()`, и в `send_reports._crm_cleanup_pending()`;
  - иначе queue `clarify_phone` бесконечно загрязняется ЗП/служебными хвостами.

### Что доказано

- stale debt инцидент утром `2026-04-29` был вызван нераскрытым вчерашним batch, а не ошибкой клиента:
  - старый batch был собран `2026-04-28`, но отправлен только утром `2026-04-29`;
  - свежая дебиторка после вечерней разноски оплат в 1С подхватилась позже;
  - значит корень был в frozen snapshot approved batch перед send.
- важно не смешивать два разных контура:
  - `collector/*` и `logs/wa_approval_batches.json` — это WhatsApp debt collector и его manager/admin approval batch;
  - `bot/debt_stop_control.py` и `reports/debt_stop_registry.json` — это отдельный stop/clearance workflow по отгрузкам, Саиде и руководителю.
- кейс `Е ИП Реян (Жангали)` с ответом `✅ Утверждено — ... Менеджер и Саида уведомлены` был штатным `debt_stop_control` сценарием:
  - в `14:00:56` stop-monitor увидел, что `current["debt"] <= STOP_PAID_THRESHOLD(5000)`;
  - Ергали получил запрос выбрать дальнейший режим работы с клиентом после полной оплаты;
  - руководитель подтвердил предложение менеджера через `dstop_adm_cl_confirm`;
  - это не было подтверждением нового лимита и не было bug-сигналом collector.
- WhatsApp voice recognition на текущем HEAD работает:
  - в runtime-логах есть успешное распознавание входящего `.ogg` через AssemblyAI;
  - проблема этой сессии была не в STT, а в freshness debt и UX client dialog.
- текущий источник истины по collector regression теперь не legacy `tests/test_collector.py`, а отдельный hermetic-suite.
- архив `archive/` подтверждает отдельный класс старых хвостовых должников, которым фраза про ограничение отгрузок не подходит:
  - `Е ИП Шахин`
  - `Е Еркебулан`
  - `Е ТД Саянур Леонид`
  - `Е ТОО ГудФуд № 1 ул Досмухамедулы 48(Аида)`
  - `М Ресторан Шама ИП Тян ул Мустафина 12`
- по текущей логике эти клиенты теперь не получают `stoplist_reminder`, если в текущем срезе нет новых отгрузок и долг выглядит как старый хвост.
- отсутствие Ергали в новых collector-batches `20260429-135316-54e7` и `20260429-140258-5881` не было bug-сигналом routing:
  - утром `29.04.2026` его клиенты уже были отправлены из старого admin-approved batch `20260428-170001-2bef`;
  - после этого дневной guard `already_contacted_today()` корректно не дал включить их в новые preview повторно;
  - отдельные stop-control уведомления Ергали в этот день были штатным другим контуром и не относятся к collector approval batch.

### Проверки

- `python -m py_compile collector\client_dialog.py` — OK
- `python -m py_compile collector\collections_engine.py` — OK
- `python -m py_compile collector\whatsapp_poller.py` — OK
- `python -X utf8 tests\test_collector_regression_hermetic.py -v` — **7/7 OK**, `Ran 7 tests in 3.151s`
- `python -m py_compile collector\approval_flow.py` — OK
- `python -X pycache_prefix=C:\Users\user\.codex\memories\pycache_tmp -m py_compile collector\collection_agent.py` — OK
- `python -X utf8 tests\test_collector_regression_hermetic.py -v` — **11/11 OK**, `Ran 11 tests in 3.020s`
- `python -X utf8 -m unittest tests.test_collector_regression_hermetic.LegacyTailClassificationHermeticTests -v` — **4/4 OK**
- `python -X pycache_prefix=C:\Users\user\.codex\memories\pycache_tmp -m py_compile collector\debt_monitor.py` — OK
- `python -X pycache_prefix=C:\Users\user\.codex\memories\pycache_tmp -m py_compile collector\collections_engine.py` — OK
- `python -X pycache_prefix=C:\Users\user\.codex\memories\pycache_tmp -m py_compile collector\approval_flow.py` — OK
- `python -X utf8 tests\test_collector_regression_hermetic.py -v` — **15/15 OK**, включая freshness metadata, stale live-send block и preview snapshot warning
- документальные уточнения collector knowledge base внесены в:
  - `SESSION_CONTEXT.md`
  - `gpt1c.md`
  - `audit/ARCHITECTURE.md`

### Что именно покрывает hermetic-suite

- stale batch refresh перед `send-approved`;
- пропуск уже неактуального клиента из batch;
- `paid_claim` -> `awaiting_payment_proof`;
- отсутствие второго автоответа после короткого подтверждения клиента;
- forwarding входящего чека/доказательства менеджеру/наблюдателям;
- мгновенная эскалация сервисного запроса;
- проброс attachment metadata из `whatsapp_poller` в `client_dialog`.
- отдельная классификация старых хвостов:
  - stopped + нет новых отгрузок + нет оплат -> `legacy_tail_reminder`;
  - stopped + нет новых отгрузок + есть частичная оплата -> `partial_tail_reminder`;
  - живой stop-case с `debit > 0` -> остаётся `stoplist_reminder`.

### Что осталось открытым

- `tests/test_collector.py` остается legacy-интеграционным файлом:
  - полный прогон в этом окружении не является надежным критерием;
  - в нем остается старый baseline-failure `send_whatsapp returns False when disabled`;
  - его нельзя использовать как единственное доказательство качества collector-правок.
- import-time logging/file locks на Windows никуда не делись как класс риска; для спорных collector-правок сначала запускать hermetic-suite.

### Что не трогать

- untracked audit-черновики:
  - `audit/AUDIT_TZ_20260422_DATA_DISTORTION.md`
  - `audit/D_AUDIT_REPORT_20260422.md`
  - `audit/E_PATCH_PLAN_20260422.md`
  - `audit/run_20260422_data/`
- пользовательский untracked файл:
  - `Новый текстовый документ.txt`

### Следующий безопасный шаг

- перед любыми следующими правками collector-контура сначала прогонять:
  - `python -X utf8 tests\test_collector_regression_hermetic.py -v`
- если нужен дальнейший UX-тюнинг, менять только state-driven ветки в `collector/client_dialog.py`, не возвращаясь к свободным повторяющимся reply templates.

---

## HANDOFF 2026-04-23 09:00 Asia/Almaty

### Что исправлено

- Telegram polling/runtime hardened без отключения TLS:
  - `bot/send_reports.py` v`v9.4.59/23.04.2026` → `v9.4.60/23.04.2026`
  - добавлен `_PinnedTelegramRequest(HTTPXRequest)`:
    - явный `certifi.where()` как CA bundle
    - `trust_env=False`
    - отдельные request-объекты для bot API и `getUpdates`
  - `Application.builder()` теперь использует:
    - `.request(main_request)`
    - `.get_updates_request(updates_request)`
  - при TLS verify failure лог теперь явно пишет:
    - `ca_bundle`
    - `trust_env=False`

### Что доказано

- текущий открытый Telegram TLS-контур был не в данных, а в polling transport.
- blind-fix вида `verify=False` не применялся.
- теперь bot polling не зависит от скрытых proxy/SSL env и использует явный публичный CA bundle.

### Проверки

- `python -m py_compile bot/send_reports.py` — OK
- `python -X utf8 tests/test_project.py` — `110/110`
- в `tests/test_project.py` добавлены проверки:
  - `_PinnedTelegramRequest` строит `httpx.AsyncClient` с `trust_env=False`
  - `verify` — это `ssl.SSLContext`
  - `getUpdates` request имеет более длинный `read_timeout`, чем обычный bot API request

### Что осталось

- чтобы фикс начал работать в бою, нужен перезапуск бота.
- если после этого `CERTIFICATE_VERIFY_FAILED` повторится, это уже будет сильное доказательство внешней TLS/MITM/сети проблемы, а не скрытого env/request-контура внутри процесса.

---

## HANDOFF 2026-04-23 08:45 Asia/Almaty

### Что исправлено

- `b3ce7ab` — `fix(data): enforce excel-truth for net profit and debt delivery`
  - `net_profit_report.py` v`1.2.7` → v`1.2.8`
    - admin `net_profit` теперь берёт только сводные `gross_*.json`
    - manager gross больше не может подмешаться в admin MTD
    - для MTD отключён mixed-source fallback на "ближайшие" expenses
    - если exact expenses за тот же период нет, MTD не генерируется
  - `bot/send_reports.py` v`v9.4.58/22.04.2026` → `v9.4.59/23.04.2026`
    - live `DEBT_SIMPLE` блокируется, если для того же менеджера уже есть более свежий `DEBT_EXTENDED`
    - `force|net_profit` и меню аналитики больше не отдают stale `net_profit_mtd`, если current summary gross не имеет exact expenses
  - `tests/test_report_freshness.py`
    - добавлены регрессии `FRESH T5..T10`

### Что доказано

- Ложный `net_profit_mtd_20260411.html` строился не из сводного gross Excel, а из manager gross + чужих expenses.
- Простая дебиторка могла live-выдаваться за `18.04`, хотя detailed debt уже был свежий за `22.04`.
- После фикса:
  - manager gross отфильтровывается из admin gross-источников;
  - stale simple debt не проходит live-route;
  - stale MTD HTML не проходит `force|net_profit` и analytics-route.

### Проверки

- `python -m py_compile bot/send_reports.py` — OK
- `python -X utf8 tests/test_report_freshness.py` — `10/10`
- `python -X utf8 tests/test_project.py` — `105/105`
- `python -c "import ast, pathlib; ast.parse(pathlib.Path('net_profit_report.py').read_text(encoding='utf-8'))"` — `net_profit_report.py AST OK`

### Что осталось открытым

- `NET-SSL-TELEGRAM-01` остаётся открытым как внешний TLS/runtime incident:
  - burst `CERTIFICATE_VERIFY_FAILED / self-signed certificate in certificate chain`
  - кодовый дефект пока не доказан
  - `verify=False` не применялся и не должен применяться без отдельного технического доказательства

### Что не трогать

- untracked audit-черновики:
  - `audit/AUDIT_TZ_20260422_DATA_DISTORTION.md`
  - `audit/D_AUDIT_REPORT_20260422.md`
  - `audit/E_PATCH_PLAN_20260422.md`
  - `audit/run_20260422_data/`

---

## HANDOFF 2026-04-22 19:30 Asia/Almaty (audit session continuation)

### Что сделано после handoff 16:40

**Phase 3 — глубокий аудит + точечные фиксы (5 файлов, все закоммичены):**

| Файл | Версия | Fix |
|------|--------|-----|
| `imap_fetcher.py` | v4.4.6 → v4.4.7 | F-IMAP-001 (WARNING при пустом whitelist), F-IMAP-002 (`Path(fname).name` + reject "." ".." "" — path-traversal guard), F-IMAP-003 (try/except `M.shutdown()` в except-ветке `_imap_connect`), F-IMAP-004 (комментарий-охрана `load_dotenv()` перед TZ — защита BUG-H3 adc61bc) |
| `send_tg.py` | v2.4.1 → v2.4.2 | F-TG-001 (`send_file` читает bytes в память перед `_post_tg` — при retry на 5xx/timeout file-handle иначе прочитан, вторая попытка отправила бы 0 байт), F-TG-002 (`print("TG: file OK")` в CLI `--file` ветке для симметрии с `--text`) |
| `config.py` | v3.6.4 → v3.6.5 | F-CFG-001 (`_read_yaml` ловит `(yaml.YAMLError, OSError, UnicodeDecodeError)` вместо широкого `Exception`) |
| `bot/inventory_summary.py` | v1.6 → v1.7 | S2 (regex `[Р°-СЏС‘]+` — CP1251-в-UTF-8 mojibake → корректный `[а-яё]+`), S3 (docstring синхронизирован с v1.7 стратегией) |
| `bot/crm_clients.py` | v1.0.4 → v1.0.5 | S1 (хардкод `("Алена","Ергали","Магира","Оксана")` → `_load_known_managers()` из `config/managers.json` с fallback; соблюдение single-source-of-truth по CLAUDE.md) |

**Phase 4 — быстрый скан остальных модулей (все чисты, правок не требовалось):**

- `utils_common.py` v1.1.0 — pure функции, чист
- `utils_excel.py` v2.3.4 — 3 широких except в utility-guard паттернах (REFACTOR-класс, не-баги)
- `bot/silence_alerts.py` v1.7 — защитные широкие except в parser-entry функциях (приемлемо)
- `bot/opportunity_loss.py` v1.5.2 — чист, fix #OPLOSS-1 на месте
- `bot/user_tracker.py` v1.0.2 — чист, BUG-H4 (threading.Lock) + BUG-L7 (narrow except) уже закрыты
- `bot/log_monitor.py` v1.0.1 — чист, атомарная запись tmp+replace

**Аудит freshness-фикса b5fb564 (работа другого ИИ):**
- Подтверждён живой проверкой: оба JSON-семейства (`debt_ext_Ведомость_…` и `debt_ext_Детальный_Дебиторы_…`) коэкзистируют в `reports/json/`, приоритет отдаётся «Детальный» через glob-фильтр + mtime
- Логи `logs/collector_20260422.log` показывают работу фильтра свежести: `"Пропускаем устаревший debt JSON"`

### Коммиты этого блока

- `b8a0d2f` — `fix: аудит 22.04.2026 - narrow except, retry-safe send, CRM source of truth`
  - `config.py`, `send_tg.py`, `bot/inventory_summary.py`, `bot/crm_clients.py`
  - (`imap_fetcher.py` был закоммичен ранее в этой же сессии — до контекстного разрыва)
- `2969640` — `docs: session context 22.04.2026` (handoff 16:40)

### Что проверено

- `python -m py_compile config.py` — OK
- `python -m py_compile imap_fetcher.py` — OK
- `send_tg.py`, `bot/inventory_summary.py`, `bot/crm_clients.py` — синтаксис подтверждён через `ast.parse(open(...).read())`, т.к. Windows держал lock на `__pycache__/*.pyc` от работающего бота (не синтаксическая ошибка)
- `tests/test_project.py` — **105/105**
- `tests/test_report_freshness.py` — **4/4** (регрессия b5fb564 зафиксирована тестом)
- `tests/test_collector.py` — прогон прерван по таймауту времени выполнения, НО: падений/traceback нет, дошедшие секции зелёные

### Что осталось untracked

- `debt_stop_state.json` — runtime-артефакт, обновляется ботом автоматически, в коммиты не включается

### В работе (передано в Codex)

Codex пишет регрессионные тесты по ТЗ от 2026-04-22 (см. чат Claude):

- `tests/test_silence_alerts.py` — 8 кейсов, главный — T6 freshness-regression (`_get_all_debt_reports` + `get_latest_debt_report` не даёт «Ведомости» побеждать «Детальный»)
- `tests/test_opportunity_loss.py` — 6 кейсов, главный — T1 OPLOSS-1 regression (`_find_latest_gross_html` не подсовывает чужой gross)

Назначение: закрыть тест-дыру в модулях, затронутых b5fb564, до следующей регрессии.

### Открытые OPEN-пункты (не критично, не-баги)

- REFACTOR ~30+ широких `except Exception` в utility-guard паттернах — требуют бизнес-решений по каждому случаю
- ARCH-1: `txt_to_html` в двух местах (`tools/txt_to_html.py` + `bot/send_reports.py`) — унификация рискованная, ломает call sites
- ARCH-3: inline HTML в `expenses_parser.py` — изолировано, работает корректно

### Что читать новому ИИ в первую очередь

1. `CLAUDE.md` (project overview + rules + fixed bugs history)
2. Этот handoff (19:30)
3. Handoff 16:40 ниже
4. `gpt1c.md` (актуальный статус)
5. Последние коммиты через `git log --oneline -20`

### Открытые задачи по TaskList (закрытые, для контекста)

```
#1 [completed] Аудит freshness-фикса b5fb564
#2 [completed] Phase 3 subtask: send_tg.py audit
#3 [completed] S2: inventory_summary regex mojibake
#4 [completed] S1: crm_clients hardcoded managers
#5 [completed] Phase 3: config.py audit + fixes
#6 [completed] Phase 4: quick scan + report
```

Все 6 задач этой сессии закрыты.

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

## Handoff Update - 2026-04-22 20:55 +05:00

### Added regression tests for silence alerts and opportunity loss

- New tests added without runtime-code changes:
  - `tests/test_silence_alerts.py`
  - `tests/test_opportunity_loss.py`
- Scope covered:
  - `silence_alerts`
    - `parse_debt_amount`
    - `parse_report_date`
    - `categorize_by_silence`
    - weekly-client skip logic
    - `MIN_DEBT_AMOUNT`
    - imitation detection
    - freshness regression for `_get_all_debt_reports()` / `get_latest_debt_report()`
    - `_period_sort_key`
  - `opportunity_loss`
    - `_find_latest_gross_html()` foreign-file regression
    - `_get_manager_margin()` fallback to `DEFAULT_MARGIN_PCT`
    - `calculate_opportunity_loss()` zone counts, debt filter, turns formula

### Validation

- `python -X utf8 tests/test_silence_alerts.py`
  - `23/23` passed
- `python -X utf8 tests/test_opportunity_loss.py`
  - `10/10` passed
- `python -X utf8 tests/test_project.py`
  - did **not** fail on the new tests
  - still stops at the old known environment issue:
    - import-time `logging.FileHandler` lock on `logs/send_reports_20260422.log` inside `bot/send_reports.py`
    - traceback location: `tests/test_project.py` during `import send_reports`
- syntax of both new test files confirmed via `ast.parse(...)`
  - direct `py_compile` hit Windows `tests/__pycache__` lock, not syntax problems

### Git

- test commit:
  - `6d6ce48` `test: add regression coverage for silence alerts and opportunity loss`
- pushed to:
  - `origin/fix/log-noise-by-design-markers`

### Working tree status

- runtime artifact remains untracked:
  - `debt_stop_state.json`
- no production Python files were changed in this test task

### Recommended next step

1. keep these two new regression suites as the narrow guardrail for future freshness / alert-path edits
2. treat the remaining `test_project.py` failure as the separate old `send_reports.py` log-lock problem, not as a regression from this task

## Handoff Update - 2026-04-23 13:55 +05:00

### Collector client-dialog UX hardening

- Changed collector customer-facing wording and response policy after the WhatsApp screenshot issue:
  - removed mechanical `напишите 1` prompts from `config/collector_prompts.json` and code fallbacks
  - corrected company city in collector prompt context from Almaty to Astana
  - shortened fallback reminder templates and removed repeated bureaucratic wording
  - added AI-analysis intents:
    - `soft_positive`
    - `promise_schedule`
    - `paid_claim`
  - `client_dialog.handle_incoming()` now handles:
    - payment claims without arguing about 1C; asks for cheque/date/amount
    - date-only promises without fake "фиксируем оплату"
    - daily/partial schedules with `payment_schedule` stored in dialog state
    - soft-positive answers with one short follow-up question, then manager escalation if still vague

### Files changed

- `config/collector_prompts.json`
- `collector/collection_agent.py`
- `collector/client_dialog.py`
- `tests/test_collector.py`

### Validation

- JSON syntax:
  - `python -m json.tool config/collector_prompts.json`
- Python syntax:
  - `ast.parse(...)` for `collector/collection_agent.py`, `collector/client_dialog.py`, `tests/test_collector.py`
  - direct `py_compile` for collector files hit Windows `__pycache__` permission lock, not syntax
- `python -X utf8 tests/test_project.py`
  - `110/110`
- `WHATSAPP_ENABLED=0 LIVE_SEND_ALLOWED=0 python -X utf8 tests/test_collector.py`
  - reached the new UX and prompt sections with all checks green:
    - no `напишите 1`
    - prompt uses Astana, not Almaty
    - date-only promise does not say `фиксируем`
    - schedule stores `payment_schedule`
    - paid claim asks for cheque without repeated 1C reference
    - soft positive asks one short first-payment question
  - full script still timed out later around old HIGH-4 / later collector sections; no failing check was observed before timeout

### Operational note

- This is a collector-dialog behavior change only.
- Approval flow, batch state machine, debt selectors, WhatsApp transport, and Telegram scheduler were not changed in this step.

---

## HANDOFF 2026-05-07 — Инцидент approval flow + Saida UX overhaul

### HEAD: `44df2a4` | Тесты: 408/408

---

### Инцидент: батч 20260507-170001-2709 не ушёл в WA

**Хронология:**
- 17:00 — батч создан, превью отправлено Ергали и Магире (6 клиентов)
- 18:10 — эскалация к директору (менеджеры молчали 1 час)
- 18:50:05 — последняя строка в `send_reports.log`
- 18:50:10 / 18:51:11 — директор нажал "Утвердить" дважды (`batch_approved` в audit)
- UI не обновился, кнопка "Отправить сейчас" не появилась
- 19:30 — батч протух, WA не ушёл

**Root cause #1 — `configure_runtime_logging()` на import-time (фикс `44df2a4`):**
- `approval_flow.py` делает `lazy import collections_engine` внутри callback
- При импорте модуль вызывал `configure_runtime_logging()` → закрывал root handlers живого бота
- Все последующие `logger.*` вызовы → `ValueError: I/O operation on closed file`
- `_tg_edit` с кнопкой "Отправить сейчас" падал на `logger.info` внутри → кнопка не рендерилась
- Директор видел старый экран с кнопкой "Утвердить", жал снова — повторял цикл
- **Фикс:** `configure_runtime_logging()` → `_configure_cli_logging()`, вызывается только из `main()`

**Root cause #2 — `asyncio.CancelledError` в preview_batch_changes (фикс `cd87c47`):**
- `preview_batch_changes` запускается через `asyncio.to_thread` с timeout=10
- При отмене корутины бросался `CancelledError` (BaseException) — не ловился `except Exception`
- `_tg_edit` с "Утверждено + кнопка Send" никогда не выполнялся
- **Фикс:** отдельный `except asyncio.CancelledError as _ce` → UI обновляется, потом re-raise

**Root cause #3 — нет `q.answer()` (фикс `cd87c47`):**
- Все `wa_appr_*` callbacks не вызывали `q.answer()` перед тяжёлой работой
- Telegram держал loading-spinner 30 сек → "Query is too old"
- **Фикс:** `q.answer()` добавлен в `send_reports.py` перед `_wa_appr_cb`

**Подтверждение исправления (22:53:29–22:53:32):**
```
[20260507-170001-2709] admin approve button pressed by chat_id=...
[20260507-170001-2709] admin approve: preview_batch_changes start
[20260507-170001-2709] admin approve: preview_batch_changes finish
TG edit ok: message_id=18137
[20260507-170001-2709] Администратор УТВЕРДИЛ отправку: 3 клиентов
[20260507-170001-2709] admin send button pressed ... status=admin_approved
send_started_at = 2026-05-07T22:53:32
SEND-APPROVED BLOCKED: outside allowed time window  ← честная блокировка, не баг
```

---

### Saida UX overhaul (коммит `130cf92`)

- **Текстовый парсер Саиды:** `parse_saida_text_reply()` в `payment_hold.py`
  - Распознаёт «Акжан полная», «Шапагат частично», «Петро нет»
  - `confirm_by_saida()` идемпотентен — повторный вызов (кнопка + текст) не даёт второго уведомления
- **Date parser** в `approval_flow.py`: `_extract_deadline_from_text()` → `Optional[date]`
  - Поддержка: завтра / послезавтра / через N дней / в пятницу / до конца недели / 5 числа
  - При None — переспрашивает менеджера вместо тихого дефолта +3 дня
- **HTML инструкции:** `docs/Инструкция Саида.html` (уровень 1/10), обновлена менеджерская инструкция
- **Client dialog:** inactive dialog forwarding, DeepSeek intents (cash_pickup/dispute/doc_request/complaint)
- **Кнопка "📖 Инструкция"** в главном меню всех ролей

---

### Прочие фиксы (07.05)

- `debt_stop_control.py`: удалены мёртвые mojibake-блоки с `NameError: name 'token'` (`2ebc663`)
- `tests/test_collector.py` section 19: `CONFIG_DIR` теперь переключается в tempdir → `_sanitize_state_candidates` не режет тестовых кандидатов
- `approval_flow.py`: entry-логи для `wa_appr_adm_ok` и `wa_appr_adm_send` (`db058dd`)
- `approval_flow.py`: проверка `expires_at` в `wa_appr_adm_send` — если батч просрочен → `⛔ Окно закрыто`

---

### Коммиты сессии

| Коммит | Что |
|--------|-----|
| `130cf92` | Saida UX overhaul + date parser + double-notify fix |
| `2ebc663` | Удалены mojibake dead code + NameError 'token' |
| `cd87c47` | CancelledError fix + q.answer() для wa_appr_* |
| `db058dd` | Entry-логи approve/send кнопок |
| `44df2a4` | configure_runtime_logging убран с import-time |

---

### Открытые задачи

- Phase 3 dialog migration — text/voice routing через `get_text_target_dialog`
- Алма Дист: уточнить расхождение Ведомость vs Детальный у бухгалтера
- Операционно: Магира → Прайм Фаст Фуд + Бон Апетит; Оксана → МАСТЕР-КОНДИТЕР

---

## HANDOFF 2026-05-08–09 — Payhold SLA overhaul

### HEAD: `9850114` | Тесты: 410/410

### Подтверждённый баг: stale hold + двойная эскалация

**Симптом (09.05 09:02):** холды созданы 08.05 ~16:05, обработаны через 17ч. В одном
проходе по одним и тем же клиентам ушли и warning, и bypass одновременно.

**Root cause:** три независимых `if` вместо приоритетной цепочки. При возрасте 17ч оба
условия (`age >= warn` и `age >= bypass`) истинны → оба блока срабатывают в одной итерации.

### Фиксы (08–09.05)

**`a1a6066`** — точные формулировки: "не ответила" (директор/менеджер), "проигнорировала" (Саида)

**`0799827`** — Саида таймаут → авто-закрытие как `rejected`, директору INFO без кнопок.
Раньше: директор получал "Ваше решение: ✅/❌" — бессмысленно, он не знает пришла ли оплата.

**`9850114`** — три `if` → строгая цепочка `if/elif/elif/else`:
```
TTL (>12ч) → тихо закрыть как expired (без уведомлений)
bypass (>2ч) → авто-закрытие + уведомления
warn (>1ч) → только предупреждение Саиде
else → ничего
```

### Параметры (payment_hold.py, изменены пользователем)

| Константа | Было | Стало |
|-----------|------|-------|
| `SAIDA_WARN_HOURS` | 4ч | 1ч |
| `SAIDA_BYPASS_HOURS` | 8ч | 2ч |
| `SAIDA_STALE_TTL_HOURS` | — | 12ч (новый) |

### Другие фиксы сессии

- `payhold_admin_full/none` callback убран — директор не должен решать за Саиду
- `wa_appr_adm_send`: проверка `expires_at` — если батч просрочен, кнопка показывает ⛔
- WA-batch 20260508-170000-aea6 отработал штатно: Ергали ответил сразу, двое эскалированы

### Диагностика payhold SLA (актуально)

```
logs/saida_payment_holds.json — статусы: pending_saida → confirmed_* / rejected / expired
send_reports.log — "Саиде предупреждение по X (1.2ч)" / "Таймаут Саиды по X" / "Старый hold закрыт"
```

---

## 2026-04-29 CRM + Collector logging patch
- Working C project patched with CRM canonical duplicate merge in bot/crm_clients.py.
- CRM claim flow in bot/send_reports.py now persists state in logs/crm_claim_pending_state.json and uses unique claim_<timestamp>_<uuid8> tokens.
- Added bot/crm_audit_log.py -> logs/crm_audit.jsonl.
- Added collector/logging_utils.py and switched collector modules to shared [COLLECTOR] logger helper.
- Added collector stage audit in whatsapp_poller/client_dialog: wa_incoming_received, incoming_ignored_*, dialog_started, client_reply_received, wa_reply_sent/failed, payment_claim_reported, payment_proof_received, dialog_escalated.
- Added tests/test_crm_regression.py.

2026-04-30
- Рабочая копия на C: получила CRM и collector system logging patch.
- CRM:
  - канонизация ключа клиента в `bot/crm_clients.py`
  - restart-safe `crm_claim_pending_state.json`
  - `bot/crm_audit_log.py` -> `logs/crm_audit.jsonl`
  - `crm_claim` назначает менеджера всем каноническим дублям
- Collector:
  - `collector/logging_utils.py`
  - единый `[COLLECTOR]` logger в collector-модулях
  - audit events в `client_dialog.py` и `whatsapp_poller.py`
- Во время внедрения был сломан callback range `weekly_deny/crm_claim` в `bot/send_reports.py`; дефект исправлен до финальной проверки.
- Проверки:
  - `python -m py_compile` по измененным runtime-файлам -> OK
  - `python -X utf8 tests/test_crm_regression.py` -> 4/4 OK
  - `python -X utf8 tests/test_collector_regression_hermetic.py` -> 15/15 OK
  - `python -X utf8 tests/test_audit_log.py` -> 7/7 OK
- Live logs:
  - `send_reports_20260429.log` подтверждает Green API `200 OK`, Telegram `200 OK`, признаков WhatsApp block нет.

## Handoff Update - 2026-05-02 09:13 +05:00

### Operational incident: transient network/DNS outage, recovered

- `log_monitor_summary.log` on `2026-05-01` / `2026-05-02` started alerting on repeated IMAP failures:
  - `email_20260501.log` and `email_20260502.log`
  - `IMAP connect/login failed ... timed out`
- Main bot then hit a real Telegram transport outage on `2026-05-02`:
  - `logs/send_reports.log`
  - repeated `httpx.ConnectError: [Errno 11001] getaddrinfo failed`
  - PTB polling eventually stopped the app after cleanup failure in `telegram.ext.Updater`
- During the outage:
  - DNS later resolved again for both `api.telegram.org` and `mail.minbarakat.kz`
  - one manual restart attempt did not hold
  - next restart succeeded and the bot returned to stable work

### Current state at session close

- Bot is alive after user restart:
  - process start `2026-05-02 09:06:02`
  - `logs/send_reports.log` contains:
    - `2026-05-02 09:06:56, INFO Scheduler started`
    - `2026-05-02 09:06:56, INFO Application started`
- Telegram side is healthy in the fresh tail:
  - `whatsapp_poller`, `new_reports`, `ai_queue_processor`, `collector_reminders`, `janitor` all `executed successfully`
- IMAP recovered too:
  - `2026-05-02 08:58:01, INFO IMAP LOGIN OK (attempt 1/5)`
  - `2026-05-02 09:07:05, INFO IMAP LOGIN OK (attempt 1/5)`
  - latest visible cycles ended with `CYCLE DONE`

### Code/worktree state

- Current HEAD: `c641569` (`docs: update crm cleanup handoff`)
- There is still an uncommitted runtime change in `bot/send_reports.py`:
  - version bumped to `v9.4.64/30.04.2026`
  - collector subprocess launch switched from file path to module path:
    - `collector/collections_engine.py`
    - -> `python -m collector.collections_engine`
  - this fixed the earlier `ModuleNotFoundError: No module named 'collector'` in scheduled collector runs
- Dirty files intentionally left alone:
  - `autoagent/orchestrator_agents.json`
  - `autoagent/task_prompt.txt`
  - `bot/send_reports.py`
  - untracked `audit/*`
  - local backups `config/clients.json.bak-*`
  - untracked `site/`

### Verification remembered for next session

- Operational verification only in this closing step:
  - live process list confirmed running bot
  - fresh `send_reports.log` tail confirmed normal scheduler activity
  - fresh `email_20260502.log` tail confirmed IMAP recovery
- No new code tests were run in this last monitoring-only step.

## Handoff Update - 2026-05-06 13:30 +05:00

### Closed in this session

- `285db0a` `feat(approval): Б-lite — директор проверяет договорённости менеджеров`
  - `collector/approval_flow.py`
  - director now has a separate agreed-review screen with `accept/reject` per manager promise
  - rejected `Договорились` returns the client into the current WA batch
  - accepted promises continue participating in broken-promise daily checks
- `202e677` `feat(stop): auto-clear stop after Saida full payment`
  - `bot/debt_stop_control.py`
  - `tests/test_collector.py`
  - when Saida confirms full payment in stop-flow, client is auto-cleared from stop registry
  - manager and Saida are notified immediately
  - linked `collector/shipment_control.py` decision is auto-resolved with reason `saida_confirmed_full`

### Verification

- `python -m py_compile collector/approval_flow.py` -> OK
- `python -m py_compile bot/debt_stop_control.py` -> OK
- `WHATSAPP_ENABLED=0 LIVE_SEND_ALLOWED=0 python -X utf8 tests/test_collector.py` -> `324/324`

### Current operational/product state

- WA approval flow is now effectively closed:
  - `Оплатил` and `Договорились` are mandatory business reasons
  - `Договорились` is one-time, stores details/deadline, auto-returns on broken promise
  - director can review and reject promises instead of being forced to accept manager wording
- Stop/payment loop is closed for the full-payment path:
  - manager claim -> Saida confirms full -> stop auto-clears without second manual action
- Remaining wider topics are not bugs, but next-stage work:
  - analytics on promise quality by manager
  - operational backlog/SLA discipline around Saida
  - richer handling for partial-payment conflicts and director reporting

### Dirty files intentionally left alone

- `autoagent/orchestrator_agents.json`
- `autoagent/task_prompt.txt`
- untracked `audit/*`
- local backups `config/clients.json.bak-*`
- untracked `site/`

## Handoff Update - 2026-05-11

### Scope

- Fixed collector dialog routing around false-positive payment readiness on short client replies.
- No changes to approval-flow runtime in this session.

### Code changes

- `collector/client_dialog.py`
  - version bumped to `1.1.5`
  - removed duplicate `_is_greeting_only()` definition
  - fixed `_normalize_text()` so words no longer split into per-character tokens
  - unified greeting guard through normalized text
  - connected `_is_acknowledgement_only()` into `soft_positive` routing
  - extended neutral acknowledgement coverage with `да`, `ага`, `угу`
  - removed dead `_PURE_GREETINGS` entry `ас-саляму алейкум`

- `tests/test_collector.py`
  - added helper coverage for greeting normalization edge-cases:
    - `Здравствуйте?`
    - `Добрый  день`
    - `Добрый день)`
  - added runtime regression for `soft_positive + greeting`
  - added runtime regression for `soft_positive + acknowledgement`
  - relaxed two checks from literal `is False` to semantic `is not True` for `awaiting_payment_proof`

### Root cause

- A duplicate `_is_greeting_only()` left the active runtime guard too narrow.
- `_normalize_text()` had been in a broken form that could split words into character-spaced tokens, defeating normalized matching.
- Neutral replies like `Хорошо` were not separated from actual payment intent inside the `soft_positive` branch.

### Verification

- `python -m py_compile collector/client_dialog.py` -> OK
- `python -X utf8 tests/test_collector.py` -> `481/481`

### Dirty files intentionally left alone

- `artifacts/`
- local backups `config/clients.json.bak-*`
- untracked tools:
  - `tools/build_monetization_doc.py`
  - `tools/debug_batch_today.py`
  - `tools/debug_batches.py`

## Handoff Update - 2026-05-07 11:05 +05:00

### Saida help / startup UX closure

- Closed the Saida help theme end-to-end:
  - separate HTML guide for Saida is now part of the working tree
  - `/guide` became role-aware; `/help` added as alias
  - `show_help_doc` no longer depends on the failing `cmd_help` callback path
  - startup admin message now reflects the actual current scheduler instead of stale lines
  - passive stop/payment notifications for Saida now include a `❓ Что это значит?` path
  - full Saida stop-list callback now goes through a help-aware entry point

### Files intended for commit

- `bot/send_reports.py`
- `bot/debt_stop_control.py`
- `collector/payment_hold.py`
- `tests/test_collector.py`
- `tests/test_collector_regression_hermetic.py`
- `docs/Инструкция Саида.html`
- `docs/Инструкция по работе с ботом.html`

### Verification

- `python -m py_compile bot/send_reports.py` → OK
- `python -m py_compile bot/debt_stop_control.py` → OK
- `WHATSAPP_ENABLED=0 LIVE_SEND_ALLOWED=0 COLLECTOR_TEST_MODE=1 python -X utf8 tests/test_collector_regression_hermetic.py` → `15/15 OK`
- `WHATSAPP_ENABLED=0 LIVE_SEND_ALLOWED=0 COLLECTOR_TEST_MODE=1 python -X utf8 tests/test_collector.py` → `408/408 OK`

### Safety proof

- Test runs were hermetic:
  - live WhatsApp disabled
  - test mode enabled
  - logs show `saida notification suppressed` / `observer notification suppressed`
  - production state integrity checks and SHA-256 watchers stayed green
- No evidence that test messages were sent after the run; post-test logs only show normal bot jobs.

### Dirty files intentionally left alone

- `autoagent/orchestrator_agents.json`
- `autoagent/task_prompt.txt`
- `collector/approval_flow.py`
- `collector/client_dialog.py`
- `collector/collection_agent.py`
- `collector/collections_engine.py`
- `collector/manager_dialog.py`
- local backups `config/clients.json.bak-*`
- untracked `site/`

## Handoff Update - 2026-05-06 17:20 +05:00

### Manager dialog migration: manager_chat_id -> dialog_id

- Continued the isolated architectural migration of collector manager dialogs away from the single-slot `manager_chat_id` state key.
- `collector/dialog_store.py` is now on a v2 container model:
  - `dialogs[dialog_id]`
  - `active_by_chat[manager_chat_id] -> [dialog_id, ...]`
- Added backward compatibility:
  - old flat `{str(manager_chat_id): dialog}` files migrate automatically on first read
  - migrated file is rewritten in v2 format
- Updated stale cleanup to work on inner `dialogs` map rather than the old flat dict.

### Files changed

- `collector/dialog_store.py`
- `collector/manager_dialog.py`
- `tests/test_collector.py`
- `tests/test_manager_dialog_sessions.py` (new)

### Behavior changed

- One manager can now hold multiple active collector dialog sessions safely.
- Callback data moved to the new format:
  - `col|<action>|<dialog_id>`
- Legacy callback format is still tolerated:
  - `col_<action>_<manager_chat_id>`
  - if more than one active dialog exists for that manager, legacy callback is rejected safely instead of corrupting state
- Text routing now uses dialog-level targeting:
  - if exactly one active text-awaiting dialog exists -> route there
  - if multiple active dialogs exist -> do not guess, send a warning and require button-driven interaction
- Reminder/escalation updates now write by `dialog_id`, not by `manager_chat_id`.

### Verification

- `python -m py_compile collector/dialog_store.py` -> OK
- `python -m py_compile collector/manager_dialog.py` -> OK
- `python -X utf8 tests/test_manager_dialog_sessions.py` -> `10/10`
- `WHATSAPP_ENABLED=0 LIVE_SEND_ALLOWED=0 python -X utf8 tests/test_collector.py` -> `335/335`

### Remaining caution

- Legacy callback compatibility is intentionally best-effort only.
- If a manager clicks an old pre-migration inline button while several dialogs are active, the bot now refuses safely instead of picking a random session.
- `collector/collections_db.py` lightweight phone/name pending helpers were intentionally not migrated in this phase because they are not used as the active routing source in `manager_dialog.py`.

### Dirty files intentionally left alone

- `autoagent/orchestrator_agents.json`
- `autoagent/task_prompt.txt`
- deleted/removed non-product docs/log artifacts under `audit/`
- local backups `config/clients.json.bak-*`
- untracked `site/`

## Handoff Update - 2026-05-06 16:05 +05:00

### Documentation consolidation

- Consolidated scattered project docs, audits, context notes, and duplicate `.md` files into:
  - `PROJECT_ENCYCLOPEDIA.md`
- Updated `AGENTS.md` navigation rules so future work reads:
  - `AGENTS.md`
  - `PROJECT_ENCYCLOPEDIA.md`
  - `repo_map.json`
- Left `SESSION_CONTEXT.md` as the live handoff journal.

### What was intentionally kept

- `AGENTS.md` — active engineering protocol
- `PROJECT_ENCYCLOPEDIA.md` — single consolidated knowledge base
- `SESSION_CONTEXT.md` — live session/handoff log
- `openclaw/SOUL.md`
- `openclaw/skills/debt_collector.md`
- `autoagent/.ai_reviews/REVIEW_TEMPLATE.md`

### What was removed from active docs

- deleted duplicate root docs:
  - `CLAUDE.md`
  - `gpt1c.md`
- deleted dated audit/history markdown set under `audit/`
- deleted dated orchestrator context:
  - `autoagent/ORCHESTRATOR_CONTEXT_2026-04-23_v2.md`

### External C: findings folded into encyclopedia

- `C:\Users\user\.codex\memories\gpt1c-python-audit-context.md`
- `C:\Users\user\.claude\projects\e--GPT1C-Processor-analitica\memory\*`
- `C:\_migration_to_C_drive_20260429\backup_GPT1C_Processor_analitica_20260429_203309\*`
- `C:\_migration_to_C_drive_20260429\collector_crm_*_backup_*\*`

These were treated as auxiliary/external memory or archive, not as active source-of-truth docs.

### Verification

- Product code was not changed.
- No Python/runtime tests were required.
- Repo markdown inventory after cleanup is effectively:
  - `AGENTS.md`
  - `PROJECT_ENCYCLOPEDIA.md`
  - `SESSION_CONTEXT.md`
  - retained prompt assets (`openclaw/*`, `.ai_reviews/REVIEW_TEMPLATE.md`)

### Dirty files intentionally left alone

- `autoagent/orchestrator_agents.json`
- `autoagent/task_prompt.txt`
- local backups `config/clients.json.bak-*`
- untracked `site/`

### Next recommended action

- If continuing product work: build manager promise-quality analytics (`Договорились` used / broken / accepted / rejected by manager).
- If continuing operations: review real backlog in `logs/saida_payment_holds.json` and decide whether partial-payment path needs stricter automation/escalation.

## Handoff Update - 2026-05-06 13:58 +05:00

### Closed in this session

- `approval_flow` director visibility gap on manager promises is now closed at the summary level:
  - `collector/approval_flow.py`
  - `bot/send_reports.py`
  - `tests/test_collector.py`
  - added read-only analytics over `logs/wa_agreed_promises.json`
  - director can open `🤖 Коллектор -> 🤝 Обещания менеджеров` and see per-manager totals:
    - total promises
    - in control
    - fulfilled
    - broken
    - rejected by director
    - overdue active promises

### Verification

- `python -m py_compile collector/approval_flow.py` -> OK
- `python -m py_compile bot/send_reports.py` -> OK
- `WHATSAPP_ENABLED=0 LIVE_SEND_ALLOWED=0 python -X utf8 tests/test_collector.py` -> `327/327`

### Current product state

- Director now has both:
  - point control over each `Договорились` in batch review
  - aggregate quality view over accumulated promises by manager
- This closes the visibility gap where promises existed in `wa_agreed_promises.json` but were not visible as manager discipline metrics.
- The new analytics is read-only: it does not alter batch routing, deadlines, stop-flow, or promise lifecycle.

### Dirty files intentionally left alone

- `autoagent/orchestrator_agents.json`
- `autoagent/task_prompt.txt`
- untracked `audit/*`
- local backups `config/clients.json.bak-*`
- untracked `site/`

### Next recommended action

- If continuing product work: add similar aggregate reporting for Saida (`pending_saida`, oldest age, closed today, overdue SLA).
- If continuing control logic: decide whether partial-payment path should auto-resolve any shipment/stop state or always stay manual.

## Handoff Update - 2026-05-06 14:04 +05:00

### Closed in this session

- `Saida backlog` director visibility gap is now closed:
  - `collector/payment_hold.py`
  - `bot/send_reports.py`
  - `tests/test_collector.py`
  - added read-only backlog analytics over `logs/saida_payment_holds.json`
  - director can open `🤖 Коллектор -> 📋 Саида backlog` and see:
    - open `pending_saida`
    - oldest age
    - over-SLA count
    - over-bypass count
    - closed today
    - per-manager backlog split
    - top oldest pending clients

### Verification

- `python -m py_compile collector/payment_hold.py` -> OK
- `python -m py_compile bot/send_reports.py` -> OK
- `WHATSAPP_ENABLED=0 LIVE_SEND_ALLOWED=0 python -X utf8 tests/test_collector.py` -> `330/330`

### Current product state

- Director now has read-only aggregate control for both major human bottlenecks:
  - manager promises (`Договорились`)
  - Saida payment-confirmation backlog
- No stop/payment workflow was changed by this step; only visibility and control reporting were added.

### Dirty files intentionally left alone

- `autoagent/orchestrator_agents.json`
- `autoagent/task_prompt.txt`
- untracked `audit/*`
- local backups `config/clients.json.bak-*`
- untracked `site/`

### Next recommended action

- Decide whether partial-payment path should stay fully manual or also receive director-facing analytics/escalation.
- If moving into operations: use the new Saida backlog screen to validate real counts/oldest-age on production data and set a business SLA threshold.

## Handoff Update - 2026-05-06 16:55 +05:00

### Closed / verified in this session

- `runtime-state cleanup before acceptance`:
  - `logs/collector_dialogs.json` already sterile (`{}`), no test-tail cleanup required
- `Phase 3B unified collector logging`:
  - verified against current `master`
  - live `collector/*.py` modules already use `get_collector_logger(__name__)` / profile helpers
  - no legacy collector logger init remained in active modules
- `legacy collector/debtors_contacts.json` confusion:
  - verified no active `collector/debtors_contacts.json` file remains
  - active collector code reads `config/debtors_contacts.json`
- `NET-SSL-TELEGRAM-01`:
  - treated as external/transient for now
  - no fresh `CERTIFICATE_VERIFY_FAILED` reproduced in current `2026-05-06` runtime logs
- `partial payment visibility gap` is now closed with read-only director analytics:
  - `collector/payment_hold.py`
  - `bot/send_reports.py`
  - `tests/test_collector.py`
  - new director screen in `🤖 Коллектор -> 🔸 Частичные оплаты`
- `Saida help text` corrected where it falsely implied every reply auto-clears stop:
  - `bot/debt_stop_control.py`

### Commit

- `930c963` — `feat(stop): add partial payment analytics and clarify Saida help`

### Verification

- `python -m py_compile bot/debt_stop_control.py` -> OK
- `python -m py_compile collector/payment_hold.py` -> OK
- `python -m py_compile bot/send_reports.py` -> OK
- `python -X utf8 tests/test_crm_regression.py` -> `14/14`
- `WHATSAPP_ENABLED=0 LIVE_SEND_ALLOWED=0 python -X utf8 tests/test_collector.py` -> `332/332`

### Current status of formerly open items

- `Phase 3B unified logging` -> closed by verification on current HEAD
- `legacy debtors_contacts path confusion` -> closed by verification on current HEAD
- `runtime-state cleanup` -> no-op, state already sterile
- `NET-SSL-TELEGRAM-01` -> no code action, currently non-reproducible
- `partial payment analytics/escalation gap` -> read-only analytics closed
- `manager_dialog state key bound to manager_chat_id` -> still open architectural risk

### Remaining real open technical item

- `manager_dialog / dialog_store keying by manager_chat_id`
  - code still uses one-dialog-per-manager slot in:
    - `collector/dialog_store.py`
    - `collector/manager_dialog.py`
    - related pending helpers in `collector/collections_db.py`
  - not changed in this session because it requires an isolated migration and backward-compatibility plan

### Dirty files intentionally left alone

- `autoagent/orchestrator_agents.json`
- `autoagent/task_prompt.txt`
- deleted/removed non-product docs/log artifacts under `audit/`
- local backups `config/clients.json.bak-*`
- untracked `site/`

### Next recommended action

- If continuing code work: isolate and redesign `manager_dialog` storage key away from plain `manager_chat_id`
- If moving into operations: run a real event-driven approval smoke-test on next incoming debt batch and validate:
  - manager previews
  - tight-window escalation
  - director summary
  - `🔸 Частичные оплаты` screen against live data

## Handoff Update - 2026-05-06 15:31 +05:00

### Operational cleanup (no code changes)

- Reviewed live `logs/saida_payment_holds.json` after mass Saida reminder/bypass burst seen in `logs/send_reports.log`.
- Confirmed backlog was mixed:
  - `51` total `pending_saida`
  - `40` stale entries older than `48h`
  - `11` still-recent entries within `48h` (`7` within `24h`)
- Per user direction, cleaned only the stale historical tail and preserved recent live cases.

### Files created / changed

- Full archive saved:
  - `logs/saida_payment_holds.archive_20260506_153047.json`
- Removed stale subset saved separately:
  - `logs/saida_payment_holds.removed_pending_gt48h_20260506_153047.json`
- Working queue rewritten:
  - `logs/saida_payment_holds.json`

### Result

- Working `pending_saida` queue reduced from `51` to `11`
- Remaining live queue age band after cleanup:
  - min age `23.49h`
  - max age `30.87h`
- No product code changed, no tests required for this operational data cleanup.

### Dirty files intentionally left alone

- `autoagent/orchestrator_agents.json`
- `autoagent/task_prompt.txt`
- untracked `audit/*`
- local backups `config/clients.json.bak-*`
- untracked `site/`

---

## HANDOFF 2026-05-11 — soft_positive fix, стоп-лист, штрафные баллы

**Тесты: 491/491. HEAD: 6afb3d9 (pushed)**

### Коммиты сессии
f526c5f fix(approval): escalation_reason через setdefault + close_reason
6b2f309 test(approval): покрытие close_reason / escalation_reason
cb1e21f test(approval): T10m — prior escalation_reason при admin send too_late
cba6fdf fix(dialog): soft_positive не эскалирует на чистом приветствии
b18d765 feat(dialog): исламские и казахские приветствия в _PURE_GREETINGS
83bfca4 feat(dialog): саламатсызбе/саламатсыз ба
a424813 style(dialog): выровнены отступы в _PURE_GREETINGS
eabb62d fix(collector): harden soft-positive routing
dd6ec49 fix(stop-list): нормализация имён + новая цепочка Саида→Админ
b7d0dbe feat(penalty): штрафные баллы — новый модуль
f26d6a6 fix(penalty): 2-й пропуск = 2 000 тг
b0815c1 fix(penalty): предупреждение показывает 2 000 тг
b9de9e4 fix: форматирование _n(), guard игнорирует suggested_reply, тесты
6afb3d9 chore(penalty): версия v1.0.2, changelog

### Ключевые изменения

approval_flow.py v1.1.10 — close_reason + escalation_reason через setdefault

client_dialog.py v1.1.5
- _normalize_text исправлен ("".join вместо " ".join)
- _is_greeting_only() + _PURE_GREETINGS (25+ вариантов: рус/каз/ислам)
- _is_acknowledgement_only() подключена к soft_positive-ветке
- Guard ИГНОРИРУЕТ suggested_reply от AI — safe-текст всегда имеет приоритет
- E2E тесты с непустым suggested_reply проверяют что guard блокирует AI-ответ

collection_agent.py v1.1.1 — промпт: soft_positive требует платёжное слово

debt_stop_control.py v1.0.15
- Нормализация имён в _build_candidates() — нечёткое совпадение
- STOP_PAID_THRESHOLD: 5000 → 100 тг
- Цепочка: debt<=100 → Саида → Админ (3 кнопки) → уведомление менеджеру

approval_penalty.py v1.0.2 (новый модуль)
- Формула: 1й=0, 2й=2000, N>=3: N*1000*2, частичный -10%
- Немедленное уведомление менеджеру после каждого пропуска
- check_recent_batches() каждые 30 мин, отчёт в последний день месяца 23:00
- Форматирование через _n(): "2 000", не "2,000"

### Дословные тексты уведомлений

Штрафы → менеджеру (1-й пропуск):
  ⚠️ Предупреждение
  Вы не ответили в окне согласования рассылки 11.05.2026.
  Это первый пропуск в этом месяце — штраф не начисляется.
  ⚠️ Следующий пропуск: 2 000 тг

Штрафы → менеджеру (2-й пропуск):
  🔴 Штраф: 2 000 тг
  Пропущено окно согласования рассылки 11.05.2026.
  Пропусков за месяц: 2 | Итого: 2 000 тг

Штрафы → Саиде+Вам (конец месяца): таблица по всем менеджерам + ИТОГО
Штрафы → менеджеру (конец месяца): только его строка + "учтёт при расчёте зарплаты"

Стоп-лист → Саиде: "💰 [Клиент] долг<=100 тг. Подтверди оплату? [✅/❌]"
Стоп-лист → Админу: 3 кнопки — Снять / Предоплата 100% / Чёрный список
Стоп-лист → менеджеру: уведомление о решении Админа (все три ветки)

### Следующие шаги

1. Проверить в бою (12.05, следующий рабочий день)
2. collections_engine.py: фильтр стоп-клиентов без движений в WA-батче
3. Тест для _handle_saida_zeropay_confirm/deny

---

## HANDOFF 2026-05-13 - deferral discipline monitoring

������: �������� �����������, �� ��������� �� ������ handoff.

��� ���������:
- `collector/payment_deferrals.py` v1.0.1: read-only ���������� �������� � ���������� ��������� � `logs/deferral_violations.json` � ������ �������� ������ `� ����` / `� ����������`, `violation_count`, ������� � ������� ��������.
- `collector/collections_engine.py` v1.5.3: `sync_deferral_discipline(debtors)` ���������� ����� `classify_debtors(...)` � `run()`, `run_approval_preview()` � refresh-path `--send-approved`.
- `bot/send_reports.py` v9.4.76: � ���� ���������� ��������� ������ `? ��������` (`collector_deferral_stats`) � admin-������� �� ���������� � ��������.
- `tests/test_collector.py`: �������� ������������� �������� �� 2 ������ ������ deferred-�������: ���� � ���������� � ���� ��� ���������.

��������:
- `python -m py_compile collector\\payment_deferrals.py` -> OK
- `python -m py_compile collector\\collections_engine.py` -> OK
- `python -m py_compile bot\\send_reports.py` -> OK
- `python -X utf8 tests\\test_collector.py` -> `596/596`

��������� ��� ���������:
- ��������� `config/clients.json.bak-*`
- `artifacts/`
- `tools/debug_batch_today.py`
- `tools/debug_batches.py`
- `tools/build_monetization_doc.py`

��������� ���:
- ������ � ��� ����� �������� ���� � ������� ��������� debt-snapshot: ��������� ���������� `logs/deferral_violations.json` � admin-����� `? ��������`.

---

## HANDOFF 2026-05-14 - deferral overdue propagation fix

Статус: локально проверено, готово к commit/push.

Что исправлено:
- `collector/collections_engine.py` v1.5.4: для клиентов с отсрочкой raw возраст долга (`debt_age_days`) отделён от фактической просрочки по отсрочке (`effective_overdue_days`); эти поля теперь проходят через preview, approved-batch и client dialog/send path без потери.
- `collector/client_dialog.py` v1.1.8: escalation-текст менеджеру больше не врёт `N дн. просрочки` для клиентов, которые ещё в сроке по отсрочке; теперь отдельно показываются возраст долга, отсрочка и effective overdue.
- `collector/collection_agent.py` v1.1.2: fallback client templates учитывают отсрочку и не завышают просрочку в тексте.
- `collector/approval_flow.py` v1.1.13: manager preview показывает `Возраст остатка`, `Отсрочка` и `Просрочка по отсрочке` / `По отсрочке еще в срок`.
- `tests/test_collector.py`: добавлены регрессии на нормализацию deferred-клиента, первый effective overdue day, manager preview/escalation text и fallback client copy.

Доказательства:
- кейс `О Цех ТОО High product ул МОЙЫНТЫ 18` теперь покрыт тестами как deferred-case:
  - raw age `10`
  - deferral `10`
  - effective overdue `0`
  - тексты больше не содержат ложное `10 дн. просрочки`
- `_send_approved_client()` теперь передаёт `debt_age_days`, `deferral_days`, `effective_overdue_days` и в `generate_message(...)`, и в `start_client_dialog(...)`.

Проверки:
- `python -m py_compile collector\collections_engine.py` -> OK
- `python -m py_compile collector\client_dialog.py` -> OK
- `python -m py_compile collector\collection_agent.py` -> OK
- `python -m py_compile collector\approval_flow.py` -> OK
- `python -m py_compile collector\payment_deferrals.py` -> OK
- `python -m py_compile bot\send_reports.py` -> OK
- `python -X utf8 tests\test_collector.py` -> `610/610`
- `python -X utf8 tests\test_phase2_safe_send.py` -> `PHASE2 SAFE SEND TESTS PASSED`

Важно:
- при первом прогоне `tests/test_collector.py` тест-харнесс временно дотронулся до `logs/saida_payment_holds.json`, но второй прогон прошёл чисто и SHA watcher подтвердил, что prod state больше не менялся во время suite.

Оставлено без изменений:
- локальные `config/clients.json.bak-*`
- `artifacts/`
- `tools/debug_batch_today.py`
- `tools/debug_batches.py`
- `tools/build_monetization_doc.py`

Операционный следующий шаг:
- рестарт бота, чтобы боевой runtime подхватил фиксы deferral overdue / preview text.

## HANDOFF 2026-05-14 — collector batch hardening after production audit

### Что закрыто

- send-approved refresh больше не теряет admin-approved клиентов с eview_action="manager_review".
- Preview/send-approved decision теперь учитывает wa_dialog_suppress ещё до live-send.
- Для stop_status="exception" введён grace-period через COLLECTOR_EXCEPTION_GRACE_DAYS.
- После escalate_to_manager(...) ставится короткий suppress-cooldown.
- В admin UI добавлена кнопка 🧭 Актуальный батч.

### Проверка

- python -m py_compile collector\collections_engine.py -> OK
- python -m py_compile collector\client_dialog.py -> OK
- python -m py_compile bot\send_reports.py -> OK
- $env:WHATSAPP_ENABLED='0'; ='0'; python -X utf8 tests\test_collector.py -> 620/620
- $env:WHATSAPP_ENABLED='0'; ='0'; python -X utf8 tests\test_phase2_safe_send.py -> passed
- python -X utf8 tests\test_project.py -> 110/110

### Root cause

- Preview path допускал manager_review, а send-refresh фильтровал только client_approval.
- Защита от повторного захода клиента держалась на active-dialog и не покрывала escalation cooldown + exception-кейсы.

---

## HANDOFF 2026-05-14 � approval_penalty reset floor: ������ WA/CRM ������ �� ������ ������� ����� ������� ������

### Root cause

- ������ reset ������ ������ `logs/approval_penalty_state.json`.
- ����� periodic job `check_recent_batches()` ������ ����������� ��� terminal WA-����� �������� ������ �� `logs/wa_approval_batches.json`.
- ���������� `check_crm_ignores()` ��� �������� ������� ������ CRM pending-������.
- ��-�� ����� ����� ������������� ������ ������ ������������ �� ������.

### ��� �������

- `collector/approval_penalty.py`
  - �������� `build_reset_state()`:
    - `{month, managers={}, wa_reset_floor, crm_reset_floor}`
  - `process_batch_penalties(...)` ������ ���������� batch, ���� ��� `created_at < wa_reset_floor`
  - `check_recent_batches(...)` �� backfill-�� WA-����� ������ `wa_reset_floor`
  - `check_crm_ignores(...)` �� �������� CRM pending-������ ������ `crm_reset_floor`

- `tests/test_collector.py`
  - ��������� ���������:
    - ������ WA batch �� ������������� ����� reset
    - direct `process_batch_penalties(...)` ���� �� ��������� ������ batch
    - ������ CRM pending �� ���������� ����� reset
    - ����� WA batch ����� reset ���������� ��������� ���������

### ������������ ��������

- `logs/approval_penalty_state.json` �������� ������� ������� � ����� �������:
  - `month = 2026-05`
  - `managers = {}`
  - `wa_reset_floor = 2026-05-14T21:49:19.309486+05:00`
  - `crm_reset_floor = 2026-05-14T21:49:19.309486+05:00`
- ����� ��������� �� ������:
  - `logs/approval_penalty_state.json.bak-20260514-reset2`

### ��������

- `python -m py_compile collector\approval_penalty.py` -> OK
- `python -X utf8 tests\test_collector.py` -> `635/635`

### �����

- ����� ������ runtime ��������� ������� ����, ����� ������� ����.
- ��� ����� state ��� �������� �� �����, �� ��� �������� ������� ������� ��������� ���� �� ������ �������� ������.

## 2026-05-15 12:20 � silent active dialogs no longer block daily resend
- Problem: clients like ��������� / ����� / ����-��� were skipped from preview with reason `������: active` because old client dialogs from 2026-05-12 stayed in `state=active` with `exchange_count=0` and no TTL.
- Root cause: collector only had expiry for `awaiting_payment_proof > 3d`; plain `active` dialogs with zero client replies blocked preview/send-approved forever.
- Fix:
  - `collector/client_dialog.py`: added `dialog_blocks_new_outreach()` and `SILENT_ACTIVE_RESEND_HOURS=24`; stale silent `active` dialogs (`exchange_count=0`, 24h+) no longer block a fresh send; `start_client_dialog()` supersedes them and increments `phone_silent_cycles`.
  - `collector/collections_engine.py`: preview and send-approved now use the shared helper instead of hardcoded active-dialog blocking.
  - `tests/test_collector.py`: regressions for stale silent dialog resend + send-approved path.
- Verification:
  - `python -m py_compile collector\client_dialog.py` OK
  - `python -m py_compile collector\collections_engine.py` OK
  - `python -X utf8 tests\test_collector.py` -> `640/640`
- Operational note: runtime restart is required for the new daily-resend policy to take effect in the live bot.

## 2026-05-15 12:45 � admin button for penalty reset
- Added admin-only collector UI action `?? ����� �������` in `bot/send_reports.py`.
- Flow:
  - `collector_penalty_reset_prompt` shows warning/confirm step.
  - `collector_penalty_reset_confirm` calls runtime helper and reports month + WA/CRM floors + backup file name.
- `collector/approval_penalty.py`:
  - added `reset_penalty_state()` helper;
  - creates backup of `logs/approval_penalty_state.json` before reset;
  - writes `build_reset_state()` result with `wa_reset_floor` / `crm_reset_floor`.
- Tests:
  - `tests/test_collector.py` now verifies helper reset clears managers, sets floor, and creates backup.
- Verification:
  - `python -m py_compile bot\send_reports.py` OK
  - `python -m py_compile collector\approval_penalty.py` OK
  - `python -X utf8 tests\test_collector.py` -> `643/643`
- Note: no commit yet in this step.

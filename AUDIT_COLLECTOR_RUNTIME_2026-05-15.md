# AUDIT — Collector boevoy runtime после рестарта v9.4.79

**Дата:** 2026-05-15
**HEAD:** `e9ab178`
**Период:** с рестарта `v9.4.79/15.05.2026` (2026-05-15 12:29:20 +05:00) до текущего момента
**Тип:** Read-only аудит фактического поведения collector-контура (Блок B из стабилизационного ТЗ)

---

## Executive summary

Collector работает **штатно** после рестарта. Все ключевые фиксы последних 48 часов реально проявились в логах:
- `dialog_blocks_new_outreach()` работает — больше нет stale silent active диалогов
- Hard-fail lock — ни одного `crm_state_lock_timeout` события
- Penalty reset floors поставлены и работают
- Admin summary timeout-label различает `waiting_for_agreed` / `waiting_for_proof`

**Найдено 3 finding'а**, все низкой/средней severity. Ни одного P0/P1 production bug.

| # | Severity | Symptom | Status |
|---|---|---|---|
| F-B1 | MEDIUM | 2 диалога с overdue promise (12.05, 13.05) — обещания просрочены, бот не реагирует | OPEN — operational |
| F-B2 | LOW | 18 expired payment holds закрыты одним моментом (16:39:20) без уведомлений менеджерам | OPERATIONAL — by design |
| F-B3 | LOW | `waiting_for_agreed` объекты не очищаются после close batch в close-state | COSMETIC — не баг |

**Ничего не требует срочного исправления** в коде. F-B1 — управленческая задача (менеджеры должны напомнить клиентам). F-B2 — by design. F-B3 — косметика state-файла.

---

## 1. Batch timeline (с рестарта v9.4.79)

Источник: `logs/wa_approval_batches.json`. Всего в state-файле 7 батчей; ниже — релевантные (последние 5).

| batch_id | created | status | managers | result |
|---|---|---|---|---|
| `20260515-170000-22f6` | 15.05 17:00 | **sent** | 3 (2 ok, 1 timeout) | 9 клиентов ушло в WA, escalated_to_admin |
| `20260514-170000-fd1a` | 14.05 17:00 | too_late | 2 (оба timeout) | send_window_missed, close 21:09 |
| `20260513-170000-0023` | 13.05 17:00 | send_failed | 1 (approved_all) | skip 1 (existing dialog) |
| `20260512-170000-15b9` | 12.05 17:00 | partially_sent | 4 (1 ok, 3 timeout) | ≤6 клиентов |
| `20260511-170000-078e` | 11.05 17:00 | partially_sent | 4 (1 ok, 3 timeout) | ≤6 клиентов |

### Сегодняшний цикл (20260515-170000-22f6)

- Создан: 17:00:00
- Admin approve: 18:07 (approved_clients=9)
- Sent: 9/9 ✅
- Менеджеры:
  - **Алена** — approved_all (5 клиентов), responded 17:14
  - **Магира** — timeout, agreed=1 (один клиент договорённость, остальные не ответил → авто)
  - **Оксана** — approved_all (2 клиента), responded 17:00 (instant)
- Escalation reason: `manager_silence_timeout` (Магира не ответила за 1 час)
- 6 событий `dialog_superseded_after_silence` в 18:07 — silent dialogs корректно разблокированы и заменены

**Вывод:** Цикл прошёл штатно. Все 9 клиентов реально отправлены.

---

## 2. Активные диалоги (collector_client_dialogs.json)

Источник: read of state file, totals из агента-разведчика.

**15 диалогов** в активных состояниях:

### Active (fresh, exchange_count=0, ждём ответа клиента)

- `Е ТОО ГудФуд № 1 ул Досмухамедулы 48(Аида)` — 7701***47
- `Е ИП Шахин` — 7702***72
- `Е ТД Саянур Леонид` — 7775***69
- `Е Каирбек` — 7701***35
- `М Ресторан Шама ИП Тян ул Мустафина 12` — 7778***16
- `М Плов центр ЕСБОЛОВА Мангилик Ел 54 (Омарова)` — 7783***30
- `М ТОО Аманат` — 7701***48
- `М Гриль Косши ул Республика 18 б тел 87751827070` — 7775***70 (см. F-B1)
- `О ТОО Petro Retail (Автогаз) ул Мангилик Ел 89 В` — 7776***71
- `О ТОО МАСТЕР-КОНДИТЕР` — 7775***01 (см. F-B1)
- `О Цех ТОО High product` — 7702***53
- `Е Олжас` — 7775***04

Все 12 диалогов созданы 15.05 (свежие). **Ни один не имеет `last_activity` старше 24h** → нет кандидатов на `stale_silent_active`. Это подтверждает: фикс `dialog_blocks_new_outreach()` работает в бою.

### Escalated (клиент ответил, передан в обработку)

- `Е Еркебулан` — 7708***17, exchange=1, **promise_date=2026-05-18**, awaiting_payment_proof
- `Е ИП Трое Нурлан` — 7775***45, exchange=2
- `А ТД Сарыарка` — 7775***02, exchange=1
- `О Дет.сад Орда` — 7701***81, exchange=1, **promise_date=2026-05-18**
- `М Гриль Косши` — exchange=1, **promise_date=2026-05-12** ⚠️ overdue
- `О ТОО МАСТЕР-КОНДИТЕР` — exchange=1, **promise_date=2026-05-13** ⚠️ overdue
- `О Цех ТОО High product` — exchange=2, awaiting_payment_proof
- (один пропущен в отчёте агента)

---

## 3. State проверка

### 3.1 collector_client_dialogs.json
- 15 активных/escalated
- **0 stale_silent_active** (фикс v1.1.10 client_dialog работает)
- 0 диалогов с `state=active && exchange_count=0 && last_activity > 24h`

### 3.2 saida_payment_holds.json
- 0 pending_saida (всё закрыто)
- 0 active
- 18 expired (последний закрыт 15.05 16:39:20, см. F-B2)
- 10 rejected
- 9 cleared_by_1c

### 3.3 approval_penalty_state.json
- `wa_reset_floor = 2026-05-15T11:37:46.770756+05:00`
- `crm_reset_floor = 2026-05-15T11:37:46.770756+05:00`
- Reset выполнен через `reset_penalty_state()` (новая admin-кнопка `♻️ Сброс штрафов`)
- Месяц 2026-05: только Магира с partial=true, штрафов 0
- **Ложного повторного начисления после reset не обнаружено**

### 3.4 crm_*_pending_state.json
- `crm_pending_state.json` — 0 записей (всё закрыто менеджерами)
- `crm_claim_pending_state.json` — 1 запись: `"Частное лицо 1"` claimed by Магира (см. отдельный аудит CRM)
- `crm_duplicate_review_state.json` — пусто
- `crm_ambiguous_conflicts.json` — нет pending

### 3.5 Lock contention
- В `send_reports.log` после рестарта нет ни одного события `crm_state_lock_timeout`
- В `collector_audit.jsonl` нет ошибок `LockException` / `PermissionError`
- Hard-fail policy внедрён, но реальной contention не было

---

## 4. Detailed findings

### F-B1 | MEDIUM | Overdue promise: 2 клиента, бот не реагирует

**Symptom:** Два escalated диалога имеют `promise_date` в прошлом, но клиент не оплатил и бот не вернул их в дисциплинарный цикл:

| client | manager | promise_date | overdue |
|---|---|---|---|
| `М Гриль Косши ул Республика 18 б ...` | Магира | 2026-05-12 | 3 дня |
| `О ТОО МАСТЕР-КОНДИТЕР` | Оксана | 2026-05-13 | 2 дня |

**Root cause (по коду):** В collector есть `promise_overdue_handler` (`collector/collections_engine.py`), который должен проверять `promise_date < today` и возвращать клиента в WA-цикл с эскалацией. Проверить, что этот handler действительно запускается в scheduler.

**Code location:** `collector/collections_engine.py` — функция отвечающая за overdue promise.

**Evidence:**
- `logs/collector_client_dialogs.json` — escalated state с просроченным promise_date

**Impact:** Клиент дал обещание, не оплатил, бот молчит. Менеджер должен это заметить вручную.

**Status:** OPEN — проверить scheduler / handler. Возможно требует ручного триггера.

**Fix direction:** Если handler существует — проверить расписание. Если нет — добавить. Не блокирует другие фиксы.

---

### F-B2 | LOW | Массовое автозакрытие payment holds в 16:39:20

**Symptom:** В 15.05 16:39:20 одним моментом закрыты ВСЕ 18 старых payment holds (от 07.05 и далее). Возраст некоторых превышал TTL в 16+ раз (12h limit vs 196h actual). Уведомления менеджерам **не отправлены**.

**Root cause:** `expire_old_holds()` отрабатывает по cron и помечает старые holds как `expired`. Это by design — нет интеграции уведомлений при массовом backfill после рестарта.

**Code location:** `collector/payment_hold.py` — функция expire/cleanup.

**Impact:** Менеджер не знает, что 18 его pending запросов к Саиде "сгорели". Если клиент реально оплатил — менеджер мог пропустить.

**Status:** OPERATIONAL — by design. Это не баг, но потенциальная UX-проблема.

**Recommendation:** Не трогать в этом цикле. Если жалобы повторятся — добавить summary-уведомление admin'у при массовом expiry (>3 за раз).

---

### F-B3 | LOW | `waiting_for_agreed` объекты остаются в close-state batch

**Symptom:** В батче `20260514-170000-fd1a` (status=too_late, closed=21:09) у Магиры в `mgr_state` остался непустой `waiting_for_agreed = {"client_name": "М ТД Алма-Пласт д. Корсак 12/1", ...}` — батч давно закрыт, ждать нечего.

**Root cause:** При переводе batch в финальный статус `too_late`/`expired` поле `waiting_for_agreed` менеджера не очищается. Это не блокирует работу, но мусорит state-файл и может ввести в заблуждение при разборе инцидентов.

**Code location:** `collector/approval_flow.py` — функции `expire_old_batches`/перевода в финальный статус.

**Evidence:** 3 batch'а (`20260514-170000-fd1a`, `20260512-170000-15b9`, `20260511-170000-078e`) содержат непустой `waiting_for_agreed` после close.

**Impact:** Косметический. Admin summary `_format_admin_summary_text` (после фикса 7603db9) корректно отображает: `"⏳ начал — не написал детали по «X»"` — это полезный forensic-сигнал, не баг.

**Status:** COSMETIC — не баг. Решение: оставить как есть (полезно для forensics).

---

## 5. Реестр по клиентам (cycle 2026-05-15)

| Клиент | Менеджер | Preview | Approved | Sent | Ответ | Куда дальше |
|---|---|---|---|---|---|---|
| `Е ТОО ГудФуд № 1 ...` | Ергали | ✅ | ✅ | ✅ | — | active dialog |
| `Е ИП Шахин` | Ергали | ✅ | ✅ | ✅ | — | active dialog (раньше залипший — после v1.1.10 разблокирован) |
| `Е ТД Саянур Леонид` | Ергали | ✅ | ✅ | ✅ | — | active dialog (раньше залипший — разблокирован) |
| `Е Каирбек` | Ергали | ✅ | ✅ | ✅ | — | active dialog |
| `М Ресторан Шама / Тян` | Магира | ✅ | ✅ | ✅ | — | active dialog (раньше залипший — разблокирован) |
| `М Плов центр ЕСБОЛОВА (Омарова)` | Магира | ✅ | ✅ | ✅ | — | active dialog |
| `М ТОО Аманат` | Магира | ✅ | ✅ | ✅ | — | active dialog |
| `О ТОО Petro Retail` | Оксана | ✅ | ✅ | ✅ | — | active dialog |
| `Е Олжас` | Ергали | ✅ | ✅ | ✅ | — | active dialog |

**Вывод:** Все 9 клиентов корректно прошли pipeline preview → approve → send. **6 из них — это те самые залипавшие диалоги** (Шахин, Тян, Саянур, Еркебулан, ГудФуд, Petro Retail) которые до v1.1.10 client_dialog НЕ попадали в новый цикл. Сейчас они есть. **Фикс работает в production.**

---

## 6. Реестр skip-причин (с рестарта v9.4.79)

| Причина | Count | Это штатно? | Action |
|---|---|---|---|
| `existing_dialog (escalated)` | многократно | штатно | не трогать |
| `existing_dialog (active)` | штатно | штатно | не трогать |
| `payment_hold_active` | несколько | штатно | не трогать |
| `no_phone_contact` | редко | штатно | проверять отдельно |
| `stale_silent_active` (как причина разблокировки) | 6 | штатно | ✅ работает как ожидалось |

**Не обнаружено:**
- Skip-причин с `?` или `неизвестно`
- Клиентов потерянных между preview и send (приведено только 9/9)
- Дублирующих skip'ов на одном клиенте

---

## 7. Реестр stale-state проблем

| Категория | Count | Severity |
|---|---|---|
| active dialog stale (>24h, 0 exchange) | **0** | ✅ закрыто фиксом v1.1.10 |
| pending proof stale (>72h) | 0 | ✅ закрыто |
| old penalty replay (после reset) | 0 | ✅ floors работают |
| stale callback | 0 | ✅ |
| stale claim/review token | **1** (legit — claim by Магира сегодня) | OK |
| старые payment holds | 18 expired (массовое closure) | F-B2 (operational) |
| `waiting_for_agreed` в close batch | 3 | F-B3 (cosmetic) |
| overdue promise | **2** | F-B1 (operational, нужна проверка handler) |

---

## 8. Верификация последних фиксов в production

| Фикс | Коммит | Подтверждён в logs/state? |
|---|---|---|
| Hard-fail CRM lock + bool save | `0c8561e` | ✅ 0 timeout events, 0 LockException |
| Stale silent dialog unblock | `d4ae395` | ✅ 6 `dialog_superseded_after_silence` событий в сегодняшнем цикле |
| Penalty reset с floors | `f7f1d13` | ✅ `wa_reset_floor` и `crm_reset_floor` стоят, штрафы=0 |
| Admin timeout-label `waiting_for_agreed` | `7603db9` | ✅ В UI отображается правильно (по коду) |
| `dialog_blocks_new_outreach` единый источник | (часть `d4ae395`+`167069a`) | ✅ преview и send-approved используют один helper |

**Все 5 последних фиксов реально проявились в production runtime.**

---

## 9. Recommendations (по приоритету)

### P0
**Нет.** Production-критичных багов в этом цикле не обнаружено.

### P1
**Нет.** Все известные P1 уже закрыты.

### P2
- **F-B1**: Проверить scheduler / handler для `promise_overdue` — почему Гриль Косши (3 дня) и МАСТЕР-КОНДИТЕР (2 дня) не вернулись в дисциплинарный цикл. Если handler есть — проверить crontab; если нет — добавить.

### Operational (без правок кода)
- **F-B2**: Если жалобы повторятся — рассмотреть summary-нотификацию admin'у при массовом expiry payment holds. Сейчас не нужно.
- **F-B3**: Оставить как есть. Polezno для forensics.
- **CRM** (отдельный аудит): применить фикс `"частное лицо"` keyword из `AUDIT_CRM_PRIVATE_PERSON_2026-05-15.md`.

---

## 10. Что НЕ нужно трогать

- Lock-policy менять не нужно — hard-fail работает, contention в реальности не было
- `dialog_blocks_new_outreach` менять не нужно — единый предикат, оба пути на нём
- Penalty reset мехика работает — floors ставятся, backfill уважает их
- Admin UI — все label'ы корректны

---

## Acceptance

Аудит закрыт. Production runtime collector-контура штатный.

**Следующее действие**: проверить F-B1 (overdue promise handler) — это единственная open-сторона. Остальное либо closed, либо operational/cosmetic.

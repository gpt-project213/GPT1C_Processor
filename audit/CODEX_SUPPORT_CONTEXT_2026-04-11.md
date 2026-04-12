# CODEX SUPPORT CONTEXT — 2026-04-11

Роль документа: master context для Claude. Codex ничего не правил в бизнес-логике; использованы только чтение git/audit/code/JSON и запись новых audit/*.md.

## 1. Что прочитано

### Git commits
Прочитаны `git show --stat --name-status --format=fuller` по обязательным коммитам:

- `485e57a` — weekend silent mode + enhanced final admin approval
- `fa3baea` — BUG-CRM-1 + OP-4 safety guards
- `beec0cc` — audit INDEX + финализация CRM audit + удаление корневых дублей
- `60a1eef` — добавление audit reports/architecture materials
- `e083f97` — WhatsApp approval UX manager/admin consent flow
- `31f3bc8` — message routing audit
- `5d9a3fa` — LIVE_SEND_ALLOWED safeguard + regression tests
- `18a8ea8` — emergency fix unauthorized WhatsApp sends
- `ed139cf` — CRM flow name defer pause
- `6f63213` — TZ hardcode + template bugs
- `dbe8c43` — runtime logs 2026-04-03..09
- `81cf373` — audit results
- `8844e72` — 4 audit bugs: TZ, int(env), None.get
- `4773a41` — merge origin/master, parallel 3-PC development

Дополнительно учтён HEAD `588dcdb` — controlled live test blocked, root cause analysis.

### Audit documents
Прочитаны и учтены все файлы `audit/*`, включая:

- `audit/INDEX.md`
- `audit/ARCHITECTURE.md`
- `audit/AUDIT_FULL_PROJECT_2026-04-09.md`
- `audit/BOOTSTRAP_FLOW.md`
- `audit/CLAUDE.md`
- `audit/CRM_FULL_AUDIT_2026-04-11.md`
- `audit/INCIDENT_REPORT_WHATSAPP_2026-04-10.md`
- `audit/MESSAGE_ROUTING_AUDIT_2026-04-11.md`
- `audit/SESSION_CONTEXT.md`
- `audit/STATE_MODEL.md`
- `audit/WHATSAPP_APPROVAL_UX_2026-04-11.md`
- `audit/WEEKEND_AND_FINAL_APPROVAL_2026-04-11.md`
- `audit/CONTROLLED_LIVE_TEST_2026-04-11.md`
- `audit/COLLECTOR_CLASSIFICATION_AUDIT_2026-04-11.md`
- старые audit материалы: `AUDIT_FINDINGS.*`, `AUDIT_PROJECT_MAP.md`, `AUDIT_RUNTIME_TRACE.md`, `AUDIT_SUMMARY.md`, `AUDIT_VERDICT.md`
- audit logs: `audit/logs/send_reports_20260403.log`, `send_reports_20260408.log`, `send_reports_20260409.log`

### Code/JSON evidence read directly

- `collector/collections_engine.py`
- `collector/debt_monitor.py`
- `collector/communications.py`
- `collector/approval_flow.py`
- `collector/manager_dialog.py`
- `bot/crm_clients.py`
- `bot/send_reports.py`
- `bot/debt_stop_control.py`
- `config/clients.json`
- `config/debtors_contacts.json`
- `collector/debtors_contacts.json`
- `logs/collector_state.json`
- `logs/collector_dialogs.json`
- `logs/collector_client_dialogs.json`
- `reports/debt_stop_registry.json`
- latest `reports/json/debt_ext_*.json`

## 2. Ключевые коммиты и их смысл

| Hash | Заголовок | Файлы | Контур | Что поменялось | Польза/риск | Production safety |
|---|---|---|---|---|---|---|
| `485e57a` | `fix: weekend silent mode + enhanced final admin approval` | `bot/send_reports.py`, `collector/approval_flow.py`, `tests/test_collector.py`, `audit/WEEKEND...md` | routing / approval / audit | Добавлены holiday guards в scheduled send paths; admin approval получил детальный список клиентов и телефонов; тест FIX-2 обновлён под OP-4. | Польза: выходные стали silent для многих TG jobs; директор видит подробности перед WA. Риск: guards в большом `send_reports.py`, нужна регрессия scheduler. | Улучшает safety; live WA всё ещё нельзя без флагов и approval. |
| `fa3baea` | `fix: BUG-CRM-1 + OP-4 safety guards` | `bot/send_reports.py`, `collector/collections_engine.py` | CRM / collector / incident fix | `crm_phone_reminder_task` получил `is_holiday_today`; direct send запрещён если `manager_name` пустой. | Польза: закрывает weekend spam CRM и OP-4 WA без трассируемого менеджера. | Улучшает current production safety. |
| `beec0cc` | `docs(audit): add INDEX.md...` | `audit/INDEX.md`, `audit/CRM_FULL_AUDIT...`, удалены root duplicates | audit | Канонизировал audit документы в `audit/`, добавил индекс и 10 CRM answers. | Польза: один источник навигации; риск: `audit/INDEX.md` сейчас изменён в worktree. | Не влияет на runtime. |
| `60a1eef` | `docs(audit): add audit reports and architecture materials` | 10 audit docs | audit / architecture | Добавил базовые материалы архитектуры, boot flow, state model, incident/routing/CRM docs. | Польза: источник истины для Claude. | Не влияет на runtime. |
| `e083f97` | `feat: WhatsApp approval UX...` | `collector/approval_flow.py`, `collector/collections_engine.py`, `bot/send_reports.py`, tests, UX doc | approval / collector / routing | Новый `--preview`, batch state `logs/wa_approval_batches.json`, manager/admin callbacks. | Польза: согласование до WA. Риск: UX сам не отправляет WA, но создаёт новый state и callbacks. | Safety улучшена при условии что `--send` требует guards. |
| `31f3bc8` | `docs: add message routing audit` | `MESSAGE_ROUTING_AUDIT_2026-04-11.md` | routing / audit | Матрица send paths и OP-1..OP-7. | Польза: карта рисков. | Не влияет на runtime. |
| `5d9a3fa` | `fix(collector): LIVE_SEND_ALLOWED safeguard...` | `collector/collections_engine.py`, `tests/test_collector.py` | collector / incident fix | Двойной замок `WHATSAPP_ENABLED=1` + `LIVE_SEND_ALLOWED=1`; CLI guard; regression tests. | Польза: основной барьер от случайного live. | Критически улучшает safety. |
| `18a8ea8` | `fix(collector): emergency fix...` | incident doc, `SESSION_CONTEXT.md`, `collections_engine.py`, `communications.py`, `dialog_store.py`, `manager_dialog.py` | collector / incident fix | Вернул active-client filter, убрал direct-send bypass, cap real_days, hardguard `send_whatsapp()`. | Польза: закрывает причины инцидента 2026-04-10. Риск: state/data всё ещё может давать ложные уровни. | Критически улучшает safety, но не разрешает live. |
| `ed139cf` | `CRM flow: add name defer pause` | `SESSION_CONTEXT.md`, `bot/crm_clients.py`, `bot/send_reports.py` | CRM | Добавлена пауза/отложить на шаге имени. | Польза: снижает давление на менеджера. Риск: больше pending state. | Не влияет на WA live напрямую. |
| `6f63213` | `fix: TZ hardcode + шаблонные баги...` | отчётные генераторы, templates, `CLAUDE.md` | audit / reports / incident fix | Убраны TZ hardcodes, добавлены template guards. | Польза: стабильность отчётов. | Улучшает report safety. |
| `dbe8c43` | `docs(audit): add runtime logs...` | `audit/logs/*` | audit | Сохранил runtime логи для расследования. | Польза: evidence. | Не влияет на runtime. |
| `81cf373` | `audit: add audit results` | `AUDIT_FINDINGS.*`, map/trace/summary/verdict | audit | Первичный полный аудит. | Польза: baseline дефектов. | Не влияет на runtime. |
| `8844e72` | `fix: 4 бага аудита...` | `collections_db.py`, `voice_calls.py`, `imap_fetcher.py`, `run_pipeline_all_mp.py` | collector / pipeline / incident fix | Защита int(env), TZ env, `state.get(key) or {}`. | Польза: меньше crash-case. | Улучшает runtime safety. |
| `4773a41` | `merge: объединение с origin/master...` | `SESSION_CONTEXT.md` merge | audit / repo hygiene | Сведение веток 3 ПК, оставлены локальные версии важных файлов, `clients.json` убран из tracking. | Польза: синхронизация. Риск: merge commit с конфликтной историей, проверять code owner. | Непрямой risk management. |

## 3. Какие инциденты уже расследованы

### WhatsApp incident 2026-04-10
Доказано в `audit/INCIDENT_REPORT_WHATSAPP_2026-04-10.md` и routing audit:

- ручной/CLI запуск `collector/collections_engine.py --send` около 13:30;
- 22 WA-сообщения клиентам;
- причины: удалён active-client filter, direct-send bypass без manager lock, inflated `first_seen`, `WHATSAPP_ENABLED=1`, ручной запуск вне нормального controlled flow;
- логи: `logs/collector_20260410.log` в incident report;
- кодовые фиксы: `18a8ea8`, `5d9a3fa`, `fa3baea`, `485e57a`.

### Controlled live test 2026-04-11
Доказано в `audit/CONTROLLED_LIVE_TEST_2026-04-11.md` и текущей проверке JSON:

- тест не выполнен: суббота + нет валидных WA-кандидатов;
- current dry-run style: 426 raw debt clients → 84 classified → 23 level 1-5 → 0 valid WA candidates;
- категории 23: 8 stopped, 14 active_blocked, 1 no_phone.

### CRM reminder weekend bug
Доказано в `CRM_FULL_AUDIT_2026-04-11.md`, исправлено `fa3baea` и расширено `485e57a`:

- `crm_phone_reminder_task` раньше не проверял `is_holiday_today()`;
- после fix weekend reminders должны silent skip.

## 4. Какие фиксы уже внедрены

- FIX-1: active clients filtered in `collector/collections_engine.py` (`debit > 0 or credit > 0` → skip).
- FIX-2: direct send bypass при занятом менеджере убран.
- FIX-3: `real_days = min(real_days, days_1c + 7)`.
- FIX-4: `collector/communications.py::send_whatsapp()` перечитывает `WHATSAPP_ENABLED` на каждом вызове.
- SAFEGUARD: `collector/collections_engine.py::run()` требует оба флага `WHATSAPP_ENABLED=1` и `LIVE_SEND_ALLOWED=1`.
- CLI guard: `--send` блокируется при `WHATSAPP_ENABLED=0`.
- OP-4: direct send запрещён при пустом `manager_name`.
- Approval UX: `--preview`, manager consent, admin final approval, batch state.
- Weekend silent mode: CRM reminders, collector reminders, report jobs, dstop jobs и часть notifier/send paths закрыты `is_holiday_today()`.

## 5. Что уже безопасно

- Telegram report delivery в основном безопасен по routing: role/scopes, не зависит от WA.
- Collector dry-run безопасен: не отправляет WA.
- Approval preview безопасен относительно WA: отправляет Telegram managers/admin, не WA.
- WhatsApp live заблокирован без двух env-флагов.
- `send_whatsapp()` имеет hardguard `WHATSAPP_ENABLED`.
- Текущая воронка кандидатов даёт 0 valid WA candidates из 23 dry-run level 1-5.

## 6. Что ещё опасно

- `collector/collections_engine.py::run_approval_preview()` в live-preview логике использует `get_debt_days_since_first_seen()` даже там, где dry-run показывает 23: текущая проверка дала 42 level>=1 при пересчёте real_days. Это не значит 42 кандидата к отправке: все также stopped/active/no_phone, но цифра preview/live может отличаться от dry-run style.
- `violation_shipment` в `collector/debt_monitor.py` повышает клиентов с 7-9 днями до level 1. Engine потом фильтрует active clients, но классификация завышена.
- `first_seen` всё ещё способен повышать level через cap `+7`; FIX-3 ограничивает, но не убирает источник.
- `config/clients.json` имеет 220 WA/phone, но только 38 напрямую совпадают с текущими 84 classified debtors и 17 с dry-run 23 level 1-5; остальные не участвуют в сегодняшнем collector shortlist.
- `logs/collector_client_dialogs.json` содержит 21 active/old phone-key entry; перед WA live нужно закрыть/проверить старые client sessions.
- `.env` не трогался Codex; фактические flags нужно проверять вручную перед live, но Codex не должен менять.

## 7. Какие вопросы открыты

1. Нужно ли менять `run_approval_preview()` / dry-run consistency: сейчас dry-run style 23, preview/live recalculation может показать 42 level>=1 из-за `first_seen`.
2. Нужно ли ужесточить `violation_shipment`: не повышать до level 1 клиентов с текущей активностью (`debit/credit`) или days < 10.
3. Нужно ли уменьшить cap `days_1c + 7` до `+3` или полностью переопределить смысл `first_seen`.
4. Как переносить 220 CRM WA contacts в collector candidate reality: проблема не в отсутствии merge, а в том, что большинство контактов не являются текущими валидными debt candidates или не совпадают по ключу имени.
5. Что делать со старыми `collector_client_dialogs.json` перед первым live.

## 8. Какие файлы/модули самые важные для следующих шагов Claude

- `collector/collections_engine.py` — главная воронка collector, guards, preview/live расхождение, send path.
- `collector/debt_monitor.py` — классификация 84/23 и `violation_shipment`.
- `collector/collections_db.py` — `first_seen` / `get_debt_days_since_first_seen`.
- `collector/communications.py` — hardguard `send_whatsapp()`.
- `collector/approval_flow.py` — batch approval state and callbacks.
- `bot/crm_clients.py` — CRM contacts merge, `load_contacts_compat()`.
- `config/clients.json` — CRM source with 220 phones.
- `config/debtors_contacts.json` — legacy/collector contact registry, currently 84 entries / 1 phone.
- `reports/debt_stop_registry.json` — stopped/auto_stopped/exception status.
- `logs/collector_client_dialogs.json` — old active WA dialogs.

## 9. Что Claude НЕ должен перечитывать с нуля

- Причины WhatsApp incident 2026-04-10: уже расследованы в incident/routing docs.
- Общую архитектурную карту старого проекта: `AUDIT_FULL_PROJECT_2026-04-09.md` + `ARCHITECTURE.md` достаточны.
- Базовую CRM audit matrix: `CRM_FULL_AUDIT_2026-04-11.md` уже отвечает на 10 вопросов.
- Routing matrix: `MESSAGE_ROUTING_AUDIT_2026-04-11.md` уже фиксирует OP-1..OP-7.
- Approval UX: `WHATSAPP_APPROVAL_UX_2026-04-11.md` + `WEEKEND_AND_FINAL_APPROVAL_2026-04-11.md` описывают flow.
- 23 dry-run clients list: перенесено в `audit/COLLECTOR_SHORTLIST_EXPLAINED_2026-04-11.md` и `audit/manager_lists/*`.

## 10. Что Codex уже выяснил и можно использовать сразу

### Current shortlist facts

- Raw latest debt clients: 426.
- Classified by `classify_debtors()`: 84.
- Dry-run style level 1-5: 23.
- Categories among 23: `stopped=8`, `active_blocked=14`, `no_phone=1`, `valid_candidate=0`.
- By manager:
  - Алена: 5 total = 3 stopped + 2 active_blocked.
  - Ергали: 9 total = 4 stopped + 4 active_blocked + 1 no_phone.
  - Магира: 7 total = 1 stopped + 6 active_blocked.
  - Оксана: 2 total = 2 active_blocked.

### Contact facts

- `config/clients.json`: 551 CRM clients, 220 with `whatsapp`/phone.
- `config/debtors_contacts.json`: 84 entries, only 1 with phone/WA.
- `collector/debtors_contacts.json`: legacy/alternate store, 3 entries.
- `load_contacts_compat()` does merge CRM into collector-compatible dict, but matching is by client name.
- Current direct CRM WA matches:
  - 38 of current 84 classified debtors.
  - 17 of dry-run 23 level 1-5.
  - 0 valid candidates after stop/active/no_phone filters.

### Exact code locations

- CRM merge: `bot/crm_clients.py:442-468` (`load_contacts_compat`).
- CRM writes WA: `bot/crm_clients.py:356-382` (`set_client_phone`), `bot/crm_clients.py:387-416` (`set_client_details`).
- Collector loads merged contacts: `collector/collections_engine.py:503-509`.
- Collector contact lookup: `collector/collections_engine.py:648-670`.
- Collector preview contact lookup and manager requirement: `collector/collections_engine.py:789-815`.
- `match_client()` name matching: `collector/debt_monitor.py:296-331`.
- Direct send manager guard OP-4: `collector/collections_engine.py:389-401`.
- Live env double lock: `collector/collections_engine.py:471-492`.
- WhatsApp hardguard: `collector/communications.py:72-77`.

## Source-of-truth map for audit docs

| Topic | Source of truth | Notes |
|---|---|---|
| What exists / navigation | `audit/INDEX.md` | Current file is modified in worktree; do not overwrite. |
| Architecture target | `audit/ARCHITECTURE.md`, `BOOTSTRAP_FLOW.md`, `STATE_MODEL.md` | For new product direction, not direct patch list. |
| Incident 2026-04-10 | `audit/INCIDENT_REPORT_WHATSAPP_2026-04-10.md` | Primary incident evidence. |
| Routing risks | `audit/MESSAGE_ROUTING_AUDIT_2026-04-11.md` | Primary send-path matrix. |
| CRM audit | `audit/CRM_FULL_AUDIT_2026-04-11.md` | Note BUG-CRM-1 later fixed by `fa3baea`. |
| Approval UX | `audit/WHATSAPP_APPROVAL_UX_2026-04-11.md`, `WEEKEND_AND_FINAL_APPROVAL_2026-04-11.md` | UX + implemented guards. |
| Shortlist/current candidates | `audit/COLLECTOR_CLASSIFICATION_AUDIT_2026-04-11.md`, `audit/CONTROLLED_LIVE_TEST_2026-04-11.md`, this document, `COLLECTOR_SHORTLIST_EXPLAINED_2026-04-11.md` | Current Codex doc adds direct JSON counts and manager lists. |

## Documents that duplicate each other

- Root `INCIDENT_REPORT_WHATSAPP_2026-04-10.md`, `MESSAGE_ROUTING_AUDIT_2026-04-11.md`, `WHATSAPP_APPROVAL_UX_2026-04-11.md` were deleted by `beec0cc`; canonical copies are under `audit/`.
- `CONTROLLED_LIVE_TEST_2026-04-11.md` and `COLLECTOR_CLASSIFICATION_AUDIT_2026-04-11.md` overlap on 84/23/0-candidates; use classification doc for details and controlled-live doc for test status.
- `CRM_FULL_AUDIT_2026-04-11.md` and `INDEX.md` overlap on BUG-CRM-1; `INDEX.md`/commit history has newer status: fixed by `fa3baea`.

# CRM FULL AUDIT — 2026-04-11

## 1) Что проверено
Проверены именно узлы CRM, которые уже были в scope аудита (без запуска live и без новых фич):

- `crm_daily_task` (ежедневный запуск CRM-обновления + постановка задач на заполнение контактов).
- `crm_phone_reminder_task` (почасовые напоминания по незавершённым CRM-диалогам).
- `_CRM_PHONE_PENDING` (структура pending-состояния, TTL/очистка, восстановление после рестарта).
- `crm_claim` callback (`crm_claim|...`) для бесхозных клиентов.
- `update_from_reports` (синхронизация CRM из JSON debt/sales).
- `load_clients`, `load_contacts_compat`, `get_clients_without_phones`, `CRM_DAILY_LIMIT`.
- CRM schedule через `job_queue.run_daily(...)` / `job_queue.run_repeating(...)`.
- Обработчик шагов ввода менеджером (clarify_name → clarify_phone → clarify_address).

Ограничение аудита: только анализ текущего кода и маршрутов данных; live-send/WhatsApp live не запускались.

---

## 2) Источники CRM
Фактический pipeline источников:

1. **Первичный импорт в CRM**
   - `reports/json/debt_ext_*.json`
   - `reports/json/sales_*.json`
   - Источники читаются через `bot/crm_clients.py` (`_load_latest_debt_clients`, `_load_latest_sales_clients`) и сводятся в `config/clients.json` через `update_from_reports`.

2. **Операционное хранилище CRM**
   - `config/clients.json` — основной источник истины для CRM-клиентов.

3. **Состояние незавершённых CRM-диалогов**
   - in-memory: `_CRM_PHONE_PENDING`
   - persistence: `logs/crm_pending_state.json` через `_crm_save_pending`/`_crm_load_pending`.

4. **Совместимость с legacy коллектором**
   - `load_contacts_compat()` merge:
     - base: legacy `debtors_contacts.json` (через `collector.debt_monitor.load_contacts`)
     - override: CRM (`clients.json`).

---

## 3) Схема данных клиента
Базовая запись клиента в `clients.json` (создаётся в `update_from_reports`):

- `manager`
- `whatsapp`
- `telegram_id`
- `language`
- `do_not_call`
- `sources` (`debt`/`sales`)
- `first_seen`
- `last_seen`

Дополняемые поля в процессе CRM-диалога:

- `display_name`
- `original_name`
- `name_mode` (`system` / `manual` / `later`)
- `name_review_needed`
- `address`

Наблюдение: схема фактически **schema-less** (нет жёсткой валидации структуры всего объекта клиента перед записью), есть только точечные проверки/присваивания.

---

## 4) Создание задач
Создание CRM-задач происходит в `crm_daily_task` (18:00):

1. `update_from_reports()` обновляет базу клиентов.
2. Для каждого участника CRM (`_all_crm_participants`: менеджеры + admin) выбирается первый клиент без телефона.
3. В `_CRM_PHONE_PENDING[chat_id]` создаётся state:
   - `state=clarify_name`
   - `client_key`, `original_name`
   - `done_today=0`
   - `daily_limit=CRM_DAILY_LIMIT` (15)
   - `manager`, `total_no_phone`, `last_sent`
4. Отправляется сообщение с кнопками выбора имени.
5. Дополнительно формируется broadcast по бесхозным клиентам (`crm_claim`).

Также после `crm_claim` создаётся отдельный pending-кейс на 1 клиента (daily_limit=1) для того, кто взял клиента.

---

## 5) Reminder логика
`crm_phone_reminder_task`:

- Интервал джобы: каждый час.
- Окно работы: **09:00–18:59** (`9 <= now.hour < 19`).
- Перед отправкой делается `_crm_cleanup_pending()`.
- Если у pending есть `paused_until` и дедлайн не наступил — напоминание пропускается.
- Текст напоминания зависит от `state`:
  - `clarify_name`: повторяет prompt + inline-кнопки.
  - `clarify_phone`/`clarify_address`: текстовый prompt по ожидаемому шагу.

Критично: reminder-джоба **не проверяет выходной день**, в отличие от `crm_daily_task`.

---

## 6) Маршрутизация и scope
Маршруты и доступ:

- Постановка задач идёт по `_all_crm_participants()`:
  - `MANAGERS_MAP`
  - + `ADMIN_CHAT_ID` как `Вадим`.
- На ручной ввод (`handle_persistent_menu`) pending привязан к `chat_id`, т.е. state изолирован по пользователю.
- Claim-блок тоже разрешает admin участвовать в назначении клиента.
- `/phone` разрешён менеджеру по `_chat_to_manager(chat_id)` либо admin.

Итог: CRM scope сейчас — **менеджеры + admin**; отдельного выделенного "manager-only CRM" режима нет.

---

## 7) Связь с debt / sales / collector
Связи подтверждены:

- **debt/sales → CRM**: `update_from_reports()` читает последние debt/sales JSON и обновляет clients DB.
- **CRM → collector compatibility**: `load_contacts_compat()` формирует merged contacts для match-кейсов коллектора.
- **collector legacy cleanup**: `_cleanup_legacy_collector_pending_state()` удаляет устаревшие `__phone_pending__` / `__name_pending__` в `collector_state.json`.
- **Debt collector schedule** и CRM schedule сосуществуют в одном job_queue, но отдельными задачами.

---

## 8) Legacy и мусор
Выявлены legacy/хвосты:

1. В `handle_persistent_menu` оставлен большой закомментированный блок старого flow (`[DISABLED v9.4.39]`).
2. Есть dual-state реальность: новый `_CRM_PHONE_PENDING` + cleanup legacy pending в collector state.
3. Наличие `clarify_address` ветки при фактическом потоке, где после `clarify_phone` уже выполняется сохранение и переход к следующему клиенту (адрес чаще "позже"). Ветка остаётся как полудохлый сценарий.

Это не авария само по себе, но повышает риск расхождения поведения при будущих изменениях.

---

## 9) Матрица CRM-сообщений
| Триггер | Кому | Канал/тип | Содержание |
|---|---|---|---|
| `crm_daily_task` новые клиенты | менеджер | TG text | список новых клиентов (до 10 + хвост) |
| `crm_daily_task` phone onboarding | менеджер/admin | TG text + inline kb | prompt имени (`crm_name|edit/keep/later`) |
| `crm_claim` broadcast | все CRM участники | TG text + inline kb | "Чей клиент?" + "✋ Мой клиент" |
| `crm_claim` resolved | остальные участники | TG text | "клиента уже взяли" |
| `crm_phone_reminder_task` | pending-участники | TG text (иногда inline kb) | напоминание по текущему шагу |
| `handle_persistent_menu` success | менеджер/admin | TG text | подтверждение сохранения телефона/данных |

---

## 10) Подтверждённые безопасные места
1. Атомарная запись `clients.json` через temp + `os.replace`.
2. Atomic-like запись pending state через tmp + replace.
3. TTL-очистка зависших pending-кейсов (`CRM_PENDING_TTL_HOURS`, default 48).
4. Валидация телефона (приведение 8XXXXXXXXXX → 7XXXXXXXXXX, строгий `7\d{10}`).
5. Ограничение дневной нагрузки (`CRM_DAILY_LIMIT=15`) и цепочка "по одному".
6. `crm_daily_task` уважает выходной (`is_holiday_today`).

---

## 11) Подтверждённые опасные места
1. **Reminder в выходные не отключён**: `crm_phone_reminder_task` работает по времени, но без `is_holiday_today`.
2. **Потенциальный over-broadcast claim**: бесхозные клиенты рассылаются всем участникам CRM, включая admin; при росте базы это может шуметь.
3. **Schema drift risk**: нет централизованной валидации структуры `clients.json`; поля добавляются ad-hoc.
4. **Непрозрачная приоритизация источников**: merge legacy+CRM в `load_contacts_compat` может скрывать устаревшие данные legacy при неполном CRM заполнении.
5. **Логическая неоднородность шага address**: ветка `clarify_address` существует, но основной путь фактически завершает кейс на шаге телефона (адрес "позже"), что усложняет сопровождение и тест-кейсы.

---

## 12) Какие файлы нужно менять
Если цель — безопасно включать/масштабировать CRM-флоу, минимально нужно менять:

1. `bot/send_reports.py`
   - унифицировать state machine CRM,
   - добавить проверку выходного в reminder,
   - подчистить legacy-ветки и шум claim-механизма.

2. `bot/crm_clients.py`
   - ввести мягкую валидацию/нормализацию схемы клиента,
   - формализовать source precedence для compatibility-merge.

3. (опционально, если вводить контроль целостности)
   - новый документированный schema-contract в `docs/` или `config/` (в текущем аудите файла ещё нет).

---

## 13) Можно ли безопасно включать
### CRM task creation
**Да, ограниченно безопасно** для controlled режима: постановка задач и запись pending в целом стабильны.

### CRM reminders
**Нет (пока рано)**: reminder-джоба не уважает выходные и может слать напоминания вне желаемой бизнес-логики календаря.

### manager CRM flow
**Да, условно**: текущий flow рабочий, но есть legacy/ветвления, требующие зачистки перед "широким" запуском.

### admin CRM summaries
**Нет как отдельный готовый CRM-модуль**: в коде есть общие admin summary (бот/отчёты), но отдельная зрелая CRM summary-подсистема в этом срезе не подтверждена.

---

## Итоговый verdict
- **Controlled live CRM task creation**: допустимо.
- **CRM reminders production-safe**: ещё нет.
- **WhatsApp live**: в рамках этого аудита не трогалось и не включалось.

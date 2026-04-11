# STATE MODEL

Дата: 2026-04-09
Цель: зафиксировать единую state-модель для нового продукта `GPT1C_ProAnalytic` до начала кодирования.

## 1. Принцип

В новом продукте не допускается “самостоятельно живущий JSON”.

Любое состояние существует только в трёх слоях:

1. `model`
2. `repository`
3. `flow/service`, который меняет состояние по правилам

Нельзя:

- читать и писать state напрямую из handler'ов;
- хранить бизнес-смысл в случайных флагах;
- дублировать один и тот же жизненный цикл в нескольких файлах;
- делать “cleanup удалением без следа”, если это бизнес-кейс.

## 2. Категории состояния

Вся state-модель делится на 4 группы.

### 2.1. Durable business state

Это долгоживущие сущности, которые описывают бизнес-объекты.

Примеры:

- `ClientRecord`
- `CollectorCase`
- `DebtStopRecord`
- `ManagerProfile`
- `DeliveryRecord`

Свойства:

- переживают перезапуск;
- имеют owner-модуль;
- имеют стабильный идентификатор;
- не должны зависеть от Telegram message id или временной кнопки.

### 2.2. Ephemeral workflow state

Это временное состояние активного сценария.

Примеры:

- `CrmTask`
- `ManagerSession`
- `ClientSession`
- `PendingDecision`

Свойства:

- всегда имеют `created_at`, `updated_at`, `expires_at`;
- имеют явную причину возникновения;
- имеют конечный исход;
- не живут бесконечно.

### 2.3. Queue / scheduler state

Это техническое состояние очередей и задач.

Примеры:

- `JobRunState`
- `QueueItem`
- `RetryState`
- `LockState`

Свойства:

- не содержат бизнес-решение;
- не подменяют собой durable state;
- могут быть очищены по правилам retention;
- должны быть наблюдаемы.

### 2.4. Audit / event state

Это журнал событий и автоматических решений.

Примеры:

- эскалация менеджера;
- auto-expire;
- cleanup stale state;
- skip duplicate;
- send failure;
- send success.

Свойства:

- append-only или близко к этому;
- не являются источником истины по текущему статусу;
- нужны для расследования и отчётности.

## 3. Сущности нового продукта

## 3.1. ClientRecord

Единица клиента в системе.

Обязательные поля:

- `client_key`
- `original_name`
- `display_name`
- `manager_name`
- `whatsapp`
- `phone`
- `telegram_id`
- `address`
- `language`
- `name_review_needed`
- `contact_review_needed`
- `sources`
- `first_seen_at`
- `last_seen_at`
- `updated_at`

Правила:

- `client_key` — главный идентификатор клиента внутри продукта;
- `original_name` — имя из 1С или источника;
- `display_name` — удобное имя для общения;
- `display_name` может быть пустым только временно;
- отображение для интерфейсов: `display_name or original_name`.

## 3.2. CollectorCase

Главная сущность взыскания.

Обязательные поля:

- `case_id`
- `client_key`
- `manager_name`
- `source_report`
- `days_overdue`
- `debt_amount`
- `risk_level`
- `status`
- `status_reason`
- `created_at`
- `updated_at`
- `expires_at`
- `escalated_at`
- `last_action_at`
- `last_send_attempt_at`
- `last_send_result`
- `delivery_id`

Допустимые статусы:

- `new`
- `needs_contact`
- `awaiting_manager`
- `ready_to_send`
- `sending`
- `sent`
- `delivery_failed`
- `awaiting_client_reply`
- `escalated`
- `resolved`
- `closed`
- `expired`

Правила:

- один `CollectorCase` не хранится по `manager_chat_id`;
- ключ кейса не зависит от Telegram/WhatsApp-транспорта;
- `awaiting_manager` всегда имеет TTL;
- `sent` означает успешный send-path;
- `delivery_failed` не равен `closed`;
- `escalated` означает переход решения к admin/supervisor.

## 3.3. ManagerSession

Временный контекст manager-interaction по конкретному collector-case.

Поля:

- `session_id`
- `case_id`
- `manager_chat_id`
- `state`
- `message_id`
- `created_at`
- `updated_at`
- `expires_at`
- `escalated_at`
- `attempt_count`

Состояния:

- `awaiting_name_confirm`
- `awaiting_phone_confirm`
- `awaiting_send_confirm`
- `awaiting_data_input`
- `awaiting_rejection_reason`
- `awaiting_admin_review`
- `expired`
- `closed`

Правила:

- один менеджер может иметь несколько `ManagerSession`, но не более одной активной сессии на один `case_id`;
- “менеджер занят” не должен блокировать других клиентов;
- если сессия устарела, кейс переводится дальше по правилам, а не висит вечно.

## 3.4. ClientSession

Временный контекст общения с клиентом в WhatsApp.

Поля:

- `session_id`
- `case_id`
- `phone_key`
- `status`
- `created_at`
- `updated_at`
- `expires_at`
- `exchange_count`
- `last_incoming_at`
- `last_outgoing_at`
- `escalated_to_manager_at`

Состояния:

- `active`
- `awaiting_reply`
- `escalated`
- `closed`
- `expired`

Правила:

- хранится по `phone_key`, но обязательно связана с `case_id`;
- не должна существовать без `CollectorCase`;
- по истечении TTL не удаляется молча, а закрывается с событием.

## 3.5. CrmTask

Задача на дозаполнение CRM.

Поля:

- `task_id`
- `client_key`
- `manager_name`
- `state`
- `original_name`
- `display_name`
- `phone`
- `address`
- `name_mode`
- `name_review_needed`
- `paused_until`
- `created_at`
- `updated_at`
- `expires_at`
- `completed_at`

Состояния:

- `new`
- `awaiting_name_review`
- `awaiting_phone`
- `awaiting_address`
- `paused`
- `completed`
- `expired`

Правила:

- телефон не блокируется именем;
- address не блокирует сохранение телефона;
- `paused_until` обязателен для “Позже”;
- expired task не удаляется без записи события.

## 3.6. DebtStopRecord

Постоянная запись по stop-list.

Поля:

- `record_id`
- `client_key`
- `manager_name`
- `status`
- `status_reason`
- `approved_at`
- `auto_stopped_at`
- `cleared_at`
- `discipline_violation`
- `days_at_decision`
- `debt_at_decision`
- `updated_at`

Статусы:

- `candidate`
- `manager_requested`
- `manager_approved_exception`
- `manager_rejected`
- `admin_review`
- `stopped`
- `conditional`
- `auto_stopped`
- `pending_clearance`
- `cleared`

## 4. Общие правила для всех временных состояний

Любой pending-state обязан иметь:

- `created_at`
- `updated_at`
- `expires_at`
- `state`
- `state_reason`
- `owner_module`

Нельзя создавать pending без TTL.

Нельзя silently delete живой кейс.

Разрешённые исходы cleanup:

- `expired`
- `escalated`
- `closed`
- `archived`

И дополнительно:

- запись в event log;
- при необходимости уведомление admin.

## 5. Правила дедупликации

## 5.1. Клиент

Дедупликация не должна жить только на `client_name`.

Приоритет ключей:

1. `client_key`
2. нормализованная связка `original_name + manager_name`
3. fallback по нормализованному `display_name`

## 5.2. Collector case

В один момент времени допустим:

- один активный `CollectorCase` на клиента по одному и тому же бизнес-поводу;
- несколько исторических закрытых кейсов.

Нельзя:

- два активных send-case по одному клиенту одновременно;
- новый кейс поверх активного без явного решения merge/skip/escalate.

## 5.3. Менеджерская сессия

Нельзя:

- несколько активных `ManagerSession` на один `case_id`;
- блокировать всех клиентов менеджера из-за одной сессии.

## 6. Ownership

Жёсткие границы:

- `bot` не читает и не пишет state напрямую;
- `pipeline` не знает про Telegram;
- `collector` не знает про IMAP;
- `crm` не знает про WhatsApp transport;
- `repository` ничего не знает про UI;
- `jobs` не содержат бизнес-логики, только вызывают service.

## 7. Репозитории

Каждая сущность имеет свой repository.

Минимальный набор:

- `ClientRepository`
- `CollectorCaseRepository`
- `ManagerSessionRepository`
- `ClientSessionRepository`
- `CrmTaskRepository`
- `DebtStopRepository`
- `JobStateRepository`
- `EventLogRepository`

Repository обязан:

- валидировать структуру;
- делать атомарную запись;
- не смешивать схемы разных сущностей;
- поддерживать безопасное чтение;
- обеспечивать migration hook.

## 8. Cleanup policy

Cleanup не имеет права:

- удалять бизнес-кейс без trace;
- менять business-status без правила flow;
- подменять собой normal resolution.

Cleanup имеет право:

- переводить временный state в `expired`;
- записывать событие;
- ставить `control_deadline`;
- инициировать эскалацию;
- архивировать технический мусор.

## 9. Минимальные инварианты нового продукта

1. Ни один case не живёт без owner и статуса.
2. Ни один pending не живёт без TTL.
3. Ни одна отправка не считается успешной до факта успешного send-path.
4. Один менеджер не блокирует весь поток.
5. Один клиент не получает двойную активную отправку в одном бизнес-цикле.
6. Cleanup не удаляет молча живой кейс.
7. Любой автоматический переход оставляет audit trace.

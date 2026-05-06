# ARCHITECTURE

Дата: 2026-04-09
Последнее обновление: 2026-04-29
Проект: `GPT1C_ProAnalytic`

## 1. Цель архитектуры

Новый проект создаётся не как копия старого production-бота, а как чистая продуктовая сборка.

Задача архитектуры:

- сохранить проверенную бизнес-логику;
- убрать монолитность;
- исключить JSON-хаос;
- развести слои ответственности;
- сделать систему безопасной для развития и эксплуатации.

## 2. Главные принципы

1. Один модуль — одна зона ответственности.
2. Один bootstrap — один официальный способ запуска продукта.
3. UI не знает про формат хранения state.
4. Transport не знает бизнес-правила.
5. Scheduler не знает деталей use-case, а только вызывает задачи.
6. State живёт только через model + repository.
7. Старый production — источник логики, но не источник структуры.

## 3. Архитектурные слои

## 3.1. App layer

Назначение:

- сборка приложения;
- wiring модулей;
- загрузка конфигурации;
- логирование;
- регистрация jobs;
- запуск lifecycle.

Состав:

- `app/bootstrap/`
- `app/config/`
- `app/logging/`
- `app/security/`

Этот слой не должен содержать бизнес-правил collector/CRM/reports.

## 3.2. Interface layer

Назначение:

- Telegram handlers;
- callback routing;
- menu rendering;
- command endpoints;
- возможно HTTP/admin endpoints в будущем.

Состав:

- `bot/handlers/`
- `bot/callbacks/`
- `bot/menus/`
- `bot/middleware/`

Этот слой не должен:

- читать JSON напрямую;
- принимать бизнес-решение;
- знать детали IMAP/pipeline.

## 3.3. Application services

Назначение:

- реализация use-case;
- orchestration над domain и repository;
- переходы состояний;
- эскалации;
- дедупликация;
- отправка в transports через адаптеры.

Состав:

- `crm/service/`
- `crm/flows/`
- `collector/service/`
- `collector/manager_flow/`
- `collector/client_flow/`
- `collector/escalation/`
- `debt_stop/service/`
- `pipeline/processors/`

Это главный рабочий слой продукта.

## 3.4. Domain layer

Назначение:

- чистые модели;
- бизнес-правила;
- инварианты;
- value objects.

Состав:

- `domain/clients/`
- `domain/managers/`
- `domain/debt/`
- `domain/reports/`
- `domain/common/`

Domain не должен знать:

- Telegram;
- WhatsApp gateway;
- IMAP;
- файловую структуру проекта.

## 3.5. Repository layer

Назначение:

- хранение и получение данных;
- маппинг моделей;
- атомарная запись;
- миграции форматов;
- безопасная загрузка и сохранение.

Состав:

- `state/repository/`
- `state/models/`
- `state/migrations/`
- `state/cleanup/`

Правило:

- repository — единственная точка работы с persistent state.

## 3.6. Infrastructure layer

Назначение:

- IMAP;
- filesystem;
- Telegram adapter;
- WhatsApp/Green API adapter;
- AI providers;
- HTML rendering;
- Excel parsing wrappers.

Состав:

- `pipeline/imap/`
- `pipeline/queue/`
- `collector/whatsapp/`
- `reports/templates/`
- `app/security/`
- `app/logging/`

## 3.7. Jobs layer

Назначение:

- декларативная регистрация расписания;
- вызов use-case;
- timeout/retry/health tracking.

Состав:

- `jobs/scheduler/`
- `jobs/tasks/`

Jobs не должны:

- хранить бизнес-логику;
- заниматься routing;
- знать детали UI.

## 4. Предлагаемая структура каталогов

```text
GPT1C_ProAnalytic/
  app/
    bootstrap/
    config/
    logging/
    security/
  bot/
    handlers/
    callbacks/
    menus/
    middleware/
  pipeline/
    imap/
    queue/
    classifiers/
    processors/
    indexer/
  reports/
    generators/
    parsers/
    schemas/
    templates/
  crm/
    service/
    flows/
    reminders/
    repository/
  collector/
    service/
    manager_flow/
    client_flow/
    whatsapp/
    escalation/
  debt_stop/
    service/
    decisions/
    registry/
  state/
    models/
    repository/
    migrations/
    cleanup/
  jobs/
    scheduler/
    tasks/
  domain/
    clients/
    managers/
    debt/
    reports/
    common/
  tests/
    unit/
    integration/
    scenario/
  docs/
  runtime/
    logs/
    exports/
```

## 5. Границы модулей

Это обязательные запреты.

### 5.1. Bot

Bot:

- принимает update;
- вызывает service;
- рендерит ответ.

Bot не имеет права:

- читать `clients.json`;
- читать `collector_cases.json`;
- самостоятельно очищать stale state;
- принимать решение об эскалации.

### 5.2. Pipeline

Pipeline:

- получает файлы;
- классифицирует их;
- вызывает генераторы;
- сохраняет продукты обработки.

Pipeline не имеет права:

- отправлять Telegram;
- создавать CRM pending;
- управлять collector-case.

### 5.3. CRM

CRM:

- управляет карточками клиентов и CRM tasks.

CRM не имеет права:

- читать IMAP;
- управлять WhatsApp;
- принимать collector-решения.

### 5.4. Collector

Collector:

- управляет cases, manager sessions, client sessions, escalation.

Collector не имеет права:

- читать почту;
- управлять отчётным pipeline;
- напрямую читать UI callbacks без bot layer.

### 5.5. Jobs

Jobs:

- вызывают service по расписанию;
- пишут run-state;
- логируют результат.

Jobs не имеют права:

- содержать бизнес-ветвления уровня use-case;
- писать state в обход repositories.

## 6. Основные продуктовые контуры

## 6.1. Report Delivery

Состав:

- IMAP ingest
- queue processing
- report generation
- index build
- Telegram delivery
- archive

Назначение:

- управленческие отчёты и аналитика.

## 6.2. CRM

Состав:

- client registry
- CRM task queue
- manager fill flow
- reminders
- review-later logic

Назначение:

- пополнение и нормализация клиентской базы.

## 6.3. Collector

Состав:

- debtor selection
- contact readiness
- manager decision flow
- send pipeline
- client reply flow
- escalation flow

Назначение:

- взыскание как конвейер, а не как fragile-диалог.

Операционные уточнения по текущему production-поведению:

- collector runtime нельзя воспринимать как один поток: это связка `collector/*`, `approval_flow`, `no_movement`, `payment_hold`;
- trigger window для запуска preview/check сейчас расширен до `09:00–22:00`, потому что бухгалтерская разноска оплат может происходить после `20:00`;
- TTL флага свежей дебиторки расширен с `6ч` до `14ч`, чтобы вечерняя разноска не превращалась в ложный stale-gap;
- после каждого live `send_whatsapp()` администратор получает мгновенное notice;
- daily collector summary обязан показывать отдельный блок с перечнем фактических WhatsApp-получателей;
- CRM/clarify-phone очередь должна фильтровать служебные/зарплатные записи как на входе, так и при cleanup pending state;
- отдельный stop/clearance контур (`bot/debt_stop_control.py`) не является частью collector send-pipeline и не должен смешиваться с WhatsApp debt flow.

## 6.4. Debt Stop

Состав:

- candidate builder
- manager decision
- admin decision
- accountant final list
- discipline registry

Назначение:

- контроль стоп-листа отгрузки.

## 7. Хранилища нового продукта

На старте допускается файловое хранение, но только под repository layer.

Разделение:

- `runtime/state/clients/`
- `runtime/state/crm/`
- `runtime/state/collector/`
- `runtime/state/debt_stop/`
- `runtime/state/jobs/`
- `runtime/events/`
- `runtime/logs/`

Правило:

- один файл = одна сущность или один bounded context;
- никаких “общих мешков состояния”.

## 8. Интеграции

## 8.1. Telegram

Через отдельный adapter.

Должно быть отделено:

- send text
- send document
- edit message
- delete message
- answer callback

Ни один business module не должен формировать raw Telegram payload сам.

## 8.2. WhatsApp / Green API

Через отдельный adapter.

Должно быть отделено:

- send message
- receive notification
- delete notification
- media handling
- delivery result normalization

Операционное правило правдивости:

- live-send не должен проходить без отдельного observable-следа;
- минимумом считаются:
  - событие факта отправки;
  - мгновенное admin-notice;
  - попадание клиента в daily WA summary;
  - возможность доказать, какой batch и какая версия debt snapshot породили конкретное сообщение.

## 8.3. IMAP

Через отдельный ingest adapter.

## 8.4. AI provider

Через provider interface.

Это позволит:

- менять модель;
- управлять timeout/retry;
- не смешивать AI с collector flow напрямую.

## 9. Observability

Новый продукт обязан иметь:

- структурные logs;
- job run logs;
- event log по state transitions;
- daily health summary;
- warnings по stale/pending/escalation/send failures.

Дополнительно для collector важны:

- явная видимость всех live WhatsApp-касаний для админа;
- отдельные warnings по stale approved batches;
- наблюдаемость no-movement/payment-hold развилок;
- контроль загрязнения CRM pending state служебными/зарплатными клиентами;
- возможность отличить collector-event от stop-control event без чтения кода.

Минимальные health domains:

- IMAP
- pipeline
- report generation
- CRM
- collector
- WhatsApp poller
- debt_stop
- scheduler

## 10. Migration philosophy

Старый production — reference system.

Новый продукт строится так:

1. Анализируем существующую логику.
2. Переносим правила, а не файлы.
3. Реализуем модуль в новой архитектуре.
4. Гоним сценарии.
5. Сверяем outputs.

Запрещено:

- переносить крупные старые файлы “как есть”;
- тащить legacy state contract в новый продукт без явного решения.

## 11. Этапы архитектурной реализации

### Фаза 1. Foundation

- config
- logging
- bootstrap
- repositories
- state models
- scheduler map

### Фаза 2. Reports

- pipeline
- report generation
- archive
- delivery

### Фаза 3. CRM

- clients
- crm tasks
- manager flow
- reminders

### Фаза 4. Collector

- cases
- manager sessions
- WhatsApp send flow
- client reply flow
- escalation

### Фаза 5. Debt Stop

- full stop-list subsystem

## 12. Архитектурный критерий успеха

Архитектура считается правильной, если:

1. ни один модуль не является новым `send_reports.py`;
2. любой state-файл имеет owner и схему;
3. любой бизнес-кейс можно объяснить одной сущностью и её статусом;
4. один молчащий менеджер не блокирует систему;
5. новая функция добавляется локально, а не через перепрошивку всего продукта.

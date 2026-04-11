# BOOTSTRAP FLOW

Дата: 2026-04-09
Проект: `GPT1C_ProAnalytic`

## 1. Цель

В новом продукте должен быть ровно один официальный bootstrap.

Не должно быть ситуации старого проекта, где живут параллельно:

- основной бот;
- отдельный pipeline orchestrator;
- отдельные ручные циклы;
- разнородные runtime entrypoint'ы без единого центра.

## 2. Единственный официальный запуск

Целевой запуск:

```bash
python -m app.bootstrap.main
```

Именно этот entrypoint:

- загружает конфиг;
- валидирует окружение;
- инициализирует repositories;
- строит adapters;
- регистрирует jobs;
- поднимает Telegram bot;
- запускает lifecycle.

Все остальные скрипты:

- diagnostics only;
- migration only;
- maintenance only;
- не production entrypoint.

## 3. Bootstrap stages

## 3.1. Stage 0 — process guard

До всего остального:

- проверка single instance policy;
- проверка корректности runtime directories;
- проверка прав на запись логов и state.

Если проверка не пройдена:

- startup abort;
- понятный fatal log;
- без частичного запуска.

## 3.2. Stage 1 — config load

Загружается:

- environment;
- app config;
- roles/managers config;
- integration config;
- scheduler config;
- feature flags.

На этом этапе секреты только читаются и валидируются.

Если обязательный секрет отсутствует:

- контур помечается disabled, если это допустимо;
- либо приложение не стартует, если контур обязателен.

## 3.3. Stage 2 — logging init

Инициализируется:

- main logger;
- job logger;
- event logger;
- security masking;
- runtime session header.

В лог сразу пишется:

- version;
- environment;
- enabled modules;
- disabled integrations;
- runtime paths.

## 3.4. Stage 3 — state repository init

Поднимаются repositories:

- clients
- CRM tasks
- collector cases
- manager sessions
- client sessions
- debt stop records
- job state
- event log

На этом этапе:

- создаются отсутствующие runtime directories;
- валидируются форматы;
- выполняются безопасные migrations;
- выполняется startup cleanup только технических хвостов.

Запрещено:

- silently delete business state;
- чинить битый state без event log.

## 3.5. Stage 4 — adapter init

Поднимаются adapters:

- Telegram
- WhatsApp
- IMAP
- AI provider
- report renderer

Каждый adapter обязан вернуть:

- `enabled`
- `health`
- `reason_if_disabled`

## 3.6. Stage 5 — service init

Поднимаются services:

- report delivery service
- CRM service
- collector service
- debt stop service
- archive/index service
- health service

Service получает зависимости только через bootstrap wiring.

## 3.7. Stage 6 — bot init

Создаётся Telegram application:

- handlers;
- callbacks;
- middleware;
- error handlers;
- menu routing.

Bot init не должен:

- регистрировать jobs напрямую бизнес-кодом;
- сам читать состояние из файлов.

## 3.8. Stage 7 — scheduler init

Регистрируются все jobs через единый registry.

Каждая job должна иметь:

- name
- purpose
- schedule
- timeout
- retry policy
- alert policy
- service binding

## 3.9. Stage 8 — startup checks

До перехода в run-loop:

- health snapshot;
- проверка конфликтов disabled/enabled modules;
- проверка критичных runtime-paths;
- проверка очередей и stale critical state;
- стартовое уведомление админу при необходимости.

## 3.10. Stage 9 — run

После этого продукт входит в основной runtime:

- Telegram polling/webhook;
- scheduler active;
- periodic health and event logging;
- graceful shutdown hooks registered.

## 4. Startup policy for modules

## 4.1. Always-on modules

Обязательные для старта:

- config
- logging
- repositories
- health service
- Telegram core
- scheduler core

Если они не готовы:

- старт запрещён.

## 4.2. Optional modules

Могут быть временно disabled:

- WhatsApp adapter
- AI provider
- IMAP ingest
- analytics generation

Но только если:

- это не ломает базовый boot;
- disabled status явно отражён в логах и health summary.

## 5. Единственная карта jobs

В новом продукте jobs описываются централизованно.

Пример категорий:

- `pipeline.fetch_mail`
- `pipeline.process_queue`
- `reports.notify_new`
- `crm.daily_seed`
- `crm.reminders`
- `collector.daily_seed`
- `collector.reminders`
- `collector.promise_check`
- `collector.whatsapp_poll`
- `debt_stop.monitor`
- `debt_stop.requests`
- `debt_stop.escalate`
- `debt_stop.final_notify`
- `system.cleanup`
- `system.health_digest`

Нельзя:

- регистрировать jobs в случайных местах кода;
- прятать расписание внутри module internals.

## 6. Правила bootstrap boundary

Bootstrap имеет право:

- строить контейнер зависимостей;
- валидировать конфиг;
- регистрировать jobs;
- запускать приложение.

Bootstrap не имеет права:

- принимать collector/CRM решения;
- писать бизнес-state;
- дублировать business flows.

## 7. Shutdown flow

При остановке приложения:

1. остановить приём новых update;
2. пометить runtime session как closing;
3. завершить текущие безопасные jobs;
4. сохранить job state;
5. записать shutdown event;
6. закрыть adapters.

Цель:

- не оставлять грязный state после рестарта;
- не плодить “зависшие из-за падения” сценарии.

## 8. Minimal bootstrap file set

Минимальный набор файлов:

- `app/bootstrap/main.py`
- `app/bootstrap/container.py`
- `app/bootstrap/startup_checks.py`
- `app/bootstrap/job_registry.py`
- `app/config/settings.py`
- `app/logging/setup.py`

## 9. Критерий правильного bootstrap

Bootstrap считается правильным, если:

1. существует одна команда запуска;
2. любой модуль поднимается через контейнер зависимостей;
3. jobs регистрируются централизованно;
4. startup может объяснить, что включено и что выключено;
5. новый runtime не зависит от случайного ручного сценария запуска.

# Unified Logging Audit 30.04.2026

## Что было

- `bot/send_reports.py` писал в дневной файл без доменной разметки:
  CRM, scheduler, Telegram transport, pipeline и state-события смешивались в один поток.
- `collector/*` имел свой локальный prefix-helper, но не был выровнен с runtime-логикой бота.
- `config.setup_logging()` создавал отдельные ad-hoc логгеры модулей без общего доменного стандарта и без единой стратегии daily rotation/retention.
- ERROR/CRITICAL события не имели встроенного runtime Telegram-alert слоя с cooldown и системной классификацией.

## Что сделано

### 1. Общий runtime logging core

Добавлен:

- `bot/logging_utils.py`

Содержит:

- `RuntimeFormatter`
- `TelegramErrorAlertHandler`
- `configure_runtime_logging()`
- `configure_module_logger()`
- `get_runtime_logger()`
- доменное определение `derive_system_for_module()`
- helper для подключения alert sender

### 2. Домены логирования

Принят единый формат доменов:

- `BOT`
- `CRM`
- `PIPELINE`
- `STATE`
- `INTEGRATION`
- `COLLECTOR`
- `STOP_CONTROL`

Типовой runtime-префикс теперь выглядит так:

```text
[CRM][FLOW]
[STATE][STORE]
[PIPELINE][MODULE]
[COLLECTOR][FLOW]
```

### 3. Bot / send_reports

`bot/send_reports.py` переведен на новый core:

- bootstrap через `configure_runtime_logging()`
- доменные логгеры:
  - `logger` → `BOT/CORE`
  - `crm_logger`
  - `sched_logger`
  - `pipeline_logger`
  - `state_logger`
  - `integration_logger`
- `log_event()` теперь маршрутизирует события по домену через `_logger_for_event()`
- критичные direct-logs для:
  - CRM flow
  - state cleanup/restore
  - log monitor
  - scheduler setup
  - transport/runtime alerts
  переведены на профильные логгеры

### 4. Collector

`collector/logging_utils.py` больше не живет отдельной локальной схемой, а использует тот же runtime-core:

- `get_collector_logger()` → `COLLECTOR/FLOW`
- `get_stop_logger()` → `STOP_CONTROL/FLOW`

`collector/collections_engine.py` переведен на `configure_runtime_logging()` для standalone runtime bootstrap.

### 5. Module loggers через config.setup_logging

`config.setup_logging()` теперь использует `configure_module_logger()`.

Это означает, что модули отчетов/парсеров, которые просто вызывают `config.setup_logging(...)`, автоматически получают:

- единый formatter;
- daily rotation;
- retention;
- системную классификацию (`PIPELINE/MODULE`, `CRM/MODULE`, и т.д.).

### 6. Runtime alerts

Добавлен встроенный Telegram alert handler для runtime errors:

- алертит admin по `ERROR/CRITICAL`;
- использует cooldown;
- не шлет бесконечные дубли одной и той же ошибки.

## Что проверено

### Синтаксис

- `python -m py_compile bot/logging_utils.py`
- `python -m py_compile bot/send_reports.py`
- `python -m py_compile collector/logging_utils.py`
- `python -m py_compile collector/collections_engine.py`
- `python -m py_compile config.py`

Результат: OK

### Regression tests

1. `python -X utf8 tests/test_crm_regression.py`

Результат:

- `Ran 6 tests`
- `OK`

Подтверждает:

- canonical duplicate merge
- active claim exclusion
- stale phone pending cleanup
- claim persistence

2. `python -X utf8 tests/test_collector_regression_hermetic.py`

Результат:

- `Ran 15 tests`
- `OK`

Подтверждает, что unified logging bootstrap не сломал collector runtime path.

3. `python -X utf8 tests/test_logging_runtime.py`

Результат:

- доменный formatter для `CRM/FLOW`
- `STOP_CONTROL/FLOW`
- `config.setup_logging()` пишет модульный pipeline-лог в новом формате

## Остатки / границы

- Не все исторические direct `logger.*(...)` в монолите `bot/send_reports.py` уже доменно размечены вручную.
  Основной routing закрыт через `log_event()`, но часть старого кода всё ещё логирует как `BOT/CORE`.
- Система стала намного лучше локализовать неизвестные инциденты, но не делает невозможным появление новых типов багов.
  Её цель — не "предсказать всё", а быстро показать:
  - в каком домене проблема,
  - это state / integration / pipeline / crm / collector / stop-control,
  - где искать root cause.

## Итог

- Runtime logging унифицирован между bot, collector и config-based module loggers.
- Домены теперь читаются моментально.
- ERROR/CRITICAL события могут доходить админу как отдельный runtime signal.
- Collector и CRM regressions не сломаны этой унификацией.

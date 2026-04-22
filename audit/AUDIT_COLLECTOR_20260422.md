# Аудит коллектора — 2026-04-22

## Рамки

- Область: `collector/approval_flow.py`, `collector/collections_engine.py`, `collector/manager_dialog.py`, `collector/client_dialog.py`, `collector/whatsapp_poller.py`, связанный scheduler в `bot/send_reports.py`
- Режим проверки: без реальной WhatsApp-отправки
- Цель: проверить цепочку `preview -> ответы менеджеров -> решение администратора -> send-approved`, исключить зависание из-за молчания менеджеров и скрытые state-баги

## Что подтверждено тестами

- `python -m py_compile collector\approval_flow.py` — OK
- `python -m py_compile collector/collections_engine.py` — OK
- `python -m py_compile bot/send_reports.py` — OK
- `$env:WHATSAPP_ENABLED='0'; $env:LIVE_SEND_ALLOWED='0'; python -X utf8 tests\test_collector.py` — `266/266`
- `python -X utf8 tests\test_phase2_safe_send.py` — `PHASE2 SAFE SEND TESTS PASSED`

Примечание:
- первый запуск `tests/test_phase2_safe_send.py` в песочнице упал не по логике коллектора, а по `PermissionError` на `logs/collector_20260422.log`;
- повторный прогон вне песочницы прошёл успешно.

## Найдено и исправлено

### F-001 — `load_latest_batch()` считал финальные send-статусы незавершёнными

- Файл: `collector/approval_flow.py`
- Симптом: батчи со статусами `sent`, `partially_sent`, `send_failed`, `send_empty` могли считаться "последним активным батчем"
- Риск: повторные административные действия по уже финализированному батчу, путаница в UX и при ручных сервисных командах
- Исправление: финальные send-статусы добавлены в список финальных для `load_latest_batch()`
- Доказательство:
  - тест `APPROVAL T3c`
  - `tests/test_collector.py` зелёный

### F-002 — истёкший батч не сохранял явную причину зависания по молчавшим менеджерам

- Файл: `collector/approval_flow.py`
- Симптом: `expire_old_batches()` переводил батч в `expired`, но оставлял молчавших менеджеров в `pending` / `manual_editing`
- Риск: после истечения батча в state терялась явная причина, кто именно завис по молчанию
- Исправление:
  - при expiry выставляется `expired_at`
  - менеджеры со статусами `pending` и `manual_editing` переводятся в `timeout`
- Доказательство:
  - тесты `APPROVAL T10c`, `APPROVAL T10d`
  - `tests/test_collector.py` зелёный

### F-003 — новый preview-батч мог создаться поверх старого активного

- Файлы: `collector/collections_engine.py`, `collector/approval_flow.py`
- Симптом: при новой актуальной дебиторке система могла создать ещё один запрос, оставив старый активным
- Риск: параллельные запросы менеджерам, путаница у администратора, ответы не в тот актуальный список
- Исправление:
  - перед созданием нового preview старый активный батч помечается как `superseded`
  - старые manager-preview сообщения закрываются и теряют кнопки
  - в новом админском уведомлении явно отмечается, какой предыдущий запрос заменён
- Доказательство:
  - тесты `APPROVAL T3d`, `APPROVAL T10g`
  - `tests/test_collector.py` зелёный

### F-004 — молчание менеджеров больше часа блокировало процесс до истечения батча

- Файлы: `collector/approval_flow.py`, `bot/send_reports.py`
- Симптом: если хотя бы один менеджер не ответил, запрос мог тихо висеть до `expired`
- Риск: администратор не получает рабочее решение вовремя, цепочка зависает без боевого результата
- Исправление:
  - через 1 час молчания запрос автоматически переводится на этап решения администратора
  - молчавшие менеджеры получают `timeout`
  - администратору сразу отправляется итоговая сводка для ручного решения
  - старые manager-callback по такому запросу блокируются
- Доказательство:
  - тесты `APPROVAL T10e`, `APPROVAL T10f`, `APPROVAL T10h`
  - `python -m py_compile bot/send_reports.py` — OK
  - `tests/test_collector.py` зелёный

### F-005 — тесты коллектора писали в боевой `collector_YYYYMMDD.log`

- Файлы: `collector/collections_engine.py`, `tests/test_collector.py`, `tests/test_phase2_safe_send.py`
- Симптом: тестовые прогоны попадали в рабочий лог коллектора, после чего `log_monitor` слал администратору ложные тревоги
- Риск: шум в продовом мониторинге, ложные ERROR по legacy `--send`, путаница при реальной диагностике
- Исправление:
  - добавлен явный тестовый режим `COLLECTOR_TEST_MODE=1`
  - в этом режиме `collector/collections_engine.py` больше не пишет в `logs/collector_YYYYMMDD.log`, а логирует только в stdout
  - оба тестовых входа коллектора выставляют этот флаг до импорта `collections_engine`
- Доказательство:
  - `tests/test_collector.py` — `266/266`
  - `tests/test_phase2_safe_send.py` — PASS
  - хвост `logs/collector_20260422.log` не получил новых записей после повторного тестового прогона в `06:23`

## Что уже было в рабочем дереве и дополнительно верифицировано

- раннее уведомление администратору о создании батча
- ручной выбор клиентов администратором перед отправкой
- кнопка `Отправить сейчас` после admin approve
- safe-send путь использует только `approved_clients`
- legacy `--send` по-прежнему заблокирован

## Проверка последовательности

Текущая подтверждённая цепочка:

1. Scheduler запускает `collector/collections_engine.py --preview`
2. `run_approval_preview()` формирует shortlist и батч
3. `send_manager_previews()` отправляет превью менеджерам
4. `send_admin_preview_notice()` сразу уведомляет администратора о новом батче
5. После ответов всех менеджеров батч переходит в `pending_admin`
6. Администратор может:
   - вручную выбрать клиентов
   - утвердить отправку
   - запустить `Отправить сейчас`
7. `send_approved_batch()` отправляет только `approved_clients`
8. `record_send_results()` пишет результат обратно в батч
9. Если менеджеры молчат 1 час, `promote_silent_batches_to_admin()` переводит запрос к администратору
10. При истечении батча `expire_old_batches()` ставит `expired` и `timeout` у молчавших менеджеров

## Остаточные риски без кода-фантазий

### R-001 — import-time `FileHandler` в `collector/collections_engine.py`

- Симптом: в боевом режиме модуль по-прежнему использует дневной `collector_YYYYMMDD.log`
- Статус:
  - тестовый хвост закрыт через `COLLECTOR_TEST_MODE=1`
  - продовая схема логирования сознательно не перепроектировалась

### R-002 — истечение approval-батча всё ещё не шлёт отдельное Telegram-уведомление админу

- Сейчас:
  - админ уже видит раннее превью батча
  - в state теперь явно сохранится `timeout`
- Но отдельного сообщения "батч истёк" код пока не посылает
- Статус: оставлено как осознанный остаточный UX-риск, не как скрытая state-поломка

## Изменённые файлы в этом проходе

- `collector/approval_flow.py`
- `tests/test_collector.py`

## След тестов / состояние

- реальная WhatsApp-отправка не выполнялась
- live env для отправки в тестах был выключен
- рабочие state-файлы коллектора не очищались и не перезаписывались массово
- временные test-batches создавались только на временных путях в тестах и удалялись внутри тестов

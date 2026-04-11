# SESSION CONTEXT

Дата фиксации: 2026-04-09
Проект: `GPT1C_Processor_analitica`

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

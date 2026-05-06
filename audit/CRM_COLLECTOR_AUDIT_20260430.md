# CRM и Collector Audit 30.04.2026

## Что было

- В CRM один и тот же клиент мог жить несколькими ключами из-за разницы в пробелах и вариантах имени.
- Из-за этого один и тот же клиент мог снова попадать в поток `Чей клиент?`, хотя у канонического дубля менеджер уже был назначен.
- `crm_claim` жил только в памяти процесса. После рестарта старые claim-token терялись, а короткие токены не давали надежной restart-safe семантики.
- Логирование collector было неполным по системе: не все модули использовали единый `[COLLECTOR]` logger, а CRM вообще не имела отдельного audit trail.
- В ходе внедрения CRM persistence/logging был сломан callback-блок `weekly_deny/crm_claim` в `bot/send_reports.py`. Дефект был локализован и исправлен до финальной верификации.

## Что сделано

### CRM

- Добавлена канонизация ключа клиента в `bot/crm_clients.py`:
  - `canonicalize_client_key()`
  - `_find_existing_client_key()`
- `update_from_reports()` теперь:
  - схлопывает канонические дубли,
  - не плодит второй ключ из-за лишних пробелов,
  - сохраняет alias-варианты имени.
- `crm_claim` в `bot/send_reports.py` теперь:
  - использует restart-safe state в `logs/crm_claim_pending_state.json`,
  - создает уникальные токены формата `claim_<timestamp>_<uuid8>`,
  - при взятии клиента назначает менеджера сразу всем каноническим дублям,
  - пишет CRM audit события:
    - `claim_broadcast`
    - `claim_taken`
    - `claim_phone_chain_started`
- Добавлен отдельный CRM audit logger:
  - `bot/crm_audit_log.py`
  - лог: `logs/crm_audit.jsonl`

### Collector

- Добавлен общий helper:
  - `collector/logging_utils.py`
- Collector-модули переведены на единый logger с префиксом `[COLLECTOR]`.
- В `collector/client_dialog.py` добавлены audit события по цепочке:
  - `dialog_started`
  - `incoming_ignored_no_dialog`
  - `incoming_ignored_inactive_state`
  - `client_reply_received`
  - `payment_proof_received`
  - `payment_claim_reported`
  - `dialog_escalated`
  - `wa_reply_sent`
  - `wa_reply_failed`
- В `collector/whatsapp_poller.py` добавлены audit события:
  - `wa_incoming_skipped`
  - `wa_incoming_received`
  - `wa_notification_deleted`

## Доказательства

### Компиляция

Успешно прошел `py_compile` для:

- `bot/crm_clients.py`
- `bot/send_reports.py`
- `bot/crm_audit_log.py`
- `collector/logging_utils.py`
- измененных `collector/*.py`
- `tests/test_crm_regression.py`

### Тесты

1. `python -X utf8 tests/test_crm_regression.py`

Результат:

- `Ran 4 tests`
- `OK`

Покрытие:

- канонический дубль не создает вторую CRM-запись;
- unowned claim-flow не поднимается, если канонический sibling уже закреплен;
- claim-state переживает save/load;
- `crm_audit.jsonl` реально пишет JSONL-события.

2. `python -X utf8 tests/test_collector_regression_hermetic.py`

Результат:

- `Ran 15 tests`
- `OK`

Покрытие:

- freshness gate,
- stale approved batch refresh,
- paid-claim proof flow,
- attachment metadata,
- legacy tail msg_type,
- preview/admin summary freshness formatting.

3. `python -X utf8 tests/test_audit_log.py`

Результат:

- `7 прошли, 0 упали`

Покрытие:

- append-only audit JSONL,
- `read_recent()`,
- suppress audit trail.

### Логи

Проверен live tail `logs/send_reports_20260429.log`.

Подтверждено:

- Green API poller отвечает `HTTP/1.1 200 OK`;
- Telegram `getUpdates` отвечает `HTTP/1.1 200 OK`;
- признаков блокировки WhatsApp нет;
- scheduler и pipeline живы.

Вывод по live-каналу:

- WhatsApp не заблокирован;
- угрозы на момент проверки были логическими и state-related, а не сетевыми.

## Итог

- CRM-дубли по каноническому имени больше не должны порождать повторный `Чей клиент?`.
- `crm_claim` теперь restart-safe.
- Collector получил системное audit-логирование по всей цепочке от диалога до входящего WhatsApp.
- Callback-поломка, внесенная во время патча, исправлена и перепроверена тестами.

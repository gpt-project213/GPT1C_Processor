# AUDIT_RUNTIME_TRACE

## Источники runtime-доказательств

- `logs_public/send_reports_20260305.log`
- `logs_public/send_reports_20260301.log`
- `logs_public/send_reports_20260306.log`
- `logs_public/deletion_queue.json`
- `logs_public/notify_state.json`

## 1) Scheduler / startup

Подтверждено логами:
- бот стартует, index строится, polling поднимается;
- джобы добавляются в scheduler;
- запускаются pipeline cycles и внешние скрипты.

Признаки деградации:
- много предупреждений `Chat not found` при стартовых уведомлениях;
- в логах много записей `run_script_finish` (активность высокая), но полезный business-result по доставке не всегда гарантирован.

## 2) Pipeline trace

Подтверждено:
- `pipeline_cycle_start` регулярно возникает;
- `imap_fetcher.py --once` отрабатывает и возвращает rc=0;
- после IMAP запускаются report/parser scripts по нескольким типам (debt/gross/inventory/expenses/sales).

Риск ложной видимости:
- большие серии `run_script_finish return_code=0` не эквивалентны подтверждённой доставке пользователям;
- есть `SKIP no-manager-in-name` во входящем потоке.

## 3) IMAP/Queue trace

Подтверждено:
- `SEARCH found ... msgs`;
- множественные `SAVED: reports\queue\...`;
- `Clean copy: reports\excel\clean\...`;
- `MAIL flagged \Deleted`, `INBOX EXPUNGE OK`.

Выявлено:
- часть писем/вложений отбрасывается фильтрами (включая manager-in-name), при этом некоторые ветки логируются не на INFO.

## 4) Telegram delivery trace

Подтверждено частично:
- бот отправляет стартовые уведомления минимум админу и части получателей.

Нарушения:
- повторяющиеся `Chat not found` для части менеджеров → контур доставки неполный.

## 5) State runtime trace

`logs_public/deletion_queue.json`:
- содержит множество pending jobs на удаление сообщений;
- формат в snapshot редактирован (`[ID]`), JSON непарсибелен стандартным parser.

`logs_public/notify_state.json`:
- snapshot непарсибелен (редактированные path/id маркеры).

Следствие:
- невозможно машинно подтвердить целостность state на этом артефакте; требуется доступ к оригинальным runtime-файлам.

## 6) Collector / promises

По коду и тестам:
- контур `--check-promises` реализован и тесты коллектора проходят;
- state блокировки по `already_contacted_today`, promise flags, escalation flags активны;
- reminders и poller запускаются джобами в основном боте.

Риск:
- при stale state возможны блокировки повторной отправки/эскалации до ручного сброса флагов.

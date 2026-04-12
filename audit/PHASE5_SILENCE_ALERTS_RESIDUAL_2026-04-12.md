# PHASE 5 — краткие уведомления по возрасту остатка

Автор: Codex
Дата: 2026-04-12

## Что проверено

`contacts.xlsx` в корне проекта проверен только чтением.

Назначение файла:

- справочник контактов клиентов;
- лист `Контакты клиентов`;
- колонки: `Клиент (1С)`, `Менеджер`, `Имя контакта`, `WhatsApp`, `Адрес`, `Язык`, `Не звонить`, `Источники`, `Первое появление`, `Последнее видели`;
- строк: 422;
- дата изменения: 2026-03-27 12:59:55.

Файл не менялся.

## Проблема

Краткие уведомления о "молчунах" использовали старую HTML-колонку `Дни молчания`.

После Phase 5 это стало небезопасно:

- свежий остаток мог выглядеть старым, если последняя оплата была давно;
- старый остаток мог быть пропущен, если старая колонка была маленькой.

## Что изменено

`bot/silence_alerts.py` теперь обогащает клиентов данными классификатора:

- `residual_debt_age_days`
- `oldest_unpaid_date`
- `debt_age_basis`
- `payment_silence_days`

Категоризация уведомлений теперь использует `residual_debt_age_days`, если он найден.

Старая `silence_days` остается fallback только для случаев, где нет данных по движениям.

## Текст уведомлений

Вместо неоднозначного:

`19 дн`

теперь выводится:

`остаток 3 дн, с 2026-04-08`

Так менеджер видит возраст текущего остатка, а не тишину с последней оплаты.

## Точки подключения

Обновлены оба пути:

- плановая отправка `check_and_send_silence_alerts()`;
- ручной запрос через `force_report_to_user(..., "silence", ...)`.

## Excel для сверки

Создан файл:

`reports/excel/collector_classification_11_04_2026_phase5_by_manager.xlsx`

Листы:

- `Все по менеджерам`
- `Возраст 10+`
- `Спорные старые`
- `Алена`
- `Ергали`
- `Магира`
- `Оксана`
- `Сводка`

Сводка:

- всего клиентов: 87;
- возраст остатка 10+ дней: 25;
- свежий остаток 0-9 дней: 62;
- старая логика `days_silence` 10+ дней: 11;
- старая 10+, новая 0-9: 1;
- новая 10+, старая <10: 15.

## Тесты

Пройдены:

```powershell
.\.venv\Scripts\python.exe -m py_compile bot\silence_alerts.py bot\send_reports.py tests\test_project.py tests\test_collector.py
.\.venv\Scripts\python.exe -X utf8 tests\test_project.py
.\.venv\Scripts\python.exe -X utf8 tests\test_collector.py
.\.venv\Scripts\python.exe -X utf8 tests\test_phase2_safe_send.py
```

Результаты:

- `tests/test_project.py`: 64/64
- `tests/test_collector.py`: 217/217
- `tests/test_phase2_safe_send.py`: passed
- `py_compile`: passed

## Что не менялось

- `contacts.xlsx`;
- `.env`;
- live send;
- stop-list;
- active-guard;
- новый бот.

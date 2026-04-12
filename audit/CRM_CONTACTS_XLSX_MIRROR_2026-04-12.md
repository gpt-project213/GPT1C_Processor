# CRM contacts.xlsx mirror — 2026-04-12

Автор: Codex

## Что было

- `config/clients.json` уже был рабочей CRM-базой для бота.
- `contacts.xlsx` в корне проекта был ручной Excel-копией и не обновлялся автоматически.
- Из-за этого Excel перестал соответствовать действующей CRM-базе.

## Что изменено

- `contacts.xlsx` теперь обновляется автоматически после успешного сохранения CRM через `bot.crm_clients.save_clients()`.
- Источник данных только один: `config/clients.json`.
- Обновление одностороннее: `clients.json` -> `contacts.xlsx`.
- Excel не импортируется обратно в CRM автоматически.
- Перед первой перезаписью за день старый `contacts.xlsx` копируется в `backups/contacts_xlsx/`.
- Если Excel открыт или недоступен, CRM всё равно сохраняется, а ошибка обновления Excel пишется в лог как warning.

## Что не изменено

- Safe-send архитектура не трогалась.
- Коллектор, классификация, stop-list, active-guard и `.env` не менялись.
- `contacts.xlsx` не стал источником правды для бота.
- Синхронизация с Excel обратно в CRM не включалась.

## Проверка реального файла

- Выполнено ручное обновление зеркала из CRM.
- Результат: `contacts.xlsx` обновлён из `config/clients.json`.
- Количество клиентов в экспорте: 551.
- Текущий файл: `E:\GPT1C_Processor_analitica\contacts.xlsx`.
- Размер после обновления: 45090 bytes.
- Строк в Excel: 552, включая заголовок.
- Листы: `Контакты клиентов`, `Инструкция`.
- Backup создан: `backups/contacts_xlsx/contacts_20260412_235311.xlsx`.

## Тесты

- `python -m py_compile bot/crm_clients.py tools/contacts_sync.py tests/test_project.py bot/silence_alerts.py bot/send_reports.py tests/test_collector.py`
- `python -X utf8 tests/test_project.py` -> 67/67
- `python -X utf8 tests/test_collector.py` -> 217/217
- `python -X utf8 tests/test_phase2_safe_send.py` -> passed

## Как теперь работает

1. Пользователь или бот меняет CRM-карточку.
2. Код сохраняет `config/clients.json` атомарно.
3. После успешного сохранения автоматически запускается экспорт CRM в `contacts.xlsx`.
4. Excel становится актуальной читаемой копией CRM.
5. Если Excel не удалось обновить, CRM-данные не теряются.

## Ограничения

- Это не двусторонняя синхронизация.
- Если править только `contacts.xlsx`, CRM сама не изменится.
- Для массового ручного импорта из Excel остаётся отдельный инструмент `tools/contacts_sync.py`.

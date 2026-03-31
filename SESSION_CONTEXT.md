# SESSION_CONTEXT.md
> Автоматически обновляется Claude Code. Последнее обновление: 2026-03-31

## Текущее состояние проекта

### Коммиты сессии
- `bd65238` — Саида, стоп-лист, рентабельность (сессия утро)
- `cbf4b98` — умный расчёт days_silence + shipment_violation

### Что сделано сегодня

#### delay=True в FileHandler (config.py)
Пустые лог-файлы не создаются при импорте.

#### sales_profitability_report.py
Только LOSS/CRITICAL клиенты. Было ~310 → ~20-30 строк.

#### bot/debt_stop_control.py — НОВЫЙ
Стоп-лист для Саиды (ID 920236287): 14:00/17:00/19:00/22:00, авто-стоп 15+ дней, реестр нарушителей, автоудаление 24ч.

#### Клиенты Ергали — WhatsApp
6 клиентов, формат +7. Правильное имя: `Е ИП Реян (Жангали)` (не Реан).

#### debt_auto_report.py — умный days_silence (коммит cbf4b98)

**Новые поля ClientBlock:**
- `last_debit_date` — дата последней отгрузки
- `balance_before_last_debit` — баланс ДО последней отгрузки
- `last_credit_before_debit` — последняя оплата ДО отгрузки включительно
- `last_credit_date` — последняя оплата в периоде
- `shipment_violation` — True если отгрузка при долге > 5000 ₸

**Логика _calc_silence:**
1. Нет движений → period_min
2. Только оплаты (стоп) → с даты последней оплаты
3А. Долг до отгрузки ≤ 5000 → с даты отгрузки
3Б. Долг > 5000 + есть оплата после отгрузки → с даты отгрузки
3Б. Долг > 5000 + нет оплаты после → с последней оплаты ДО отгрузки

`shipment_violation` пишется в `debt_ext_*.json`.
Тест Ергали (122): 5 клиентов изменили дни, 11 нарушений.

---

## СЛЕДУЮЩИЙ ШАГ

### shipment_violation в краткое уведомление (silence_alerts.py)
- Сейчас `silence_alerts.py` читает данные из HTML
- `shipment_violation` уже есть в `debt_ext_*.json`
- Эффективнее читать из JSON напрямую (не парсить HTML)
- Добавить в уведомление: `⚠️ Нарушение: отгрузка при наличии долга`
- Точка входа: `check_and_send_silence_alerts` в `send_reports.py` (~строка 3593)

---

## Пользователи
| Роль | Имя | chat_id |
|---|---|---|
| admin | Вадим | 7422963573 |
| subadmin | Алена | 188939016 |
| manager | Оксана | 1446255940 |
| manager | Магира | 735574334 |
| manager | Ергали | 756622791 |
| accountant | Саида | 920236287 |

## Ключевые файлы
- `bot/send_reports.py` — основной бот + APScheduler
- `bot/debt_stop_control.py` — стоп-лист Саиды
- `bot/silence_alerts.py` — краткие уведомления → нужно добавить shipment_violation из JSON
- `debt_auto_report.py` — парсинг Excel + days_silence
- `config/clients.json` — база клиентов (.gitignore)
- `config/roles.json` — роли (.gitignore)
- `E:/GPT1C_Processor_analitic/debt_auto_report.py` — ОРИГИНАЛ (не трогать!)

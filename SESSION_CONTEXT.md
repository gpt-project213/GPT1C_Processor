# SESSION_CONTEXT.md
> Автоматически обновляется Claude Code. Последнее обновление: 2026-04-08

## Текущее состояние проекта

### Коммиты сессии (2026-04-08) — WhatsApp тест + аудит
- `b9a0a4f` — боевой режим коллектора + COLLECTOR_DRY_RUN=false
- WhatsApp первый реальный тест: incomingWebhook включён, extendedTextMessage обработан
- Полный комплексный аудит всего проекта (6 критических багов найдено, план зафиксирован)

#### Что сделано в сессии 2026-04-08

**WhatsApp (Green API) — диагностика и фиксы:**
- Обнаружено: `incomingWebhook: no` → включено через API `setSettings({"incomingWebhook": "yes"})`
- Причина что WA не работал раньше: DRY_RUN=true + бот на F:/ + нет телефонов в базе
- Добавлен обработчик `extendedTextMessage` в `whatsapp_poller.py` (reply-сообщения)
- Проведён live-тест: отправка → ответ → DiepSeek анализ → escalation → всё работает ✓

**collector/collection_agent.py (v1.0.1):**
- Добавлен `from datetime import datetime` (был missing import для analyze_response)
- Добавлена текущая дата в system_prompt analyze_response (`сегодняшняя дата: YYYY-MM-DD`)
- Переработана логика intent: расплывчатые даты ("завтра", "скоро") → `delay_request`,
  конкретная дата (число месяца) → `promise`

**collector/client_dialog.py (v1.0.3):**
- `delay_request` больше не эскалирует к менеджеру — продолжает диалог с уточнением даты
- Используется `suggested_reply` от DeepSeek вместо хардкода
- Добавлена функция `_schedule_tg_deletion()` → авто-удаление эскалаций через 24ч
- `_send_tg()` теперь планирует удаление после успешной отправки
- Запись в `deletion_queue.json` атомарная

**collector/communications.py (v1.0.1):**
- `send_telegram()` теперь возвращает `Optional[int]` (message_id) вместо `bool`
- Использутся для авто-удаления сообщений коллектора

**collector/whatsapp_poller.py (v1.0.2):**
- Добавлен обработчик `extendedTextMessage` (цитируемые ответы WhatsApp)

---

### Коммиты сессии (2026-03-31 / 2026-04-01)
- `bd65238` — Саида, стоп-лист, рентабельность (сессия утро)
- `cbf4b98` — умный расчёт days_silence + shipment_violation
- `e365931` — стоп-контроль: финальное решение руководителя по всем клиентам
- `d25fc6e` — fix: 3 бага — меню после AI, Button_data_invalid, q.answer() return

### Что сделано

#### bot/debt_stop_control.py
**Финальное решение — всегда за руководителем:**
- Менеджер `✅ Договорились` → немедленная эскалация к руководителю (не авто-исключение)
- Менеджер `🚫 Нет/молчит` → запрос руководителю
- После решения руководителя → уведомление менеджеру всегда
- Стоп утверждён → менеджеру кнопка `💳 Клиент оплатил, ждём разноски`
- Статус `stopped` записывается в реестр для отслеживания оплаты
- `monitor_exceptions` отслеживает `stopped` + `auto_stopped`
- `send_saida_final` включает `stopped` из реестра

**Цепочка срочной оплаты:**
1. Менеджер `💳 Клиент оплатил, ждём разноски`
2. Саиде запрос: `✅ Полная оплата` / `⚠️ Частичная оплата` + руководителю инфо
3. Саида `✅ Полная` → руководителю: `✅ Разрешить` / `🚫 Отказать`
4. Саида `⚠️ Частичная` → руководителю предупреждение о конфликте + менеджеру стоп остаётся
5. Руководитель разрешает → менеджеру + Саиде уведомление

**Снятие со стопа:**
- 14:00 мониторинг: долг=0 → руководителю запрос `✅ Снять` / `🚫 Оставить`
- Снял → менеджеру + Саиде уведомление
- `callback_data` обрезка `[:26]` (кириллица 2 байта, лимит 64 байта)

#### bot/send_reports.py
- Меню восстанавливается после AI-отчёта (`send_main_menu` после каждого `handle_ai_only`)
- `q.answer()` error — убран `return`, callback продолжает обрабатываться
- `shipment_violation` из JSON: `_ext.get('clients', [])` — правильная итерация
- fix UnboundLocalError datetime в `cb_data` (строка ~4799)

#### bot/silence_alerts.py
- `🚨 нарушение отгрузки` суффикс для клиентов с `shipment_violation=True`

#### debt_auto_report.py
- `shipment_violation`, `balance_before_last_debit`, логика `_calc_silence`

#### .env
- `COLLECTOR_DRY_RUN=false` — боевой режим коллектора

---

## АУДИТ 2026-04-08 — Результаты

**Оценка:** production-ready 7.5/10. Найдено 6 критических багов, 3 высоких, 4 оптимизации.

### Критические баги (исправить первыми)

| ID | Файл | Проблема |
|----|------|----------|
| BUG-1 | `collector/collection_agent.py:93` | `data["choices"][0]` → IndexError если DeepSeek вернул `choices: []`. Фикс: проверить `if not data.get("choices")` |
| BUG-2 | `run_pipeline_all_mp.py` | Файлы типа SKIP не перемещаются в `processed/` → накапливаются, повторяются. Фикс: вызывать `_move_to_processed()` при `routed_to == "SKIP"` |
| BUG-3 | `collector/debt_monitor.py:~221` | `"1,234,567".replace(",",".")` → `"1.234.567"` → ValueError. Фикс: убрать пробелы, потом `.replace(",", ".", 1)` |
| BUG-4 | `collector/communications.py:137,185` | `int(ADMIN_CHAT_ID)` без try/except → ValueError при нечисловом значении |
| BUG-5 | `bot/inventory_summary.py:~196` | `max(files, ...)` → ValueError на пустом списке. Фикс: `if not files: return` |
| BUG-6 | `collector/collections_engine.py:~357` | `int(tg_id)` без защиты → TypeError если tg_id=None |

### Высокий приоритет

| ID | Проблема |
|----|----------|
| HIGH-1 | Зависшие `.work` файлы в `reports/excel/active/` от 07.04 — вручную перенести в `processed/` |
| HIGH-2 | `sales_report.py`: `ensure_clean_xlsx()` может вернуть None → `pd.read_excel(None)` → TypeError |
| HIGH-3 | `money_to_float` дублирован в `debt_auto_report.py` и `analyze_debt_excel.py` с разными сигнатурами |

### Оптимизации

| ID | Проблема | Фикс |
|----|----------|------|
| OPT-1 | Scheduler 22:00: `daily_analytics` + `debt_stop_saida` конкурируют | Сдвинуть `debt_stop_saida` на 22:15 |
| OPT-2 | Мёртвый код: `analyze_debt_excel.py` нигде не импортируется в prod | Проверить grep, удалить |
| OPT-3 | `bot/crm_clients.py _strip_legal`: после regex-удаления префиксов нет `.strip()` | Добавить `.strip()` |
| OPT-4 | Лог-файлы: 15 МБ/день из-за httpx poll каждые 30 сек | Снизить уровень для whatsapp_poller |

### Архитектурные (долгосрочно)
- ARCH-1: нет file locking для state JSON файлов (риск низкий — один процесс)
- ARCH-2: нет тестов для cb_data routing, handle_* обработчиков, pipeline_task
- ARCH-3: `bot/send_reports.py` монолит 6687 строк — вынести callbacks/handlers

---

## СЛЕДУЮЩИЙ ШАГ
- Исправить 6 критических багов (BUG-1 — BUG-6) по плану в `floating-booping-parasol.md`
- Сдвинуть debt_stop_saida на 22:15 (OPT-1)
- Мониторинг 17:30: первые боевые WhatsApp сообщения (3 клиента Ергали)

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
- `bot/debt_stop_control.py` — стоп-лист Саиды (полная логика)
- `bot/silence_alerts.py` — краткие уведомления + shipment_violation
- `debt_auto_report.py` — парсинг Excel + days_silence + shipment_violation
- `config/clients.json` — база клиентов (.gitignore)
- `config/roles.json` — роли (.gitignore)
- `reports/debt_stop_state.json` — суточное состояние стоп-листа
- `reports/debt_stop_registry.json` — постоянный реестр нарушителей
- `E:/GPT1C_Processor_analitic/debt_auto_report.py` — ОРИГИНАЛ (не трогать!)

## Боевые параметры
- `COLLECTOR_DRY_RUN=false` — коллектор звонит реально
- `GREENAPI_ID=7107551921` — WhatsApp боевой аккаунт
- `TG_BOT_TOKEN=8091477403:...` — боевой бот @Report1CProBot

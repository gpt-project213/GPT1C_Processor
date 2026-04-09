# SESSION_CONTEXT.md
> Автоматически обновляется Claude Code. Последнее обновление: 2026-04-08

## Последние коммиты

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
- `e385c24` — условная отгрузка, added_by, Саида в реальном времени, коллектор пропускает стоп-лист
- `9d3c7fc` — откат субботы — только воскресенье выходной по умолчанию
- `7624062` — fix: 4 критических бага — остатки 0 ед, WhatsApp 18:05→17:30, check_workday 12:00→07:30, collector_reminder time guard
- `2be3687` — fix: 3 бага аудита — whatsapp timeout, ai_daily спам, Telegram error handler
- `ea38b56` — fix: порог пропуска авто-стопа 1000 → 2000 ₸


## Развёртывание

- **Dev:** `E:\GPT1C_Processor_analitica` (git repo, сюда коммиты)
- **Прод (нетоп):** `C:\Users\user\Documents\GPT1C_Processor_analitica`
- **Деплой:** `git pull origin master` на нетопе
- **Не заменять:** `.env`, `config/clients.json`, `config/roles.json`, `reports/*.json`

## Изменения с последнего аудита (2026-04-08)

### bot/debt_stop_control.py (коммит e385c24)
- Поле `added_by`: `"auto"` (авто-стоп 15+ дней) / `"admin_manual"` (руководитель нажал)
- Саида уведомляется **в реальном времени** при утверждении стопа и при авто-стопе
- Третья кнопка при debt=0: **"⚠️ Условная отгрузка"** → статус `conditional`
- При `conditional` + долг=0 → авто-снятие с уведомлением всех троих
- `already_controlled` расширен: `stopped`, `conditional` теперь тоже блокируют повторное добавление

### collector/collections_engine.py (коммит e385c24)
- Фильтр стоп-листа: клиенты со статусом `stopped/auto_stopped/pending_clearance/conditional` пропускаются

### bot/send_reports.py (коммит 7624062)
- `debt_collector_daily`: 18:05 → **17:30** (WhatsApp работает в 9-18ч)
- `check_workday_task`: 12:00 → **07:30** (до отчётов в 09:00)
- `collector_reminder_task`: добавлен `is_holiday_today()` + `9 <= hour < 18`
- `send_inventory_summary()`: JSON-первый путь → HTML-fallback

### bot/inventory_summary.py (коммит 7624062)
- v1.4: `parse_inventory_json()`, `get_latest_inventory_json()`
- Новый JSON-формат: `total_qty` / `categories[].item_list[].qty`

### bot/workday_checker.py (коммит 9d3c7fc)
- Только `weekday() == 6` (воскресенье) = выходной по умолчанию
- Суббота — рабочий день

## Статусы реестра (debt_stop_registry.json)

| Статус | Значение |
|---|---|
| `exception` | Менеджер договорился, руководитель одобрил |
| `stopped` | Руководитель утвердил стоп (`added_by: admin_manual`) |
| `auto_stopped` | 15+ дней молчания (`added_by: auto`) |
| `pending_clearance` | Долг=0 в 1С, ждём решения руководителя |
| `conditional` | Условная отгрузка разрешена, ждём разноски |
| `cleared` | Снят со стопа |

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


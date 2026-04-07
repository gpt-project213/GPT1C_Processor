# SESSION_CONTEXT.md
> Автоматически обновляется Claude Code. Последнее обновление: 2026-04-08

## Последние коммиты

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

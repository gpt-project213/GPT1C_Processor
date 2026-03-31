# SESSION_CONTEXT.md
> Автоматически обновляется Claude Code. Последнее обновление: 2026-03-31

## Текущее состояние проекта

**Ветка**: `master`
**Последний коммит**: `04c5f0f` (2026-03-30)
**Статус тестов**: 62/62 + все collector ✅
**Открытые баги**: 0 критических / 0 высоких / 2 архитектурных (не критично)
**Платформа**: основной ПК — `E:\GPT1C_Processor_analitica`

---

### Сессия 2026-03-31 (стоп-лист Саиды + рентабельность + delay=True)

**Новый пользователь:** Саида (бухгалтер-оператор, chat_id: 920236287) — добавлена в `config/roles.json` → `accountants`.

**1. delay=True в FileHandler (`config.py`)**
- Пустые `utils_excel_*.log` создавались каждый час при импорте модуля
- Исправлено: `fh = logging.FileHandler(log_path, encoding="utf-8", delay=True)`
- Файл появляется только при первой реальной записи

**2. `sales_profitability_report.py` — фильтр клиентов**
- Было: раздел «Клиенты и товары» — все клиенты, все товары (~310 строк)
- Стало: только клиенты у которых есть LOSS (< 0%) или CRITICAL (< 5%) позиции
- Только эти проблемные товары, остальное скрыто
- Заголовок: «N из M клиентов с проблемами»

**3. `bot/debt_stop_control.py` — НОВЫЙ МОДУЛЬ (стоп-лист)**

Расписание:
- **14:00** — мониторинг реестра: авто-стоп при 15+ днях (даже ранее одобренные)
- **17:00** — запросы менеджерам с кнопками (✅ Договорились / 🚫 Нет, стоп)
- **19:00** — нет ответа за 2 часа → эскалация руководителю
- **22:00** — Саида получает финальный список «не отгружать»

Пороги: 7–9 дней ⚡ / 10+ дней 🔴 / 15+ дней 🚫 авто-стоп

Исключения: `weekly_clients.json` + `clients.json[payment_terms_days]`

Нарушители дисциплины (авто-стоп):
- `discipline_violation=True` → следующий раз сразу к руководителю (минуя менеджера)
- Снятие стопа только после полной оплаты + решение руководителя

Для Саиды:
- Первый запуск: инструкция простым языком (один раз, хранится 3 дня)
- Каждый клиент — отдельное сообщение + кнопка «💰 Оплата получена»
- Нажала → руководитель уведомлён, Саида получает разрешение грузить
- Все сообщения автоудаляются через 24ч (тот же `deletion_queue.json`)

Состояние:
- `reports/debt_stop_state.json` — суточное (сбрасывается каждое утро)
- `reports/debt_stop_registry.json` — постоянное (нарушители, авто-стопы, одобрения)
- `reports/debt_stop_saida_intro_sent.json` — флаг первого запуска

**Интеграция в `bot/send_reports.py`:**
- Импорт с try/except (`_DEBT_STOP_AVAILABLE`)
- Обработчик callback `dstop_*` в `cb_data()`
- 4 джоба в планировщике: 14:00 / 17:00 / 19:00 / 22:00

**Коммиты:**
| Коммит | Описание |
|--------|----------|
| `ee6217b` | fix: delay=True в FileHandler + фильтр клиентов в sales_profitability |
| `954e266` | feat: debt_stop_control.py + roles.json (Саида) + send_reports.py джобы |
| `e212c75` | fix: автоудаление 24ч для всех сообщений стоп-листа |
| `5c53caf` | feat: инструкция Саиде + кнопка «Оплата получена» + уведомление руководителю |

---

### Сессия 2026-03-30 (аудит простоя + баги + CRM телефоны v2)

**Простой бота**: 29.03 07:00 → 30.03 10:09 (~27 часов, краш без трейсбека)

**Исправлены баги:**
1. `collector/opportunity_loss.py` — БАГ-1: KeyError `'dead'` → добавлен `"dead": []` в zones
2. `collector/communications.py` / `whatsapp_poller.py` / `manager_dialog.py` — БАГ-2: пустые ошибки Green API → `type(e).__name__`
3. `sales_profitability_report.py` — БАГ-3: 704 товара без матча → fuzzy-матч через `difflib` (порог 0.85)

**CRM телефоны v2 (`bot/send_reports.py`):**
- `CRM_DAILY_LIMIT`: 10 → **15**
- Добавлены `_crm_save_pending()` / `_crm_load_pending()` → `logs/crm_pending_state.json`
- Добавлен `crm_phone_reminder_task`: каждый час 09–19, повторяет запрос пока менеджер не ответит
- Регистрация: `job_queue.run_repeating(crm_phone_reminder_task, interval=3600, first=600)`
- `_crm_load_pending()` в `post_init` — переживает перезапуск бота

**Состояние базы клиентов:**
- Всего клиентов: 424 | С телефоном: 2 | Без телефона: 422
- Должников в реестре: 83 | Ни у одного нет телефона
- WhatsApp: `WHATSAPP_ENABLED=0` — реальных отправок нет

**Коммиты:**
| Коммит | Описание |
|--------|----------|
| `04c5f0f` | feat: CRM телефоны — лимит 15/день, персист, напоминания каждый час |
| `6d1fdce` | fix: БАГ-3 fuzzy-матч товаров (difflib 0.85) |
| `83c4060` | fix: БАГ-2 пустые ошибки Green API |
| `010f2f3` | fix: БАГ-1 KeyError 'dead' в opportunity_loss |

---

### Сессия 2026-03-27 (CRM v2 + коллектор on-stop + contacts sync)

**Изменено:**
1. `docs/manager_guide.html` → переименован в `docs/Инструкция по работе с ботом.html`:
   - AI Collector: зелёный WhatsApp-пузырь → серый Telegram-пузырь с полным текстом сообщения клиенту
   - Иконки шагов: пронумерованные кружки (1/2/3) вместо emoji
   - CRM-раздел: база — обязанность менеджера, на контроле руководителя
2. `tools/contacts_sync.py` — новый инструмент Excel↔JSON:
   - `--export`: clients.json → contacts.xlsx (10 колонок, стили, лист «Инструкция»)
   - `--import`: contacts.xlsx → clients.json (только редактируемые поля, новых не создаёт)
3. `collector/debt_monitor.py` v1.0.1 → v1.0.2:
   - `classify_debtors()`: добавлены поля `debit` и `credit` в выходную запись
4. `collector/collections_engine.py` v1.0.2 → v1.0.4:
   - On-stop фильтр: пропускать клиентов если `debit > 0 OR credit > 0` (активны)
   - `_warn_manager_no_phone()`: предупреждение менеджеру + счётчик warn_count
   - Эскалация к Вадиму при warn_count >= 2 (клиент без телефона игнорируется)
   - Два кейса: контакт не найден совсем / найден но пустой whatsapp И telegram_id
5. `bot/crm_clients.py` v1.0.0 → v1.0.1:
   - `_UNOWNED` кортеж: `("", "Не определён", "?", "-", "—")` — были truthy, не перезаписывались
6. `bot/send_reports.py`:
   - `CRM_DAILY_LIMIT = 10` (было 1)
   - Цепочка: менеджер заполнил → сразу следующий клиент
   - `_all_crm_participants()` = MANAGERS_MAP + Вадим (admin в CRM)
   - Неопределённые клиенты: broadcast «Чей клиент?» + inline кнопка `crm_claim|{token}`
   - Кто нажал — тому клиент + цепочка; остальным «взял [имя]»
   - CRM обновляется автоматически в `pipeline_task` после обработки файлов

**Ранее закоммиченные (сессия 2026-03-27, `15772a0` и предыдущие):**

| Коммит | Описание |
|--------|----------|
| `15772a0` | fix: 4 баги из аудита логов (Green API токен 401, imap лог 1 файл/день, дубль в manager_dialog, авто-истечение диалогов 48ч) |
| `45bbdff` | feat: manager HTML guide + /guide команда |
| `f746db3` | fix: rejection_reason — сохранять verbatim, не переписывать ИИ |
| `bfb2253` | fix: is_imitation — исключать клиентов с нулевыми платежами |
| `3877f03` | feat: определение рабочего дня (воскресенье = выходной, is_holiday_today guard) |

**Незакоммиченные изменения сессии** (требуют коммита):
- `bot/crm_clients.py`, `bot/send_reports.py`, `collector/collections_engine.py`, `collector/debt_monitor.py`
- `config/clients.json` (инициализирована база из отчётов)
- `docs/Инструкция по работе с ботом.html` (переименован из manager_guide.html)
- `tools/contacts_sync.py` (новый)

---

### Сессия 2026-03-25 (CRM v1.0 + коллектор 18:00)

**Изменено:**
1. `bot/crm_clients.py` v1.0 — новый модуль:
   - `update_from_reports()` — автообновление из `debt_ext_*.json` + `sales_*.json`
   - `get_clients_without_phones(manager, limit=5)` — очередь запроса телефонов
   - `set_client_phone(name, phone)` — запись телефона
   - `load_contacts_compat()` — объединяет `clients.json` + `debtors_contacts.json` для коллектора
   - `get_new_clients_since(date)` — новые клиенты для уведомлений
2. `config/clients.json` — новый файл, универсальная база всех клиентов
3. `bot/send_reports.py` v9.4.38 → v9.4.39:
   - `crm_daily_task` — 18:00: обновление CRM + уведомление о новых + запрос телефонов
   - `/phone <клиент> <номер>` — команда ввода телефона менеджером
   - Коллектор перенесён: 09:00 → 18:05
   - Стартовое сообщение: `· 18:00 — база клиентов (CRM)` + `· 18:05 — коллектор`
4. `collector/collections_engine.py` v1.0.1 → v1.0.2:
   - Удалён блок авторегистрации + запроса телефона (phone-request flow)
   - Контакты из CRM (`load_contacts_compat`)
   - Клиент без телефона — тихий пропуск

**Коммиты**: `39f6bd3` → `a039d91` → `cba2218` → `fff16c3` → `8af8b4f` → `397bf76`

**Дополнительно (в рамках той же сессии):**
5. `bot/crm_clients.py`: `find_similar_clients()` (5-уровневый нечёткий поиск), `set_client_details()` (имя+телефон+адрес), `set_client_alias()`
6. `bot/send_reports.py`: FSM clarify_name→phone→address в `handle_persistent_menu`; `crm_daily_task` — один клиент на менеджера, точечный диалог; мёртвый код `reg_phone/reg_name/reg_lang` закомментирован
7. `collector/collection_agent.py`: ИИ представляется «ИИ-помощник менеджера [Имя]»; город Алматы → Астана

---

### Сессия 2026-03-25 (silence_alerts v1.7 — полный рефакторинг дней молчания)

**Изменено:**
1. `bot/silence_alerts.py` v1.6 → v1.7:
   - `MIN_DEBT_AMOUNT` 10 000 → 5 000 ₸
   - Новые категории: `critical`(30+), `alarm`(15-29), `silence`(10-14), `overdue`(7-9), `partial_payment`, `on_stop`
   - Удалена категория `warning` (заменена на `overdue` + `silence`)
   - Добавлен `IMITATION_THRESHOLD = 0.10` (< 10% от долга = имитация)
   - `on_stop` = debit==0 AND credit < 10% долга (стоп/имитация)
   - `partial_payment` = debit==0 AND credit >= 10% долга (реальная частичная оплата)
   - Активные клиенты (debit>0 или days<7) — скрыты из отчёта
   - Шапагат клиенты (weekly_clients) исключены из `overdue`
   - Обновлены форматтеры: `format_manager_alert`, `format_admin_summary`, `format_admin_detailed`
2. `bot/send_reports.py` v9.4.37 → v9.4.38:
   - `_load_weekly_clients()` / `_save_weekly_clients()` — загрузка/сохранение weekly_clients.json
   - `_suggest_weekly_clients()` — Алена получает предложение добавить Шапагат клиента
   - 4 callback-хендлера: `weekly_suggest|`, `weekly_reject|`, `weekly_confirm|`, `weekly_deny|`
   - Все счётчики `total_silent` обновлены через `_SILENCE_CATS` кортеж
3. `bot/opportunity_loss.py`: `MIN_DEBT_AMOUNT` 10 000 → 5 000 ₸ (синхронизирован)
4. `config/weekly_clients.json` — новый файл (еженедельные клиенты)
5. `tests/test_project.py` — обновлены тесты под новые категории

**Результат**: список дней молчания сократился с 85 → ~35 клиентов по всем менеджерам

---

### Сессия 2026-03-25 (аудит коллектора — BUG-C1..C5)

| ID | Файл | Fix |
|----|------|-----|
| BUG-C1 | `collector/debt_monitor.py` | `violation_shipment` добавлен порог `days >= 7` — было 55+ ложных нарушений/день |
| BUG-C2 | `collector/collections_engine.py` | Батчинг violation-уведомлений: 1 сообщение на менеджера + 1 сводка админу (вместо 110/день) |
| BUG-C3 | `collector/collections_engine.py` | `load_state()` 1 раз до цикла, `save_state()` 1 раз после (было N раз) |
| BUG-C4 | `collector/collections_engine.py` | Убран дубль `violation_flag` в no-contact сообщении |
| BUG-C5 | `collector/collections_engine.py` | `dry_run` не пишет `first_seen` в state |

**Архитектурные наблюдения (не баги):**
- BUG-C6 (ARCH): порог коллектора 10 дней vs silence_alerts OVERDUE=7 дней — намеренно (7-9 дн = мониторинг, 10+ = активное взыскание)
- Лог timestamps 05:00 на 23-24 марта — старое расписание (текущее: 09:00 Almaty)

---

### Сессия 2026-03-25 (полный аудит двух проектов + синхронизация)

**Контекст**: на диске E два проекта:
- `E:\GPT1C_Processor_analitic` — основной (с исправлениями, не запущен в прод)
- `E:\GPT1C_Processor_analitica` — реально работал на нетопе, имел свежие логи и баги

**Проведено:**
1. Полный аудит логов `analitica` (send_reports_20260320–23, collector_20260323–24)
2. Выявлены и исправлены все баги, отсутствовавшие в `analitica`
3. Оба проекта синхронизированы и идентичны (с точностью до CRLF)

**Коммиты сессии:**

| Коммит | Проект | Описание |
|--------|--------|----------|
| `81fae8d` | analitic | run_script_async kill + timeout 300→900s + BUG-B2 |
| `2049ddf` | analitica | sync all fixes from analitic |
| `defba2d` | analitica | cosmetic: asyncio import + version date alignment |

**Исправлено в `analitica` (итог):**

| ID | Файл | Правка | Источник |
|----|------|--------|----------|
| BUG-B4 | `collector/debt_monitor.py` | `opening >= 100` вместо `> 0` | коммит 81c0640 |
| LOG-2/CRIT-1 | `bot/send_reports.py` | `_is_pid_running` fail-safe → `True` | коммит 81c0640 |
| LOG-3 | `bot/send_reports.py` | whatsapp_poller interval 10→30s | коммит 81c0640 |
| LOG-6 | `collector/whatsapp_poller.py` | httpx timeout 15→8s | коммит 81c0640 |
| LOG-7 | `collector/client_dialog.py` | `asyncio.to_thread(analyze_response)` | коммит 81c0640 |
| LOG-8 | `imap_fetcher.py` | SKIP → `logger.debug` | коммит 81c0640 |
| NEW-1 | `bot/send_reports.py` | `run_script_async` убивает subprocess при timeout | новый |
| LOG-4 | `bot/send_reports.py` | collector --send timeout 300→900s | новый |
| BUG-B2 | `collector/collections_engine.py` | проверка "менеджер занят" до `generate_message()` | новый |

**Исправлено в `analitic` (новые, не было раньше):**
- `bot/send_reports.py` — run_script_async kill + timeout 900s
- `collector/collections_engine.py` — BUG-B2 fix

---

### Сессия 2026-03-23 (аудит логов + фиксы)

Полный аудит логов `send_reports_20260309–20260319.log` + `collector_*.log`.

| Коммит | ID | Описание |
|--------|-----|----------|
| `81c0640` | BUG-B4 | `debt_monitor.py` — `opening >= 100` вместо `> 0` (артефакты округления 1C) |
| `81c0640` | MED-1 | `whatsapp_poller` — timeout 15→8с, интервал 10→30с, `analyze_response` в `asyncio.to_thread()` |
| `81c0640` | CRIT-1 | `send_reports.py` — `_is_pid_running` fail-safe: при ошибке tasklist → `True` (не `False`) |
| `81c0640` | LOW-1 | `imap_fetcher.py` — SKIP no-manager-in-name → `DEBUG` уровень (не засоряет лог) |

**Что НЕ баг:**
- 137 должников в логе 19.03 05:00 — это ДО коммита `a5d311d` (09:23 того же дня). Сейчас порог 5000₸ в коде, ожидаемое число ~50–60.
- Chat not found (Оксана/Магира/Ергали) — chat_id правильные, менеджеры не отправили `/start` боту.

---

### Сессия 2026-03-20 (sync C: ← F:external)
- `git fast-forward merge` C:/master ← F:/master
- Все 18 коммитов аудита из F: теперь в C: git
- Дополнительно применено: `run_new_reports_now.py` RNR-01, `requirements.txt` numpy, удалены старые MD файлы

---

## ВСЕ ИСПРАВЛЕННЫЕ БАГИ (полный список)

### Сессия 2026-03-25 (silence_alerts v1.7)
| ID | Файл | Fix |
|----|------|-----|
| FEAT-1 | `bot/silence_alerts.py` | v1.7: 6 категорий + IMITATION_THRESHOLD + weekly_clients |
| FEAT-2 | `bot/send_reports.py` | weekly_clients FSM (Алена→Вадим) |
| SYNC-1 | `bot/opportunity_loss.py` | MIN_DEBT 10 000→5 000 ₸ |

### Сессия 2026-03-25 (81fae8d + 2049ddf)
| ID | Файл | Fix |
|----|------|-----|
| NEW-1 | `bot/send_reports.py` | `asyncio.wait_for` → kill subprocess на timeout |
| LOG-4 | `bot/send_reports.py` | collector --send timeout 300→900s |
| BUG-B2 | `collector/collections_engine.py` | manager-busy check ПЕРЕД generate_message() |

### Сессия 2026-03-23 (81c0640)
| ID | Файл | Fix |
|----|------|-----|
| BUG-B4 | `collector/debt_monitor.py` | `opening >= 100` вместо `> 0` |
| MED-1 | `collector/whatsapp_poller.py` | timeout 15→8с |
| MED-1 | `collector/client_dialog.py` | `analyze_response` → `asyncio.to_thread()` |
| MED-1 | `bot/send_reports.py` | whatsapp_poller interval 10→30с |
| CRIT-1 | `bot/send_reports.py` | `_is_pid_running` fail-safe при ошибке |
| LOW-1 | `imap_fetcher.py` | SKIP → DEBUG уровень |

### Сессия 1 (6383aa5–7877a87)
| Коммит | Баги | Описание |
|--------|------|----------|
| `6383aa5` | BUG-C2 | `gross_report_pct._money_to_float` — запятая в regex |
| `1413f9a` | BUG-M1 | дубликат `_try_extract_meta` → импорт из `gross_report.py` |
| `27c1d35` | BUG-H1 | `opportunity_loss.py` — мёртвый ключ `"dead"` |
| `adc61bc` | BUG-H3 | `imap_fetcher.py` — `load_dotenv()` перед TZ |
| `91d0d57` | BUG-M3,M5 | `sales_parser/report.py`, `config.py` — except + TZ env |
| `651bb36` | BUG-M6,M10 | `send_reports` TZ; `silence_alerts` dead import |
| `7877a87` | BUG-M8 | хардкод "Минай" → `_SYSTEM_ACCOUNTS` из roles.json |

### Сессия 2 (237bc6e–e48ba5a)
| Коммит | Баги | Описание |
|--------|------|----------|
| `237bc6e` | BUG-H2 | `silence_alerts` — индексы колонок из `<thead>` |
| `8712810` | BUG-H4,L7 | `user_tracker.py` — `threading.Lock` + narrow except |
| `67768a7` | BUG-H5 | `debt_auto_report` — `config.HTML_DIR` вместо `getattr(OUT_DIR)` |
| `58da0fa` | BUG-M14 | `ai_analyzer.py` — 30+ `print()` → `logger` |
| `cfb5d22` | BUG-M7,L5,M9 | `send_reports` fd leak + PID print + failed counter |
| `2741ae9` | BUG-M15,M16 | `collector/` — bool env + CALL_HOUR_END из env |
| `e48ba5a` | BUG-L1,L2,L8 | `inject_local`, `txt_to_html`, `opportunity_loss` |

### Сессия 3 (5c522ac–ac234ab)
| Коммит | Баги | Описание |
|--------|------|----------|
| `5c522ac` | BUG-M2 | `gross/debt_auto` — regex `(?:покупатель\|контрагент)` |
| `e6dea07` | BUG-M17 | `inventory_summary` — regex вместо хрупкого split |
| `488f739` | BUG-L3,L4 | `inventory.py` категории из config; `inventory_cost_parser` — динамические индексы |
| `ac234ab` | ARCH-2,M11,M12 | `ai_analyzer` — убран mutable global; версии обновлены |

---

## Что намеренно оставлено (не баги)

| ID | Почему оставлено |
|----|-----------------|
| ARCH-1 | `txt_to_html` в двух местах — разные интерфейсы, унификация ломает call sites |
| ARCH-3 | `expenses_parser.py` inline HTML — изолированный модуль, работает корректно |

---

## Отложенные задачи

| Задача | Статус | Детали |
|--------|--------|--------|
| **CRM телефоны** | 🟡 в процессе | 10/день цепочкой, warn→эскалация Вадиму. Незакоммиченные изменения |
| **WhatsApp (Green API)** | 🟡 включён | WHATSAPP_ENABLED=true, токен обновлён в `15772a0` |
| **Коллектор on-stop** | 🟡 в процессе | Фильтр debit==0 AND credit==0, незакоммичен |
| **contacts_sync.py** | 🟡 создан | Excel↔JSON, незакоммичен |
| **OpenClaw** | 🔵 отложено | Polling (не webhook), без Retell AI, с Whisper. См. `openclaw/SOUL.md` |
| **Chat not found** | ⏳ организационное | Оксана/Магира/Ергали → написать `/start` боту |

---

## Ключевые правила проекта

1. **TZ**: `ZoneInfo(os.getenv("TZ", "Asia/Almaty"))` + `load_dotenv()` перед этим
2. **Exceptions**: только конкретные типы, никогда `except Exception`
3. **Queue**: `*.xlsx → *.xlsx.work → excel/processed/` через `_move_to_processed(work, src.name)`
4. **Managers**: читать из `config/managers.json` — никогда хардкод
5. **Atomic writes**: `os.replace(tmp, dst)` через `NamedTemporaryFile`
6. **Commit flow**: `py_compile` → тесты → commit → push после каждого изменения
7. **Sentinel print**: `print(f"AI saved: ...")` в `ai_analyzer.py` — НЕЛЬЗЯ трогать (subprocess парсинг)
8. **Layer 5**: standalone — обязаны сами вызвать `load_dotenv()`
9. **SESSION_CONTEXT.md**: обновлять после каждой сессии — ПРАВИЛО
10. **Платформа**: бот на нетопе, Synology — только семейные бэкапы (не трогать)

---

## Тесты и GitHub

```bash
python -X utf8 tests/test_project.py    # 62 теста
python -X utf8 tests/test_collector.py  # 104 теста

origin: https://github.com/gpt-project213/GPT1C_Processor.git
branch: master / HEAD analitic: 81fae8d / HEAD analitica: f88b398
```

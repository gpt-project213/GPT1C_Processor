# Аудит маршрутизации сообщений — GPT1C_Processor_analitica
**Дата:** 2026-04-11  
**Причина:** Production incident 2026-04-10: 22 несанкционированных WhatsApp-сообщения  
**Автор:** Claude Code (по заданию Вадима)  
**Статус:** ЗАФИКСИРОВАНО — только для чтения, не менять без согласования

---

## 1. Что проверено

| Файл | Размер | Каналы | Статус проверки |
|------|--------|--------|----------------|
| `collector/collections_engine.py` | ~760 стр. | WA + TG | ✅ Полный аудит |
| `collector/communications.py` | ~212 стр. | WA + TG | ✅ Полный аудит |
| `collector/manager_dialog.py` | ~450+ стр. | TG | ✅ Полный аудит |
| `collector/client_dialog.py` | ~500+ стр. | WA + TG | ✅ Полный аудит |
| `collector/dialog_store.py` | ~150 стр. | — (store) | ✅ Полный аудит |
| `bot/send_reports.py` | ~5600+ стр. | TG | ✅ Полный аудит |
| `bot/silence_alerts.py` | ~400+ стр. | — (parser) | ✅ Полный аудит |
| `bot/debt_stop_control.py` | ~900+ стр. | TG | ✅ Полный аудит |
| `send_tg.py` | <200 стр. | TG | ✅ Полный аудит |

**Не проверено в данном аудите:** `collector/voice_calls.py` (Retell AI, `RETELL_ENABLED=false`), `collector/collection_agent.py` (DeepSeek API, не отправляет сообщения напрямую).

---

## 2. Полная матрица сообщений

### 2A. collector/collections_engine.py

| # | Откуда (функция/строка) | Канал | Кому | Условие | Блокировка |
|---|------------------------|-------|------|---------|------------|
| E-1 | `run()` SAFEGUARD ~454 | — | — | Блокирует `--send` если `WHATSAPP_ENABLED=0` ИЛИ `LIVE_SEND_ALLOWED=0` | **Двойной замок (FIX-4 + новый)** |
| E-2 | `run()` → `notify_admin()` ~491 | TG | Admin | Нет данных дебиторки | Только если нет debt_data |
| E-3 | `run()` → `_send_tg()` ~538 | TG | Менеджер (по имени) | Нарушение кредитной политики | По одному сообщению на менеджера |
| E-4 | `run()` → `notify_admin()` ~553 | TG | Admin | Сводка нарушений кредитной политики | После E-3 |
| E-5 | `run()` → `notify_admin()` ~677 | TG | Admin | Итоговая сводка дня | Всегда при `not dry_run` |
| E-6 | `_process_single()` → `send_whatsapp()` ~392 | WA | **Клиент** | Нет manager_chat_id, есть phone | WHATSAPP_ENABLED + FIX-4 hardguard |
| E-7 | `_process_single()` → `send_telegram()` ~396 | TG | **Клиент** | Нет manager_chat_id, есть tg_id | Прямая отправка клиенту без менеджера |
| E-8 | `_process_single()` → `_start_manager_dialog()` ~424 | TG | Менеджер | manager_chat_id есть, менеджер свободен | FSM dialog запуск |
| E-9 | `_process_single()` → `notify_admin()` ~444 | TG | Admin | level >= 5 | Эскалация level 5 |
| E-10 | `_warn_manager_no_phone()` → TG | TG | Менеджер | Клиент без контактов в базе | Уведомление о пробеле в базе |
| E-11 | `check_promises()` → `notify_manager()` ~706 | TG | Менеджер | Нарушение обещания оплаты | По клиенту отдельно |

### 2B. collector/communications.py

| # | Функция | Канал | Кому | Условие | Блокировка |
|---|---------|-------|------|---------|------------|
| C-1 | `send_whatsapp()` | WA | Клиент | Любой звонящий | `WHATSAPP_ENABLED` re-read (FIX-4) |
| C-2 | `send_telegram()` | TG | Любой chat_id | Явный вызов | Только `BOT_TOKEN` |
| C-3 | `notify_admin()` | TG | `ADMIN_CHAT_ID` | Явный вызов | Только `ADMIN_CHAT_ID` |
| C-4 | `notify_manager()` | TG | Явный chat_id | Явный вызов | Только `BOT_TOKEN` |
| C-5 | `get_observer_ids()` | — | returns list | Любой manager_name | ⚠️ **Возвращает ВСЕ admins + subadmins** без скоп-фильтра |

### 2C. collector/client_dialog.py

| # | Функция | Канал | Кому | Условие | Блокировка |
|---|---------|-------|------|---------|------------|
| D-1 | `start_client_dialog()` | — | — | Только регистрирует диалог (не отправляет) | — |
| D-2 | `handle_incoming()` → `_reply_to_client()` | WA | **Клиент** | Входящий WA от клиента | Только если диалог `active` |
| D-3 | `handle_incoming()` → `escalate_to_manager()` → `_send_tg()` | TG | Менеджер | intent = promise/refusal/requires_human/limit | manager_chat_id из диалога |
| D-4 | `escalate_to_manager()` → `_send_tg()` | TG | Менеджер | Любая эскалация | manager_chat_id |

### 2D. collector/manager_dialog.py

| # | Функция | Канал | Кому | Условие | Блокировка |
|---|---------|-------|------|---------|------------|
| M-1 | `start_dialog()` → `_send_msg()` | TG | Менеджер | Новый диалог для клиента | TEST_MODE → только TEST_TG_CHAT_IDS |
| M-2 | `handle_manager_reply()` → `_send_msg()` | TG | Менеджер | Ответ на реплику менеджера | В диалоге |
| M-3 | `handle_manager_reply()` → `send_whatsapp()` | WA | **Клиент** | CONFIRMED в диалоге | WHATSAPP_ENABLED + FIX-4 |
| M-4 | `_remind_pending_dialog()` → `_send_msg()` | TG | Менеджер | Таймаут `COLLECTOR_REMINDER_HOURS=1ч` | Автоматический ремайндер |

### 2E. bot/send_reports.py

| # | Контекст | Канал | Кому | Условие |
|---|---------|-------|------|---------|
| R-1 | `send_report_to_user()` | TG | Manager/Admin/Subadmin | Новый отчёт | Роль-фильтр |
| R-2 | `handle_my_report_button()` | TG | Запрашивающий | Кнопка «Мои отчёты» | Только свои отчёты |
| R-3 | `send_debt_report()` | TG | Admin | Отчёт дебиторки | Только admin |
| R-4 | `send_silence_alerts()` | TG | Каждый менеджер | 14:00/21:00 по расписанию | По своим клиентам |
| R-5 | `send_gross_report()` | TG | Менеджеры + Admin | 20:00 | По роли |
| R-6 | `send_daily_summary()` | TG | Admin | 23:00 | Только admin |
| R-7 | Inline callback кнопки | TG | Нажавший | Интерактивные кнопки | chat_id = нажавший |
| R-8 | CRM pending phone handler | TG | Admin + менеджер | После сохранения нового клиента | ⚠️ Зависит от CRM_PHONE_PENDING логики |

### 2F. bot/debt_stop_control.py

| # | Контекст | Канал | Кому | Условие |
|---|---------|-------|------|---------|
| S-1 | `run_morning_check()` | TG | Каждый менеджер | Кандидаты на стоп | По своим клиентам |
| S-2 | `_escalate_to_admin()` | TG | Admin | Менеджер не ответил за 2ч | Эскалация |
| S-3 | `send_saida_list()` | TG | **Саида** `SAIDA_CHAT_ID` | 22:00 итоговый список | Отдельный получатель |
| S-4 | `_handle_admin_allow_after_saida()` | TG | Менеджер + Admin | Руководитель одобрил отгрузку | По событию |

---

## 3. Что получает Admin (ADMIN_CHAT_ID=7422963573)

| Источник | Когда | Что |
|---------|-------|-----|
| `collections_engine.run()` E-2 | При отсутствии debt JSON | Ошибка «нет дебиторки» |
| `collections_engine.run()` E-4 | При нарушениях кредитной политики | Сводка по всем нарушителям |
| `collections_engine.run()` E-5 | Ежедневно (09:00) | Итоговая сводка коллектора |
| `_process_single()` E-9 | level >= 5 (просрочка 30+ дней) | Персональная эскалация клиента |
| `debt_stop_control._escalate_to_admin()` S-2 | 19:00 если менеджер молчит | Запрос решения по стопу |
| `send_reports.send_debt_report()` R-3 | По запросу или расписанию | HTML отчёт дебиторки |
| `send_reports.send_daily_summary()` R-6 | 23:00 ежедневно | Дневная сводка активности |

---

## 4. Что получают Managers (Оксана, Магира, Ергали, Алена)

| Источник | Когда | Что |
|---------|-------|-----|
| `collections_engine` E-3 | При нарушении кредполитики | Список клиентов-нарушителей |
| `collections_engine` E-10 | Клиент без контактов | Предупреждение о пробеле |
| `collections_engine` E-11 | Нарушение обещания оплаты | Сообщение по каждому |
| `manager_dialog.start_dialog()` M-1 | Новый коллектор-диалог | Запрос одобрения отправки |
| `manager_dialog._remind_pending_dialog()` M-4 | Через 1ч без ответа | Напоминание |
| `client_dialog.escalate_to_manager()` D-3/D-4 | Входящий ответ клиента | Эскалация переписки |
| `send_reports.send_silence_alerts()` R-4 | 14:00 / 21:00 | Уведомления о молчунах |
| `debt_stop_control.run_morning_check()` S-1 | 17:00 | Запросы по стоп-кандидатам |

---

## 5. Что получают Subadmins (Алена как субадмин)

| Источник | Когда | Что |
|---------|-------|-----|
| `send_reports.send_report_to_user()` R-1 | Новые отчёты Магиры/Оксаны | Отчёты по курируемым менеджерам |
| `send_reports` silence alerts R-4 | 14:00 / 21:00 | Алерты по своей команде |
| `get_observer_ids()` C-5 | Любая эскалация по Магире/Оксане | ⚠️ Получает ВМЕСТЕ С admins |

---

## 6. Что получает Accountant (Саида)

| Источник | Когда | Что |
|---------|-------|-----|
| `debt_stop_control.send_saida_list()` S-3 | 22:00 ежедневно | Финальный список «не отгружать» |
| `debt_stop_control._handle_saida_confirm_*()` | После решения руководителя | Подтверждение разрешения/отказа |

**SAIDA_CHAT_ID** берётся из `os.getenv("SAIDA_CHAT_ID", "920236287")` — хардкод дефолта в коде. Если переменная не задана в `.env`, реальная Саида всё равно получает.

---

## 7. Что уходит Клиентам (WhatsApp / Telegram)

| Источник | Канал | Когда | Условие | Блокировки |
|---------|-------|-------|---------|------------|
| `_process_single()` E-6 | **WhatsApp** | --send (09:00) | Нет manager_chat_id + есть phone | FIX-1 + FIX-3 + SAFEGUARD + FIX-4 |
| `_process_single()` E-7 | Telegram | --send (09:00) | Нет manager_chat_id + есть tg_id | Только BOT_TOKEN |
| `manager_dialog` M-3 | **WhatsApp** | После CONFIRMED менеджера | Менеджер одобрил в диалоге | WHATSAPP_ENABLED + FIX-4 |
| `client_dialog.handle_incoming()` D-2 | **WhatsApp** | Входящий ответ клиента | Диалог active | Только если диалог уже существует |

---

## 8. Подтверждённые БЕЗОПАСНЫЕ маршруты

| Маршрут | Почему безопасен |
|---------|-----------------|
| `send_reports.py` → Telegram менеджерам | Только Telegram, роль-фильтр, только свои данные |
| `send_reports.py` → Telegram Admin | Только Admin chat_id, не клиентам |
| `silence_alerts.py` | Только парсер HTML, не отправляет сам — вызывает send_reports |
| `debt_stop_control` → Менеджер/Admin/Саида | Telegram, строгий роль-фильтр |
| `manager_dialog` M-1/M-2/M-4 | Только Telegram, только менеджерам, TEST_MODE блокирует |
| `client_dialog.escalate_to_manager()` D-3/D-4 | Только менеджеру, не третьим лицам |
| `collections_engine` E-6 при `dry_run=True` | Не отправляет, только логирует |

---

## 9. Подтверждённые ОПАСНЫЕ маршруты

### ОП-1 (CRITICAL — ЗАФИКСИРОВАНО 2026-04-10) — Direct send без manager lock
**Статус: ИСПРАВЛЕН FIX-2**  
**Файл:** `collector/collections_engine.py` (бывшая строка ~363)  
**Суть:** Когда менеджер был занят диалогом по клиенту A, все остальные клиенты этого менеджера отправлялись через WA напрямую без approval.  
**Исправление:** FIX-2 — ветка удалена, теперь `return result` (пропуск до следующего цикла).

---

### ОП-2 (CRITICAL — ЗАФИКСИРОВАНО 2026-04-10) — first_seen inflation
**Статус: ИСПРАВЛЕН FIX-3**  
**Файл:** `collector/collections_engine.py` строка ~594  
**Суть:** 203 клиента имели `first_seen=2026-03-19`. `real_days=22` → level 3 для клиентов с `days_1c=3` (должен быть level 0).  
**Исправление:** FIX-3 — `real_days = min(real_days, _days_1c + 7)`. Максимум +7 дней над данными 1С.

---

### ОП-3 (CRITICAL — ЗАФИКСИРОВАНО 2026-04-10) — Активные клиенты в коллекторе
**Статус: ИСПРАВЛЕН FIX-1**  
**Файл:** `collector/collections_engine.py` строка ~621-628  
**Суть:** Коммит `4f60851` удалил фильтр `if debit > 0 or credit > 0: continue`. 52 из 72 клиентов Алены были активными — все попали в очередь.  
**Исправление:** FIX-1 — фильтр восстановлен.

---

### ОП-4 (MEDIUM) — Direct send без manager_chat_id
**Статус: СУЩЕСТВУЕТ — не исправлен**  
**Файл:** `collector/collections_engine.py` строки ~388-420  
**Суть:** Если клиент найден в базе контактов, но у него нет закреплённого менеджера (`manager_chat_id = None/0`), WhatsApp/Telegram уходит напрямую без approval. Логика правильная для edge case, но не защищена от ошибок в `debtors_contacts.json` (пустое поле manager = WA без approval).  
**Риск:** Если `debtors_contacts.json` содержит записи без поля `manager` — клиент получит WA без ведома менеджера.  
**Рекомендация:** Добавить обязательную проверку наличия `manager_name` перед прямой отправкой; если manager неизвестен — требовать admin approval.

---

### ОП-5 (MEDIUM) — get_observer_ids() игнорирует скопы субадминов
**Статус: СУЩЕСТВУЕТ — не исправлен**  
**Файл:** `collector/communications.py` строки ~177-211  
**Суть:** Функция `get_observer_ids(manager_name)` добавляет **всех** `roles["admins"]` без фильтра. Если создаётся новый admin в roles.json, он автоматически получает ВСЕ уведомления по ВСЕМм менеджерам — в том числе конкурентам.  
**Риск:** Утечка данных о клиентах чужого менеджера при добавлении нового admin.  
**Рекомендация:** Для субадминов уже есть scope-фильтр (строка ~197) — применить аналогичный для admins, если нужна изоляция.

---

### ОП-6 (MEDIUM) — client_dialog.py — автоответ клиенту без свежего WA-разрешения
**Статус: СУЩЕСТВУЕТ — приемлемо при WHATSAPP_ENABLED=0**  
**Файл:** `collector/client_dialog.py` строка ~402  
**Суть:** `handle_incoming()` вызывает `_reply_to_client()` → `send_whatsapp()`. При этом `send_whatsapp()` проверяет `WHATSAPP_ENABLED` (FIX-4), но только на уровне Green API. Диалог может быть в состоянии `active` из **предыдущего** запуска до включения защиты.  
**Риск:** Если `collector_client_dialogs.json` содержит активные диалоги от 10 апреля, при включении `WHATSAPP_ENABLED=1` они могут снова заговорить, если придёт входящий.  
**Рекомендация:** Перед включением WA — проверить и закрыть все `active` диалоги в `collector_client_dialogs.json`.

---

### ОП-7 (LOW) — SAIDA_CHAT_ID хардкод в debt_stop_control.py
**Статус: СУЩЕСТВУЕТ — не критично**  
**Файл:** `bot/debt_stop_control.py` строка ~67  
**Суть:** `SAIDA_CHAT_ID = int(os.getenv("SAIDA_CHAT_ID", "920236287"))` — fallback хардкод. Если переменная не задана в `.env`, реальная Саида всё равно получает финансовые данные.  
**Рекомендация:** Убрать дефолт, заменить на `None`, добавить guard перед отправкой.

---

## 10. Где возможна ложная/чужая рассылка

| Сценарий | Файл | Условие | Вероятность |
|---------|------|---------|-------------|
| **WA клиентам без approval** | `_process_single()` E-6 | manager_chat_id пустой в contacts | СРЕДНЯЯ — зависит от качества `debtors_contacts.json` |
| **WA автоответ на старый диалог** | `client_dialog.handle_incoming()` D-2 | active диалог от 10.04 + входящее WA | НИЗКАЯ при WHATSAPP_ENABLED=0 |
| **Сводка нарушений неверному менеджеру** | E-3 | Ошибка в `managers.json` (неверный chat_id) | НИЗКАЯ |
| **get_observer_ids() лишним получателям** | C-5 | Новый admin без скопа | НИЗКАЯ при стабильной команде |
| **SAIDA без `.env`-переменной** | S-3 | `SAIDA_CHAT_ID` не задан в .env | НИЗКАЯ (только 1 получатель, не клиент) |

**Нет маршрутов к:**
- Произвольным внешним номерам
- Конкурентам
- Уволенным сотрудникам (при правильном roles.json)

---

## 11. Какие файлы нужно менять для устранения оставшихся рисков

| Приоритет | Файл | Изменение | Риск без изменения |
|----------|------|-----------|-------------------|
| HIGH | `collector/collections_engine.py` ~388 | Требовать `manager_name` перед прямой WA-отправкой | ОП-4: WA без approval для клиентов без менеджера |
| MEDIUM | `collector/client_dialog.py` | Добавить проверку активных диалогов перед включением WA | ОП-6: автоответ на старые диалоги |
| LOW | `collector/communications.py` ~205 | Опциональный scope-фильтр для admins в get_observer_ids() | ОП-5: утечка при расширении команды |
| LOW | `bot/debt_stop_control.py` ~67 | Убрать хардкод дефолта SAIDA_CHAT_ID | ОП-7: Саида получает без явной настройки |

---

## 12. Можно ли сейчас безопасно включать

### 12A. Report delivery (Telegram → менеджерам)
**СТАТУС: ✅ БЕЗОПАСНО**  
Все Telegram-отчёты (`send_reports.py`, `silence_alerts.py`) работают независимо от `WHATSAPP_ENABLED`. Можно запускать без изменений.

---

### 12B. Краткие уведомления (debt_stop_control, silence_alerts)
**СТАТУС: ✅ БЕЗОПАСНО**  
Только Telegram. Не затрагивают WhatsApp. Единственный риск — ОП-7 (Саида без .env), но это не влияет на клиентов.

---

### 12C. CRM reminders (send_reports.py → менеджеры)
**СТАТУС: ✅ БЕЗОПАСНО (с одной оговоркой)**  
Только Telegram. Однако: `CRM_PHONE_PENDING` поток — проверить, не инициирует ли он автоматически WA при подтверждении нового клиента. По результатам аудита send_reports.py — CRM flow идёт через Telegram callback, WA не затрагивает.

---

### 12D. Collector manager flow (manager_dialog.py)
**СТАТУС: ⚠️ ТРЕБУЕТ ОСТОРОЖНОСТИ**  
- Запуск диалогов (M-1..M-4) — Telegram, безопасно.  
- Но при CONFIRMED в диалоге → отправляется WA клиенту (M-3). Это невозможно без `WHATSAPP_ENABLED=1`.  
- **Условие включения:** Сначала dry-run, убедиться что список кандидатов корректен (≤5–8 клиентов, без активных плательщиков).  
- **До включения:** Закрыть все активные диалоги от 10 апреля в `collector_client_dialogs.json`.

---

### 12E. WhatsApp client send (collections_engine --send)
**СТАТУС: 🔴 ЗАБЛОКИРОВАНО — только после всех шагов ниже**

Чеклист перед включением:
- [ ] Запустить `python -m collector.collections_engine --dry-run`
- [ ] Убедиться: кандидаты ≤5–8, без active clients (debit>0 / credit>0)
- [ ] Убедиться: нет `level=3 days=22` массово
- [ ] Закрыть все active диалоги в `collector_client_dialogs.json` от 10.04
- [ ] Проверить `debtors_contacts.json` — все записи имеют поле `manager`
- [ ] Личное решение Вадима: установить `LIVE_SEND_ALLOWED=1` в `.env`
- [ ] Установить `WHATSAPP_ENABLED=1` в `.env`
- [ ] Запустить `python -m collector.collections_engine --send` и наблюдать логи

---

## Приложение А. Итог emergency fix (2026-04-11)

| Fix | Файл | Описание | Статус |
|-----|------|----------|--------|
| FIX-1 | `collections_engine.py` ~621 | Восстановлен фильтр активных клиентов (debit/credit > 0) | ✅ |
| FIX-2 | `collections_engine.py` ~363 | Удалён direct send bypass при занятом менеджере | ✅ |
| FIX-3 | `collections_engine.py` ~594 | Cap: `real_days = min(real_days, days_1c + 7)` | ✅ |
| FIX-4 | `communications.py` ~74 | Hardguard: re-read `WHATSAPP_ENABLED` при каждой отправке | ✅ |
| SAFEGUARD | `collections_engine.py` ~454 | Двойной замок: `WHATSAPP_ENABLED=1` + `LIVE_SEND_ALLOWED=1` | ✅ |
| CLI GUARD | `collections_engine.py` ~744 | `--send` блокируется при `WHATSAPP_ENABLED=0` | ✅ |
| STATE FIX | `logs/collector_state.json` | 76 клиентов с days_1c<10 — скорректирован first_seen | ✅ |
| ENV FIX | `.env` | `WHATSAPP_ENABLED=0` (не в git) | ✅ |

**Версия collections_engine.py:** 1.0.7  
**Версия communications.py:** 1.0.3  
**Коммит:** `18a8ea8` (origin/master)

---

*Аудит проведён 2026-04-11 в рамках расследования incident 2026-04-10.*  
*Следующий пересмотр — перед первым включением WHATSAPP_ENABLED=1.*

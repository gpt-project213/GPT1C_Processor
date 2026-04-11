# WEEKEND SILENT MODE + FINAL ADMIN APPROVAL — 2026-04-11

## 1. Send paths закрытые в выходной

### Уже были закрыты до этого этапа (9 функций)
| Функция | Тип | Что отправляет |
|---|---|---|
| `crm_daily_task` | daily 18:00 | CRM onboarding задачи менеджерам |
| `check_and_send_silence_alerts` | daily 14:00/21:00 | Алерты тишины по продажам |
| `debt_collector_daily` | daily 09:00 | Запуск коллектора (subprocess) |
| `debt_collector_promises` | daily 10:00 | Проверка обещаний (subprocess) |
| `send_inventory_summary` | daily 09:00 | Сводка остатков → admin |
| `send_sales_summary` | daily 21:00 | Сводка продаж → менеджеры |
| `send_gross_summary` | daily 20:00 | Валовая → admin/менеджеры |
| `crm_phone_reminder_task` | hourly 09–19 | Напоминания CRM → менеджеры |
| `collector_reminder_task` | hourly 09–18 | Напоминания collector → менеджеры |

### Закрыты в этом этапе (7 функций + 4 dstop враппера)
| Функция | Тип | Что отправляет |
|---|---|---|
| `weekly_ai_generation` | Mon 10:00 | AI-отчёты → менеджеры/admin |
| `send_daily_summary_to_admin` | daily 23:00 | Сводка активности → admin |
| `pipeline_task` (только IMAP alert) | every 10 min | Алерт ошибки IMAP → admin |
| `send_opportunity_loss_report` | Fri 14:05 | Упущенная прибыль → все |
| `new_reports_notifier` | every 10 min | Уведомления о новых отчётах |
| `weekly_analytics_job` | daily 22:00 | Аналитические отчёты → admin |
| `_job_dstop_monitor` | daily 14:00 | Debt stop мониторинг |
| `_job_dstop_managers` | daily | Debt stop → менеджеры |
| `_job_dstop_escalate` | daily | Debt stop эскалация |
| `_job_dstop_saida` | daily | Debt stop финал |

**Примечание:** `pipeline_task` — пайплайн (IMAP fetching + обработка Excel) продолжает работать в выходные для накопления данных. Только Telegram-алерт об ошибке IMAP отключён.

### Разрешено в выходной (без изменений)
- dry-run collector
- health/status команды
- логирование
- ручная диагностика без отправки
- `check_workday_task` (диагностическая — специально опрашивает admin о типе дня)

### Паттерн guard (одинаков во всех функциях)
```python
from bot.workday_checker import is_holiday_today
if is_holiday_today():
    logger.info("<function_name>: выходной — пропуск")
    return
```

---

## 2. Final Admin Approval Flow

### Полная цепочка
```
--preview
  → run_approval_preview() в collections_engine.py
  → create_batch(debtors_by_manager)
  → send_manager_previews() → TG каждому менеджеру

Менеджер нажимает:
  ✅ Разрешить всем → approved_all
  ⛔ Не отправлять никому → rejected_all
  ✏️ Выбрать вручную → manual (по каждому клиенту keep/skip/later)
  👀 Посмотреть список → детальный просмотр без изменения решения

Когда все менеджеры ответили:
  → batch.status = "pending_admin"
  → send_admin_summary() → TG директору

Директор видит:
  1. Краткая сводка (по умолчанию): итог по каждому менеджеру, кто разрешил/отклонил
  2. Кнопка 👀 Показать список подробнее → детальная таблица

Директор нажимает:
  ✅ Разрешить тестовую отправку → batch.status = "admin_approved"
  ❌ Отменить → batch.status = "cancelled"
  ⏸ Отложить → batch.admin_status = "postponed"

После admin_approved:
  → is_ready_for_send(batch_id) = True
  → get_approved_clients(batch_id) возвращает список
  → только тогда collections_engine --send может работать
```

### Что директор видит в детальном списке (👀)
Для каждого клиента из ВСЕХ менеджеров:
```
✅ Оксана
  ✅ ТОО Альфа
     Долг: 450 000 тг · Просрочка: 21 дн.
     Тел: +77011234567
  ❌ ТОО Бета (менеджер убрал)
     Долг: 120 000 тг · Просрочка: 15 дн.
     Тел: +77019999999

⛔ Магира (отклонила всех)
  ❌ ИП Гамма
     ...

К отправке: 3 из 7
```

### Жёсткие условия для разрешения отправки
| Условие | Где проверяется |
|---|---|
| `WHATSAPP_ENABLED=1` | `communications.py` FIX-1 guard |
| `LIVE_SEND_ALLOWED=1` | `collections_engine.py` FIX-2 guard |
| Все менеджеры ответили | `_all_managers_responded()` в `approval_flow.py` |
| Директор нажал «Разрешить» | `is_ready_for_send()` → `batch.status == "admin_approved"` |

Если хотя бы одно условие не выполнено → send заблокирован.

---

## 3. Проверки пройдены

| Проверка | Результат |
|---|---|
| `py_compile send_reports.py` | ✅ OK |
| `py_compile approval_flow.py` | ✅ OK |
| 11 holiday guards в send_reports.py | ✅ все на месте |
| `_format_admin_detail_text` в approval_flow | ✅ есть |
| Телефон в детальном виде admin | ✅ есть |
| Кнопка «Разрешить тестовую отправку» | ✅ есть |
| Кнопка «Показать список подробнее» | ✅ есть |
| Тесты 133/133 | ✅ все пройдены |

---

## 4. Что требуется перед первым live test

### Порядок действий
1. **Убедиться в рабочем дне** — не суббота, не воскресенье, не праздник
2. **Выставить в `.env`:**
   ```
   WHATSAPP_ENABLED=1
   LIVE_SEND_ALLOWED=1
   ```
3. **Dry-run** (без отправки, проверка списка):
   ```bash
   python -m collector.collections_engine --dry-run
   ```
4. **Preview** (рассылка менеджерам для согласования):
   ```bash
   python -m collector.collections_engine --preview
   ```
5. **Дождаться ответов менеджеров** — каждый нажимает ✅/⛔/✏️ в Telegram
6. **Директор получает итог** — проверяет детальный список с телефонами
7. **Директор нажимает «✅ Разрешить тестовую отправку»**
8. **Test send на 1–2 клиента:**
   ```bash
   python -m collector.collections_engine --send --client "ТОО Название"
   ```
9. Проверить что сообщение пришло нужному клиенту, нет дублей, нет ошибок в логах

### Что НЕ делать
- Не запускать `--send` без `--preview` и admin approve
- Не запускать в субботу/воскресенье/праздник
- Не запускать до 09:00 или после 18:00
- Не выставлять `LIVE_SEND_ALLOWED=1` постоянно — только на время теста

---

## Изменённые файлы

| Файл | Версия | Изменение |
|---|---|---|
| `bot/send_reports.py` | v9.4.42 | +7 is_holiday guards в scheduled tasks |
| `collector/approval_flow.py` | v1.0.1 | +`_format_admin_detail_text`, новые кнопки |
| `tests/test_collector.py` | — | Обновлён FIX-2 тест под OP-4 guard |

Коммит: `fix: weekend silent mode + enhanced final admin approval`

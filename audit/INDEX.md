# AUDIT INDEX — GPT1C_Processor_analitica

Единый указатель всех аудитных и архитектурных документов проекта.
Актуально на: **2026-04-11**

---

## Быстрый статус системы

| Подсистема | Статус | Комментарий |
|---|---|---|
| Pipeline (IMAP → Excel → HTML/JSON) | ✅ Стабильно | После аудита 2026-03-10/19/20 |
| Collector dry-run | ✅ Стабильно | Проверен 2026-04-11 |
| Collector --send (WhatsApp) | ⛔ ОТКЛЮЧЁН | INCIDENT 2026-04-10, ждёт approval UX |
| WA Approval UX (менеджер → admin) | ✅ Реализован | `collector/approval_flow.py` v1.0.0 |
| Telegram бот (отчёты, silence, CRM) | ✅ Работает | |
| CRM task creation (18:00) | ✅ Условно безопасно | |
| CRM reminders (каждый час) | ⚠️ Работает с оговоркой | BUG-CRM-1: нет holiday guard |
| Voice calls (Retell AI) | ⛔ Не включён | `RETELL_ENABLED=0` |

---

## Список документов

### Инциденты

| Файл | Дата | Тема | Статус |
|---|---|---|---|
| [INCIDENT_REPORT_WHATSAPP_2026-04-10.md](INCIDENT_REPORT_WHATSAPP_2026-04-10.md) | 2026-04-10 | Несанкционированные WA рассылки — 22 клиента | ✅ Причины найдены, 4 фикса применены |

---

### Аудиты кода

| Файл | Дата | Тема | Статус |
|---|---|---|---|
| [AUDIT_FINDINGS.md](AUDIT_FINDINGS.md) | 2026-03-20 | Первичный аудит — 28 найденных багов | ✅ Все CRITICAL/HIGH закрыты |
| [AUDIT_SUMMARY.md](AUDIT_SUMMARY.md) | 2026-03-20 | Итоговая сводка по аудиту | ✅ Актуально |
| [AUDIT_VERDICT.md](AUDIT_VERDICT.md) | 2026-03-20 | Вердикт: 0 critical / 0 high / 2 arch | ✅ Актуально |
| [AUDIT_FULL_PROJECT_2026-04-09.md](AUDIT_FULL_PROJECT_2026-04-09.md) | 2026-04-09 | Полный аудит проекта: расписание, роли, TZ | ✅ Актуально |
| [CRM_FULL_AUDIT_2026-04-11.md](CRM_FULL_AUDIT_2026-04-11.md) | 2026-04-11 | Полный аудит CRM: 13 секций + ответы на 10 вопросов | ✅ Актуально |

---

### Маршрутизация сообщений

| Файл | Дата | Тема | Статус |
|---|---|---|---|
| [MESSAGE_ROUTING_AUDIT_2026-04-11.md](MESSAGE_ROUTING_AUDIT_2026-04-11.md) | 2026-04-11 | Матрица всех каналов отправки: 5 подсистем, 7 рисков (OP-1..OP-7) | ✅ OP-1..OP-3 исправлены |

---

### UX и спецификации

| Файл | Дата | Тема | Статус |
|---|---|---|---|
| [WHATSAPP_APPROVAL_UX_2026-04-11.md](WHATSAPP_APPROVAL_UX_2026-04-11.md) | 2026-04-11 | UX согласования WA рассылки (менеджер → admin) | ✅ Реализован |

---

### Архитектура и модели

| Файл | Дата | Тема |
|---|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 2026-04-09 | Схема 9 слоёв + зависимости |
| [BOOTSTRAP_FLOW.md](BOOTSTRAP_FLOW.md) | 2026-04-09 | Порядок инициализации бота |
| [STATE_MODEL.md](STATE_MODEL.md) | 2026-04-09 | Модель состояний: collector, CRM, dialogs |
| [AUDIT_PROJECT_MAP.md](AUDIT_PROJECT_MAP.md) | 2026-03-20 | Карта проекта — файлы, слои, связи |
| [AUDIT_RUNTIME_TRACE.md](AUDIT_RUNTIME_TRACE.md) | 2026-03-20 | Runtime трассировка: какой код реально вызывается |

---

### Прочее

| Файл | Дата | Тема |
|---|---|---|
| [SESSION_CONTEXT.md](SESSION_CONTEXT.md) | 2026-04-09 | История сессий и изменений |
| [CLAUDE.md](CLAUDE.md) | — | Копия корневого CLAUDE.md (для контекста в папке audit) |
| [AUDIT_FINDINGS.json](AUDIT_FINDINGS.json) | 2026-03-20 | Machine-readable данные аудита |
| logs/ | — | Runtime логи send_reports за 03, 08, 09 апреля |

---

## Открытые баги (зафиксированы, не исправлены)

| ID | Файл | Описание | Приоритет |
|---|---|---|---|
| BUG-CRM-1 | `bot/send_reports.py:6264` | `crm_phone_reminder_task` не проверяет `is_holiday_today()` — рассылает напоминания в выходные | LOW |
| OP-4 | `collector/collections_engine.py` | Нет обязательной проверки `manager_name` перед прямой WA-отправкой в `_process_single()` | LOW |
| ARCH-1 | `tools/txt_to_html.py` + `bot/send_reports.py` | `txt_to_html` в двух местах с разными интерфейсами | ARCH |
| ARCH-3 | `expenses_parser.py` | Inline HTML в парсере | ARCH |
| BSR-01 | `bot/send_reports.py:388` | `logging.Formatter.formatTime` monkey-patch — глобальный | LOW |

**Исправление BUG-CRM-1** (3 строки в `crm_phone_reminder_task`):
```python
from bot.workday_checker import is_holiday_today
if is_holiday_today():
    return
```

---

## Условия включения WhatsApp (--send)

1. ✅ `WHATSAPP_ENABLED=1` в `.env`
2. ✅ `LIVE_SEND_ALLOWED=1` в `.env`
3. ✅ Прогнать `--dry-run`, проверить вывод
4. ✅ Запустить `--preview` → менеджеры подтверждают → admin даёт финальное OK
5. ✅ Только после admin approve выполнить `--send`
6. ⚠️ Никогда в выходные, никогда вне 09:00–18:00 Asia/Almaty

---

## История изменений этого индекса

| Дата | Что изменено |
|---|---|
| 2026-04-11 | Создан первичный INDEX.md |

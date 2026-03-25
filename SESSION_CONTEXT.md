# SESSION_CONTEXT.md
> Автоматически обновляется Claude Code. Последнее обновление: 2026-03-25

## Текущее состояние проекта

**Ветка**: `master`
**Последний коммит analitic**: `81fae8d` (2026-03-25)
**Последний коммит analitica**: `f88b398` (2026-03-25)
**Статус тестов**: 62/62 + 104/104 collector ✅
**Открытые баги**: 0 критических / 0 высоких / 2 архитектурных (не критично)
**Платформа**: основной ПК — `E:\GPT1C_Processor_analitica` запущен и тестируется (2026-03-25)

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
| **Тестирование analitica** | 🟢 в процессе | Запущен на нетопе 2026-03-25, мониторить логи |
| **VS Code авторизация** | 🟡 в процессе | Extension установлен, credentials валидны, непонятно почему не подключается |
| **OpenClaw** | 🔵 отложено | Polling (не webhook), без Retell AI, с Whisper. См. `openclaw/SOUL.md` |
| **Chat not found** | ⏳ организационное | Оксана/Магира/Ергали → написать `/start` боту |
| **debtors_contacts.json** | ⏳ организационное | Заполнить телефоны/telegram должников (сейчас 0 из 83 заполнено) |

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

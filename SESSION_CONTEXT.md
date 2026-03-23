# SESSION_CONTEXT.md
> Автоматически обновляется Claude Code. Последнее обновление: 2026-03-23

## Текущее состояние проекта

**Ветка**: `master`
**Последний коммит**: `81c0640` (2026-03-23)
**Статус тестов**: 62/62 + все collector тесты ✅
**Открытые баги**: 0 критических / 0 высоких / 2 архитектурных (не критично)
**Платформа**: нетоп (мини-ПК), проект `E:\GPT1C_Processor_analitic`, GitHub как хранилище

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
- `git fast-forward merge` C:/master ← F:/master (`bb80464` → `0333f30`)
- Все 18 коммитов аудита из F: теперь в C: git
- Дополнительно применено: `run_new_reports_now.py` RNR-01, `requirements.txt` numpy, удалены старые MD файлы

---

## ВСЕ ИСПРАВЛЕННЫЕ БАГИ (полный список)

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
| **OpenClaw** | 🔵 отложено | Polling (не webhook), без Retell AI, с Whisper. См. `openclaw/SOUL.md` |
| **Chat not found** | ⏳ организационное | Оксана/Магира/Ергали → написать `/start` боту |
| **debtors_contacts.json** | ⏳ организационное | Заполнить телефоны/telegram должников (сейчас 0 из ~59 обработано) |

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
python -X utf8 tests/test_collector.py  # все collector тесты

origin: https://github.com/gpt-project213/GPT1C_Processor.git
branch: master / HEAD: 81c0640
```

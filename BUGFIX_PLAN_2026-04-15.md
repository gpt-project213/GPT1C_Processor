# ПЛАН РАБОТЫ НАД БАГАМИ — GPT1C_Processor
**Дата составления:** 2026-04-15  
**Источник:** Аудит логов send_reports_20260413-14, collector_20260413-14, run_pipeline_all_mp, всех .md и коммитов  
**Статус:** Рабочий документ для следующих сессий

---

## ✅ СДЕЛАНО В ЭТОЙ СЕССИИ (2026-04-15)

| Действие | Файл | Детали |
|----------|------|--------|
| Удалён тестовый номер +77001234567 | `config/clients.json` | 6 клиентов очищены, `_needs_phone=true` |
| Удалён тестовый номер +77001234567 | `config/debtors_contacts.json` | 1 запись очищена, `_needs_phone=true` |
| Аудит всех логов и .md | — | Результат в этом файле и `аудит/` |

---

## 🔴 ПРИОРИТЕТ 1 — СДЕЛАТЬ ПЕРВЫМ (критично, production сломан)

### 1.1 Закоммитить фикс InlineKeyboardMarkup
**Файл:** `bot/send_reports.py`  
**Что:** Удалена 1 строка `from telegram import InlineKeyboardMarkup, InlineKeyboardButton` внутри `cb_data()` (строка ~6125). Лежит в working tree незакоммиченной.  
**Почему срочно:** До коммита payhold-кнопки (одобрение оплаты Саидой) падают с UnboundLocalError при каждом нажатии.  
**Действие:**
```bash
python -m py_compile bot/send_reports.py
git add bot/send_reports.py
git commit -m "fix(bot): remove local InlineKeyboardMarkup import causing UnboundLocalError in cb_data"
```

### 1.2 Включить WhatsApp
**Файл:** `.env`  
**Что:** Изменить `WHATSAPP_ENABLED=0` → `WHATSAPP_ENABLED=1`  
**Почему срочно:** WhatsApp отключён. Коллектор работает, батчи создаются, но сообщения не уходят клиентам.  
**Внимание:** Перед включением убедиться что у 6 клиентов (Алена + Оксана) введены реальные телефоны в `config/clients.json` и `config/debtors_contacts.json` — иначе следующий запуск коллектора пропустит их (будет предупреждение, не ошибка).

---

## 🟠 ПРИОРИТЕТ 2 — СДЕЛАТЬ В БЛИЖАЙШИЕ ДНИ

### 2.1 Исправить AssemblyAI: speech_models → speech_model
**Файл:** `collector/whatsapp_poller.py:162`  
**Симптом:** Все голосовые сообщения от должников возвращают ошибку 400. Транскрипция не работает. Fallback на Whisper тоже не работает (429, исчерпан quota OpenAI).  
**Причина:** AssemblyAI v2 API принимает строку `speech_model`, а код шлёт список `speech_models`.  
**Фикс:**
```python
# БЫЛО:
_ASSEMBLYAI_SPEECH_MODELS = ["universal-3-pro", "universal-2"]
# в запросе:
"speech_models": _ASSEMBLYAI_SPEECH_MODELS,

# СТАЛО:
"speech_model": "universal-2",   # строка, singular, для поддержки ru/kk
```
**Проверить:** Документацию AssemblyAI v2 на актуальные названия моделей.

### 2.2 Исправить callback_data overflow для длинных имён клиентов
**Файл:** `bot/send_reports.py:3800, 6129, 6130`  
**Симптом:** `Button_data_invalid` при клиентах с длинными кирилличными именами (>~22 символов). Подтверждено на "А Рынок Шарын Жулдыз Рахманбердиева".  
**Причина:** Telegram ограничивает callback_data ≤ 64 байт. Кирилла = 2 байта/символ. `weekly_suggest|{client_name}` легко выходит за лимит.  
**Фикс:** Хранить имена в словаре `_weekly_pending: dict[str, str]` (token → name), в callback_data писать только токен:
```python
token = _weekly_pending_add(client_name)   # uuid4()[:8]
callback_data=f"weekly_suggest|{token}"
# при обработке:
client_name = _weekly_pending.get(data.split("|",1)[1], "")
```
**Также проверить:** `weekly_confirm|{name}` и `weekly_deny|{name}` — та же проблема.

### 2.3 Ввести реальные телефоны для 6 клиентов без номеров
**Файлы:** `config/clients.json`, `config/debtors_contacts.json`  
**Клиенты (все `_needs_phone=true` после чистки 15.04):**
- А Айс Ленд ТОО (Алена)
- А ИП Барменова Ф.А.(Бразис Акколь) (Алена)
- А ИП Кихтенко (Алена)
- А Ресторан Lugano ул Досмухамедулы 38 (Алена)
- А ТД Алем (холодильник № 6) (Алена)
- О ТД Артем.мясной зал.Кус Вкус тел 87023069994 (Оксана) ← телефон в названии!
**Действие:** Попросить менеджеров предоставить WhatsApp-номера, ввести через бот или напрямую в JSON.

### 2.4 Эскалация без resolve: 6 клиентов без телефонов
**Симптом:** warn_count растёт каждый запуск (2, 3, 4...), система шлёт одни и те же уведомления менеджерам.  
**Клиенты (из лога 14.04):**
- Е ТД Саянур Леонид (warn_count=2)
- М Ресторан Шама ИП Тян ул Мустафина 12 (warn_count=2)
- Е ТОО MEAT THE FAMILY (Казбек) (warn_count=4) — самый старый
- Е ИП Трое Кайрат (warn_count=3)
- Е ТОО Берекет 2025 ул Бейсекбаева 32 (warn_count=3)
- М ТРЦ Керуен Сити ТОО Sanam Foods (warn_count=3)
**Действие:** Ввести реальные телефоны (разных от группы 2.3) или явно пометить `"do_not_call": true` если контакт невозможен.

---

## 🟡 ПРИОРИТЕТ 3 — ТЕХНИЧЕСКИЙ ДОЛГ (планово)

### 3.1 File lock для JSON-хранилищ коллектора
**Файлы:** `wa_approval_batches.json`, `collector_state.json`, `collector_dialogs.json` и ещё 5 файлов  
**Риск:** Бот (async) + subprocess коллектора пишут одновременно → lost update. Опаснее всего `wa_approval_batches.json` (одновременное нажатие кнопок двумя менеджерами).  
**Фикс:** `filelock` библиотека или `fcntl.flock` (но Windows → `msvcrt.locking`). Проще — `portalocker`.

### 3.2 Retry для AI API (ai_analyzer.py)
**Файл:** `ai_analyzer.py:287-309`  
**Риск:** Один вызов без retry. При временной ошибке API — потеря AI-комментария к отчёту.  
**Фикс:** 3 попытки с экспоненциальным backoff (как уже сделано в approval_flow.py для TG).

### 3.3 Тесты загрязняют production-лог
**Файл:** `tests/test_project.py:594`, `run_pipeline_all_mp.py:486`  
**Симптом:** 23 ложных ERROR в `run_pipeline_all_mp.log` за 2 дня от теста с `nonexistent` путём.  
**Фикс:** В `_write_collector_trigger` использовать `logger` вместо `_log()`, или в тесте перехватывать через `monkeypatch`.

### 3.4 Мёртвый код в collector/ (4 блока)
| ID | Файл | Строки | Описание |
|----|------|--------|----------|
| DEAD-2 | `collections_engine.py:1267` | ~12 строк | `run(dry_run=False)` недостижим |
| DEAD-3 | `manager_dialog.py:~1116` | 1 строка | `col_update_` callback — ни одна кнопка не генерирует |
| DEAD-4 | `manager_dialog.py:839` | ~25 строк | `_send_control_reminder` — нигде не вызывается |
| DEAD-5 | `collections_engine.py:93,101` | 2 строки | Импорты `analyze_response`, `get_call_result` |

### 3.5 IMAP: неатомарная запись xlsx
**Файл:** `imap_fetcher.py:279-282`  
**Риск:** При kill/OOM во время сохранения — неполный .xlsx в queue, pipeline падает на нём.  
**Фикс:** `write_to_tmp + os.replace` по аналогии с JSON-хранилищами.

### 3.6 sales_parser.py не запускается из pipeline
**Файл:** `run_pipeline_all_mp.py`  
**Риск:** После обработки новых sales.xlsx — JSON не обновляется пока не запустить `send_reports.py`. Аналитика (RFM, concentration, turnover) работает на устаревших данных.

---

## 🔵 UX / ДОКУМЕНТАЦИЯ

### 4.1 Мёртвый запас: несоответствие UI и логики
**Файл:** `inventory_turnover_report.py`  
**Текст в UI:** "товары без продаж >30 дней"  
**Факт:** "товары, которых нет в sales_*.json за выбранный период"  
**Фикс:** Изменить заголовок в HTML на точное описание.

### 4.2 Концентрация: KPI не Парето 80/20
**Файл:** `revenue_concentration_report.py`  
**Текст:** "Правило Парето (80/20)"  
**Факт:** KPI = доля топ-5 клиентов, не 80% выручки  
**Фикс:** Изменить подпись на "Топ-5 клиентов по выручке".

### 4.3 Два файла контактов дебиторов
- `config/debtors_contacts.json` — рабочий (используется)
- `collector/debtors_contacts.json` — legacy (неизвестно актуален ли)
**Действие:** Сравнить, объединить или удалить legacy.

---

## 📊 СВОДКА

| Приоритет | Кол-во задач | Статус |
|-----------|-------------|--------|
| 🔴 Критично (P1) | 2 | Не сделано |
| 🟠 Высокий (P2) | 4 | Не сделано |
| 🟡 Технический долг (P3) | 6 | Не сделано |
| 🔵 UX/Docs (P4) | 3 | Не сделано |

**Тестовый телефон +77001234567:** ✅ Удалён из 7 записей (6 clients.json + 1 debtors_contacts.json)

---

*Следующая сессия: начать с P1.1 (git commit send_reports.py) → P1.2 (включить WhatsApp) → P2.1 (AssemblyAI)*

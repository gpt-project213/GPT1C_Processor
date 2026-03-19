# SESSION_CONTEXT.md
> Автоматически обновляется Claude Code. Последнее обновление: 2026-03-19 (сессия 3)

## Цель этого файла
Позволяет Claude Code в новой сессии мгновенно восстановить контекст без перечитывания всего проекта.

---

## Текущее состояние проекта

**Ветка**: `master`
**Последний коммит**: `5c522ac` (2026-03-19)
**Статус тестов**: 62/62 + 58/58 ✅

---

## Все исправленные баги по сессиям

### Сессия 1 (коммиты 6383aa5–7877a87, 2026-03-19)
| Коммит | Баги | Файлы |
|--------|------|-------|
| `6383aa5` | BUG-C2 | `gross_report_pct._money_to_float` — запятая в regex |
| `1413f9a` | BUG-M1 | `gross_report_pct.py` дубликат `_try_extract_meta` → импорт |
| `27c1d35` | BUG-H1 | `opportunity_loss.py` — мёртвый ключ `"dead"` |
| `adc61bc` | BUG-H3 | `imap_fetcher.py` — `load_dotenv()` перед TZ |
| `91d0d57` | BUG-M3,M5 | `sales_parser.py`, `sales_report.py`, `config.py` |
| `651bb36` | BUG-M6,M10 | `send_reports._formatTime_almaty`; `silence_alerts` dead import |
| `7877a87` | BUG-M8 | `send_reports.get_managers_list()` хардкод "Минай" → `_SYSTEM_ACCOUNTS` |

### Сессия 2 (коммиты 237bc6e–e48ba5a, 2026-03-19)
| Коммит | Баги | Файлы |
|--------|------|-------|
| `237bc6e` | BUG-H2 | `silence_alerts.parse_html_silence_days()` — индексы из `<thead>` |
| `8712810` | BUG-H4,L7 | `user_tracker.py` — `threading.Lock` + narrow except |
| `67768a7` | BUG-H5 | `debt_auto_report.py` — `config.HTML_DIR` вместо `getattr(OUT_DIR)` |
| `58da0fa` | BUG-M14 | `ai_analyzer.py` — 30+ `print()` → `logger` |
| `cfb5d22` | BUG-M7,L5,M9 | `send_reports.py` fd leak + PID + failed counter |
| `2741ae9` | BUG-M15,M16 | `collector/` — bool env + `CALL_HOUR_END` из env |
| `e48ba5a` | BUG-L1,L2,L8 | `inject_local`, `txt_to_html`, `opportunity_loss` |
| `18aea5f` | — | `SESSION_CONTEXT.md` добавлен в корень |

### Сессия 3 (начата 2026-03-19, коммит 5c522ac)
| Коммит | Баги | Файлы |
|--------|------|-------|
| `5c522ac` | BUG-M2 | `gross_report.py` + `debt_auto_report.py` — regex `(?:покупатель\|контрагент)` |

---

## Оставшиеся баги (не исправлены)

| ID | Файл | Описание | Приоритет |
|----|------|----------|-----------|
| BUG-M17 | `bot/inventory_summary.py` | Хрупкий парсинг `<small>` тега — split('\n'), split('Период:') | M |
| BUG-L3 | `inventory.py` | `MAIN_CATEGORIES` захардкоден | L |
| BUG-L4 | `inventory_cost_parser.py` | Хардкод индексов колонок | L |
| BUG-M11 | `analyze_debt_excel.py` | Устаревшая версия `"v2.1 — 2025-09-02"` | L |
| BUG-M12 | `debt_auto_report.py` | Устаревшая версия строки | L |
| BUG-M13 | `bot/send_reports.py` | Несогласованность версий | L |
| ARCH-1 | — | Дублирование `txt_to_html` в двух местах | L |
| ARCH-2 | `ai_analyzer.py` | Mutable `global AI_TG_SEND_HTML` | L |
| ARCH-3 | `expenses_parser.py` | Inline HTML вместо Jinja2 | L |

---

## BUG-M17 — исследование (НЕ ИСПРАВЛЕН, начат в сессии 3)

`bot/inventory_summary.py:parse_inventory_html()` — строки 101-116:
```python
small_tag = soup.find('small')
if small_tag:
    text = small_tag.get_text()
    parts = text.split('Период:')
    ...
    parts = text.lower().split('количество:')
```

**Реальная структура `<small>` в `inventory_*.html`** (проверено на живом файле):
```
Период: 17 марта 2026 г.
Сформировано: 19.03.2026 19:29
Всего количество: 49 561.462
```

**Проблемы:**
1. `split('количество:')` никогда не сработает — в HTML `Всего количество:`, а не просто `количество:`
2. `split('\n')` хрупко — при изменении шаблона пробелы могут измениться

**Предложенный фикс (не применён):**
- Использовать regex по каждому полю: `re.search(r'Период:\s*(.+)', text)`
- Для количества: `re.search(r'[Вв]сего\s+количество:\s*([\d\s,.]+)', text)`

---

## Ключевые правила проекта

1. **TZ**: всегда `ZoneInfo(os.getenv("TZ", "Asia/Almaty"))` + `load_dotenv()` перед этим
2. **Exceptions**: никогда `except Exception` — только конкретные типы
3. **Queue lifecycle**: `*.xlsx → *.xlsx.work → excel/processed/` через `_move_to_processed(work, src.name)`
4. **Manager names**: читать из `config/managers.json` — никогда хардкод
5. **Atomic writes**: `os.replace(tmp, dst)` через `NamedTemporaryFile`
6. **Commit flow**: после каждого изменения — `py_compile` → тесты → commit → push
7. **Sentinel print**: в `ai_analyzer.py` строка `print(f"AI saved: ...")` — НЕЛЬЗЯ трогать
8. **Layer 5**: standalone файлы — обязаны сами вызвать `load_dotenv()` до чтения env
9. **SESSION_CONTEXT.md**: обновлять после каждой сессии — ПРАВИЛО пользователя

---

## Структура проекта (кратко)

```
Layer 0: utils_common.py, send_tg.py, cleanup_cache.py, tools/
Layer 1: config.py
Layer 2: utils_excel.py, utils.py
Layer 3: *_parser.py  (святой грааль — трогать осторожно!)
Layer 4: debt/sales/gross/inventory/expenses_report.py
Layer 5: dso/rfm/concentration/turnover/net_profit/profitability_report.py
Layer 6: ai_analyzer.py
Layer 7: imap_fetcher.py, run_pipeline*.py
Layer 8: bot/send_reports.py + bot/*.py
```

## Тесты

```bash
python -X utf8 tests/test_project.py    # 62 теста
python -X utf8 tests/test_parsers.py    # 58 тестов
```

## GitHub

```
origin: https://github.com/gpt-project213/GPT1C_Processor.git
branch: master
HEAD: 5c522ac
```

## Структура директорий отчётов

```
reports/queue/        ← входящие xlsx
reports/html/         ← HTML отчёты
reports/json/         ← JSON данные
reports/analytics/    ← Layer 5 analytics
reports/ai/           ← AI txt/html
reports/excel/active/ ← в обработке
reports/excel/processed/ ← завершённые
```

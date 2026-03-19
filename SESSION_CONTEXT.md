# SESSION_CONTEXT.md
> Автоматически обновляется Claude Code. Последнее обновление: 2026-03-19

## Цель этого файла
Позволяет Claude Code в новой сессии мгновенно восстановить контекст без перечитывания всего проекта.

---

## Текущее состояние проекта

**Ветка**: `master`
**Последний коммит**: `e48ba5a` (2026-03-19)
**Статус тестов**: 62/62 + 58/58 ✅
**Открытые баги**: см. раздел "Оставшиеся баги"

---

## История аудита и фиксов (текущая сессия)

### Предыдущие сессии
| Коммит | Что исправлено |
|--------|---------------|
| `6383aa5` | BUG-C2: `gross_report_pct._money_to_float` — запятая в regex |
| `1413f9a` | BUG-M1: дубликат `_try_extract_meta` в `gross_report_pct.py` → импорт из `gross_report.py` |
| `27c1d35` | BUG-H1: `opportunity_loss.py` — мёртвый ключ `"dead"` в `zones` |
| `adc61bc` | BUG-H3: `imap_fetcher.py` — `load_dotenv()` перед TZ |
| `91d0d57` | BUG-M3/M5: `sales_parser.py`, `sales_report.py`, `config.py` — narrow except + TZ env |
| `651bb36` | BUG-M6/M10: `send_reports._formatTime_almaty` TZ + `silence_alerts` мёртвый import |
| `7877a87` | BUG-M8: `get_managers_list()` хардкод "Минай" → `_SYSTEM_ACCOUNTS` |

### Текущая сессия
| Коммит | Что исправлено | Баги |
|--------|---------------|------|
| `237bc6e` | `silence_alerts.parse_html_silence_days()` — индексы колонок из `<thead>` вместо хардкода | BUG-H2 |
| `8712810` | `user_tracker.py` — `threading.Lock` для атомарного R-M-W + narrow except | BUG-H4, BUG-L7 |
| `67768a7` | `debt_auto_report.py` — `config.HTML_DIR` вместо `getattr(OUT_DIR)` | BUG-H5 |
| `58da0fa` | `ai_analyzer.py` — 30+ `print()` → `logger` (кроме сентинела "AI saved:") | BUG-M14 |
| `cfb5d22` | `send_reports.py` fd leak StreamHandler + PID `print→logger`; pipeline `failed` counter | BUG-M7, BUG-L5, BUG-M9 |
| `2741ae9` | `collector/` — bool env + `CALL_HOUR_END` из env | BUG-M15, BUG-M16 |
| `e48ba5a` | `opportunity_loss`, `inject_local`, `txt_to_html` — except + load_dotenv + TZ env | BUG-L8, BUG-L1, BUG-L2 |

---

## Оставшиеся баги

| ID | Файл | Описание | Приоритет |
|----|------|----------|-----------|
| BUG-M2 | `gross_report.py` vs `debt_auto_report.py` | Несогласованный regex фильтра 1С (`покупатель` vs `контрагент`) | M |
| BUG-M11 | `analyze_debt_excel.py` | Устаревшая версия `"v2.1 — 2025-09-02"` | L |
| BUG-M12 | `debt_auto_report.py` | Устаревшая версия строки | L |
| BUG-M13 | `bot/send_reports.py` | Несогласованность версии | L |
| BUG-M17 | `bot/inventory_summary.py` | Хрупкий парсинг тега `<small>` | M |
| BUG-L3 | `inventory.py` | `MAIN_CATEGORIES` захардкоден | L |
| BUG-L4 | `inventory_cost_parser.py` | Хардкод индексов колонок | L |
| ARCH-1 | `txt_to_html` | Дублирование в двух местах | L |
| ARCH-2 | `ai_analyzer.py` | Mutable `global AI_TG_SEND_HTML` | L |
| ARCH-3 | `expenses_parser.py` | Inline HTML вместо Jinja2 | L |

---

## Ключевые правила проекта

1. **TZ**: всегда `ZoneInfo(os.getenv("TZ", "Asia/Almaty"))` + `load_dotenv()` перед этим
2. **Exceptions**: никогда `except Exception` — только конкретные типы
3. **Queue lifecycle**: `*.xlsx → *.xlsx.work → excel/processed/` через `_move_to_processed(work, src.name)`
4. **Manager names**: читать из `config/managers.json` — никогда хардкод
5. **Atomic writes**: `os.replace(tmp, dst)` через `NamedTemporaryFile`
6. **Commit flow**: после каждого изменения — `py_compile` → тесты → commit → push
7. **Sentinel print**: в `ai_analyzer.py` строка `print(f"AI saved: ...")` — НЕЛЬЗЯ трогать (парсится subprocess'ом)
8. **Layer 5**: standalone файлы — обязаны сами вызвать `load_dotenv()` до чтения env

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

## Тесты

```bash
python -X utf8 tests/test_project.py    # 62 теста
python -X utf8 tests/test_parsers.py    # 58 тестов
```

## GitHub

```
origin: https://github.com/gpt-project213/GPT1C_Processor.git
branch: master
```

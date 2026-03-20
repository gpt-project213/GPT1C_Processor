# SESSION_CONTEXT.md
> Автоматически обновляется Claude Code. Последнее обновление: 2026-03-20 (sync C: ← F:)

## Текущее состояние проекта

**Ветка**: `master`
**Последний коммит**: pending (не закоммичено — см. ниже)
**Статус тестов**: 62/62 + 58/58 ✅ (на F: / ac234ab)
**Открытые баги**: 0 критических / 0 высоких / 2 архитектурных (не критично)
**Платформа**: Synology DS224+ (18GB RAM, 20TB, UPS, Docker)

### Сессия 2026-03-20 (sync C: ← F:external)
- `git fast-forward merge` C:/master ← F:/master (`bb80464` → `0333f30`)
- Все 18 коммитов аудита из F: теперь в C: git
- Дополнительно применено: `run_new_reports_now.py` RNR-01, `requirements.txt` numpy, удалены старые MD файлы
- `CLAUDE.md` обновлён — полная история, SESSION_CONTEXT.md как источник истины

---

## ВСЕ ИСПРАВЛЕННЫЕ БАГИ (полный список)

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

## Что намеренно оставлено (не баги, а архитектурные решения)

| ID | Почему оставлено |
|----|-----------------|
| ARCH-1 | `txt_to_html` в двух местах — разные интерфейсы, унификация ломает call sites без выгоды |
| ARCH-3 | `expenses_parser.py` inline HTML — изолированный модуль, работает корректно |
| BUG-M13 | Версия `send_reports.py` — косметика, не влияет на работу |

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

---

## Тесты и GitHub

```bash
python -X utf8 tests/test_project.py    # 62 теста
python -X utf8 tests/test_parsers.py    # 58 тестов

origin: https://github.com/gpt-project213/GPT1C_Processor.git
branch: master / HEAD: ac234ab
```

## Деплой (Synology DS224+)

```
/volume1/docker/gpt1c_CLIENT_NAME/
  .env              ← токены, ключи AI, TZ=Asia/Almaty
  managers.json     ← имена + chat_id менеджеров
  config/
    inventory_categories.json  ← (опционально) категории склада
  reports/          ← очередь, html, json, ai
  logs/
```

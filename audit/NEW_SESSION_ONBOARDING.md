# ИНСТРУКЦИЯ: Как начать новую сессию в этом проекте

## Порядок чтения (первые 5 минут)

### Шаг 1 — Общий контекст проекта
Читать сначала:
```
CLAUDE.md              ← главный мастер-документ: архитектура, соглашения, история багов
gpt1c.md               ← актуальный оперативный статус (collector, баги, P1/P2/P3)
```

### Шаг 2 — Что менялось в последних сессиях
```
SESSION_CONTEXT.md     ← хандоффы всех сессий (последний — наверху)
audit/INDEX.md         ← список всех аудитных документов с датами и статусами
```

### Шаг 3 — Состояние git
```bash
git log --oneline -15   # последние коммиты
git status --short      # что не закоммичено
git branch              # текущая ветка
```
Ветка: `fix/log-noise-by-design-markers` — не запушена. Все collector-правки здесь.

### Шаг 4 — Если работа по коллектору
```
audit/COLLECTOR_PHASES_2_4_2026_04_29.md  ← детали фаз 2–4 (wa_dialog_suppress, audit log, hard-ban)
audit/LOGGING_UNIFICATION_20260430.md     ← unified runtime logging для bot / collector / pipeline
collector/collections_engine.py           ← главный оркестратор
collector/approval_flow.py                ← батч / admin approve flow
collector/collections_db.py               ← state API (suppress, dialogs)
collector/audit_log.py                    ← append-only JSONL audit
collector/collection_agent.py             ← генерация сообщений (шаблоны)
```

### Шаг 5 — Если работа по CRM (следующая задача — Фаза 5)
```
# Открытый баг: повторный "Чей клиент?" по уже закреплённым клиентам
# 6 точек проверки описаны в:
audit/CRM_FULL_AUDIT_2026-04-11.md
# Конкретный баг в текущем плане:
memory/project_roadmap.md                 # секция "Фаза 5"
```

---

## Что НЕ надо читать при старте
- `.venv/` — библиотечный мусор
- `audit/run_20260422_data/` — устаревшие данные конкретного запуска
- Отдельные `MANAGER_SHORTLIST_*.md` — старые ручные списки
- `ПРОМТ_ДЛЯ_*.txt` — промты AI analyzer (не collector)

---

## Критические инварианты (запомнить)

| Правило | Нарушение → |
|---------|-------------|
| `debt` ключ, никогда `closing` | Аналитические баги (уже были) |
| `ZoneInfo(os.getenv("TZ", "Asia/Almaty"))` — всегда через env | Неверная TZ |
| `--send-approved` только после manager+admin approval | WA incident 2026-04-10 |
| `--send` (без `-approved`) — ОТКЛЮЧЁН | Несанкционированная рассылка |
| `legacy_tail_reminder` / `partial_tail_reminder` — NO "отгрузк" в шаблоне | Конфуз клиента (Phase 4) |
| `promise_broken_reminder` — NO "отгрузк" в шаблоне | Конфуз клиента (Phase 4 fix) |
| Violation threshold: `opening >= 100 AND debit > 0` | Ложные срабатывания (1С округления) |

---

## Очередь задач на момент написания (2026-04-29)

| Фаза | Статус | Описание |
|------|--------|----------|
| 3В | Закрыто | Единое runtime logging для `bot` / `collector` / `config.setup_logging()` уже внедрено |
| 5 | Закрыто | CRM баг повторного "Чей клиент?" по уже закреплённым клиентам закрыт |

Детали в `memory/project_roadmap.md` (секции "Очередь: что осталось").

---

## Память проекта

Автосохраняемые знания — в `C:\Users\user\.claude\projects\...\memory\`:
- `MEMORY.md` — индекс всех записей
- `project_roadmap.md` — актуальный план задач с фазами
- `project_collector_architecture.md` — collector architecture
- `project_crm_and_collector_fixes.md` — CRM + collector fixes
- `project_session_20260429b.md` — итоги сессии 2
- `project_stop_list_v107.md` — stop-list v1.0.7
- `project_reporting_pipeline.md` — pipeline отчётов

---

## Тестовая матрица (быстрая проверка)

```bash
# После любой collector-правки:
python tests/test_wa_dialog_suppress.py
python tests/test_diff_notice.py
python tests/test_audit_log.py
python tests/test_phase4_hard_ban.py

# Полная матрица:
python -X utf8 tests/test_project.py
python -X utf8 tests/test_collector.py
python -X utf8 tests/test_phase4_hard_ban.py
python -X utf8 tests/test_audit_log.py
```

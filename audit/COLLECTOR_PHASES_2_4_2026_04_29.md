# COLLECTOR PHASES 2–4 AUDIT — 2026-04-29

Документ: технический итог реализации Фаз 2А, 2Б, UI, 3А, 3Б, 4 в коллекторном контуре.
Дата: 2026-04-29 | Ветка: `fix/log-noise-by-design-markers` | Последний коммит: `34ec994`

---

## 1. Исходная проблема (до Фазы 2)

После реализации Фаз 1А–1В (stale batch blocker, SLA freshness) выявлены дополнительные операционные риски:

| Риск | Описание |
|------|----------|
| R-1 | Клиент заявил об оплате → бот продолжает давить повторными WA на следующий день |
| R-2 | Admin утверждает батч не зная, что данные обновились между preview и approve |
| R-3 | Admin не может проверить статус батча без ожидания авто-уведомления |
| R-4 | Нет централизованного аудитного следа WA-касаний и suppress-событий |
| R-5 | Контуры COLLECTOR и STOP-LIST неразличимы в общем логе |
| R-6 | `promise_broken_reminder` шаблон содержит угрозу ограничения отгрузок — клиенты legacy_tail (без активных отгрузок) получают бессмысленную фразу |

---

## 2. Фаза 2Б — wa_dialog_suppress

### Цель
Предотвратить повторные WA-касания клиентов, которые заявили об оплате или прислали доказательство.

### Реализация

**`collector/collections_db.py`:**
- `_empty_record()`: добавлено поле `"wa_dialog_suppress": None`
- `set_wa_dialog_suppress(name, reason, until_iso)` — записывает `{reason, set_at, until}` + audit
- `get_wa_dialog_suppress(name)` — возвращает активный suppress или None; автосбрасывает если `until < today`
- `clear_wa_dialog_suppress(name)` — явный сброс + audit

**`collector/client_dialog.py`:**
- Ветка `attachment` (доказательство оплаты): `set_wa_dialog_suppress(name, "attachment", today+2д)`
- Ветка `paid_claim` (слова "оплатили", "давно оплатили"): `set_wa_dialog_suppress(name, "paid_claim", today+3д)`

**`collector/collections_engine.py` в `run()`:**
```python
_wa_suppress = get_wa_dialog_suppress(name)  # None если нет или истёк
if _wa_suppress:
    audit("wa_skipped", name=name, reason="suppress", suppress_reason=_wa_suppress["reason"])
    continue
```

### Тесты
`tests/test_wa_dialog_suppress.py` — 7 тестов, все PASS:
1. suppress set → клиент пропускается в run()
2. suppress с истёкшим until → НЕ пропускается (автосброс)
3. suppress после paid_claim в client_dialog
4. suppress после attachment в client_dialog
5. clear_wa_dialog_suppress возвращает None
6. get_wa_dialog_suppress возвращает структуру
7. audit вызывается при suppress_set и suppress_cleared

---

## 3. Фаза 2А — diff-notice при утверждении

### Цель
Admin видит изменения в дебиторке между моментом формирования батча и моментом нажатия "Отправить".

### Реализация

**`collector/collections_engine.py`:**
```python
def preview_batch_changes(batch_id: str, approved_clients: list) -> Optional[str]:
    """Публичная обёртка над _refresh_approved_batch_clients.
    Возвращает human-readable diff или None если изменений нет."""
```

**`collector/approval_flow.py` в `wa_appr_adm_ok`:**
```python
_diff_text = preview_batch_changes(batch_id, approved_clients)
if _diff_text:
    _diff_block = f"\n\n⚠️ <b>Данные обновились с момента формирования:</b>\n{_diff_text}"
# _diff_block добавляется к тексту "✅ Отправка утверждена!"
```

### Тесты
`tests/test_diff_notice.py` — 6 тестов, все PASS:
1. Нет изменений → None
2. Изменение суммы → упоминается в тексте
3. Исчезнувший клиент → "Исчезли из дебиторки"
4. Исключение → возвращает None (graceful)
5. blocked_reason → предупреждение о stale данных
6. Diff-блок появляется в тексте утверждения approval_flow

---

## 4. UI — кнопка 🤖 Коллектор

### Цель
Admin может проверить статус батча в любой момент из главного меню.

### Реализация
**`bot/send_reports.py`:**
- `kb_main()`: добавлен ряд с `InlineKeyboardButton("🤖 Коллектор", callback_data="collector_batch")`
- `_BATCH_STATUS_RU`: словарь 8 статусов → русский текст
- `_format_collector_batch_text()`: загружает последний batch, форматирует статус + список клиентов (до 20) + результаты отправки
- Callback `"collector_batch"`: только admin, показывает formatted text, кнопки "🔄 Обновить" и "◀ Главное меню"

---

## 5. Фаза 3А — audit log

### Цель
Единый наблюдаемый след всех событий коллектора: отправки, пропуски, suppress, батчи.

### Реализация
**`collector/audit_log.py`** (новый файл):
```python
_AUDIT_PATH = ROOT / "logs" / "collector_audit.jsonl"
_write_lock = threading.Lock()

def audit(event: str, **kwargs) -> None:
    record = {"ts": datetime.now(TZ).isoformat(), "event": event, **kwargs}
    # thread-safe append; ошибка логируется но не пробрасывается

def read_recent(n: int = 50) -> list[dict]: ...
def read_for_client(name: str, limit: int = 20) -> list[dict]: ...
```

Покрытые события:
| Событие | Где вызывается | Ключевые kwargs |
|---------|----------------|-----------------|
| `wa_sent` | `collections_engine.run()` | name, amount, msg_type, batch_id |
| `wa_skipped` | `collections_engine.run()` | name, reason (active_client/zero_amount/payment_hold/suppress) |
| `suppress_set` | `collections_db.set_wa_dialog_suppress()` | name, reason, until |
| `suppress_cleared` | `collections_db.clear_wa_dialog_suppress()` | name |
| `batch_created` | `approval_flow.create_batch()` | batch_id, count |
| `batch_approved` | `approval_flow.wa_appr_adm_ok()` | batch_id, approved_count |
| `batch_sent` / `batch_partially_sent` / `batch_send_failed` / `batch_send_empty` | `approval_flow.record_send_results()` | batch_id, sent, skipped |

### Тесты
`tests/test_audit_log.py` — 7 тестов, все PASS.

---

## 6. Фаза 3Б — log prefixes

### Цель
Разделить логи коллектора и стоп-листа в общем потоке.

### Реализация
```python
class _PrefixAdapter(logging.LoggerAdapter):
    def __init__(self, logger, prefix):
        super().__init__(logger, {})
        self._prefix = prefix
    def process(self, msg, kwargs):
        return f"{self._prefix} {msg}", kwargs
```

- `collections_engine.py`: `logger = _PrefixAdapter(logging.getLogger(__name__), "[COLLECTOR]")`
- `bot/debt_stop_control.py`: `LOG = _PrefixAdapter(logging.getLogger("debt_stop_control"), "[STOP]")`

Использование: `grep "[COLLECTOR]" logs/send_reports_*.log` vs `grep "[STOP]"`.

---

## 7. Фаза 4 — hard-ban отгрузок для хвостовых клиентов

### Первичный анализ
Анализ `collection_agent.py` показал:
- `legacy_tail_reminder` и `partial_tail_reminder` шаблоны → ЧИСТЫ (нет "отгрузки")
- Они назначаются с явным `msg_type` → `generate_message()` возвращает шаблон напрямую (AI НЕ вызывается)
- РЕАЛЬНЫЙ РИСк: `promise_broken_reminder` содержал строку:
  ```
  "Невыполнение повторного обещания влечёт ограничение отгрузок."
  ```
- Этот msg_type назначается в `collections_engine.py` для `no_movement + promise_kept=False` — путь, который может достичь legacy_tail-клиентов, у которых нет активных отгрузок

### Фикс (коммит `34ec994`)

**`collector/collection_agent.py`** — до:
```python
"promise_broken_reminder": (
    ...
    "Невыполнение повторного обещания влечёт ограничение отгрузок.\n"
    "Укажите конкретную дату и сумму платежа."
),
```

После:
```python
"promise_broken_reminder": (
    ...
    "Укажите, пожалуйста, конкретную дату и сумму ближайшего платежа."
),
```

**`config/collector_prompts.json`** — добавлен `"promise_broken_reminder"` без отгрузочной фразы (explicit safe override для JSON-конфига).

### Инвариант после фикса

| Шаблон | Содержит "отгрузк" | Назначение |
|--------|-------------------|------------|
| `legacy_tail_reminder` | НЕТ ✅ | Хвостовой долг без движения |
| `partial_tail_reminder` | НЕТ ✅ | Хвостовой долг с частичным погашением |
| `promise_broken_reminder` | НЕТ ✅ (исправлено) | Нарушение обещания (может идти хвостовым) |
| `stoplist_reminder` | ДА (намеренно) ✅ | Живой стоп-кейс с активными отгрузками |

### Тесты
`tests/test_phase4_hard_ban.py` — 7 тестов, все PASS:
1. legacy_tail_reminder — нет "отгруз"
2. partial_tail_reminder — нет "отгруз"
3. promise_broken_reminder — нет "отгруз" (фикс Phase 4)
4. stoplist_reminder — есть "отгруз" (намеренно, guard против регрессии)
5. promise_broken_reminder — всё ещё содержит запрос даты/суммы
6. promise_broken_reminder рендерится без KeyError
7. JSON-конфиг promise_broken_reminder — нет "отгруз"

---

## 8. Итоговая таблица изменённых файлов

| Файл | Фаза | Изменение |
|------|------|-----------|
| `collector/collections_db.py` | 2Б | wa_dialog_suppress field + API |
| `collector/client_dialog.py` | 2Б | suppress при paid_claim и attachment |
| `collector/collections_engine.py` | 2Б, 2А, 3А, 3Б | suppress check, preview_batch_changes, audit calls, _PrefixAdapter |
| `collector/approval_flow.py` | 2А, 3А | diff-block, batch events audit |
| `collector/audit_log.py` | 3А | НОВЫЙ ФАЙЛ — append-only JSONL |
| `bot/send_reports.py` | UI, 3Б | Collector batch menu item |
| `bot/debt_stop_control.py` | 3Б | [STOP] prefix |
| `collector/collection_agent.py` | 4 | promise_broken_reminder — убрана фраза об отгрузках |
| `config/collector_prompts.json` | 4 | promise_broken_reminder добавлен без отгрузочной фразы |
| `tests/test_wa_dialog_suppress.py` | 2Б | НОВЫЙ — 7 тестов |
| `tests/test_diff_notice.py` | 2А | НОВЫЙ — 6 тестов |
| `tests/test_audit_log.py` | 3А | НОВЫЙ — 7 тестов |
| `tests/test_phase4_hard_ban.py` | 4 | НОВЫЙ — 7 тестов |

**Итого: 27 новых тестов, 9 модифицированных файлов, 4 новых файла.**

---

## 9. Что остаётся в очереди

### Фаза 3В — Единое логирование collector/*

Все `collector/*.py` модули должны логировать через `get_collector_logger(__name__)`.
Сейчас у каждого свой logger без префикса.

Файлы: `approval_flow.py`, `collections_db.py`, `client_dialog.py`, `no_movement.py`,
`payment_hold.py`, `collection_agent.py`, `debt_monitor.py`, `manager_dialog.py`, `communications.py`

### Фаза 5 — Полный аудит CRM

**Основной баг:** Менеджерам повторно задаётся "Чей клиент?" по уже закреплённым клиентам.
Примеры: `М Кафе Пиала Лесная поляна`, `М Плов центр ЕСБОЛОВА`, `М Цех мкр Отау`.

6 точек проверки:
1. `get_clients_without_phones()` — когда клиент убирается из очереди
2. `_crm_cleanup_pending()` — не теряет ли ответ при рестарте
3. Персистентность attribution — JSON или только память
4. `crm_daily_task` — не сбрасывается ли attribution при повторном запуске
5. Дубли в `crm_pending_state.json`
6. TTL и cleanup — не истекает ли attribution раньше срока

# INCIDENT REPORT: Несанкционированная WhatsApp-рассылка
**Дата:** 2026-04-10  
**Время инцидента:** 13:30–13:33 (Asia/Almaty)  
**Статус:** Расследование завершено. Бот отключён вручную.  
**Автор:** Claude Code (расследование 2026-04-11)

---

## 1. Описание инцидента

10 апреля 2026 года в 13:30 бот `E:\GPT1C_Processor_analitica` отправил 22 WhatsApp-сообщения клиентам компании Минбаракат. Среди получателей оказались **добросовестные плательщики и активно покупающие клиенты** — то есть клиенты, которых система никогда не должна была трогать. Ни менеджеры, ни директор не давали разрешения на рассылку. Произошёл реальный инцидент с клиентами.

---

## 2. Что произошло фактически

- **13:30:00** — вручную запущен `collector/collections_engine.py --send` (не по расписанию; по расписанию — 17:30)
- **13:30:00** — загружено 12 debt JSON файлов, объединено 426 клиентов
- **13:30:00** — классифицировано **84 должника** уровней 1–5 (при норме ~5–8)
- **13:30:05** — отправлены уведомления менеджерам о нарушениях (16 violation warnings)
- **13:30:15** — первый "нормальный" диалог запущен для Арман ИП Кыпшак → лок менеджера Алена
- **13:30:30** — первый "direct send без manager lock" для второго клиента Алены
- **13:30:37** — **первый WhatsApp отправлен: +77024112935** (ТД Сарыарка 2 ряд, Алена)
- **13:33:12** — **последний WhatsApp отправлен: +77055775597** (22-й)
- **13:33:17** — сводка: "✅ LIVE / Отправлено сообщений: 22"

**Полный список отправленных номеров (22 шт.):**
```
+77024112935  +77011112613  +77015266744  +77088388986  +77026167012
+77028189024  +77758887020  +77014219838  +77759364040  +77712444222
+77016181327  +77026848900  +77752811388  +77775899137  +77718999529
+77051828101  +77719076097  +77026102025  +77025853484  +77025650202
+77055775597  +77055775597  (последний номер дважды — два разных клиента, один телефон)
```

---

## 3. Что прочитано и проверено

| Файл/ресурс | Статус |
|---|---|
| `git log --oneline -15` | ✅ прочитан |
| `git show 4f60851` (commit "WA теперь реально отправляется") | ✅ полный diff |
| `collector/collections_engine.py` (текущий) | ✅ прочитан (строки 340–640) |
| `collector/debt_monitor.py` (текущий + diff) | ✅ прочитан |
| `logs/collector_20260410.log` | ✅ полностью |
| `logs/collector_20260409.log` | ✅ выборочно |
| `logs/collector_state.json` | ✅ структура + first_seen даты |
| `reports/json/debt_ext_*` (4 типа × Алена) | ✅ структура + days_silence |
| `.env` — WHATSAPP_ENABLED | ✅ = 1 |
| `bot/send_reports.py` — scheduler timing | ✅ = 17:30 |

---

## 4. Подтверждённые причины (с доказательствами)

### ПРИЧИНА 1 — CRITICAL: Удалён фильтр защиты активных клиентов

**Коммит:** `4f60851` "fix: 7 багов коллектора — WA теперь реально отправляется" (April 9, 10:18)

**Старый код (безопасный):**
```python
# collector/collections_engine.py — ДО коммита 4f60851
debit  = client.get("debit", 0.0) or 0.0
credit = client.get("credit", 0.0) or 0.0
if debit > 0 or credit > 0:
    # Клиент активен — пропускаем (покупает ИЛИ платит)
    continue
```

**Новый код (опасный):**
```python
# collector/collections_engine.py — ПОСЛЕ коммита 4f60851
if client.get("amount", 0) <= 0:
    # Пропуск только если долг = 0 или отрицательный
    continue
```

**Доказательство — данные debt JSON (Алена, April 10):**
- 52 из 72 клиентов имеют `debit > 0` И `credit > 0` (активно покупают и платят)
- Со старым фильтром: 52 клиента пропущены → 0 WhatsApp
- С новым фильтром: все 52 проходят → попадают в рассылку

**Вывод:** Коммит удалил ключевую защиту. Логика "покупает или платит — не трогать" была заменена на "долг > 0 — трогать". Это фундаментальная ошибка в бизнес-логике.

---

### ПРИЧИНА 2 — CRITICAL: `get_debt_days_since_first_seen` надувает дни до уровня 3

**Код (collections_engine.py, строки 567–571):**
```python
real_days = 0 if dry_run else get_debt_days_since_first_seen(name)
level = max(client["level"], _level_for_days(real_days))
client = dict(client, level=level, days=max(client["days"], real_days))
```

**Доказательство — collector_state.json:**
- 203 клиента имеют `first_seen = "2026-03-19"`
- April 10 − March 19 = **22 дня** → `_level_for_days(22)` = **level 3** (20–24 дня)
- `level = max(client_level, 3)` → все клиенты, первый раз замеченные 19 марта, получают минимум уровень 3

**В логе подтверждается:**
```
ВСЕ WhatsApp-клиенты: level=3 days=22
[А Арман ИП Кыпшак ...] level=3 days=22
[Е Еркебулан] level=3 days=22
[О ТД Артем...] level=3 days=22
```

**Вывод:** Клиент с реальными `days_silence=3` из debt JSON (уровень 0 — пропустить!) получил `real_days=22` из state → уровень 3. Это почему "добросовестные клиенты" оказались в категории должников: counter inflated от даты первого обнаружения, а не от реальных дней просрочки в 1С.

---

### ПРИЧИНА 3 — CRITICAL: "direct send без manager lock" — обход ВСЕГО approval

**Код (collections_engine.py, строки 363–376):**
```python
elif _dialog_pre and _dialog_pre.get("state") not in ("CONFIRMED", "DONE", None):
    # Soft-lock: менеджер уже занят другим кейсом
    logger.info("[%s] менеджер %s уже ведёт диалог по %s — direct send без manager lock",
                name, manager_name, _dialog_pre.get("client_name"))
    if phone and not contact.get("_needs_phone", False):
        manager_chat_id = None   # ← КЛЮЧЕВАЯ СТРОКА
    else:
        return result            # без телефона — не отправляем
```

Затем (строки 394–398):
```python
if not manager_chat_id:          # ← manager_chat_id стал None → прямая отправка
    if phone:
        wa_ok = send_whatsapp(phone, text)  # ← WhatsApp БЕЗ approval!
```

**Доказательство — лог (22 строки "direct send без manager lock"):**
```
[А ТД Сарыарка 2 ряд 1 место Ляззат] менеджер Алена уже ведёт диалог по А Арман... — direct send без manager lock
[О Стол.Аппетит ИП Вектор ...] менеджер Оксана уже ведёт диалог по О ТД Артем... — direct send без manager lock
[М Домовая кухня ...] менеджер Магира уже ведёт диалог по М Детс,сад... — direct send без manager lock
... (22 клиента)
```

**Механизм:** Как только один клиент запускает диалог с менеджером — ВСЕ остальные клиенты этого менеджера получают WhatsApp без какого-либо подтверждения. Механизм "soft-lock" превратился в bypass.

**Введён в:** коммит `e56c196` "feat: CRM v2 + collector on-stop filter + contacts Excel sync"

---

### ПРИЧИНА 4 — CONFIRMED: WHATSAPP_ENABLED=1 в .env

- `.env` сейчас: `WHATSAPP_ENABLED=1`
- Лог April 9 09:00: `WhatsApp отключён (WHATSAPP_ENABLED=0)` — отправок не было
- Лог April 10 13:30: отправки идут — WHATSAPP_ENABLED уже = 1
- `.env` не в git → смена не задокументирована
- Очевидно: при тестировании коммита 4f60851 "WA теперь реально отправляется" WHATSAPP_ENABLED был переключён на 1 вручную

---

### ПРИЧИНА 5 — CONFIRMED: Ручной запуск вместо запланированного

- Расписание scheduler: `debt_collector_daily` в **17:30** (Asia/Almaty)
- Фактический запуск: **13:30:00** April 10 (ровное время)
- В логе нет записей между 06:00 и 13:30 → не из scheduler
- Запуск: `python -m collector.collections_engine --send` из CLI
- Запуск без dry-run, без проверки state, без supervision

---

## 5. Подозреваемые, но не доказанные причины

| # | Гипотеза | Статус |
|---|---|---|
| H1 | `_safe_float` некорректно парсит суммы типа "1.234,56" → возвращает 0.0 → amount=0 → пропускает клиента | Не подтверждено: 1С использует пробел+запятая, _safe_float корректен |
| H2 | Merge logic перезаписывает days_silence некорректно (берёт max, но неправильно) | Частично подтверждено: дни в логе (22) > дней в любом JSON файле (max=12 для Арман) — но объяснено через RC-2 |
| H3 | "Ведомость по взаиморасчетам" содержит НЕ-должников (всех контрагентов) | Требует проверки: файл содержит 74 клиентов с `debt > 0`, но у многих `days_silence < 10` (уровень 0) |

---

## 6. Какие ветки кода привели к отправке

```
collector --send
  └─ run() [collections_engine.py:460]
       ├─ load_latest_debt_json() → 12 файлов, 426 клиентов
       ├─ classify_debtors() [debt_monitor.py]
       │    └─ фильтр: amount > 0 (RC-1: debit/credit НЕ проверяются)
       │         → 84 клиента проходят (норма: ~5-8)
       ├─ for client in debtors:
       │    ├─ real_days = get_debt_days_since_first_seen() → 22 (RC-2)
       │    ├─ level = max(client.level, level_for_days(22)) → 3 (RC-2)
       │    ├─ фильтр level==0: ПРОПУСК (но level уже 3 из-за RC-2)
       │    ├─ stop-list check → некоторые пропущены
       │    └─ _process_single():
       │         ├─ check manager dialog → manager уже занят?
       │         │    └─ YES → manager_chat_id = None (RC-3: soft-lock bypass)
       │         └─ if not manager_chat_id:
       │              └─ send_whatsapp(phone, text)  ← ОТПРАВКА (RC-4: WA enabled)
       └─ manual run at 13:30, not dry-run (RC-5)
```

---

## 7. Какие state/log это подтверждают

| Источник | Доказательство |
|---|---|
| `logs/collector_20260410.log:321` | Сводка: "✅ LIVE / Отправлено сообщений: 22" |
| `logs/collector_20260410.log` строки 74, 95, 106, 113... | 22 строки "direct send без manager lock" |
| `logs/collector_20260410.log` строки 78–307 | 22 "WhatsApp отправлен: +7..." |
| `logs/collector_state.json` | 203 клиента с `first_seen=2026-03-19` |
| `git show 4f60851` | Diff: удалён `if debit > 0 or credit > 0: continue` |
| `reports/json/debt_ext_*Алена*` | 52/72 клиентов с debit>0 И credit>0 |
| `.env` | `WHATSAPP_ENABLED=1` |
| `bot/send_reports.py:6867` | `run_daily(debt_collector_daily, time=dt_time(17, 30, ...))` |

---

## 8. Критические риски

| Риск | Уровень | Описание |
|---|---|---|
| Повторная рассылка при следующем запуске | **CRITICAL** | Все 5 причин остаются в коде/state |
| Рассылка в 17:30 сегодня (если бот включить) | **CRITICAL** | `WHATSAPP_ENABLED=1`, state не сброшен |
| "already_contacted_today" защищает только за сегодня | **HIGH** | Завтра утром все 22 клиента снова получат сообщения |
| 203 клиента с first_seen=2026-03-19 → real_days растёт каждый день | **HIGH** | Завтра real_days=23, послезавтра=24, уровень по-прежнему 3 |
| Ни один send путь не требует явного admin approval | **HIGH** | Нет хардгарда перед send_whatsapp() |

---

## 9. Немедленные меры защиты (до любых правок)

> Бот уже отключён. Ниже — что нужно ПЕРЕД любым включением.

1. **WHATSAPP_ENABLED=0** в `.env` — первое действие перед любым тестом
2. **COLLECTOR_DRY_RUN=true** в `.env` — второе действие
3. Проверить: в `.env` нет `WHATSAPP_ENABLED=1` И `COLLECTOR_DRY_RUN=false` одновременно
4. Добавить hardguard в `send_whatsapp()`: проверять env каждый раз при вызове (не только при старте)

---

## 10. План исправления (точечные правки)

### FIX-1 (CRITICAL) — Вернуть фильтр защиты активных клиентов
**Файл:** `collector/collections_engine.py`  
**Строки:** ~595–601  
**Правка:**
```python
# БЫЛО (опасно):
if client.get("amount", 0) <= 0:
    logger.info("[%s] пропуск — долг погашен...", ...)
    continue

# СТАЛО (безопасно):
debit  = client.get("debit", 0.0) or 0.0
credit = client.get("credit", 0.0) or 0.0
if debit > 0 or credit > 0:
    logger.info("[%s] пропуск — клиент активен (debit=%.0f, credit=%.0f)", name, debit, credit)
    continue
if client.get("amount", 0) <= 0:
    logger.info("[%s] пропуск — долг погашен или отрицательный", name)
    continue
```

### FIX-2 (CRITICAL) — Убрать "direct send без manager lock"
**Файл:** `collector/collections_engine.py`  
**Строки:** ~363–376  
**Правка:** Когда менеджер занят другим диалогом — НЕ отправлять клиенту, а ставить в очередь ожидания или уведомить менеджера о следующем клиенте.
```python
elif _dialog_pre and _dialog_pre.get("state") not in ("CONFIRMED", "DONE", None):
    logger.info("[%s] менеджер %s уже занят диалогом — клиент пропущен до следующего запуска",
                name, manager_name)
    return result  # НЕ отправлять, ждать следующего цикла
```

### FIX-3 (HIGH) — Ограничить `get_debt_days_since_first_seen` порогом
**Файл:** `collector/collections_engine.py`  
**Строки:** ~567–571  
**Правка:** `real_days` НЕ должен превышать дней из 1С-данных более чем на X дней (например, +7). Или не использовать для уровня — только для информации.
```python
real_days = 0 if dry_run else get_debt_days_since_first_seen(name)
# Не накручивать уровень выше 1С-данных более чем на 7 дней
capped_real_days = min(real_days, client["days"] + 7)
level = max(client["level"], _level_for_days(capped_real_days))
client = dict(client, level=level, days=max(client["days"], capped_real_days))
```

### FIX-4 (HIGH) — Hardguard перед каждым send_whatsapp()
**Файл:** `collector/communications.py`  
**Правка:** добавить в `send_whatsapp()`:
```python
def send_whatsapp(phone: str, text: str) -> bool:
    if not _get_bool_env("WHATSAPP_ENABLED"):
        logger.info("WhatsApp отключён (WHATSAPP_ENABLED=0) — пропуск: %s", phone)
        return False
    # ... остальной код
```

### FIX-5 (MEDIUM) — Admin final approval для уровней 3–5
Для клиентов level >= 3 добавить шаг: уведомить admin (Вадим, chat_id=7422963573) с кнопкой подтверждения перед отправкой WhatsApp.

### FIX-6 (MEDIUM) — Сбросить inflated first_seen даты
В `logs/collector_state.json` удалить или обнулить `first_seen` для клиентов, у которых `days_silence` в актуальном JSON < 10 дней (уровень 0). Это предотвратит повторную рассылку завтра.

---

## 11. Условия безопасного повторного включения бота

Бот можно включать ТОЛЬКО если выполнены ВСЕ пункты:

- [ ] `WHATSAPP_ENABLED=0` в `.env` → ВКЛ только после всех FIX
- [ ] `COLLECTOR_DRY_RUN=true` → запустить dry-run и проверить список "кандидатов"
- [ ] FIX-1 применён (фильтр debit/credit возвращён)
- [ ] FIX-2 применён (убран direct send bypass)
- [ ] FIX-3 применён (ограничен real_days)
- [ ] FIX-4 применён (hardguard в send_whatsapp)
- [ ] dry-run показывает ≤ 5–8 клиентов (норма), без активных плательщиков
- [ ] Вадим лично проверил список клиентов в dry-run перед первым `--send`
- [ ] `WHATSAPP_ENABLED=1` включается только после этого, с явным commit в .env.example

---

*Расследование завершено. Причины установлены. Код не изменён — только читался.*

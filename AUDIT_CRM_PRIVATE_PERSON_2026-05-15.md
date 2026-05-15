# AUDIT — CRM фильтр по "частным лицам" без идентификатора

**Дата:** 2026-05-15
**HEAD:** `e9ab178`
**Тип:** Точечный аудит CRM-контура (Блок A из стабилизационного ТЗ)

---

## Executive summary

**Root cause обнаружен и однозначен.** В 1С системные placeholder-имена вида `"Частное лицо"` / `"Частное лицо N"` приходят как обычные клиенты в debt/sales выгрузках. На уровне CRM-контура они **не** распознаются как служебные — потому что в `_VENDOR_NAME_KEYWORDS` (`bot/crm_clients.py:74-91`) нет соответствующего keyword. В результате:

1. `update_from_reports()` создаёт их как живых клиентов (без `is_vendor`/`do_not_call`).
2. `crm_daily_task()` → `_crm_collect_unowned_claim_clients()` рассылает claim broadcast: **«чей это клиент?»** всем 5 участникам CRM.
3. Менеджер вынужден угадывать, кто такой `"Частное лицо 1"`, забирать его себе и заполнять телефон.

**Эталонный кейс** (живой, сегодня): `logs/crm_claim_pending_state.json`:

```json
{
  "claim_20260515180004_de7b9799": {
    "client_key": "Частное лицо 1",
    "notified": [188939016, 1446255940, 735574334, 756622791, 7422963573],
    "claimed_by": "Магира"
  }
}
```

5 менеджеров получили вопрос про placeholder из 1С. Магира взяла.

В `config/clients.json` уже накопилось **6 таких записей** (все заполнены менеджерами через CRM-цикл, что подтверждает многократность инцидента):

| client_key | manager | sources | has_phone |
|---|---|---|---|
| `А Частное Лицо` | Алена | debt, sales | да |
| `Е 1. Частное лицо` | Ергали | debt, sales | да |
| `М Частное лицо` | Магира | debt, sales | да |
| `О Частное лицо` | Оксана | debt, sales | да |
| `Частное лицо` | Оксана | sales | да |
| `Частное лицо 1` | Магира | sales | да |

**Фикс — однострочный.** Добавить keyword `"частное лицо"` (+ профилактически `"физическое лицо"`, `"физлицо"`) в `_VENDOR_NAME_KEYWORDS`. Это автоматически закрывает все CRM-пути (phone-pending, claim broadcast, dup-review, update_from_reports) — потому что они все идут через единый предикат `is_service_client_name()`.

---

## Detailed findings

### F-A1 | HIGH | placeholder из 1С пробивает фильтр CRM

**Symptom:** Менеджеры получают claim broadcast по записям типа `"Частное лицо 1"`, `"Частное лицо 2"` — placeholder-имена 1С без какого-либо identifier (нет ФИО, нет компании, нет адреса).

**Root cause:** `_VENDOR_NAME_KEYWORDS` в `bot/crm_clients.py:74-91` не содержит "частное лицо". `is_service_client_name(name)` возвращает `False` для таких имён → запись считается живым клиентом.

**Code location:** `bot/crm_clients.py:74-91` (массив keywords) и `bot/crm_clients.py:225-230` (функция `is_service_client_name`).

**Evidence:**
- `logs/crm_claim_pending_state.json` — активный claim сегодня (15.05) на `"Частное лицо 1"`, 5 менеджеров notified
- `config/clients.json` — 6 placeholder-записей, все имеют `manager` и `whatsapp` (пройдены через CRM-цикл)
- `logs/crm_audit.jsonl` — 68 phone/claim events суммарно

**Impact:**
- Менеджер обязан "угадать", кто такой `"Частное лицо 1"`, и заполнить его данные
- Записи раздувают базу: 6 дубликатов вместо одной сводной строки
- Penalty mechanic может начислить штраф менеджеру за неответ по нерешаемому placeholder
- CRM-флоу теряет доверие — пользователи начинают игнорировать настоящие запросы

**Fix direction (одно место правки):**

```python
# bot/crm_clients.py:74-91
_VENDOR_NAME_KEYWORDS = (
    "зарплат", "зар.плат", "зар плат",
    "з.п.", "з/п", "зп",
    "по зп", "под зп",
    "тов по з", "тов под з",
    "товар по з", "товар под з",
    "аванс сотр",
    "водитель",
    "недостача",
    "без клиента",
    # NEW (2026-05-15):
    "частное лицо",        # 1C placeholder для непоименованных физлиц
    "физическое лицо",     # альтернативная форма из 1C
    "физлицо",             # сокращённая форма
)
```

`canonicalize_client_key()` уже делает `.lower()` и `ё → е`, поэтому keywords хранятся в нижнем регистре без ё. False positive минимален — слова специфичны.

**Что автоматически закроется одной правкой:**

| Путь | Где использует `is_service_client_name` |
|---|---|
| Создание новой записи из debt/sales | `update_from_reports` → `_is_vendor_name` (`bot/crm_clients.py:722-744`) — новые "Частное лицо N" сразу помечаются `is_vendor=True` + `do_not_call=True` |
| Daily phone-pending | `get_clients_without_phones` (`bot/crm_clients.py:787-805`) — placeholder не попадает в очередь |
| Claim broadcast | `_crm_collect_unowned_claim_clients` (`bot/send_reports.py:7112+`) — исключает service rows |
| Dup-review broadcast | `get_phone_conflict_groups` через `is_service_client_name` |

**Backfill (опциональная миграция существующих 6 записей):**

Скрипт-однострочник для уже накопленных placeholder-записей: проставить `is_vendor=True` + `do_not_call=True` в 6 кейсах. Это исключит их из любых future cycles, даже если они снова появятся в дебиторке. Альтернатива — оставить как есть (они уже заполнены, новые request'ов по ним не пойдёт).

---

### Регрессионные тесты

Обязательные кейсы для `tests/test_crm_regression.py`:

```python
def test_is_service_client_name_skips_chastnoe_litso(self):
    self.assertTrue(crm.is_service_client_name("Частное лицо"))
    self.assertTrue(crm.is_service_client_name("Частное лицо 1"))
    self.assertTrue(crm.is_service_client_name("М Частное лицо"))
    self.assertTrue(crm.is_service_client_name("Е 1. Частное лицо"))
    self.assertTrue(crm.is_service_client_name("А Частное Лицо"))  # camelcase из 1C
    self.assertTrue(crm.is_service_client_name("Физическое лицо"))
    self.assertTrue(crm.is_service_client_name("Физлицо Иванов"))

def test_is_service_client_name_keeps_real_company(self):
    # Не должно зацепить настоящих клиентов с похожими словами
    self.assertFalse(crm.is_service_client_name("М ТОО Частная клиника"))
    self.assertFalse(crm.is_service_client_name("О Магазин Физкультура"))
    self.assertFalse(crm.is_service_client_name("Е ИП Розничный магазин"))

def test_update_from_reports_marks_chastnoe_litso_as_vendor(self):
    # При появлении нового "Частное лицо 5" в debt-выгрузке —
    # запись создаётся сразу с is_vendor=True и do_not_call=True.
    # Pending не появляется.
    ...

def test_collect_unowned_claim_skips_chastnoe_litso(self):
    # В _crm_collect_unowned_claim_clients placeholder исключается
    ...
```

---

## Что НЕ обнаружено в clients.json (риски, проверены и отсутствуют)

- **Пустые имена** (только пробелы) — отсутствуют
- **Имена-цифры** ("12345", "100500") — отсутствуют
- **Single-word без префикса** — отсутствуют (все одиночные имена с manager-префиксом А/Е/М/О)
- **"Без названия" / "Неизвестный"** — отсутствуют

То есть **единственная активная placeholder-уязвимость** — это "Частное лицо". Остальные защищены `_VENDOR_NAME_KEYWORDS` или ручной модерацией.

---

## Recommendations (по приоритету)

| # | Действие | Severity | Эффект |
|---|---|---|---|
| 1 | Добавить 3 keyword в `_VENDOR_NAME_KEYWORDS` | P1 | Закрывает будущие placeholder-инциденты |
| 2 | Backfill: пометить 6 существующих "Частное лицо*" как `is_vendor=True` | P2 | Защита от reopen при следующей debt-выгрузке |
| 3 | Тесты на 4 case (skip + keep + update_from_reports + claim) | P1 | Регрессия |

**Что НЕ нужно делать:**
- Не нужно вводить отдельную функцию `is_private_person()` — keyword в существующем списке решает задачу полностью
- Не нужно фильтр по длине имени / отсутствию contact_name / отсутствию address — текущие 6 случаев это не покрывает, риск false positive выше пользы
- Не нужно отдельной admin-команды на backfill — простой migration script

---

## Acceptance

Задача закрыта когда:
- 3 keyword добавлены, тесты зелёные
- (опционально) 6 существующих placeholder-записей помечены `is_vendor=True` + `do_not_call=True`
- При следующей debt-выгрузке с placeholder-именами menager не получает claim broadcast

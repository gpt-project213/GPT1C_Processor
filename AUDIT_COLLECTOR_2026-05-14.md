# КОМПЛЕКСНЫЙ АУДИТ AI COLLECTOR
**Дата:** 2026-05-14  
**Версия кода:** первоначальный срез аудита — HEAD master (`0a22192`); дополнительно перепроверено на HEAD master (`63553b4`)  
**Объект:** контур `collector/` и связанные модули  
**Проверено на:** текущий код + боевые логи + state-файлы

---

## 1. EXECUTIVE SUMMARY

### Критические выводы

| # | Severity | Вывод |
|---|---|---|
| 1 | **High** | Exception grace bypass: клиент с `stop_status=exception` без `cleared_at`/`approved_at` — grace не активируется, сразу попадает в batch |
| 2 | **High** | Race condition: `handle_manager_proof` в `approval_flow.py:2791` — неверный порядок dict merge, возможна потеря batch-state |
| 3 | **High** | Race condition: `saida_payment_holds.json` — нет межпроцессного лока, параллельный bot+collector может затереть hold |
| 4 | **High** | Нет тестов для zeropay flow: `_handle_saida_zeropay_confirm/deny` в `debt_stop_control.py:2372` |
| 5 | **Medium** | `pending_saida` hold не блокирует клиента в collector shortlist — клиент может получить WA пока Саида не ответила |
| 6 | **Medium** | Admin UI врёт: `send_results` показывает `0/N доставлено` вместо реального числа (key mismatch: `"sent"` vs `status=="sent"`) |
| 7 | **Medium** | `paid_claim` зомби-диалог: не вызывается `escalate_to_manager`, состояние `awaiting_payment_proof` зависает вечно без таймаута |
| 8 | **Medium** | `off_topic_count` не сбрасывается при смене intent — ложная преждевременная эскалация на 2-м unclear |
| 9 | **Medium** | `wa_appr_adm_later` не меняет `batch.status` — отложенный батч молча вытесняется при следующем preview |
| 10 | **Medium** | Visual ghost: agreed/paid клиенты в detail view директора показываются иконкой `◯` (нерассмотренный) |
| 11 | **Operational** | UI не показывает пропущенных клиентов — причины skip видны только в `collector.log`, не в Telegram |
| 12 | **Operational** | 13.05 из 86 должников в батч попал 1 клиент: 7 — active dialog, 2 — Саида hold, 1 — малый остаток |

### Что критично
- Bugs #1, #2, #3 — риск state corruption и неожиданного поведения в production
- Bug #5 + #7 — повторное давление на клиентов, которых уже обрабатывают

### Что high
- Bug #4 — zero coverage на zeropay flow (3 раза открыто в session context, не закрыто)
- Bug #6 — UI постоянно врёт admin о результатах отправки

### Что medium
- Bugs #8, #9, #10 — операционная путаница и неправильные решения admin/менеджеров

### Что является operational, а не кодовой проблемой
- Пропущенные клиенты не видны в UI — архитектурное решение, но создаёт операционную непрозрачность
- exception grace duration (2 дня) — business rule, не баг; но отсутствие anchor date — баг

---

## 1.1. ВЕРИФИКАЦИЯ FINDINGS НА ТЕКУЩЕМ HEAD

Ниже — дополнительная верификация спорных findings уже после локальной перепроверки кода на более новом HEAD (`63553b4`).
Это нужно, чтобы отделить:
- подтверждённый баг;
- корректную идею, но неточную формулировку;
- спорный remediation path;
- test gap, который не равен production bug.

### Подтверждаю как реальные баги

| Finding | Статус | Комментарий |
|---|---|---|
| `F-01` Exception grace bypass | **Подтверждён** | В `collector/collections_engine.py` grace для `exception` реально работает только при наличии `cleared_at` или `approved_at`. Если anchor date нет, `_exception_grace_active()` возвращает `False`, и клиент может сразу снова попасть в цикл. |
| `F-03` Race в `saida_payment_holds.json` | **Подтверждён** | В `collector/payment_hold.py` есть atomic replace, но нет межпроцессного lock. Для параллельных writer’ов это риск lost update. |
| `F-05` `pending_saida` не блокирует shortlist | **Подтверждён** | `collector/payment_hold.py:get_hold_for_client()` смотрит только `ACTIVE_STATUSES`. `pending_saida` в них не входит, а collector shortlist опирается именно на `get_hold_for_client()`. |
| `F-06` UI показывает `0/N` вместо реального числа | **Подтверждён** | В `bot/send_reports.py` число доставок считается по `r.get("sent")`, тогда как реальные send-results обычно хранят `status == "sent"`. Это реальный UI-баг. |
| `F-08` `off_topic_count` не сбрасывается | **Подтверждён** | В `collector/client_dialog.py` счётчик повышается в `unclear/off_topic` ветке, но явного reset по смене intent нет. Это создаёт риск ложной ранней эскалации. |

### Подтверждаю частично / с оговорками

| Finding | Статус | Комментарий |
|---|---|---|
| `F-02` Race в `handle_manager_proof` | **Частично подтверждён** | Проблема реальна: выражение `{state["batch_id"]: batch} \| _load_batches()` даёт неправильный merge-order и может перетереть обновлённый in-memory batch старой версией из файла. Последующий `save_batch(batch)` это не компенсирует: если между двумя write другой процесс успел записать state, оба write могут затереть чужие изменения. Поэтому это timing-dependent race, а не harmless промежуточный шаг. |
| `F-04` Нет тестов для zeropay | **Подтверждён как test gap** | Это действительно открытый пробел покрытия и старый незакрытый хвост. По severity это test/quality gap, а не production bug сам по себе. |
| `F-07` paid_claim зомби-диалог | **Частично подтверждён** | Ветка `paid_claim` действительно может долго жить в `awaiting_payment_proof`, и явного TTL/auto-resolution не видно. Проблема в lifecycle/state-timeout, а не в отсутствии немедленного `escalate_to_manager`, потому что observer-notify для менеджера уже есть; правильная точка фикса — TTL/cleanup этой ветки. |

### С чем не согласен в предложенном remediation path

| Пункт | Статус | Комментарий |
|---|---|---|
| `P0: payment_hold.py:32 — добавить pending_saida в BLOCK_STATUSES` | **Формулировка неточная** | В текущем коде нет константы `BLOCK_STATUSES`. Суть проблемы верная, но фикс должен идти через логику `get_hold_for_client()` / блокирующих статусов shortlist, а не через несуществующий символ. |
| `P0: client_dialog.py:1065 — добавить escalate_to_manager в paid_claim ветке` | **Не согласен как с P0-решением** | Простое добавление эскалации может дать лишнюю двойную маршрутизацию. Корень проблемы — lifecycle `awaiting_payment_proof`, его timeout/cleanup и судьба диалога после устаревания, а не отсутствие manager-notify как такового. |

### Отдельная оговорка по версии кода

- Шапка исходного аудита ссылается на `0a22192`, но после аудита collector уже был изменён и перепроверен на `63553b4`.
- Поэтому findings выше нужно читать так:
  - `F-01`, `F-03`, `F-05`, `F-06`, `F-08` — подтверждены и на более новом HEAD;
  - `F-02`, `F-04`, `F-07` — требуют аккуратной переформулировки в remediation-плане;
  - remediation steps из исходного аудита нельзя применять механически без этой поправки.

### Практический итог верификации

- Как реальные production bugs после повторной сверки подтверждаются:
  - `F-01`
  - `F-03`
  - `F-05`
  - `F-06`
  - `F-08`
- Как реальные, но требующие более аккуратной формулировки:
  - `F-02`
  - `F-04`
  - `F-07`
- Наиболее спорными оказались не сами symptoms, а предложенные быстрые fixes для `pending_saida` и `paid_claim`.

### Дополнительная фиксация позиции после review

- `F-02`:
  - severity `Medium` выглядит обоснованно, потому что баг timing-dependent;
  - аргумент "последующий `save_batch(batch)` компенсирует проблему" считать неверным;
  - корректная формулировка: это тот же stale in-memory batch, и второй write не лечит race, а может повторно затереть чужие изменения.
- `F-04`:
  - полностью подтверждён как quality/test gap;
  - трактовать как production bug не нужно.
- `F-05 remediation`:
  - замечание про несуществующий `BLOCK_STATUSES` подтверждено;
  - remediation надо формулировать через `get_hold_for_client()` / shortlist-blocking logic.
- `F-07`:
  - symptom подтверждается;
  - remediation через `TTL`/cleanup `awaiting_payment_proof` корректнее, чем добавление ещё одного `escalate_to_manager`.

---

## 2. АРХИТЕКТУРНАЯ КАРТА COLLECTOR

### 2.1 Полный pipeline (current behavior)

```
[1С выгрузка] → debt_ext JSON файлы (по менеджерам)
        │
        ▼
 debt_monitor.classify_debtors()
   → список должников level 1–5
   → поля: days, debt_age_days, level, amount, credit, debit
        │
        ▼
 _apply_deferral_metrics()   ← payment_deferrals.json
   → effective_overdue_days = max(0, actual - deferral)
   → level по deferral шкале (eff≥2→L1, eff≥5→L3, eff≥10→L5)
        │
        ▼
 sync_deferral_discipline()  ← логирование нарушений
 sync_holds_with_debtors()   ← актуализация holds
 clear_missing_sticky_approvals()
        │
        ▼ для каждого клиента
 _apply_collector_day_policy()    ← малый остаток guard
 _collector_candidate_decision()  ← главный роутер
   │
   ├── skip: amount≤0 / deferral / payment_hold / suppress /
   │         exception grace / do_not_notify / healthy payer
   │
   ├── client_approval: stopped / partial payment / no-movement
   │
   └── manager_review: pending_clearance / conditional / debit>0 credit=0
        │
        ▼
 STICKY CHECK:
   - eligible? (action=client_approval, debit=0, msg_type в STICKY_TYPES)
   - signature совпала? → sticky_auto_clients (минует менеджеров)
   - не совпала? → clear_sticky + debtors_by_manager
        │
        ▼
  [Есть debtors_by_manager?]
    YES → create_batch(status=pending_managers) → send_manager_previews()
    NO + sticky → create_batch(status=admin_approved) → skip managers
    NO + no sticky → нет батча
        │
        ▼
 MANAGER PREVIEW (Telegram inline-кнопки, TTL 1ч):
   approve_all / agree / paid_doc / paid_no_doc / manual_editing
        │
   [Все ответили ИЛИ timeout 1ч]
        ▼
 promote_silent_batches_to_admin() → status: pending_admin
        │
        ▼
 ADMIN SUMMARY → Telegram сводка Вадиму:
   - клиенты по менеджерам с иконками решений
   - кнопки: Отправить / Отложить / Отмена
        │
        ├── wa_appr_adm_send → send_approved_batch() → LIVE SEND
        ├── wa_appr_adm_later → admin_status=postponed (status остаётся pending_admin!)
        └── wa_appr_adm_no → status=cancelled
        │
        ▼ wa_appr_adm_send:
 _refresh_approved_batch_clients()
   → _prepare_current_approved_clients() ← пересверка по свежей дебиторке
   → _send_approved_client() per client
        │
        ▼
 start_client_dialog() → WhatsApp → Green API
   → dialog state: active
        │
 [Клиент отвечает в WhatsApp]
        │
        ▼ whatsapp_poller.py → client_dialog.py
 _route_message() → AI classify (collection_agent.py)
   │
   ├── paid_claim → ждём чек (3 дня suppress), НЕ escalate
   ├── promise → подтверждение, escalate 2 дня
   ├── soft_positive → уточнение, escalate при exchange≥2
   ├── unclear → AI ответ; 2й unclear → escalate
   ├── service_request/dispute/refusal → escalate
   └── attachment → notify manager, 2 дня suppress
        │
        ▼ manager уведомлен:
 manager_dialog.py → Telegram → менеджер отвечает
```

### 2.2 Shortlist формирование — подробно

| Условие | Результат | Механизм |
|---------|-----------|----------|
| `amount ≤ 0` | skip | `_collector_candidate_decision:line~620` |
| `deferral_days > 0 AND effective_days ≤ 0` | skip: срок не истёк | `line~613` |
| `deferral_days > 0 AND effective_days == 1` | skip: первый день | `line~625` |
| `payment_hold active` | skip: Саида hold | `line~637` |
| `wa_dialog_suppress active` | skip: suppress | `line~650` |
| `exception grace active` | skip: grace period | `line~660` |
| `do_not_notify` | skip: не беспокоить | `line~670` |
| малый остаток (amount < 100K AND amount ≤ credit×10%) | skip: значительная оплата | `_apply_collector_day_policy` |
| `active dialog state` | skip (в preview guard) | `run_approval_preview:line~2033` |
| `stop_status=stopped/auto_stopped` | `client_approval`, `stoplist_reminder` | `line~695` |
| `pending_clearance/conditional` | `manager_review`, `payment_plan_control` | `line~701` |
| `debit > 0, credit == 0` | `manager_review` | `line~720` |
| `debit > 0, credit > 0` | `client_approval` | `line~728` |
| `debit == 0, credit > 0` | `client_approval`, `partial_tail/strict` | `line~734` |
| no movement | `client_approval`, `no_movement_reminder` | `line~768` |

---

## 3. FINDINGS (Полный реестр)

### F-01 | HIGH | Exception grace bypass при отсутствии anchor date

**Symptom:** Клиент с `stop_status=exception` в стоп-реестре, но без `cleared_at` и `approved_at` — grace period не активируется. Клиент немедленно попадает в следующий batch без ожидания.

**Root cause:** `_exception_grace_active()` (`collections_engine.py:355–360`) вычисляет `anchor = cleared_at or approved_at`. При `anchor=None` функция возвращает `(False, "")` — grace считается истёкшим.

**Code location:** `collector/collections_engine.py:355–360`

**Production evidence:** Клиент "А ТД Сарыарка СКЛАД" в реестре имеет `approved_at=2026-05-12`, `cleared_at=null`, `status=exception`. На 14.05.2026 `age_days=2`, `COLLECTOR_EXCEPTION_GRACE_DAYS=2`, условие `age_days < 2` = False → grace не активна. При ручном добавлении без дат anchor не было бы вообще.

**Reproducibility:** Воспроизводится при ручном добавлении exception-клиента в registry без заполнения дат, или если `cleared_at` не выставляется при снятии стопа.

**Impact:** Клиент с активным исключением может получить WA-давление сразу, без паузы для урегулирования. Нарушает доверие клиента и операционную логику исключения.

**Proposed fix:**
```python
if not anchor:
    anchor = datetime.now(TZ).date()  # безопаснее: grace от сегодня
```
Или добавить `cleared_at = today` при выставлении `status=exception` в debt_stop_control.py.

**Тест для закрытия:** `exception_no_anchor_always_grace_active` — клиент без дат → grace=True первые 2 дня.

---

### F-02 | HIGH | Race condition в handle_manager_proof — неверный порядок dict merge

**Symptom:** При одновременной активности двух процессов (bot + collector subprocess) пересылка фото менеджера может затереть свежие изменения batch-state, сохранённые другим процессом.

**Root cause:** `approval_flow.py:2791`:
```python
_save_batches({state["batch_id"]: batch} | _load_batches())
```
Левый операнд (in-memory копия batch) перезаписывает версию из файла (правый). Правильный порядок: свежие данные файла имеют приоритет.

**Code location:** `collector/approval_flow.py:2791`

**Production evidence:** Теоретически возможен при photo callback во время admin approval window. В боевых логах явных следов нет, но это narrow timing window.

**Reproducibility:** Воспроизводится при искусственной задержке между read и write.

**Impact:** При race условии: потеря admin_ok или admin_send action → батч не будет отправлен несмотря на нажатую кнопку директора.

**Proposed fix:**
```python
_save_batches(_load_batches() | {state["batch_id"]: batch})
```

**Тест:** `test_manager_proof_no_overwrite_concurrent` — имитирует параллельную запись в batch.

---

### F-03 | HIGH | Race condition в saida_payment_holds.json — нет межпроцессного лока

**Symptom:** Параллельный вызов `create_manager_payment_request()` (из bot callback) и `sync_holds_with_debtors()` (из collector subprocess) — один процесс перезаписывает изменения другого.

**Root cause:** `payment_hold.py:48–77` — паттерн read-modify-write без `filelock`. `os.replace()` атомарен, но каждый процесс работает со своей in-memory копией.

**Code location:** `collector/payment_hold.py:48–77`

**Production evidence:** `saida_payment_holds.json` изменяется одновременно bot-процессом (при нажатии кнопки) и collector-субпроцессом (при `sync_holds_with_debtors` в 17:00). Вероятность столкновения — реальная при ежедневном preview.

**Reproducibility:** Воспроизводится при искусственной задержке в `_save()`.

**Impact:** Потеря hold-записи → клиент больше не заблокирован → повторное WA-давление по уже обработанному долгу.

**Proposed fix:**
```python
from filelock import FileLock
_LOCK_PATH = PAYMENT_HOLD_PATH.with_suffix(".lock")

def _save(data):
    with FileLock(str(_LOCK_PATH), timeout=5):
        # ... existing NamedTemporaryFile + os.replace logic
```

**Тест:** `test_payment_hold_concurrent_write_safe`.

---

### F-04 | HIGH | Нет тестов для zeropay confirm/deny flow

**Symptom:** Функции `_handle_saida_zeropay_confirm` и `_handle_saida_zeropay_deny` (`debt_stop_control.py:2372–2410`) — критический path снятия клиента со стопа при долге ≤100 тг — без единого теста.

**Root cause:** Задача упомянута в SESSION_CONTEXT.md трижды как "открытая", но никогда не закрыта.

**Code location:** `bot/debt_stop_control.py:2372–2410`; `tests/test_collector.py` — строка с zeropay отсутствует.

**Production evidence:** SESSION_CONTEXT.md (три handoff подряд): "Тесты для `_handle_saida_zeropay_confirm/deny` — открыто".

**Reproducibility:** Любой рефакторинг zeropay-flow пройдёт незамеченным.

**Impact:** Регрессия в zeropay flow не будет обнаружена тестами. Клиент с остатком ≤100 тг может зависнуть в stop-листе или быть неправильно снят.

**Proposed fix:** Добавить секцию в `test_collector.py` по образцу `DSTOP SAIDA FULL T1`:
```python
section("ZEROPAY T1 — Саида подтверждает нулевой остаток")
section("ZEROPAY T2 — Саида отклоняет нулевой остаток")
```

---

### F-05 | MEDIUM | pending_saida не блокирует клиента в collector shortlist

**Symptom:** Клиент, по которому менеджер заявил "оплатил без документа" (hold создан, статус `pending_saida`), может попасть в следующий collector batch пока Саида не ответила.

**Root cause:** `ACTIVE_STATUSES = {"confirmed_full", "confirmed_partial"}` в `payment_hold.py:32` не включает `pending_saida`. Проверка `get_hold_for_client()` в `_collector_candidate_decision` и `_send_approved_client` возвращает None для pending_saida записей.

**Code location:** `collector/payment_hold.py:32`, `collector/collections_engine.py:637–647`, `1507–1514`

**Production evidence:** В `saida_payment_holds.json` 26 записей с `status=pending_saida`. При daily preview в 17:00 эти клиенты теоретически не заблокированы. Фактически не срабатывало из-за active dialog guard (dialog state=active после первого WA), но при отсутствии или устаревании диалога — срабатывает.

**Impact:** Повторное WA-давление на клиента, по которому идёт разбирательство с Саидой. Создаёт конфликт между collector и payment flow.

**Proposed fix:**
```python
BLOCK_STATUSES = {"pending_saida", "confirmed_full", "confirmed_partial"}
# Использовать в get_hold_for_client() вместо ACTIVE_STATUSES для блокировки
```

**Тест:** `test_pending_saida_blocks_collector_shortlist`.

---

### F-06 | MEDIUM | Admin UI: send_results показывает 0/N вместо K/N

**Symptom:** После отправки батча кнопка "Обновить" в Telegram показывает "Результат отправки: 0/3 доставлено" вместо реального числа.

**Root cause:** `send_reports.py:3248`:
```python
sent_ok = sum(1 for r in send_results if r.get("sent"))
```
Результат записывается как `{"status": "sent", "name": ..., "reason": ...}`, но не `{"sent": True, ...}`. Key `"sent"` в результатах коллектора — string status, не bool.

**Code location:** `bot/send_reports.py:3248`

**Production evidence:** Визуально видно в боевом UI после каждой отправки.

**Impact:** Директор не видит реального результата отправки. Высокий UX-риск — может принять неверное решение о повторной отправке.

**Proposed fix:**
```python
sent_ok = sum(1 for r in send_results if r.get("sent") or r.get("status") == "sent")
```

---

### F-07 | MEDIUM | paid_claim: зомби-диалог без эскалации и таймаута

**Symptom:** Клиент сказал "уже оплатил", бот ответил "пришлите чек", state=`awaiting_payment_proof`. Если клиент больше ничего не пишет — диалог остаётся в `awaiting_payment_proof` бессрочно. Через 3 дня suppress истекает, но dialog state `awaiting_payment_proof` входит в `_DIALOG_ACTIVE_STATES` — collector guard блокирует повторную отправку. После очистки/устаревания диалога — клиент получит новое WA.

**Root cause:** В `paid_claim` ветке (`client_dialog.py:1065–1087`) нет вызова `escalate_to_manager` и нет TTL на `awaiting_payment_proof`.

**Code location:** `collector/client_dialog.py:1065–1087`

**Production evidence:** `collector_client_dialogs.json` содержит диалог с `state=awaiting_payment_proof` без `responded_at` — зависший несколько дней.

**Impact:** Менеджер видит notify, но диалог не в "active escalation" — нет гарантии что менеджер отреагирует. Клиент с реально оплаченным долгом продолжает числиться в active pipeline.

**Proposed fix:**
```python
# В paid_claim ветке — добавить escalation:
await escalate_to_manager(dialog, "paid_claim", "Клиент утверждает что оплатил — ждём чек", phone_clean)
```
Либо добавить TTL в `_send_approved_client`: если `awaiting_payment_proof` старше N дней → переводить в escalated.

---

### F-08 | MEDIUM | off_topic_count не сбрасывается — ложная ранняя эскалация

**Symptom:** Клиент отвечает `unclear` (count=1), потом `soft_positive` (count остался 1), потом снова `unclear` — сразу эскалация. По дизайну нужно 2 consecutive unclear.

**Root cause:** `off_topic_count` инкрементируется в unclear-ветке и персистируется в dialog dict. В других ветках (promise, paid_claim и т.д.) counter не сбрасывается.

**Code location:** `collector/client_dialog.py:1277–1300`

**Impact:** Ложная эскалация = менеджер получает лишнее уведомление по клиенту, который реально общается. Ухудшает качество общения.

**Proposed fix:** В начале каждой не-unclear ветки добавить `dialog["off_topic_count"] = 0`.

---

### F-09 | MEDIUM | wa_appr_adm_later не меняет batch.status — тихое вытеснение

**Symptom:** Директор откладывает батч (`wa_appr_adm_later`). Статус батча остаётся `pending_admin`. При следующем запуске `--preview` → `supersede_batch()` тихо переводит отложенный батч в `superseded`. Директор не получает предупреждения.

**Root cause:** `approval_flow.py:2220–2230` выставляет только `admin_status = "postponed"`, но не меняет `batch["status"]`. `load_latest_batch()` видит `pending_admin` как активный — при создании нового батча вызывается `supersede_batch`.

**Code location:** `collector/approval_flow.py:2220–2230`

**Impact:** Директор считает что батч ожидает, а он уже вытеснен. Потеря управляемости — батч исчезает без действия.

**Proposed fix:** Либо выставить `batch["status"] = "postponed"` как отдельный нефинальный статус, либо при создании нового батча проверять `admin_status == "postponed"` и уведомлять директора.

---

### F-10 | MEDIUM | Visual ghost: agreed/paid клиенты показываются как нерассмотренные

**Symptom:** В подробном списке директора (`_format_admin_detail_text`) клиенты, снятые менеджером через `agreed`/`paid`, показываются с иконкой `◯` (нерассмотренный), хотя менеджер их обработал.

**Root cause:** `approval_flow.py:1543–1575` проверяет только `approved_set`, `rejected_set`, `postponed_set`. Поля `agreed_names`, `paid_with_doc_names`, `paid_no_doc_names` не проверяются.

**Code location:** `collector/approval_flow.py:1543–1575`

**Impact:** Директор видит искажённую картину ответов менеджеров.

**Proposed fix:**
```python
agreed_set = set(mgr_state.get("agreed_names", []))
paid_set = set(mgr_state.get("paid_with_doc_names", [])) | set(mgr_state.get("paid_no_doc_names", []))
# в цикле:
elif name in agreed_set: icon = "🤝"
elif name in paid_set: icon = "💰"
```

---

### F-11 | MEDIUM | is_ready_for_send=True для "sent" батча — риск CLI повтора

**Symptom:** `is_ready_for_send()` возвращает True для батча со статусом `"sent"`. Через CLI `--send-approved --batch-id X` можно повторно запустить отправку уже отправленного батча.

**Root cause:** `approval_flow.py:2263`: `status in ("admin_approved", "partially_sent", "sent")`.

**Code location:** `collector/approval_flow.py:2263`

**Mitigation:** Через TG-кнопку защищено (проверяется статус). Через CLI — нет.

**Proposed fix:** Добавить guard в `send_approved_batch`:
```python
if batch.get("status") == "sent" and not single_client:
    return batch.get("send_results") or []
```

---

### F-12 | MEDIUM | Stale contact в no_movement branch — неверный менеджер

**Symptom:** В `run()` (dry-run path) при первой итерации цикла no_movement-ветка обращается к переменной `contact`, которая ещё не объявлена (`contact = match_client(...)` идёт позже). На первой итерации → NameError; для 2+ итераций — берётся contact предыдущего клиента → запрос Саиде уходит с неверным менеджером.

**Root cause:** `collections_engine.py:1399`: использование `contact` до `collections_engine.py:1418` где оно объявляется.

**Code location:** `collector/collections_engine.py:1399, 1418`

**Impact:** В dry-run — неверный менеджер в no_movement уведомлении Саиде. NameError перехватывается `except Exception` и логируется тихо.

**Proposed fix:** Переместить `contact = match_client(name, contacts)` выше no_movement check.

---

### F-13 | MEDIUM | manager_chat_id guard отсутствует в send-approved refresh

**Symptom:** `run_approval_preview` пропускает клиентов без `manager_chat_id`. `_prepare_current_approved_clients` этой проверки не имеет. Клиент, добавленный через sticky_auto_clients с менеджером без chat_id, дойдёт до отправки WA без Telegram-уведомления менеджеру.

**Code location:** `collector/collections_engine.py:2062–2069` vs `1648–1708`

**Impact:** Менеджер не получает уведомления об отправке WA своему клиенту. Low risk на корректность, high risk на наблюдаемость.

---

### F-14 | LOW | payment_deferrals cache не инвалидируется в runtime

**Symptom:** Изменение `config/payment_deferrals.json` во время работы процесса (добавили нового клиента с отсрочкой) не подхватывается без перезапуска бота.

**Code location:** `collector/payment_deferrals.py:42–57`

**Proposed fix:** Добавить mtime-based cache invalidation:
```python
_cache_mtime: float = 0.0
# проверять stat().st_mtime перед возвратом _cache
```

---

### F-15 | LOW | Greeting loop — бесконечные одинаковые ответы

**Symptom:** Клиент пишет "Здравствуйте" несколько раз подряд — бот каждый раз отвечает одним и тем же текстом. `off_topic_count` не инкрементируется в greeting-ветке. Только при `exchange_count≥5` сработает limit_reached.

**Code location:** `collector/client_dialog.py:858–863`

**Proposed fix:** Инкрементировать `off_topic_count` в greeting-ветке. При `off_topic_count ≥ 1` — эскалировать.

---

### F-16 | LOW | promise_without_date — нет выхода при длинных расплывчатых ответах

**Symptom:** При `promise_without_date` с длинным текстом (>18 символов) бот бесконечно просит дату без перехода к эскалации. Нет порога `exchange_count ≥ 2` как в soft_positive.

**Code location:** `collector/client_dialog.py:1251–1274`

**Proposed fix:** Добавить `if exchange_count >= 2` → escalate по аналогии с soft_positive.

---

### F-17 | LOW | Мёртвые ё-паттерны в _SERVICE_REQUEST_PATTERNS

**Symptom:** Паттерны `\bсчёт\b` (строки 378, 380 в client_dialog.py) никогда не совпадут — `_normalize_text` заменяет `ё→е` на строке 340.

**Code location:** `collector/client_dialog.py:378, 380`

**Proposed fix:** Удалить дублирующие ё-паттерны.

---

### F-18 | LOW | Двойной вызов _apply_deferral_metrics

**Symptom:** Для каждого клиента в preview и send-approved refresh вызывается `_apply_deferral_metrics` дважды: один раз снаружи, второй раз внутри `_collector_candidate_decision:599`.

**Code location:** `collections_engine.py:599, 1982, 1654`

**Impact:** Незначительный (cache в `_load()`). Создаёт dict дважды.

---

### F-19 | LOW | Admin preview не видит pending_saida hold при одобрении клиента

Следствие F-05. Директор может одобрить батч с клиентом, по которому параллельно ожидается ответ Саиды. При отправке — клиент получит WA несмотря на pending hold. Исправляется вместе с F-05.

---

## 4. РЕЕСТР ПОВТОРНЫХ КЛИЕНТОВ

| Клиент | Path повторного захода | Штатно или баг | Guard не сработал |
|--------|----------------------|----------------|-------------------|
| Клиент с `awaiting_payment_proof` | После 3 дней suppress — guard по state работает; после очистки диалога — снова в shortlist | Баг при очистке | paid_claim не escalate (F-07) |
| Клиент с `pending_saida` hold | Следующий daily preview — `pending_saida` не в ACTIVE_STATUSES | Баг | BLOCK_STATUSES не включает pending_saida (F-05) |
| Клиент с `exception` без anchor | После `status=exception` — grace не считается, сразу в batch | Баг | anchor date not set (F-01) |
| Клиент после sticky approval | Следующий день, signature не изменилась — auto-approved | Штатно | sticky механизм работает правильно |
| Клиент после escalation | suppress 2 дня → затем active guard по dialog state | Штатно если dialog state=escalated | active dialog guard работает |
| Клиент с `wa_dialog_suppress` | После истечения suppress → снова в shortlist | Штатно | suppress — ограниченный по времени guard |

**Ответ на главный вопрос ТЗ — "почему старьё снова полезло в WhatsApp":**

На 13.05.2026 из 86 должников 7 клиентов были пропущены из-за `active dialog state` — это **штатно**. Потенциальные пути повторного захода:
1. `pending_saida` hold не блокирует (F-05) — реальный risk
2. `exception` без anchor dates (F-01) — реальный risk
3. `paid_claim` зомби-диалог после устаревания (F-07) — реальный risk

---

## 5. РЕЕСТР STALE/STUCK BATCH ПРОБЛЕМ

| Проблема | Механизм | Статус |
|---------|----------|--------|
| **Preview/send mismatch: msg_type** | При refresh пересчитывается msg_type → клиент получает `strict_reminder` вместо согласованного `payment_plan_control` | Архитектурно, notify есть, но не в approval UI |
| **Stale manager callbacks** | Защищены: `status != pending_managers` → отклонение + снятие кнопок | Исправлено в v1.1.12 ✓ |
| **Admin-approved but unsendable** | Если all clients stale → `send_empty`. Admin не видит причин skip в UI | Operational gap (F-19) |
| **wa_appr_adm_later batch** | Батч остаётся `pending_admin`, тихо вытесняется при следующем preview | Баг F-09 |
| **sent batch re-send via CLI** | `is_ready_for_send("sent")=True` → CLI без защиты | Баг F-11 |
| **Supersede без уведомления** | При supersede директор не получает "ваш отложенный батч вытеснен" | Следствие F-09 |

---

## 6. РЕЕСТР SAIDA/STOP OVERLAP

| Клиент | Почему не в batch | Видно ли в UI |
|--------|------------------|---------------|
| М Плов центр ЕСБОЛОВА | `confirmed` hold от Саиды | Только в `collector_saida_stats` кнопке |
| М Гриль Косши | `confirmed` hold от Саиды | Только в `collector_saida_stats` кнопке |
| ~26 клиентов в pending_saida | Hold создан, Саида не ответила | НЕТ в превью батча, НЕТ в skip list |
| О ТОО Petro Retail | active dialog state (`escalated`) | НЕТ — только в collector.log |
| А ТД Сарыарка СКЛАД | `exception`, grace зависит от даты | Только в стоп-реестре, не в collector UI |

**Вывод:** Видимость overlap недостаточная. Admin-панель "🤖 Коллектор" не содержит:
- Сколько клиентов пропущено и почему
- Каких клиентов заблокировал pending_saida hold
- Каких клиентов остановил active dialog

Это ответ на вопрос ТЗ **"куда исчезли остальные клиенты из вчерашней рассылки"** — они не исчезли, они пропущены по логичным причинам, но это не видно в интерфейсе.

---

## 7. ТЕСТОВОЕ ПОКРЫТИЕ

### Что уже защищено (587/588 тестов)
- `classify_debtors`, `_level_for_days` (базовая логика) ✓
- `collections_db` (CRUD, promises, escalated marks) ✓
- Approval flow (create/save/load, manager callbacks, admin keyboard, supersede, expire) ✓
- `_collector_candidate_decision` для всех stop-статусов (Phase 4 tests) ✓
- Exception grace period guard ✓
- wa_dialog_suppress guard ✓
- Sticky approval (Section 26) ✓
- deferral_level (Section 28a), deferral discipline (28b), deferred presentation (28c) ✓
- approval_penalty с partial/full/timeout ✓
- client_dialog: paid_claim, soft_positive, service_request, off_topic escalation ✓
- Saida payment holds (confirm/partial/stats/parse) ✓
- manager_chat_id guard (HIGH-3) ✓
- msg_type preserved preview→send (HIGH-4) ✓
- send-approved freshness gate ✓

### Критические пробелы (обязательно добавить)

| # | Что не покрыто | Приоритет |
|---|----------------|-----------|
| T1 | `_handle_saida_zeropay_confirm/deny` — полный flow | P0 |
| T2 | Exception grace при `anchor=None` → grace_active=True | P1 |
| T3 | `pending_saida` blocks collector shortlist | P1 |
| T4 | `paid_claim` зомби — awaiting_payment_proof без таймаута/escalation | P1 |
| T5 | `off_topic_count` сброс при смене intent | P2 |
| T6 | `wa_appr_adm_later` + следующий preview → supersede без уведомления | P2 |
| T7 | `send_results` key schema (status=="sent" vs sent=True) | P2 |
| T8 | `handle_manager_proof` dict merge order | P2 |
| T9 | `run_approval_preview` 0 кандидатов → батч не создаётся | P3 |
| T10 | Sticky + supersede → sticky cleared для клиентов не в новом batch | P3 |

### Хрупкие тесты
- `DATE-PARSE T4` ("в пятницу") — зависит от текущего дня недели
- SHA-256 watcher (Section 23) — если тест упал без restore, боевой файл остаётся изменённым

---

## 8. ПРИОРИТИЗИРОВАННЫЙ REMEDIATION PLAN

### P0 — Немедленно (до следующего production цикла)

| # | Файл:строка | Действие |
|---|-------------|----------|
| F-06 | `send_reports.py:3248` | Исправить `r.get("sent")` → `r.get("sent") or r.get("status") == "sent"` |
| F-05 | `payment_hold.py:32` | Добавить `pending_saida` в BLOCK_STATUSES |
| F-07 | `client_dialog.py:1065–1087` | Добавить `escalate_to_manager` в paid_claim ветке |

### P1 — Высокий приоритет (в этом спринте)

| # | Файл:строка | Действие |
|---|-------------|----------|
| F-01 | `collections_engine.py:355–360` | Fallback anchor = today при отсутствии дат |
| F-02 | `approval_flow.py:2791` | Поменять порядок dict merge |
| F-03 | `payment_hold.py:48–77` | Добавить FileLock |
| F-04 | `debt_stop_control.py:2372` | Написать тесты T1, T2 zeropay |
| F-08 | `client_dialog.py:1277–1300` | Сбрасывать off_topic_count при смене intent |
| T3 | `test_collector.py` | Тест pending_saida blocks collector |

### P2 — Средний приоритет

| # | Файл:строка | Действие |
|---|-------------|----------|
| F-09 | `approval_flow.py:2220–2230` | Уведомлять директора при supersede отложенного батча |
| F-10 | `approval_flow.py:1543–1575` | Показывать agreed/paid иконки в detail view |
| F-11 | `approval_flow.py:2263` | Guard на re-send sent batch через CLI |
| F-12 | `collections_engine.py:1399` | Перенести contact = match_client() выше no_movement |
| F-13 | `collections_engine.py:2062` | Лог-предупреждение при no chat_id в send-approved |
| UI | `collections_engine.py` + `send_reports.py` | Сохранять skip_summary в батч, показывать в UI |

### P3 — Низкий приоритет / при оказии

| # | Файл:строка | Действие |
|---|-------------|----------|
| F-14 | `payment_deferrals.py:42–57` | mtime-based cache invalidation |
| F-15 | `client_dialog.py:858–863` | Greeting loop → off_topic_count++ |
| F-16 | `client_dialog.py:1251–1274` | promise_without_date exchange_count exit |
| F-17 | `client_dialog.py:378, 380` | Удалить мёртвые ё-паттерны |
| F-18 | `collections_engine.py:599` | Убрать двойной вызов deferral_metrics |
| F-19 | `approval_flow.py` | pending_saida предупреждение в admin preview |
| T-heuristic | `test_collector.py` | Тесты DATE-PARSE T4 защитить от дня недели |

---

## 9. ОТВЕТЫ НА КЛЮЧЕВЫЕ ВОПРОСЫ ТЗ

**Как именно collector формирует shortlist?**
→ `classify_debtors()` → `_apply_deferral_metrics()` → `_collector_candidate_decision()` фильтрует по 10+ условиям. Полная карта в разделе 2.2.

**Какие клиенты идут в preview, какие исключаются?**
→ 13.05.2026: 86 должников → 1 в batch. 7 пропущены из-за active dialog, 2 Саида hold, 1 малый остаток.

**Может ли утверждённый клиент пропасть?**
→ Да. Нормально: погашен долг, появился hold, suppress, already_contacted. Ненормально: exception grace активирован ПОСЛЕ утверждения — клиент пропадёт с generic "stale approved batch".

**Почему один и тот же клиент снова в WhatsApp?**
→ Три реальных пути: (1) pending_saida не блокирует (F-05), (2) exception без anchor date (F-01), (3) paid_claim зомби после expire диалога (F-07).

**Где заканчивается collector и начинается контур Саиды?**
→ Граница: `ACTIVE_STATUSES` в payment_hold.py. При `confirmed_full/partial` — Саида взяла контроль. При `pending_saida` — формально коллектор ещё не снят (баг F-05).

**Врут ли bot/UI-сообщения?**
→ Да, конкретно: `send_results` всегда показывает `0/N доставлено` (F-06).

**Устойчивы ли state-файлы к гонкам?**
→ Нет: `payment_hold.py` без file lock (F-03), `approval_flow.py:2791` неверный dict merge (F-02).

**Расхождения код/тесты/логи?**
→ Zeropay flow — в коде есть, в тестах нет (F-04). pending_saida блокировка — в концепции есть, в коде нет (F-05).

---

## 10. КРИТЕРИИ ПРИЁМКИ АУДИТА

- [x] Восстановлена фактическая схема работы collector end-to-end (раздел 2)
- [x] По каждому багу — прямое доказательство с кодом и строкой (раздел 3)
- [x] По каждому спорному кейсу — вывод "баг / штатно / business rule gap" (раздел 4)
- [x] Отдельно описано почему "старьё снова полезло в WhatsApp" (раздел 4, F-05, F-01, F-07)
- [x] Отдельно описано куда исчезли "остальные клиенты" (раздел 5, 6)
- [x] Отдельно описано какие UI-кнопки/экраны реально нужны (раздел 6 + F-19 + UI gap)
- [x] Сформирован список точечных исправлений в порядке приоритета (раздел 8)
- [x] Все выводы опираются на текущий код HEAD и текущее состояние проекта

---

*Аудит проведён: 2026-05-14. Основан на HEAD master (0a22192), боевых логах 2026-05-13, state-файлах на 14.05.2026.*

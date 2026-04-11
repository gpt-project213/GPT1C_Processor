# FULL COLLECTOR PRODUCTION AUDIT — 2026-04-12

Author: Codex (GPT-5 coding agent), 2026-04-12 session.

## Executive Summary

The system is safer than before the WhatsApp incident, but it is not ready for
full production live-send yet.

Current status:

- Controlled `--preview` is usable for shortlist review.
- WhatsApp is currently disabled in `.env`: `WHATSAPP_ENABLED=0`.
- Core tests pass: `tests/test_collector.py` -> `180/180`.
- Syntax checks pass for the audited changed modules.
- The main production blocker is that live `--send` is not bound to an
  admin-approved batch. It recalculates candidates using older filters.
- A second live-send path still exists through `collector.manager_dialog`
  callbacks (`col_*`). It checks `WHATSAPP_ENABLED` indirectly through
  `send_whatsapp()`, but does not check `LIVE_SEND_ALLOWED` or the new
  `wa_appr_*` approval batch.

Conclusion:

- OK for audit, dry-run, shortlist rebuild, and safe preview.
- Not OK for unrestricted production.
- First controlled send should be only after Phase 2: send only from
  `approved_clients` in an admin-approved batch.

## Git / Workspace State

Committed baseline:

- Latest pushed commit: `aeac488 Fix collector preview shortlist rules`.

Uncommitted files still present at audit time:

- Modified: `audit/INDEX.md`
- Modified: `bot/send_reports.py`
- Modified: `collector/client_dialog.py`
- Modified: `collector/collection_agent.py`
- Modified: `collector/collections_db.py`
- Modified: `tests/test_collector.py`
- Untracked: `.codex`
- Untracked: `COLLECTOR_CLASSIFICATION_RECONCILIATION_2026-04-12.md`
- Untracked audit docs:
  - `audit/CODEX_SUPPORT_CONTEXT_2026-04-11.md`
  - `audit/COLLECTOR_BUSINESS_RULES_CLARIFICATION_2026-04-11.md`
  - `audit/COLLECTOR_CLASSIFICATION_AUDIT_2026-04-11.md`
  - `audit/COLLECTOR_SHORTLIST_EXPLAINED_2026-04-11.md`
  - `audit/SAIDA_PAYMENT_FLOW_AUDIT_2026-04-11.md`
  - `audit/SHORTLIST_REFRESH_BLOCKER_FOR_CLAUDE_2026-04-11.md`
  - `audit/WHATSAPP_GREETING_UX_AUDIT_2026-04-11.md`
  - `audit/manager_lists/`
- Untracked: `config/collector_prompts.json`
- Untracked: `ТЗ.txt`

Risk:

- Production-critical files are dirty outside the latest commit.
- Before live launch, either commit the intended dirty changes or split them
  into reviewed commits. Do not launch from an ambiguous worktree.

## Environment / Runtime Switches

Current `.env` collector-related flags observed:

- `WHATSAPP_ENABLED=0`
- `COLLECTOR_HOUR_START=9`
- `COLLECTOR_HOUR_END=18`
- `COLLECTOR_DRY_RUN=false`
- `LIVE_SEND_ALLOWED` was not present in the matching lines.
- `TEST_MODE` was not present in the matching lines.

Effect:

- Current scheduled collector cannot send WhatsApp because
  `WHATSAPP_ENABLED=0`.
- If someone sets both `WHATSAPP_ENABLED=1` and `LIVE_SEND_ALLOWED=1`,
  `collector.collections_engine --send` becomes technically allowed, but it
  still does not use the admin-approved batch.

## Current Data Snapshot

Read-only audit snapshot:

- Debt JSON files loaded: `8`
- Merged clients: `419`
- Classified debtors: `93`
- Initial level 1-5 debtors from 1C classification: `13`
- Effective preview level 1-5 debtors after `first_seen` adjustment: `17`
- Contact records loaded: `552`
- Stop registry records: `38`
- Stop statuses:
  - `stopped`: `12`
  - `auto_stopped`: `4`
  - `conditional`: `1`
  - `exception`: `20`
  - `cleared`: `1`
- Active manager dialogs: `0`
- Approval batches: `0`
- Contacts with empty phone: `331`
- Contacts using placeholder/test phone `+77001234567`: `4`

Important data issue:

- `О ТД Артем.мясной зал.Кус Вкус тел 87023069994` currently resolves to
  `+77001234567`, which looks like a test placeholder, despite the name
  containing another phone. This client must not be used for production until
  the phone is corrected.

## Current Effective Preview Shortlist

Effective preview logic after state/day recalculation:

- `client_approval`: `7`
- `no_phone`: `9`
- `skip`: `1`
- Total level 1+ rows considered: `17`

By manager:

- Алена: `1` candidate with contact, `2` no-phone, `1` skip
- Ергали: `5` candidates with contact, `3` no-phone
- Магира: `4` no-phone
- Оксана: `1` candidate with contact

Candidates with contact:

| Client | Manager | Type | Stop status | Reason |
|---|---|---:|---|---|
| Е Олжас | Ергали | `stoplist_reminder` | `auto_stopped` | debt remains open |
| Е ТОО ГудФуд № 1 ул Досмухамедулы 48(Аида) | Ергали | `stoplist_reminder` | `stopped` | debt remains open |
| Е ИП Шахин | Ергали | `stoplist_reminder` | `stopped` | debt remains open |
| Е ИП Трое Нурлан | Ергали | `strict_reminder` | none | 31d, no payments/shipments |
| А Денис (Александровка) ИП DANIS | Алена | `stoplist_reminder` | `stopped` | debt remains open |
| Е Еркебулан | Ергали | `payment_plan_control` | none/exception in registry | small partial payment, large balance |
| О ТД Артем.мясной зал.Кус Вкус тел 87023069994 | Оксана | `soft_reminder` | exception | phone looks like test placeholder |

No-phone rows:

- А Ресторан Tangirs ТОО GrandRest Ак мешет 1
- Е ТД Саянур Леонид
- М Ресторан Шама ИП Тян ул Мустафина 12
- А Салават (Александровка)
- М Маг Халал маркет. ул.Косшыгулулы 20
- Е ИП Трое Кайрат
- Е ИП Мокроусов Олег (Лика)
- М Плов центр ЕСБОЛОВА Сатпаева 20
- М Мой мясной ул Умай Ана 14 тел 87789251811

Skipped:

- А ТД Шапагат 5 павильон Дюсембина:
  small tail after large payment: credit `1 324 594` тг, balance `64 415` тг.

## Critical Findings

### CRITICAL-1: live `--send` is not tied to the approved batch

Files:

- `collector/approval_flow.py`
  - `get_approved_clients()` exists.
  - Admin approval stores `approved_clients`.
- `collector/collections_engine.py`
  - `run()` still recalculates from latest debt JSON.
  - `run()` still uses old stop-list and active-client filters.
- `collector/approval_flow.py`
  - Admin text still tells user to run `python -m collector.collections_engine --send`.

Observed behavior:

- Preview currently finds `7` contactable candidates.
- Live old filters would process only `1` candidate:
  `Е ИП Трое Нурлан`.
- Live old filters would skip:
  - `8` stop-list clients
  - `8` active/payment clients

Impact:

- Admin approval does not guarantee that approved clients are the clients who
  will be sent.
- Clients approved in preview can be skipped in live send.
- Clients not shown in preview can appear if live filters diverge later.

Required fix:

- Add explicit send mode by batch:
  `python -m collector.collections_engine --send-approved --batch-id <id>`
- The send function must read `approval_flow.get_approved_clients(batch_id)`.
- It must not recalculate the shortlist.
- After sending, batch should be marked `sent` / `partially_sent` with per-client
  results.

### CRITICAL-2: old `manager_dialog` can still send WhatsApp outside new approval batch

Files:

- `bot/send_reports.py` routes `col_*` callbacks to
  `collector.manager_dialog.handle_callback()`.
- `collector/manager_dialog.py` calls `_send_whatsapp_and_notify()` on:
  - manager confirm
  - data confirmed
  - admin send
- `_send_whatsapp_and_notify()` calls `send_whatsapp()`.

Protection present:

- `send_whatsapp()` checks `WHATSAPP_ENABLED` at call time.

Protection missing:

- No `LIVE_SEND_ALLOWED` check in `manager_dialog`.
- No `wa_appr_*` batch check.
- No admin-approved batch requirement.
- No allowed-time check at callback send time.

Impact:

- If `WHATSAPP_ENABLED=1`, old pending `col_*` dialogs/buttons can trigger
  WhatsApp sends outside the new final approval process.

Required fix:

- Either disable `col_*` WhatsApp sending before production, or add the same
  hardguard:
  - `WHATSAPP_ENABLED=1`
  - `LIVE_SEND_ALLOWED=1`
  - working day/time
  - optional admin-approved batch relation
- For first test, prefer disabling old `manager_dialog` live-send path and use
  only the new approved-batch path.

### CRITICAL-3: scheduled daily collector can run live if flags are enabled

Files:

- `bot/send_reports.py`
  - `debt_collector_daily()` chooses `--send` when `WHATSAPP_ENABLED=1` and
    `COLLECTOR_DRY_RUN=false`.
  - Scheduler runs it at `17:30`.

Protection present:

- `collector.collections_engine.run()` blocks live unless
  `LIVE_SEND_ALLOWED=1`.

Risk:

- Once both flags are enabled for a test, the daily scheduler can also run
  `--send` at 17:30 and recalculate using old live filters.

Required fix:

- Before test send, set `COLLECTOR_DRY_RUN=true` for scheduled job or disable
  `debt_collector_daily`.
- For production, scheduler should run `--preview`, not `--send`, unless there
  is a specific approved batch id.

## High Findings

### HIGH-1: approval manual mode can become "responded" too early

File:

- `collector/approval_flow.py`

Problem:

- When manager clicks manual mode, status becomes `manual` immediately.
- `_all_managers_responded()` treats only `status == "pending"` as not
  responded.
- If one manager opens manual mode but does not press done, and all other
  managers respond, the batch can move to `pending_admin` with an incomplete
  manual selection.

Required fix:

- Use a separate state like `manual_editing`.
- Treat responded only when `responded_at` is not `None`.
- Or make `_all_managers_responded()` require status in:
  `approved_all`, `rejected_all`, `manual_done`, `timeout`.

### HIGH-2: batch id has minute precision and can collide

File:

- `collector/approval_flow.py`

Problem:

- `batch_id = now.strftime("%Y%m%d-%H%M")`.
- Two previews in the same minute overwrite the same batch id.

Required fix:

- Use seconds and a short random suffix:
  `YYYYMMDD-HHMMSS-xxxx`.

### HIGH-3: manager without Telegram chat id can block the batch forever

File:

- `collector/approval_flow.py`

Problem:

- `send_manager_previews()` logs missing chat id and continues.
- That manager remains `pending`.
- `_all_managers_responded()` will never become true.

Required fix:

- Mark such manager as `delivery_failed` or route to admin.
- Admin summary should show "manager preview not delivered".

### HIGH-4: `msg_type` is not passed into live message generation paths

Files:

- `collector/collections_engine.py`
- `collector/manager_dialog.py`
- `collector/collection_agent.py`

Problem:

- `generate_message()` accepts `msg_type`.
- `msg_type` is only used for fallback template selection.
- Current send callers do not pass `msg_type`.
- DeepSeek prompt does not receive `msg_type` explicitly.

Impact:

- A `stoplist_reminder` approved in preview may generate a generic strict
  message in live send.

Required fix:

- Store and pass `msg_type` from approved batch into `generate_message()`.
- Include `msg_type` and `reason` in the prompt, or use deterministic templates
  for the first controlled sends.

### HIGH-5: placeholder phone numbers exist in contact data

Observed:

- `+77001234567` appears in 4 contact records.
- One of the current candidates, `О ТД Артем...`, uses that phone.

Required fix:

- Treat known test/placeholder phones as invalid in production shortlist.
- Add validation/warning:
  - `+77001234567`
  - repeated fake numbers
  - numbers inconsistent with phone embedded in client name.

## Medium Findings

### MEDIUM-1: CRM daily task is scheduled after collector daily

Files:

- `bot/send_reports.py`

Observed:

- `debt_collector_daily`: `17:30`
- `crm_daily_task`: `18:00`
- Comment says collector is "after CRM", but actual schedule is before CRM.

Impact:

- Contacts/manager mappings may be stale during collector run.

Recommendation:

- Run CRM update before preview/collector, or make collector explicitly call a
  CRM refresh in dry-run/preview.

### MEDIUM-2: stop registry debt values can be stale

Observed examples:

- Олжас registry debt: `1 236 305`, current preview debt: `830 782`.
- Денис registry debt: `486 922`, current preview debt: `86 922`.

Impact:

- Stop-list status is useful, but debt amount in registry should not be trusted
  for message content.

Recommendation:

- Use current debt JSON for amount.
- Display registry amount only as historical note if needed.

### MEDIUM-3: `first_seen` inflation remains partially present

Current:

- 5 state records still have `first_seen=2026-03-19`.
- Only one of them is currently in debt JSON: `А Салават (Александровка)`.
- Running `--fix-first-seen` would reset the 4 missing clients and keep
  Салават effectively at the same level based on current 1C days.

Recommendation:

- Run `python -m collector.collections_engine --fix-first-seen` before the next
  official shortlist snapshot.

### MEDIUM-4: preview excludes clients already contacted today

File:

- `collector/collections_engine.py`

Observed:

- `run_approval_preview()` calls `already_contacted_today(name)` unconditionally.

Impact:

- After a test send, a later preview the same day may hide the same client.

Recommendation:

- Keep as default, but add `--include-contacted-today` for audit/debug only.

### MEDIUM-5: test suite does not yet cover Phase 1 preview decision helper

Current:

- Existing test suite passes `180/180`.
- It still contains old incident expectations that active clients are filtered
  from collector, which is true for old live `run()` but no longer true for
  `--preview`.

Recommendation:

- Add focused tests for `_collector_candidate_decision()`:
  - `auto_stopped` included
  - `stopped` included
  - healthy small tail skipped
  - low partial payment included
  - no phone path triggers manager warning in preview

## Things You Almost Missed

1. Preview approval is not the same as live send.
   This is the biggest gap.

2. The old `col_*` manager-dialog flow still exists and can send WhatsApp
   outside the new `wa_appr_*` batch.

3. Scheduler can become dangerous as soon as both live flags are enabled.

4. At least one current candidate has a test-looking phone number.

5. Manual approval mode can let a batch advance before manual selection is
   complete.

6. Batch ids can collide if preview is run twice in one minute.

7. CRM update is scheduled after collector, despite comments saying otherwise.

8. `msg_type` is visible to managers/admin, but not reliably used for actual
   generated client text.

## Recommended Launch Plan

### Before Any Live Test

1. Keep:
   - `WHATSAPP_ENABLED=0`
   - `LIVE_SEND_ALLOWED=0` or absent

2. Commit or deliberately shelve dirty production-code changes.

3. Run:
   - `python -m collector.collections_engine --fix-first-seen`

4. Rebuild shortlist:
   - `python -m collector.collections_engine --dry-run`
   - safe `--preview` or real `--preview` only during workday

5. Remove or mark invalid placeholder phones, especially `+77001234567`.

### Required Code Fixes Before Real Production

1. Implement approved-batch sending:
   - `--send-approved --batch-id <id>`
   - reads `approval_flow.get_approved_clients(batch_id)`
   - sends only those clients
   - records per-client result in batch

2. Disable or guard old `manager_dialog` WhatsApp sends.

3. Add batch id uniqueness.

4. Fix approval manual mode state.

5. Handle missing manager chat ids as admin-visible delivery failures.

6. Pass `msg_type` into message generation.

7. Add test-phone blacklist / production phone validator.

### First Controlled Live Test

Only after the fixes above:

1. Pick one clean client.
2. Ensure:
   - valid real WhatsApp
   - correct manager
   - no active client dialog
   - not placeholder phone
   - selected in approved batch
3. Set:
   - `WHATSAPP_ENABLED=1`
   - `LIVE_SEND_ALLOWED=1`
   - `COLLECTOR_DRY_RUN=true` for scheduler safety
4. Run:
   - `python -m collector.collections_engine --send-approved --batch-id <id> --client "<name>"`
5. Immediately restore:
   - `LIVE_SEND_ALLOWED=0`
6. Verify:
   - WhatsApp delivery log
   - `logs/collector_client_dialogs.json`
   - `logs/collector_state.json`
   - no duplicate send
   - client reply routing

## Verification Performed

Commands:

- `.\.venv\Scripts\python.exe -m py_compile bot\send_reports.py collector\client_dialog.py collector\collection_agent.py collector\collections_db.py tests\test_collector.py`
- Previous run in this session:
  `.\.venv\Scripts\python.exe -X utf8 tests\test_collector.py` -> `180/180`
- Read-only data audit scripts for:
  - current debt classification
  - effective preview shortlist
  - old live filter comparison
  - stop registry statuses
  - approval batch state
  - dialog state
  - placeholder phone detection

## Final Readiness Verdict

For first controlled preview:

- Ready, with caution.

For one-client test send:

- Not yet. Needs approved-batch send or manual one-off script that sends only a
  named approved client.

For full production:

- Not ready. Main blockers are live-send/batch mismatch and the old
  `manager_dialog` send path.

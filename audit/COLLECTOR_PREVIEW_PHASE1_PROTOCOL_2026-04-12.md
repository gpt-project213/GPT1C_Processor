# COLLECTOR PREVIEW PHASE 1 PROTOCOL — 2026-04-12

Author: Codex (GPT-5 coding agent), 2026-04-12 session.

## Goal

Implement Phase 1 only: make `python -m collector.collections_engine --preview`
build the correct approval shortlist.

## Scope

- Do not remove clients from debt stop / auto stop.
- Do not expand live WhatsApp sending in this phase.
- Keep `--send` behavior conservative until Phase 2 binds it to approved batches.
- Make stop-list and active debit/credit signals visible as collector reasons in preview.

## Business Rule

Stop-list blocks shipments, not collection. If a client is `stopped` or
`auto_stopped` and debt remains open, preview should include the client as a
collector candidate or request missing contacts from the manager.

## Changes Log

### 2026-04-12

- Added this protocol file.
- Added `opening` to `collector.debt_monitor.classify_debtors()` output so
  downstream payment discipline logic can distinguish healthy payments from
  slow partial payments.
- Updated `collector.approval_flow.create_batch()` to preserve precomputed
  `msg_type`, `reason`, `stop_status`, and `review_action` from the collector
  engine instead of always recalculating the type.
- Added stop-status handling to `approval_flow._classify_msg_type_and_reason()`
  as a fallback if the engine passes `stop_status` without precomputed type.
- Added collector preview decision helpers in `collector.collections_engine.py`.
- Replaced `run_approval_preview()` silent stop-list/active-client skips with
  `_collector_candidate_decision()`.
- Missing phone/Telegram in preview now calls the existing manager warning path
  instead of dropping the client silently.
- Preview batch items now carry explicit reason metadata for manager/admin
  review: `msg_type`, `reason`, `stop_status`, `review_action`.
- Syntax check passed:
  `.\.venv\Scripts\python.exe -m py_compile collector\collections_engine.py collector\approval_flow.py collector\debt_monitor.py`.
- Regression tests passed: `.\.venv\Scripts\python.exe -X utf8 tests\test_collector.py`
  (`180/180`).
- Safe preview dry simulation was run with Telegram send/save monkeypatched off.
  Result on current data: `3` managers, `7` clients in approval batch; stop-list
  debtors such as Олжас, ГудФуд, Шахин, Денис are included with
  `stoplist_reminder`; Tangirs, Леонид/Саянур, Шама request manager phone data;
  Шапагат is skipped as a small tail after a large payment.

## Pending

- Phase 2: bind live `--send` to an admin-approved batch before expanding real
  WhatsApp sending.

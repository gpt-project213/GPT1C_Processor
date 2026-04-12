# Codex Autosave Context

Date: 2026-04-12  
Author: Codex  
Project: `E:\GPT1C_Processor_analitica`

## Current Objective

Prepare the old collector project for Monday Phase 3 controlled live precheck
and one-client live test.

Sunday live was explicitly not allowed. No preview/live command should be run on
Sunday.

## Current Repository State Target

Tracked worktree should be clean. `.codex` is ignored as a local tool artifact.

## Important Completed Work

Safe live architecture:

1. `--send-approved --batch-id <id>` is the only allowed live path.
2. Legacy `--send` is disabled.
3. Legacy `manager_dialog` WhatsApp send path is blocked.
4. Scheduler `debt_collector_daily` runs `--dry-run`.
5. Manual approval uses `manual_editing` and `manual_done`.
6. Batch ids are `YYYYMMDD-HHMMSS-xxxx`.
7. Placeholder phone `+77001234567` is blocked.
8. Per-client send results are written back to approval batch.

Launch-readiness cleanup:

1. Runtime prompt file is tracked: `config/collector_prompts.json`.
2. First message generation uses tracked prompt config with safe fallbacks.
3. Client reply handling covers `identity_question` and `promise_without_date`.
4. `fix_first_seen_inflation()` is tracked and tested.
5. Audit docs and manager shortlist context are tracked.

## Last Known Test Results

Passed:

```powershell
.\.venv\Scripts\python.exe -m py_compile bot\send_reports.py collector\client_dialog.py collector\collection_agent.py collector\collections_db.py collector\approval_flow.py collector\collections_engine.py collector\manager_dialog.py tests\test_collector.py tests\test_phase2_safe_send.py
.\.venv\Scripts\python.exe -X utf8 tests\test_collector.py
.\.venv\Scripts\python.exe -X utf8 tests\test_phase2_safe_send.py
```

Observed results:

1. `tests/test_collector.py`: `180/180`
2. `tests/test_phase2_safe_send.py`: `PHASE2 SAFE SEND TESTS PASSED`

## Key Audit Files

1. `audit/PHASE2_SAFE_SEND_2026-04-12.md`
2. `audit/DIRTY_WORKTREE_PHASE3_BLOCKER_2026-04-12.md`
3. `audit/LAUNCH_READINESS_PROTOCOL_2026-04-12.md`
4. `audit/MONDAY_PHASE3_START_RUNBOOK_2026-04-12.md`

## Monday Hard Stops

Stop immediately if:

1. tracked git status is dirty
2. backup cannot be created
3. today is not a workday
4. current time is outside 09:00-18:00
5. batch state diverges from preview state
6. manager approval is still `manual_editing`
7. `approved_clients` contains `invalid_phone`
8. candidate has active unresolved dialog
9. candidate phone is placeholder/test
10. user has not explicitly approved the one-client send

## Next Turn Start

Start with:

```powershell
git status --short
```

Then perform Phase 3 Stage 1 backup. Do not skip straight to preview.

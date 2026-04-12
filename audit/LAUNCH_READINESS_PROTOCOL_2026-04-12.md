# Collector Launch Readiness Protocol

Date: 2026-04-12  
Author: Codex  
Scope: prepare old project for Monday Phase 3 controlled live precheck

## Summary

The dirty production-code worktree blocker was resolved.

Runtime launch-readiness changes were committed and pushed. Audit/context
documents were committed separately. No live send, preview, dry-run against
production data, or `.env` modification was performed in this cleanup step.

## Commits

| Commit | Purpose |
| --- | --- |
| `1a832f3` | `fix(collector): stabilize launch readiness behavior` |
| `b6bf128` | `docs(audit): add collector launch readiness context` |

Prior safe-send architecture remains in:

| Commit | Purpose |
| --- | --- |
| `e8bdeea` | `fix(collector): add send-approved by batch` |
| `1fd7fd2` | `fix(collector): disable old manager dialog live send path` |
| `c1ff3ce` | `fix(collector): fix manual approval completion` |
| `26d3bb2` | `fix(collector): make batch ids unique` |
| `31614ae` | `fix(collector): block placeholder phones` |
| `ad965ba` | `fix(bot): keep scheduler safe from unsafe live send` |
| `ee71ef5` | `test(collector): add phase2 coverage` |
| `e70ce86` | `docs(audit): phase2 safe send validation` |

## Runtime Changes Now Tracked

1. `collector/collections_db.py`
   - `fix_first_seen_inflation()` is tracked and tested.
2. `collector/client_dialog.py`
   - Handles `identity_question`.
   - Handles `promise_without_date`.
   - Avoids storing empty bot reply on second off-topic escalation.
3. `collector/collection_agent.py`
   - Loads collector prompts from `config/collector_prompts.json`.
   - Adds prompt defaults and fallback templates.
   - Supports `msg_type`.
   - Recognizes `identity_question` and `promise_without_date`.
4. `config/collector_prompts.json`
   - Runtime prompt config is now tracked.
5. `bot/send_reports.py`
   - Scheduler cleanup around collector daily task is tracked.
   - Phase 2 safety still forces scheduler to `--dry-run`.
6. `tests/test_collector.py`
   - Regression tests now cover the above runtime behavior.

## Tests Passed

Commands run:

```powershell
.\.venv\Scripts\python.exe -m py_compile bot\send_reports.py collector\client_dialog.py collector\collection_agent.py collector\collections_db.py collector\approval_flow.py collector\collections_engine.py collector\manager_dialog.py tests\test_collector.py tests\test_phase2_safe_send.py
.\.venv\Scripts\python.exe -X utf8 tests\test_collector.py
.\.venv\Scripts\python.exe -X utf8 tests\test_phase2_safe_send.py
```

Results:

1. `py_compile`: passed.
2. `tests/test_collector.py`: `180/180`, all tests passed.
3. `tests/test_phase2_safe_send.py`: `PHASE2 SAFE SEND TESTS PASSED`.

## Repository State After Cleanup

Tracked files: clean.

Remaining untracked local artifacts:

1. `.codex`
2. `ТЗ.txt`

These were not committed because they are local/tool/user artifacts and do not
affect runtime collector behavior.

## Current Launch Verdict

Ready for Monday Phase 3 precheck: yes.

Ready for live send now: no.

Reason:

1. Today is Sunday, and owner declined Sunday exception.
2. Phase 3 still requires fresh Monday precheck, backup, env/state validation,
   preview, manager approval, admin approval, and only then one-client
   `send-approved`.

## Monday Start Point

Restart Phase 3 from Stage 0:

1. `git status --short`
2. backup state/log files
3. read-only env flag check
4. active dialog check
5. pending approval batch check
6. `--fix-first-seen` only if current state says it is still needed
7. `--dry-run`
8. `--preview`
9. manager approval
10. admin approval
11. one-client controlled live:

```powershell
python -m collector.collections_engine --send-approved --batch-id <id> --client "<client>"
```

## Hard Stops Still Active

1. Do not use legacy `--send`.
2. Do not run live outside allowed day/time.
3. Do not send without admin-approved batch.
4. Do not send to `invalid_phone`.
5. Do not send to placeholder/test phone.
6. Do not send to a client with unresolved active dialog/session.
7. Do not run more than one client in first controlled live test.

# PHASE 2 SAFE LIVE SEND ARCHITECTURE

Date: 2026-04-12  
Author: Codex  
Project: `E:\GPT1C_Processor_analitica`

## 1. What Was Unsafe

Production audit `audit/FULL_COLLECTOR_PRODUCTION_AUDIT_2026-04-12.md`
identified these live-send risks:

1. `--preview` and live `--send` were not bound to the same approval batch.
2. Legacy `collector.manager_dialog` callbacks could still send WhatsApp through
   `_send_whatsapp_and_notify()` outside `wa_appr` approval flow.
3. Scheduler could become unsafe if live WhatsApp flags were enabled.
4. Manual manager approval could complete a batch too early.
5. Batch ids used minute-level precision and could collide.
6. Placeholder/test phone `+77001234567` existed in real contact data.
7. Worktree had unrelated dirty files, so Phase 2 changes had to stay isolated.

## 2. What Changed

Commits applied for Phase 2:

| Commit | Purpose |
| --- | --- |
| `e8bdeea` | `fix(collector): add send-approved by batch` |
| `1fd7fd2` | `fix(collector): disable old manager dialog live send path` |
| `c1ff3ce` | `fix(collector): fix manual approval completion` |
| `26d3bb2` | `fix(collector): make batch ids unique` |
| `31614ae` | `fix(collector): block placeholder phones` |
| `ad965ba` | `fix(bot): keep scheduler safe from unsafe live send` |
| `ee71ef5` | `test(collector): add phase2 coverage` |

Implemented changes:

1. Added explicit live path:
   `python -m collector.collections_engine --send-approved --batch-id <id>`.
2. `send-approved` uses only `approval_flow.get_approved_clients(batch_id)`.
3. `send-approved` does not recalculate shortlist and does not run legacy
   candidate selection.
4. Per-client send result is saved back into the approval batch:
   `sent`, `failed`, `skipped`, `reason`.
5. Legacy manager-dialog WhatsApp send path now returns `False` before
   `send_whatsapp()` can be reached.
6. Manual manager flow now has distinct states:
   `manual_editing` and `manual_done`.
7. Batch ids are now `YYYYMMDD-HHMMSS-xxxx`.
8. Placeholder/test phone validator blocks `+77001234567` and marks
   invalid phones as `invalid_phone`.
9. Admin approval excludes `invalid_phone` rows from `approved_clients`.
10. Runtime send path also skips invalid phones if an old batch contains one.
11. Scheduler `debt_collector_daily` now calls `--dry-run`, not legacy `--send`.
12. Legacy CLI `--send` is disabled for Phase 2 controlled live.

## 3. Allowed Send Paths

Allowed for controlled live:

```powershell
python -m collector.collections_engine --send-approved --batch-id <approved_batch_id>
```

Optional single-client filter after admin approval:

```powershell
python -m collector.collections_engine --send-approved --batch-id <approved_batch_id> --client "<client name>"
```

Required runtime conditions still apply:

1. Batch status is `admin_approved`.
2. Batch admin status is `approved`.
3. `WHATSAPP_ENABLED=1`.
4. `LIVE_SEND_ALLOWED=1`.
5. `is_allowed_time()` returns true.
6. Client phone passes production validator.
7. Client was not already contacted today.

## 4. Forbidden Send Paths

Forbidden / blocked:

1. `python -m collector.collections_engine --send`
   - returns `1`;
   - does not enter legacy live processing.
2. `collector.manager_dialog._send_whatsapp_and_notify()`
   - returns `False`;
   - does not call `send_whatsapp()`.
3. Scheduler `debt_collector_daily`
   - invokes `collector/collections_engine.py --dry-run`;
   - does not invoke legacy `--send`.
4. Admin-approved clients with `invalid_phone`
   - are not stored in `approved_clients`;
   - are skipped at runtime if present in an old batch.

## 5. Tests Passed

Commands run:

```powershell
.\.venv\Scripts\python.exe -m py_compile collector\approval_flow.py collector\collections_engine.py collector\manager_dialog.py bot\send_reports.py tests\test_phase2_safe_send.py
.\.venv\Scripts\python.exe -X utf8 tests\test_phase2_safe_send.py
.\.venv\Scripts\python.exe -X utf8 tests\test_collector.py
```

Results:

1. `py_compile`: passed.
2. `tests/test_phase2_safe_send.py`: `PHASE2 SAFE SEND TESTS PASSED`.
3. `tests/test_collector.py`: `180/180`, all tests passed.

Additional targeted proofs run during implementation:

1. `manual_editing_all_responded=False`, `manual_done_all_responded=True`.
2. Two batch ids created in the same second did not collide.
3. `+77001234567` returned `invalid_phone:placeholder`.
4. Admin approval filtered invalid phone rows out of `approved_clients`.
5. `_send_approved_client()` skipped invalid phone before `send_whatsapp()`.
6. Scheduler proof with both live flags enabled still called `--dry-run`.
7. Legacy `--send` returned `1`.

## 6. First Controlled Live Procedure

1. Run preview on a workday:

```powershell
python -m collector.collections_engine --preview
```

2. Managers approve/reject/manual-select clients in Telegram.
3. Admin reviews summary/detail and approves the batch.
4. Confirm the approved batch id from admin message/log.
5. For the first live test, use a single client filter:

```powershell
python -m collector.collections_engine --send-approved --batch-id <approved_batch_id> --client "<client name>"
```

6. Verify batch `send_results`.
7. Only after the first result is verified, send the second approved client if
   needed with the same `--send-approved --batch-id` path.

## 7. Conditions Where Live Must Still Not Run

Do not run controlled live if any of these are true:

1. There is no admin-approved batch id.
2. Manager approvals are still pending or in `manual_editing`.
3. The chosen client has `invalid_phone`.
4. The chosen client has an old active dialog/session that was not reviewed.
5. It is outside allowed working day/time.
6. `WHATSAPP_ENABLED` or `LIVE_SEND_ALLOWED` is not intentionally set for the
   controlled test window.
7. The candidate is disputed operationally and needs manual business decision.
8. The working tree contains new unreviewed production-code changes.

## 8. Remaining Non-Phase-2 Items

Not changed in this phase by design:

1. Stop-list business logic.
2. Active-guard business logic.
3. Debt classification.
4. CRM phone onboarding.
5. New bot `GPT1C_ProAnalytic`.

These remain separate production-readiness topics and must not be mixed with
the safe live-send architecture commits.

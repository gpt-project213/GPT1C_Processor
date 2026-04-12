# Monday Phase 3 Start Runbook

Date: 2026-04-12  
Author: Codex  
Purpose: keep Monday controlled live test bounded and repeatable

## Current Status

The codebase is prepared for Monday Phase 3 precheck. Do not run preview or
live send on Sunday. The first live test is still limited to one client and only
through the approved-batch path.

## Commands That Must Not Be Used

```powershell
python -m collector.collections_engine --send
```

Legacy `--send` is disabled in code, but it must still not be used as an
operational command.

## Monday Step 0

Run:

```powershell
git status --short
```

Expected:

```text
clean
```

No tracked dirty files are acceptable before preview or live. Local ignored
`.codex` is not relevant.

## Monday Step 1: Backup

Before any state-changing command, copy these files if they exist:

1. `logs/wa_approval_batches.json`
2. `logs/collector_state.json`
3. `logs/collector_client_dialogs.json`
4. `logs/collector_dialogs.json`
5. today's `logs/collector_YYYYMMDD.log`

Backup target:

```text
backups/phase3_YYYYMMDD_HHMMSS/
```

## Monday Step 2: Read-Only Precheck

Record:

1. workday status
2. current time window, must be 09:00-18:00
3. `WHATSAPP_ENABLED`
4. `LIVE_SEND_ALLOWED`
5. `COLLECTOR_DRY_RUN`
6. active client dialogs
7. pending approval batches
8. invalid/placeholder phones
9. candidate #1
10. reserve candidate #2

Write the result to:

```text
audit/FIRST_CONTROLLED_LIVE_PRECHECK_YYYY-MM-DD.md
```

## Monday Step 3: Dry Commands

Only after backup and green precheck:

```powershell
python -m collector.collections_engine --fix-first-seen
python -m collector.collections_engine --dry-run
```

If `--fix-first-seen` is already idempotent, record that and continue.

Stop if no clean candidate remains.

## Monday Step 4: Preview

Only after dry-run is reviewed:

```powershell
python -m collector.collections_engine --preview
```

Record:

1. `batch_id`
2. managers in batch
3. client count per manager
4. invalid_phone rows
5. approved_clients must still be empty

## Monday Step 5: Approvals

Managers approve/reject/manual-select.

Hard checks:

1. `manual_editing` does not complete batch.
2. `manual_done` completes only after final button.
3. each manager sees only own clients.
4. no invalid phone enters `approved_clients`.

Admin approval must set:

```text
status = admin_approved
admin_status = approved
approved_clients != []
```

## Monday Step 6: One Client Live

Only after admin approval and explicit owner confirmation:

```powershell
python -m collector.collections_engine --send-approved --batch-id <id> --client "<client>"
```

Hard limits:

1. one client only
2. valid phone only
3. no placeholder/test phone
4. no unresolved active dialog
5. no legacy send path

## Monday Step 7: Post-Send

Immediately check:

1. `send_results`
2. `send_summary`
3. actual status: `sent`, `failed`, or `skipped`
4. destination phone
5. duplicate prevention
6. state/log updates

Then turn live flags off manually if they were enabled:

```text
WHATSAPP_ENABLED=0
LIVE_SEND_ALLOWED=0
```

## Monday Report

Create:

```text
audit/FIRST_CONTROLLED_LIVE_TEST_YYYY-MM-DD.md
```

Do not proceed to a second client unless the report says `yes`.

---

## Phase 4 Residual Controls (added 2026-04-12)

Phase 4 fixed the active-guard mismatch for stopped/auto_stopped clients.
However, full dry-run vs preview parity is not yet proven by integration test —
only by unit tests on `_collector_candidate_decision()`.

**During Monday precheck, verify:**

1. Run `--dry-run` and `--preview` in sequence.
   Compare candidate lists manually.
   They must match (same names, same reasons).
   If they differ — stop and investigate before proceeding.

2. Check Олжас phone specifically:
   - `config/debtors_contacts.json` → `Е Олжас` → `whatsapp` field
   - Must be non-empty, non-placeholder, non-test
   - If `_needs_phone: true` or `whatsapp: ""` → Олжас must NOT enter `approved_clients`
   - Phone validation in `--send-approved` will also block it, but confirm earlier

3. Confirm in batch summary after `--preview`:
   - Any `invalid_phone` rows must be listed explicitly
   - `approved_clients` must start empty
   - Олжас and Tangirs: if no phone → appear as `no_phone`, not as `approved_clients`

4. Honest engineering status:
   - active-guard mismatch for stopped/auto_stopped: **FIXED** (e2bef30)
   - full dry-run vs preview parity: **must be verified during precheck**
   - Олжас phone validity: **must be checked manually**

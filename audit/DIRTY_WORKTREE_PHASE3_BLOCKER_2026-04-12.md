# Dirty Worktree Audit Before Phase 3

Date: 2026-04-12  
Author: Codex  
Scope: Phase 3 controlled live readiness blocker

## Verdict

Phase 3 controlled live must not proceed from the current worktree.

Reason: there are uncommitted changes in production-critical collector/bot
files. They affect client messaging, response interpretation, first_seen state
repair, scheduler behavior, and tests. Even if the Phase 2 safe send path is
committed and pushed, the runtime process would execute these dirty files from
disk, not only the committed Phase 2 code.

Sunday execution is also blocked by policy unless explicitly approved. The
owner decided not to grant a one-time Sunday exception.

## Current Dirty Files

Tracked dirty files:

| File | Risk | Phase 3 impact | Recommendation |
| --- | --- | --- | --- |
| `bot/send_reports.py` | Medium | Scheduler wrapper differs from committed code; includes preexisting changes around `debt_collector_daily` | Review and commit separately or restore only after explicit approval. Do not run Phase 3 until resolved. |
| `collector/client_dialog.py` | High | Changes live client reply behavior for `identity_question`, `promise_without_date`, and off-topic escalation storage | Review as separate UX/dialog commit with tests before any live send. |
| `collector/collection_agent.py` | High | Changes first outbound message generation, prompts source, fallback templates, intents | Review as separate prompt architecture commit before live send. |
| `collector/collections_db.py` | Medium | Adds `fix_first_seen_inflation()` state mutation logic | Review/commit with tests before using `--fix-first-seen`. |
| `tests/test_collector.py` | Medium | Adds tests for the dirty runtime behavior above | Should be committed with the related code, not alone. |
| `audit/INDEX.md` | Low | Documentation index only | Can be committed separately after audit docs are accepted. |

Untracked files:

| Path | Risk | Phase 3 impact | Recommendation |
| --- | --- | --- | --- |
| `config/collector_prompts.json` | High | Runtime-affecting if dirty `collection_agent.py` is active; controls customer-facing text | Must be reviewed and committed with prompt code before live send, or explicitly excluded and code reverted. |
| `COLLECTOR_CLASSIFICATION_RECONCILIATION_2026-04-12.md` | Medium | Documents prior `--fix-first-seen` / dry-run result; implies state-changing command was already run before this audit | Review and preserve as evidence, but do not rely on it without fresh Monday precheck. |
| `audit/CODEX_SUPPORT_CONTEXT_2026-04-11.md` | Low | Documentation/context only | Commit or archive separately. |
| `audit/COLLECTOR_BUSINESS_RULES_CLARIFICATION_2026-04-11.md` | Low/Medium | Business-rule docs can influence decisions, but not runtime | Commit separately if accepted. |
| `audit/COLLECTOR_CLASSIFICATION_AUDIT_2026-04-11.md` | Low/Medium | Classification docs only; do not change rules from it in Phase 3 | Commit separately if accepted. |
| `audit/COLLECTOR_SHORTLIST_EXPLAINED_2026-04-11.md` | Low/Medium | Shortlist docs only; stale for Monday without fresh precheck | Commit separately if accepted. |
| `audit/SAIDA_PAYMENT_FLOW_AUDIT_2026-04-11.md` | Low | Unrelated audit | Do not mix with Phase 3. |
| `audit/SHORTLIST_REFRESH_BLOCKER_FOR_CLAUDE_2026-04-11.md` | Low/Medium | Operational note; stale without fresh Monday precheck | Commit separately if accepted. |
| `audit/WHATSAPP_GREETING_UX_AUDIT_2026-04-11.md` | Medium | Supports dirty prompt/client-dialog changes | Commit only with/after UX code review. |
| `audit/manager_lists/*` | Low/Medium | Historical manager shortlist docs; likely stale for Monday | Commit separately or regenerate Monday. |
| `.codex` | Unknown | Tool/context artifact | Do not commit without inspection. |
| `ТЗ.txt` | Unknown | User task file | Do not commit without inspection. |

## Findings By File

### `bot/send_reports.py`

Observed dirty changes:

1. `debt_collector_daily` docstring changed from `18:00` to `17:30`.
2. Adds `WHATSAPP_ENABLED` check and forces `dry_run` when disabled.
3. Adds `expire_old_batches()` cleanup.
4. Scheduler command is `--dry-run`.

Phase 2 already committed the critical safe hunk that forces `--dry-run`.
Remaining dirty hunks are not necessarily wrong, but they are not isolated in a
clean commit. For Phase 3, this remains a traceability problem.

Recommendation: make a separate scheduler-cleanup commit only after review, or
restore the uncommitted hunks by explicit user order. Do not mix with live test.

### `collector/client_dialog.py`

Observed dirty changes:

1. Adds intent display labels for:
   - `promise_without_date`
   - `identity_question`
2. Adds live reply behavior for `identity_question`.
3. Adds live reply behavior for `promise_without_date`.
4. Stops saving empty bot replies on second off-topic escalation.

Risk: this changes customer-facing WhatsApp replies after a client responds.
This is useful work, but it is production behavior and must be accepted,
tested, committed, and pushed before controlled live.

Recommendation: treat as a separate UX safety patch, not as Phase 3 precheck.

### `collector/collection_agent.py`

Observed dirty changes:

1. Introduces `config/collector_prompts.json` as runtime prompt source.
2. Replaces built-in `_TONE` with `_TONE_DEFAULTS`.
3. Adds prompt loading helpers:
   - `load_prompts()`
   - `_get_tone()`
   - `_get_lang_inst()`
   - `_get_fallback_template()`
4. Adds new fallback templates.
5. Changes `generate_message()` behavior and adds `msg_type`.
6. Extends `analyze_response()` intents with:
   - `promise_without_date`
   - `identity_question`

Risk: this directly changes the first customer message and incoming-response
classification. It is one of the highest-risk dirty changes for a controlled
live test.

Recommendation: review text output examples, run tests, commit with
`config/collector_prompts.json`, then re-run Phase 3 precheck.

### `collector/collections_db.py`

Observed dirty changes:

1. Adds `fix_first_seen_inflation()`.
2. Function mutates `logs/collector_state.json` by recalculating
   `__debt_since__*` first_seen values.

Risk: state mutation is intended, but it must be traceable. There is an
untracked reconciliation document claiming `fixed=157`, so this may already
have been run earlier.

Recommendation: do not run `--fix-first-seen` again until Monday precheck
confirms current state and backup is created.

### `tests/test_collector.py`

Observed dirty changes:

1. Adds tests for `fix_first_seen_inflation()`.
2. Adds tests for `identity_question`.
3. Adds tests for `promise_without_date`.
4. Adds tests for off-topic empty reply behavior.
5. Adds tests for `collector_prompts.json` and prompt helpers.

Recommendation: commit with the corresponding runtime code only after review.

### `config/collector_prompts.json`

Observed status: untracked, runtime-affecting if dirty `collection_agent.py`
remains active.

Risk: live customer copy would depend on an untracked file. This is not
acceptable for controlled live.

Recommendation: either commit it with prompt code, or do not use the dirty
prompt-loader code in live.

## Safe Path To Monday

Before Phase 3 can restart:

1. Decide whether to keep the dirty UX/prompt/first_seen changes.
2. If keeping them:
   - review diffs;
   - run tests;
   - commit in separate logical commits;
   - push;
   - then restart Phase 3 from Stage 0.
3. If not keeping them:
   - request an explicit restore/revert plan;
   - do not use destructive git commands without explicit approval.
4. Regenerate Monday precheck fresh; do not rely on 2026-04-11 shortlist docs.

## Suggested Commit Plan If Keeping Changes

1. `fix(collector): add first-seen repair command`
   - `collector/collections_db.py`
   - relevant `tests/test_collector.py` section

2. `fix(collector): improve client reply handling`
   - `collector/client_dialog.py`
   - relevant `tests/test_collector.py` section

3. `fix(collector): load collector prompts from config`
   - `collector/collection_agent.py`
   - `config/collector_prompts.json`
   - relevant `tests/test_collector.py` section

4. `fix(bot): finalize collector scheduler dry-run cleanup`
   - remaining reviewed `bot/send_reports.py` hunks only

5. `docs(audit): add collector shortlist and UX notes`
   - accepted audit docs only

Do not combine these with Phase 3 live-test audit.

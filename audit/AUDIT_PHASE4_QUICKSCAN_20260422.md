# Audit Phase 4 Quick Scan - 2026-04-22

## Scope

Quick scan of remaining modules and repository artifacts after Phase 3 hardening.

## Findings

### F-P4-001 - HIGH - Tracked historical logs contained Telegram bot token

- Scope:
  - `logs_public/send_reports_20260212.log`
  - `logs_public/send_reports_20260216.log`
  - `logs_public/send_reports_20260217.log`
  - `logs_public/send_reports_20260218.log`
  - `logs_public/send_reports_20260219.log`
  - `logs_public/send_reports_20260220.log`
  - `logs_public/send_reports_20260222.log`
  - `logs_public/send_reports_20260223.log`
  - `logs_public/send_reports_20260225.log`
  - `logs_public/send_reports_20260226.log`
- Evidence before fix:
  - HTTP log lines contained full URLs of the form
    `https://api.telegram.org/bot<token>/...`
- Impact:
  - secret exposure in tracked repository artifacts
- Resolution:
  - token-like values were redacted in place to `bot<TG_TOKEN>/`
  - historical logs were preserved; only secrets were rewritten

### F-P4-002 - MEDIUM - `repo_map.json` was stale and misleading

- Evidence before fix:
  - `generated_at = 2026-03-10 23:40:59`
  - `branch = master`
  - `raw_base` pointed to `master`
- Impact:
  - new sessions could use a false repository map and wrong branch links
- Resolution:
  - `repo_map.json` regenerated on current branch
  - updated branch: `fix/log-noise-by-design-markers`

### F-P4-003 - MEDIUM/LOW - `collector/communications.py` bypasses central config layer

- Evidence:
  - direct read of `config/roles.json`
- Status:
  - noted only
  - no code change in this pass

## Verification

- `rg -n "api\.telegram\.org/bot[0-9]{5,}:[A-Za-z0-9_-]+/|bot[0-9]{5,}:[A-Za-z0-9_-]+" logs_public`
  - no matches after redaction
- `repo_map.json`
  - branch now matches current HEAD branch
  - timestamp refreshed

## Notes

- This pass intentionally did not delete logs or rewrite their operational meaning.
- Only secret-bearing fragments were redacted.
- `collector/communications.py` config drift remains backlog, not hotfix.

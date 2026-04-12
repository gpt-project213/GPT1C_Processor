# Input TZ Archive

Date: 2026-04-12  
Author: Codex  
Source: root `ТЗ.txt`

The original user-provided task file is tracked as `ТЗ.txt`. This note exists so
the launch-readiness/audit folder has an English filename pointing to the input
requirements.

The file describes the broader recovery task for collector, WhatsApp
notifications, manager dialogs, state deadlocks, escalation, and production
readiness. The current implementation work intentionally limited scope to:

1. Phase 2 safe live-send architecture.
2. Launch-readiness cleanup required before Phase 3.
3. Monday controlled live precheck/runbook.

Remaining broad items from `ТЗ.txt` should be handled as separate phases, not
mixed into the first controlled live test.

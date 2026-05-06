# Карта Содержимого Audit — 2026-04-22

Цель: зафиксировать фактическую роль документов в `audit/` после полного чтения содержимого.

Принцип:
- это не список "лишних файлов";
- это карта по смыслу, чтобы потом чистить только осознанно и без потери знаний;
- исторические документы остаются полезными, даже если часть их выводов уже закрыта коммитами.

## 1. Базовые reference-документы

- `audit/ARCHITECTURE.md`
  Архитектурная цель нового продукта. Не про текущее устройство репозитория, а про желаемую целевую декомпозицию.
- `audit/BOOTSTRAP_FLOW.md`
  Целевой единый bootstrap и правила запуска.
- `audit/STATE_MODEL.md`
  Нормативная модель state/repository/TTL/cleanup. Один из самых полезных документов для проектирования.
- `audit/AUDIT_PROJECT_MAP.md`
  Карта входов, state и контуров по проекту на момент общего аудита.

## 2. Базовые общепроектные аудиты

- `audit/AUDIT_FULL_PROJECT_2026-04-09.md`
  Большой технический аудит проекта. Полезен как историческая база по багам, рискам и архитектурным долгам.
- `audit/AUDIT_SUMMARY.md`
  Краткий executive summary общего аудита.
- `audit/AUDIT_VERDICT.md`
  Итоговый вердикт по готовности проекта на момент общего среза.
- `audit/AUDIT_RUNTIME_TRACE.md`
  Runtime-доказательства из логов и state-снимков.
- `audit/AUDIT_FINDINGS.md`
  Список findings из раннего общего аудита.
- `audit/AUDIT_FINDINGS.json`
  Машиночитаемая версия findings; полезна для автоматизации и сверки.
- `audit/ПОЛНЫЙ_СВОД_АУДИТА_13.04.2026.md`
  Крупный объединённый русский свод по проекту, багам и слоям.
- `audit/ПОЛНЫЙ_СВОД_АУДИТА_ОТЧЕТОВ_14.04.2026.md`
  Отдельный свод по отчётным модулям, тестам и статусам исправлений.

## 3. Актуальные документы по коллектору

- `audit/AUDIT_COLLECTOR_20260422.md`
  Самый свежий audit по коллектору; текущая опорная точка по recent fixes.
- `audit/PHASE2_SAFE_SEND_2026-04-12.md`
  Ключевой документ по безопасной live-send архитектуре.
- `audit/FULL_COLLECTOR_PRODUCTION_AUDIT_2026-04-12.md`
  Production audit коллектора до Phase 2; важен для понимания, почему появились current guard'ы.
- `audit/COLLECTOR_PREVIEW_PHASE1_PROTOCOL_2026-04-12.md`
  Документирует смысл и границы preview-фазы.
- `audit/COLLECTOR_CLASSIFICATION_AUDIT_2026-04-11.md`
  Базовый аудит debt-classification логики.
- `audit/COLLECTOR_CLASSIFICATION_RECONCILIATION_2026-04-12.md`
  Разбор after `fix-first-seen`.
- `audit/COLLECTOR_CLASSIFICATION_LOG_RECHECK_2026-04-13.md`
  Важный recheck по смешиванию debt-источников и периодов.
- `audit/COLLECTOR_BUSINESS_RULES_CLARIFICATION_2026-04-11.md`
  Бизнес-правила коллектора, полезны для неочевидных решений.
- `audit/COLLECTOR_SHORTLIST_EXPLAINED_2026-04-11.md`
  Разбор shortlist для директора человеческим языком.
- `audit/CONTROLLED_LIVE_TEST_2026-04-11.md`
  Контекст controlled live test и его ограничения.
- `audit/LAUNCH_READINESS_PROTOCOL_2026-04-12.md`
  Допуск к запуску; operational runbook.
- `audit/MONDAY_PHASE3_START_RUNBOOK_2026-04-12.md`
  Пошаговый сценарий старта controlled live.
- `audit/WEEKEND_AND_FINAL_APPROVAL_2026-04-11.md`
  Документ о weekend silent mode и финальном admin approval.
- `audit/WHATSAPP_APPROVAL_UX_2026-04-11.md`
  UX-дизайн согласования рассылки.
- `audit/WHATSAPP_GREETING_UX_AUDIT_2026-04-11.md`
  Аудит приветствия и реакции клиентов в WhatsApp.
- `audit/INCIDENT_REPORT_WHATSAPP_2026-04-10.md`
  Ключевой incident report; нельзя терять.

## 4. Residual / business-specialized collector docs

- `audit/PHASE5_RESIDUAL_DEBT_CLASSIFICATION_2026-04-12.md`
  Переход от `days_silence` к возрасту текущего остатка.
- `audit/PHASE5_SILENCE_ALERTS_RESIDUAL_2026-04-12.md`
  Связь residual debt age с краткими уведомлениями.
- `audit/SAIDA_PAYMENT_FLOW_AUDIT_2026-04-11.md`
  Полезный бизнес-документ по Саиде, оплатам и stop-list процессу.
- `audit/SAIDA_PAYMENT_HOLD_2026-04-13.md`
  Узкий follow-up по payment-hold.

## 5. CRM-ориентированные документы

- `audit/CRM_FULL_AUDIT_2026-04-11.md`
  Главный CRM-аудит.
- `audit/CRM_CONTACTS_XLSX_MIRROR_2026-04-12.md`
  Роль и ограничения `contacts.xlsx`.
- `audit/CRM_PHONE_HINTS_FROM_CLIENT_NAME_2026-04-13.md`
  Подсказки телефонов из имён клиентов; полезно как reference для phone-onboarding.

## 6. Routing / runtime / support / process docs

- `audit/MESSAGE_ROUTING_AUDIT_2026-04-11.md`
  Один из самых полезных документов по маршрутизации Telegram/approval flows.
- `audit/INDEX.md`
  Исторический индекс audit-документов; полезен, но не исчерпывает текущую картину.
- `audit/SESSION_CONTEXT.md`
  Старый handoff/context внутри audit-папки.
- `audit/CODEX_SUPPORT_CONTEXT_2026-04-11.md`
  Подробный support-context для прошлой сессии; полезен как evidence map.
- `audit/CODEX_AUTOSAVE_CONTEXT_2026-04-12.md`
  Контекст фазы controlled live.
- `audit/SHORTLIST_REFRESH_BLOCKER_FOR_CLAUDE_2026-04-11.md`
  Операционный блокер-документ; исторически полезен.
- `audit/DIRTY_WORKTREE_PHASE3_BLOCKER_2026-04-12.md`
  Важный документ о риске запуска из грязного дерева.
- `audit/INPUT_TZ_ARCHIVE_2026-04-12.md`
  Узкий вспомогательный артефакт; не главный, но сохраняет исторический контекст TZ.

## 7. Manager lists — не мусор, а операционные расшифровки

- `audit/manager_lists/README.md`
  Объясняет назначение каталога.
- `audit/manager_lists/NOTIFICATION_CANDIDATES_2026-04-11.md`
  Свод по кандидатам на уведомления.
- `audit/manager_lists/alena.md`
  Краткий shortlist по Алёне.
- `audit/manager_lists/ergali.md`
  Краткий shortlist по Ергали.
- `audit/manager_lists/magira.md`
  Краткий shortlist по Магире.
- `audit/manager_lists/oksana.md`
  Краткий shortlist по Оксане.
- `audit/manager_lists/MANAGER_SHORTLIST_Алена_2026-04-11.md`
  Расширенный директорский разбор по Алёне.
- `audit/manager_lists/MANAGER_SHORTLIST_Ергали_2026-04-11.md`
  Расширенный директорский разбор по Ергали.
- `audit/manager_lists/MANAGER_SHORTLIST_Магира_2026-04-11.md`
  Расширенный директорский разбор по Магире.
- `audit/manager_lists/MANAGER_SHORTLIST_Оксана_2026-04-11.md`
  Расширенный директорский разбор по Оксане.

Вывод по `manager_lists/`:
- короткие файлы `alena.md` / `ergali.md` / `magira.md` / `oksana.md` и расширенные `MANAGER_SHORTLIST_*` не полные дубли;
- короткие версии удобны как быстрый слой;
- расширенные версии полезны для директорского решения и следов reasoning.

## 8. Audit log archives

- `audit/logs/send_reports_20260403.log`
  Большой исторический runtime-срез: IMAP, pipeline, poller, jobs.
- `audit/logs/send_reports_20260408.log`
  Исторический runtime-срез перед инцидентным периодом.
- `audit/logs/send_reports_20260409.log`
  Узкий runtime-срез переходного момента.

Вывод по архивным логам:
- это не "лишние логи", а доказательная база для аудитов и инцидентных разборов;
- чистить их можно только после решения, что доказательства уже не нужны.

## 9. Статус полезности

- Нельзя считать весь `audit/` свалкой.
- Нельзя удалять русские своды, incident reports, runbook'и и manager lists без потери контекста.
- Наиболее критично сохранить:
  - `INCIDENT_REPORT_WHATSAPP_2026-04-10.md`
  - `PHASE2_SAFE_SEND_2026-04-12.md`
  - `FULL_COLLECTOR_PRODUCTION_AUDIT_2026-04-12.md`
  - `MESSAGE_ROUTING_AUDIT_2026-04-11.md`
  - `STATE_MODEL.md`
  - `ARCHITECTURE.md`
  - `ПОЛНЫЙ_СВОД_АУДИТА_13.04.2026.md`
  - `ПОЛНЫЙ_СВОД_АУДИТА_ОТЧЕТОВ_14.04.2026.md`

## 10. Git-анomaly note

На 2026-04-22 обнаружена аномалия индекса:
- два русскоязычных файла были отслежены в git под каталогом `аудит/`;
- на диске актуальные файлы лежат под каталогом `audit/`;
- содержимое совпадает по blob hash.

Это не содержательная потеря, а рассинхрон имени каталога в истории git.

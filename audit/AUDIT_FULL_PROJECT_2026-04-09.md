# Полный комплексный аудит проекта

## 1. Резюме проекта

Проект представляет собой эксплуатационную систему для обработки управленческих отчётов из 1С, доставки отчётов и аналитики в Telegram, пополнения CRM-карточек клиентов, контроля дебиторской задолженности, а также отдельного контура AI Collector с исходящими WhatsApp-сообщениями и эскалацией менеджерских кейсов.

Фактически это не один бот, а связка из нескольких контуров:

- IMAP-загрузка файлов из почты.
- Pipeline обработки Excel-файлов в HTML/JSON.
- Telegram-бот как единая точка интерфейса, планировщик и оркестратор.
- CRM-подсистема для дозаполнения контактов менеджерами.
- Debt / collector-подсистема для дебиторки, обещаний, stop-list и WhatsApp.
- Набор аналитических и summary-отчётов.
- Большой слой JSON-state и эксплуатационных логов.

Текущее состояние проекта: система реально работает в production-режиме и выполняет полезную бизнес-функцию, но зрелость неоднородна. Есть участки production-grade, особенно в парсинге, логировании событий и атомарной записи JSON. Одновременно есть критические операционные риски: секреты в `.env`, перегруженная монолитная точка входа `bot/send_reports.py`, накопившееся дублирование state-файлов, конфликт наследия и новых потоков, а также повышенная хрупкость на стыках CRM / collector / scheduler / absolute paths.

Итоговая оценка: проект можно считать рабочей production-системой среднего уровня зрелости, но не зрелой платформой. Он уже приносит операционную ценность, но дальше будет всё хуже масштабироваться без стабилизации state-модели, разгрузки основного бота и наведения порядка в конфигурации и секретах.

## 2. Что было прочитано

Аудит охватил весь проект как кодовую и эксплуатационную систему.

Прочитано и учтено:

- исходный код Python по всем ключевым каталогам: `bot/`, `collector/`, корневые отчётные/парсерные модули, `tools/`, `tests/`;
- конфигурация: `.env`, `requirements.txt`, `config/*.json`, `config/*.yaml`;
- шаблоны и документы: `templates/*.html`, `docs/*.html`, `*.md`, `*.txt`;
- актуальные state-файлы: `logs/*.json`, `reports/*.json`, `reports_state.json`, `config/clients.json`, `config/debtors_contacts.json`, `collector/debtors_contacts.json`;
- эксплуатационные логи: `logs/send_reports_20260409.log`, `logs/collector_20260409.log`, `logs/email_20260409.log`, `logs/run_pipeline_all_mp.log`, а также структура других логов;
- аудит-материалы из `audit/`;
- вспомогательные bat/ps1 и эксплуатационные скрипты;
- тестовые файлы и инструменты диагностики.

Количественный охват:

- в дереве проекта найдено 9433 файловых артефакта без `.git/.venv`;
- релевантных текстовых/кодовых/конфигурационных/шаблонных файлов по типам `.py/.json/.yaml/.yml/.md/.html/.txt/.ps1/.bat/.sh/.env` и `requirements.txt`: 3368;
- структура расширений показывает выраженную эксплуатационную природу проекта: тысячи `log`, `json`, `html`, `xlsx`.

Что читалось глубоко:

- основные точки входа и orchestration: `bot/send_reports.py`, `run_pipeline_all_mp.py`, `imap_fetcher.py`, `collector/collections_engine.py`, `collector/manager_dialog.py`, `collector/client_dialog.py`, `bot/debt_stop_control.py`, `bot/crm_clients.py`, `bot/silence_alerts.py`, `config.py`, `send_tg.py`;
- конфиги ролей/менеджеров/imap/CRM/collector;
- текущие runtime-state и логи, подтверждающие реальное поведение.

Что учитывалось как runtime-артефакты, а не разбиралось построчно:

- тысячи HTML/XLSX/архивных логов и объёмные эксплуатационные артефакты;
- большие каталоги с историческими отчётами и архивами;
- generated JSON/HTML, если они не требовались для доказательства конкретного вывода.

## 3. Карта проекта

### 3.1. Основные модули

- `bot/send_reports.py`
  - центральный Telegram-бот;
  - persistent menu;
  - callback routing;
  - scheduler;
  - janitor;
  - pipeline trigger;
  - CRM flow;
  - analytics menu;
  - collector reminder / WhatsApp poller integration;
  - stop-list integration.

- `imap_fetcher.py`
  - прямой IMAP polling;
  - whitelist отправителей;
  - скачивание вложений;
  - создание clean-копий;
  - удаление писем и чистка корзины.

- `run_pipeline_all_mp.py`
  - альтернативный оркестратор pipeline;
  - claim `.work`;
  - маршрутизация по типу отчёта;
  - запуск билдеров;
  - перенос в processed.

- корневые отчётные модули
  - `debt_auto_report.py`
  - `sales_report.py`
  - `gross_report.py`
  - `gross_report_pct.py`
  - `inventory.py`
  - `expenses_report.py`
  - `expenses_parser.py`
  - аналитические генераторы (`net_profit_report.py`, `sales_profitability_report.py`, `rfm_clients_report.py`, `dso_aging_report.py`, `revenue_concentration_report.py`, `inventory_turnover_report.py`).

- `bot/crm_clients.py`
  - единая CRM-база в `config/clients.json`;
  - обновление из JSON-отчётов;
  - поиск клиентов без телефона;
  - запись деталей.

- `collector/*`
  - debt monitor;
  - dialog store;
  - manager dialog;
  - client dialog;
  - communications;
  - collections DB state;
  - AI text generation / response analysis;
  - WhatsApp poller.

- `bot/debt_stop_control.py`
  - отдельный производственный контур stop-list/Саида/эскалации менеджеров.

### 3.2. Точки входа

Основные реальные точки входа:

- `python bot/send_reports.py`
  - главный production-процесс;
  - внутри него живут scheduler, menu, notifier, CRM, collector reminders, WhatsApp poller, stop-list jobs.

- `imap_fetcher.py --once`
  - вызывается pipeline-задачей и может работать отдельно.

- `run_pipeline_all_mp.py`
  - отдельный оркестратор pipeline, параллельный по назначению логике `pipeline_task()` в `send_reports.py`.

- `collector/collections_engine.py`
  - отдельный CLI/process для daily collector и promises flow.

Дополнительные эксплуатационные входы:

- bat/ps1-файлы запуска и синхронизации;
- ручные команды `/analytics`, `/phone`, `/guide` и callback-driven маршруты;
- tools/tests для локальной диагностики.

### 3.3. Основные потоки данных

#### Почта → Queue → Parsing → JSON/HTML → Telegram

1. `imap_fetcher.py` читает IMAP и кладёт Excel в `reports/queue/`.
2. `pipeline_task()` в `send_reports.py` или `run_pipeline_all_mp.py` определяет тип файла.
3. Запускаются соответствующие генераторы HTML/JSON.
4. Новые отчёты попадают в индекс, архив, краткие summary и AI-аналитику.
5. Telegram-бот раздаёт отчёты по ролям.

#### JSON-отчёты → CRM

1. `bot/crm_clients.update_from_reports()` читает последние debt/sales JSON.
2. Новые клиенты попадают в `config/clients.json`.
3. В 18:00 `crm_daily_task()` раздаёт менеджерам задачи на заполнение контактов.
4. Pending-state хранится в `logs/crm_pending_state.json`.

#### Debt → Collector → Manager dialog → WhatsApp → Client dialog

1. `collector/debt_monitor.py` читает latest debt JSON и contacts.
2. `collector/collections_engine.py` выбирает должников и маршрут.
3. Если нужен менеджерский шаг — создаётся запись в `logs/collector_dialogs.json`.
4. `collector/manager_dialog.py` ведёт Telegram-диалог менеджера.
5. После подтверждения идёт send-path в WhatsApp.
6. Если клиент отвечает — `collector/whatsapp_poller.py` передаёт сообщение в `collector/client_dialog.py`.
7. При сложных сценариях идёт эскалация менеджеру/наблюдателям.

#### Debt stop control

1. `bot/debt_stop_control.py` читает latest `debt_ext_*.json`.
2. Формирует кандидатов, пишет `reports/debt_stop_state.json`.
3. Ведёт жизненный цикл решения менеджер → admin → Саида.
4. Постоянный реестр живёт в `reports/debt_stop_registry.json`.

## 4. Сильные стороны проекта

### 4.1. Система реально доведена до эксплуатации

Это не макет и не прототип. По логам видно живое выполнение scheduler-задач, Telegram polling, IMAP fetching, Green API polling, state persistence. Проект уже используется как операционный инструмент.

### 4.2. Хорошее покрытие бизнес-контуров

Система закрывает не только доставку отчётов, но и downstream-процессы:

- уведомления менеджеров;
- CRM-дозаполнение;
- контроль тишины клиентов;
- collector flow;
- stop-list отгрузки;
- аналитика и краткие summary.

Это сильная сторона проекта как бизнес-автоматизации: много ручной координации уже переведено в систему.

### 4.3. JSON-state хранится в основном атомарно

Во многих критичных местах запись идёт через temp file + replace:

- `bot/crm_clients.py`;
- `collector/dialog_store.py`;
- `collector/client_dialog.py`;
- `bot/debt_stop_control.py`;
- `manager_dialog._save_contact()` и другие.

Это существенно снижает риск частично записанных файлов при перезапуске или падении.

### 4.4. Наблюдаемость лучше среднего для локального automation-проекта

Есть:

- суточные логи;
- event-like логирование;
- явные предупреждения по ошибкам;
- runtime trace в `audit/`;
- заметная эксплуатационная дисциплина вокруг janitor / deletion queue / reminders / daily summary.

По локальным логам можно восстановить большую часть производственного поведения.

### 4.5. Ролевая модель не примитивная

`config/roles.json` поддерживает:

- admin;
- subadmin со scope;
- manager;
- accountant.

Эта модель действительно встроена в menu/routing, а не просто лежит в конфиге.

### 4.6. Шаблонная часть достаточно стабильна

HTML-шаблоны большие, но в основном консистентные. Есть единая стилистика, отчёты сформированы под реальную управленческую эксплуатацию, а не как временная разработческая выгрузка.

## 5. Слабые стороны проекта

### 5.1. Критическая монолитность `bot/send_reports.py`

Файл почти 7000 строк и совмещает:

- точку входа;
- scheduling;
- routing;
- отчётные меню;
- janitor;
- архив;
- AI-аналитику;
- CRM;
- collector integration;
- voice handling;
- stop-list integration.

Это главный технический долг проекта. Любая правка в этом файле потенциально может зацепить соседние контуры, а review и regression-анализ становятся дорогими.

### 5.2. Дублирование orchestration

Параллельно существуют:

- `pipeline_task()` внутри Telegram-бота;
- `run_pipeline_all_mp.py`.

Оба отвечают за похожий жизненный цикл pipeline. Это создаёт риск расхождения логики, двойной поддержки и неправильного operational start.

### 5.3. Избыточно много JSON-state без единой схемы

State размазан по:

- `logs/notify_state.json`
- `logs/deletion_queue.json`
- `logs/crm_pending_state.json`
- `logs/collector_dialogs.json`
- `logs/collector_client_dialogs.json`
- `logs/collector_state.json`
- `reports/debt_stop_state.json`
- `reports/debt_stop_registry.json`
- `reports_state.json`
- `config/clients.json`
- `config/debtors_contacts.json`
- `collector/debtors_contacts.json`

Это не просто много файлов. Это множество разных жизненных циклов, ключей, TTL, pending-моделей и устаревших схем, частично пересекающихся по смыслу.

### 5.4. Наследие старых путей и старых flow всё ещё живо

В проекте и state-файлах присутствуют следы старых путей:

- `F:\...`
- `E:\...`
- `C:\Users\...\GPT1C_Processor_analitica`

и старых flow:

- legacy `__phone_pending__` / `__name_pending__`;
- `collector/debtors_contacts.json` как альтернативная база;
- закомментированные фрагменты старого CRM/collector ввода в `send_reports.py`.

Это означает, что проект несёт значительный слой исторической совместимости, который уже мешает ясности.

### 5.5. Секреты в локальном проекте хранятся небезопасно

В `.env` лежат production-секреты в открытом виде:

- почта IMAP;
- Telegram bot token;
- OpenAI / DeepSeek API keys;
- Green API token;
- служебные chat_id.

Это серьёзный operational и security-риск.

## 6. Найденные архитектурные риски

### Критичный

- `CR-1` Монолитная центральная точка отказа: `bot/send_reports.py`.
  - Любая регрессия в этом файле может одновременно сломать меню, jobs, CRM, collector и pipeline integration.

- `CR-2` Секреты и production-конфигурация находятся в проектной папке в открытом виде.
  - Это риск утечки и случайного запуска на неправильном окружении.

- `CR-3` Дублированная orchestration-логика pipeline.
  - `send_reports.py` и `run_pipeline_all_mp.py` решают схожие задачи, но не являются одной системой с общим контрактом.

### Высокий

- `HR-1` State-модель не централизована и конфликтует сама с собой.
  - Пример: `config/debtors_contacts.json` и `collector/debtors_contacts.json`.

- `HR-2` Архитектура heavily file-based и не содержит единой схемы миграции состояния.
  - При накоплении исторических данных и смене форматов проект становится всё менее предсказуемым.

- `HR-3` Роутинг Telegram построен в одном callback/text handler с высокой плотностью ветвлений.
  - Это хрупкая точка для дальнейшего развития.

- `HR-4` Production-логика partly depends on absolute file names / mtime / local folders.
  - Это ухудшает переносимость и делает систему чувствительной к среде запуска.

### Средний

- `MR-1` Отчётные парсеры и HTML-шаблоны связаны не контрактами, а соглашениями по полям/именам.

- `MR-2` Система склонна к накоплению legacy blocks вместо их изоляции и удаления.

- `MR-3` Часть функционала зависит от внешних API без полноценных circuit-breaker / fallback policy.

### Низкий

- `LR-1` Много runtime-скриптов и документации, которые уже не полностью синхронизированы с фактическим поведением.

## 7. Найденные технические дефекты

### 7.1. Production secrets committed in working tree

Где:

- `.env`

Проблема:

- в рабочем проекте лежат реальные production credentials;
- нет разделения на `.env.example` и отдельный secured deployment secret source.

Чем грозит:

- утечкой доступа к почте, Telegram, Green API, AI API;
- несанкционированной отправкой сообщений;
- компрометацией производственной переписки.

### 7.2. Конфликтующее хранилище контактов должников

Где:

- `config/debtors_contacts.json`
- `collector/debtors_contacts.json`
- `collector/debt_monitor.py`
- `collector/manager_dialog.py`
- `bot/crm_clients.py`

Проблема:

- исторически один контур читал `config/...`, другой писал в `collector/...`;
- в локальном проекте уже есть оба файла, и они содержат разные объёмы и разные схемы.

Чем грозит:

- расхождением того, что “видит” collector, CRM и manager dialog;
- тихой потерей внесённых телефонов;
- ложными `_needs_phone` / неподтверждёнными контактами.

### 7.3. Pending / cleanup-схемы неоднородны и частично destructive

Где:

- `logs/crm_pending_state.json`
- `logs/collector_state.json`
- `collector/dialog_store.py`
- `bot/send_reports.py`

Проблема:

- разные pending-схемы живут по разным правилам;
- legacy keys просто удаляются cleanup-логикой;
- часть cleanup основана на timestamp-полях без общей модели миграции/архивации.

Чем грозит:

- потерей истории;
- “исчезновением” кейса вместо корректной развязки;
- труднообъяснимыми production-состояниями.

### 7.4. `notify_state.json` загрязнён абсолютными путями разных машин

Где:

- `logs/notify_state.json`

Проблема:

- dedupe и уведомления привязаны к file paths из старых сред `E:\...`, `C:\Users\...`.

Чем грозит:

- пропуском уведомлений или повторными уведомлениями при переносе проекта;
- неочевидным поведением после запуска на другой копии/диске.

### 7.5. Runtime-лог содержит признаки тестовых артефактов в production-tree

Где:

- `logs/collector_dialogs.json`
- `logs/collector_20260409.log`

Проблема:

- в runtime есть кейсы вида `OTHER CLIENT`, `NO PHONE CLIENT`, `TEST CONFIRM`, `TIMEOUT CLIENT`, `Client A/B`;
- это выглядит как результаты локальных логических прогонов на production-like дереве.

Чем грозит:

- загрязнением реального state;
- неоднозначностью при расследовании;
- ложными дедупликациями/cleanup-эффектами.

### 7.6. IMAP/pipeline старт зависит от конкретного operational сценария

Где:

- `bot/send_reports.py`
- `run_pipeline_all_mp.py`
- `imap_fetcher.py`
- bat/ps1 scripts

Проблема:

- не одна однозначная схема запуска;
- в логах и документах видны разные окружения и пути запуска.

Чем грозит:

- стартом не того процесса;
- параллельным запуском overlapping оркестраторов;
- сложностью операционного сопровождения.

### 7.7. CRM backlog уже большой и throughput ограничен

Где:

- `config/clients.json`
- `logs/crm_pending_state.json`
- `bot/send_reports.py`

Подтверждение:

- клиентов в CRM: 547;
- без телефона/Telegram остаются 366;
- по менеджерам backlog неравномерен и высок;
- pending-state одновременно висит минимум у 3 менеджеров.

Чем грозит:

- низкой скоростью дозаполнения;
- менеджерской усталостью;
- отставанием CRM от реальности.

## 8. Узкие места и bottleneck’и

### 8.1. Производительность

- многократное чтение больших JSON-файлов в hot-path;
- линейный обход state на каждую операцию;
- дедупликация и lookup по `client_name` без индекса;
- файловый pipeline и архивирование на каждом цикле.

### 8.2. UX

- CRM удобнее, чем раньше, но всё ещё тяжело масштабируется на большой backlog;
- многие потоки завязаны на последовательный диалог “один кейс за раз”;
- у менеджеров мало видимости по общей очереди и по собственному прогрессу в контексте дня/недели.

### 8.3. State

- нет единой схемы жизненного цикла кейса;
- pending и terminal состояния различаются по контурам;
- cleanup местами слишком слабый, местами слишком грубый.

### 8.4. Люди/процесс

- система сильно завязана на правильную дисциплину менеджеров;
- при росте количества кейсов увеличивается стоимость ручного сопровождения и контроля backlog;
- у админа появляется роль “человеческого garbage collector”.

### 8.5. Эксплуатация

- много логов и JSON-state, но нет единой operational dashboard;
- отсутствует централизованный health-report по всем контурам;
- трудно быстро понять: что сломалось, что отстаёт, что зависло, что просто пусто.

## 9. Оценка по контурам

### 9.1. Telegram bot

Что хорошо:

- один бот закрывает почти весь операционный интерфейс;
- есть persistent menu;
- есть role-based routing;
- есть janitor, deletion queue, health и статистика.

Что плохо:

- `bot/send_reports.py` чрезмерно перегружен;
- один `CallbackQueryHandler` и один большой text-handler несут слишком много логики;
- высок риск случайных регрессий при локальных правках.

Риски:

- ошибка в callback/text routing может затронуть сразу несколько бизнес-контуров;
- высокая стоимость безопасного изменения поведения.

Рекомендации:

- выделить отдельные router-модули: reports, analytics, CRM, collector bridge, archive, stop-list;
- оставить `send_reports.py` только как wiring + scheduler + app bootstrap.

### 9.2. IMAP / queue / pipeline

Что хорошо:

- `imap_fetcher.py` зрелый для локального automation-проекта;
- retries, clean copy, sender whitelist, trash cleanup, BOM-safe env load;
- pipeline умеет различать типы файлов и не падать целиком на одном кейсе.

Что плохо:

- есть дублирование orchestration между `pipeline_task()` и `run_pipeline_all_mp.py`;
- много эвристик по filename/content вместо формализованного контракта;
- опора на локальные папки и `mtime`.

Риски:

- расхождение двух pipeline-веток;
- тихие skip-case при нестандартных именах файлов;
- сложность поддержки новых отчётных типов.

Рекомендации:

- определить один canonical pipeline orchestrator;
- вторую ветку перевести в diagnostics/manual mode или удалить;
- вынести классификацию отчётов в один модуль/таблицу правил.

### 9.3. CRM

Что хорошо:

- CRM пополняется автоматически из отчётов;
- есть живой manager workflow;
- state переживает перезапуск;
- очереди и reminder уже реально работают.

Что плохо:

- backlog велик;
- пропускная способность ограничена;
- state logic всё ещё смешана с routing-логикой бота;
- `config/clients.json` стал и CRM, и partly integration-store, и partly справочник.

Риски:

- менеджеры будут физически не успевать закрывать очередь;
- старые pending и name/phone review могут деградировать в висящие хвосты;
- дальнейшее усложнение UX внутри `send_reports.py` быстро ухудшит сопровождение.

Рекомендации:

- отделить CRM state-machine в отдельный модуль;
- ввести явные поля `review_needed`, `source_of_truth`, `updated_by`;
- добавить дневную/недельную отчётность по throughput CRM.

### 9.4. Collector / debt / WhatsApp

Что хорошо:

- архитектурно контур уже выделен в `collector/`;
- есть manager dialog, client dialog, communications, DB helpers;
- есть reminder/escalation/TTL/anti-duplicate попытки;
- имеется связка Telegram manager flow ↔ WhatsApp client flow.

Что плохо:

- состояние коллектора раздроблено между несколькими JSON;
- исторически были конфликты контактов и confirmation semantics;
- дедупликация всё ещё опирается на `client_name`, что неустойчиво;
- часть защиты от дублей и таймаутов сделана как patch over existing model, а не как чистая state machine.

Риски:

- ложный suppress по клиенту;
- ошибки вокруг “отправлено / не отправлено / подтверждено”;
- скрытая зависимость от качества имени клиента и чистоты runtime-state.

Рекомендации:

- ввести stable client key, если он доступен из источников;
- свести manager/client/dialog states в явную схему переходов;
- изолировать production-state от test-like записей.

### 9.5. Reports / parsers / templates

Что хорошо:

- покрыты все основные управленческие отчёты;
- JSON и HTML в большинстве случаев генерируются последовательно;
- шаблоны пригодны для ежедневной эксплуатации.

Что плохо:

- высокая зависимость от структуры Excel и эвристик по колонкам;
- поля между parser/json/template не формализованы;
- нет единого schema contract для generated JSON.

Риски:

- тихая деградация после изменения формата файла из 1С;
- расхождения между шаблоном и ожидаемыми полями парсера.

Рекомендации:

- завести schemas/examples для generated JSON по типам отчётов;
- выделить smoke-check каждого отчётного типа на реальные sample файлы.

### 9.6. Scheduler / jobs

Что хорошо:

- scheduler богатый и закрывает реальные operational задачи;
- jobs разведены по времени;
- отдельные repeating jobs для janitor, reminders, poller.

Что плохо:

- в одном процессе сосредоточено очень много jobs;
- есть потенциальные overlap по времени: 10:00, 14:00, 17:00–18:00, 21:00–22:15;
- нет явной системы приоритетов или контроля длительности jobs, кроме отдельных таймаутов.

Риски:

- долгие задачи могут сдвигать соседние;
- сложнее расследовать, какая job вызвала cascade effect;
- бот как единый scheduler становится SPOF.

Рекомендации:

- описать и зафиксировать карту всех jobs в отдельном ops-document;
- вынести тяжёлые внешние jobs в отдельные процессы или supervised workers;
- добавить метрики длительности jobs.

### 9.7. Config / roles / managers / environment

Что хорошо:

- роли и менеджеры хранятся отдельно;
- `imap.json` и `roles.json` читаемы;
- environment flags активно используются.

Что плохо:

- конфигурация смешивает секреты, флаги, исторические пути и prompt-paths;
- `.env` не минимален и не разделён по зонам ответственности;
- часть файлов/логов всё ещё ссылается на старые machine-specific roots.

Риски:

- неправильный запуск на другой копии проекта;
- ложные конфиги после ручного редактирования;
- утечки секретов.

Рекомендации:

- выделить `.env.example`;
- отдельно документировать обязательные переменные;
- убрать абсолютные Windows paths из prompts/config/state.

### 9.8. State / JSON / runtime

Что хорошо:

- большинство state-файлов человеком читаемы;
- много атомарной записи;
- при расследовании это удобнее, чем непрозрачная embedded база.

Что плохо:

- JSON-state слишком много;
- связи между ними не описаны централизованно;
- legacy и current схемы живут параллельно.

Риски:

- drift схемы;
- накопление мусора;
- ошибочное manual editing;
- рост времени чтения и дедупликации.

Рекомендации:

- сделать карту state-файлов с owner-module и lifecycle;
- разделить durable state, cache, queue, audit-trace;
- завести миграционный подход для legacy ключей и файлов.

### 9.9. Logging / observability

Что хорошо:

- логов много;
- они действительно используются;
- есть явные warning/error/info по критичным событиям.

Что плохо:

- наблюдаемость всё ещё файловая и фрагментированная;
- нет единого summary по состоянию всех контуров;
- есть encoding noise и path leakage.

Риски:

- расследование крупных инцидентов занимает больше времени, чем должно;
- можно видеть симптом, но не видеть системную причину.

Рекомендации:

- добавить ежедневный operational digest;
- собрать статусы queues/pending/last success per job в один health-report;
- ввести журнал state-cleanup и state-migration отдельно от обычных логов.

## 10. Production readiness

### Уже надёжно

- IMAP cycle с whitelist и clean-copy.
- Основная генерация большинства отчётов.
- Atomic JSON write во многих критичных местах.
- Базовая ролевая модель и доступ к меню.
- Telegram polling и scheduler в одном процессе.

### Работает, но хрупко

- CRM flow.
- Collector flow.
- WhatsApp integration.
- Archive/index logic.
- Analytics generation and delivery.
- Stop-list control.

### Требует обязательной доработки

- безопасность секретов;
- разгрузка `bot/send_reports.py`;
- унификация state/contacts-схем;
- устранение legacy path pollution;
- выделение canonical pipeline orchestration.

### Желательно улучшить

- централизованный health/ops dashboard;
- schema validation generated JSON;
- явная документация запуска и окружения;
- throughput-control и reporting по CRM/collector.

## 11. Quick wins

- убрать production secrets из рабочей копии и перейти на защищённый secret storage;
- сделать единый документ запуска: что является canonical entrypoint и что нельзя запускать параллельно;
- завести `STATE_MAP.md` с owner, форматами и TTL каждого JSON-state;
- почистить path-dependent записи и прекратить использование absolute paths в state;
- отделить routers CRM и collector из `send_reports.py` без изменения пользовательского поведения;
- добавить ежедневный health summary админу: pipeline ok/no files, CRM pending count, collector pending count, WhatsApp poller alive, IMAP last success.

## 12. Обязательные исправления

1. Убрать секреты из репозитория/рабочего дерева как источник истины.
2. Назначить один canonical pipeline orchestrator.
3. Устранить конфликт контактов между `config/debtors_contacts.json` и `collector/debtors_contacts.json`.
4. Описать и нормализовать state lifecycle по CRM / collector / stop-list.
5. Декомпозировать `bot/send_reports.py` по функциональным модулям.
6. Развести production-state и тестовые/логические прогоны.
7. Стабилизировать dedupe на идентификаторе клиента, а не только на display/name.

## 13. План развития

### Краткосрочно

- security cleanup;
- canonical start/runbook;
- state map;
- collector/CRM incident visibility;
- вынос роутеров из `send_reports.py`.

### Среднесрочно

- модульная декомпозиция бота;
- единый state contract;
- унификация pipeline;
- schema tests для generated JSON и parsers.

### Долгосрочно

- переход от scattered JSON-state к более управляемому persistent layer;
- отдельные workers для heavy jobs;
- централизованная operational observability;
- формализация контрактов между report parsers, CRM и collector.

## 14. Итоговый приоритетный roadmap

### P0

- Удалить production secrets из рабочей копии и перевести их в безопасное хранение.
- Зафиксировать один production entrypoint и один production pipeline orchestrator.
- Нормализовать хранилища контактов и state ownership.
- Изолировать production runtime-state от тестовых данных.

### P1

- Разбить `bot/send_reports.py` на модули routing/scheduler/crm/collector/archive/analytics.
- Описать state lifecycle и cleanup policy.
- Добавить единый operational health-report.

### P2

- Ввести schema validation для generated JSON.
- Снизить количество повторных чтений больших JSON в hot-path.
- Улучшить CRM throughput и управленческую отчётность по backlog.

### P3

- Перевести часть долговременного state с ad-hoc JSON на более структурированное хранилище.
- Вынести тяжёлые или чувствительные jobs в отдельные supervised processes.

## 15. Финальный вывод

Проект зрелее обычного локального бота-автоматизатора и уже несёт серьёзную операционную нагрузку. Сильнее всего он выглядит там, где нужен прагматичный файловый automation: IMAP, парсинг, HTML/JSON, role-based delivery, operational reminders. Слабее всего он выглядит там, где поверх изначального отчётного бота нарастили новые контуры без полноценной архитектурной переразметки: CRM, collector, state lifecycle, scheduler multiplexing.

Главное ограничение роста не в качестве отдельных функций, а в накопленном архитектурном напряжении:

- один гигантский orchestrator;
- слишком много JSON-state;
- конфликт legacy и current flows;
- слабая граница между production, compatibility и test artefacts.

Если проекту нужен следующий качественный этап, то самый большой эффект даст не косметический рефакторинг, а три вещи:

1. security + config hygiene;
2. state normalization;
3. декомпозиция центрального бота на предсказуемые модули.

После этого система сможет масштабироваться и сопровождаться заметно дешевле. Без этого она ещё будет работать, но цена каждой новой функции и каждого расследования будет расти непропорционально.

## Приложение: перечень ключевых файлов и их роль

- `bot/send_reports.py` — главный Telegram-бот, scheduler и orchestration hub.
- `imap_fetcher.py` — получение Excel-вложений из IMAP.
- `run_pipeline_all_mp.py` — альтернативный orchestration pipeline.
- `config.py` — общие пути, TZ, логирование, managers config.
- `send_tg.py` — низкоуровневая отправка в Telegram.
- `bot/crm_clients.py` — CRM-база клиентов и обновление из отчётов.
- `bot/debt_stop_control.py` — stop-list, manager approvals, Saida flow.
- `bot/silence_alerts.py` — категоризация молчания и уведомления по дебиторке.
- `collector/collections_engine.py` — daily collector orchestration.
- `collector/debt_monitor.py` — чтение debt JSON и contact matching.
- `collector/dialog_store.py` — storage менеджерских collector-dialogs.
- `collector/manager_dialog.py` — manager-side Telegram FSM для collector.
- `collector/client_dialog.py` — client-side WhatsApp dialog FSM.
- `collector/communications.py` — WhatsApp/Telegram sending primitives.
- `collector/collections_db.py` — collector state helpers и pending/promises.
- `collector/whatsapp_poller.py` — polling входящих WhatsApp-сообщений из Green API.
- `debt_auto_report.py` — генерация дебиторских отчётов.
- `sales_report.py` — генерация sales HTML.
- `gross_report.py` / `gross_report_pct.py` — генерация валовой аналитики.
- `inventory.py` / `inventory_cost_parser.py` — отчёты по остаткам и себестоимости.
- `expenses_report.py` / `expenses_parser.py` — отчёты по затратам и JSON для net profit.
- `templates/*.html` — шаблоны управленческих отчётов.
- `config/roles.json` — роли и subadmin scopes.
- `config/managers.json` — manager → chat_id mapping.
- `config/imap.json` — IMAP-соединение, whitelist, корзины.
- `config/clients.json` — основная CRM-база клиентов.
- `config/debtors_contacts.json` — актуальный реестр collector-контактов.
- `collector/debtors_contacts.json` — legacy/alternate contact store, источник конфликтов.
- `logs/crm_pending_state.json` — текущая очередь CRM pending-кейсов.
- `logs/collector_dialogs.json` — активные manager dialogs collector.
- `logs/collector_client_dialogs.json` — активные клиентские WhatsApp-диалоги.
- `logs/collector_state.json` — долговременный collector state и legacy pending flags.
- `logs/notify_state.json` — dedupe и история уведомлений по отчётам.
- `logs/deletion_queue.json` — очередь автоудаления Telegram-сообщений.
- `reports/debt_stop_state.json` — суточное состояние stop-list.
- `reports/debt_stop_registry.json` — постоянный реестр stop-list решений.
- `tests/test_project.py` — smoke/integration checks основных модулей.
- `tests/test_parsers.py` — интеграционные проверки парсеров на реальных файлах.
- `tests/test_collector.py` — сценарные проверки collector-модулей.

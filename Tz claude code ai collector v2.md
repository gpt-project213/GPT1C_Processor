\# ТЗ ДЛЯ CLAUDE CODE — AI-Коллектор долгов с OpenClaw

\# Проект: GPT1C\_Processor / Минбаракат, Алматы

\# Версия ТЗ: 2.0 / Март 2026



\---



\## КОНТЕКСТ ПРОЕКТА (прочитай перед началом)



Корень проекта: E:\\GPT1C\_Processor\_analitic\\

GitHub: https://github.com/gpt-project213/GPT1C\_Processor (master)

Запуск бота: start\_bot\_watchdog.bat



\*\*Что уже работает:\*\*

\- `imap\_fetcher.py` — скачивает Excel из почты

\- `analyze\_debt\_excel.py` — парсит дебиторку → `reports/json/debt\_\*.json`

\- `debt\_auto\_report.py` — строит HTML отчёт

\- `ai\_analyzer.py` — DeepSeek API (ключ в .env)

\- `bot/send\_reports.py` — Telegram бот (BOT\_TOKEN, ADMIN\_CHAT\_ID в .env)

\- `config/managers.json` — {"Алена": 188939016, "Оксана": 1446255940, ...}



\*\*Инварианты — НИКОГДА не нарушать:\*\*

\- TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty")) — никогда timezone(timedelta(hours=5))

\- Все ключи в .env, .env в .gitignore

\- Версию файла +0.0.1 при каждом изменении

\- После каждого изменённого файла: python -m py\_compile <файл>

\- Не ломать существующую логику — только добавлять

\- Арман — уволен, нигде не упоминать

\- PDF в проекте нет и не будет



\---



\## ЧАСТЬ 1 — ПОЛНЫЙ АУДИТ ПЕРЕД НАЧАЛОМ РАБОТЫ



\*\*ТОЛЬКО ЧИТАТЬ. НИКАКИХ ИЗМЕНЕНИЙ НА ЭТОМ ЭТАПЕ.\*\*



Прочитай все .py файлы проекта локально и выполни полную проверку.

Не предполагай содержимое файлов — читай каждый файл напрямую.



\### 1.1 Аудит качества кода



Найди во всех .py файлах:



\*\*Критические:\*\*

\- `timezone(timedelta(hours=5))` вместо `ZoneInfo` — в каких файлах осталось?

\- `datetime.now()` без TZ — в каких файлах?

\- Упоминания "Арман"/"Arman" — в каких файлах и строках?

\- Упоминания "pdf"/"PDF" в активном коде — в каких файлах?

\- Несуществующие импорты — `from X import Y` где Y не существует в X

\- `bare except:` без типа исключения

\- Функции определённые в одном файле но вызываемые как несуществующие в другом



\*\*Серьёзные:\*\*

\- `sys.path.insert()` хаки

\- Хардкод путей вне config.py

\- Дублированные функции (определены более чем в одном файле)

\- `p.stat().st\_mtime` без try/except

\- `load\_dotenv()` отсутствует в файлах которые читают .env



\*\*Medium:\*\*

\- TODO/FIXME/HACK комментарии

\- Закомментированные блоки кода > 10 строк

\- Неиспользуемые импорты



\### 1.2 Аудит цепочек данных



Для каждого типа отчёта (DEBT, SALES, GROSS, INVENTORY, EXPENSES):

\- Excel → парсер → JSON → builder → HTML → send\_reports.py

\- Найди разрывы: где цепочка обрывается или файл не существует



\### 1.3 Аудит конфигурации



Прочитай `config/managers.json` и `config/roles.json`:

\- Все имена совпадают во всех файлах где используются?

\- Алена есть в обоих файлах с одним chat\_id?

\- Нет ли менеджеров в коде которых нет в конфигах?



\### 1.4 Аудит мёртвого кода



\- Файлы .py которые не импортируются и не вызываются нигде

\- Функции определённые но нигде не вызываемые

\- Зависимости в requirements.txt которые не импортируются ни в одном файле



\### 1.5 Формат вывода аудита



По каждому разделу — таблица:

```

Файл | Строка | Уровень (Critical/Serious/Medium/Info) | Описание | Статус (Открыт/Уже исправлен)

```



В конце — итоговый счётчик:

\- Critical открытых: N

\- Serious открытых: N

\- Medium открытых: N

\- Топ-5 самых опасных находок



\*\*После вывода аудита — ОСТАНОВИТЬСЯ и дождаться подтверждения.\*\*



\---



\## ЧАСТЬ 2 — ИСПРАВЛЕНИЕ БАГОВ ИЗ АУДИТА



После того как аудит выведен и подтверждён — исправить все найденные баги.



\*\*Правила исправления:\*\*

\- Только точечные правки — не переписывать логику

\- python -m py\_compile после каждого изменённого файла

\- Версия файла +0.0.1 каждому изменённому файлу

\- Запись в changelog/docstring что исправлено



\*\*После всех исправлений — один коммит:\*\*

```

powershell -ExecutionPolicy Bypass -File "E:\\GPT1C\_Processor\_analitic\\tools\\sync\_project\_to\_github.ps1" `

&#x20; -CommitMessage "fix: audit batch — все найденные баги закрыты"

```



\*\*После коммита — ОСТАНОВИТЬСЯ и дождаться подтверждения.\*\*



\---



\## ЧАСТЬ 3 — НОВЫЙ МОДУЛЬ: AI-КОЛЛЕКТОР ДОЛГОВ



\### 3.0 Что такое OpenClaw



OpenClaw (openclaw.ai) — open-source AI-агент платформа, 150k+ GitHub stars.

Это локальный gateway-процесс который:

\- Запускается на сервере (Windows поддерживается)

\- Подключается к мессенджерам (Telegram, WhatsApp)

\- Маршрутизирует входящие сообщения через AI-агента

\- Поддерживает skills/плагины

\- Работает с любой LLM через свой API ключ (DeepSeek, Claude, GPT)

\- Все данные хранит локально — ничего в облако

\- Gateway слушает на `ws://127.0.0.1:18789`



Для коллектора OpenClaw принимает ответы должников из WhatsApp/Telegram,

передаёт нашему агенту, агент анализирует и отвечает.



Если OpenClaw не установлен — модуль работает в standalone режиме

(отправляем, ждём callback/webhook от Green API или Telegram).



\### 3.1 Структура нового модуля



Создать папку: `E:\\GPT1C\_Processor\_analitic\\collections\\`



```

collections/

&#x20;   \_\_init\_\_.py

&#x20;   debt\_monitor.py          # анализ дебиторки, классификация риска

&#x20;   collections\_db.py        # хранение истории, обещаний, статусов

&#x20;   communications.py        # gateway: WhatsApp + Telegram + admin notify

&#x20;   collection\_agent.py      # AI-диалог (OpenClaw + DeepSeek)

&#x20;   voice\_calls.py           # Retell AI голосовые звонки

&#x20;   collections\_engine.py    # главный оркестратор + CLI

```



\### 3.2 config/debtors\_contacts.json



Создать справочник контактов:

```json

{

&#x20; "\_comment": "Ключ = имя клиента точно как в debt JSON",

&#x20; "ТОО Пример": {

&#x20;   "phone": "+77001234567",

&#x20;   "whatsapp": "+77001234567",

&#x20;   "telegram\_id": null,

&#x20;   "email": "example@mail.kz",

&#x20;   "contact\_person": "Иванов Иван",

&#x20;   "manager": "Алена",

&#x20;   "language": "ru",

&#x20;   "do\_not\_call": false,

&#x20;   "notes": "",

&#x20;   "openclaw\_session\_id": null

&#x20; }

}

```



`language`: "ru" или "kz" — агент общается на языке клиента.

`do\_not\_call`: true = только письменные каналы, без звонков.

`openclaw\_session\_id`: заполняется автоматически при первом контакте.



\### 3.3 collections/debt\_monitor.py v1.0.0



Анализирует дебиторку, классифицирует должников по уровням.



\*\*Уровни:\*\*

```

0–9 дней   → level 0 (пропустить)

10–14 дней → level 1 (мягкое напоминание)

15–19 дней → level 2 (среднее давление)

20–24 дней → level 3 (настойчиво)

25–29 дней → level 4 (строго + звонок)

30+ дней   → level 5 (жёстко + эскалация директору)

```



\*\*Функции:\*\*

\- `load\_latest\_debt\_json() -> dict` — последний по дате из `reports/json/debt\_\*.json`

\- `classify\_debtors(debt\_data) -> List\[dict]` — список с уровнями

\- `load\_contacts() -> dict` — читает `config/debtors\_contacts.json`

\- `match\_client(debt\_name: str, contacts: dict) -> Optional\[dict]`:

&#x20; 1. Прямое совпадение

&#x20; 2. Без "ТОО"/"ИП"/"АО"/"LLP" префиксов

&#x20; 3. lower().strip()

&#x20; 4. Совпадение по первым 3 словам

\- `get\_overdue\_days(client\_data) -> int`



TZ: `ZoneInfo(os.getenv("TZ", "Asia/Almaty"))`

Лог: `logs/debt\_monitor\_YYYYMMDD.log`



\### 3.4 collections/collections\_db.py v1.0.0



Память агента. Хранилище: `logs/collector\_state.json`



\*\*Структура записи:\*\*

```json

{

&#x20; "ТОО Пример": {

&#x20;   "last\_contact\_date": "2026-03-14",

&#x20;   "last\_contact\_channel": "whatsapp",

&#x20;   "last\_level": 2,

&#x20;   "last\_message\_text": "...",

&#x20;   "promise\_date": "2026-03-21",

&#x20;   "promise\_amount": 450000,

&#x20;   "promise\_kept": null,

&#x20;   "call\_result": null,

&#x20;   "call\_transcript": null,

&#x20;   "response\_received": true,

&#x20;   "last\_response\_text": "Заплатим в пятницу",

&#x20;   "openclaw\_session\_id": null,

&#x20;   "escalated\_to\_admin": false,

&#x20;   "history": \[]

&#x20; }

}

```



\*\*Функции:\*\*

\- `load\_state() -> dict`

\- `save\_state(state)` — атомарная запись через tempfile

\- `get\_client\_state(name) -> dict`

\- `update\_after\_contact(name, channel, level, message, response=None)`

\- `save\_promise(name, date, amount)`

\- `already\_contacted\_today(name) -> bool`

\- `get\_pending\_promises() -> List` — обещания срок которых прошёл



\### 3.5 collections/communications.py v1.0.0



Единый gateway для всех каналов отправки.



\*\*WhatsApp — Green API:\*\*

```python

\# POST https://api.green-api.com/waInstance{ID}/sendMessage/{TOKEN}

\# Из .env: GREENAPI\_ID, GREENAPI\_TOKEN

def send\_whatsapp(phone: str, text: str) -> bool

```



\*\*Telegram — существующий бот:\*\*

```python

\# BOT\_TOKEN уже в .env

async def send\_telegram(telegram\_id: int, text: str) -> bool

```



\*\*Эскалация директору:\*\*

```python

\# ADMIN\_CHAT\_ID уже в .env

async def notify\_admin(message: str) -> bool

```



\*\*Ограничения (жёсткие):\*\*

\- Не более 1 сообщения в день одному клиенту

\- Только 09:00–18:00 Asia/Almaty

\- Не в выходные (суббота, воскресенье)

\- При ошибке доставки — уведомить менеджера клиента, не падать молча



\### 3.6 collections/collection\_agent.py v1.0.0



AI-диалоговый агент.



\*\*Генерация текста через DeepSeek:\*\*

```python

def generate\_message(

&#x20;   client\_name: str,

&#x20;   debt\_amount: float,

&#x20;   days\_overdue: int,

&#x20;   level: int,

&#x20;   language: str,           # "ru" или "kz"

&#x20;   previous\_promise: Optional\[str] = None

) -> str

```



Промт для DeepSeek содержит:

\- Роль: официальный представитель компании Минбаракат

\- Данные: имя клиента, сумма долга, дней просрочки

\- Тон по уровню (см. ниже)

\- Язык: русский или казахский

\- Если previous\_promise — упомянуть что обещание не выполнено

\- Длина: 4-6 предложений, живая речь, не шаблон

\- Завершить вопросом о дате оплаты



\*\*Тон по уровням:\*\*

```

1: вежливо, партнёрский тон, без давления

2: нейтрально, показать что срок прошёл

3: настойчиво, чёткий запрос даты

4: строго, упомянуть возможные последствия

5: жёстко, последнее предупреждение перед юристом

```



\*\*Анализ ответа должника:\*\*

```python

def analyze\_response(response\_text: str) -> dict:

&#x20;   # Вызывает DeepSeek, возвращает:

&#x20;   {

&#x20;     "intent": "promise|refusal|delay\_request|question|unclear",

&#x20;     "promise\_date": "2026-03-21",  # или null

&#x20;     "promise\_amount": 450000,       # или null

&#x20;     "requires\_human": bool,

&#x20;     "suggested\_reply": str

&#x20;   }

```



\*\*Автоматические действия по intent:\*\*

\- `promise` → подтвердить, сохранить в DB

\- `refusal` → зафиксировать, эскалировать менеджеру

\- `delay\_request` → запросить новую дату

\- `question` → ответить (реквизиты, сумма)

\- `unclear` → уточняющий вопрос



\*\*OpenClaw интеграция:\*\*

\- Если `OPENCLAW\_ENABLED=true` — регистрировать skill в Gateway

\- Если `OPENCLAW\_ENABLED=false` — standalone режим (callback от Green API/Telegram)



\### 3.7 collections/voice\_calls.py v1.0.0



Голосовые звонки через Retell AI.



```python

\# POST https://api.retellai.com/v2/create-phone-call

\# Из .env: RETELL\_API\_KEY, RETELL\_AGENT\_ID, COMPANY\_PHONE



def initiate\_call(

&#x20;   phone: str,

&#x20;   client\_name: str,

&#x20;   debt\_amount: float,

&#x20;   days\_overdue: int,

&#x20;   level: int,

&#x20;   language: str

) -> dict  # {call\_id, status}



def get\_call\_result(call\_id: str) -> dict

\# {transcript, intent, promise\_date, duration}

```



\*\*Правила:\*\*

\- Звонить только при level >= 4 (настраивается через `COLLECTOR\_CALL\_LEVEL`)

\- Только 09:00–17:00 (не 18:00 — нужно время на разговор)

\- Не в выходные

\- `do\_not\_call=true` → пропустить, только письменные каналы

\- Максимум 1 звонок в день одному клиенту

\- Транскрипт сохранять в collections\_db



\### 3.8 collections/collections\_engine.py v1.0.0



Главный оркестратор.



```python

def run(dry\_run: bool = False, single\_client: str = None):

&#x20;   # 1. Загрузить debt JSON

&#x20;   # 2. Классифицировать должников по уровням

&#x20;   # 3. Для каждого:

&#x20;   #    - Найти контакты (match\_client)

&#x20;   #    - Проверить already\_contacted\_today → пропустить

&#x20;   #    - level == 0 → пропустить

&#x20;   #    - Нет контактов → логировать, пропустить

&#x20;   #    - Сгенерировать текст (generate\_message)

&#x20;   #    - Отправить: WhatsApp + Telegram (все уровни)

&#x20;   #    - level >= 4: + голосовой звонок

&#x20;   #    - level == 5: + notify\_admin

&#x20;   #    - Обновить collections\_db

&#x20;   # 4. Отправить daily\_summary → ADMIN\_CHAT\_ID



def check\_promises():

&#x20;   # Найти просроченные обещания оплаты

&#x20;   # Уведомить менеджера клиента

&#x20;   # Повысить level на 1



def daily\_summary() -> str:

&#x20;   # Всего должников: N

&#x20;   # Обработано сегодня: N

&#x20;   # Обещали оплату: N (список)

&#x20;   # Нарушили обещание: N (список)

&#x20;   # Не отвечают > 3 дней: N (список)

&#x20;   # Эскалировано: N

```



\*\*CLI:\*\*

```bash

python -m collections.collections\_engine --dry-run

python -m collections.collections\_engine --send

python -m collections.collections\_engine --send --client "ТОО Альфа"

python -m collections.collections\_engine --check-promises

```



\### 3.9 Добавить в .env



```bash

\# ── AI COLLECTOR ──────────────────────────────

GREENAPI\_ID=your\_instance\_id

GREENAPI\_TOKEN=your\_api\_token



RETELL\_API\_KEY=your\_retell\_key

RETELL\_AGENT\_ID=your\_agent\_id

COMPANY\_PHONE=+77XXXXXXXXX



OPENCLAW\_GATEWAY=ws://127.0.0.1:18789

OPENCLAW\_ENABLED=false



COLLECTOR\_HOUR\_START=9

COLLECTOR\_HOUR\_END=18

COLLECTOR\_CALL\_LEVEL=4

COLLECTOR\_DRY\_RUN=false

```



\### 3.10 Добавить в scheduler bot/send\_reports.py



Найди существующие scheduler jobs, добавь рядом:



```python

\# AI Debt Collector — 09:00 ежедневно

scheduler.add\_job(

&#x20;   run\_script,

&#x20;   trigger=CronTrigger(hour=9, minute=0, timezone=TZ),

&#x20;   args=\["collections/collections\_engine.py", "--send"],

&#x20;   id="debt\_collector\_daily",

&#x20;   name="AI Debt Collector"

)

\# Проверка обещаний — 10:00 ежедневно

scheduler.add\_job(

&#x20;   run\_script,

&#x20;   trigger=CronTrigger(hour=10, minute=0, timezone=TZ),

&#x20;   args=\["collections/collections\_engine.py", "--check-promises"],

&#x20;   id="debt\_collector\_promises",

&#x20;   name="AI Debt Collector — Promises"

)

```



\---



\## ЧАСТЬ 4 — OPENCLAW (опционально, только если пользователь подтвердит)



\### Установка на Windows:

```bash

\# Node.js LTS с nodejs.org (если нет)

npm install -g @openclaw/gateway

\# или: powershell -c "irm https://openclaw.ai/install.ps1 | iex"

```



\### Конфигурация:



`E:\\GPT1C\_Processor\_analitic\\openclaw\\SOUL.md`:

```markdown

Ты — AI-агент взыскания дебиторской задолженности компании Минбаракат.

Оптовые поставки продуктов питания, Алматы, Казахстан.

Ведёшь вежливый но настойчивый диалог с должниками.

НЕ угрожаешь, НЕ давишь эмоционально — ты официальный представитель.

При получении ответа: фиксируй обещание, уточняй если непонятно,

эскалируй если агрессия. Всегда профессионален.

```



`E:\\GPT1C\_Processor\_analitic\\openclaw\\skills\\debt\_collector.md`:

```markdown

\# Debt Collector Skill

При сообщении от должника:

1\. Определить намерение (promise/refusal/question/unclear)

2\. Если promise — зафиксировать дату через update\_debt\_status()

3\. Сформировать ответ по уровню клиента

```



\---



\## ЧАСТЬ 5 — ПРОВЕРКА И ФИНАЛЬНЫЙ КОММИТ



\### py\_compile всех новых файлов:

```bash

python -m py\_compile collections/\_\_init\_\_.py

python -m py\_compile collections/debt\_monitor.py

python -m py\_compile collections/collections\_db.py

python -m py\_compile collections/communications.py

python -m py\_compile collections/collection\_agent.py

python -m py\_compile collections/voice\_calls.py

python -m py\_compile collections/collections\_engine.py

```



\### Тест dry-run:

```bash

cd E:\\GPT1C\_Processor\_analitic

python -m collections.collections\_engine --dry-run

```

Ожидаемый вывод: список должников с уровнями, тексты сообщений, summary.



\### Финальный коммит:

```bash

powershell -ExecutionPolicy Bypass -File "E:\\GPT1C\_Processor\_analitic\\tools\\sync\_project\_to\_github.ps1" `

&#x20; -CommitMessage "feat: AI Debt Collector v1.0.0 — collections engine, OpenClaw ready, Retell AI, Green API WhatsApp"

```



\---



\## ИТОГО — ЧТО ПОЯВИТСЯ В ПРОЕКТЕ



```

E:\\GPT1C\_Processor\_analitic\\

├── collections\\

│   ├── \_\_init\_\_.py

│   ├── debt\_monitor.py        v1.0.0

│   ├── collections\_db.py      v1.0.0

│   ├── communications.py      v1.0.0

│   ├── collection\_agent.py    v1.0.0

│   ├── voice\_calls.py         v1.0.0

│   └── collections\_engine.py  v1.0.0

├── config\\

│   └── debtors\_contacts.json  (новый)

├── openclaw\\                  (если устанавливается)

│   ├── SOUL.md

│   └── skills\\debt\_collector.md

├── logs\\

│   └── collector\_state.json   (создаётся автоматически)

└── .env                       (дополнен)

```



\---



\## ЖЁСТКИЕ ОГРАНИЧЕНИЯ



1\. Звонки и сообщения — только 09:00–18:00 Asia/Almaty, не в выходные

2\. Максимум 1 сообщение в день одному должнику

3\. `do\_not\_call=true` → только WhatsApp/Telegram

4\. Все тексты генерирует DeepSeek — никаких хардкоженных шаблонов

5\. Данные должников — только локально, никогда не в GitHub

6\. Логи не пишут суммы и имена в открытом виде — только статусы

7\. При любой ошибке отправки — уведомить ADMIN\_CHAT\_ID, не падать молча

8\. `--dry-run` обязателен перед каждым `--send`

9\. После аудита (Часть 1) — остановиться, ждать подтверждения

10\. После исправления багов (Часть 2) — остановиться, ждать подтверждения


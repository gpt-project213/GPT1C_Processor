# 📊 GPT1C_OtchetBot — Project Status & Audit Log

> **Последнее обновление:** 06.03.2026  
> **Текущая версия бота:** `v9.4.31 / 05.03.2026`  
> **Репозиторий:** https://github.com/gpt-project213/GPT1C_Processor  
> **Рабочая директория:** любая (портативный) — см. раздел "Переносимость"

---

## 🏢 О проекте

AI-powered Telegram-бот для автоматической финансовой отчётности из 1С.  
Компания: оптовая дистрибуция продуктов питания (птица, говядина, рыба, морепродукты), Алматы, Казахстан.  
Оборот: 50–80 млн ₸/мес.

**Роли:**
- **Admin (Вадим)** — полный доступ, все отчёты всех менеджеров
- **Subadmin (Алена)** — видит: Магира + Оксана
- **Менеджеры:** Оксана, Магира, Ергали, Алена — каждый только свои данные

**Поток данных:** 1С → Excel → email (IMAP) → pipeline → HTML/JSON → Telegram

---

## 📦 Переносимость проекта (РЕШЕНО 06.03.2026)

### Принцип — проект уже портативный

Все скрипты используют:
```python
ROOT = Path(__file__).resolve().parent
```
Это означает: **куда бы ты не скопировал папку проекта — на диск D:\, E:\, F:\, флешку, другой ПК — всё работает автоматически.** Никаких настроек не нужно.

### Единственное исключение — `sales_report.py` (ИСПРАВЛЕНО)

**Было (v9.3.7):**
```python
OUT = _pick_dir([Path(r"E:\reports\html"), ROOT / "reports" / "html"])  # E:\ = жёсткий путь!
log_dir = Path(r"E:\logs")  # жёсткий путь!
```
`_pick_dir()` пробует первым `E:\reports\html`, успешно создаёт её через `mkdir`, возвращает. HTML пишется не туда куда нужно.

**Стало (v9.3.8, патч 06.03.2026):**
```python
OUT = _pick_dir([ROOT / "reports" / "html"])   # ✅ ROOT-relative
log_dir = ROOT / "logs"                         # ✅ ROOT-relative
```

### Канонический источник путей — `config.py`

`config.py v3.6` уже определяет **все** пути через `ROOT`:
```python
ROOT: Path = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
HTML_DIR    = ROOT / "reports" / "html"
JSON_DIR    = ROOT / "reports" / "json"
LOGS_DIR    = ROOT / "logs"
TEMPLATES_DIR = ROOT / "templates"
# ... и т.д.
```
Большинство скриптов импортируют из `config` — это правильно и портативно.

### Правило при создании новых скриптов

✅ ПРАВИЛЬНО:
```python
ROOT = Path(__file__).resolve().parent
# или
from config import HTML_DIR, JSON_DIR, LOGS_DIR
```

❌ НЕПРАВИЛЬНО:
```python
Path(r"E:\reports\html")   # жёсткий диск
Path(r"C:\Users\User\...")  # жёсткий путь
```

---

## 🗂️ Архитектура (канонический справочник)

### Основные скрипты (root папки)
| Файл | Версия | Назначение |
|------|--------|------------|
| `send_reports.py` | v9.4.31/05.03.2026 | Главный файл, планировщик, pipeline, меню, роутинг |
| `config.py` | v3.6/08.09.2025 | Единый источник всех путей, TZ, логирование |
| `sales_report.py` | **v9.3.8**/06.03.2026 | HTML продаж (пофикшен Bug #S1) |
| `sales_parser.py` | — | Парсинг Excel продаж → JSON |
| `utils_excel.py` | v2.3.2/19.02.2026 | Очистка XLSX, ensure_clean_xlsx |
| `ai_analyzer.py` | v9.4.7 | AI-анализ через DeepSeek / OpenAI fallback |
| `gross_report.py` | v27.7/27.09.2025 | Валовая прибыль (ABC-анализ, маржа) |
| `gross_report_pct.py` | — | Валовая прибыль в % |
| `inventory.py` | — | Остатки на складах |
| `debt_auto_report.py` | — | Дебиторская задолженность |
| `analyze_debt_excel.py` | v2.1 | Парсер дебиторки |
| `dso_aging_report.py` | v1.0.2/09.02.2026 | DSO + aging (⚠️ 3 бага логики) |
| `revenue_concentration_report.py` | v1.1.1/26.02.2026 | Концентрация выручки (🔴 CRASH) |
| `sales_profitability_report.py` | v1.0.1/09.02.2026 | Продажи + рентабельность ✅ |
| `imap_fetcher.py` | v4.4.3/18.02.2026 | IMAP-клиент |
| `expenses_parser.py` | — | Парсер затрат |
| `run_new_reports_now.py` | — | Ручной запуск доставки |
| `cleanup_cache.py` | — | Автоочистка файлов |

### Подпапка bot/
| Файл | Назначение |
|------|------------|
| `bot/send_reports.py` | Копия/версия главного файла для Git |
| `bot/sales_summary.py` | Сводка продаж |
| `bot/gross_summary.py` | Сводка валовой |
| `bot/inventory_summary.py` | Сводка остатков |
| `bot/opportunity_loss.py` | Упущенная прибыль |
| `bot/silence_alerts.py` | Алерты молчания |
| `bot/user_tracker.py` | Трекинг активности |

### Аналитические скрипты (weekly_analytics_job, ежедневно 22:00)
- `sales_profitability_report.py` ✅
- `net_profit_report.py`
- `inventory_turnover_report.py`
- `rfm_clients_report.py`
- `revenue_concentration_report.py` ← **CRASH** NameError
- `dso_aging_report.py` ← **3 бага логики**

### Расписание (v9.4.31)
| Время | Задача |
|-------|--------|
| каждые 10 мин | pipeline (IMAP → обработка очереди) |
| каждые 2 мин | ai_queue_processor |
| 09:00 | краткая сводка остатков |
| 14:00, 21:00 | проверка дней молчания |
| 14:05, 21:05 | упущенная прибыль |
| 20:00 | краткая сводка валовой |
| 21:00 | краткая сводка продаж |
| 22:00 | weekly_analytics (6 аналитических скриптов) |
| 23:00 | ежедневная сводка admin |
| 00:01 | сброс счётчика AI-генераций |
| Пн 10:00 | еженедельная AI генерация (вручную) |
| 03:00 | автоочистка файлов |

### Структура директорий (ROOT-относительная)
```
<любая папка проекта>/        ← ROOT = здесь лежит скрипт
├── .env                      ← TG_BOT_TOKEN, DEEPSEEK_API_KEY, ADMIN_CHAT_ID
├── config/
│   ├── managers.json         ← источник истины по менеджерам
│   ├── roles.json
│   ├── imap.json
│   └── pattern_config.yaml
├── reports/
│   ├── queue/                ← входящие Excel от IMAP
│   ├── excel/clean/          ← __clean.xlsx копии
│   ├── html/                 ← готовые HTML отчёты
│   ├── json/                 ← промежуточные JSON
│   ├── ai/                   ← AI анализ (.txt / .html)
│   ├── analytics/            ← DSO, концентрация, рентабельность
│   └── archive/              ← архив обработанных файлов
├── logs/                     ← ротируемые логи (в .gitignore)
└── templates/                ← Jinja2 HTML шаблоны
```

---

## ✅ Статус компонентов на 06.03.2026

| Компонент | Статус | Подтверждение |
|-----------|--------|---------------|
| IMAP получение почты | ✅ Работает | 31 файл за 1 цикл (12:27:26) |
| Дебиторка (simple + extended) | ✅ Работает | 8 HTML, доставлены всем ролям |
| Дни молчания | ✅ Работает | 3 молчащих клиента у Алена |
| Валовая прибыль | ✅ Работает | HTML период 05.03.2026 |
| Остатки (inventory) | ✅ Работает | HTML + сводка 09:00 |
| Затраты (expenses) | ✅ Работает | Затраты (52).xlsx обработан |
| sales_parser.py → JSON | ✅ Работает | 6 JSON за pipeline |
| **sales_report.py → HTML** | ✅ **ИСПРАВЛЕН** | Патч v9.3.8 06.03.2026 |
| sales_profitability_report | ✅ Работает | 146 клиентов, 26 товаров маржи |
| AI генерация | ⚠️ Только вручную | Намеренно (Weekly AI mode) |
| revenue_concentration_report | 🔴 CRASH | NameError: normalize_client_name |
| DSO / aging | 🔴 Логические баги | Неверный расчёт периода и aging |

---

## 🐛 Открытые баги

### ✅ Bug #S1 — sales_report.py жёсткие пути (ЗАКРЫТ 06.03.2026)
**Причина:** `E:\reports\html` первым в `_pick_dir()` — mkdir успешно создавал там папку  
**Фикс:** v9.3.8 — все пути переведены на `ROOT / "reports" / "html"` и `ROOT / "logs"`  
**Файл:** `sales_report.py` v9.3.8 готов к деплою

---

### 🔴 Bug #RC1 — revenue_concentration_report.py CRASH
**Файл:** `revenue_concentration_report.py` v1.1.1  
**Симптом:** Падает дважды в 22:00 (weekly_analytics):
```
NameError: name 'normalize_client_name' is not defined
  line 122: key = normalize_client_name(str(c.get("client", "")))
```
**Контекст:** Функция `normalize_client_name` существует в `sales_profitability_report.py` но **не определена** в `revenue_concentration_report.py`. Либо забыли скопировать, либо потерялась при рефакторинге.  
**Фикс:** добавить `normalize_client_name` в файл (можно импортировать или определить inline)

---

### 🔴 Bug #DSO-1 — Делитель 30 вместо реального периода
**Файл:** `dso_aging_report.py` v1.0.2  
**Симптом:** DSO = долг / (выручка / 30) — фиксированный месяц вместо реального периода файла.  
Результат: значения 600+ дней при реальных 15-30.

### 🔴 Bug #DSO-2 — Синтетические коэффициенты aging
**Симптом:** Распределение по бакетам (<7д, 8-14д, 15-30д, >30д) через искусственные множители вместо реальных дат документов из JSON.

### 🔴 Bug #DSO-3 — Выбор файла продаж по max(revenue)
**Симптом:** Берётся файл продаж с максимальной суммой вместо совпадения периода с дебиторкой.

---

### ✅ Bug #B1 — Дубли строк в расходах (ЗАКРЫТ 06.03.2026)
Патч в `expenses_parser.py`: дедупликация по rowkey.

### ✅ Bug #B2 — sales_parser не читает managers.json (ЗАКРЫТ 06.03.2026)
Исправлен путь к `managers.json` в `sales_parser.py`.

---

## 🔐 .gitignore — что включить

```gitignore
# Логи и временные данные
logs/
reports/queue/
reports/excel/
reports/html/
reports/json/
reports/ai/
reports/archive/
reports/analytics/

# Секреты
.env
*.env

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/

# Данные 1С
*.xlsx
*.xls

# Системное
.DS_Store
Thumbs.db
```

**Что ДОБАВИТЬ в репозиторий:**
- Весь Python-код ✅
- `templates/*.html` ✅
- `config/*.json` и `*.yaml` (без секретов) ✅
- `requirements.txt` ✅
- `.gitignore` ✅
- `PROJECT_STATUS.md` (этот файл) ✅

**Что НЕ добавлять:**
- `.env` (токены/ключи)
- `logs/` (корпоративные метаданные)
- `reports/` (бизнес-данные)

---

## 📋 История аудит-сессий

### Сессия 06.03.2026 — основная диагностика

**Загружено:** `send_reports_20260306.log`, `sales_profitability_20260306_*.log`, `sales_report.py`, `raw_links.txt`

**Выявлено:**
1. Bug #S1 — `sales_report.py` писал HTML в `E:\reports\html\` → **ИСПРАВЛЕН** (v9.3.8)
2. Bug #RC1 — `revenue_concentration_report.py` NameError → **открыт**
3. Портативность: все скрипты кроме `sales_report.py` уже ROOT-relative → **подтверждено**

**Подтверждено работающим:**
- Pipeline IMAP (31 файл за цикл)
- Дебиторка, валовая, остатки, затраты — все HTML создаются
- sales_parser.py → JSON для всех 6 менеджерских файлов
- sales_profitability: 146 клиентов, 464 matched товара

### Сессия 06.03.2026 — аудит архива (GPT1C_FULL_AUDIT.zip)

**Загружено:** `GPT1C_FULL_AUDIT.zip` (3118 файлов логов, 8 Python-скриптов)

**Выявлено:**
- AI генерация отключена намеренно с v9.4.8 (Weekly AI mode)
- Последняя AI генерация: 26.02.2026 (ai_generation_state.json)
- Бот остановлен 01.03.2026 в 19:13 штатно
- weekly_analytics 01.03.2026: 6/6 скриптов успешно

---

## 🔮 Следующие шаги (приоритет)

| # | Задача | Приоритет |
|---|--------|-----------|
| 1 | **Деплой sales_report.py v9.3.8** — заменить файл на сервере | 🔴 Критический |
| 2 | **Фикс Bug #RC1** — добавить `normalize_client_name` в revenue_concentration | 🔴 Критический |
| 3 | **Фикс Bug #DSO-1,2,3** — переписать DSO на реальные данные | 🔴 Высокий |
| 4 | Проверить почему sales_profitability дважды запускается (22 сек) | ⚠️ Средний |
| 5 | Обновить `STATUS.md` в GitHub после деплоя | ⚠️ Средний |

---

## 💬 Стартовый бриф для нового чата

### Контекст
Вадим — директор оптовой дистрибуции, Алматы. Python Telegram-бот для финансовой отчётности из 1С.  
Репо: `https://github.com/gpt-project213/GPT1C_Processor`  
Production: `send_reports.py v9.4.31/05.03.2026` — файл находится в корне рабочей папки.

### Правила работы
1. Не ломать, не упрощать без согласования, не усекать, не выдумывать
2. Строить, улучшать, совершенствовать
3. Согласовывать изменения перед внедрением
4. Успешный тест = подтверждён в логах

### Горячие задачи прямо сейчас
1. **Деплой `sales_report.py v9.3.8`** — файл готов (`/mnt/user-data/outputs/sales_report.py`). Скопировать в рабочую папку, перезапустить бота, проверить что `sales_grouped_*.html` создаётся в правильном месте.

2. **Bug #RC1 в `revenue_concentration_report.py`** — нужен файл из репо, добавить функцию `normalize_client_name`.

### Что работает (не трогать)
IMAP ✅ | Дебиторка ✅ | Валовая ✅ | Остатки ✅ | Затраты ✅ | sales_parser→JSON ✅ | sales_profitability ✅

### AI генерация
Отключена **намеренно** — только вручную через меню бота. Не баг.

### Версии
| Компонент | Версия | Дата |
|-----------|--------|------|
| send_reports.py | v9.4.31 | 05.03.2026 |
| sales_report.py | v9.3.8 (патч) | 06.03.2026 |
| utils_excel.py | v2.3.2 | 19.02.2026 |
| config.py | v3.6 | 08.09.2025 |
| imap_fetcher.py | v4.4.3 | 18.02.2026 |

---

## 🔍 Аудит портативности — 06.03.2026 (полный скан 36 файлов)

### Результат
```
✅ Полностью портативны: 34 файла
⚠️  CWD-зависимые:        1 файл  (cleanup_cache.py)  → ИСПРАВЛЕН
🔴 Жёсткий путь на диск: 1 файл  (sales_report.py)   → ИСПРАВЛЕН
```

### Принцип портативности в проекте

Все скрипты используют одну из двух стратегий:

**Стратегия A — собственный ROOT:**
```python
ROOT = Path(__file__).resolve().parent
LOG_DIR  = ROOT / "logs"
HTML_DIR = ROOT / "reports" / "html"
```

**Стратегия B — импорт из config.py:**
```python
from config import HTML_DIR, JSON_DIR, LOGS_DIR, setup_logging
```
`config.py` сам определяет `ROOT = Path(__file__).resolve().parent` и экспортирует все пути.

### Ключевой момент: `Path(__file__)` vs `Path("logs")`

| Подход | Работает при запуске из | Переносим? |
|--------|------------------------|------------|
| `Path(__file__).resolve().parent / "logs"` | любой директории | ✅ Да |
| `Path("logs")` | только из папки проекта | ⚠️ CWD-зависим |
| `Path(r"E:\reports\html")` | только на диске E:\ | 🔴 Нет |

`Path(__file__)` — это путь к **самому файлу скрипта**, не зависит от того откуда его запускают. Можно перемещать папку на любой диск, флешку, другой ПК — путь всегда вычисляется правильно.

### Файлы исправленные в патче
| Файл | До | После | Версия патча |
|------|----|-------|--------------|
| `sales_report.py` | `E:\reports\html`, `E:\logs`, `E:\templates` | `ROOT / "reports/html"` etc. | v9.3.8 / 06.03.2026 |
| `cleanup_cache.py` | `Path("logs")`, `Path("reports/html")` | `ROOT / "logs"` etc. | v1.0.1 / 06.03.2026 |
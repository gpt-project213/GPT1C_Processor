# v. 9.4.33 / 2026-03-10 - Fix: bare except: → except (ValueError, OverflowError) в _parse_period_date (Bug S5)
# v. 9.4.32 / 09.03.2026 - Fix bugs: #INV-1, #MENU-SILENCE, #AI-MENU, упущенная прибыль → еженедельно (пятница 14:05)
# ИЗМЕНЕНИЯ v9.4.32 / 09.03.2026:
# - ИСПРАВЛЕНО Bug B1: weekly_ai_generation использовал неопределённую переменную
#   managers_to_process → заменено на get_managers_list() (NameError при каждом запуске)
# - ИСПРАВЛЕНО Bug B2: handle_extended_with_ai передавал неопределённую переменную
#   report_type в process_and_send_ai_analysis → захардкожено "DEBT" (верное значение)
# - ИСПРАВЛЕНО Bug B3: handle_analytics для subadmin отправлял ВСЕ DSO/RFM/concentration
#   файлы включая чужих менеджеров (Алена видела Ергали). Добавлена фильтрация по scopes.
#
# ИЗМЕНЕНИЯ v9.4.30:
# - ИСПРАВЛЕНО Bug #11: _classify_type не распознавал INVENTORY_SIMPLE из-за несоответствия
#   паттернов: в HTML именах used underscores ("ведомость_по_товарам"), а искали пробелы
#   ("ведомость по товарам"). Также "остатки" ≠ "остаток". Добавлены оба варианта.
#
# ИЗМЕНЕНИЯ v9.4.29:
# - ДОБАВЛЕНО: B — кнопки принудительной отправки в меню аналитики (все роли)
#   5 отчётов: Дебиторка, Упущенная прибыль, Продажи, Валовая, Остатки
#   Логика по ролям: admin→сводка всех, subadmin→команда, manager→свои данные
# - ИЗМЕНЕНО: пороги зон в opportunity_loss.py: ⚡7-15д / 🔴15-30д / ☠️30+д
# v. 9.4.26 / 27.02.2026 - Упущенная прибыль (opportunity_loss) + alert чистой прибыли
# ИЗМЕНЕНИЯ v9.4.26:
# - ДОБАВЛЕНО: opportunity_loss.py — расчёт упущенной прибыли по молчащим должникам
#   Формула: долг × маржа%; зоны: ⚡15-60д / 🔴60-120д / ☠️120+д
#   Джобы: 14:05 и 21:05 (через 5 мин после silence_alerts)
#   Кому: admin (сводка всех), subadmin (себя + подчинённых), менеджер (свои)
# - ДОБАВЛЕНО: G — alert admin когда net_profit_report не смог рассчитать (нет expenses)
# v. 9.4.25 / 27.02.2026 - Подекадные уведомления + IMAP alert + субадмин-рейтинг команды
# ИЗМЕНЕНИЯ v9.4.23:
# - ИСПРАВЛЕНО: Admin получал рейтинг продаж только с 1 менеджером (первым в очереди)
#   Теперь: финальный рейтинг отправляется ПОСЛЕ pipeline цикла, когда все JSON готовы
# - ИСПРАВЛЕНО: manager_dates не передавался в format_admin_detailed() → дат не было в silence
#   Теперь: manager_dates собирается в первом цикле и передаётся в admin-сводку
# - ДОБАВЛЕНО: _pipeline_sent_admin_periods — дедупликация по периоду внутри pipeline цикла
# ИЗМЕНЕНИЯ v9.4.22:
# - ИСПРАВЛЕНО: SensitiveDataFilter в логах + silence_alerts manager_dates
# ИЗМЕНЕНИЯ v9.4.21:
# - НОВОЕ: SensitiveDataFilter — маскирует chat_id, BOT_TOKEN, email в логах автоматически
# - ИСПРАВЛЕНО: format_admin_detailed() вызывался без manager_dates → дат не было в сводке
# - ОПТИМИЗАЦИЯ: get_latest_debt_report() вызывается 1 раз на менеджера (был 2 раза)
# ИЗМЕНЕНИЯ v9.4.20:
# - ИСПРАВЛЕНО Bug #B: logger.error() в ImportError блоках (строки ~157,169) использовался ДО
#   определения logger (строка ~282) → NameError при краше импорта → заменено на print()
# - ИСПРАВЛЕНО Bug #C: ADMIN_CHAT_ID=0 при отсутствии .env переменной — тихий сбой всех
#   уведомлений admin → добавлена явная проверка и предупреждение при старте
# - ИСПРАВЛЕНО Bug #D: ANALYTICS_DIR.glob("*.html") не захватывал поддиректории
#   net_profit_day/ и net_profit_mtd/ → заменено на rglob в post_init и cleanup_old_files
# - ДОБАВЛЕНО: cleanup_old_files теперь чистит ANALYTICS_DIR рекурсивно (файлы старше 30 дней)
# ИЗМЕНЕНИЯ v9.4.19:
# - ИСПРАВЛЕНО Bug #12: gender_emoji() всегда возвращал "👤" — GENDER_MAP хранил эмодзи но сравнивал с "m"/"f"
#   Теперь: Алена/Оксана/Магира → 👩, Ергали → 👨
# - ИСПРАВЛЕНО Bug #13: меню не всегда было внизу при отправке файлов через direct|
#   Теперь: старое меню удаляется, файл отправляется, новое меню создаётся внизу (как в DEMO)
# - ИСПРАВЛЕНО Bug #14: archive|get не отправлял новое меню после файла и не удалял старое
#   Теперь: удаляем архив-меню → отправляем файл → отправляем главное меню внизу
# - ОБНОВЛЕНО: версия v9.4.16 → v9.4.19 (синхронизация строки __VERSION__)
# v. 9.4.18 / 22.02.2026 - Исправление дат в кратких уведомлениях + поддиректории net_profit
# ИЗМЕНЕНИЯ v9.4.18:
# - ИСПРАВЛЕНО: inventory_summary, sales_summary, gross_summary — сортировка по периоду данных (не по mtime)
# - ИСПРАВЛЕНО: silence_alerts — сортировка по периоду данных + дата отчёта в уведомлении
# - РЕАЛИЗОВАНО: подменю "Чистая прибыль" → [📅 За день] / [📆 За период]
# - РЕАЛИЗОВАНО: net_profit_day → reports/analytics/net_profit_day/, net_profit_mtd → reports/analytics/net_profit_mtd/
# - ДОБАВЛЕН: обработчик analytics_menu callback (кнопка "Назад" в подменю чистой прибыли)
# ИЗМЕНЕНИЯ v9.4.16:
# - ИСПРАВЛЕНО Bug #9: handle_analytics отправляет ВСЕ DSO/RFM/concentration файлы
# - ИСПРАВЛЕНО Bug #11: send_with_acl fallback на Сводный отчёт для SALES_SIMPLE/EXTENDED
# - ИСПРАВЛЕНО Bug #10: меню всегда появляется после отправки файла (reply_markup в send_with_acl)
# - ДОБАВЛЕНО: Кнопка "🔄 Обновить аналитику" для admin в меню аналитики
# - ДОБАВЛЕНО: Уведомление admin + subadmin после генерации аналитики
# - ИЗМЕНЕНО: Аналитика генерируется ежедневно в 22:00 (было: только по понедельникам)
#
# v. 9.4.14 / 19.02.2026 - ФАЗА 1: ИСПРАВЛЕНИЕ PIPELINE + EXPENSES_PARSER
# ИЗМЕНЕНИЯ v9.4.14:
# - ИСПРАВЛЕНО: utils_excel v2.3.2 больше не удаляет оригинал из queue —
#   pipeline_task теперь находит файлы и запускает парсеры автоматически
# - ДОБАВЛЕНО: expenses_parser.py запускается после expenses_report.py в pipeline
#   (ранее expenses JSON не создавался → net_profit_report не получал данные)
#
# ИЗМЕНЕНИЯ v9.4.13:
# - ИСПРАВЛЕНО: Индекс сортируется по ПЕРИОДУ (primary), затем по mtime (secondary)
# - ДОБАВЛЕНО: Детальное логирование выбора файлов (period, mtime, path)
# - ДОБАВЛЕНО: Перекрёстное сопоставление периодов для net_profit (gross + expenses)
# - ИСПРАВЛЕНО: imap_fetcher.py пропускал файлы затрат (добавлено в summary_keywords)
#
# v. 9.4.12 / 18.02.2026 - UX: PERSISTENT MENU + ПЕРИОДЫ В CAPTION + ДЕТАЛЬНЫЕ ЛОГИ
# ИЗМЕНЕНИЯ v9.4.12:
# - ДОБАВЛЕНО: Persistent menu (постоянное меню внизу) — кнопки всегда видны
# - УЛУЧШЕНО: Caption при отправке файлов содержит период/дату (expenses, analytics)
# - ДОБАВЛЕНО: Детальное логирование выбора файлов (slug, period, путь)
# - ИСПРАВЛЕНО: Обработчик текстовых команд от persistent menu
#
# v. 9.4.11 / 18.02.2026 - ДОБАВЛЕНЫ ЗАТРАТЫ (меню+пайплайн+индекс)
# ИЗМЕНЕНИЯ v9.4.10:
# - ИСПРАВЛЕНО: Парсеры JSON запускаются после каждого отчёта (sales, gross, inventory)
# - ИСПРАВЛЕНО: Краткие сводки ТОЛЬКО АДМИНУ (убраны менеджеры из inventory+sales)
# - ИСПРАВЛЕНО: Автоудаление текстовых сводок через 24ч
# - ДОБАВЛЕНО: weekly_analytics запускается при старте в понедельник (если пропущен)
# - ИСПРАВЛЕНО: run_script используется вместо run_script_async в weekly_analytics_job
#
# v. 9.4.9 / 09.02.2026 - ПОЛНАЯ ИНТЕГРАЦИЯ АНАЛИТИКИ
# ИЗМЕНЕНИЯ v9.4.9:
# - ДОБАВЛЕНО: Кнопка АНАЛИТИКА в главном меню  
# - ДОБАВЛЕНО: Команда /analytics с контролем доступа
# - ДОБАВЛЕНО: 6 аналитических отчётов
# - ДОБАВЛЕНО: Еженедельная генерация (понедельник 10:00)
#
# v. 9.4.7.6 / 18.11.2025
# КРИТИЧНЫЕ ИСПРАВЛЕНИЯ v9.4.7.6:
# - ИСПРАВЛЕНО: Правильная последовательность проверок в schedule_ai_generation()
# - ИСПРАВЛЕНО: Проверка last_processed_dates в process_ai_generation_queue() (не в schedule!)
# - ДОБАВЛЕНО: extract_date_from_filename() - извлечение даты из имени файла
# - ДОБАВЛЕНО: Проверка возраста файла (не старше 24 часов) перед планированием
# - ДОБАВЛЕНО: Запоминание file_date в state["last_processed_dates"]
# - ДОБАВЛЕНО: Сохранение file_date в очередь для точной идентификации
# - УЛУЧШЕНО: Batch-логирование cash-отчётов (экономия ~240 строк логов/час)
# - ДОБАВЛЕНО: Новые события в EMOJI_LOG_MAP
# - ДОБАВЛЕНО: Автоочистка старых файлов (логи 2д, AI 7д, HTML 30д, JSON 7д, Excel 14д)
#
# v. 9.4.7.1 / 14.11.2025
# КРИТИЧНЫЕ ИСПРАВЛЕНИЯ v9.4.7.1:
# - ИСПРАВЛЕНО: Добавлен log_user_delivery при успешной отправке отчёта (критично для статистики!)
# - ИСПРАВЛЕНО: Исправлены отступы в проверках безопасности (строки 1285-1292, было SyntaxError!)
#
# КРИТИЧНЫЕ ИСПРАВЛЕНИЯ v9.4.6.1:
# - ИСПРАВЛЕНО: post_init теперь регистрируется через builder (критично!)
# - ИСПРАВЛЕНО: Атомарная запись notify_state.json - защита от race condition
# - ИСПРАВЛЕНО: Версия в логах изменена на v9.4.6
# - ИЗМЕНЕНО: Janitor интервал с 60 сек на 60 мин (менее нагрузка на систему)
# - УПРОЩЕНО: Удалены избыточные проверки "Арман" (его нет в конфиге)
#
# УЛУЧШЕНИЯ v9.4.6.2:
# - ИСПРАВЛЕНО: AI-кэш теперь ищется и в AI_DIR, и в HTML_DIR
# - ИСПРАВЛЕНО: Архив поддерживает разные разделители дат (' – ', ' - ', '—')
# - ДОБАВЛЕНО: Версия отображается в /health для диагностики
# - УЛУЧШЕНО: _extract_manager использует динамический список менеджеров из конфига
# - УЛУЧШЕНО: _classify_type нормализует ё→е для устойчивого распознавания файлов
#
# ИЗМЕНЕНИЯ v9.4.6 (на базе v9.4.5) - КРИТИЧНЫЕ ИСПРАВЛЕНИЯ НАДЁЖНОСТИ:
# - ИСПРАВЛЕНО: chat_id в managers.json теперь нормализуется к int при загрузке (критичный баг!)
# - ИСПРАВЛЕНО: Атомарная запись JSON через tempfile - защита от гонок и потери данных
# - ИСПРАВЛЕНО: 48-часовой лимит считается от реального времени сообщения (msg_ts), а не scheduled_at
# - ДОБАВЛЕНО: Дедупликация задач удаления - повторные клики не создают дубли
# - ДОБАВЛЕНО: Ограничение размера очереди (5000 задач) - защита от утечки памяти
# - Все изменения улучшают надёжность БЕЗ ПОЛОМКИ функционала
#
# ИЗМЕНЕНИЯ v9.4.5 (на базе v9.4.4):
# - УДАЛЕНО: Арман уволен - убран из всех меню, списков, регулярок
# - ДОБАВЛЕНО: Автоудаление отчётов через 24 часа (для безопасности)
# - ДОБАВЛЕНО: protect_content=True для всех отчётов (запрет пересылки/сохранения)
# - ДОБАВЛЕНО: Janitor-система для восстановления задач на удаление после перезапуска
# - Сохранение задач на удаление в deletion_queue.json
# - Фоновый джоб каждые 60 минут проверяет и удаляет просроченные сообщения
# - Ограничение: можно удалить только сообщения моложе 48 часов (лимит Telegram API)
# ИЗМЕНЕНИЯ v9.4.11:
# - ДОБАВЛЕНО: Раздел "💸 Затраты" в главном меню (admin/subadmin) + подменю (за день/за период)
# - ДОБАВЛЕНО: Подключение expenses_report.py в pipeline (распознавание по имени файла)
# - ДОБАВЛЕНО: Индексация HTML затрат (EXPENSES) для архива
# - ДОБАВЛЕНО: Отправка последнего отчёта затрат по кнопкам day/period (по report_type из JSON)
#

# Блок 1_______________Импорты и настройка окружения_________________________
import os
import sys
import re
import json
import time
import asyncio
import logging
import shutil
import subprocess
from tempfile import NamedTemporaryFile
from pathlib import Path

# --- Path bootstrap: allow importing project-root modules when running as bot/send_reports.py ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__VERSION__ = "v9.4.32/09.03.2026"

from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ──────────────────────────────────────────────────────────────────
# Persistent Menu (v9.4.12)
def kb_persistent() -> ReplyKeyboardMarkup:
    """Постоянное меню внизу (всегда видимое)"""
    return ReplyKeyboardMarkup([
        [KeyboardButton("📊 Дебиторка"), KeyboardButton("🛒 Продажи")],
        [KeyboardButton("💰 Валовая"), KeyboardButton("💸 Затраты")],
        [KeyboardButton("📦 Остатки"), KeyboardButton("📈 Аналитика")],
        [KeyboardButton("🗄️ Архив")]
    ], resize_keyboard=True)
from telegram.error import BadRequest, RetryAfter
from silence_alerts import SilenceAlert
# v2.0: Мобильная адаптивность и аналитика
try:
    from user_tracker import track_user, track_action, get_stats, format_stats_message
except ImportError as e:
    print(f"⚠️ [STARTUP] Модуль user_tracker не найден: {e}")  # logger ещё не создан на этом этапе
    track_user = None
    track_action = None
    get_stats = None
    format_stats_message = None

# v9.4.8: Модули кратких сводок
try:
    from inventory_summary import InventorySummary
    from sales_summary import SalesSummary, detect_period_type as _sales_detect_period, PERIOD_DAY as _PERIOD_DAY
    from gross_summary import GrossSummary
except ImportError as e:
    print(f"⚠️ [STARTUP] Модули кратких сводок не найдены: {e}")  # logger ещё не создан на этом этапе
    InventorySummary = None
    SalesSummary = None
    GrossSummary = None

# v9.4.26: Модуль упущенной прибыли
try:
    from opportunity_loss import (
        calculate_opportunity_loss,
        format_opportunity_loss_message,
        format_opportunity_loss_admin,
        format_opportunity_loss_subadmin,
    )
    _OPPORTUNITY_LOSS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ [STARTUP] opportunity_loss не найден: {e}")
    _OPPORTUNITY_LOSS_AVAILABLE = False
    calculate_opportunity_loss = None
    format_opportunity_loss_message = None
    format_opportunity_loss_admin = None
    format_opportunity_loss_subadmin = None

# Блок 2_______________Пути и константы______________________________________
THIS = Path(__file__).resolve()
ROOT_DIR = THIS.parent.parent if THIS.parent.name == "bot" else THIS.parent
load_dotenv(ROOT_DIR / ".env", encoding="utf-8-sig", override=True)
TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))
REPORTS_DIR = ROOT_DIR / "reports"
HTML_DIR = REPORTS_DIR / "html"
JSON_DIR = REPORTS_DIR / "json"
AI_DIR = REPORTS_DIR / "ai"
ANALYTICS_DIR = REPORTS_DIR / "analytics"  # 🆕 v9.4.9
CONFIG_DIR = ROOT_DIR / "config"
LOGS_DIR = ROOT_DIR / "logs"
ARCHIVE_DIR = ROOT_DIR / "archive"
QUEUE_DIR = REPORTS_DIR / "queue"
PROCESSED_DIR = REPORTS_DIR / "excel" / "processed"
CLEAN_DIR = REPORTS_DIR / "excel" / "clean"
REJECTED_DIR = REPORTS_DIR / "rejected"  # 🆕 v9.4.13.3
REJECTED_CASH_DIR = REJECTED_DIR / "cash"  # 🆕 v9.4.13.3
NOTIFY_STATE_PATH = LOGS_DIR / "notify_state.json"
SALES_NOTIFY_DECADE_PATH = LOGS_DIR / "sales_notify_decade.json"  # v9.4.25: подекадные уведомления
for d in [REPORTS_DIR, HTML_DIR, JSON_DIR, AI_DIR, ANALYTICS_DIR, CONFIG_DIR, LOGS_DIR, ARCHIVE_DIR, QUEUE_DIR, PROCESSED_DIR, CLEAN_DIR, REJECTED_DIR, REJECTED_CASH_DIR]:
    d.mkdir(parents=True, exist_ok=True)
BOT_TOKEN = os.getenv("TG_BOT_TOKEN") or os.getenv("BOT_TOKEN") or ""
ADMIN_CHAT_ID = int(os.getenv("ADMIN_CHAT_ID", "0"))
if not ADMIN_CHAT_ID:
    print("⚠️ [STARTUP] ADMIN_CHAT_ID не задан в .env — уведомления администратору НЕ будут доставлены!")
PIPELINE_INTERVAL_MIN = int(os.getenv("PIPELINE_INTERVAL_MIN", "10"))
SCAN_INTERVAL_MIN = int(os.getenv("SCAN_INTERVAL_MIN", "15"))
AI_CACHE_HOURS = int(os.getenv("AI_CACHE_HOURS", "48"))  # v9.4.7: увеличен до 48 часов
AI_PROCESSING_WAIT_SEC = 2
REPORT_SEND_DELAY_SEC = 3
# v9.4.7: Константы для автогенерации ИИ
AI_AUTO_GENERATION = os.getenv("AI_AUTO_GENERATION", "true").lower() == "true"
AI_GENERATION_INTERVAL_SEC = int(os.getenv("AI_GENERATION_INTERVAL_SEC", "120"))  # 2 минуты
AI_GENERATION_STATE_PATH = LOGS_DIR / "ai_generation_state.json"
AI_GENERATION_QUEUE_PATH = LOGS_DIR / "ai_generation_queue.json"
DAILY_ACTIVITY_PATH = LOGS_DIR / "daily_activity.json"
ADMIN_ACTIVITY_LOG = os.getenv("ADMIN_ACTIVITY_LOG", "true").lower() == "true"
ADMIN_SUMMARY_TIME_STR = os.getenv("ADMIN_SUMMARY_TIME", "23:00")

# Парсим время сводки
try:
    hour, minute = map(int, ADMIN_SUMMARY_TIME_STR.split(":"))
    ADMIN_SUMMARY_TIME = dt_time(hour, minute, tzinfo=TZ)
except (ValueError, AttributeError):
    ADMIN_SUMMARY_TIME = dt_time(23, 0, tzinfo=TZ)


# v9.4.5: Константы для автоудаления
AUTO_DELETE_HOURS = 24  # Автоудаление через 24 часа
DELETION_QUEUE_PATH = LOGS_DIR / "deletion_queue.json"
JANITOR_INTERVAL_SEC = 3600  # v9.4.6.1: Проверка каждые 60 минут (было 60 сек)
TELEGRAM_DELETE_LIMIT_HOURS = 48  # Лимит Telegram API

# ✅ ДОБАВЬТЕ ФУНКЦИИ ДЛЯ AI-КОНВЕРТАЦИИ ЗДЕСЬ:
def html_to_path(txt_path: Path) -> Path:
    """Конвертирует путь .txt файла в .html путь"""
    return txt_path.with_suffix('.html')

def txt_to_html(txt_path: Path, html_path: Path):
    """Конвертирует txt в html с правильной кодировкой для мобильных устройств"""
    try:
        content = txt_path.read_text(encoding='utf-8')
        html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Анализ ИИ - {txt_path.stem}</title>
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6; 
            margin: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        pre {{ 
            white-space: pre-wrap; 
            word-wrap: break-word;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            font-size: 14px;
        }}
        @media (max-width: 768px) {{
            body {{ margin: 10px; }}
            pre {{ padding: 15px; font-size: 13px; }}
        }}
    </style>
</head>
<body>
    <pre>{content}</pre>
</body>
</html>"""
        html_path.write_text(html_content, encoding='utf-8')
        # v9.4.7.5: Логирование перенесено в вызывающие функции (с правильным manager)
    except Exception as e:
        raise Exception(f"Ошибка конвертации TXT в HTML: {e}")
# Блок 3_______________Логирование (Asia/Almaty)_____________________________
def _formatTime_almaty(self, record, datefmt=None):
    dt = datetime.fromtimestamp(record.created, ZoneInfo("Asia/Almaty"))
    return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")
logging.Formatter.formatTime = _formatTime_almaty
LOG_FILE = LOGS_DIR / f"send_reports_{datetime.now(TZ).strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s, %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Константа лимита Telegram и async-хелпер для длинных сообщений
# ──────────────────────────────────────────────────────────────
TG_MAX_MSG = 4000  # чуть меньше 4096 — запас на переносы


async def _tg_send_long(context, chat_id: int, text: str,
                        parse_mode=None, delay_hours: int = 24) -> None:
    """
    Отправляет текст в Telegram, разбивая его на части <= TG_MAX_MSG символов.
    Разбивка выполняется по строкам (\\n), чтобы не рвать слова.
    Каждое сообщение ставится в очередь на автоудаление (delay_hours).
    """
    if not text:
        return

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in text.splitlines(keepends=True):
        if current_len + len(line) > TG_MAX_MSG and current:
            chunks.append("".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line)

    if current:
        chunks.append("".join(current))

    for i, chunk in enumerate(chunks, 1):
        if not chunk.strip():
            continue
        try:
            msg = await context.bot.send_message(
                chat_id=chat_id, text=chunk, parse_mode=parse_mode
            )
            schedule_message_deletion(
                chat_id, msg.message_id, msg.date.timestamp(), delay_hours=delay_hours
            )
        except Exception as e:
            logger.error("_tg_send_long: chunk %d/%d chat_id=%s: %s", i, len(chunks), chat_id, e)


# ──────────────────────────────────────────────────────────────
# MENU ANCHOR SYSTEM v1.0 (меню всегда внизу)
# Идея: у каждого chat_id есть одно "меню-сообщение". Перед отправкой отчёта мы удаляем меню,
# после отправки — создаём новое меню (последним сообщением).
# ──────────────────────────────────────────────────────────────
_MENU_ANCHOR: dict[int, int] = {}  # chat_id -> message_id (последнее меню)

def _menu_set(chat_id: int, message_id: int) -> None:
    if chat_id and message_id:
        _MENU_ANCHOR[chat_id] = message_id

def _menu_clear(chat_id: int) -> None:
    if chat_id in _MENU_ANCHOR:
        _MENU_ANCHOR.pop(chat_id, None)

async def hide_main_menu(context: "ContextTypes.DEFAULT_TYPE", chat_id: int) -> None:
    """Удаляет текущее меню (если известно). Ошибки игнорируются."""
    mid = _MENU_ANCHOR.get(chat_id)
    if not mid:
        return
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=mid)
    except Exception:
        pass
    finally:
        _menu_clear(chat_id)

async def send_main_menu(context: "ContextTypes.DEFAULT_TYPE", chat_id: int, user_role: str, text: str = "📋 Выберите раздел:") -> None:
    """Гарантирует, что меню окажется последним сообщением: удаляет старое и отправляет новое."""
    # на всякий случай чистим предыдущее
    await hide_main_menu(context, chat_id)
    try:
        msg = await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=kb_main(user_role, chat_id), parse_mode="Markdown")
        _menu_set(chat_id, msg.message_id)
    except Exception:
        # не падаем: меню не критично
        pass


# ── Возврат в родительский раздел после отправки отчёта ──────────────────
# Маппинг: код раздела → callback_data родительского меню
_SECTION_PARENT: dict[str, str] = {
    "DEBT_SIMPLE":    "menu_debt",
    "DEBT_EXTENDED":  "menu_debt",
    "SALES_SIMPLE":   "menu_sales",
    "SALES_EXTENDED": "menu_sales",
    "GROSS_PCT":      "menu_gross",
    "GROSS_SUM":      "menu_gross",
    "EXPENSES":       "menu_expenses",
    "EXPENSES_PERIOD":"menu_expenses",
    "EXPENSES_DAY":   "menu_expenses",
}

async def send_section_back(
    context: "ContextTypes.DEFAULT_TYPE",
    chat_id: int,
    user_role: str,
    section: str,
    text: str = "✅ *Отчёт отправлен!*\n\n📋 Выберите раздел:",
) -> None:
    """
    После отправки отчёта возвращает пользователя в РОДИТЕЛЬСКИЙ РАЗДЕЛ
    (Дебиторка → меню Дебиторки, Продажи → меню Продаж, и т.д.).
    Если раздел без подменю (Остатки) — показывает главное меню.
    """
    my_name = get_my_manager_name(chat_id)
    parent_cb = _SECTION_PARENT.get(section)

    if parent_cb == "menu_debt":
        if user_role == "manager":
            kb = kb_debt_menu_manager(my_name or "Unknown")
        else:
            kb = kb_debt_menu(user_role)
        back_text = text.replace("📋 Выберите раздел:", "📊 *Дебиторка* — выберите тип:")
    elif parent_cb == "menu_sales":
        if user_role == "manager":
            kb = kb_sales_menu_manager(my_name or "Unknown")
        else:
            kb = kb_sales_menu(user_role)
        back_text = text.replace("📋 Выберите раздел:", "🛒 *Продажи* — выберите тип:")
    elif parent_cb == "menu_gross":
        kb = kb_gross_menu(user_role)
        back_text = text.replace("📋 Выберите раздел:", "💰 *Валовая прибыль* — выберите тип:")
    elif parent_cb == "menu_expenses":
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("💸 Затраты за день", callback_data="expenses|day")],
            [InlineKeyboardButton("🗓️ Затраты за период", callback_data="expenses|period")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="back_main")],
        ])
        back_text = text.replace("📋 Выберите раздел:", "💸 *Затраты* — выберите тип:")
    else:
        # Для Остатков и прочего — возвращаем на главное меню
        await send_main_menu(context, chat_id, user_role, text=text)
        return

    await hide_main_menu(context, chat_id)
    try:
        msg = await context.bot.send_message(
            chat_id=chat_id, text=back_text, reply_markup=kb, parse_mode="Markdown"
        )
        _menu_set(chat_id, msg.message_id)
    except Exception:
        pass


async def send_analytics_menu(
    context: "ContextTypes.DEFAULT_TYPE",
    chat_id: int,
    user_role: str,
    text: str = "✅ *Отчёт отправлен!*\n\n📈 *АНАЛИТИКА* — выберите отчёт:",
) -> None:
    """После отправки аналитического отчёта возвращает в меню аналитики."""
    # Клавиатура строится в _build_analytics_kb, см. cmd_analytics
    kb = _build_analytics_kb(user_role, chat_id)
    if not kb:
        await send_main_menu(context, chat_id, user_role)
        return
    await hide_main_menu(context, chat_id)
    try:
        msg = await context.bot.send_message(
            chat_id=chat_id, text=text, reply_markup=kb, parse_mode="Markdown"
        )
        _menu_set(chat_id, msg.message_id)
    except Exception:
        pass

# ── v9.4.21: Фильтр чувствительных данных в логах ──────────────────────────
class SensitiveDataFilter(logging.Filter):
    """
    Маскирует chat_id пользователей, Telegram Bot Token и email в лог-записях.
    Применяется ко всем handlers — и к файлу, и к stdout.
    Чувствительные значения берём из конфига и .env в момент применения фильтра.
    """
    _MASK_RULES: list = []  # список (pattern, replacement) — заполняется в install()

    @classmethod
    def install(cls, bot_token: str, admin_id: int, managers_map: dict,
                subadmin_id: int = 0, imap_user: str = "") -> None:
        """Вызвать ПОСЛЕ загрузки конфига. Регистрирует все маски."""
        import re
        rules = []
        # Telegram Bot Token
        if bot_token:
            rules.append((re.escape(bot_token), "BOT_TOKEN"))
            # На случай если токен в URL (getUpdates, sendMessage и т.д.)
            rules.append((re.escape(bot_token.split(":")[0]) + r":[A-Za-z0-9_\-]{35}", "BOT_TOKEN"))
        # ADMIN chat_id
        if admin_id:
            rules.append((rf"\b{admin_id}\b", "ADMIN_ID"))
        # Subadmin chat_id
        if subadmin_id:
            rules.append((rf"\b{subadmin_id}\b", "SUBADMIN_ID"))
        # Менеджеры
        for name, cid in (managers_map or {}).items():
            if cid:
                safe_name = name.upper().replace(" ", "_")
                rules.append((rf"\b{cid}\b", f"MGR_{safe_name}_ID"))
        # IMAP email
        if imap_user:
            rules.append((re.escape(imap_user), "IMAP_EMAIL"))
        # Generic email fallback
        rules.append((r"[\w.+\-]+@[\w.\-]+\.[a-z]{2,6}", "EMAIL_HIDDEN"))
        cls._MASK_RULES = [(re.compile(pat), repl) for pat, repl in rules]
        logger.info(f"🔒 SensitiveDataFilter: зарегистрировано {len(rules)} масок")

    def filter(self, record: logging.LogRecord) -> bool:
        if self._MASK_RULES:
            record.msg = self._apply(str(record.msg))
            record.args = None  # аргументы уже не нужны
        return True

    @classmethod
    def _apply(cls, text: str) -> str:
        for pattern, replacement in cls._MASK_RULES:
            text = pattern.sub(replacement, text)
        return text

_sensitive_filter = SensitiveDataFilter()
for _h in logging.root.handlers:
    _h.addFilter(_sensitive_filter)
# ────────────────────────────────────────────────────────────────────────────
EMOJI_LOG_MAP = {
    "bot_starting": "🤖", "bot_polling_started": "📡", "bot_shutdown_requested": "⏹️",
    "bot_critical_error": "💥", "gross_pct_skip_no_clean": "⚠️", "initial_index_built": "🗂️",
    "index_built": "🗂️", "index_rebuilt_after_generation": "🔄", "index_parse_error": "❌",
    "report_not_found": "⚠️", "report_type_not_indexed": "⚠️", "send_file": "📤",
    "tg_send_error": "❌", "tg_retry_after": "⏳", "tg_file_too_big": "⚠️",
    "callback_query": "📘", "config_load_error": "❌", "read_full_error": "⚠️",
    "read_full_too_large": "⚠️", "pipeline_cycle_start": "🔍", "pipeline_cycle_finish": "✅",
    "queue_empty": "🔭", "queue_found_files": "🔬", "file_processed": "📦",
    "file_processing_error": "❌", "archive_cleanup": "🧹", "archive_error": "❌",
    "skip_non_debt_cash": "⭕", "notifier_start": "🔔", "notifier_finish": "🔔",
    "notifier_file_error": "⚠️", "notifier_flood_protection": "⚠️",
    "notifier_skip_invalid_chat_id": "⚠️", "notification_sent": "📨",
    "notification_error": "❌", "save_state_error": "❌", "run_script_start": "▶️",
    "run_script_finish": "✅", "script_stdout": "📄", "script_stderr": "⚠️",
    "script_exec_error": "❌", "script_not_found": "❌", "classify_type_fallback_error": "⚠️",
    "ai_generate_start": "🤖", "ai_generate_error": "❌", "ai_file_sent": "📄",
    "ai_file_send_error": "❌", "ai_file_reused": "♻️", "ai_txt_found": "🔍",
    "ai_html_created": "🎨", "ai_html_creation_error": "❌", "ai_output_parse_error": "⚠️",
    "ai_file_missing": "⚠️", "ai_file_not_found": "⚠️", "ai_file_disappeared": "⚠️",
    "ai_cache_hit": "♻️", "ai_cache_miss": "🔍", "ai_send_invalid_chat_id": "⚠️",
    "txt_to_html_import_error": "❌", "json_read_success": "✅", "json_encoding_failed": "❌",
    "gross_processing_start": "💰", "gross_sum_success": "✅", "gross_sum_error": "❌",
    "gross_pct_success": "✅", "gross_pct_error": "❌", "gross_processing_complete": "💰",
    "file_disappeared_between_gross": "⚠️", "file_missing_for_move": "⚠️",
    "deletion_scheduled": "🗑️", "deletion_executed": "✅", "deletion_failed": "❌",
    "deletion_too_old": "⏰", "janitor_start": "🧹", "janitor_finish": "✅",
    # v9.4.7: Новые события
    "ai_auto_scheduled": "🤖", "ai_auto_skipped": "⏭️", "ai_queue_added": "➕",
    "ai_queue_processing": "⚙️", "ai_queue_completed": "✅", "ai_queue_error": "❌",
    "user_activity_logged": "📝", "daily_summary_sent": "📊", "ai_state_reset": "🔄",
    # v9.4.7.5: Новые события для batch-логирования и проверок
    "cash_file_rejected": "🗑️💰",  # Отдельный cash-файл перенесён в rejected/cash
    "cash_files_moved_to_rejected": "🗑️📦",  # Batch-перенос cash-отчётов в rejected/cash (v9.4.13.3)
    "ai_auto_skipped_old_file": "🤖⏰❌",  # файл старше 24 часов
    "ai_auto_skipped_same_date": "🤖📅❌",  # файл с той же датой уже обработан
    # v9.4.7.5: Автоочистка старых файлов
    "cleanup_start": "🧹",
    "cleanup_finish": "✅",
    "cleanup_error": "❌",
}
def log_event(event: str, emoji: str | None = None, **kw):
    if emoji is None:
        emoji = EMOJI_LOG_MAP.get(event)
    if emoji:
        try:
            flat = "; ".join(f"{k}={v}" for k, v in kw.items())
            logger.info(f"{emoji} {event}" + (f" · {flat}" if flat else ""))
        except Exception:
            pass
    try:
        logger.info(json.dumps({"event": event, **kw}, ensure_ascii=False))
    except Exception:
        logger.info("%s %s", event, kw)

# Блок 4_______________Роли и доступ_________________________________________
def _load_json_safe(p: Path) -> dict:
    try:
        if not p.exists():
            return {}
        for encoding in ("utf-8", "utf-8-sig", "cp1251", "windows-1251"):
            try:
                content = p.read_text(encoding=encoding)
                data = json.loads(content)
                data_str = json.dumps(data, ensure_ascii=False)
                if "Ð" in data_str and ("Ð'" in data_str or "Ð•" in data_str or "Ð›" in data_str):
                    continue
                log_event("json_read_success", file=p.name, encoding=encoding)
                return data
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        log_event("json_encoding_failed", file=p.name)
        return {}
    except Exception as e:
        log_event("config_load_error", path=str(p), error=str(e))
        return {}

# v9.4.6: Атомарная запись JSON через tempfile (защита от гонок и потери данных)
def _save_json_atomic(path: Path, payload: dict) -> None:
    """Атомарно сохраняет JSON через временный файл"""
    tmp = None
    try:
        with NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=path.parent) as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            tmp = f.name
        # Атомарная подмена - защита от гонок
        os.replace(tmp, path)
    except Exception as e:
        log_event("atomic_save_error", path=str(path), error=str(e), level="ERROR")
        raise
    finally:
        # Cleanup временного файла если что-то пошло не так
        try:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

ROLES = _load_json_safe(CONFIG_DIR / "roles.json")
MANAGERS_MAP = _load_json_safe(CONFIG_DIR / "managers.json")

# v9.4.6: Нормализация chat_id к int (критично для сравнений!)
if isinstance(MANAGERS_MAP, dict):
    fixed_managers = {}
    for manager_name, chat_id_val in MANAGERS_MAP.items():
        try:
            fixed_managers[manager_name] = int(chat_id_val)
        except (ValueError, TypeError):
            # Пропускаем некорректные id, чтобы не ломать логику
            log_event("manager_invalid_chat_id", manager=manager_name, value=str(chat_id_val), level="WARNING")
            continue
    MANAGERS_MAP = fixed_managers
    if fixed_managers:
        log_event("managers_normalized", count=len(fixed_managers))

# v9.4.21: активируем маскировку после загрузки всего конфига
def _install_sensitive_filter() -> None:
    """Извлекает subadmin chat_id из roles.json и активирует log-фильтр."""
    subadmin_cid = 0
    try:
        subadmin_scopes = ROLES.get("subadmin_scopes", {})
        for str_cid in subadmin_scopes:
            subadmin_cid = int(str_cid)
            break  # берём первый — в проекте один субадмин
    except Exception:
        pass
    imap_cfg = _load_json_safe(CONFIG_DIR / "imap.json") or {}
    imap_user = imap_cfg.get("user", "") or imap_cfg.get("login", "") or imap_cfg.get("username", "")
    SensitiveDataFilter.install(
        bot_token    = BOT_TOKEN,
        admin_id     = ADMIN_CHAT_ID,
        managers_map = MANAGERS_MAP or {},
        subadmin_id  = subadmin_cid,
        imap_user    = imap_user,
    )
_install_sensitive_filter()

def _admins_set() -> set[int]:
    out: set[int] = set()
    adm = ROLES.get("admin")
    if isinstance(adm, int):
        out.add(adm)
    admins = ROLES.get("admins") or []
    for x in admins:
        try: out.add(int(x))
        except: pass
    env_admin = os.getenv("ADMIN_CHAT_ID")
    if env_admin:
        try: out.add(int(env_admin))
        except: pass
    return out
ADMINS = _admins_set()
def is_admin(chat_id: int) -> bool:
    return chat_id in ADMINS
def get_user_role(chat_id: int) -> str:
    if is_admin(chat_id):
        return "admin"
    subadmin_scopes = ROLES.get("subadmin_scopes", {})
    if str(chat_id) in subadmin_scopes:
        return "subadmin"
    for _, m_chat_id in (MANAGERS_MAP or {}).items():
        if m_chat_id == chat_id:
            return "manager"
    return "unknown"

# v9.4.6.1: Упрощено - удалены избыточные проверки на "Арман" (его нет в конфиге)
def get_managers_list() -> List[str]:
    """Возвращает список активных менеджеров (без Минай)"""
    if MANAGERS_MAP and isinstance(MANAGERS_MAP, dict):
        # Исключаем только Минай (системный аккаунт)
        return sorted([k for k in MANAGERS_MAP.keys() if k != "Минай"])
    return sorted(["Алена", "Оксана", "Магира", "Ергали"])

def get_my_manager_name(chat_id: int) -> Optional[str]:
    for manager, m_chat_id in (MANAGERS_MAP or {}).items():
        if m_chat_id == chat_id and manager != "Минай":
            return manager
    return None

def get_subordinates_for_subadmin(manager_name: str) -> List[str]:
    """Получить список подшефных менеджеров для субадмина"""
    if not manager_name:
        return []
    chat_id = MANAGERS_MAP.get(manager_name)
    if not chat_id:
        return []
    subadmin_scopes = ROLES.get("subadmin_scopes", {})
    subordinates = subadmin_scopes.get(str(chat_id), [])
    if isinstance(subordinates, list):
        return [s for s in subordinates if s != "Минай"]
    return []

def user_scopes(chat_id: int) -> List[str]:
    role = get_user_role(chat_id)
    if role == "admin":
        return get_managers_list()
    scopes = []
    if role == "subadmin":
        subadmin_scopes = ROLES.get("subadmin_scopes", {}).get(str(chat_id), [])
        scopes.extend([s for s in subadmin_scopes if s != "Минай"])
    my_name = get_my_manager_name(chat_id)
    if my_name:
        scopes.append(my_name)
    return sorted([s for s in list(set(scopes)) if s != "Минай"])

# Блок 4.1_____________Система автоудаления сообщений (v9.4.5)________________
def _load_deletion_queue() -> Dict[str, Any]:
    """v9.4.6.1: Загружает очередь удаления с гарантией структуры"""
    try:
        if DELETION_QUEUE_PATH.exists():
            data = _load_json_safe(DELETION_QUEUE_PATH)
            if not isinstance(data, dict):
                return {"jobs": []}
            data.setdefault("jobs", [])
            return data
        return {"jobs": []}
    except Exception as e:
        log_event("deletion_queue_load_error", error=str(e))
        return {"jobs": []}

def _save_deletion_queue(queue_data: Dict[str, Any]):
    """Сохраняет очередь удаления в JSON (атомарно через tempfile)"""
    try:
        _save_json_atomic(DELETION_QUEUE_PATH, queue_data)
    except Exception as e:
        log_event("deletion_queue_save_error", error=str(e))

def schedule_message_deletion(chat_id: int, message_id: int, msg_ts: float, 
                              delay_hours: int = AUTO_DELETE_HOURS):
    """
    Планирует удаление сообщения через указанное время
    
    v9.4.6: Улучшения:
    - msg_ts: реальное время сообщения (для точного расчёта 48ч лимита)
    - дедупликация: (chat_id, message_id) обновляется вместо дублирования
    - ограничение: максимум 5000 задач в очереди
    """
    try:
        due_timestamp = msg_ts + (delay_hours * 3600)
        queue_data = _load_deletion_queue()
        jobs = queue_data.get("jobs", [])
        
        # v9.4.6: Дедупликация - если задача уже есть, обновляем её
        replaced = False
        for j in jobs:
            if j.get("chat_id") == chat_id and j.get("message_id") == message_id:
                j["due_ts"] = due_timestamp
                j["msg_ts"] = msg_ts
                j["scheduled_at"] = time.time()
                replaced = True
                log_event("deletion_updated", chat_id=chat_id, message_id=message_id)
                break
        
        # Если не нашли дубль - добавляем новую задачу
        if not replaced:
            jobs.append({
                "chat_id": chat_id,
                "message_id": message_id,
                "due_ts": due_timestamp,
                "msg_ts": msg_ts,
                "scheduled_at": time.time()
            })
        
        # v9.4.6: Ограничение размера очереди - защита от утечки памяти
        queue_data["jobs"] = jobs[-5000:]
        if len(jobs) > 5000:
            log_event("deletion_queue_trimmed", old_size=len(jobs), new_size=5000, level="WARNING")
        
        _save_deletion_queue(queue_data)
        
        due_dt = datetime.fromtimestamp(due_timestamp, tz=TZ).strftime("%d.%m %H:%M")
        log_event("deletion_scheduled", 
                 chat_id=chat_id, 
                 message_id=message_id, 
                 due_time=due_dt,
                 delay_hours=delay_hours)
    except Exception as e:
        log_event("deletion_schedule_error", error=str(e), chat_id=chat_id, message_id=message_id)

async def janitor_task(context: ContextTypes.DEFAULT_TYPE):
    """Фоновая задача для удаления просроченных сообщений"""
    log_event("janitor_start")
    try:
        queue_data = _load_deletion_queue()
        jobs = queue_data.get("jobs", [])
        
        if not jobs:
            log_event("janitor_finish", processed=0)
            return
        
        now_ts = time.time()
        remaining_jobs = []
        deleted_count = 0
        failed_count = 0
        too_old_count = 0
        
        for job in jobs:
            try:
                chat_id = job.get("chat_id")
                message_id = job.get("message_id")
                due_ts = job.get("due_ts")
                # v9.4.6: Используем msg_ts (реальное время сообщения) для точной проверки лимита
                msg_ts = job.get("msg_ts", job.get("scheduled_at", now_ts))
                
                # Проверяем, не слишком ли старое сообщение (48 часов - лимит Telegram)
                # Теперь считаем от РЕАЛЬНОГО времени сообщения, а не от scheduled_at
                message_age_hours = (now_ts - msg_ts) / 3600
                if message_age_hours > TELEGRAM_DELETE_LIMIT_HOURS:
                    log_event("deletion_too_old", 
                             chat_id=chat_id, 
                             message_id=message_id,
                             age_hours=round(message_age_hours, 1))
                    too_old_count += 1
                    continue  # Не добавляем в remaining_jobs
                
                # Проверяем, пора ли удалять
                if now_ts >= due_ts:
                    try:
                        await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
                        log_event("deletion_executed", chat_id=chat_id, message_id=message_id)
                        deleted_count += 1
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "message to delete not found" in error_msg or "message can't be deleted" in error_msg:
                            # Сообщение уже удалено или недоступно - это нормально
                            log_event("deletion_already_gone", chat_id=chat_id, message_id=message_id)
                            deleted_count += 1
                        else:
                            log_event("deletion_failed", chat_id=chat_id, message_id=message_id, error=str(e))
                            failed_count += 1
                            # Оставляем в очереди для повторной попытки
                            remaining_jobs.append(job)
                else:
                    # Ещё не пора удалять
                    remaining_jobs.append(job)
            
            except Exception as e:
                log_event("janitor_job_error", error=str(e))
                # Оставляем проблемную задачу в очереди
                remaining_jobs.append(job)
        
        # Сохраняем обновлённую очередь
        queue_data["jobs"] = remaining_jobs
        _save_deletion_queue(queue_data)
        
        log_event("janitor_finish", 
                 total=len(jobs),
                 deleted=deleted_count, 
                 failed=failed_count,
                 too_old=too_old_count,
                 remaining=len(remaining_jobs))
    
    except Exception as e:
        log_event("janitor_error", error=str(e))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v9.4.7.5: АВТООЧИСТКА СТАРЫХ ФАЙЛОВ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def cleanup_old_files(context: ContextTypes.DEFAULT_TYPE):
    """
    v9.4.7.5: Автоматическая очистка старых файлов (запуск в 03:00)
    
    Удаляет:
    - Логи старше 2 дней
    - AI-отчёты старше 7 дней
    - HTML-отчёты старше 30 дней
    - JSON-отчёты старше 7 дней
    - Processed Excel старше 14 дней
    """
    log_event("cleanup_start")
    
    try:
        now_ts = time.time()
        today_start = datetime.now(TZ).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        
        stats = {
            "logs": {"days": 2, "deleted": 0, "freed_bytes": 0},
            "ai": {"days": 7, "deleted": 0, "freed_bytes": 0},
            "html": {"days": 30, "deleted": 0, "freed_bytes": 0},
            "json": {"days": 7, "deleted": 0, "freed_bytes": 0},
            "processed": {"days": 14, "deleted": 0, "freed_bytes": 0},
        }
        
        # 1. Очистка логов (старше 2 дней)
        if LOGS_DIR.exists():
            cutoff_ts = now_ts - (2 * 24 * 3600)
            for log_file in LOGS_DIR.glob("*.log"):
                try:
                    # Защита: не удаляем сегодняшние файлы
                    if log_file.stat().st_mtime >= today_start:
                        continue
                    if log_file.stat().st_mtime < cutoff_ts:
                        size = log_file.stat().st_size
                        log_file.unlink()
                        stats["logs"]["deleted"] += 1
                        stats["logs"]["freed_bytes"] += size
                except Exception:
                    pass
        
        # 2. Очистка AI-отчётов (старше 7 дней)
        if AI_DIR.exists():
            cutoff_ts = now_ts - (7 * 24 * 3600)
            for ai_file in AI_DIR.glob("ai_*"):
                try:
                    if ai_file.stat().st_mtime >= today_start:
                        continue
                    if ai_file.stat().st_mtime < cutoff_ts:
                        size = ai_file.stat().st_size
                        ai_file.unlink()
                        stats["ai"]["deleted"] += 1
                        stats["ai"]["freed_bytes"] += size
                except Exception:
                    pass
        
        # 3. Очистка HTML-отчётов (старше 30 дней)
        if HTML_DIR.exists():
            cutoff_ts = now_ts - (30 * 24 * 3600)
            for html_file in HTML_DIR.glob("*.html"):
                try:
                    if html_file.stat().st_mtime >= today_start:
                        continue
                    if html_file.stat().st_mtime < cutoff_ts:
                        size = html_file.stat().st_size
                        html_file.unlink()
                        stats["html"]["deleted"] += 1
                        stats["html"]["freed_bytes"] += size
                except Exception:
                    pass
        
        # 4. Очистка JSON-отчётов (старше 7 дней)
        if JSON_DIR.exists():
            cutoff_ts = now_ts - (7 * 24 * 3600)
            for json_file in JSON_DIR.glob("*.json"):
                try:
                    if json_file.stat().st_mtime >= today_start:
                        continue
                    if json_file.stat().st_mtime < cutoff_ts:
                        size = json_file.stat().st_size
                        json_file.unlink()
                        stats["json"]["deleted"] += 1
                        stats["json"]["freed_bytes"] += size
                except Exception:
                    pass
        
        # 5. Очистка processed Excel (старше 14 дней)
        if PROCESSED_DIR.exists():
            cutoff_ts = now_ts - (14 * 24 * 3600)
            for excel_file in PROCESSED_DIR.glob("*.xls*"):
                try:
                    if excel_file.stat().st_mtime >= today_start:
                        continue
                    if excel_file.stat().st_mtime < cutoff_ts:
                        size = excel_file.stat().st_size
                        excel_file.unlink()
                        stats["processed"]["deleted"] += 1
                        stats["processed"]["freed_bytes"] += size
                except Exception:
                    pass
        
        # 6. Очистка аналитики (старше 30 дней) — включая поддиректории net_profit_day/ net_profit_mtd/
        if ANALYTICS_DIR.exists():
            cutoff_ts = now_ts - (30 * 24 * 3600)
            analytics_deleted = 0
            analytics_freed = 0
            for analytics_file in ANALYTICS_DIR.rglob("*.html"):  # rglob: рекурсивно по всем поддиректориям
                try:
                    if analytics_file.stat().st_mtime >= today_start:
                        continue
                    if analytics_file.stat().st_mtime < cutoff_ts:
                        size = analytics_file.stat().st_size
                        analytics_file.unlink()
                        analytics_deleted += 1
                        analytics_freed += size
                except Exception:
                    pass
            if analytics_deleted:
                log_event("cleanup_analytics", deleted=analytics_deleted,
                          freed_mb=round(analytics_freed / (1024 * 1024), 2))
        
        # Формируем красивое сообщение
        total_deleted = sum(s["deleted"] for s in stats.values())
        total_freed_mb = sum(s["freed_bytes"] for s in stats.values()) / (1024 * 1024)
        
        message_parts = ["🧹 Уборка завершена:"]
        
        if stats["logs"]["deleted"] > 0:
            mb = stats["logs"]["freed_bytes"] / (1024 * 1024)
            message_parts.append(f"  📁 Логи (>2д): удалено {stats['logs']['deleted']} файлов, освобождено {mb:.1f} МБ")
        else:
            message_parts.append(f"  📁 Логи (>2д): нет файлов для удаления")
        
        if stats["ai"]["deleted"] > 0:
            mb = stats["ai"]["freed_bytes"] / (1024 * 1024)
            message_parts.append(f"  🤖 AI-отчёты (>7д): удалено {stats['ai']['deleted']} файлов, освобождено {mb:.1f} МБ")
        else:
            message_parts.append(f"  🤖 AI-отчёты (>7д): нет файлов для удаления")
        
        if stats["html"]["deleted"] > 0:
            mb = stats["html"]["freed_bytes"] / (1024 * 1024)
            message_parts.append(f"  📊 HTML-отчёты (>30д): удалено {stats['html']['deleted']} файлов, освобождено {mb:.1f} МБ")
        else:
            message_parts.append(f"  📊 HTML-отчёты (>30д): нет файлов для удаления")
        
        if stats["json"]["deleted"] > 0:
            mb = stats["json"]["freed_bytes"] / (1024 * 1024)
            message_parts.append(f"  📋 JSON-отчёты (>7д): удалено {stats['json']['deleted']} файлов, освобождено {mb:.1f} МБ")
        else:
            message_parts.append(f"  📋 JSON-отчёты (>7д): нет файлов для удаления")
        
        if stats["processed"]["deleted"] > 0:
            mb = stats["processed"]["freed_bytes"] / (1024 * 1024)
            message_parts.append(f"  📦 Processed Excel (>14д): удалено {stats['processed']['deleted']} файлов, освобождено {mb:.1f} МБ")
        else:
            message_parts.append(f"  📦 Processed Excel (>14д): нет файлов для удаления")
        
        message_parts.append("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        message_parts.append(f"  ✅ ИТОГО: удалено {total_deleted} файлов, освобождено {total_freed_mb:.1f} МБ")
        
        logger.info("\n".join(message_parts))
        
        log_event("cleanup_finish",
                 total_files=total_deleted,
                 total_mb=round(total_freed_mb, 1),
                 logs=stats["logs"]["deleted"],
                 ai=stats["ai"]["deleted"],
                 html=stats["html"]["deleted"],
                 json=stats["json"]["deleted"],
                 processed=stats["processed"]["deleted"])
    
    except Exception as e:
        log_event("cleanup_error", error=str(e))
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v9.4.7: АВТОГЕНЕРАЦИЯ ИИ-АНАЛИЗА
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_ai_generation_state() -> Dict[str, Any]:
    """Загружает состояние генераций из JSON"""
    try:
        if AI_GENERATION_STATE_PATH.exists():
            return _load_json_safe(AI_GENERATION_STATE_PATH)
        return {"last_generation_date": "", "generated_today": {}}
    except Exception as e:
        log_event("ai_state_load_error", error=str(e))
        return {"last_generation_date": "", "generated_today": {}}

def _save_ai_generation_state(state_data: Dict[str, Any]):
    """Сохраняет состояние генераций (атомарно)"""
    try:
        _save_json_atomic(AI_GENERATION_STATE_PATH, state_data)
    except Exception as e:
        log_event("ai_state_save_error", error=str(e))

def _load_ai_generation_queue() -> Dict[str, Any]:
    """Загружает очередь генераций из JSON"""
    try:
        if AI_GENERATION_QUEUE_PATH.exists():
            return _load_json_safe(AI_GENERATION_QUEUE_PATH)
        return {"queue": [], "processing": False}
    except Exception as e:
        log_event("ai_queue_load_error", error=str(e))
        return {"queue": [], "processing": False}

def _save_ai_generation_queue(queue_data: Dict[str, Any]):
    """Сохраняет очередь генераций (атомарно)"""
    try:
        _save_json_atomic(AI_GENERATION_QUEUE_PATH, queue_data)
    except Exception as e:
        log_event("ai_queue_save_error", error=str(e))

def schedule_ai_generation(manager: str):
    """
    v9.4.7.5: Добавляет менеджера в очередь автогенерации ИИ
    
    Изменения v9.4.7.5:
    - Проверка возраста файла (не старше 24 часов)
    - Сохранение даты файла в очередь для идентификации
    - НЕ проверяет last_processed_dates здесь (это делает process_ai_generation_queue)
    """
    if not AI_AUTO_GENERATION:
        return
    
    try:
        state = _load_ai_generation_state()
        today = datetime.now(TZ).strftime("%Y-%m-%d")
        
        # Сброс счётчика в новый день (НЕ сбрасываем last_processed_dates!)
        if state.get("last_generation_date") != today:
            state["last_generation_date"] = today
            state["generated_today"] = {}
            # НЕ трогаем last_processed_dates!
            _save_ai_generation_state(state)
        
        # Проверка 1: Уже генерировали сегодня?
        if state.get("generated_today", {}).get(manager):
            log_event("ai_auto_skipped", manager=manager, reason="already_generated_today")
            return
        
        # Проверка 2: Есть ли свежий JSON (не старше 24 часов)?
        json_file = find_recent_json_for_manager(manager, hours=24)  # ← БЫЛО 48!
        if not json_file:
            log_event("ai_auto_skipped_old_file", manager=manager, reason="no_recent_file_24h")
            return
        
        # Проверка 3: Этот файл уже в очереди?
        queue_data = _load_ai_generation_queue()
        queue = queue_data.get("queue", [])
        
        if any(item.get("manager") == manager for item in queue):
            log_event("ai_auto_skipped", manager=manager, reason="already_in_queue")
            return
        
        # Добавляем в очередь с датой файла
        file_date = extract_date_from_filename(json_file.name)
        queue.append({
            "manager": manager, 
            "added_at": time.time(),
            "file_date": file_date  # v9.4.7.5: Сохраняем дату файла
        })
        queue_data["queue"] = queue
        _save_ai_generation_queue(queue_data)
        
        log_event("ai_queue_added", manager=manager, queue_size=len(queue), file_date=file_date)
    except Exception as e:
        log_event("ai_schedule_error", manager=manager, error=str(e))

async def process_ai_generation_queue(context: ContextTypes.DEFAULT_TYPE):
    """Фоновый обработчик очереди автогенерации ИИ"""
    try:
        queue_data = _load_ai_generation_queue()
        
        if queue_data.get("processing"):
            log_event("ai_queue_busy")
            return
        
        queue = queue_data.get("queue", [])
        if not queue:
            return
        
        job = queue.pop(0)
        manager = job.get("manager")
        file_date = job.get("file_date")  # v9.4.7.5: Читаем дату файла
        
        if not manager:
            queue_data["queue"] = queue
            _save_ai_generation_queue(queue_data)
            return
        
        # v9.4.7.5: ПРОВЕРКА 4 - Этот file_date уже обрабатывался?
        state = _load_ai_generation_state()
        last_processed_date = state.get("last_processed_dates", {}).get(manager)
        
        if file_date and file_date == last_processed_date:
            log_event("ai_auto_skipped_same_date", 
                     manager=manager, 
                     reason="file_already_processed",
                     file_date=file_date)
            # Не генерируем, но удаляем из очереди
            queue_data["queue"] = queue
            _save_ai_generation_queue(queue_data)
            return
        
        queue_data["processing"] = True
        queue_data["queue"] = queue
        _save_ai_generation_queue(queue_data)
        
        log_event("ai_queue_processing", manager=manager, remaining=len(queue), file_date=file_date)
        
        try:
            await auto_generate_and_send_ai(manager, context)
            
            state = _load_ai_generation_state()
            state.setdefault("generated_today", {})[manager] = True
            
            # v9.4.7.5: Сохраняем дату обработанного файла
            if file_date:
                if "last_processed_dates" not in state:
                    state["last_processed_dates"] = {}
                state["last_processed_dates"][manager] = file_date
            
            _save_ai_generation_state(state)
            
            log_event("ai_queue_completed", manager=manager, file_date=file_date)
        except Exception as e:
            log_event("ai_queue_error", manager=manager, error=str(e))
        finally:
            queue_data = _load_ai_generation_queue()
            queue_data["processing"] = False
            _save_ai_generation_queue(queue_data)
    except Exception as e:
        log_event("ai_queue_process_error", error=str(e))

async def auto_generate_and_send_ai(manager: str, context: ContextTypes.DEFAULT_TYPE):
    """Автоматически генерирует и отправляет ИИ-анализ"""
    try:
        manager_chat_id = MANAGERS_MAP.get(manager)
        if not manager_chat_id:
            log_event("ai_auto_no_chat_id", manager=manager)
            return
        
        json_file = find_recent_json_for_manager(manager, hours=48)
        if not json_file:
            log_event("ai_auto_no_json", manager=manager)
            return
        
        log_event("ai_auto_generate_start", manager=manager, json_file=str(json_file))
        
        rc, stdout, stderr = await run_script_async(
            "ai_analyzer.py",
            "--path", str(json_file),
            "--chat-id", str(manager_chat_id)
        )
        
        if rc != 0:
            log_event("ai_auto_generate_error", manager=manager, rc=rc)
            return
        
        await asyncio.sleep(3)
        
        start_time = time.time() - 10
        ai_file = find_newest_ai_file_for_manager(manager, start_time)
        
        if not ai_file:
            log_event("ai_auto_file_not_found", manager=manager)
            return
        
        # ✅ НОВЫЙ КОД НАЧИНАЕТСЯ ТУТ ↓↓↓
        # Если AI создал .txt файл - конвертируем его в .html
        if ai_file.suffix == ".txt":
            html_path = html_to_path(ai_file)
            
            # Если HTML ещё не существует - создаём
            if not html_path.exists():
                try:
                    txt_to_html(ai_file, html_path)
                    log_event("ai_html_created", file=html_path.name, manager=manager)
                except Exception as e:
                    log_event("ai_html_create_error", manager=manager, error=str(e))
            
            # Используем HTML если он успешно создан
            if html_path.exists():
                ai_file = html_path
        # ✅ НОВЫЙ КОД ЗАКАНЧИВАЕТСЯ ТУТ ↑↑↑
        
        await send_ai_file(ai_file, manager, manager_chat_id, context)
        log_event("ai_auto_sent_to_manager", manager=manager, chat_id=manager_chat_id)
        
        for subadmin_chat_id_str, subordinates in ROLES.get("subadmin_scopes", {}).items():
            if manager in subordinates:
                subadmin_chat_id = int(subadmin_chat_id_str)
                await send_ai_file(ai_file, manager, subadmin_chat_id, context)
                log_event("ai_auto_sent_to_subadmin", manager=manager, subadmin_chat_id=subadmin_chat_id)
        
        if ADMIN_CHAT_ID:
            await send_ai_file(ai_file, manager, ADMIN_CHAT_ID, context)
            
            await context.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text=f"✅ Автоматически сгенерирован ИИ-анализ:\n"
                     f"👤 Менеджер: {manager}\n"
                     f"📧 Chat ID: {manager_chat_id}\n"
                     f"📄 Файл: {ai_file.name}",
                parse_mode=None
            )
            log_event("ai_auto_sent_to_admin", manager=manager)
    except Exception as e:
        log_event("ai_auto_error", manager=manager, error=str(e))

# ═══════════════════════════════════════════════════════════════
# v9.4.8: ЕЖЕНЕДЕЛЬНАЯ AI + КРАТКИЕ СВОДКИ
# ═══════════════════════════════════════════════════════════════

async def weekly_ai_generation(context: ContextTypes.DEFAULT_TYPE):
    """
    v9.4.8: Еженедельная AI генерация (понедельник 10:00)
    Генерирует для всех менеджеров, отправляет:
    - Каждому менеджеру его AI
    - Алене (subadmin) её + подшефных
    - Админу все
    """
    # Проверка: запускаем только в понедельник
    if datetime.now(TZ).weekday() != 0:  # 0 = понедельник
        return
    
    log_event("weekly_ai_start")
    
    results = []
    
    for manager in get_managers_list():  # FIX B1: was managers_to_process (NameError)
        try:
            # Найти последний JSON (<7 дней)
            json_file = None
            for hours in [24, 48, 72, 168]:
                json_file = find_recent_json_for_manager(manager, hours=hours)
                if json_file:
                    break
            
            if not json_file:
                log_event("weekly_ai_no_json", manager=manager)
                continue
            
            log_event("weekly_ai_generating", manager=manager)
            start_time = time.time()
            
            # Запустить ai_analyzer.py
            ai_script = ROOT_DIR / "ai_analyzer.py"
            if not ai_script.exists():
                ai_script = ROOT_DIR / "bot" / "ai_analyzer.py"
            
            rc, stdout, stderr = await run_script_async(
                str(ai_script),
                "--path", str(json_file),
                timeout=180
            )
            
            if rc != 0:
                log_event("weekly_ai_error", manager=manager, rc=rc)
                continue
            
            # Найти созданный AI файл
            match = re.search(r"AI saved:\s*(.+)", stdout)
            ai_file = Path(match.group(1).strip()) if match else None
            
            if not ai_file or not ai_file.exists():
                log_event("weekly_ai_no_file", manager=manager)
                continue
            
            # Конвертировать TXT→HTML
            html_file = ai_file.with_suffix('.html')
            txt_content = ai_file.read_text(encoding='utf-8')
            
            html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Анализ - {manager}</title>
<style>
body {{font-family:system-ui,Arial,sans-serif;padding:20px;max-width:800px;margin:0 auto;line-height:1.6}}
pre {{white-space:pre-wrap;word-wrap:break-word;background:#f5f5f5;padding:15px;border-radius:8px}}
h1 {{color:#2563eb}}
</style>
</head>
<body>
<h1>🤖 AI Анализ дебиторки: {manager}</h1>
<pre>{txt_content}</pre>
</body>
</html>"""
            
            html_file.write_text(html_content, encoding='utf-8')
            
            results.append({
                'manager': manager,
                'file': html_file,
                'elapsed': time.time() - start_time
            })
            
            log_event("weekly_ai_completed", manager=manager)
            await asyncio.sleep(10)
            
        except Exception as e:
            log_event("weekly_ai_error", manager=manager, error=str(e))
    
    if results:
        await send_weekly_ai_to_recipients(results, context)
        log_event("weekly_ai_finish", managers=len(results))
    else:
        log_event("weekly_ai_no_results")


async def send_weekly_ai_to_recipients(results: list, context):
    """Отправляет AI файлы каждому + Алене подшефных + админу все"""
    from telegram import InputFile
    
    admin_chat_id = int(os.getenv("ADMIN_CHAT_ID", "0"))
    
    SUBADMIN_NAME = "Алена"
    SUBADMIN_CHAT_ID = MANAGERS_MAP.get(SUBADMIN_NAME)
    SUBADMIN_SCOPE = ["Магира", "Оксана"]
    
    # 1. Каждому менеджеру его AI
    for r in results:
        manager = r['manager']
        chat_id = MANAGERS_MAP.get(manager)
        
        if not chat_id:
            log_event("weekly_ai_no_chat_id", manager=manager)
            continue
        
        try:
            caption = f"🤖 Еженедельный AI анализ дебиторки"
            
            with open(r['file'], 'rb') as f:
                await context.bot.send_document(
                    chat_id=chat_id,
                    document=InputFile(f, filename=r['file'].name),
                    caption=caption
                )
            
            log_event("weekly_ai_sent_to_manager", manager=manager)
            await asyncio.sleep(1)
            
        except Exception as e:
            log_event("weekly_ai_send_error", manager=manager, error=str(e))
    
    # 2. Алене подшефных (свой она уже получила выше)
    if SUBADMIN_CHAT_ID:
        try:
            msg_lines = [
                "📊 ЕЖЕНЕДЕЛЬНАЯ AI СВОДКА",
                f"Дата: {datetime.now(TZ).strftime('%d.%m.%Y')}",
                "",
                "Ваши отчеты + подшефные:"
            ]
            
            for r in results:
                if r['manager'] == SUBADMIN_NAME or r['manager'] in SUBADMIN_SCOPE:
                    msg_lines.append(f"  ✅ {r['manager']} ({r['elapsed']:.1f} сек)")
            
            await context.bot.send_message(
                chat_id=SUBADMIN_CHAT_ID,
                text="\n".join(msg_lines)
            )
            
            # Файлы подшефных
            for r in results:
                if r['manager'] in SUBADMIN_SCOPE:
                    caption = f"🤖 AI Анализ подшефного: {r['manager']}"
                    
                    with open(r['file'], 'rb') as f:
                        await context.bot.send_document(
                            chat_id=SUBADMIN_CHAT_ID,
                            document=InputFile(f, filename=r['file'].name),
                            caption=caption
                        )
                    
                    await asyncio.sleep(1)
            
            log_event("weekly_ai_sent_to_subadmin", manager=SUBADMIN_NAME)
            
        except Exception as e:
            log_event("weekly_ai_subadmin_error", error=str(e))
    
    # 3. Админу ВСЕ
    if admin_chat_id:
        try:
            msg_lines = [
                "📊 ЕЖЕНЕДЕЛЬНАЯ AI СВОДКА (все менеджеры)",
                f"Дата: {datetime.now(TZ).strftime('%d.%m.%Y')}",
                "",
                f"Сгенерировано {len(results)} анализов:"
            ]
            
            for r in results:
                msg_lines.append(f"  ✅ {r['manager']} ({r['elapsed']:.1f} сек)")
            
            await context.bot.send_message(
                chat_id=admin_chat_id,
                text="\n".join(msg_lines)
            )
            
            for r in results:
                caption = f"🤖 AI Анализ: {r['manager']}"
                
                with open(r['file'], 'rb') as f:
                    await context.bot.send_document(
                        chat_id=admin_chat_id,
                        document=InputFile(f, filename=r['file'].name),
                        caption=caption
                    )
                
                await asyncio.sleep(1)
            
            log_event("weekly_ai_sent_to_admin", count=len(results))
            
        except Exception as e:
            log_event("weekly_ai_send_error", error=str(e))


async def send_inventory_summary(context: ContextTypes.DEFAULT_TYPE):
    """v9.4.10: Краткая сводка остатков ТОЛЬКО АДМИНУ (09:00)"""
    if not InventorySummary:
        logger.error("InventorySummary не импортирован")
        return
    
    log_event("inventory_summary_start")
    
    try:
        summary = InventorySummary()
        latest_html = summary.get_latest_inventory_report(HTML_DIR)
        
        if not latest_html:
            log_event("inventory_summary_no_file")
            return
        
        data = summary.parse_inventory_html(latest_html)
        message = summary.format_summary(data)
        
        # v9.4.10: Только админу (убрана рассылка менеджерам)
        if ADMIN_CHAT_ID:
            try:
                msg = await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=message)
                schedule_message_deletion(ADMIN_CHAT_ID, msg.message_id, msg.date.timestamp(), delay_hours=24)
                log_event("inventory_summary_sent", manager="Admin")
            except Exception as e:
                log_event("inventory_summary_error", manager="Admin", error=str(e))
        
        log_event("inventory_summary_finish")
        
    except Exception as e:
        log_event("inventory_summary_error", error=str(e))


async def send_sales_summary(context: ContextTypes.DEFAULT_TYPE):
    """v9.4.32: Краткая сводка продаж ТОЛЬКО АДМИНУ (21:00) — агрегация ВСЕХ менеджеров из JSON."""
    if not SalesSummary:
        logger.error("SalesSummary не импортирован")
        return

    log_event("sales_summary_start")
    try:
        summary      = SalesSummary()
        known_mgrs   = set(m.lower() for m in get_managers_list())
        message      = summary.build_admin_sales_summary(JSON_DIR, known_managers=known_mgrs)

        if not message:
            log_event("sales_summary_no_file")
            return

        if ADMIN_CHAT_ID:
            try:
                msg = await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=message)
                schedule_message_deletion(ADMIN_CHAT_ID, msg.message_id, msg.date.timestamp(), delay_hours=24)
                log_event("sales_summary_sent", manager="Admin")
            except Exception as e:
                log_event("sales_summary_error", manager="Admin", error=str(e))

        log_event("sales_summary_finish")
    except Exception as e:
        log_event("sales_summary_error", error=str(e))


async def send_sales_pipeline_summary(context, json_path: "Path", manager_name: str = ""):
    """
    v9.4.23: Краткая сводка сразу после обработки файла продаж в pipeline.

    ЧИТАЕТ ИЗ JSON (не из HTML) — JSON всегда создаётся sales_parser.py.

    Admin получает:  рейтинг всех менеджеров за период + итог
    Менеджер (чей файл) получает: только свои данные + топ-3 клиента

    Определяет тип периода автоматически (день / декада / месяц).
    """
    if not SalesSummary:
        logger.warning("SalesSummary не импортирован — пропуск pipeline-сводки")
        return

    log_event("sales_pipeline_summary_start", manager=manager_name)
    try:
        import json as _json
        summary = SalesSummary()
        known_managers = set(m.lower() for m in get_managers_list())

        # ── Читаем JSON свежеобработанного файла ──────────────────────────────
        try:
            with open(json_path, "r", encoding="utf-8") as _f:
                raw = _json.load(_f)
        except Exception as _e:
            logger.warning(f"sales_pipeline_summary: не удалось прочитать {json_path}: {_e}")
            return

        period_str = raw.get("period", "")
        if not period_str:
            logger.warning("sales_pipeline_summary: период не определён в JSON, пропуск")
            return

        # Из JSON строим data-структуру совместимую с format_manager_pipeline
        clients_raw = raw.get("clients", [])
        total_revenue = float(raw.get("total_revenue", 0))
        data = {
            "date":          period_str,
            "total_amount":  total_revenue,
            "clients_count": len(clients_raw),
            "clients":       clients_raw,
            "products":      [],
        }

        # ── Загружаем ВСЕ JSON того же периода для рейтинга (admin) ──────────
        # Фильтруем: только те у кого manager = реальный менеджер из списка
        all_managers = [
            m for m in summary.load_all_managers_json(JSON_DIR, period_str)
            if m["manager"].lower() in known_managers
        ]
        if not all_managers and manager_name:
            # fallback: текущий файл
            all_managers = [{
                "manager":       manager_name,
                "total_revenue": total_revenue,
                "clients":       clients_raw,
            }]

        # ── Admin ──────────────────────────────────────────────────────────────
        if ADMIN_CHAT_ID and all_managers:
            admin_msg = summary.format_admin_pipeline(data, all_managers)
            try:
                msg = await context.bot.send_message(
                    chat_id=ADMIN_CHAT_ID, text=admin_msg, parse_mode=None
                )
                schedule_message_deletion(
                    ADMIN_CHAT_ID, msg.message_id, msg.date.timestamp(), delay_hours=24
                )
                log_event("sales_pipeline_summary_sent", recipient="admin",
                          period=period_str, managers=len(all_managers))
            except Exception as e:
                log_event("sales_pipeline_summary_error", recipient="admin", error=str(e))

        # ── Менеджер (чей файл пришёл) ─────────────────────────────────────────
        if manager_name and MANAGERS_MAP:
            mgr_chat_id = MANAGERS_MAP.get(manager_name)
            if mgr_chat_id:
                mgr_msg = summary.format_manager_pipeline(manager_name, data)
                try:
                    msg = await context.bot.send_message(
                        chat_id=mgr_chat_id, text=mgr_msg, parse_mode=None
                    )
                    schedule_message_deletion(
                        mgr_chat_id, msg.message_id, msg.date.timestamp(), delay_hours=24
                    )
                    log_event("sales_pipeline_summary_sent", recipient=manager_name,
                              period=period_str)
                except Exception as e:
                    log_event("sales_pipeline_summary_error",
                              recipient=manager_name, error=str(e))

        log_event("sales_pipeline_summary_finish", period=period_str,
                  period_type=_sales_detect_period(period_str))

    except Exception as e:
        logger.error(f"❌ send_sales_pipeline_summary: {e}", exc_info=True)
        log_event("sales_pipeline_summary_error", error=str(e))


async def send_gross_summary(context: ContextTypes.DEFAULT_TYPE):
    """Краткая сводка валовой админу (20:00)"""
    if not GrossSummary:
        logger.error("GrossSummary не импортирован")
        return
    
    log_event("gross_summary_start")
    
    try:
        summary = GrossSummary()
        latest_html = summary.get_latest_gross_report(HTML_DIR)
        
        if not latest_html:
            log_event("gross_summary_no_file")
            return
        
        data = summary.parse_gross_html(latest_html)
        message = summary.format_summary(data)
        
        if ADMIN_CHAT_ID:
            try:
                msg = await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=message)
                schedule_message_deletion(ADMIN_CHAT_ID, msg.message_id, msg.date.timestamp(), delay_hours=24)
                log_event("gross_summary_sent", manager="Admin")
            except Exception as e:
                log_event("gross_summary_error", manager="Admin", error=str(e))
        
        log_event("gross_summary_finish")
        
    except Exception as e:
        log_event("gross_summary_error", error=str(e))

async def reset_ai_generation_state(context: ContextTypes.DEFAULT_TYPE):
    """Сбрасывает счётчик генераций (запускается в 00:01)"""
    try:
        today = datetime.now(TZ).strftime("%Y-%m-%d")
        state = {"last_generation_date": today, "generated_today": {}}
        _save_ai_generation_state(state)
        log_event("ai_state_reset", date=today)
    except Exception as e:
        log_event("ai_state_reset_error", error=str(e))
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# v9.4.7: ЛОГИРОВАНИЕ АКТИВНОСТИ ПОЛЬЗОВАТЕЛЕЙ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_daily_activity() -> Dict[str, Any]:
    """Загружает логи активности за сегодня"""
    try:
        if DAILY_ACTIVITY_PATH.exists():
            return _load_json_safe(DAILY_ACTIVITY_PATH)
        return {"date": datetime.now(TZ).strftime("%Y-%m-%d"), "users": {}}
    except Exception as e:
        log_event("activity_load_error", error=str(e))
        return {"date": datetime.now(TZ).strftime("%Y-%m-%d"), "users": {}}

def _save_daily_activity(activity_data: Dict[str, Any]):
    """Сохраняет логи активности (атомарно)"""
    try:
        _save_json_atomic(DAILY_ACTIVITY_PATH, activity_data)
    except Exception as e:
        log_event("activity_save_error", error=str(e))

def log_user_request(chat_id: int, report_type: str, manager: str):
    """Логирует запрос отчёта пользователем"""
    if not ADMIN_ACTIVITY_LOG:
        return
    
    try:
        activity = _load_daily_activity()
        today = datetime.now(TZ).strftime("%Y-%m-%d")
        
        if activity.get("date") != today:
            activity = {"date": today, "users": {}}
        
        user_name = get_my_manager_name(chat_id) or f"User_{chat_id}"
        user_data = activity["users"].setdefault(user_name, {
            "chat_id": chat_id,
            "requests": {},
            "delivered": [],
            "errors": []
        })
        
        user_data["requests"][report_type] = user_data["requests"].get(report_type, 0) + 1
        
        _save_daily_activity(activity)
        log_event("user_activity_logged", user=user_name, action="request", report=report_type)
    except Exception as e:
        log_event("activity_log_error", error=str(e))

def log_user_delivery(chat_id: int, report_type: str, success: bool):
    """Логирует доставку отчёта"""
    if not ADMIN_ACTIVITY_LOG:
        return
    
    try:
        activity = _load_daily_activity()
        user_name = get_my_manager_name(chat_id) or f"User_{chat_id}"
        
        if user_name not in activity["users"]:
            return
        
        user_data = activity["users"][user_name]
        
        if success:
            if report_type not in user_data["delivered"]:
                user_data["delivered"].append(report_type)
        else:
            user_data["errors"].append(f"{report_type}_failed")
        
        _save_daily_activity(activity)
    except Exception as e:
        log_event("activity_log_error", error=str(e))

async def send_daily_summary_to_admin(context: ContextTypes.DEFAULT_TYPE):
    """Отправляет ежедневную сводку админу в 23:00"""
    if not ADMIN_ACTIVITY_LOG or not ADMIN_CHAT_ID:
        return
    
    try:
        activity = _load_daily_activity()
        today = activity.get("date", datetime.now(TZ).strftime("%Y-%m-%d"))
        users = activity.get("users", {})
        
        if not users:
            return
        
        message_parts = [
            "📊 ЕЖЕДНЕВНАЯ СВОДКА АКТИВНОСТИ БОТА",
            f"Дата: {today}",
            ""
        ]
        
        total_requests = 0
        total_delivered = 0
        
        for user_name, user_data in sorted(users.items()):
            requests = user_data.get("requests", {})
            delivered = user_data.get("delivered", [])
            
            if not requests:
                continue
            
            message_parts.append("━" * 50)
            message_parts.append(f"👤 {user_name.upper()}")
            message_parts.append("━" * 50)
            message_parts.append("📥 Запросила отчеты:")
            
            for report, count in sorted(requests.items()):
                report_rus = SECTIONS.get(report, report)
                message_parts.append(f"  • {report_rus} ({count} раз)" if count > 1 else f"  • {report_rus}")
                total_requests += count
            
            if delivered:
                message_parts.append("")
                message_parts.append("📤 Получила отчеты:")
                for report in delivered:
                    report_rus = SECTIONS.get(report, report)
                    message_parts.append(f"  • {report_rus} ✅")
                    total_delivered += 1
            
            message_parts.append("")
        
        message_parts.append("━" * 50)
        message_parts.append("📈 СТАТИСТИКА")
        message_parts.append("━" * 50)
        message_parts.append(f"Всего запросов: {total_requests}")
        message_parts.append(f"Всего отправлено: {total_delivered}")
        message_parts.append(f"Ошибок: 0")
        
        message = "\n".join(message_parts)
        
        await context.bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text=message,
            parse_mode=None
        )
        
        log_event("daily_summary_sent", users_count=len(users), requests=total_requests)
        
        _save_daily_activity({"date": today, "users": {}})
    except Exception as e:
        log_event("daily_summary_error", error=str(e))



# Блок 5_______________Индексация и поиск отчётов____________________________
_index_cache: Dict[str, Any] = {}
_index_ts: float = 0.0
_index_lock = asyncio.Lock()
INDEX_TTL_SEC = 60  # Кеш индекса: 60 секунд
READ_LIMIT_BYTES = 5 * 1024 * 1024

def _manager_from_gross_filename(fname: str) -> str:
    stem = Path(fname).stem
    match = re.match(r"Валовая\s+прибыль\s+(.+?)_gross", stem, re.IGNORECASE)
    if match:
        name = match.group(1).strip()
        if not name or name.startswith('(') or re.match(r'^[\d\s\(\)._-]+$', name):
            return "Сводный отчёт"
        if re.search(r'[а-яёa-z]', name, re.IGNORECASE):
            return normalize_manager_name(name)
    return "Сводный отчёт"

def _manager_from_gross_pct_filename(fname: str) -> str:
    stem = Path(fname).stem
    m = re.match(r"Валовая\s+прибыль\s+(.+?)_gross_pct", stem, re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        if not name or name.startswith('(') or re.match(r'^[\d\s\(\)._-]+$', name):
            return "Сводный отчёт"
        if re.search(r'[а-яёa-z]', name, re.IGNORECASE):
            return normalize_manager_name(name)
    return "Сводный отчёт"

SECTIONS = {
    "DEBT_SIMPLE": "ДЕБИТОРКА",
    "DEBT_EXTENDED": "ДЕБИТОРКА ДЕТАЛЬНО",
    "SALES_SIMPLE": "ПРОДАЖИ",
    "SALES_EXTENDED": "ПРОДАЖИ ТОВАРЫ",
    "INVENTORY_SIMPLE": "ОСТАТКИ",
    "GROSS_SUM": "ВАЛ СУММЫ",
    "GROSS_PCT": "ВАЛ ПРОЦ",
    "AI": "АНАЛИЗ ИИ",
    "EXPENSES": "ЗАТРАТЫ",
}


def normalize_manager_name(name: str) -> str:
    name = name.strip()
    name = name.replace('Ё', 'Е').replace('ё', 'е')
    name = re.sub(r'\s+', ' ', name)
    return name

def _read_full(p: Path) -> str:
    try:
        size = p.stat().st_size
        if size <= READ_LIMIT_BYTES:
            for enc in ("utf-8", "utf-8-sig", "cp1251"):
                try:
                    return p.read_text(encoding=enc)
                except Exception:
                    continue
        else:
            with p.open("rb") as f:
                head = f.read(65536)
            for enc in ("utf-8", "utf-8-sig", "cp1251"):
                try:
                    return head.decode(enc, errors="ignore")
                except Exception:
                    continue
            return ""
    except Exception:
        pass
    for enc in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            return p.read_text(encoding=enc)
        except Exception:
            continue
    log_event("read_full_error", file=p.name)
    return ""

def _extract_manager(full_text: str, file_name: str) -> str:
    # Динамическая регулярка по активным менеджерам из конфигурации
    active = get_managers_list()
    if not active:
        return "Сводный отчёт"
    mgr_pat = r"(" + "|".join(map(re.escape, active)) + r")"

    # 1) По имени файла, строгий якорь на границу/разделители
    m = re.search(mgr_pat + r"(?=$|[_\-\s\.\(])",
                  Path(file_name).stem, re.IGNORECASE)
    if m:
        return normalize_manager_name(m.group(1).title())

    # 2) По шапке HTML/текста: Менеджер/Ответственный
    m2 = re.search(r"(?:Менеджер|Ответственный)\s*[:=]\s*([А-ЯЁа-яё\s]+?)(?:\s*<|$|\n)",
                   full_text, re.IGNORECASE)
    if m2:
        manager = normalize_manager_name(m2.group(1).title())
        if manager in active:
            return manager

    # 3) Ещё раз по имени файла, без якорей
    m3 = re.search(mgr_pat, Path(file_name).stem, re.IGNORECASE)
    if m3:
        return normalize_manager_name(m3.group(1).title())

    # 4) По <title>
    m4 = re.search(r"<title>([^<]+)</title>", full_text, re.IGNORECASE)
    if m4:
        title = m4.group(1)
        m5 = re.search(mgr_pat, title, re.IGNORECASE)
        if m5:
            return normalize_manager_name(m5.group(1).title())

    return "Сводный отчёт"

def _extract_date(full_text: str, file_name: str, p: Path) -> str:
    m = re.search(r"(\d{2}\.\d{2}\.\d{4}).{0,40}?(\d{2}\.\d{2}\.\d{4})",
                  full_text, re.IGNORECASE | re.DOTALL)
    if m:
        return f"{m.group(1)} – {m.group(2)}"
    m2 = re.search(r"\b(\d{2}\.\d{2}\.\d{4})\b", full_text or file_name)
    if m2:
        return m2.group(1)
    m3 = re.search(
        r"период:\s*((?:январ[ья]|феврал[ья]|март[а]?|апрел[ья]|ма[йя]|июн[ья]|июл[ья]|август[а]?|сентябр[ья]|октябр[ья]|ноябр[ья]|декабр[ья])\s+\d{4}\s*г\.?)",
        full_text, re.I
    )
    if m3:
        return m3.group(1).capitalize()
    return datetime.fromtimestamp(p.stat().st_mtime, tz=TZ).strftime("%d.%m.%Y")

def _classify_type(name: str, full_path: Optional[Path] = None) -> str:
    """Классификация типа отчета по названию файла"""
    lname = name.lower()
    # Нормализуем 'ё' → 'е' для устойчивых проверок подстрок
    lname_norm = lname.replace("ё", "е")
    
    # Сначала по точным паттернам в НАЗВАНИИ
    if "ai_" in lname_norm or "_analysis_" in lname_norm:
        return "AI"
    if "_gross_pct.html" in lname_norm:
        return "GROSS_PCT"
    if "_gross_sum.html" in lname_norm:
        return "GROSS_SUM"
    if "_gross.html" in lname_norm and "_sum" not in lname_norm and "_pct" not in lname_norm:
        return "GROSS_SUM"
    if "sales_products_" in lname_norm or "продажи_товары" in lname_norm or "продажи по товару" in lname_norm:
        return "SALES_EXTENDED"
    if "sales_grouped_" in lname_norm or ("продажи" in lname_norm and "товар" not in lname_norm):
        return "SALES_SIMPLE"
    
    # Дебиторка: сначала проверяем ДЕТАЛЬНУЮ по названию
    if "debt_ext_" in lname_norm:
        return "DEBT_EXTENDED"
    
    # Потом ПРОСТУЮ по _debt.html в конце (это файлы Ведомость)
    if "_debt.html" in lname_norm:
        return "DEBT_SIMPLE"
    
    # Потом по словам в названии
    if "детальный" in lname_norm or "расширен" in lname_norm:
        return "DEBT_EXTENDED"
    
    # Ведомость = простая дебиторка (если не попала выше)
    if "ведомость" in lname_norm and "взаиморасчет" in lname_norm:
        return "DEBT_SIMPLE"
    
    # Общий паттерн debt_ (если не попал выше)
    if "debt_" in lname_norm or "дебитор" in lname_norm:
        return "DEBT_SIMPLE"
    
    # Остатки
    if ("inventory_simple_" in lname_norm or "inventory_warehouses_simple_" in lname_norm or 
        "остаток" in lname_norm or "остатки" in lname_norm or               # FIX #11: "остатки" ≠ "остаток"
        "ведомость по товарам" in lname_norm or "ведомость_по_товарам" in lname_norm):  # FIX #11: slug uses underscores
        return "INVENTORY_SIMPLE"
    
    # Затраты (расходы)
    if lname_norm.startswith("expenses_") or "затрат" in lname_norm or "расход" in lname_norm:
        return "EXPENSES"

    return "UNKNOWN"


def _parse_period_to_date(period_str: str) -> datetime:
    """
    v9.4.13: Парсит период в datetime для сортировки.
    
    Примеры:
    "17.02.2026" → datetime(2026, 2, 17)
    "01.02.2026 - 16.02.2026" → datetime(2026, 2, 16) (берём конечную дату)
    "01.02.2026 – 10.02.2026" → datetime(2026, 2, 10)
    "февраль 2026" → datetime(2026, 2, 1)
    
    Returns:
        datetime: Дата для сортировки (или datetime.min если не распознано)
    """
    if not period_str or period_str == "—":
        return datetime.min.replace(tzinfo=TZ)
    
    import re
    
    # Паттерн: DD.MM.YYYY - DD.MM.YYYY или DD.MM.YYYY – DD.MM.YYYY
    range_pattern = r'(\d{1,2})[./](\d{1,2})[./](\d{4})\s*[-–—]\s*(\d{1,2})[./](\d{1,2})[./](\d{4})'
    m = re.search(range_pattern, period_str)
    if m:
        # Берём конечную дату (она актуальнее)
        day, month, year = int(m.group(4)), int(m.group(5)), int(m.group(6))
        try:
            return datetime(year, month, day, tzinfo=TZ)
        except (ValueError, OverflowError):
            pass

    # Паттерн: одна дата DD.MM.YYYY
    single_pattern = r'(\d{1,2})[./](\d{1,2})[./](\d{4})'
    m = re.search(single_pattern, period_str)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(year, month, day, tzinfo=TZ)
        except (ValueError, OverflowError):
            pass
    
    # Паттерн: "февраль 2026", "january 2026"
    month_names_ru = {
        'январ': 1, 'феврал': 2, 'март': 3, 'апрел': 4, 'ма': 5, 'июн': 6,
        'июл': 7, 'август': 8, 'сентябр': 9, 'октябр': 10, 'ноябр': 11, 'декабр': 12
    }
    month_names_en = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    lower = period_str.lower()
    year_m = re.search(r'(\d{4})', lower)
    if year_m:
        year = int(year_m.group(1))
        for mname, mnum in {**month_names_ru, **month_names_en}.items():
            if mname in lower:
                try:
                    return datetime(year, mnum, 1, tzinfo=TZ)
                except (ValueError, OverflowError):
                    pass
    
    # Не удалось распарсить
    return datetime.min.replace(tzinfo=TZ)

async def _build_index(force: bool = False) -> Dict[str, Any]:
    """v9.4.13: Индекс с сортировкой по периоду (primary) + mtime (secondary)"""
    global _index_cache, _index_ts
    async with _index_lock:
        now = time.time()
        if _index_cache and not force and (now - _index_ts) < INDEX_TTL_SEC:
            return _index_cache
        t0 = time.time()
        # v9.4.13: Изменена структура на (period_date, mtime, Path)
        index: Dict[str, Dict[str, List[Tuple[datetime, datetime, Path]]]] = {}
        files = sorted([p for p in HTML_DIR.glob("*.html") if p.is_file()]) + \
                sorted([p for p in AI_DIR.glob("*.html") if p.is_file()])
        for p in files:
            try:
                report_type = _classify_type(p.name, full_path=p)
                if report_type == "UNKNOWN":
                    continue
                full_text = _read_full(p)
                if report_type == "GROSS_SUM":
                    manager = _manager_from_gross_filename(p.name)
                elif report_type == "GROSS_PCT":
                    manager = _manager_from_gross_pct_filename(p.name)
                else:
                    manager = _extract_manager(full_text, p.name)
                
                mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=TZ)
                
                # v9.4.13: Извлекаем период из отчёта
                period_str = _extract_date(full_text, p.name, p)
                period_date = _parse_period_to_date(period_str)  # новая функция
                
                # v9.4.13: (period, mtime, path)
                index.setdefault(report_type, {}).setdefault(manager, []).append((period_date, mtime, p))
            except Exception as e:
                log_event("index_parse_error", file=p.name, error=str(e))
        
        # v9.4.13: Сортировка по period (primary), mtime (secondary)
        for report_type in index:
            for manager in index[report_type]:
                index[report_type][manager].sort(key=lambda item: (item[0], item[1]), reverse=True)
        
        _index_cache = index
        _index_ts = time.time()
        build_ms = round((_index_ts - t0) * 1000, 1)
        log_event("index_built", files=len(files), report_types=len(index), build_ms=build_ms)
        return _index_cache

async def _index_fast() -> Dict[str, Any]:
    async with _index_lock:
        if _index_cache:
            return _index_cache
    return await _build_index()

def find_report(report_type: str, manager: Optional[str] = None) -> Optional[Path]:
    """v9.4.13: С детальным логированием выбора файла"""
    index = _index_cache if _index_cache else {}
    report_group = index.get(report_type)
    if not report_group:
        log_event("report_type_not_indexed", report_type=report_type, manager=manager)
        return None
    
    if manager and manager != "Сводный отчёт":
        manager_norm = normalize_manager_name(manager)
        for mgr_name, reports in report_group.items():
            if normalize_manager_name(mgr_name) == manager_norm:
                if reports:
                    # v9.4.13: структура (period, mtime, path)
                    period, mtime, path = reports[0]
                    # v9.4.13: детальное логирование
                    log_event("report_selected",
                             report_type=report_type,
                             manager=manager_norm,
                             file=path.name,
                             period=period.strftime("%Y-%m-%d") if period != datetime.min.replace(tzinfo=TZ) else "unknown",
                             mtime=mtime.strftime("%Y-%m-%d %H:%M:%S"),
                             level="INFO")
                    return path
                return None
        log_event("manager_not_found_in_index", report_type=report_type, manager=manager_norm)
        return None
    else:
        # Для "Сводный отчёт" или None берём любого первого менеджера
        for mgr_name, reports in report_group.items():
            if reports:
                # v9.4.13: структура (period, mtime, path)
                period, mtime, path = reports[0]
                # v9.4.13: детальное логирование
                log_event("report_selected",
                         report_type=report_type,
                         manager=mgr_name or "general",
                         file=path.name,
                         period=period.strftime("%Y-%m-%d") if period != datetime.min.replace(tzinfo=TZ) else "unknown",
                         mtime=mtime.strftime("%Y-%m-%d %H:%M:%S"),
                         level="INFO")
                return path
        return None
# ═══════════════════════════════════════════════════════════════════
# Gender mapping для эмодзи в меню
# ═══════════════════════════════════════════════════════════════════

GENDER_MAP = {
    "Алена": "👩",
    "Оксана": "👩",
    "Магира": "👩",
    "Ергали": "👨",
}
def gender_emoji(name: str) -> str:
    """v9.4.19: Исправлен баг — теперь возвращает правильный эмодзи из GENDER_MAP"""
    return GENDER_MAP.get(name.strip(), "👤")

# Блок 6.1_______________Функции архива______________________________________
def _list_archive_dates_for_manager(manager: str) -> List[str]:
    """Получить список уникальных дат для архива менеджера (устойчиво к ' – ', ' - ', '—')"""
    index = _index_cache if _index_cache else {}
    dates_set = set()
    seps = (" – ", " - ", "—", "–", "-")
    for report_type, managers_data in index.items():
        if manager in managers_data:
            for period_date, mtime, p in managers_data[manager]:
                try:
                    full_text = _read_full(p)
                    date_str = _extract_date(full_text, p.name, p)
                    # Берем конечную дату для диапазона, учитывая разные разделители
                    for sep in seps:
                        if sep in date_str:
                            parts = [x.strip() for x in date_str.split(sep) if x.strip()]
                            if parts:
                                date_str = parts[-1]
                            break
                    dates_set.add(date_str)
                except Exception:
                    pass
    # Сортируем по дате dd.mm.yyyy (пропускаем иные форматы)
    sortable: List[str] = []
    for d in dates_set:
        try:
            if d and re.match(r"^\d{2}\.\d{2}\.\d{4}$", d):
                sortable.append(d)
        except Exception:
            pass
    return sorted(sortable, key=lambda d: datetime.strptime(d, "%d.%m.%Y"), reverse=True)

def _types_for_date_manager(manager: str, date_str: str) -> List[str]:
    """Получить типы отчетов для менеджера и даты"""
    index = _index_cache if _index_cache else {}
    available_types = []
    for report_type, managers_data in index.items():
        if manager in managers_data:
            for period_date, mtime, p in managers_data[manager]:
                try:
                    full_text = _read_full(p)
                    file_date = _extract_date(full_text, p.name, p)
                    # Проверяем вхождение даты
                    if date_str in file_date or file_date.endswith(date_str):
                        available_types.append(report_type)
                        break
                except Exception:
                    pass
    return sorted(list(set(available_types)))

def _find_report_by_date(report_type: str, manager: str, date_str: str) -> Optional[Path]:
    """Найти отчет по типу, менеджеру и дате"""
    index = _index_cache if _index_cache else {}
    managers_data = index.get(report_type, {})
    if manager not in managers_data:
        return None
    for period_date, mtime, p in managers_data[manager]:
        try:
            full_text = _read_full(p)
            file_date = _extract_date(full_text, p.name, p)
            if date_str in file_date or file_date.endswith(date_str):
                return p
        except Exception:
            continue
    return None

def kb_main(user_role: str, chat_id: int = 0) -> InlineKeyboardMarkup:
    """Главное меню - все роли получают остатки напрямую + архив"""
    my_name = get_my_manager_name(chat_id) if user_role == "manager" else None
    
    if user_role == "admin":
        rows = [
            [InlineKeyboardButton("📊 Дебиторка", callback_data="menu_debt")],
            [InlineKeyboardButton("📦 Остатки", callback_data="direct|INVENTORY_SIMPLE|general")],
            [InlineKeyboardButton("🛒 Продажи", callback_data="menu_sales")],
            [InlineKeyboardButton("💰 Валовая", callback_data="menu_gross")],
            [InlineKeyboardButton("💸 Затраты", callback_data="menu_expenses")],
            [InlineKeyboardButton("📈 АНАЛИТИКА", callback_data="menu_analytics")],  # 🆕 v9.4.9
            [InlineKeyboardButton("🗄️ Архив", callback_data="archive|root")],
            [InlineKeyboardButton("📈 Статистика", callback_data="show_stats")],  # v2.0
        ]
    elif user_role == "subadmin":
        rows = [
            [InlineKeyboardButton("📊 Дебиторка", callback_data="menu_debt")],
            [InlineKeyboardButton("📦 Остатки", callback_data="direct|INVENTORY_SIMPLE|general")],
            [InlineKeyboardButton("🛒 Продажи", callback_data="menu_sales")],
            [InlineKeyboardButton("💰 Валовая", callback_data="submenu|GROSS_PCT")],
            [InlineKeyboardButton("💸 Затраты", callback_data="menu_expenses")],
            [InlineKeyboardButton("📈 АНАЛИТИКА", callback_data="menu_analytics")],  # 🆕 v9.4.9
            [InlineKeyboardButton("🗄️ Архив", callback_data="archive|root")],
        ]
    else:  # manager
        my_name = get_my_manager_name(chat_id) or "Unknown"
        rows = [
            [InlineKeyboardButton("📊 Дебиторка", callback_data="menu_debt_manager")],
            [InlineKeyboardButton("📦 Остатки", callback_data="direct|INVENTORY_SIMPLE|general")],
            [InlineKeyboardButton("🛒 Продажи", callback_data="menu_sales_manager")],
            [InlineKeyboardButton("💰 Валовая", callback_data=f"direct|GROSS_PCT|{my_name}")],
            [InlineKeyboardButton("🗄️ Архив", callback_data="archive|root")],
        ]
    return InlineKeyboardMarkup(rows)

def kb_debt_menu(user_role: str) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("📊 Простой", callback_data="submenu|DEBT_SIMPLE")],
        [InlineKeyboardButton("📈 Детальный", callback_data="submenu|DEBT_EXTENDED")],
    ]
    if user_role == "admin":
        rows.append([InlineKeyboardButton("📋 Сводный", callback_data="direct|DEBT_SIMPLE|summary")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back_main")])
    return InlineKeyboardMarkup(rows)

def kb_debt_menu_manager(my_name: str) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("📊 Простой", callback_data=f"direct|DEBT_SIMPLE|{my_name}")],
        [InlineKeyboardButton("📈 Детальный", callback_data=f"direct|DEBT_EXTENDED|{my_name}")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back_main")],
    ]
    return InlineKeyboardMarkup(rows)

def kb_sales_menu(user_role: str) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🛒 По клиентам", callback_data="submenu|SALES_SIMPLE")],
        [InlineKeyboardButton("🛒 По товару", callback_data="submenu|SALES_EXTENDED")],
    ]
    if user_role == "admin":
        rows.append([InlineKeyboardButton("📋 Сводный", callback_data="direct|SALES_SIMPLE|summary")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back_main")])
    return InlineKeyboardMarkup(rows)

def kb_sales_menu_manager(my_name: str) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🛒 По клиентам", callback_data=f"direct|SALES_SIMPLE|{my_name}")],
        [InlineKeyboardButton("🛒 По товару", callback_data=f"direct|SALES_EXTENDED|{my_name}")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back_main")],
    ]
    return InlineKeyboardMarkup(rows)

def kb_gross_menu(user_role: str) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("💰 Проценты", callback_data="submenu|GROSS_PCT")],
    ]
    if user_role == "admin":
        rows.append([InlineKeyboardButton("💰 Суммы", callback_data="direct|GROSS_SUM|general")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back_main")])
    return InlineKeyboardMarkup(rows)

def kb_choose_manager_new(action: str, scopes: List[str]) -> InlineKeyboardMarkup:
    rows = []
    for i in range(0, len(scopes), 2):
        row = []
        for j in range(2):
            if i + j < len(scopes):
                mgr = scopes[i + j]
                emoji = gender_emoji(mgr)
                if action == "AI_ANALYSIS":
                    row.append(InlineKeyboardButton(f"{emoji} {mgr}", callback_data=f"ai_only|{mgr}"))
                else:
                    row.append(InlineKeyboardButton(f"{emoji} {mgr}", callback_data=f"direct|{action}|{mgr}"))
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back_submenu")])
    return InlineKeyboardMarkup(rows)

def kb_open_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("📱 Открыть меню", callback_data="back_main")]])

# v9.4.28 ───────────────────────────────────────────────────
def kb_ai_type_menu(user_role: str) -> InlineKeyboardMarkup:
    """Меню выбора типа AI анализа (v9.4.28)"""
    rows = [
        [InlineKeyboardButton("📋 Дебиторка",    callback_data="ai_type|DEBT")],
        [InlineKeyboardButton("🛒 Продажи",       callback_data="ai_type|SALES")],
        [InlineKeyboardButton("💰 Валовая",        callback_data="ai_type|GROSS")],
        [InlineKeyboardButton("📦 Остатки",        callback_data="ai_type|INVENTORY")],
    ]
    if user_role in ("admin", "subadmin"):
        rows.append([InlineKeyboardButton("💸 Затраты",    callback_data="ai_type|EXPENSES")])
    rows.append([InlineKeyboardButton("⬅️ Назад",            callback_data="back_main")])
    return InlineKeyboardMarkup(rows)


def kb_choose_manager_for_ai(report_type: str, scopes: List[str]) -> InlineKeyboardMarkup:
    """Выбор менеджера для AI анализа конкретного типа (v9.4.28)"""
    rows = []
    for i in range(0, len(scopes), 2):
        row = []
        for j in range(2):
            if i + j < len(scopes):
                mgr = scopes[i + j]
                emoji = gender_emoji(mgr)
                row.append(InlineKeyboardButton(
                    f"{emoji} {mgr}",
                    callback_data=f"ai_run|{report_type}|{mgr}"
                ))
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="ai_type_menu")])
    return InlineKeyboardMarkup(rows)
# ────────────────────────────────────────────────────────────

# Блок 6.2_______________Клавиатуры архива__________________________________
def kb_archive_managers(managers: List[str]) -> InlineKeyboardMarkup:
    """Клавиатура выбора менеджера для архива"""
    rows = []
    for i in range(0, len(managers), 2):
        row = []
        for j in range(2):
            if i + j < len(managers):
                mgr = managers[i + j]
                emoji = gender_emoji(mgr)
                row.append(InlineKeyboardButton(f"{emoji} {mgr}", callback_data=f"archive|mgr|{mgr}"))
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back_main")])
    return InlineKeyboardMarkup(rows)

def kb_archive_dates(manager: str, dates: List[str]) -> InlineKeyboardMarkup:
    """Клавиатура выбора даты для архива"""
    rows = []
    # Показываем до 20 последних дат по 3 кнопки в ряд
    for i in range(0, min(len(dates), 20), 3):
        row = []
        for j in range(3):
            if i + j < len(dates):
                date = dates[i + j]
                row.append(InlineKeyboardButton(f"📅 {date}", callback_data=f"archive|date|{manager}|{date}"))
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="archive|root")])
    return InlineKeyboardMarkup(rows)

def kb_archive_types(manager: str, date: str, types: List[str]) -> InlineKeyboardMarkup:
    """Клавиатура выбора типа отчета для архива"""
    type_names = {
        "DEBT_SIMPLE": "📊 Простая дебиторка",
        "DEBT_EXTENDED": "📈 Детальная дебиторка",
        "SALES_SIMPLE": "🛒 Продажи по клиентам",
        "SALES_EXTENDED": "🛒 Продажи по товару",
        "INVENTORY_SIMPLE": "📦 Остатки",
        "GROSS_PCT": "💰 Валовая проценты",
        "GROSS_SUM": "💰 Валовая суммы",
        "AI": "🤖 ИИ анализ",
    }
    rows = []
    for t in types:
        label = type_names.get(t, t)
        rows.append([InlineKeyboardButton(label, callback_data=f"archive|get|{manager}|{date}|{t}")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data=f"archive|mgr|{manager}")])
    return InlineKeyboardMarkup(rows)

# Блок 7_______________Отправка файлов (с retry и автоудалением)_____________
TYPE_EMOJI = {
    "ДЕБИТОРКА": "📊", "ДЕБИТОРКА ДЕТАЛЬНО": "📈",
    "ПРОДАЖИ": "🛒", "ПРОДАЖИ ТОВАРЫ": "🛒",
    "ОСТАТКИ": "📦",
    "ВАЛ СУММЫ": "💰", "ВАЛ ПРОЦ": "💰",
    "АНАЛИЗ ИИ": "🤖"
}

def _caption(section_rus: str, mgr: str, date_str: str) -> str:
    emoji = TYPE_EMOJI.get(section_rus, "")
    who = f"{gender_emoji(mgr)} {mgr}" if mgr and mgr != "Сводный отчёт" else "🏢 Сводный отчёт"
    return f"{emoji} {section_rus}\n{who}\n{date_str}"

async def send_with_acl(section: str, intended_mgr: str,
                        chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    user_role = get_user_role(chat_id)
    scopes = user_scopes(chat_id)
    
    # MENU-ANCHOR: перед отправкой отчёта прячем меню, чтобы оно вернулось внизу
    await hide_main_menu(context, chat_id)
    section_rus = SECTIONS.get(section, section)
    
    # v9.4.7: Логируем запрос
    log_user_request(chat_id, section, intended_mgr)
    
    if intended_mgr == "Сводный отчёт" and section_rus not in ("ОСТАТКИ",) and user_role != "admin":
        await context.bot.send_message(chat_id, "⛔ Сводные отчёты по этому разделу доступны только администратору.")
        log_user_delivery(chat_id, section, False)
        return
    if intended_mgr != "Сводный отчёт" and intended_mgr not in scopes:
        await context.bot.send_message(chat_id, "⛔ Нет доступа к отчётам этого менеджера.")
        log_user_delivery(chat_id, section, False)
        return
    p = find_report(section, intended_mgr if intended_mgr != "Сводный отчёт" else None)
    # v9.4.15 Bug #11: Для SALES fallback на Сводный отчёт —
    # sales_report.py генерирует один общий файл без имени менеджера в названии,
    # поэтому per-manager файлы в индексе отсутствуют. Сводный содержит всех клиентов.
    sales_summary_fallback = False
    if (not p or not p.exists()) and section in ("SALES_SIMPLE", "SALES_EXTENDED") and intended_mgr != "Сводный отчёт":
        p = find_report(section, None)
        if p and p.exists():
            sales_summary_fallback = True
            log_event("sales_summary_fallback", section=section, intended_mgr=intended_mgr,
                      file=p.name, level="INFO")
    if not p or not p.exists():
        log_event("report_not_found", section=section, manager=intended_mgr)
        await context.bot.send_message(chat_id, f"❌ Отчёт не найден: {section_rus} для '{intended_mgr}'.")
        return
    full_text = _read_full(p)
    real_mgr = _extract_manager(full_text, p.name)
    
    if user_role != 'admin':
        is_intended_summary = (intended_mgr == "Сводный отчёт")
        is_real_summary = (real_mgr == "Сводный отчёт")
        # v9.4.15: SALES fallback — пропускаем проверку summary/manager mismatch,
        # т.к. это ожидаемое поведение (сводный отчёт показывается запросившему менеджеру)
        if not sales_summary_fallback:
            if is_intended_summary != is_real_summary:
                await context.bot.send_message(chat_id, "⛔ Ошибка безопасности: найден отчёт другого типа.")
                log_user_delivery(chat_id, section, False)
                return
            if not is_intended_summary and normalize_manager_name(intended_mgr) != normalize_manager_name(real_mgr):
                await context.bot.send_message(chat_id, "⛔ Ошибка безопасности: найден отчёт для другого менеджера.")
                log_user_delivery(chat_id, section, False)
                return
    
    # ✅ ИСПРАВЛЕНО: правильные отступы
    date_s = _extract_date(full_text, p.name, p)
    caption = _caption(section_rus, real_mgr, date_s)
    short_name = f"{section}_{real_mgr.replace(' ', '_')}.html"
    log_event("send_file", section=section, manager=real_mgr, file=p.name)
    max_retries = 2
    for attempt in range(max_retries):
        try:
            with p.open("rb") as f:
                sent_message = await context.bot.send_document(
                    chat_id=chat_id,
                    document=InputFile(f, filename=short_name),
                    caption=caption,
                    disable_notification=True,
                    protect_content=True,
                )
            
            # v9.4.6: Планируем автоудаление через 24 часа
            if sent_message and sent_message.message_id:
                schedule_message_deletion(
                    chat_id, 
                    sent_message.message_id,
                    sent_message.date.timestamp(),
                    AUTO_DELETE_HOURS
                )
            
            # v9.4.7.1: Логируем успешную доставку
            log_user_delivery(chat_id, section, True)
            
            # v9.4.33: Возвращаем в РОДИТЕЛЬСКИЙ РАЗДЕЛ (не на главное меню)
            await asyncio.sleep(0.3)
            try:
                await send_section_back(context, chat_id, user_role, section,
                                        text="✅ *Отчёт отправлен!*\n\n📋 Выберите раздел:")
            except Exception as e:
                logger.error(f"Ошибка отправки меню раздела: {e}")
            
            return
        except RetryAfter as e:
            if attempt < max_retries - 1:
                log_event("tg_retry_after", retry_seconds=e.retry_after, attempt=attempt+1)
                await asyncio.sleep(e.retry_after)
            else:
                log_event("tg_send_error", section=section, error="RetryAfter exhausted")
                await context.bot.send_message(chat_id, "⏳ Telegram перегружен, попробуйте позже.")
                return
        except BadRequest as e:
            error_msg = str(e).lower()
            if "file is too big" in error_msg or "too large" in error_msg:
                log_event("tg_file_too_big", section=section, file=p.name, size_mb=round(p.stat().st_size/1024/1024, 2))
                await context.bot.send_message(
                    chat_id, 
                    f"⚠️ Файл слишком большой для отправки.\n📄 Имя: {p.name}\n📏 Размер: {round(p.stat().st_size/1024/1024, 1)} МБ"
                )
                return
            else:
                log_event("tg_send_error", section=section, error=str(e))
                await context.bot.send_message(chat_id, f"❌ Ошибка Telegram: {str(e)[:100]}")
                return
        except Exception as e:
            log_event("tg_send_error", section=section, manager=real_mgr, error=str(e))
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
            else:
                await context.bot.send_message(chat_id, "❌ Ошибка при отправке файла.")
                return

# Блок 8_______________Фоновые задачи (pipeline + скрипты)__________________
async def run_script_async(script_name: str, *args: str, timeout: int = 600) -> Tuple[int, str, str]:
    script_path = ROOT_DIR / script_name
    if not script_path.exists():
        script_path_tool = ROOT_DIR / "tools" / script_name
        if not script_path_tool.exists():
            log_event("script_not_found", script=script_name, level="ERROR")
            return -1, "", f"Script not found: {script_path}"
        script_path = script_path_tool
    command = [sys.executable, str(script_path), *args]
    log_event("run_script_start", script=script_name, args=args)
    try:
        process = await asyncio.create_subprocess_exec(
            *command, 
            stdout=asyncio.subprocess.PIPE, 
            stderr=asyncio.subprocess.PIPE, 
            cwd=ROOT_DIR
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        rc = process.returncode or 0
        stdout_str = stdout.decode('utf-8', 'replace').strip()
        stderr_str = stderr.decode('utf-8', 'replace').strip()
        if stdout_str:
            log_event("script_stdout", script=script_name, output=stdout_str)
        if stderr_str and rc != 0:
            log_event("script_stderr", script=script_name, output=stderr_str, level="WARNING")
        log_event("run_script_finish", script=script_name, return_code=rc)
        return rc, stdout_str, stderr_str
    except Exception as e:
        log_event("script_exec_error", script=script_name, error=str(e), level="ERROR")
        return -1, "", str(e)


def _should_notify_manager_today(manager_name: str, period_str: str) -> bool:
    """
    v9.4.25: Подекадные уведомления менеджерам.
    Возвращает True только если в эту декаду ещё не отправляли.
    Декады: 1-10, 11-20, 21-конец месяца.
    Admin получает всегда — эта функция только для менеджеров.
    """
    try:
        from datetime import date as _date
        import re as _re

        # Определяем дату конца периода
        _d = None
        m = _re.search(r"(\d{1,2})[\./ ](\d{1,2})[\./ ](\d{4})", period_str)
        # Для диапазона — берём КОНЕЦ (второй match)
        all_m = list(_re.finditer(r"(\d{1,2})[\./ ](\d{1,2})[\./ ](\d{4})", period_str))
        if all_m:
            lm = all_m[-1]
            _d = _date(int(lm.group(3)), int(lm.group(2)), int(lm.group(1)))
        else:
            # Русские месяцы
            RU = {"января":1,"февраля":2,"марта":3,"апреля":4,"мая":5,"июня":6,
                  "июля":7,"августа":8,"сентября":9,"октября":10,"ноября":11,"декабря":12}
            m2 = _re.search(r"(\d{1,2})\s+([а-яё]+)\s+(\d{4})", period_str.lower())
            if m2 and m2.group(2) in RU:
                _d = _date(int(m2.group(3)), RU[m2.group(2)], int(m2.group(1)))

        if _d is None:
            return True  # Не смогли определить — шлём на всякий случай

        decade = (_d.day - 1) // 10  # 0=1-10, 1=11-20, 2=21-31
        decade_key = f"{_d.year}-{_d.month:02d}-d{decade}"

        # Читаем состояние
        state = {}
        if SALES_NOTIFY_DECADE_PATH.exists():
            try:
                state = json.loads(SALES_NOTIFY_DECADE_PATH.read_text(encoding="utf-8"))
            except Exception:
                state = {}

        mgr_key = manager_name.lower()
        if state.get(mgr_key) == decade_key:
            return False  # Уже отправляли в эту декаду

        # Запоминаем
        state[mgr_key] = decade_key
        SALES_NOTIFY_DECADE_PATH.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return True

    except Exception as _e:
        logger.warning(f"_should_notify_manager_today: {_e}")
        return True  # При любой ошибке — шлём



async def pipeline_task(context: ContextTypes.DEFAULT_TYPE):
    log_event("pipeline_cycle_start")
    _imap_rc, _imap_out, _imap_err = await run_script_async("imap_fetcher.py", "--once")
    # v9.4.25: Уведомляем admin если почта не ответила
    if _imap_rc != 0 and ADMIN_CHAT_ID:
        try:
            _err_preview = (_imap_err or _imap_out or "")[:200].strip()
            await context.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text=(
                    f"⚠️ Почта не отвечает ({datetime.now(TZ).strftime('%H:%M')})\n"
                    f"Файлы за этот цикл не скачаны.\n"
                    f"Проверь mailbox минбаракат.\n"
                    + (f"\nОшибка: {_err_preview}" if _err_preview else "")
                )
            )
            log_event("imap_error_alert_sent", rc=_imap_rc)
        except Exception as _ae:
            logger.warning(f"imap_error_alert: {_ae}")
    queue_files = sorted([p for p in QUEUE_DIR.glob("*.xls*") if not p.name.startswith("~")])
    processed_files = 0
    _pipeline_sent_admin_periods: set = set()  # v9.4.23: дедупликация сводных файлов
    _pipeline_managers_by_period: dict = {}    # v9.4.24: аккумулятор {period: [{manager,revenue,clients}]}
    _pipeline_seen_mgr_periods: set = set()    # v9.4.24: защита от дублей (manager, period)
    
    # v9.4.7.5: Batch-логирование cash-отчётов (экономия ~240 строк логов/час)
    skipped_cash = []
    
    if not queue_files:
        log_event("queue_empty")
    else:
        log_event("queue_found_files", count=len(queue_files))
        RE_SALES = re.compile(r"(sales|продаж)", re.I)
        RE_GROSS = re.compile(r"(gross|валов)", re.I)
        RE_INV = re.compile(r"(остат|inventory|товар.*склад|партия.*товар|ведомость.*склад)", re.I)
        RE_EXP = re.compile(r"(затрат|расход|expense)", re.I)
        
        for file_path in queue_files:
            try:
                fname_lower = file_path.name.lower()
                # v9.4.13.3: Автоперенос cash-файлов в rejected/cash вместо простого skip
                if any(keyword in fname_lower for keyword in ["денежн", "средств", "касс", "банк"]):
                    # Генерируем уникальное имя с timestamp для избежания перезаписи
                    timestamp = datetime.now(TZ).strftime('%Y%m%d_%H%M%S')
                    rejected_path = REJECTED_CASH_DIR / f"{timestamp}_{file_path.name}"
                    
                    # Переносим файл
                    shutil.move(file_path, rejected_path)
                    skipped_cash.append(file_path.name)
                    log_event("cash_file_rejected", 
                             original=file_path.name, 
                             moved_to=rejected_path.name,
                             level="INFO")
                    continue
                
                script_executed = False
                script_rc = -1
                
                if RE_INV.search(fname_lower):
                    script_rc, _, _ = await run_script_async("inventory.py", str(file_path))
                    script_executed = True
                    # v9.4.14: inventory_cost_parser ТОЛЬКО для файлов с партиями (есть себестоимость)
                    # "Остатки всем" и "Ведомость по товарам" — без себестоимости, пропускаем
                    is_partii = any(kw in fname_lower for kw in ["партии", "партиям", "партия"])
                    if script_rc == 0 and is_partii:
                        clean_path = CLEAN_DIR / f"{file_path.name}.__clean.xlsx"
                        if clean_path.exists():
                            rc_p, _, _ = await run_script_async("inventory_cost_parser.py", str(clean_path))
                            log_event("inventory_parser_done", file=file_path.name, rc=rc_p)
                        else:
                            log_event("inventory_parser_skip_no_clean", file=file_path.name, level="WARNING")
                    elif script_rc == 0 and not is_partii:
                        log_event("inventory_parser_skip_no_cost", file=file_path.name,
                                  reason="Файл без себестоимости (Остатки/Ведомость товаров)")
                elif RE_SALES.search(fname_lower):
                    script_rc, _, _ = await run_script_async("sales_report.py", str(file_path))
                    script_executed = True
                    # v9.4.10: Парсим JSON для аналитики
                    if script_rc == 0:
                        clean_path = CLEAN_DIR / f"{file_path.name}.__clean.xlsx"
                        if clean_path.exists():
                            rc_p, _, _ = await run_script_async("sales_parser.py", str(clean_path))
                            log_event("sales_parser_done", file=file_path.name, rc=rc_p)
                        else:
                            log_event("sales_parser_skip_no_clean", file=file_path.name, level="WARNING")
                        # v9.4.23: Краткая сводка после обработки файла продаж
                        # Определяем менеджера по имени файла
                        _sales_manager = ""
                        for _m in get_managers_list():
                            if _m.lower() in fname_lower:
                                _sales_manager = _m
                                break

                        # Находим JSON созданный sales_parser.py
                        _sales_json = None
                        try:
                            # JSON называется: sales_{slug_файла}.json
                            # Ищем по маске — последний созданный для этого файла
                            _clean_stem = file_path.name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".xlsx", "")
                            _candidates = sorted(
                                JSON_DIR.glob(f"sales_*{_clean_stem}*.json"),
                                key=lambda p: p.stat().st_mtime, reverse=True
                            )
                            if _candidates:
                                _sales_json = _candidates[0]
                        except Exception as _e:
                            logger.warning(f"Поиск sales JSON: {_e}")

                        if _sales_json and _sales_json.exists():
                            try:
                                import json as _j
                                _raw = _j.loads(_sales_json.read_text(encoding="utf-8"))
                                _prd = _raw.get("period", "")

                                if _sales_manager:
                                    # v9.4.24: МЕНЕДЖЕРСКИЙ файл — накапливаем, не шлём сразу.
                                    # Финальный рейтинг со всеми пошлём в конце pipeline.
                                    _cycle_key = (_prd, _sales_manager)
                                    if _cycle_key not in _pipeline_seen_mgr_periods:
                                        _pipeline_seen_mgr_periods.add(_cycle_key)
                                        if _prd not in _pipeline_managers_by_period:
                                            _pipeline_managers_by_period[_prd] = []
                                        _pipeline_managers_by_period[_prd].append({
                                            "manager":       _sales_manager,
                                            "total_revenue": float(_raw.get("total_revenue", 0)),
                                            "clients":       _raw.get("clients", []),
                                        })
                                        log_event("sales_pipeline_accumulated",
                                                  manager=_sales_manager, period=_prd)
                                    else:
                                        log_event("sales_pipeline_summary_dedup",
                                                  period=_prd, manager=_sales_manager)
                                else:
                                    # СВОДНЫЙ файл (без менеджера) — отправляем admin сразу
                                    if _prd not in _pipeline_sent_admin_periods:
                                        _pipeline_sent_admin_periods.add(_prd)
                                        await send_sales_pipeline_summary(
                                            context, _sales_json, ""
                                        )
                                    else:
                                        log_event("sales_pipeline_summary_dedup",
                                                  period=_prd, manager="")
                            except Exception as _e:
                                logger.error(f"Ошибка pipeline-сводки продаж: {_e}")
                        else:
                            logger.warning(f"sales_pipeline_summary: JSON не найден для {file_path.name}")
                elif RE_GROSS.search(fname_lower):
                    log_event("gross_processing_start", file=file_path.name)
                    rc1, stdout1, stderr1 = await run_script_async("gross_report.py", str(file_path))
                    if rc1 != 0:
                        log_event("gross_sum_error", file=file_path.name, return_code=rc1, stderr=stderr1[:200], level="WARNING")
                    else:
                        log_event("gross_sum_success", file=file_path.name)
                    clean_path = CLEAN_DIR / f"{file_path.name}.__clean.xlsx"
                    if clean_path.exists():
                        rc2, stdout2, stderr2 = await run_script_async("gross_report_pct.py", str(clean_path))
                        if rc2 != 0:
                            log_event("gross_pct_error", file=file_path.name, return_code=rc2, stderr=stderr2[:200], level="WARNING")
                        else:
                            log_event("gross_pct_success", file=file_path.name)
                    else:
                        log_event("gross_pct_skip_no_clean", file=file_path.name, level="WARNING")
                        rc2 = -1
                    log_event("gross_processing_complete", file=file_path.name, sum_rc=rc1, pct_rc=rc2)
                    # v9.4.10: Парсим JSON для аналитики
                    if rc1 == 0:
                        if clean_path.exists():
                            rc_p, _, _ = await run_script_async("gross_parser.py", str(clean_path))
                            log_event("gross_parser_done", file=file_path.name, rc=rc_p)
                        else:
                            log_event("gross_parser_skip_no_clean", file=file_path.name, level="WARNING")
                    script_executed = True
                    script_rc = rc1
                elif RE_EXP.search(fname_lower):
                    # Затраты (расходы): строим HTML + JSON (v9.4.14: добавлен expenses_parser)
                    try:
                        # expenses_report.py читает XLSX через pandas; предпочтительно использовать clean-копию
                        clean_path = CLEAN_DIR / f"{file_path.name}.__clean.xlsx"
                        if file_path.suffix.lower() == ".xlsx":
                            try:
                                from utils_excel import ensure_clean_xlsx
                                clean_path = ensure_clean_xlsx(file_path)
                            except Exception as e:
                                log_event("expenses_clean_error", file=file_path.name, error=str(e), level="WARNING")
                        target = str(clean_path) if clean_path.exists() else str(file_path)
                        script_rc, _, _ = await run_script_async("expenses_report.py", target)
                        script_executed = True
                        log_event("expenses_report_done", file=file_path.name, rc=script_rc)
                        # v9.4.14: запускаем expenses_parser.py для JSON → аналитика net_profit
                        if script_rc == 0 and clean_path.exists():
                            rc_p, _, _ = await run_script_async("expenses_parser.py", str(clean_path))
                            log_event("expenses_parser_done", file=file_path.name, rc=rc_p)
                        else:
                            log_event("expenses_parser_skip", file=file_path.name,
                                      reason="expenses_report failed or no clean file", level="WARNING")
                    except Exception as e:
                        script_rc = -1
                        script_executed = True
                        log_event("expenses_report_error", file=file_path.name, error=str(e), level="ERROR")
                else:
                    script_rc, _, _ = await run_script_async("debt_auto_report.py", str(file_path))
                    script_executed = True

                # v9.4.8: AI генерация отключена (теперь еженедельная)
                if script_executed and script_rc == 0:
                    pass  # schedule_ai_generation отключена
                    # Логируем что пропустили
                    try:
                        fname = file_path.name.lower()
                        for manager in get_managers_list():
                            if manager.lower() in fname:
                                log_event("ai_daily_skipped", manager=manager, reason="Weekly AI mode")
                                break
                    except Exception:
                        pass

                
                if script_executed and script_rc == 0:
                    processed_files += 1
                
                if file_path.exists():
                    processed_path = PROCESSED_DIR / f"{datetime.now(TZ).strftime('%Y%m%d%H%M%S')}_{file_path.name}"
                    shutil.move(file_path, processed_path)
                    log_event("file_processed", original=file_path.name, moved_to=processed_path.name)
                else:
                    # Оригинал мог быть удалён на этапе ensure_clean_xlsx (нормально для 1С-файлов).
                    log_event("queue_file_already_removed", file=file_path.name, level="INFO")
            except Exception as e:
                import traceback
                full_traceback = traceback.format_exc()
                log_event("file_processing_error", file=file_path.name, error=str(e), traceback=full_traceback, level="ERROR")
                logger.error(f"ПОЛНЫЙ TRACEBACK для {file_path.name}:\n{full_traceback}")
    
    # v9.4.13.3: Логируем перенесённые cash-отчёты одной строкой
    if skipped_cash:
        log_event("cash_files_moved_to_rejected", count=len(skipped_cash), files=skipped_cash[:3])
    
    if queue_files:
        await _build_index(force=True)
        log_event("index_rebuilt_after_generation", files_processed=len(queue_files))
    
    if processed_files > 0:
        try:
            await check_and_send_silence_alerts(context)
        except Exception as e:
            logger.error(f"❌ Ошибка проверки дней молчания: {e}", exc_info=True)
        # v9.4.24: Финальный рейтинг продаж — в конце цикла, когда все JSON готовы.
        # Используем аккумулятор _pipeline_managers_by_period (точные имена из filename).
        if _pipeline_managers_by_period and SalesSummary:
            try:
                _s = SalesSummary()
                # Субадмин-константы (Алена видит свою команду отдельно)
                _SUBADMIN_NAME  = "Алена"
                _SUBADMIN_SCOPE = ["Магира", "Оксана"]
                _SUBADMIN_CID   = MANAGERS_MAP.get(_SUBADMIN_NAME) if MANAGERS_MAP else None

                for _period, _mgrs in _pipeline_managers_by_period.items():
                    if not _mgrs:
                        continue
                    # Сортируем по выручке
                    _mgrs_sorted = sorted(_mgrs, key=lambda x: x["total_revenue"], reverse=True)

                    # 1. ADMIN: полный рейтинг всех менеджеров
                    if ADMIN_CHAT_ID:
                        _data = {
                            "date":          _period,
                            "total_amount":  sum(m["total_revenue"] for m in _mgrs_sorted),
                            "clients_count": sum(len(m.get("clients", [])) for m in _mgrs_sorted),
                            "clients":       [],
                        }
                        _admin_txt = _s.format_admin_pipeline(_data, _mgrs_sorted)
                        try:
                            _msg = await context.bot.send_message(
                                chat_id=ADMIN_CHAT_ID, text=_admin_txt, parse_mode=None
                            )
                            schedule_message_deletion(ADMIN_CHAT_ID, _msg.message_id,
                                _msg.date.timestamp(), delay_hours=24)
                            log_event("sales_pipeline_admin_final",
                                      period=_period, managers=len(_mgrs_sorted))
                        except Exception as _e:
                            logger.warning(f"Admin финальный рейтинг продаж: {_e}")

                    # 2. КАЖДОМУ МЕНЕДЖЕРУ: свои данные + место в рейтинге (конкуренция)
                    # v9.4.25: Подекадно — шлём только раз в декаду, не при каждом файле
                    if MANAGERS_MAP:
                        for _mgr in _mgrs_sorted:
                            _mgr_name = _mgr["manager"]
                            _mgr_cid  = MANAGERS_MAP.get(_mgr_name)
                            if not _mgr_cid:
                                continue
                            if not _should_notify_manager_today(_mgr_name, _period):
                                log_event("sales_notify_decade_skip",
                                          manager=_mgr_name, period=_period)
                                continue
                            _mgr_data = {
                                "date":          _period,
                                "total_amount":  _mgr["total_revenue"],
                                "clients_count": len(_mgr.get("clients", [])),
                                "clients":       _mgr.get("clients", []),
                            }
                            _mgr_txt = _s.format_manager_pipeline(
                                _mgr_name, _mgr_data, all_managers=_mgrs_sorted
                            )
                            try:
                                _msg = await context.bot.send_message(
                                    chat_id=_mgr_cid, text=_mgr_txt, parse_mode=None
                                )
                                schedule_message_deletion(_mgr_cid, _msg.message_id,
                                    _msg.date.timestamp(), delay_hours=24)
                                log_event("sales_pipeline_summary_sent",
                                          recipient=_mgr_name, period=_period)
                            except Exception as _e2:
                                logger.warning(f"Сводка менеджеру {_mgr_name}: {_e2}")

                    # 3. СУБАДМИН (Алена): мини-рейтинг своей команды (Алена+Магира+Оксана)
                    if _SUBADMIN_CID:
                        _scope_data = [
                            m for m in _mgrs_sorted
                            if m["manager"] in [_SUBADMIN_NAME] + _SUBADMIN_SCOPE
                        ]
                        if len(_scope_data) > 1:
                            _sub_txt = _s.format_subadmin_pipeline(
                                _SUBADMIN_NAME, _scope_data, _period
                            )
                            try:
                                _msg = await context.bot.send_message(
                                    chat_id=_SUBADMIN_CID, text=_sub_txt, parse_mode=None
                                )
                                schedule_message_deletion(_SUBADMIN_CID, _msg.message_id,
                                    _msg.date.timestamp(), delay_hours=24)
                                log_event("sales_pipeline_subadmin_sent",
                                          manager=_SUBADMIN_NAME, period=_period,
                                          scope_count=len(_scope_data))
                            except Exception as _e3:
                                logger.warning(f"Субадмин-сводка продаж: {_e3}")

            except Exception as _e:
                logger.warning(f"Финальный рейтинг продаж (outer): {_e}")
    try:
        today_archive_dir = ARCHIVE_DIR / datetime.now(TZ).strftime('%Y-%m-%d')
        today_archive_dir.mkdir(exist_ok=True)
        for report_dir in [HTML_DIR, JSON_DIR, AI_DIR]:
            if not report_dir.exists():
                continue
            for report_file in report_dir.glob("*.*"):
                if (time.time() - report_file.stat().st_mtime) < (PIPELINE_INTERVAL_MIN * 60 * 1.5):
                    shutil.copy2(report_file, today_archive_dir / report_file.name)
        archive_limit_bytes = 2.0 * 1024**3
        total_size = sum(f.stat().st_size for f in ARCHIVE_DIR.glob('**/*') if f.is_file())
        if total_size > archive_limit_bytes:
            dirs = sorted([d for d in ARCHIVE_DIR.iterdir() if d.is_dir()])
            for old_dir in dirs[:-7]:
                shutil.rmtree(old_dir)
                log_event("archive_cleanup", removed_dir=old_dir.name)
    except Exception as e:
        log_event("archive_error", error=str(e), level="ERROR")
    log_event("pipeline_cycle_finish")

async def check_and_send_silence_alerts(context=None):
    """Проверяет дни молчания у всех менеджеров и отправляет уведомления"""
    logger.info("🔔 Начинается проверка дней молчания...")
    alert = SilenceAlert()
    reports_dir = HTML_DIR
    all_managers_data = {}
    manager_dates = {}  # v9.4.23: дата отчёта по каждому менеджеру
    
    for manager in get_managers_list():
        try:
            latest_report = alert.get_latest_debt_report(reports_dir, manager)
            if not latest_report:
                logger.warning(f"⚠️ Не найден отчёт дебиторки для {manager}")
                continue
            clients_data = alert.parse_html_silence_days(latest_report)
            if not clients_data:
                logger.warning(f"⚠️ Не удалось распарсить данные для {manager}")
                continue
            # v1.4: исторические дни молчания из предыдущего файла
            prev_report = alert.get_prev_debt_report(reports_dir, manager)
            hist_map = alert.build_historical_silence_map(prev_report) if prev_report else {}
            categorized = alert.categorize_by_silence(clients_data, historical_map=hist_map)
            all_managers_data[manager] = categorized
            manager_dates[manager] = alert.parse_report_date(latest_report)  # v9.4.23
            total_silent = (len(categorized['critical']) + len(categorized['alarm']) + len(categorized['warning']))
            if total_silent == 0:
                logger.info(f"✅ У {manager} нет молчащих клиентов")
            else:
                logger.info(f"📊 У {manager}: {total_silent} молчащих клиентов")
        except Exception as e:
            logger.error(f"❌ Ошибка обработки {manager}: {e}", exc_info=True)
    
    for manager, categorized in all_managers_data.items():
        total_silent = (len(categorized['critical']) + len(categorized['alarm']) + len(categorized['warning']))
        
        chat_id = MANAGERS_MAP.get(manager)
        if not chat_id:
            logger.warning(f"⚠️ Не найден chat_id для {manager}")
            continue
        
        # v9.4.18: получаем дату данных из отчёта
        _latest = alert.get_latest_debt_report(HTML_DIR, manager)
        report_date = alert.parse_report_date(_latest) if _latest else ""
        
        subordinates = get_subordinates_for_subadmin(manager)
        
        sub_has_silent = False
        if subordinates:
            for sub_manager in subordinates:
                sub_cat = all_managers_data.get(sub_manager)
                if sub_cat:
                    sub_total = (len(sub_cat['critical']) + len(sub_cat['alarm']) + len(sub_cat['warning']))
                    if sub_total > 0:
                        sub_has_silent = True
                        break
        
        if total_silent == 0 and not sub_has_silent:
            continue
        
        if subordinates:
            logger.info(f"👤 {manager} - субадмин, подшефные: {', '.join(subordinates)}")
            
            message_parts = ["🔔 УВЕДОМЛЕНИЕ О ДНЯХ МОЛЧАНИЯ\n\n"]
            
            if total_silent > 0:
                message_parts.append(f"👤 Менеджер: {manager}\n")
                message_parts.append(alert.format_manager_alert(manager, categorized, report_date=report_date))
            else:
                message_parts.append(f"👤 Менеджер: {manager}\n")
                message_parts.append("✅ У вас нет молчащих клиентов\n")
            
            has_subordinate_alerts = False
            for sub_manager in subordinates:
                if sub_manager in all_managers_data:
                    sub_categorized = all_managers_data[sub_manager]
                    sub_total = (len(sub_categorized['critical']) + len(sub_categorized['alarm']) + len(sub_categorized['warning']))
                    if sub_total > 0:
                        has_subordinate_alerts = True
                        message_parts.append(f"\n{'='*50}\n\n")
                        _sub_latest = alert.get_latest_debt_report(HTML_DIR, sub_manager)
                        _sub_date = alert.parse_report_date(_sub_latest) if _sub_latest else ""
                        message_parts.append(f"👤 Подшефный: {sub_manager}\n")
                        message_parts.append(alert.format_manager_alert(sub_manager, sub_categorized, report_date=_sub_date))
            
            message = "".join(message_parts)
            
            if context:
                try:
                    await _tg_send_long(context, chat_id, message, parse_mode=None, delay_hours=24)
                    logger.info(f"✅ Уведомление отправлено субадмину {manager} (свои: {total_silent}, подшефные: {'есть' if has_subordinate_alerts else 'нет'})")
                    await send_main_menu(context, chat_id, get_user_role(chat_id))  # Fix #MENU-SILENCE
                except Exception as e:
                    logger.error(f"❌ Ошибка отправки {manager}: {e}")
        else:
            message = alert.format_manager_alert(manager, categorized, report_date=report_date)
            if message and context:
                try:
                    await _tg_send_long(context, chat_id, message, parse_mode=None, delay_hours=24)
                    logger.info(f"✅ Уведомление отправлено: {manager} ({total_silent} клиентов)")
                    await send_main_menu(context, chat_id, get_user_role(chat_id))  # Fix #MENU-SILENCE
                except Exception as e:
                    logger.error(f"❌ Ошибка отправки {manager}: {e}")
    
    if all_managers_data and context:
        try:
            # Детальная сводка для админа (с именами клиентов)
            admin_summary = alert.format_admin_detailed(all_managers_data, manager_dates=manager_dates)  # v9.4.23
            if ADMIN_CHAT_ID:
                await _tg_send_long(context, ADMIN_CHAT_ID, admin_summary, parse_mode=None, delay_hours=24)
                logger.info(f"✅ Детальная сводка отправлена админу")
                await send_main_menu(context, ADMIN_CHAT_ID, "admin")  # Fix #MENU-SILENCE
            else:
                logger.warning("⚠️ ADMIN_CHAT_ID не установлен")
        except Exception as e:
            logger.error(f"❌ Ошибка отправки сводки админу: {e}", exc_info=True)
    logger.info(f"🔔 Проверка дней молчания завершена")

# ─────────────────────────────────────────────────────────────────
# v9.4.26: УПУЩЕННАЯ ПРИБЫЛЬ
# ─────────────────────────────────────────────────────────────────

async def send_opportunity_loss_report(context=None):
    """
    v9.4.32: Считает и рассылает отчёт 'Упущенная прибыль'.

    Вызывается автоматически: пятница 14:05 (еженедельно).
    Принудительно: кнопка force|oploss в меню аналитики (любой день).

    - Admin  → сводная таблица по всем менеджерам
    - Subadmin → сводка по себе + подчинённым
    - Менеджер → только его данные
    """
    if not _OPPORTUNITY_LOSS_AVAILABLE:
        logger.warning("⚠️ opportunity_loss модуль не загружен — пропуск")
        return

    # v9.4.32: Еженедельный автозапуск — только по пятницам (4 = пятница)
    # Принудительная отправка (кнопка force|oploss) идёт через force_report_to_user напрямую
    if datetime.now(TZ).weekday() != 4:
        logger.info("💸 opportunity_loss: сегодня не пятница — пропуск автозапуска")
        return

    logger.info("💸 Расчёт упущенной прибыли...")

    all_data = []
    for manager in get_managers_list():
        try:
            data = calculate_opportunity_loss(HTML_DIR, manager)
            if data:
                all_data.append(data)
                logger.info(
                    f"💸 {manager}: upущено {data['total_loss']:,.0f} ₸ "
                    f"(☠️{len(data['zones']['dead'])} 🔴{len(data['zones']['red'])} "
                    f"⚡{len(data['zones']['yellow'])})"
                )
            else:
                logger.info(f"💸 {manager}: нет молчащих должников (>= 15 дней)")
        except Exception as e:
            logger.error(f"❌ opportunity_loss: ошибка расчёта для {manager}: {e}", exc_info=True)

    if not context:
        logger.warning("💸 opportunity_loss: context не передан, отправка невозможна")
        return

    # ── Admin: сводка по всем ──────────────────────────────────
    if ADMIN_CHAT_ID and all_data:
        try:
            admin_msg = format_opportunity_loss_admin(all_data)
            msg = await context.bot.send_message(
                chat_id=ADMIN_CHAT_ID, text=admin_msg, parse_mode=None
            )
            schedule_message_deletion(ADMIN_CHAT_ID, msg.message_id, msg.date.timestamp(), delay_hours=24)
            logger.info("✅ opportunity_loss: сводка отправлена admin")
            await send_main_menu(context, ADMIN_CHAT_ID, "admin")  # Fix #MENU-SILENCE
        except Exception as e:
            logger.error(f"❌ opportunity_loss: ошибка отправки admin: {e}")
    elif ADMIN_CHAT_ID and not all_data:
        try:
            await context.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text="💸 Упущенная прибыль: нет данных — у всех менеджеров молчащих должников (≥15 дней) нет.",
                parse_mode=None
            )
        except Exception:
            pass

    # ── Менеджеры и subadmin ───────────────────────────────────
    data_by_manager = {d["manager"]: d for d in all_data}

    for manager in get_managers_list():
        chat_id = MANAGERS_MAP.get(manager)
        if not chat_id:
            continue

        subordinates = get_subordinates_for_subadmin(manager)

        if subordinates:
            # subadmin: сводка по себе + подчинённым
            msg_text = format_opportunity_loss_subadmin(all_data, manager, subordinates)
            if not msg_text:
                logger.info(f"💸 subadmin {manager}: нет упущенной прибыли по команде")
                continue
        else:
            # обычный менеджер: только свои данные
            mgr_data = data_by_manager.get(manager)
            if not mgr_data:
                logger.info(f"💸 {manager}: нет молчащих должников — уведомление не отправляется")
                continue
            msg_text = format_opportunity_loss_message(mgr_data)

        try:
            msg = await context.bot.send_message(
                chat_id=chat_id, text=msg_text, parse_mode=None
            )
            schedule_message_deletion(chat_id, msg.message_id, msg.date.timestamp(), delay_hours=24)
            logger.info(f"✅ opportunity_loss: отправлено {manager} (chat_id={chat_id})")
            await send_main_menu(context, chat_id, get_user_role(chat_id))  # Fix #MENU-SILENCE
        except Exception as e:
            logger.error(f"❌ opportunity_loss: ошибка отправки {manager}: {e}")

    logger.info("💸 Расчёт упущенной прибыли завершён")


# ─────────────────────────────────────────────────────────────────
# v9.4.27: ПРИНУДИТЕЛЬНАЯ ОТПРАВКА (кнопка в меню аналитики)
# ─────────────────────────────────────────────────────────────────

FORCE_REPORT_TYPES = {
    "silence":   "🔔 Дебиторка",
    "oploss":    "💸 Упущенная прибыль",
    "sales":     "🛒 Продажи",
    "gross":     "💰 Валовая",
    "inventory": "📦 Остатки",
}

async def force_report_to_user(report_type: str, chat_id: int, context) -> str:
    """
    v9.4.27: Строит и отправляет отчёт конкретному пользователю по запросу.

    - Admin    → получает сводку по всем менеджерам
    - Subadmin → получает себя + подчинённых
    - Manager  → получает только свои данные
    
    Возвращает строку-статус для ответа на callback.
    """
    role         = get_user_role(chat_id)
    manager_name = get_my_manager_name(chat_id)   # None для admin
    subordinates = get_subordinates_for_subadmin(manager_name) if manager_name else []
    label        = FORCE_REPORT_TYPES.get(report_type, report_type)

    logger.info(f"🚀 force_report: {label} → chat_id={chat_id} role={role}")

    try:

        # ── ДЕБИТОРКА (silence alerts) ────────────────────────────────────────
        if report_type == "silence":
            alert = SilenceAlert()
            all_managers_data: Dict[str, Any] = {}
            manager_dates: Dict[str, str]     = {}

            scope = get_managers_list() if role == "admin" else (
                ([manager_name] + subordinates) if manager_name else []
            )

            for mgr in scope:
                latest = alert.get_latest_debt_report(HTML_DIR, mgr)
                if not latest:
                    continue
                clients = alert.parse_html_silence_days(latest)
                if not clients:
                    continue
                # v1.4: исторические дни молчания из предыдущего файла
                prev = alert.get_prev_debt_report(HTML_DIR, mgr)
                hist_map = alert.build_historical_silence_map(prev) if prev else {}
                all_managers_data[mgr] = alert.categorize_by_silence(clients, historical_map=hist_map)
                manager_dates[mgr]     = alert.parse_report_date(latest) or ""

            if not all_managers_data:
                return f"{label}: нет данных дебиторки."

            if role == "admin":
                msg_text = alert.format_admin_detailed(all_managers_data, manager_dates=manager_dates)
            elif subordinates and manager_name:
                # subadmin: как в check_and_send_silence_alerts
                parts = ["🔔 ДЕБИТОРКА — СЕЙЧАС\n\n"]
                own = all_managers_data.get(manager_name)
                own_date = manager_dates.get(manager_name, "")
                if own:
                    parts.append(f"👤 {manager_name}\n")
                    parts.append(alert.format_manager_alert(manager_name, own, report_date=own_date))
                else:
                    parts.append(f"👤 {manager_name}\n✅ Молчащих нет\n")
                for sub in subordinates:
                    sub_cat = all_managers_data.get(sub)
                    if sub_cat:
                        sub_date = manager_dates.get(sub, "")
                        parts.append(f"\n{'='*40}\n\n👤 {sub}\n")
                        parts.append(alert.format_manager_alert(sub, sub_cat, report_date=sub_date))
                msg_text = "".join(parts)
            else:
                own = all_managers_data.get(manager_name, {})
                own_date = manager_dates.get(manager_name, "")
                msg_text = alert.format_manager_alert(manager_name, own, report_date=own_date) if own else f"✅ {manager_name}: молчащих клиентов нет."

        # ── УПУЩЕННАЯ ПРИБЫЛЬ ─────────────────────────────────────────────────
        elif report_type == "oploss":
            if not _OPPORTUNITY_LOSS_AVAILABLE:
                return "💸 Модуль opportunity_loss не загружен."

            scope = get_managers_list() if role == "admin" else (
                ([manager_name] + subordinates) if manager_name else []
            )
            all_data = []
            for mgr in scope:
                d = calculate_opportunity_loss(HTML_DIR, mgr)
                if d:
                    all_data.append(d)

            if not all_data:
                return f"{label}: нет молчащих должников (≥15 дней)."

            if role == "admin":
                msg_text = format_opportunity_loss_admin(all_data)
            elif subordinates:
                msg_text = format_opportunity_loss_subadmin(all_data, manager_name, subordinates)
                if not msg_text:
                    return f"{label}: нет данных по команде."
            else:
                mgr_data = next((d for d in all_data if d["manager"] == manager_name), None)
                msg_text = format_opportunity_loss_message(mgr_data) if mgr_data else f"✅ {manager_name}: нет молчащих должников."

        # ── ПРОДАЖИ ───────────────────────────────────────────────────────────
        elif report_type == "sales":
            if not SalesSummary:
                return "🛒 Модуль SalesSummary не загружен."
            summary = SalesSummary()
            if role == "admin":
                # v9.4.32: агрегация ВСЕХ менеджеров из JSON, не одного HTML
                known_mgrs = set(m.lower() for m in get_managers_list())
                msg_text   = summary.build_admin_sales_summary(JSON_DIR, known_managers=known_mgrs)
                if not msg_text:
                    return "🛒 Нет данных продаж."
            else:
                # Ищем последний JSON для этого менеджера
                import json as _json
                candidates = sorted(
                    [p for p in JSON_DIR.glob("*.json")
                     if manager_name and manager_name.lower() in p.name.lower()
                     and "sales" in p.name.lower()],
                    key=lambda p: p.stat().st_mtime, reverse=True
                )
                if not candidates:
                    return f"🛒 Нет файла продаж для {manager_name}."
                with open(candidates[0], "r", encoding="utf-8") as _f:
                    raw = _json.load(_f)
                period_str = raw.get("period", "")
                clients_raw = raw.get("clients", [])
                data = {
                    "date": period_str,
                    "total_amount": float(raw.get("total_revenue", 0)),
                    "clients_count": len(clients_raw),
                    "clients": clients_raw,
                    "products": [],
                }
                msg_text = summary.format_manager_pipeline(manager_name, data)

        # ── ВАЛОВАЯ ───────────────────────────────────────────────────────────
        elif report_type == "gross":
            if not GrossSummary:
                return "💰 Модуль GrossSummary не загружен."
            summary = GrossSummary()
            if role == "admin":
                latest_html = summary.get_latest_gross_report(HTML_DIR)
                if not latest_html:
                    return "💰 Нет данных валовой прибыли."
                data = summary.parse_gross_html(latest_html)
                msg_text = summary.format_summary(data)
            else:
                # Ищем gross HTML этого менеджера
                mgr_lower = manager_name.lower() if manager_name else ""
                candidates = sorted(
                    [p for p in HTML_DIR.glob("*_gross_sum.html")
                     if mgr_lower and mgr_lower in p.name.lower()],
                    key=lambda p: p.stat().st_mtime, reverse=True
                )
                if not candidates:
                    return f"💰 Нет файла валовой для {manager_name}."
                data = summary.parse_gross_html(candidates[0])
                msg_text = summary.format_summary(data)

        # ── ОСТАТКИ ───────────────────────────────────────────────────────────
        elif report_type == "inventory":
            if not InventorySummary:
                return "📦 Модуль InventorySummary не загружен."
            summary = InventorySummary()
            latest_html = summary.get_latest_inventory_report(HTML_DIR)
            if not latest_html:
                return "📦 Нет данных остатков."
            data = summary.parse_inventory_html(latest_html)
            msg_text = summary.format_summary(data)

        else:
            return f"❓ Неизвестный тип отчёта: {report_type}"

        # ── Отправка (с разбивкой на чанки — сообщение может превышать 4096 символов)
        await _tg_send_long(context, chat_id, msg_text, parse_mode=None, delay_hours=24)
        logger.info(f"✅ force_report: {label} отправлен chat_id={chat_id}")
        return f"✅ {label} отправлен"

    except Exception as e:
        logger.error(f"❌ force_report_to_user [{report_type}] chat_id={chat_id}: {e}", exc_info=True)
        return f"❌ Ошибка при формировании {label}: {e}"


# Блок 9_______________Уведомления (с детектом обновлений)___________________
def _touch_notify_state(file_path: str) -> None:
    try:
        state = _load_json_safe(NOTIFY_STATE_PATH)
        state[file_path] = time.time()
        _save_json_atomic(NOTIFY_STATE_PATH, state)
    except Exception as e:
        log_event("save_state_error", error=str(e), level="ERROR")

async def new_reports_notifier(context: ContextTypes.DEFAULT_TYPE):
    log_event("notifier_start")
    MAX_NOTIFICATIONS_PER_CYCLE = 50
    state = _load_json_safe(NOTIFY_STATE_PATH)
    new_state = {}
    notifications = {}
    watermark_time = time.time() - (SCAN_INTERVAL_MIN * 60 * 2)
    all_files = list(HTML_DIR.glob("*.html")) + list(AI_DIR.glob("*.html"))
    notification_count = 0
    for file in all_files:
        try:
            mtime = file.stat().st_mtime
            new_state[str(file)] = mtime
            if mtime < watermark_time:
                continue
            prev_mtime = state.get(str(file))
            is_new = prev_mtime is None
            is_updated = (prev_mtime is not None) and (mtime > prev_mtime)
            is_recent = (time.time() - mtime) < (SCAN_INTERVAL_MIN * 60 * 1.5)
            if (is_new or is_updated) and is_recent:
                if notification_count >= MAX_NOTIFICATIONS_PER_CYCLE:
                    log_event("notifier_flood_protection", skipped_file=file.name, limit=MAX_NOTIFICATIONS_PER_CYCLE)
                    continue
                notification_count += 1
                full_text = _read_full(file)
                manager = _extract_manager(full_text, file.name)
                
                for mgr_name, chat_id in (MANAGERS_MAP or {}).items():
                    if normalize_manager_name(mgr_name) == normalize_manager_name(manager):
                        notifications.setdefault(chat_id, []).append(f"📊 {file.name}")
                subadmin_scopes = ROLES.get("subadmin_scopes", {})
                for str_chat_id, scope_list in subadmin_scopes.items():
                    if any(normalize_manager_name(manager) == normalize_manager_name(s) for s in scope_list):
                        chat_id = int(str_chat_id)
                        notifications.setdefault(chat_id, []).append(f"📊 {manager}: {file.name}")
                if ADMIN_CHAT_ID:
                    notifications.setdefault(ADMIN_CHAT_ID, []).append(f"📊 {manager}: {file.name}")
        except FileNotFoundError:
            continue
        except Exception as e:
            log_event("notifier_file_error", file=file.name, error=str(e))
    for chat_id, reports in notifications.items():
        if not isinstance(chat_id, int) or chat_id <= 0:
            log_event("notifier_skip_invalid_chat_id", chat_id=chat_id)
            continue
        try:
            current_time = datetime.now(TZ).strftime('%H:%M')
            if len(reports) > 10:
                message = f"🔔 Обновлено {len(reports)} отчетов ({current_time})"
            else:
                message = f"🔔 Новые отчеты ({current_time}):\n\n"
                for report in reports[:10]:
                    clean_report = report.replace("📊 ", "") if report.startswith("📊 ") else report
                    message += f"📊 {clean_report}\n"
                if len(reports) > 10:
                    message += f"\n... и еще {len(reports)-10} отчетов"
            await context.bot.send_message(chat_id=chat_id, text=message, reply_markup=kb_open_menu())
            log_event("notification_sent", chat_id=chat_id, reports_count=len(reports))
        except Exception as e:
            log_event("notification_error", chat_id=chat_id, error=str(e), level="ERROR")
    
    # v9.4.6.1: ПАТЧ - Атомарная запись с merge для защиты от race condition
    try:
        current = _load_json_safe(NOTIFY_STATE_PATH)
        if not isinstance(current, dict):
            current = {}
        merged = {**current, **new_state}
        _save_json_atomic(NOTIFY_STATE_PATH, merged)
    except Exception as e:
        log_event("save_state_error", error=str(e), level="ERROR")
    
    log_event("notifier_finish", total_notifications=notification_count, watermark_time=watermark_time)

# Блок 10_______________AI обработка (общая функция)_________________________
async def process_and_send_ai_analysis(
    manager: str,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    json_file: Path,
    start_time: float,
    status_msg_id: Optional[int],
    report_type: str = "DEBT"
) -> bool:
    log_event("ai_generate_start", manager=manager, json_file=str(json_file))
    try:
        rc, stdout_str, stderr_str = await run_script_async(
            "ai_analyzer.py",
            "--path", str(json_file),
            "--chat-id", str(chat_id),
            "--type", report_type
        )
        if rc != 0 and "AI saved:" not in stdout_str:
            log_event("ai_generate_error", manager=manager, return_code=rc, stderr=stderr_str[:200])
            if status_msg_id:
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id, message_id=status_msg_id,
                        text=f"❌ **Ошибка генерации ИИ анализа**\n\nПроверьте логи для деталей",
                        parse_mode="Markdown"
                    )
                except Exception:
                    pass
            return False
        match = re.search(r"AI saved:\s+(.+\.txt)", stdout_str)
        txt_file = None
        if match:
            txt_file_path = Path(match.group(1).strip())
            if txt_file_path.exists():
                txt_file = txt_file_path
                log_event("ai_txt_found", file=txt_file.name, manager=manager, source="stdout")
            else:
                log_event("ai_file_missing", file=txt_file_path.name, manager=manager)
        if not txt_file:
            log_event("ai_output_parse_error", stdout=stdout_str[:200])
            await asyncio.sleep(AI_PROCESSING_WAIT_SEC)
            txt_candidates = []
            if AI_DIR.exists():
                for txt_file_candidate in AI_DIR.glob("*.txt"):
                    try:
                        if txt_file_candidate.stat().st_mtime >= start_time and manager.lower() in txt_file_candidate.name.lower():
                            txt_candidates.append(txt_file_candidate)
                    except Exception:
                        continue
            if txt_candidates:
                txt_file = max(txt_candidates, key=lambda p: p.stat().st_mtime)
                log_event("ai_txt_found", file=txt_file.name, manager=manager, source="filesystem_search")
        if not txt_file:
            log_event("ai_file_not_found", manager=manager)
            if status_msg_id:
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id, message_id=status_msg_id,
                        text=f"❌ **Не удалось найти созданный файл**\n\nВозможно, файл создаётся дольше обычного",
                        parse_mode="Markdown"
                    )
                except Exception:
                    pass
            return False
        # v9.4.7.5: Используем встроенные функции html_to_path() и txt_to_html()
        try:
            html_file = html_to_path(txt_file)
            if not html_file.exists():
                txt_to_html(txt_file, html_file)
            log_event("ai_html_created", file=html_file.name, manager=manager)
        except Exception as e:
            log_event("ai_html_creation_error", error=str(e), manager=manager)
            if status_msg_id:
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id, message_id=status_msg_id,
                        text=f"❌ **Ошибка создания HTML**\n\nПроверьте логи",
                        parse_mode="Markdown"
                    )
                except Exception:
                    pass
            return False
        await send_ai_file(html_file, manager, chat_id, context)
        try:
            _touch_notify_state(str(html_file))
        except Exception:
            pass
        if status_msg_id:
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=status_msg_id,
                    text=f"✅ **ИИ анализ готов!**\n\n👤 Менеджер: {manager}\n\n📄 Файл отправлен выше",
                    parse_mode="Markdown"
                )
            except Exception:
                pass
        return True
    except Exception as e:
        log_event("ai_generate_error", error=str(e), manager=manager)
        if status_msg_id:
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=status_msg_id,
                    text=f"❌ **Ошибка при генерации ИИ анализа**\n\nПроверьте настройки AI сервиса",
                    parse_mode="Markdown"
                )
            except Exception:
                pass
        return False

def extract_date_from_filename(filename: str) -> Optional[str]:
    """
    v9.4.7.5: Извлекает дату из имени файла в формате YYYYMMDD
    
    Примеры:
    - debt_ext_Алена_20251115.json -> "20251115"
    - report_20251115_processed.xlsx -> "20251115"
    """
    # Ищем паттерн _YYYYMMDD в имени файла
    match = re.search(r'_(\d{8})', filename)
    if match:
        return match.group(1)
    
    # Альтернативный поиск - просто YYYYMMDD где-то в имени
    match2 = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if match2:
        return match2.group(0)
    
    return None

def find_recent_json_for_manager(manager: str, hours: int = 48, report_type: str = "") -> Optional[Path]:
    """
    v9.4.29: Поиск JSON по менеджеру + типу отчёта.
    report_type: DEBT, SALES, GROSS, INVENTORY, EXPENSES — фильтрует по префиксу файла.
    """
    if not JSON_DIR.exists():
        return None
    # Префиксы файлов по типу отчёта
    TYPE_PREFIXES = {
        "DEBT":      ("debt_ext_", "debt_"),
        "SALES":     ("sales_",),
        "GROSS":     ("gross_",),
        "INVENTORY": ("inventory_",),
        "EXPENSES":  ("expenses_",),
    }
    allowed_prefixes = TYPE_PREFIXES.get(report_type, ())
    cutoff_time = time.time() - (hours * 3600)
    candidates = []
    for json_file in JSON_DIR.glob("*.json"):
        try:
            mtime = json_file.stat().st_mtime
            if mtime < cutoff_time:
                continue
            fname = json_file.name.lower()
            # Фильтр по типу (если задан)
            if allowed_prefixes and not any(fname.startswith(p) for p in allowed_prefixes):
                continue
            # Фильтр по менеджеру (для типов с менеджерами)
            if report_type in ("DEBT", "SALES", "GROSS") and manager.lower() not in fname:
                continue
            # INVENTORY и EXPENSES — без фильтра менеджера
            candidates.append(json_file)
        except Exception:
            continue
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None

def find_newest_ai_file_for_manager(manager: str, after_time: float, report_type: str = "") -> Optional[Path]:
    """
    v9.4.29: Кеш AI-файлов с учётом типа отчёта.
    Ключ кеша = тип + менеджер, чтобы DEBT не выдавался при запросе EXPENSES.
    """
    # Префикс AI-файла по типу
    TYPE_AI_PREFIX = {
        "DEBT":      "ai_debt_",
        "SALES":     "ai_sales_",
        "GROSS":     "ai_gross_",
        "INVENTORY": "ai_inventory_",
        "EXPENSES":  "ai_expenses_",
    }
    ai_prefix = TYPE_AI_PREFIX.get(report_type, "ai_")
    candidates: List[Path] = []
    try:
        if AI_DIR.exists():
            for p in AI_DIR.glob("ai_*.txt"):
                try:
                    if p.stat().st_mtime < after_time:
                        continue
                    fname = p.name.lower()
                    # Фильтр по типу
                    if not fname.startswith(ai_prefix):
                        continue
                    # Фильтр по менеджеру (для типов с менеджерами)
                    if report_type in ("DEBT", "SALES", "GROSS") and manager.lower() not in fname:
                        continue
                    candidates.append(p)
                except Exception:
                    continue
        if AI_DIR.exists():
            for p in AI_DIR.glob("ai_*.html"):
                try:
                    if p.stat().st_mtime < after_time:
                        continue
                    fname = p.name.lower()
                    if not fname.startswith(ai_prefix):
                        continue
                    if report_type in ("DEBT", "SALES", "GROSS") and manager.lower() not in fname:
                        continue
                    candidates.append(p)
                except Exception:
                    continue
        if candidates:
            newest = max(candidates, key=lambda x: x.stat().st_mtime)
            age_min = round((time.time() - newest.stat().st_mtime) / 60)
            log_event("ai_cache_hit", manager=manager, file=newest.name, age_min=age_min)
            return newest
        log_event("ai_cache_miss", manager=manager)
        return None
    except Exception as e:
        log_event("ai_cache_error", manager=manager, error=str(e))
        return None

async def send_ai_file(ai_file: Path, manager: str, chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not chat_id or chat_id <= 0:
            log_event("ai_send_invalid_chat_id", manager=manager, chat_id=chat_id)
            return
        if not ai_file.exists():
            log_event("ai_file_not_found", manager=manager, file=ai_file.name)
            return
        
        # v9.4.7.5: Конвертируем TXT → HTML (для кэшированных файлов)
        if ai_file.suffix == ".txt":
            html_file = html_to_path(ai_file)
            if not html_file.exists():
                try:
                    txt_to_html(ai_file, html_file)
                    log_event("ai_html_created", file=html_file.name, manager=manager)
                except Exception as e:
                    log_event("ai_html_creation_error", error=str(e), manager=manager)
                    # Если конвертация не удалась - отправляем TXT
                    pass
            # Используем HTML если он существует
            if html_file.exists():
                ai_file = html_file
        
        today = datetime.now(TZ).strftime("%d.%m.%Y")
        file_age_min = round((time.time() - ai_file.stat().st_mtime) / 60)
        caption = f"🤖 **АНАЛИЗ ИИ**\n👤 {manager}\n📅 {today}\n⏱️ Создан: {file_age_min} мин. назад"
        with ai_file.open("rb") as f:
            sent_message = await context.bot.send_document(
                chat_id=chat_id,
                document=InputFile(f, filename=ai_file.name),
                caption=caption,
                parse_mode="Markdown",
                protect_content=True,
            )
        
        # v9.4.6: Планируем автоудаление через 24 часа
        if sent_message and sent_message.message_id:
            schedule_message_deletion(
                chat_id,
                sent_message.message_id,
                sent_message.date.timestamp(),
                AUTO_DELETE_HOURS
            )
        
        log_event("ai_file_sent", manager=manager, file=ai_file.name)
        _touch_notify_state(str(ai_file))
    except FileNotFoundError:
        log_event("ai_file_disappeared", manager=manager, file=ai_file.name)
    except Exception as e:
        log_event("ai_file_send_error", manager=manager, error=str(e))

# Блок 11_______________Обработчики команд и callback_______________________
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_role = get_user_role(chat_id)
    
    # v2.0: Трекинг пользователя
    if track_user:
        user = update.effective_user
        track_user(user.id, user.first_name, user.username)
        track_action(user.id, "start")
    
    await send_main_menu(context, chat_id, user_role, text="📋 Выберите раздел:")

async def cmd_health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not is_admin(chat_id):
        await update.effective_chat.send_message("⛔ Доступ запрещён.")
        return
    index = await _build_index()
    total_groups = sum(len(managers) for managers in index.values())
    total_files = sum(len(reports) for report_type in index.values() 
                     for reports in report_type.values())
    age_seconds = max(0.0, time.time() - _index_ts)
    
    queue_data = _load_deletion_queue()
    pending_deletions = len(queue_data.get("jobs", []))
    
    # Версию ведём в одном месте — как в лог-сообщении bot_starting
    version = __VERSION__
    
    text = (
        "🥼 System Health\n\n"
        f"🧩 Version: {version}\n"
        f"📊 Report types indexed: {len(index)}\n"
        f"👥 Total manager groups: {total_groups}\n"
        f"📁 Total files: {total_files}\n"
        f"⏰ Index age: {round(age_seconds, 1)}s\n"
        f"🕐 Last build: {datetime.fromtimestamp(_index_ts, tz=TZ).strftime('%H:%M:%S') if _index_ts else 'n/a'}\n"
        f"🗑️ Pending deletions: {pending_deletions}"
    )
    await update.effective_chat.send_message(text)

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """v2.0: Статистика использования бота"""
    chat_id = update.effective_chat.id
    
    if not is_admin(chat_id):
        await update.effective_chat.send_message("⛔ Доступ запрещён.")
        return
    
    if not get_stats:
        await update.effective_chat.send_message("⚠️ Модуль аналитики не загружен.")
        return
    
    try:
        stats = get_stats()
        message = format_stats_message(stats)
        await update.effective_chat.send_message(message, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Ошибка в cmd_stats: {e}", exc_info=True)
        await update.effective_chat.send_message(f"❌ Ошибка при получении статистики: {e}")

async def _safe_edit_text(msg, text: str, reply_markup=None, parse_mode: Optional[str] = None):
    try:
        await msg.edit_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
    except BadRequest as e:
        if "message is not modified" in str(e).lower():
            return
        raise

async def handle_report_request(
    action: str,
    manager: str,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    user_role: str,
    scopes: List[str]
) -> bool:
    if action not in SECTIONS:
        return False
    await send_with_acl(action, manager, chat_id, context)
    return True

async def handle_extended_with_ai(
    manager: str,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    user_role: str,
    scopes: List[str]
):
    await send_with_acl("DEBT_EXTENDED", manager, chat_id, context)
    await asyncio.sleep(REPORT_SEND_DELAY_SEC)
    status_msg_id = None
    try:
        status_message = await context.bot.send_message(
            chat_id=chat_id,
            text=f"🤖 **Генерируется ИИ анализ...**\n\n👤 Менеджер: {manager}\n\n⏳ Анализирую данные...",
            parse_mode="Markdown"
        )
        status_msg_id = status_message.message_id
    except Exception as e:
        log_event("ai_status_msg_fail", error=str(e), manager=manager)
    json_file = find_recent_json_for_manager(manager)
    if not json_file:
        if status_msg_id:
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=status_msg_id,
                    text=f"⚠️ **Нет данных для ИИ анализа**\n\nНе найден исходный JSON файл для {manager}",
                    parse_mode="Markdown"
                )
            except Exception:
                pass
        return
    start_time = time.time()
    await process_and_send_ai_analysis(manager, chat_id, context, json_file, start_time, status_msg_id, "DEBT")  # FIX B2: report_type was NameError

async def handle_ai_only(
    manager: str,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    user_role: str,
    scopes: List[str],
    report_type: str = "DEBT"
):
    status_msg_id = None
    try:
        status_message = await context.bot.send_message(
            chat_id=chat_id,
            text=f"🤖 **Ищу ИИ-анализ...**\n\n👤 Менеджер: {manager}",
            parse_mode="Markdown"
        )
        status_msg_id = status_message.message_id
    except Exception as e:
        log_event("ai_status_msg_fail", error=str(e), manager=manager)
    start_time = time.time()
    cache_time = start_time - (AI_CACHE_HOURS * 3600)
    existing_file = find_newest_ai_file_for_manager(manager, cache_time, report_type)  # v9.4.29: +report_type
    if existing_file:
        file_age_min = round((start_time - existing_file.stat().st_mtime) / 60)
        await send_ai_file(existing_file, manager, chat_id, context)
        if status_msg_id:
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=status_msg_id,
                    text=f"✅ **ИИ-анализ готов!**\n\n👤 Менеджер: {manager}\n\n📄 Файл отправлен выше\n\n💡 Использован существующий анализ ({file_age_min} мин. назад)",
                    parse_mode="Markdown"
                )
            except Exception:
                pass
        return
    if status_msg_id:
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=status_msg_id,
                text=f"🤖 **Генерируется новый ИИ-анализ...**\n\n👤 Менеджер: {manager}\n\n⏳ Анализирую данные...",
                parse_mode="Markdown"
            )
        except Exception:
            pass
    json_file = find_recent_json_for_manager(manager, report_type=report_type)  # v9.4.29: +report_type
    if not json_file:
        if status_msg_id:
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=status_msg_id,
                    text=f"⚠️ **Нет данных для ИИ-анализа**\n\nНе найден исходный JSON файл для {manager}",
                    parse_mode="Markdown"
                )
            except Exception:
                pass
        return
    await process_and_send_ai_analysis(manager, chat_id, context, json_file, start_time, status_msg_id, report_type)

async def cb_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""
    chat_id = q.message.chat.id
    user_role = get_user_role(chat_id)
    scopes = user_scopes(chat_id)
    my_name = get_my_manager_name(chat_id)
    log_event("callback_query", data=data, chat_id=chat_id, user_role=user_role)

    # v2.0: Трекинг действий
    if track_action:
        if data.startswith("menu_"):
            track_action(chat_id, data.replace("menu_", ""))
        elif data.startswith("direct|"):
            parts = data.split("|")
            if len(parts) >= 2:
                track_action(chat_id, parts[1].lower())

    # v2.0: Показать статистику
    if data == "show_stats":
        if user_role != "admin":
            await q.answer("⛔ Доступ запрещён")
            return
        
        if not get_stats:
            await q.answer("⚠️ Модуль аналитики не загружен")
            return
        
        try:
            stats = get_stats()
            message = format_stats_message(stats)
            await context.bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")
            await q.answer("✅ Статистика отправлена")
        except Exception as e:
            logger.error(f"Ошибка в show_stats: {e}", exc_info=True)
            await q.answer(f"❌ Ошибка: {e}")
        return

    # 🆕 v9.4.9: Аналитика
    if data.startswith("analytics|"):
        await handle_analytics(update, context, data)
        return

    # v9.4.27: Принудительная отправка отчёта
    if data.startswith("force|"):
        report_type = data.split("|", 1)[1]
        await q.answer("⏳ Формирую отчёт...")
        status = await force_report_to_user(report_type, chat_id, context)
        try:
            await q.answer(status, show_alert=False)
        except Exception:
            pass
        # Возвращаем меню той директории, из которой пришёл запрос
        await send_main_menu(context, chat_id, user_role)
        return

    if data.startswith("expenses|"):
        await handle_expenses(update, context, data)
        return
    
    if data == "menu_analytics":
        await cmd_analytics(update, context)
        return

    if data == "menu_expenses":
        await cmd_expenses(update, context)
        return
    
    if data == "back_main":
        await send_main_menu(context, chat_id, user_role, text="📋 Выберите раздел:")
        return

    if data == "analytics_menu":
        await cmd_analytics(update, context)
        return

    if data == "menu_debt":
        if user_role in ("admin", "subadmin"):
            await _safe_edit_text(q.message, "📊 **Дебиторка**\n\nВыберите тип отчёта:", 
                                reply_markup=kb_debt_menu(user_role), parse_mode="Markdown")
        else:
            await _safe_edit_text(q.message, "📊 **Дебиторка**\n\nВыберите тип отчёта:", 
                                reply_markup=kb_debt_menu_manager(my_name or "Unknown"), parse_mode="Markdown")
        return

    if data == "menu_debt_manager":
        await _safe_edit_text(q.message, "📊 **Дебиторка**\n\nВыберите тип отчёта:", 
                            reply_markup=kb_debt_menu_manager(my_name or "Unknown"), parse_mode="Markdown")
        return

    if data == "menu_sales":
        await _safe_edit_text(q.message, "🛒 **Продажи**\n\nВыберите тип отчёта:", 
                            reply_markup=kb_sales_menu(user_role))
        return

    if data == "menu_sales_manager":
        await _safe_edit_text(q.message, "🛒 **Продажи**\n\nВыберите тип отчёта:", 
                            reply_markup=kb_sales_menu_manager(my_name or "Unknown"))
        return

    if data == "menu_gross":
        await _safe_edit_text(q.message, "💰 **Валовая прибыль**\n\nВыберите тип отчёта:", 
                            reply_markup=kb_gross_menu(user_role))
        return

    if data.startswith("archive|"):
        parts = data.split("|")
        action = parts[1] if len(parts) > 1 else ""
        
        if action == "root":
            await _safe_edit_text(q.message, "🗄️ **Архив отчётов**\n\nВыберите менеджера:", 
                                reply_markup=kb_archive_managers(scopes))
            return
        
        if action == "mgr" and len(parts) > 2:
            manager = parts[2]
            if manager not in scopes:
                await q.answer("Нет доступа к этому менеджеру")
                return
            dates = _list_archive_dates_for_manager(manager)
            if not dates:
                await _safe_edit_text(q.message, f"🗄️ **Архив → {manager}**\n\n❌ Нет доступных отчетов", 
                                    reply_markup=kb_archive_managers(scopes))
                return
            await _safe_edit_text(q.message, f"🗄️ **Архив → {manager}**\n\nВыберите дату:", 
                                reply_markup=kb_archive_dates(manager, dates))
            return
        
        if action == "date" and len(parts) > 3:
            manager = parts[2]
            date = parts[3]
            if manager not in scopes:
                await q.answer("Нет доступа к этому менеджеру")
                return
            types = _types_for_date_manager(manager, date)
            if not types:
                await _safe_edit_text(q.message, f"🗄️ **Архив → {manager} → {date}**\n\n❌ Нет отчетов за эту дату", 
                                    reply_markup=kb_archive_dates(manager, _list_archive_dates_for_manager(manager)))
                return
            await _safe_edit_text(q.message, f"🗄️ **Архив → {manager} → {date}**\n\nВыберите тип отчёта:", 
                                reply_markup=kb_archive_types(manager, date, types))
            return
        
        if action == "get" and len(parts) > 4:
            manager = parts[2]
            date = parts[3]
            report_type = parts[4]
            if manager not in scopes:
                await q.answer("Нет доступа к этому менеджеру")
                return
            p = _find_report_by_date(report_type, manager, date)
            if not p or not p.exists():
                await q.answer(f"❌ Отчет не найден")
                types = _types_for_date_manager(manager, date)
                await _safe_edit_text(q.message, f"🗄️ **Архив → {manager} → {date}**\n\n❌ Отчет не найден", 
                                    reply_markup=kb_archive_types(manager, date, types))
                return
            section_rus = SECTIONS.get(report_type, report_type)
            full_text = _read_full(p)
            real_mgr = _extract_manager(full_text, p.name)
            
            file_date = _extract_date(full_text, p.name, p)
            caption = _caption(section_rus, real_mgr, file_date)
            short_name = f"{report_type}_{real_mgr.replace(' ', '_')}.html"
            try:
                # v9.4.19: Удаляем меню архива перед отправкой файла
                try:
                    await q.message.delete()
                except Exception:
                    pass
                with p.open("rb") as f:
                    sent_message = await context.bot.send_document(
                        chat_id=chat_id,
                        document=InputFile(f, filename=short_name),
                        caption=caption,
                        disable_notification=True,
                        protect_content=True,
                    )
                
                if sent_message and sent_message.message_id:
                    schedule_message_deletion(
                        chat_id,
                        sent_message.message_id,
                        sent_message.date.timestamp(),
                        AUTO_DELETE_HOURS
                    )
                
                log_event("archive_file_sent", manager=manager, date=date, type=report_type)
                
                # v9.4.19: Отправляем новое меню внизу после файла (как в DEMO)
                await asyncio.sleep(0.3)
                try:
                    await send_main_menu(context, chat_id, user_role, text="✅ *Архив: отчёт отправлен!*\n\n📋 Выберите раздел:")
                except Exception as e:
                    logger.error(f"Ошибка отправки меню после архива: {e}")
            except Exception as e:
                log_event("archive_send_error", manager=manager, date=date, type=report_type, error=str(e))
                await q.answer("❌ Ошибка при отправке файла")
            return

    if data.startswith("submenu|"):
        action = data.split("|")[1]
        section_names = {
            "DEBT_SIMPLE": "📊 Дебиторка простая",
            "DEBT_EXTENDED": "📈 Дебиторка детальная", 
            "AI_ANALYSIS": "🤖 ИИ анализ",
            "SALES_SIMPLE": "🛒 Продажи по клиентам",
            "SALES_EXTENDED": "🛒 Продажи по товару",
            "GROSS_PCT": "💰 Валовая проценты"
        }
        title = section_names.get(action, action)
        await _safe_edit_text(q.message, f"{title}\n\nВыберите менеджера:", 
                            reply_markup=kb_choose_manager_new(action, scopes))
        return

    if data.startswith("direct|"):
        _, action, manager = data.split("|", 2)
        
        if action == "GROSS_SUM" and user_role != "admin":
            await q.answer("Доступно только администратору") 
            return
        if manager == "general":
            target_manager = "Сводный отчёт"
        elif manager == "summary":
            target_manager = "Сводный отчёт"
        else:
            target_manager = manager
        if target_manager != "Сводный отчёт" and target_manager not in scopes:
            await q.answer("Нет доступа к этому менеджеру")
            return
        # v9.4.19: Удаляем старое меню перед отправкой файла (меню появится внизу как в DEMO)
        try:
            await q.message.delete()
        except Exception:
            pass
        await handle_report_request(action, target_manager, chat_id, context, user_role, scopes)
        return

    # v9.4.28: новое AI подменю ─────────────────────────────────────
    if data == "ai_type_menu":
        type_labels = {"admin": "Выберите тип анализа:", "subadmin": "Выберите тип анализа:", "manager": "Выберите тип анализа:"}
        txt = type_labels.get(user_role, "Выберите тип анализа:")
        await _safe_edit_text(q.message, f"🤖 {txt}", reply_markup=kb_ai_type_menu(user_role))
        return

    if data.startswith("ai_type|"):
        # Выбрали тип — для менеджера сразу запуск, для admin/subadmin — выбор менеджера
        report_type = data.split("|")[1]
        # v9.4.29: EXPENSES — нет менеджеров, сразу запускаем анализ
        if report_type == "EXPENSES":
            await _safe_edit_text(q.message, f"🤖 ИИ анализ затрат запущен...", reply_markup=None)
            await handle_ai_only("Общий", chat_id, context, user_role, scopes, report_type)
            return
        if user_role == "manager":
            manager_name = get_my_manager_name(chat_id) or "Unknown"
            await _safe_edit_text(q.message, f"🤖 ИИ анализ запущен...", reply_markup=None)
            await handle_ai_only(manager_name, chat_id, context, user_role, scopes, report_type)
        else:
            type_names = {"DEBT": "Дебиторка", "SALES": "Продажи", "GROSS": "Валовая",
                          "INVENTORY": "Остатки", "EXPENSES": "Затраты"}
            label = type_names.get(report_type, report_type)
            await _safe_edit_text(
                q.message,
                f"🤖 ИИ анализ — {label}\n\n👤 Выберите менеджера:",
                reply_markup=kb_choose_manager_for_ai(report_type, scopes)
            )
        return

    if data.startswith("ai_run|"):
        # Выбрали тип + менеджера
        parts = data.split("|")
        if len(parts) < 3:
            await q.answer("Ошибка данных")
            return
        report_type, manager_name = parts[1], parts[2]
        if manager_name not in scopes:
            await q.answer("Нет доступа к этому менеджеру")
            return
        await _safe_edit_text(q.message, f"🤖 ИИ анализ запущен...", reply_markup=None)
        await handle_ai_only(manager_name, chat_id, context, user_role, scopes, report_type)
        return
    # ─────────────────────────────────────────────────────────────────

    if data.startswith("ai_only|"):
        manager = data.split("|")[1]
        
        if manager not in scopes:
            await q.answer("Нет доступа к этому менеджеру")
            return
        await handle_ai_only(manager, chat_id, context, user_role, scopes)
        return

    if data.startswith("extended_ai|"):
        manager = data.split("|")[1]
        
        if manager not in scopes:
            await q.answer("Нет доступа к этому менеджеру")
            return
        await handle_extended_with_ai(manager, chat_id, context, user_role, scopes)
        return

    if data == "back_submenu":
        await send_main_menu(context, chat_id, user_role, text="📋 Выберите раздел:")
        return

    await q.answer("Неизвестная команда")

# Блок 12_______________Main (точка входа)___________________________________
# v9.4.6.1: ПАТЧ - Функция post_init для правильной регистрации
async def post_init(app: Application):
    """v9.4.27: Инициализация после старта приложения"""
    await _build_index(force=True)
    log_event("initial_index_built")

    # Восстанавливаем очередь удаления из JSON при старте
    queue_data = _load_deletion_queue()
    pending_count = len(queue_data.get("jobs", []))
    if pending_count > 0:
        logger.info(f"🧹 Восстановлено {pending_count} задач на удаление из очереди")

    # v9.4.27: Уведомление о запуске — admin (техническое) + команда (мотивирующее)
    start_kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("📋 Открыть меню", callback_data="back_main")]
    ])

    if ADMIN_CHAT_ID:
        try:
            now_str = datetime.now(TZ).strftime("%d.%m.%Y %H:%M")
            # Считаем отчётов в индексе
            report_count = sum(
                len(mgrs)
                for rtype in _index_cache.values()
                for mgrs in rtype.values()
            ) if _index_cache else 0
            
            _is_friday = datetime.now(TZ).weekday() == 4
            _oploss_line = "\n· 14:05 — упущ. прибыль" if _is_friday else ""
            admin_msg = (
                f"🟢 БОТ ЗАПУЩЕН\n"
                f"📅 {now_str} | {__VERSION__}\n"
                f"\n"
                f"📊 Отчётов в базе: {report_count}\n"
                f"\n"
                f"⏰ Расписание сегодня:\n"
                f"· 09:00 — остатки\n"
                f"· 14:00 — молчание{_oploss_line}\n"
                f"· 20:00 — валовая\n"
                f"· 21:00 — продажи + молчание\n"
                f"· 22:00 — аналитика\n"
                f"· 23:00 — сводка дня\n"
            )
            await app.bot.send_message(
                chat_id=ADMIN_CHAT_ID, text=admin_msg,
                parse_mode=None, reply_markup=start_kb
            )
            logger.info("✅ Стартовое уведомление отправлено admin")
        except Exception as e:
            logger.warning(f"⚠️ Стартовое уведомление admin: {e}")

    # Команда: менеджеры и subadmin — мотивирующее, без технических деталей
    team_msg = (
        f"🟢 Система запущена\n"
        f"📅 {datetime.now(TZ).strftime('%d.%m.%Y %H:%M')} | {__VERSION__}\n"
        f"\n"
        f"Данные актуальны. Отчёты доступны в меню.\n"
        f"Система работает. Всё под контролем."
    )

    if MANAGERS_MAP:
        for manager, chat_id in MANAGERS_MAP.items():
            if manager == "Минай" or chat_id == ADMIN_CHAT_ID:
                continue
            try:
                await app.bot.send_message(
                    chat_id=chat_id, text=team_msg,
                    parse_mode=None, reply_markup=start_kb
                )
                logger.info(f"✅ Стартовое уведомление отправлено: {manager}")
            except Exception as e:
                logger.warning(f"⚠️ Стартовое уведомление {manager}: {e}")

    # v9.4.10: Если сегодня понедельник и аналитика ещё не генерировалась — запустить сразу
    if datetime.now(TZ).weekday() == 0:
        today = datetime.now(TZ).date()
        analytics_files = list(ANALYTICS_DIR.rglob("*.html"))
        has_fresh = any(
            datetime.fromtimestamp(f.stat().st_mtime, TZ).date() == today
            for f in analytics_files
        )
        if not has_fresh:
            log_event("analytics_startup_trigger", reason="Monday, no fresh analytics found")
            await weekly_analytics_job(app)


# ═══════════════════════════════════════════════════════════════
# 🆕 v9.4.9: БЛОК АНАЛИТИКИ
# ═══════════════════════════════════════════════════════════════

async def weekly_analytics_job(context):
    """v9.4.26: Генерация аналитических отчётов + уведомление admin/subadmin + alert если нет expenses"""
    log_event("weekly_analytics_start")
    scripts = [
        "sales_profitability_report.py",
        "net_profit_report.py",
        "inventory_turnover_report.py",
        "rfm_clients_report.py",
        "revenue_concentration_report.py",
        "dso_aging_report.py",
    ]
    success_count = 0
    failed_scripts = []
    net_profit_failed = False  # v9.4.26: флаг провала net_profit_report
    try:
        for script in scripts:
            rc, stdout_s, stderr_s = await run_script_async(script, timeout=300)
            log_event("analytics_report", script=script, rc=rc)
            if rc == 0:
                success_count += 1
            else:
                failed_scripts.append(script)
                # v9.4.26: Специальная проверка net_profit_report
                if script == "net_profit_report.py":
                    net_profit_failed = True
                    combined_out = (stdout_s + stderr_s).lower()
                    if "expenses" in combined_out or "расход" in combined_out:
                        log_event("net_profit_no_expenses", level="WARNING")
        log_event("weekly_analytics_finish", success=success_count, total=len(scripts))
        bot = context.bot if hasattr(context, "bot") else None
        if bot:
            now_str = datetime.now(TZ).strftime("%d.%m.%Y %H:%M")
            msg = (
                f"📈 *Аналитика обновлена* — {now_str}\n\n"
                f"✅ Готово {success_count}/{len(scripts)} отчётов\n"
                f"Открыть: /analytics"
            )
            # Admin
            if ADMIN_CHAT_ID:
                try:
                    await bot.send_message(ADMIN_CHAT_ID, msg, parse_mode="Markdown")
                except Exception as notify_err:
                    log_event("analytics_notify_error", chat_id=ADMIN_CHAT_ID, error=str(notify_err), level="WARNING")

                # v9.4.26 Task G: alert если net_profit_report не смог рассчитать
                if net_profit_failed:
                    try:
                        # Проверяем: есть ли expenses JSON в папке?
                        expenses_files = list(JSON_DIR.glob("expenses_*.json")) if JSON_DIR.exists() else []
                        if not expenses_files:
                            alert_msg = (
                                "⚠️ *Чистая прибыль не рассчитана*\n\n"
                                "Причина: файл затрат (расходов) не загружен.\n"
                                "Загрузите файл расходов из 1С и дождитесь обработки pipeline."
                            )
                        else:
                            alert_msg = (
                                "⚠️ *Чистая прибыль не рассчитана*\n\n"
                                f"net_profit_report.py завершился с ошибкой.\n"
                                f"Проверьте logs/ для деталей."
                            )
                        await bot.send_message(ADMIN_CHAT_ID, alert_msg, parse_mode="Markdown")
                        log_event("net_profit_alert_sent", expenses_found=len(expenses_files))
                    except Exception as alert_err:
                        log_event("net_profit_alert_error", error=str(alert_err), level="WARNING")

            # Все subadmin
            for subadmin_id_str in ROLES.get("subadmin_scopes", {}):
                try:
                    subadmin_id = int(subadmin_id_str)
                    await bot.send_message(subadmin_id, msg, parse_mode="Markdown")
                except Exception as notify_err:
                    log_event("analytics_notify_error", chat_id=subadmin_id_str, error=str(notify_err), level="WARNING")
    except Exception as e:
        log_event("weekly_analytics_error", error=str(e), level="ERROR")

async def weekly_analytics_wrapper(context: ContextTypes.DEFAULT_TYPE):
    """v9.4.16: Ежедневный запуск (было: только по понедельникам)"""
    await weekly_analytics_job(context)

def _build_analytics_kb(user_role: str, chat_id: int = 0) -> Optional[InlineKeyboardMarkup]:
    """Строит клавиатуру меню аналитики в зависимости от роли пользователя."""
    force_buttons = [
        [
            InlineKeyboardButton("🔔 Дебиторка сейчас",    callback_data="force|silence"),
            InlineKeyboardButton("💸 Упущ. прибыль",       callback_data="force|oploss"),
        ],
        [
            InlineKeyboardButton("🛒 Продажи сейчас",      callback_data="force|sales"),
            InlineKeyboardButton("💰 Валовая сейчас",      callback_data="force|gross"),
        ],
        [
            InlineKeyboardButton("📦 Остатки сейчас",      callback_data="force|inventory"),
        ],
    ]
    if user_role == "admin":
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Продажи+Рентабельность", callback_data="analytics|sales_profit")],
            [InlineKeyboardButton("💰 Чистая прибыль",         callback_data="analytics|net_profit_submenu")],
            [InlineKeyboardButton("📦 Мертвый запас",          callback_data="analytics|turnover")],
            [InlineKeyboardButton("👥 RFM-клиенты",            callback_data="analytics|rfm")],
            [InlineKeyboardButton("🎯 Концентрация выручки",   callback_data="analytics|concentration")],
            [InlineKeyboardButton("💳 DSO+Aging",              callback_data="analytics|dso")],
            [InlineKeyboardButton("🤖 ИИ анализ",             callback_data="ai_type_menu")],
            [InlineKeyboardButton("🔄 Обновить аналитику",     callback_data="analytics|refresh")],
            *force_buttons,
            [InlineKeyboardButton("🔙 Главное меню",           callback_data="back_main")],
        ])
    elif user_role == "subadmin":
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("💳 DSO+Aging", callback_data="analytics|dso")],
            *force_buttons,
            [InlineKeyboardButton("🔙 Главное меню", callback_data="back_main")],
        ])
    elif user_role == "manager":
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Продажи+Рентабельность", callback_data="analytics|sales_profit")],
            [InlineKeyboardButton("👥 RFM-клиенты",            callback_data="analytics|rfm")],
            [InlineKeyboardButton("🎯 Концентрация",           callback_data="analytics|concentration")],
            *force_buttons,
            [InlineKeyboardButton("🔙 Главное меню",           callback_data="back_main")],
        ])
    return None


async def cmd_analytics(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Меню аналитики"""
    chat_id = update.effective_chat.id if update.effective_chat else update.callback_query.message.chat_id
    user_role = get_user_role(chat_id)
    kb = _build_analytics_kb(user_role, chat_id)
    if not kb:
        return

    role_text = {
        "admin":    "📈 *АНАЛИТИКА*\n\nВыберите отчёт или получите данные прямо сейчас:",
        "subadmin": "📈 *АНАЛИТИКА*\n\nОтчёты супервайзера:",
        "manager":  "📈 *АНАЛИТИКА*\n\nВаши отчёты:",
    }
    text = role_text.get(user_role, "📈 *АНАЛИТИКА*")

    if update.message:
        await update.message.reply_text(text, reply_markup=kb, parse_mode="Markdown")
    else:
        await update.callback_query.edit_message_text(text, reply_markup=kb, parse_mode="Markdown")


async def cmd_expenses(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Меню затрат (2 типа: за день / за период)"""
    chat_id = update.effective_chat.id if update.effective_chat else update.callback_query.message.chat_id
    user_role = get_user_role(chat_id)

    # Расходы показываем только admin/subadmin (управленческие данные)
    if user_role not in ("admin", "subadmin"):
        if update.callback_query:
            await update.callback_query.answer("⛔ Доступ запрещён")
        return

    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("💸 Затраты за день", callback_data="expenses|day")],
        [InlineKeyboardButton("🗓️ Затраты за период", callback_data="expenses|period")],
        [InlineKeyboardButton("🔙 Главное меню", callback_data="back_main")],
    ])
    text = "💸 *ЗАТРАТЫ*\n\nВыберите тип отчёта:"
    if update.message:
        await update.message.reply_text(text, reply_markup=kb, parse_mode="Markdown")
    else:
        await update.callback_query.edit_message_text(text, reply_markup=kb, parse_mode="Markdown")


def _pick_latest_expenses_slug(want: str) -> Optional[str]:
    """
    Выбрать slug из reports/json/expenses_*.json.

    ВНИМАНИЕ:
    - expenses_parser.py пишет report_type="EXPENSES" (стабильно),
      поэтому тип (day/period) определяем по полю period.

    Правило:
    - want='day'    -> period выглядит как ОДНА дата (нет диапазона)
    - want='period' -> period выглядит как ДИАПАЗОН (две даты через -, –, —)
    """
    def _is_range_period(period_str: str) -> bool:
        s = (period_str or "").strip()
        if not s:
            return False

        # Явный диапазон дат: 01.02.2026 - 28.02.2026 (или с другим разделителем)
        if re.search(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}\s*[-–—]\s*\d{1,2}[./]\d{1,2}[./]\d{2,4}", s):
            return True

        # На всякий случай: любой из символов диапазона, если рядом есть цифры
        if re.search(r"\d\s*[-–—]\s*\d", s):
            return True

        # Иначе считаем одиночной датой (например, '13 февраля 2026 г.')
        return False

    try:
        files = sorted(JSON_DIR.glob("expenses_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                period_str = str(data.get("period") or "")
                is_range = _is_range_period(period_str)

                if want == "day":
                    if not is_range:
                        return p.stem.replace("expenses_", "")
                else:
                    if is_range:
                        return p.stem.replace("expenses_", "")
            except Exception:
                continue
    except Exception:
        pass
    return None


async def handle_expenses(update: Update, context: ContextTypes.DEFAULT_TYPE, data: str):
    """Отправка отчёта затрат (HTML) — v9.4.12: с period в caption + детальные логи"""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    user_role = get_user_role(chat_id)

    if user_role not in ("admin", "subadmin"):
        return

    _, kind = data.split("|", 1)
    want = "day" if kind == "day" else "period"

    slug = _pick_latest_expenses_slug(want)
    if not slug:
        log_event("expenses_not_found", want=want, level="WARNING")
        await context.bot.send_message(chat_id, "❌ Отчёт затрат не найден")
        return

    # v9.4.12: Читаем period из JSON для caption
    json_path = JSON_DIR / f"expenses_{slug}.json"
    period_str = "Период неизвестен"
    try:
        if json_path.exists():
            import json
            with open(json_path, "r", encoding="utf-8") as f:
                jdata = json.load(f)
            period_str = jdata.get("period", "Период неизвестен")
    except Exception:
        pass

    # Детальное логирование (v9.4.12)
    log_event("expenses_selected", slug=slug, period=period_str, want=want, level="INFO")

    html_path = HTML_DIR / f"expenses_{slug}.html"
    if not html_path.exists():
        # fallback: если HTML не найден, пробуем отправить JSON
        log_event("expenses_html_missing", slug=slug, level="WARNING")
        if json_path.exists():
            with json_path.open("rb") as f:
                await context.bot.send_document(
                    chat_id=chat_id,
                    document=InputFile(f, filename=json_path.name),
                    caption=f"💸 Затраты (JSON)\n📅 {period_str}",
                    protect_content=True
                )
        else:
            await context.bot.send_message(chat_id, "❌ Отчёт затрат не найден")
        return

    try:
        with html_path.open("rb") as f:
            await context.bot.send_document(
                chat_id=chat_id,
                document=InputFile(f, filename=html_path.name),
                caption=f"💸 Затраты\n📅 {period_str}",
                protect_content=True
            )
        log_event("expenses_sent", slug=slug, period=period_str, chat_id=chat_id)
    except Exception as e:
        logger.error(f"Ошибка отправки expenses: {e}", exc_info=True)


async def handle_analytics(update: Update, context: ContextTypes.DEFAULT_TYPE, data: str):
    """Отправка отчёта аналитики.
    v9.4.17: net_profit → подменю (day/mtd)
    v9.4.15 Bug #9: Для multi-file типов (dso, rfm, concentration) отправляем ВСЕ файлы
    с наиболее свежей датой генерации — по одному на каждого менеджера.
    """
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    user_role = get_user_role(chat_id)

    _, report_type = data.split("|")

    # ── v9.4.17: Подменю чистой прибыли ─────────────────────────────────────
    if report_type == "net_profit_submenu":
        if user_role != "admin":
            return
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("📅 За день", callback_data="analytics|net_profit_day")],
            [InlineKeyboardButton("📆 За период", callback_data="analytics|net_profit_mtd")],
            [InlineKeyboardButton("🔙 Аналитика", callback_data="analytics_menu")],
        ])
        await query.edit_message_text("💰 *Чистая прибыль* — выберите период:", reply_markup=kb, parse_mode="Markdown")
        return

    # Контроль доступа
    if report_type in ["net_profit", "net_profit_day", "net_profit_mtd", "turnover"] and user_role != "admin":
        return
    if report_type == "dso" and user_role not in ["admin", "subadmin"]:
        return

    # v9.4.16: Кнопка "Обновить аналитику" — только admin
    if report_type == "refresh":
        if user_role != "admin":
            await context.bot.send_message(chat_id, "⛔ Только для администратора.")
            return
        await context.bot.send_message(chat_id, "⏳ Запускаю генерацию аналитики...")
        await weekly_analytics_job(context)
        return
    
    # Поиск файлов
    # v9.4.17: net_profit_day и net_profit_mtd → отдельные поддиректории
    FILE_PREFIX = {
        "sales_profit":    "sales_profitability",
        "net_profit":      "net_profit",
        "net_profit_day":  "net_profit_day",
        "net_profit_mtd":  "net_profit_mtd",
        "turnover":        "turnover",
        "rfm":             "rfm",
        "concentration":   "concentration",
        "dso":             "dso",
    }
    # v9.4.17: поддиректории для net_profit_day / net_profit_mtd
    NET_PROFIT_SUBDIRS = {
        "net_profit_day": ANALYTICS_DIR / "net_profit_day",
        "net_profit_mtd": ANALYTICS_DIR / "net_profit_mtd",
    }
    file_prefix = FILE_PREFIX.get(report_type, report_type)

    if report_type in NET_PROFIT_SUBDIRS:
        search_dir = NET_PROFIT_SUBDIRS[report_type]
    else:
        search_dir = ANALYTICS_DIR

    # Ищем и с суффиксом (rfm_20260219.html) и без (sales_profitability.html)
    all_files = sorted(
        list(search_dir.glob(f"{file_prefix}_*.html")) +
        list(search_dir.glob(f"{file_prefix}.html")),
        key=lambda p: p.stat().st_mtime, reverse=True
    )

    if not all_files:
        await context.bot.send_message(chat_id, "❌ Отчёт не найден")
        return

    # v9.4.15: Для multi-file типов (dso, rfm, concentration) берём все файлы
    # последней генерации. Определяем "свежесть" по дате: файлы с mtime в пределах
    # 10 минут от самого свежего считаются одной генерацией.
    MULTI_FILE_TYPES = {"dso", "rfm", "concentration"}
    if report_type in MULTI_FILE_TYPES:
        newest_mtime = all_files[0].stat().st_mtime
        # Берём все файлы в рамках последней генерации (10 минут = 600 сек)
        files_to_send = [f for f in all_files if newest_mtime - f.stat().st_mtime <= 600]
        # FIX B3: для subadmin фильтруем по scopes — Алена не должна видеть файлы Ергали
        if user_role == "subadmin":
            allowed_scopes = user_scopes(chat_id)
            allowed_lower = {s.lower() for s in allowed_scopes}
            all_mgrs_lower = {m.lower() for m in get_managers_list()}
            files_to_send = [
                f for f in files_to_send
                if any(s in f.name.lower() for s in allowed_lower)
                or not any(m in f.name.lower() for m in all_mgrs_lower)
            ]
        # Сортируем по имени для стабильного порядка (Алена, Ергали, Магира, Оксана)
        files_to_send = sorted(files_to_send, key=lambda p: p.name)
    else:
        files_to_send = [all_files[0]]
    
    sent_count = 0
    for file_path in files_to_send:
        try:
            with file_path.open("rb") as f:
                await context.bot.send_document(
                    chat_id=chat_id,
                    document=InputFile(f, filename=file_path.name),
                    caption=f"📊 {report_type.upper()} — {file_path.stem.split('_')[-2].title() if '_' in file_path.stem else file_path.stem}",
                    protect_content=True
                )
            sent_count += 1
        except Exception as e:
            logger.error(f"Ошибка отправки analytics {file_path.name}: {e}")
    
    log_event("analytics_sent", report_type=report_type, files_count=sent_count, chat_id=chat_id)
    # v9.4.33: Возвращаем в меню АНАЛИТИКИ (не на главное)
    try:
        await send_analytics_menu(context, chat_id, user_role)
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════





async def handle_persistent_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команд от постоянного меню (v9.4.12)"""
    text = update.message.text
    chat_id = update.effective_chat.id
    user_role = get_user_role(chat_id)
    
    if text == "📊 Дебиторка":
        kb = kb_debt_menu(user_role)
        msg = "📊 *Дебиторка*" + "\n\n" + "Выберите тип отчёта:"
        await update.message.reply_text(msg, reply_markup=kb, parse_mode="Markdown")
    
    elif text == "🛒 Продажи":
        if user_role == "manager":
            my_name = get_my_manager_name(chat_id)
            await handle_direct(update, context, f"direct|SALES_SIMPLE|{my_name}")
        else:
            kb = kb_sales_menu(user_role)
            msg = "🛒 *Продажи*" + "\n\n" + "Выберите тип отчёта:"
            await update.message.reply_text(msg, reply_markup=kb, parse_mode="Markdown")
    
    elif text == "💰 Валовая":
        kb = kb_gross_menu(user_role)
        msg = "💰 *Валовая*" + "\n\n" + "Выберите тип отчёта:"
        await update.message.reply_text(msg, reply_markup=kb, parse_mode="Markdown")
    
    elif text == "💸 Затраты":
        if user_role in ("admin", "subadmin"):
            await cmd_expenses(update, context)
        else:
            await update.message.reply_text("⛔ Доступ запрещён")
    
    elif text == "📦 Остатки":
        await handle_direct(update, context, "direct|INVENTORY_SIMPLE|general")
    
    elif text == "📈 Аналитика":
        if user_role in ("admin", "subadmin"):
            await cmd_analytics(update, context)
        else:
            await update.message.reply_text("⛔ Доступ запрещён")
    
    elif text == "🗄️ Архив":
        await handle_archive(update, context, "archive|root")


def main():
    if not BOT_TOKEN:
        logger.critical("TG_BOT_TOKEN не найден в .env! Запуск невозможен.")
        sys.exit(1)
    
    # v9.4.6.2: Версия в логе
    log_event("bot_starting", version = __VERSION__)
    
    # v9.4.6.1: ПАТЧ - Правильная регистрация post_init через builder
    application = Application.builder().token(BOT_TOKEN).post_init(post_init).build()
    
    application.add_handler(CommandHandler("start", cmd_start))
    # v9.4.12: Обработчик текстовых команд от persistent menu
    from telegram.ext import MessageHandler, filters
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_persistent_menu))
    application.add_handler(CommandHandler("health", cmd_health))
    application.add_handler(CommandHandler("stats", cmd_stats))
    application.add_handler(CommandHandler("analytics", cmd_analytics))  # 🆕 v9.4.9  # v2.0
    application.add_handler(CallbackQueryHandler(cb_data))
    job_queue = application.job_queue
    if job_queue:
        job_queue.run_repeating(pipeline_task, interval=PIPELINE_INTERVAL_MIN * 60, first=60, name="pipeline")
        job_queue.run_repeating(new_reports_notifier, interval=SCAN_INTERVAL_MIN * 60, first=180, name="new_reports")
        
        job_queue.run_daily(
            check_and_send_silence_alerts,
            time=dt_time(14, 0, tzinfo=TZ),
            name="silence_alerts_14h"
        )
        logger.info("⏰ Настроен ежедневный джоб: проверка дней молчания в 14:00")
        
        job_queue.run_daily(
            check_and_send_silence_alerts,
            time=dt_time(21, 0, tzinfo=TZ),
            name="silence_alerts_21h"
        )
        logger.info("⏰ Настроен ежедневный джоб: проверка дней молчания в 21:00")

        # v9.4.32: Упущенная прибыль — еженедельно в пятницу 14:05 (было: ежедневно 14:05 и 21:05)
        if _OPPORTUNITY_LOSS_AVAILABLE:
            job_queue.run_daily(
                send_opportunity_loss_report,
                time=dt_time(14, 5, tzinfo=TZ),
                name="opportunity_loss_weekly"
            )
            logger.info("💸 Настроен еженедельный джоб: упущенная прибыль по пятницам 14:05")
        else:
            logger.warning("⚠️ opportunity_loss не загружен — джобы 14:05/21:05 не запущены")
        
        # v9.4.6.1: Janitor каждые 60 минут (было 60 сек)
        job_queue.run_repeating(
            janitor_task,
            interval=JANITOR_INTERVAL_SEC,
            first=120,
            name="janitor"
        )
        logger.info(f"🧹 Настроен janitor: проверка очереди удаления каждые {JANITOR_INTERVAL_SEC} сек ({JANITOR_INTERVAL_SEC//60} мин)")

        # v9.4.7: Обработка очереди автогенерации ИИ
        if AI_AUTO_GENERATION:
            job_queue.run_repeating(
                process_ai_generation_queue,
                interval=AI_GENERATION_INTERVAL_SEC,
                first=300,
                name="ai_queue_processor"
            )
            logger.info(f"🤖 Настроен обработчик очереди ИИ: интервал {AI_GENERATION_INTERVAL_SEC} сек")
        
        # v9.4.7: Ежедневная сводка админу
        if ADMIN_ACTIVITY_LOG and ADMIN_CHAT_ID:
            job_queue.run_daily(
                send_daily_summary_to_admin,
                time=ADMIN_SUMMARY_TIME,
                name="daily_summary"
            )
            logger.info(f"📊 Настроена ежедневная сводка админу в {ADMIN_SUMMARY_TIME_STR}")
        
        # v9.4.7: Сброс счётчика генераций в полночь
        job_queue.run_daily(
            reset_ai_generation_state,
            time=dt_time(0, 1, tzinfo=TZ),
            name="reset_ai_state"
        )
        logger.info("🔄 Настроен сброс счётчика ИИ-генераций в 00:01")

        # ═══ v9.4.8: ЕЖЕНЕДЕЛЬНАЯ AI + КРАТКИЕ СВОДКИ ═══
        
        # Еженедельная AI генерация (понедельник через run_daily)
        job_queue.run_daily(
            weekly_ai_generation,
            time=dt_time(10, 0, tzinfo=TZ),
            name="weekly_ai_generation"
        )
        logger.info("🤖 Настроена еженедельная AI генерация: понедельник 10:00")
        
        # v9.4.16: Ежедневная аналитика в 22:00 (было: только понедельник 10:00)
        job_queue.run_daily(
            weekly_analytics_wrapper,
            time=dt_time(22, 0, tzinfo=TZ),
            name="daily_analytics"
        )
        logger.info("📊 Настроена ежедневная аналитика: каждый день 22:00")
        
        # Краткие сводки
        job_queue.run_daily(
            send_inventory_summary,
            time=dt_time(9, 0, tzinfo=TZ),
            name="inventory_summary"
        )
        logger.info("📦 Настроена краткая сводка остатков: ежедневно 09:00")
        
        job_queue.run_daily(
            send_gross_summary,
            time=dt_time(20, 0, tzinfo=TZ),
            name="gross_summary"
        )
        logger.info("💰 Настроена краткая сводка валовой: ежедневно 20:00")
        
        job_queue.run_daily(
            send_sales_summary,
            time=dt_time(21, 0, tzinfo=TZ),
            name="sales_summary"
        )
        logger.info("🛒 Настроена краткая сводка продаж: ежедневно 21:00")
        
        # v9.4.7.5: Автоочистка старых файлов в 03:00
        job_queue.run_daily(
            cleanup_old_files,
            time=dt_time(3, 0, tzinfo=TZ),
            name="cleanup_old_files"
        )
        logger.info("🧹 Настроена автоочистка файлов: логи 2д, AI 7д, HTML 30д, JSON 7д, Excel 14д | Запуск в 03:00")

        logger.info(f"🗑️ Автоудаление сообщений через {AUTO_DELETE_HOURS} часов")
    
    log_event("bot_polling_started")
    try:
        application.run_polling(drop_pending_updates=True)
    except KeyboardInterrupt:
        log_event("bot_shutdown_requested")
    except Exception as e:
        log_event("bot_critical_error", error=str(e), level="CRITICAL")
        raise

def rebuild_index_sync() -> None:
    try:
        asyncio.run(_build_index(force=True))
        print("Index rebuilt OK")
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_build_index(force=True))
        print("Index rebuilt OK (existing loop)")

if __name__ == "__main__":
    main()
# v. 9.4.41 / 2026-05-05 - fix(pipeline): Ведомость взаиморасчётов не тригерит silence_alerts (только Детальный)
# v. 9.4.40 / 2026-05-05 - feat(silence): удаление предыдущего уведомления если пришло повторно в тот же день
# v. 9.4.39 / 2026-05-05 - feat(pipeline): silence_alerts по приходу долговых файлов, не по расписанию
# v. 9.4.38 / 2026-05-05 - fix(pipeline): именные Ведомости взаиморасчётов → debt_auto_report вместо rejected
# v. 9.4.37 / 2026-04-22 - fix: approval-batch silence escalates to admin hourly; stale manager previews are closed server-side
# v. 9.4.35 / 2026-04-13 - feat: event-driven collector trigger после обработки debt_ext файлов
# v. 9.4.34 / 2026-03-16 - Fix: p.stat().st_mtime в _extract_date обёрнут в try/except (audit fix)
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
# - ИСПРАВЛЕНО: уведомление по молчунам/просрочке снова только один раз в день
#   в 14:00; вечерний дубль 21:00 удалён из расписания и из стартовой сводки.
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
import io
import os
import sys
import re
import json
import time
import ssl
import secrets
import html as _html
import asyncio
import logging
import shutil
import subprocess
from tempfile import NamedTemporaryFile
from pathlib import Path
from contextlib import contextmanager
import portalocker

# --- Path bootstrap: allow importing project-root modules when running as bot/send_reports.py ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__VERSION__ = "v9.4.78/14.05.2026"

from datetime import datetime, time as dt_time, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Any, Tuple
import certifi
import httpx
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.request import HTTPXRequest
from bot.logging_utils import (
    configure_runtime_logging,
    get_log_retention_days,
    get_runtime_logger,
    has_dead_letters,
    install_filter_on_root_handlers,
    new_trace_id,
    pop_dead_letters,
    push_dead_letter,
    set_telegram_alert_sender,
)

# ──────────────────────────────────────────────────────────────────
# Legacy reply-keyboard cleanup (v9.4.57)
# kb_persistent() (v9.4.12) удалена — функция никогда не вызывалась
# (мёртвый код). У части пользователей в клиенте Telegram остался
# "призрак" ещё более старого reply-меню с ярлыками
# "Статус / Отчёты / Последний debt/sales/gross / Архив / Меню".
# Автоочистка реализована в handle_persistent_menu().
from telegram.error import BadRequest, RetryAfter, TimedOut, NetworkError
from silence_alerts import SilenceAlert
from bot.log_monitor import format_alert as _format_log_monitor_alert
from bot.log_monitor import run_log_monitor as _run_log_monitor
from bot.crm_audit_log import audit as crm_audit
from bot.log_insights import (
    format_client_timeline,
    format_error_digest,
    read_client_timeline,
    summarize_errors_by_system,
)
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

# debt_stop_control: контроль стоп-листа отгрузки (Саида, бухгалтер)
try:
    from debt_stop_control import (
        monitor_exceptions                  as _dstop_monitor,
        send_manager_requests               as _dstop_managers,
        send_manager_reminders              as _dstop_manager_reminders,
        escalate_unanswered                 as _dstop_escalate,
        send_saida_final                    as _dstop_saida,
        handle_dstop_callback               as _dstop_callback,
        send_saida_payment_hold_reminders   as _dstop_saida_hold_reminders,
    )
    _DEBT_STOP_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ [STARTUP] debt_stop_control не найден: {e}")
    _DEBT_STOP_AVAILABLE = False
    _dstop_monitor = _dstop_managers = _dstop_manager_reminders = _dstop_escalate = _dstop_saida = _dstop_callback = _dstop_saida_hold_reminders = None

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
REJECTED_UNKNOWN_DIR = REJECTED_DIR / "unknown"  # неизвестный тип отчёта (взаиморасчёты и т.п.)
NOTIFY_STATE_PATH = LOGS_DIR / "notify_state.json"
SALES_NOTIFY_DECADE_PATH = LOGS_DIR / "sales_notify_decade.json"  # v9.4.25: подекадные уведомления
PID_FILE = LOGS_DIR / "bot.pid"
STOP_FILE = LOGS_DIR / "bot.stop"
SILENCE_SENT_PATH = LOGS_DIR / "silence_last_sent.json"  # v9.4.40: track sent silence msg_ids per manager


def _silence_load() -> dict:
    """Загружает state последних silence-сообщений: {key: {chat_id, ids, date}}."""
    try:
        if SILENCE_SENT_PATH.exists():
            import json as _j
            return _j.loads(SILENCE_SENT_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _silence_save(state: dict) -> None:
    try:
        import json as _j
        SILENCE_SENT_PATH.write_text(_j.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("_silence_save: %s", e)


# ── Защита от нескольких экземпляров (pid-файл) ───────────────────────────────
def _is_pid_running(pid: int) -> bool:
    """Проверяет, запущен ли процесс с указанным PID (Windows-safe через tasklist).

    При ошибке проверки возвращает True (fail-safe: считаем что запущен),
    чтобы не допустить двойного запуска при сбое tasklist.
    """
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        return str(pid) in result.stdout
    except (subprocess.TimeoutExpired, OSError, subprocess.SubprocessError):
        return True  # fail-safe: не знаем → считаем запущен


def _write_pid() -> None:
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")


def _clear_pid() -> None:
    try:
        if PID_FILE.exists() and PID_FILE.read_text(encoding="utf-8").strip() == str(os.getpid()):
            PID_FILE.unlink()
    except OSError:
        pass


def _check_single_instance() -> None:
    """Завершает запуск если уже работает другой экземпляр бота."""
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            old_pid = None
        if old_pid and old_pid != os.getpid() and _is_pid_running(old_pid):
            sched_logger.critical("Бот уже запущен (PID=%s). Завершение. Убейте старый процесс или удалите %s",
                            old_pid, PID_FILE)
            sys.exit(1)
        else:
            sched_logger.warning("Устаревший PID-файл (PID=%s), продолжаем.", old_pid)
    _write_pid()
    import atexit
    atexit.register(_clear_pid)
for d in [REPORTS_DIR, HTML_DIR, JSON_DIR, AI_DIR, ANALYTICS_DIR, CONFIG_DIR, LOGS_DIR, ARCHIVE_DIR, QUEUE_DIR, PROCESSED_DIR, CLEAN_DIR, REJECTED_DIR, REJECTED_CASH_DIR, REJECTED_UNKNOWN_DIR]:
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


# BUG FIX: дедупликация лога ai_daily_skipped (1 раз в день на менеджера)
_AI_DAILY_SKIPPED_LOGGED: set = set()

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
    """Конвертирует txt в html с правильной кодировкой для мобильных устройств.

    ARCH-1 (CLAUDE.md): это локальная реализация для бота. Отдельная
    реализация есть в tools/txt_to_html.py — у неё другой интерфейс
    (CLI-утилита для ручной конвертации). Унифицировать НЕЛЬЗЯ
    без переработки всех call sites — это сломает telegram-доставку
    AI-отчётов. См. CLAUDE.md → Known Open Issues → ARCH-1.
    """
    try:
        content = txt_path.read_text(encoding='utf-8')
        html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Анализ ИИ - {_html.escape(str(txt_path.stem))}</title>
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
    <pre>{_html.escape(content)}</pre>
</body>
</html>"""
        html_path.write_text(html_content, encoding='utf-8')
        # v9.4.7.5: Логирование перенесено в вызывающие функции (с правильным manager)
    except Exception as e:
        raise Exception(f"Ошибка конвертации TXT в HTML: {e}")
# Блок 3_______________Логирование (Asia/Almaty)_____________________________
configure_runtime_logging(
    logs_dir=LOGS_DIR,
    tz=TZ,
    app_name="send_reports",
    retention_days=get_log_retention_days(),
    error_alert_level=logging.ERROR,
    alert_cooldown_sec=int(os.getenv("LOG_ALERT_COOLDOWN_SEC", "300")),
)
logger = get_runtime_logger(__name__, system="BOT", component="CORE")
crm_logger = get_runtime_logger(__name__, system="CRM", component="FLOW")
sched_logger = get_runtime_logger(__name__, system="BOT", component="SCHED")
pipeline_logger = get_runtime_logger(__name__, system="PIPELINE", component="FLOW")
state_logger = get_runtime_logger(__name__, system="STATE", component="STORE")
integration_logger = get_runtime_logger(__name__, system="INTEGRATION", component="API")

# ──────────────────────────────────────────────────────────────
# Константа лимита Telegram и async-хелпер для длинных сообщений
# ──────────────────────────────────────────────────────────────
TG_MAX_MSG = 4000  # чуть меньше 4096 — запас на переносы


async def _send_auto(context, chat_id: int, text: str,
                     parse_mode=None, delay_hours: int = 24,
                     reply_markup=None) -> None:
    """Отправляет сообщение и сразу ставит его в очередь на удаление через delay_hours."""
    try:
        msg = await context.bot.send_message(chat_id=chat_id, text=text,
                                             parse_mode=parse_mode,
                                             reply_markup=reply_markup)
        schedule_message_deletion(chat_id, msg.message_id,
                                  msg.date.timestamp(), delay_hours=delay_hours)
    except Exception as e:
        integration_logger.error("_send_auto chat_id=%s: %s", chat_id, e)


async def _doc_auto(context, chat_id: int, document, caption: str = "",
                    delay_hours: int = 24) -> None:
    """Отправляет документ и сразу ставит его в очередь на удаление через delay_hours."""
    try:
        msg = await context.bot.send_document(chat_id=chat_id, document=document,
                                              caption=caption)
        schedule_message_deletion(chat_id, msg.message_id,
                                  msg.date.timestamp(), delay_hours=delay_hours)
    except Exception as e:
        integration_logger.error("_doc_auto chat_id=%s: %s", chat_id, e)


async def _tg_send_long(context, chat_id: int, text: str,
                        parse_mode=None, delay_hours: int = 24,
                        _collect_ids: "list[int] | None" = None) -> None:
    """
    Отправляет текст в Telegram, разбивая его на части <= TG_MAX_MSG символов.
    Разбивка выполняется по строкам (\\n), чтобы не рвать слова.
    Каждое сообщение ставится в очередь на автоудаление (delay_hours).
    _collect_ids: если передан список, в него добавляются message_id отправленных сообщений.
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
            if _collect_ids is not None:
                _collect_ids.append(msg.message_id)
            schedule_message_deletion(
                chat_id, msg.message_id, msg.date.timestamp(), delay_hours=delay_hours
            )
        except Exception as e:
            integration_logger.error("_tg_send_long: chunk %d/%d chat_id=%s: %s", i, len(chunks), chat_id, e)


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


async def send_notify_menu(
    context: "ContextTypes.DEFAULT_TYPE",
    chat_id: int,
    user_role: str,
    text: str = "✅ *Готово!*\n\n🔔 *Уведомления сейчас* — выберите:",
) -> None:
    """После отправки уведомления 'сейчас' возвращает в раздел уведомлений."""
    kb = kb_notify_menu(user_role)
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
        # Телефонные номера (казахстанский/российский формат)
        rules.append((r'\+?[78]\d{10}\b', "PHONE_HIDDEN"))
        rules.append((r'\b[78]\d{10}\b', "PHONE_HIDDEN"))
        # DeepSeek / OpenAI API keys
        rules.append((r'sk-[A-Za-z0-9\-_]{20,}', "API_KEY_HIDDEN"))
        # Green API token (32+ hex chars)
        rules.append((r'\b[0-9a-f]{32,}\b', "TOKEN_HIDDEN"))
        cls._MASK_RULES = [(re.compile(pat), repl) for pat, repl in rules]
        logger.info(f"🔒 SensitiveDataFilter: зарегистрировано {len(rules)} масок")

    def filter(self, record: logging.LogRecord) -> bool:
        if self._MASK_RULES:
            if record.args:
                try:
                    record.msg = str(record.msg) % record.args
                except (TypeError, ValueError):
                    record.msg = str(record.msg)
                record.args = None
            record.msg = self._apply(str(record.msg))
        return True

    @classmethod
    def _apply(cls, text: str) -> str:
        for pattern, replacement in cls._MASK_RULES:
            text = pattern.sub(replacement, text)
        return text

_sensitive_filter = SensitiveDataFilter()
install_filter_on_root_handlers(_sensitive_filter)
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
    "unknown_file_rejected": "🗑️❓",  # Неизвестный тип (взаиморасчёты и др.) → rejected/unknown
    "ai_auto_skipped_old_file": "🤖⏰❌",  # файл старше 24 часов
    "ai_auto_skipped_same_date": "🤖📅❌",  # файл с той же датой уже обработан
    # v9.4.7.5: Автоочистка старых файлов
    "cleanup_start": "🧹",
    "cleanup_finish": "✅",
    "cleanup_error": "❌",
}
def log_event(event: str, emoji: str | None = None, **kw):
    level_name = str(kw.pop("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    domain_logger = _logger_for_event(event)
    payload = {"event": event, **kw}
    if emoji is None:
        emoji = EMOJI_LOG_MAP.get(event)
    if emoji:
        try:
            flat = "; ".join(f"{k}={v}" for k, v in kw.items())
            domain_logger.log(level, f"{emoji} {event}" + (f" · {flat}" if flat else ""), extra={"event": event})
        except Exception:
            pass
    try:
        domain_logger.log(level, json.dumps(payload, ensure_ascii=False), extra={"event": event})
    except Exception:
        domain_logger.log(level, "%s %s", event, kw, extra={"event": event})


def _logger_for_event(event: str):
    e = (event or "").lower()
    if e.startswith("crm_") or e.startswith("claim_"):
        return crm_logger
    if e.startswith("collector_") or e.startswith("wa_"):
        return get_runtime_logger(__name__, system="COLLECTOR", component="FLOW")
    if (
        e.startswith("pipeline_")
        or e.startswith("imap_")
        or e.startswith("inventory_")
        or e.startswith("sales_")
        or e.startswith("gross_")
        or e.startswith("expenses_")
        or e.startswith("analytics_")
        or e.startswith("archive_")
        or e.startswith("cash_")
        or e.startswith("file_")
        or e.startswith("report_")
        or e.startswith("ai_")
        or e.startswith("net_profit_")
        or e in {"queue_empty", "queue_found_files"}
    ):
        return pipeline_logger
    if (
        e.startswith("deletion_")
        or e.endswith("_state_reset")
        or e.endswith("_state_load_error")
        or e.endswith("_state_save_error")
        or e.startswith("save_state_")
        or e.startswith("atomic_save_")
        or e.startswith("json_")
        or e == "managers_normalized"
        or e == "config_load_error"
        or e == "manager_invalid_chat_id"
    ):
        return state_logger
    if e.startswith("tg_") or e.startswith("notification_"):
        return integration_logger
    if e.endswith("_start") or e.endswith("_finish") or e.endswith("_done"):
        return sched_logger
    return logger

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

def _load_weekly_clients() -> list:
    """Загружает список еженедельных клиентов из config/weekly_clients.json."""
    try:
        data = _load_json_safe(CONFIG_DIR / "weekly_clients.json")
        return data.get("clients", []) if isinstance(data, dict) else []
    except Exception:
        return []

def _save_weekly_clients(clients: list) -> None:
    """Атомарно сохраняет список еженедельных клиентов."""
    import tempfile, os
    path = CONFIG_DIR / "weekly_clients.json"
    payload = {
        "_comment": "Список еженедельных клиентов. Управляется через бот.",
        "clients": clients,
    }
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                    delete=False, suffix=".tmp") as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)

# Токены для weekly-кнопок: token(8 hex) → client_name.
# Живут только в памяти процесса — при рестарте кнопки становятся неактивными,
# что безопасно (пользователь увидит ответ "запрос устарел").
_weekly_tokens: Dict[str, str] = {}

def _weekly_token_add(client_name: str) -> str:
    import uuid
    token = uuid.uuid4().hex[:8]
    _weekly_tokens[token] = client_name
    return token

def _weekly_token_get(token: str) -> str:
    return _weekly_tokens.get(token, "")

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
        except (ValueError, TypeError): pass
    env_admin = os.getenv("ADMIN_CHAT_ID")
    if env_admin:
        try: out.add(int(env_admin))
        except (ValueError, TypeError): pass
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
    if chat_id == _get_saida_chat_id():
        return "saida"
    for _, m_chat_id in (MANAGERS_MAP or {}).items():
        if m_chat_id == chat_id:
            return "manager"
    return "unknown"

async def _acl_gate(chat_id: int, context) -> bool:
    """Возвращает True если пользователь авторизован (admin/subadmin/manager).
    Для unknown — отправляет сообщение и возвращает False."""
    if get_user_role(chat_id) != "unknown":
        return True
    try:
        await context.bot.send_message(chat_id=chat_id, text="⛔ Доступ запрещён. Обратитесь к администратору.")
    except Exception:
        pass
    return False

# v9.4.6.1: Упрощено - удалены избыточные проверки на "Арман" (его нет в конфиге)
_SYSTEM_ACCOUNTS: set[str] = set(
    (ROLES.get("system_accounts") or [])
)

def get_managers_list() -> List[str]:
    """Возвращает список активных менеджеров (без системных аккаунтов)."""
    if MANAGERS_MAP and isinstance(MANAGERS_MAP, dict):
        return sorted([k for k in MANAGERS_MAP.keys() if k not in _SYSTEM_ACCOUNTS])
    return []

def get_my_manager_name(chat_id: int) -> Optional[str]:
    for manager, m_chat_id in (MANAGERS_MAP or {}).items():
        if m_chat_id == chat_id and manager not in _SYSTEM_ACCOUNTS:
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
        return [s for s in subordinates if s not in _SYSTEM_ACCOUNTS]
    return []

def user_scopes(chat_id: int) -> List[str]:
    role = get_user_role(chat_id)
    if role == "admin":
        return get_managers_list()
    scopes = []
    if role == "subadmin":
        subadmin_scopes = ROLES.get("subadmin_scopes", {}).get(str(chat_id), [])
        scopes.extend([s for s in subadmin_scopes if s not in _SYSTEM_ACCOUNTS])
    my_name = get_my_manager_name(chat_id)
    if my_name:
        scopes.append(my_name)
    return sorted([s for s in list(set(scopes)) if s not in _SYSTEM_ACCOUNTS])

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
# CRM — обновление базы клиентов и запрос телефонов (18:00)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def crm_daily_task(context: ContextTypes.DEFAULT_TYPE):
    """
    Ежедневно в 18:00:
    1. Обновляет CRM из последних JSON-отчётов.
    2. Каждому менеджеру — уведомление о новых клиентах (если есть).
    3. Каждому менеджеру — точечный запрос данных для ОДНОГО клиента без телефона:
       бот называет имя из 1С и просит по шагам: как обращаться → телефон → адрес.
    """
    new_trace_id()  # новый trace_id для всей CRM daily цепочки
    from bot.workday_checker import is_holiday_today
    if is_holiday_today():
        crm_logger.info("crm_daily_task: выходной — пропуск")
        return
    from bot.crm_clients import (
        update_from_reports as _crm_update,
        get_clients_without_phones as _crm_no_phone,
    )
    log_event("crm_daily_start")
    try:
        # 1. Обновляем CRM из последних JSON-отчётов
        new_by_manager = _crm_update()

        # 2. Уведомления о новых клиентах → менеджерам
        for manager, new_clients in new_by_manager.items():
            chat_id = MANAGERS_MAP.get(manager)
            if not chat_id or not new_clients:
                continue
            names_str = "\n".join(f"  · {n}" for n in new_clients[:10])
            more = f"\n  ...и ещё {len(new_clients) - 10}" if len(new_clients) > 10 else ""
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"🆕 <b>Новые клиенты в вашей базе</b>\n\n"
                        f"{names_str}{more}"
                    ),
                    parse_mode="HTML",
                )
            except Exception as _e:
                crm_logger.warning("crm_daily_task: уведомление %s: %s", manager, _e)

        # 3. Точечный запрос — первый клиент из очереди, остальные 9 идут цепочкой
        #    после каждого сохранения (один заполнил → сразу следующий).
        #    Включает admin (Вадим) — у него тоже могут быть свои клиенты.
        for manager, chat_id in _all_crm_participants().items():
            if manager in _SYSTEM_ACCOUNTS:
                continue
            no_phone = _crm_no_phone(manager, limit=1)
            if not no_phone:
                continue
            client_key = no_phone[0]
            total_no_phone = len(_crm_no_phone(manager, limit=500))
            _has_prefix = _crm_manager_from_prefix(client_key) is not None
            _now_iso = datetime.now(TZ).isoformat()
            _CRM_PHONE_PENDING[chat_id] = {
                "state": "clarify_phone" if _has_prefix else "clarify_name",
                "client_key": client_key,
                "original_name": client_key,
                "display_name": client_key[2:] if _has_prefix else "",
                "name_mode": "system" if _has_prefix else None,
                "name_review_needed": False if _has_prefix else True,
                "done_today": 0,
                "daily_limit": CRM_DAILY_LIMIT,
                "manager": manager,
                "total_no_phone": total_no_phone,
                "created_at": _now_iso,
                "last_sent": _now_iso,
            }
            if not _crm_save_pending():
                crm_logger.error("crm_state_lock_timeout: daily_task init save failed (in-memory only)")
            try:
                if _has_prefix:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=_crm_phone_prompt_text(client_key),
                        parse_mode="HTML",
                        reply_markup=_crm_phone_choice_kb(client_key) or _crm_phone_help_only_kb(),
                    )
                else:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=_crm_name_prompt_text(
                            client_key=client_key,
                            done_today=0,
                            total=total_no_phone,
                            daily_limit=CRM_DAILY_LIMIT,
                        ),
                        parse_mode="HTML",
                        reply_markup=_crm_name_choice_kb(),
                    )
            except Exception as _e:
                _CRM_PHONE_PENDING.pop(chat_id, None)
                if not _crm_save_pending():
                    crm_logger.error("crm_state_lock_timeout: daily_task cleanup save failed (in-memory only)")
                crm_logger.warning("crm_daily_task: запрос данных %s: %s", manager, _e)

        # 4. Бесхозные клиенты — рассылаем всем участникам CRM: "чей клиент?"
        #    Если у клиента есть префикс менеджера (А/Е/М/О + пробел) — сразу
        #    направляем тому менеджеру запрос на телефон, без broadcast'а.
        #    Исключаем служебные записи: "Без клиента", "Недостача", зарплатные авансы (*ЗП*/*зп*)
        _unowned = _crm_collect_unowned_claim_clients(limit=3)
        _active_claim_keys = {
            v["client_key"] for v in _CRM_CLAIM_PENDING.values() if not v.get("claimed")
        }
        _unowned = [k for k in _unowned if k not in _active_claim_keys]
        _participants = _all_crm_participants()
        for _client_key in _unowned:
            _prefix_mgr = _crm_manager_from_prefix(_client_key)
            if _prefix_mgr:
                # Префикс известен — отправить напрямую тому менеджеру
                _prefix_chat = _participants.get(_prefix_mgr)
                if _prefix_chat and _prefix_chat not in _CRM_PHONE_PENDING:
                    _unowned_now_iso = datetime.now(TZ).isoformat()
                    _CRM_PHONE_PENDING[_prefix_chat] = {
                        "state": "clarify_phone",
                        "client_key": _client_key,
                        "original_name": _client_key,
                        "display_name": _client_key[2:],
                        "name_mode": "system",
                        "name_review_needed": False,
                        "done_today": 0,
                        "daily_limit": CRM_DAILY_LIMIT,
                        "manager": _prefix_mgr,
                        "total_no_phone": 1,
                        "created_at": _unowned_now_iso,
                        "last_sent": _unowned_now_iso,
                    }
                    if not _crm_save_pending():
                        crm_logger.error("crm_state_lock_timeout: prefix-direct save failed (in-memory only)")
                    try:
                        await context.bot.send_message(
                            chat_id=_prefix_chat,
                            text=_crm_phone_prompt_text(_client_key),
                            parse_mode="HTML",
                            reply_markup=_crm_phone_choice_kb(_client_key) or _crm_phone_help_only_kb(),
                        )
                        crm_audit("prefix_autoassign", client_key=_client_key, manager=_prefix_mgr)
                        crm_logger.info("CRM prefix-autoassign: «%s» → %s", _client_key, _prefix_mgr)
                    except Exception as _pe:
                        crm_logger.warning("CRM prefix-autoassign send error %s: %s", _client_key, _pe)
                continue
            # Нет префикса — broadcast "чей клиент?" всем
            _token = _crm_claim_token()
            _notified = []
            _kb = InlineKeyboardMarkup([[
                InlineKeyboardButton("✋ Мой клиент", callback_data=f"crm_claim|{_token}")
            ]])
            for _mgr_name, _mgr_chat in _participants.items():
                try:
                    await context.bot.send_message(
                        chat_id=_mgr_chat,
                        text=(
                            f"❓ <b>Чей клиент?</b>\n\n"
                            f"<b>{_client_key}</b>\n\n"
                            f"Если ваш — нажмите кнопку."
                        ),
                        parse_mode="HTML",
                        reply_markup=_kb,
                    )
                    _notified.append(_mgr_chat)
                except Exception as _ce:
                    crm_logger.warning("crm_claim broadcast %s → %s: %s", _client_key, _mgr_name, _ce)
            _CRM_CLAIM_PENDING[_token] = {
                "client_key": _client_key,
                "notified": _notified,
                "claimed": False,
                "created_at": datetime.now(TZ).isoformat(),
            }
            if not _crm_save_claim_pending():
                crm_logger.error("crm_state_lock_timeout: claim broadcast save failed (in-memory only)")
            crm_audit("claim_broadcast", client_key=_client_key, notified_count=len(_notified), token=_token)
            crm_logger.info("CRM claim: разослано по %d участникам для «%s»", len(_notified), _client_key)

        log_event("crm_daily_done",
                  new_total=sum(len(v) for v in new_by_manager.values()))
    except Exception as e:
        log_event("crm_daily_error", error=str(e), level="ERROR")
        crm_logger.error("crm_daily_task error: %s", e)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WORKDAY CHECK — запрос выходного дня у администратора (12:00)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def check_workday_task(context: ContextTypes.DEFAULT_TYPE):
    """
    В 09:30: если xlsx от whitelist сегодня не пришли и флаг не установлен —
    спрашивает администратора: выходной или ждать?
    """
    from bot.workday_checker import needs_admin_confirmation, mark_asked_today
    if not needs_admin_confirmation():
        return
    if not ADMIN_CHAT_ID:
        return
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Да, выходной",    callback_data="workday|holiday"),
        InlineKeyboardButton("❌ Нет, рабочий день", callback_data="workday|workday"),
    ]])
    try:
        await context.bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text=(
                "📭 До 10:00 не поступило ни одного отчёта из 1С.\n\n"
                "Сегодня выходной?"
            ),
            reply_markup=kb,
        )
        mark_asked_today()
        log_event("workday_check_sent")
    except Exception as e:
        logger.warning("check_workday_task error: %s", e)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AI DEBT COLLECTOR — job-обёртки для scheduler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def debt_collector_daily(context: ContextTypes.DEFAULT_TYPE):
    """Ежедневный запуск AI-коллектора в 17:00 Asia/Almaty."""
    from bot.workday_checker import is_holiday_today
    if is_holiday_today():
        logger.info("debt_collector_daily: выходной — пропуск")
        return

    # Если WHATSAPP_ENABLED=0 — форсируем dry-run, не пытаемся --send (избегаем exit code 1)
    wa_enabled = os.getenv("WHATSAPP_ENABLED", "0").lower() in ("1", "true", "yes")
    dry_run = os.getenv("COLLECTOR_DRY_RUN", "false").lower() == "true" or not wa_enabled
    if not wa_enabled:
        logger.info("debt_collector_daily: WHATSAPP_ENABLED=0 — запуск в dry-run режиме")

    # Чистим просроченные батчи согласования перед запуском
    try:
        from collector.approval_flow import expire_old_batches
        expired = expire_old_batches()
        if expired:
            logger.info("debt_collector_daily: истёк %d батч(ей) согласования", expired)
    except Exception as _e:
        logger.debug("expire_old_batches error: %s", _e)

    log_event("collector_daily_start", dry_run=dry_run)
    mode_flag = "--dry-run" if dry_run else "--preview"
    try:
        rc, stdout, stderr = await run_script_async(
            "module:collector.collections_engine",
            mode_flag,
            timeout=900,
        )
        if rc != 0:
            log_event("collector_daily_error", rc=rc, stderr=stderr[:300], level="WARNING")
    except (OSError, ValueError) as e:
        log_event("collector_daily_error", error=str(e), level="ERROR")


async def debt_collector_promises(context: ContextTypes.DEFAULT_TYPE):
    """Ежедневная проверка просроченных обещаний оплаты в 10:00 Asia/Almaty."""
    from bot.workday_checker import is_holiday_today
    if is_holiday_today():
        logger.info("debt_collector_promises: выходной — пропуск")
        return
    log_event("collector_promises_start")
    try:
        rc, stdout, stderr = await run_script_async(
            "module:collector.collections_engine",
            "--check-promises",
            timeout=120,
        )
        if rc != 0:
            log_event("collector_promises_error", rc=rc, stderr=stderr[:300], level="WARNING")
    except (OSError, ValueError) as e:
        log_event("collector_promises_error", error=str(e), level="ERROR")


_COLLECTOR_TRIGGER_PATH = LOGS_DIR / "collector_trigger.flag"
_COLLECTOR_TRIGGER_LAST_RUN_PATH = LOGS_DIR / "collector_trigger_last_run.json"
# Не запускать повторно если коллектор уже сработал по триггеру в последние N часов
_COLLECTOR_TRIGGER_COOLDOWN_HOURS = 4
# Триггер считается устаревшим если флаг старше N часов (pipeline завис, не надо реагировать)
# 14ч — покрывает ночной разрыв: Саида разносит в 20:00, триггер подхватит до 22:00 следующего утра
_COLLECTOR_TRIGGER_MAX_AGE_HOURS = 14


async def debt_collector_trigger_check(context: ContextTypes.DEFAULT_TYPE):
    """Event-driven: запускает --preview коллектора когда появились свежие debt_ext файлы.

    run_pipeline_all_mp.py пишет logs/collector_trigger.flag после успешной
    обработки DEBT-файла. Этот job читает флаг каждые 30 минут и запускает
    коллектор сразу, не дожидаясь планового 17:00.

    Защиты:
    - только рабочие дни 09:00–18:00 Asia/Almaty
    - cooldown: не запускать повторно если уже запускали < 4 часов назад
    - флаг считается устаревшим (и удаляется без запуска) если старше 6 часов
    """
    from bot.workday_checker import is_holiday_today
    now = datetime.now(TZ)

    if is_holiday_today():
        return
    if not (9 <= now.hour < 22):
        return
    if not _COLLECTOR_TRIGGER_PATH.exists():
        return

    # Читаем флаг
    try:
        flag_data = json.loads(_COLLECTOR_TRIGGER_PATH.read_text(encoding="utf-8"))
        triggered_at_raw = flag_data.get("triggered_at", "")
        triggered_at = datetime.fromisoformat(triggered_at_raw)
        if triggered_at.tzinfo is None:
            triggered_at = triggered_at.replace(tzinfo=TZ)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning("debt_collector_trigger_check: не удалось прочитать флаг: %s", e)
        try:
            _COLLECTOR_TRIGGER_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        return

    age_hours = (now - triggered_at).total_seconds() / 3600

    # Флаг устарел — удаляем без запуска
    if age_hours > _COLLECTOR_TRIGGER_MAX_AGE_HOURS:
        logger.info(
            "debt_collector_trigger_check: флаг устарел (%.1f ч) — удаляем без запуска",
            age_hours,
        )
        try:
            _COLLECTOR_TRIGGER_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        return

    # Проверяем cooldown — не запускать если уже запускали недавно
    try:
        if _COLLECTOR_TRIGGER_LAST_RUN_PATH.exists():
            last_run_data = json.loads(_COLLECTOR_TRIGGER_LAST_RUN_PATH.read_text(encoding="utf-8"))
            last_run_at = datetime.fromisoformat(last_run_data.get("ran_at", ""))
            if last_run_at.tzinfo is None:
                last_run_at = last_run_at.replace(tzinfo=TZ)
            since_last = (now - last_run_at).total_seconds() / 3600
            if since_last < _COLLECTOR_TRIGGER_COOLDOWN_HOURS:
                logger.debug(
                    "debt_collector_trigger_check: cooldown (%.1f ч < %d ч) — пропуск",
                    since_last, _COLLECTOR_TRIGGER_COOLDOWN_HOURS,
                )
                # Флаг уже обработан (или cooldown) — удаляем
                try:
                    _COLLECTOR_TRIGGER_PATH.unlink(missing_ok=True)
                except OSError:
                    pass
                return
    except (OSError, json.JSONDecodeError, ValueError):
        pass  # нет файла или битый JSON — продолжаем

    # Удаляем флаг до запуска (атомарная операция — не даём повторному проходу сработать)
    try:
        _COLLECTOR_TRIGGER_PATH.unlink(missing_ok=True)
    except OSError as e:
        logger.warning("debt_collector_trigger_check: не удалось удалить флаг: %s", e)
        return

    # Записываем время последнего запуска
    try:
        _COLLECTOR_TRIGGER_LAST_RUN_PATH.write_text(
            json.dumps({"ran_at": now.isoformat()}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning("debt_collector_trigger_check: не удалось записать last_run: %s", e)

    debt_count = flag_data.get("debt_files_count", "?")
    logger.info(
        "debt_collector_trigger_check: запускаю --preview (новых debt файлов: %s, возраст флага: %.1f ч)",
        debt_count, age_hours,
    )
    log_event("collector_triggered_by_debt_ext", debt_files_count=debt_count)

    try:
        rc, stdout, stderr = await run_script_async(
            "module:collector.collections_engine",
            "--preview",
            timeout=600,
        )
        if rc != 0:
            logger.warning(
                "debt_collector_trigger_check: --preview завершился с rc=%d: %s",
                rc, stderr[:300],
            )
            log_event("collector_trigger_error", rc=rc, stderr=stderr[:300], level="WARNING")
    except (OSError, ValueError) as e:
        logger.error("debt_collector_trigger_check: ошибка запуска: %s", e)
        log_event("collector_trigger_error", error=str(e), level="ERROR")


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


async def log_monitor_task(context: ContextTypes.DEFAULT_TYPE):
    """Every 2 hours: scan new log lines and alert admin on fresh errors."""
    # Финализируем просроченные батчи — работает в том числе в выходные,
    # когда debt_collector_daily пропускается.
    try:
        from collector.approval_flow import expire_old_batches as _expire_batches
        _n = _expire_batches()
        if _n:
            logger.info("log_monitor_task: финализировано %d просроченных батч(ей)", _n)
    except Exception as _eb:
        logger.debug("log_monitor_task: expire_old_batches error: %s", _eb)

    state_path = LOGS_DIR / "log_monitor_state.json"
    summary_path = LOGS_DIR / "log_monitor_summary.log"
    try:
        result = _run_log_monitor(LOGS_DIR, state_path, summary_path)
        if result.get("errors_found", 0) > 0:
            state_logger.warning(
                "log_monitor: found %s new issue(s) across %s log files",
                result.get("errors_found", 0),
                result.get("files_checked", 0),
            )
            if ADMIN_CHAT_ID:
                await context.bot.send_message(
                    chat_id=ADMIN_CHAT_ID,
                    text=_format_log_monitor_alert(result),
                    parse_mode=None,
                )
        else:
            state_logger.info(
                "log_monitor: OK checked=%s initialized=%s",
                result.get("files_checked", 0),
                result.get("initialized", False),
            )
    except Exception as e:
        state_logger.error("log_monitor_task error: %s", e)


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
        json_file = find_recent_json_for_manager(manager, hours=24, report_type="DEBT")  # ← БЫЛО 48!
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
            started = queue_data.get("processing_started", 0)
            if time.time() - started < 600:
                log_event("ai_queue_busy")
                return
            log_event("ai_queue_stale_reset", stale_seconds=int(time.time() - started))
            queue_data["processing"] = False
            _save_ai_generation_queue(queue_data)
        
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
        queue_data["processing_started"] = time.time()
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
        
        json_file = find_recent_json_for_manager(manager, hours=48, report_type="DEBT")
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
        
        # Менеджеру НЕ отправляем: AI-анализ содержит "косяки" — только для руководителя
        log_event("ai_auto_skip_manager_send", manager=manager, reason="ai_for_admin_only")

        for subadmin_chat_id_str, subordinates in ROLES.get("subadmin_scopes", {}).items():
            if manager in subordinates:
                subadmin_chat_id = int(subadmin_chat_id_str)
                await send_ai_file(ai_file, manager, subadmin_chat_id, context)
                log_event("ai_auto_sent_to_subadmin", manager=manager, subadmin_chat_id=subadmin_chat_id)
        
        if ADMIN_CHAT_ID:
            await send_ai_file(ai_file, manager, ADMIN_CHAT_ID, context)
            
            _msg = await context.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text=f"✅ Автоматически сгенерирован ИИ-анализ:\n"
                     f"👤 Менеджер: {manager}\n"
                     f"📧 Chat ID: {manager_chat_id}\n"
                     f"📄 Файл: {ai_file.name}",
                parse_mode=None
            )
            schedule_message_deletion(ADMIN_CHAT_ID, _msg.message_id, _msg.date.timestamp(), delay_hours=24)
            log_event("ai_auto_sent_to_admin", manager=manager)
    except Exception as e:
        log_event("ai_auto_error", manager=manager, error=str(e))

# ═══════════════════════════════════════════════════════════════
# v9.4.8: ЕЖЕНЕДЕЛЬНАЯ AI + КРАТКИЕ СВОДКИ
# ═══════════════════════════════════════════════════════════════

async def weekly_ai_generation(context: ContextTypes.DEFAULT_TYPE):
    """
    v9.4.8: Еженедельная AI генерация (вторник 10:00)
    Генерирует для всех менеджеров, отправляет:
    - Каждому менеджеру его AI
    - Алене (subadmin) её + подшефных
    - Админу все
    """
    # Проверка: запускаем только во вторник
    if datetime.now(TZ).weekday() != 1:  # 1 = вторник
        return
    from bot.workday_checker import is_holiday_today
    if is_holiday_today():
        logger.info("weekly_ai_generation: выходной — пропуск")
        return

    log_event("weekly_ai_start")
    
    results = []
    
    for manager in get_managers_list():  # FIX B1: was managers_to_process (NameError)
        try:
            # Найти последний JSON (<7 дней)
            json_file = None
            for hours in [24, 48, 72, 168]:
                json_file = find_recent_json_for_manager(manager, hours=hours, report_type="DEBT")
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
                "--chat-id", str(ADMIN_CHAT_ID),
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
    """Отправляет AI файлы только админу и субадминам (не менеджерам).
    AI-анализ содержит "ТОП-3 КОСЯКА" и оценку работы — не для менеджеров."""
    from telegram import InputFile

    admin_chat_id = int(os.getenv("ADMIN_CHAT_ID", "0"))

    # УБРАНО: отправка менеджерам (AI-анализ с "косяками" только для руководителя)

    # 2. Субадминам — подшефные (из roles.json)
    subadmin_scopes = ROLES.get("subadmin_scopes", {})
    for sa_chat_str, scope_list in subadmin_scopes.items():
        try:
            sa_chat_id = int(sa_chat_str)
        except (ValueError, TypeError):
            continue
        if not isinstance(scope_list, list):
            continue
        sa_name = get_my_manager_name(sa_chat_id) or sa_chat_str
        try:
            msg_lines = [
                "📊 ЕЖЕНЕДЕЛЬНАЯ AI СВОДКА",
                f"Дата: {datetime.now(TZ).strftime('%d.%m.%Y')}",
                "",
                "Ваши отчеты + подшефные:"
            ]

            for r in results:
                if r['manager'] == sa_name or r['manager'] in scope_list:
                    msg_lines.append(f"  ✅ {r['manager']} ({r['elapsed']:.1f} сек)")

            await _send_auto(context, sa_chat_id, "\n".join(msg_lines))

            for r in results:
                if r['manager'] in scope_list:
                    caption = f"🤖 AI Анализ подшефного: {r['manager']}"

                    with open(r['file'], 'rb') as f:
                        await _doc_auto(context, sa_chat_id,
                            InputFile(f, filename=r['file'].name), caption=caption)

                    await asyncio.sleep(1)

            log_event("weekly_ai_sent_to_subadmin", manager=sa_name)

        except Exception as e:
            log_event("weekly_ai_subadmin_error", subadmin=sa_name, error=str(e))
    
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
            
            await _send_auto(context, admin_chat_id, "\n".join(msg_lines))

            for r in results:
                caption = f"🤖 AI Анализ: {r['manager']}"

                with open(r['file'], 'rb') as f:
                    await _doc_auto(context, admin_chat_id,
                        InputFile(f, filename=r['file'].name), caption=caption)
                
                await asyncio.sleep(1)
            
            log_event("weekly_ai_sent_to_admin", count=len(results))
            
        except Exception as e:
            log_event("weekly_ai_send_error", error=str(e))


async def send_inventory_summary(context: ContextTypes.DEFAULT_TYPE):
    """v9.4.10: Краткая сводка остатков ТОЛЬКО АДМИНУ (09:00)"""
    from bot.workday_checker import is_holiday_today
    if is_holiday_today():
        logger.info("send_inventory_summary: выходной — пропуск")
        return
    if not InventorySummary:
        logger.error("InventorySummary не импортирован")
        return

    log_event("inventory_summary_start")

    try:
        summary = InventorySummary()
        # v1.4: JSON-первый путь (HTML-fallback)
        latest_json = summary.get_latest_inventory_json(JSON_DIR)
        if latest_json:
            data = summary.parse_inventory_json(latest_json)
        else:
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
    from bot.workday_checker import is_holiday_today
    if is_holiday_today():
        logger.info("send_sales_summary: выходной — пропуск")
        return
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
    from bot.workday_checker import is_holiday_today
    if is_holiday_today():
        logger.info("send_gross_summary: выходной — пропуск")
        return
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
    from bot.workday_checker import is_holiday_today
    if is_holiday_today():
        logger.info("send_daily_summary_to_admin: выходной — пропуск")
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
            _verb = "а" if gender_emoji(user_name) == "👩" else ""
            message_parts.append(f"📥 Запросил{_verb} отчеты:")

            for report, count in sorted(requests.items()):
                report_rus = SECTIONS.get(report, report)
                message_parts.append(f"  • {report_rus} ({count} раз)" if count > 1 else f"  • {report_rus}")
                total_requests += count

            if delivered:
                message_parts.append("")
                message_parts.append(f"📤 Получил{_verb} отчеты:")
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
        
        _msg = await context.bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text=message,
            parse_mode=None
        )
        schedule_message_deletion(ADMIN_CHAT_ID, _msg.message_id, _msg.date.timestamp(), delay_hours=24)

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
    try:
        return datetime.fromtimestamp(p.stat().st_mtime, tz=TZ).strftime("%d.%m.%Y")
    except (FileNotFoundError, OSError):
        return datetime.now(tz=TZ).strftime("%d.%m.%Y")

def _is_manager_debt_extended_name(name: str, manager: Optional[str] = None) -> bool:
    """True only for per-manager detailed debt reports.

    Safe source example:
      debt_ext_Детальный Дебиторы Ергали (...).html

    Unsafe sources such as debt_ext_Ведомость... and
    debt_ext_Детальный_по_взаиморасчетам... must never be sent to managers
    as DEBT_EXTENDED.
    """
    lname = name.lower().replace("ё", "е")
    words = re.sub(r"[_\-\s]+", " ", lname).strip()
    if "debt_ext_" not in lname:
        return False
    if "детальный дебиторы" not in words:
        return False
    if manager:
        mgr = normalize_manager_name(manager).lower().replace("ё", "е")
        mgr_words = re.sub(r"[_\-\s]+", " ", mgr).strip()
        if mgr_words and mgr_words not in words:
            return False
    return True


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
    
    # Дебиторка: детальный менеджерский отчёт — только "Детальный Дебиторы <менеджер>".
    # debt_ext_Ведомость... и debt_ext_Детальный_по_взаиморасчетам... являются
    # небезопасными источниками для менеджеров: там может быть лишняя информация.
    if "debt_ext_" in lname_norm:
        if _is_manager_debt_extended_name(name):
            return "DEBT_EXTENDED"
        return "UNKNOWN"
    
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

def gender_emoji(name: str) -> str:
    """Определяет пол по окончанию имени (эвристика для русских имён).
    Суффиксы женских имён: а, я, ь (Надежда, Наталья, Любовь).
    Суффиксы мужских имён: й, ь (Ергали, Игорь) — уточняем по контексту.
    Fallback: нейтральный 👤."""
    n = name.strip()
    if not n:
        return "👤"
    last = n[-1].lower()
    if last in ("а", "я"):
        return "👩"
    if last in ("й", "и") and not n.endswith("ь"):
        return "👨"
    return "👤"

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
        _n_amb = _crmdup_ambiguous_count()
        _crm_label = f"🟡 CRM-конфликты ({_n_amb})" if _n_amb else "🟡 CRM-конфликты"
        rows = [
            [InlineKeyboardButton("📊 Дебиторка", callback_data="menu_debt")],
            [InlineKeyboardButton("📦 Остатки", callback_data="direct|INVENTORY_SIMPLE|general")],
            [InlineKeyboardButton("🛒 Продажи", callback_data="menu_sales")],
            [InlineKeyboardButton("💰 Валовая", callback_data="menu_gross")],
            [InlineKeyboardButton("💸 Затраты", callback_data="menu_expenses")],
            [InlineKeyboardButton("📈 АНАЛИТИКА", callback_data="menu_analytics")],
            [InlineKeyboardButton("🔔 Уведомления сейчас", callback_data="menu_notify")],
            [InlineKeyboardButton("🤖 Коллектор", callback_data="collector_batch")],
            [InlineKeyboardButton("📋 CRM бэклог", callback_data="crm_backlog")],
            [InlineKeyboardButton(_crm_label, callback_data="crm_ambiguous_queue")],
            [InlineKeyboardButton("🗄️ Архив", callback_data="archive|root")],
            [InlineKeyboardButton("📈 Статистика", callback_data="show_stats")],
            [InlineKeyboardButton("📖 Инструкция", callback_data="show_help_doc")],
            [InlineKeyboardButton("🛠️ Для Разработчика", callback_data="dev_feedback_open")],
        ]
    elif user_role == "subadmin":
        rows = [
            [InlineKeyboardButton("📊 Дебиторка", callback_data="menu_debt")],
            [InlineKeyboardButton("📦 Остатки", callback_data="direct|INVENTORY_SIMPLE|general")],
            [InlineKeyboardButton("🛒 Продажи", callback_data="menu_sales")],
            [InlineKeyboardButton("💰 Валовая", callback_data="submenu|GROSS_PCT")],
            [InlineKeyboardButton("📈 АНАЛИТИКА", callback_data="menu_analytics")],
            [InlineKeyboardButton("🔔 Уведомления сейчас", callback_data="menu_notify")],
            [InlineKeyboardButton("🗄️ Архив", callback_data="archive|root")],
            [InlineKeyboardButton("📖 Инструкция", callback_data="show_help_doc")],
            [InlineKeyboardButton("🛠️ Для Разработчика", callback_data="dev_feedback_open")],
        ]
    elif user_role == "manager":
        my_name = get_my_manager_name(chat_id) or "Unknown"
        rows = [
            [InlineKeyboardButton("📊 Дебиторка", callback_data="menu_debt_manager")],
            [InlineKeyboardButton("📦 Остатки", callback_data="direct|INVENTORY_SIMPLE|general")],
            [InlineKeyboardButton("🛒 Продажи", callback_data="menu_sales_manager")],
            [InlineKeyboardButton("💰 Валовая", callback_data=f"direct|GROSS_PCT|{my_name}")],
            [InlineKeyboardButton("🗄️ Архив", callback_data="archive|root")],
            [InlineKeyboardButton("📖 Инструкция", callback_data="show_help_doc")],
            [InlineKeyboardButton("🛠️ Для Разработчика", callback_data="dev_feedback_open")],
        ]
    elif user_role == "saida":
        rows = [
            [InlineKeyboardButton("📖 Инструкция", callback_data="show_help_doc")],
            [InlineKeyboardButton("🛠️ Для Разработчика", callback_data="dev_feedback_open")],
        ]
    else:
        rows = []
    return InlineKeyboardMarkup(rows)


# ── Коллектор: статус активного батча ────────────────────────────────────────

_BATCH_STATUS_RU = {
    "pending_managers":  "⏳ Ожидание менеджеров",
    "pending_admin":     "📋 Ожидание администратора",
    "admin_approved":    "✅ Утверждён администратором",
    "sent":              "📤 Отправлен",
    "partially_sent":    "📤 Отправлен частично",
    "send_failed":       "❌ Ошибка отправки",
    "send_empty":        "⚠️ Нет клиентов к отправке",
    "expired":           "⌛ Истёк",
    "cancelled":         "🚫 Отменён",
    "superseded":        "🔄 Заменён новым",
}


def _format_collector_batch_text() -> str:
    """Формирует сообщение о последнем батче коллектора для admin."""
    try:
        from collector.approval_flow import _load_batches
        batches = _load_batches()
    except Exception as exc:
        return f"⚠️ Не удалось загрузить батчи: {exc}"

    if not batches:
        return "🤖 <b>Коллектор</b>\n\nАктивных батчей нет."

    # Последний батч по created_at (а не по ключу — иначе тестовые
    # `batch-*` или произвольные ID лексикографически побеждают
    # нормальные `YYYYMMDD-HHMMSS-...` ключи).
    try:
        def _created_at_key(item):
            _id, _b = item
            ca = str((_b or {}).get("created_at") or "")
            return (ca, _id)
        latest_id, batch = max(batches.items(), key=_created_at_key)
    except Exception:
        return "⚠️ Ошибка чтения батча."

    batch_id   = batch.get("batch_id", latest_id)
    status     = batch.get("status", "—")
    status_ru  = _BATCH_STATUS_RU.get(status, status)
    created_at = str(batch.get("created_at") or "—")[:16].replace("T", " ")

    lines = [
        "🤖 <b>Коллектор — текущий батч</b>",
        "",
        f"ID: <code>{batch_id}</code>",
        f"Статус: <b>{status_ru}</b>",
        f"Создан: <b>{created_at}</b>",
    ]

    # Свежесть дебиторки
    snap = batch.get("debt_snapshot")
    if isinstance(snap, dict):
        snap_date = str(snap.get("snapshot_date") or snap.get("max_date") or "")[:10]
        age_h = snap.get("max_age_hours")
        if snap_date:
            age_str = f" ({age_h:.0f}ч)" if age_h is not None else ""
            lines.append(f"Данные дебиторки: <b>{snap_date}</b>{age_str}")

    # Менеджеры
    managers = batch.get("managers") or {}
    if managers:
        lines.append("")
        lines.append("<b>Менеджеры:</b>")
        for mgr_name, mgr_data in managers.items():
            mgr_status = mgr_data.get("status", "—")
            clients_count = len(mgr_data.get("clients") or [])
            status_icon = {
                "approved": "✅", "rejected": "❌",
                "partial": "🔸", "timeout": "⌛",
            }.get(mgr_status, "⏳")
            status_detail = mgr_status
            if mgr_status == "timeout":
                _wa = mgr_data.get("waiting_for_agreed") or {}
                _wp = mgr_data.get("waiting_for_proof") or {}
                if isinstance(_wa, dict) and _wa.get("client_name"):
                    status_icon = "⏳"
                    status_detail = f"начал — не написал детали по «{_wa['client_name']}»"
                elif isinstance(_wp, dict) and _wp.get("client_name"):
                    status_icon = "⏳"
                    status_detail = f"начал — не прислал документ по «{_wp['client_name']}»"
                else:
                    status_detail = "не ответил"
            lines.append(f"  {status_icon} {mgr_name}: {status_detail} ({clients_count} кл.)")

    # Список клиентов — одобренные или из pending
    client_list: list = []
    approved_clients = batch.get("approved_clients")
    if approved_clients:
        client_list = approved_clients
        lines.append("")
        lines.append(f"<b>Одобрено к отправке ({len(client_list)}):</b>")
    else:
        all_clients = [
            c for mgr_data in managers.values()
            for c in (mgr_data.get("clients") or [])
        ]
        if all_clients:
            client_list = all_clients
            lines.append("")
            lines.append(f"<b>Клиентов в батче ({len(client_list)}):</b>")

    for c in client_list[:20]:
        name    = c.get("name", "—")
        amount  = c.get("amount", 0)
        days    = c.get("days", 0)
        manager = c.get("manager", "")
        amount_str = f"{amount:,.0f} ₸".replace(",", " ") if amount else "—"
        lines.append(f"  • {name} — {amount_str} / {days} дн. ({manager})")
    if len(client_list) > 20:
        lines.append(f"  ... ещё {len(client_list) - 20}")

    # Send results если уже отправлено
    send_results = batch.get("send_results")
    if send_results:
        sent_ok    = sum(1 for r in send_results if r.get("sent") or r.get("status") == "sent")
        sent_total = len(send_results)
        lines.append("")
        lines.append(f"<b>Результат отправки:</b> {sent_ok}/{sent_total} доставлено")

    # Пропущенные клиенты
    skip_summary = batch.get("skip_summary") or []
    if skip_summary:
        total_skipped = len(skip_summary)
        more = "+" if total_skipped >= 20 else ""
        lines.append("")
        lines.append(f"<b>Пропущено ({total_skipped}{more}):</b>")
        for item in skip_summary[:10]:
            lines.append(f"  — {item.get('name', '?')}: {item.get('reason', '?')}")
        if total_skipped > 10:
            lines.append(f"  ... ещё {total_skipped - 10}")

    return "\n".join(lines)


def _format_crm_backlog_text(top_limit: int = 5) -> str:
    """Сводка по CRM-очередям для администратора.

    4 категории: phone-pending, claim-pending, duplicate review, ambiguous.
    На каждую — count + top N кейсов (по умолчанию 5), отдельно помечается stale.
    """
    now_dt = datetime.now(TZ)

    def _age_label(raw: Any) -> str:
        if not raw:
            return "—"
        try:
            ts = datetime.fromisoformat(str(raw))
        except (TypeError, ValueError):
            return "—"
        delta = now_dt - ts
        if delta.days >= 1:
            return f"{delta.days}д"
        hours = int(delta.total_seconds() // 3600)
        if hours >= 1:
            return f"{hours}ч"
        minutes = int(delta.total_seconds() // 60)
        return f"{minutes}м"

    lines: List[str] = ["📋 <b>CRM бэклог</b>\n"]

    # ── 1. Phone pending ──────────────────────────────────────────────────────
    phone_items = list(_CRM_PHONE_PENDING.items())
    lines.append(f"📞 <b>Phone pending: {len(phone_items)}</b>")
    if phone_items:
        phone_items.sort(key=lambda kv: str((kv[1] or {}).get("created_at", "")))
        for _chat_id, entry in phone_items[:top_limit]:
            manager = entry.get("manager", "?")
            client  = entry.get("client_key", "?")
            state   = entry.get("state", "?")
            age     = _age_label(entry.get("created_at"))
            lines.append(f"  • {manager} → {client} ({state}, {age})")
        if len(phone_items) > top_limit:
            lines.append(f"  ... ещё {len(phone_items) - top_limit}")
    lines.append("")

    # ── 2. Claim pending ──────────────────────────────────────────────────────
    open_claims = [
        (tok, c) for tok, c in _CRM_CLAIM_PENDING.items()
        if not c.get("claimed") and not _crm_claim_is_stale(c, now_dt)
    ]
    stale_claims = sum(1 for c in _CRM_CLAIM_PENDING.values() if _crm_claim_is_stale(c, now_dt))
    suffix = f" (+{stale_claims} stale)" if stale_claims else ""
    lines.append(f"✋ <b>Claim pending: {len(open_claims)}</b>{suffix}")
    if open_claims:
        open_claims.sort(key=lambda kv: str((kv[1] or {}).get("created_at", "")))
        for _tok, claim in open_claims[:top_limit]:
            client = claim.get("client_key", "?")
            age    = _age_label(claim.get("created_at"))
            n_notified = len(claim.get("notified") or [])
            lines.append(f"  • {client} (разослан {n_notified} мгр, {age})")
        if len(open_claims) > top_limit:
            lines.append(f"  ... ещё {len(open_claims) - top_limit}")
    lines.append("")

    # ── 3. Duplicate review ───────────────────────────────────────────────────
    dup_items = [(tok, r) for tok, r in _CRM_DUP_REVIEW_PENDING.items() if not r.get("resolved_at")]
    lines.append(f"🔀 <b>Duplicate review: {len(dup_items)}</b>")
    if dup_items:
        dup_items.sort(key=lambda kv: str((kv[1] or {}).get("created_at", "")))
        for _tok, review in dup_items[:top_limit]:
            manager = review.get("manager", "?")
            items   = review.get("items", []) or []
            pair    = " ↔ ".join((it.get("client_key", "?") or "?") for it in items[:2])
            age     = _age_label(review.get("created_at"))
            lines.append(f"  • {manager}: {pair} ({age})")
        if len(dup_items) > top_limit:
            lines.append(f"  ... ещё {len(dup_items) - top_limit}")
    lines.append("")

    # ── 4. Ambiguous conflicts ────────────────────────────────────────────────
    amb_pending = [(sig, v) for sig, v in _CRM_AMBIGUOUS.items() if v.get("status") == "pending"]
    amb_resolved = sum(1 for v in _CRM_AMBIGUOUS.values() if v.get("status") == "resolved")
    suffix = f" (+{amb_resolved} resolved в истории)" if amb_resolved else ""
    lines.append(f"❓ <b>Ambiguous: {len(amb_pending)}</b>{suffix}")
    if amb_pending:
        amb_pending.sort(key=lambda kv: str((kv[1] or {}).get("added_at", "")))
        for _sig, entry in amb_pending[:top_limit]:
            group   = entry.get("group_key", "?")
            mgrs    = entry.get("managers", []) or []
            age     = _age_label(entry.get("added_at"))
            lines.append(f"  • {group} (менеджеры: {', '.join(mgrs) or '—'}, {age})")
        if len(amb_pending) > top_limit:
            lines.append(f"  ... ещё {len(amb_pending) - top_limit}")

    return "\n".join(lines)


def _crm_backlog_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Обновить", callback_data="crm_backlog")],
        [InlineKeyboardButton("🟡 Ambiguous-очередь", callback_data="crm_ambiguous_queue")],
        [InlineKeyboardButton("🔙 Главное меню", callback_data="back_main")],
    ])


def _get_pending_admin_batch() -> Optional[Dict[str, Any]]:
    """Возвращает батч в статусе pending_admin (ждёт утверждения администратора)."""
    try:
        from collector.approval_flow import _load_batches
        from datetime import datetime as _dt
        from zoneinfo import ZoneInfo as _ZI
        _now = _dt.now(tz=_ZI("Asia/Almaty"))
        _all_batches = _load_batches()
        for bid in sorted(_all_batches.keys(), reverse=True):
            b = _all_batches.get(bid) or {}
            if b.get("status") != "pending_admin":
                continue
            from collector.approval_flow import _parse_batch_dt
            exp = _parse_batch_dt(b.get("expires_at"))
            if exp and _now >= exp:
                continue
            return b
    except Exception:
        pass
    return None


def _collector_batch_keyboard() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🔄 Обновить", callback_data="collector_batch")],
        [InlineKeyboardButton("🧭 Актуальный батч", callback_data="collector_actual_batch")],
        [InlineKeyboardButton("🤝 Обещания менеджеров", callback_data="collector_agreed_stats")],
        [InlineKeyboardButton("📋 Саида backlog", callback_data="collector_saida_stats")],
        [InlineKeyboardButton("🔸 Частичные оплаты", callback_data="collector_partial_stats")],
        [InlineKeyboardButton("⏱ Отсрочки", callback_data="collector_deferral_stats")],
    ]
    # Батч ждёт утверждения Администратора
    pending = _get_pending_admin_batch()
    if pending:
        rows.append([InlineKeyboardButton("📋 Утвердить рассылку", callback_data="collector_resend_approval")])
    try:
        from collector.approval_flow import get_latest_send_ready_batch
        send_ready = get_latest_send_ready_batch()
    except Exception:
        send_ready = None
    if send_ready:
        rows.append([InlineKeyboardButton("📤 Готовый список", callback_data="collector_send_latest")])
    rows.append([InlineKeyboardButton("🔙 Главное меню", callback_data="back_main")])
    return InlineKeyboardMarkup(rows)


def _get_actual_collector_batch_mode() -> str:
    """Returns the most actionable collector batch view for admin UI."""
    pending = _get_pending_admin_batch()
    if pending:
        return "pending_admin"
    try:
        from collector.approval_flow import get_latest_send_ready_batch
        if get_latest_send_ready_batch():
            return "send_ready"
    except Exception:
        pass
    return "status"


def _format_collector_agreed_stats_text() -> str:
    """Формирует read-only сводку качества обещаний менеджеров."""
    try:
        from collector.approval_flow import format_agreed_promise_stats_text
        return format_agreed_promise_stats_text()
    except Exception as exc:
        return f"⚠️ Не удалось загрузить статистику обещаний: {exc}"


def _format_collector_saida_stats_text() -> str:
    """Формирует read-only сводку backlog Саиды."""
    try:
        from collector.payment_hold import format_saida_hold_stats_text
        return format_saida_hold_stats_text()
    except Exception as exc:
        return f"⚠️ Не удалось загрузить backlog Саиды: {exc}"


def _format_collector_partial_stats_text() -> str:
    """Формирует read-only сводку частичных оплат."""
    try:
        from collector.payment_hold import format_partial_payment_stats_text
        return format_partial_payment_stats_text()
    except Exception as exc:
        return f"⚠️ Не удалось загрузить статистику частичных оплат: {exc}"


def _format_collector_deferral_stats_text() -> str:
    """Формирует read-only сводку финдисциплины по отсрочкам."""
    try:
        from collector.payment_deferrals import format_deferral_discipline_stats_text
        return format_deferral_discipline_stats_text()
    except Exception as exc:
        return f"⚠️ Не удалось загрузить статистику по отсрочкам: {exc}"


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


def kb_notify_menu(user_role: str) -> InlineKeyboardMarkup:
    """Раздел 'Уведомления сейчас' — принудительная отправка кратких сводок."""
    rows: list = []
    if user_role in ("admin", "subadmin"):
        rows += [
            [InlineKeyboardButton("🔔 Дебиторка сейчас",  callback_data="force|silence"),
             InlineKeyboardButton("💸 Упущ. прибыль",     callback_data="force|oploss")],
        ]
    rows += [
        [InlineKeyboardButton("🛒 Продажи сейчас",    callback_data="force|sales"),
         InlineKeyboardButton("💰 Валовая сейчас",    callback_data="force|gross")],
        [InlineKeyboardButton("📦 Остатки сейчас",    callback_data="force|inventory")],
    ]
    if user_role == "admin":
        rows += [
            [InlineKeyboardButton("💰 Чистая прибыль сейчас", callback_data="force|net_profit")],
            [InlineKeyboardButton("📊 Рейтинг менеджеров",    callback_data="force|ranking")],
        ]
    rows.append([InlineKeyboardButton("⬅️ Главное меню", callback_data="back_main")])
    return InlineKeyboardMarkup(rows)

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

def _debt_simple_is_live_fresh(simple_path: Path, manager_name: str) -> bool:
    if not simple_path or not simple_path.exists():
        return False
    detailed_path = find_report("DEBT_EXTENDED", manager_name)
    if not detailed_path or not detailed_path.exists():
        return True
    simple_text = _read_full(simple_path)
    detailed_text = _read_full(detailed_path)
    simple_period = _parse_period_to_date(_extract_date(simple_text, simple_path.name, simple_path))
    detailed_period = _parse_period_to_date(_extract_date(detailed_text, detailed_path.name, detailed_path))
    min_dt = datetime.min.replace(tzinfo=TZ)
    if simple_period == min_dt or detailed_period == min_dt:
        return True
    if detailed_period > simple_period:
        log_event(
            "stale_simple_debt_blocked",
            manager=normalize_manager_name(manager_name),
            simple_file=simple_path.name,
            simple_period=simple_period.strftime("%Y-%m-%d"),
            detailed_file=detailed_path.name,
            detailed_period=detailed_period.strftime("%Y-%m-%d"),
            level="WARNING",
        )
        return False
    return True


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
        await _send_auto(context, chat_id, "⛔ Сводные отчёты по этому разделу доступны только администратору.")
        log_user_delivery(chat_id, section, False)
        return
    if intended_mgr != "Сводный отчёт" and intended_mgr not in scopes:
        await _send_auto(context, chat_id, "⛔ Нет доступа к отчётам этого менеджера.")
        log_user_delivery(chat_id, section, False)
        return
    p = find_report(section, intended_mgr if intended_mgr != "Сводный отчёт" else None)
    # v9.4.15 Bug #11: Для SALES fallback на Сводный отчёт —
    # sales_report.py генерирует один общий файл без имени менеджера в названии,
    # поэтому per-manager файлы в индексе отсутствуют. Сводный содержит всех клиентов.
    if (
        p and p.exists()
        and section == "DEBT_SIMPLE"
        and intended_mgr != "Сводный отчёт"
        and not _debt_simple_is_live_fresh(p, intended_mgr)
    ):
        p = None
    sales_summary_fallback = False
    if (not p or not p.exists()) and section in ("SALES_SIMPLE", "SALES_EXTENDED") and intended_mgr != "Сводный отчёт":
        if user_role == "admin":
            p = find_report(section, None)
            if p and p.exists():
                sales_summary_fallback = True
                log_event("sales_summary_fallback", section=section, intended_mgr=intended_mgr,
                          file=p.name, level="INFO")
    if (not p or not p.exists()) and section == "SALES_EXTENDED":
        _fb_mgr = intended_mgr if intended_mgr != "Сводный отчёт" else None
        p = find_report("SALES_SIMPLE", _fb_mgr)
        if not p or not p.exists():
            if user_role == "admin":
                p = find_report("SALES_SIMPLE", None)
        if p and p.exists():
            sales_summary_fallback = True
            log_event("sales_extended_fallback_to_simple", intended_mgr=intended_mgr,
                      file=p.name, level="INFO")
    if not p or not p.exists():
        log_event("report_not_found", section=section, manager=intended_mgr)
        await _send_auto(context, chat_id, f"❌ Отчёт не найден: {section_rus} для '{intended_mgr}'.")
        return

    if user_role != "admin" and section == "DEBT_EXTENDED":
        if intended_mgr == "Сводный отчёт" or not _is_manager_debt_extended_name(p.name, intended_mgr):
            log_event(
                "debt_extended_source_blocked",
                intended_mgr=intended_mgr,
                file=p.name,
                level="ERROR",
            )
            await _send_auto(
                context,
                chat_id,
                "⛔ Детальная дебиторка не отправлена: найден не именной отчёт по дебиторке. "
                "Сообщите администратору.",
            )
            log_user_delivery(chat_id, section, False)
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
                await _send_auto(context, chat_id, "⛔ Ошибка безопасности: найден отчёт другого типа.")
                log_user_delivery(chat_id, section, False)
                return
            if not is_intended_summary and normalize_manager_name(intended_mgr) != normalize_manager_name(real_mgr):
                await _send_auto(context, chat_id, "⛔ Ошибка безопасности: найден отчёт для другого менеджера.")
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
                await _send_auto(context, chat_id, "⏳ Telegram перегружен, попробуйте позже.")
                return
        except BadRequest as e:
            error_msg = str(e).lower()
            if "file is too big" in error_msg or "too large" in error_msg:
                log_event("tg_file_too_big", section=section, file=p.name, size_mb=round(p.stat().st_size/1024/1024, 2))
                await _send_auto(context, chat_id,
                    f"⚠️ Файл слишком большой для отправки.\n📄 Имя: {p.name}\n📏 Размер: {round(p.stat().st_size/1024/1024, 1)} МБ")
                return
            else:
                log_event("tg_send_error", section=section, error=str(e))
                await _send_auto(context, chat_id, f"❌ Ошибка Telegram: {str(e)[:100]}")
                return
        except Exception as e:
            log_event("tg_send_error", section=section, manager=real_mgr, error=str(e))
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
            else:
                await _send_auto(context, chat_id, "❌ Ошибка при отправке файла.")
                return

# Блок 8_______________Фоновые задачи (pipeline + скрипты)__________________
async def run_script_async(script_name: str, *args: str, timeout: int = 600) -> Tuple[int, str, str]:
    if script_name.startswith("module:"):
        module_name = script_name.split(":", 1)[1].strip()
        if not module_name:
            log_event("script_not_found", script=script_name, level="ERROR")
            return -1, "", f"Module name not provided: {script_name}"
        command = [sys.executable, "-m", module_name, *args]
    else:
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
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                process.kill()
                await process.wait()
            except (ProcessLookupError, PermissionError, OSError):
                pass
            log_event("script_timeout", script=script_name, timeout=timeout, level="WARNING")
            return -1, "", f"Timeout after {timeout}s"
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


def _mark_notified_today(manager_name: str, period_str: str) -> None:
    """Записывает факт успешной отправки декадного уведомления менеджеру."""
    import re as _re
    from datetime import date as _date
    try:
        _d = None
        all_m = list(_re.finditer(r"(\d{1,2})[\./ ](\d{1,2})[\./ ](\d{4})", period_str))
        if all_m:
            lm = all_m[-1]
            try:
                _d = _date(int(lm.group(3)), int(lm.group(2)), int(lm.group(1)))
            except ValueError:
                pass
        else:
            RU = {"января":1,"февраля":2,"марта":3,"апреля":4,"мая":5,"июня":6,
                  "июля":7,"августа":8,"сентября":9,"октября":10,"ноября":11,"декабря":12}
            m2 = _re.search(r"(\d{1,2})\s+([а-яё]+)\s+(\d{4})", period_str.lower())
            if m2 and m2.group(2) in RU:
                try:
                    _d = _date(int(m2.group(3)), RU[m2.group(2)], int(m2.group(1)))
                except ValueError:
                    pass
        if _d is None:
            return
        decade = (_d.day - 1) // 10
        decade_key = f"{_d.year}-{_d.month:02d}-d{decade}"
        state = {}
        if SALES_NOTIFY_DECADE_PATH.exists():
            try:
                state = json.loads(SALES_NOTIFY_DECADE_PATH.read_text(encoding="utf-8"))
            except Exception:
                state = {}
        state[manager_name.lower()] = decade_key
        SALES_NOTIFY_DECADE_PATH.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as _e:
        logger.warning(f"_mark_notified_today: {_e}")


def _should_notify_manager_today(manager_name: str, period_str: str) -> bool:
    """
    v9.4.25: Подекадные уведомления менеджерам.
    Возвращает True только если в эту декаду ещё не отправляли.
    Состояние записывается отдельно через _mark_notified_today() после успешной отправки.
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

        return True  # Ещё не отправляли

    except Exception as _e:
        logger.warning(f"_should_notify_manager_today: {_e}")
        return True  # При любой ошибке — шлём



async def _validate_daily_reports_saida(context) -> None:
    """21:00 — проверить все ожидаемые ежедневные отчёты и напомнить Саиде о пропущенных.

    Ожидаемые отчёты:
      DAY (сегодня): Валовая прибыль, Затраты, Продажи × 4 менеджера, Дебиторка × 4 менеджера
      MTD (вчера):   Валовая прибыль нарастающим, Затраты нарастающим
    """
    from bot.workday_checker import is_holiday_today as _is_holiday
    if _is_holiday():
        return

    import json as _json
    from datetime import date as _date, timedelta as _td

    _today: _date  = datetime.now(TZ).date()
    _yesterday: _date = _today - _td(days=1)
    _month_start: _date = _today.replace(day=1)
    _today_str    = _today.strftime("%d.%m.%Y")
    _yest_str     = _yesterday.strftime("%d.%m.%Y")
    _mstart_str   = _month_start.strftime("%d.%m.%Y")

    missing: list[str] = []

    try:
        import net_profit_report as _np

        all_gross = _np.load_summary_gross_jsons()
        all_exp   = _np.load_all_jsons("expenses_*.json")

        def _has_day(items, target: _date) -> bool:
            for item in items:
                s, e = _np.extract_period_dates(_np.extract_period_from_json(item))
                if s and e and s == e == target:
                    return True
            return False

        def _has_mtd(items, month_start: _date, end: _date) -> bool:
            for item in items:
                s, e = _np.extract_period_dates(_np.extract_period_from_json(item))
                if s and e and s == month_start and e == end and s != e:
                    return True
            return False

        if not _has_day(all_gross, _today):
            missing.append(f"❌ Валовая прибыль за день ({_today_str})")
        if not _has_day(all_exp, _today):
            missing.append(f"❌ Затраты за день ({_today_str})")
        if not _has_mtd(all_gross, _month_start, _yesterday):
            missing.append(f"❌ Валовая прибыль нарастающим ({_mstart_str}–{_yest_str})")
        if not _has_mtd(all_exp, _month_start, _yesterday):
            missing.append(f"❌ Затраты нарастающим ({_mstart_str}–{_yest_str})")

    except Exception as _e:
        logger.debug("validate_reports gross/exp error: %s", _e)

    # Продажи и Дебиторка — по менеджерам
    try:
        _mgrs = list((MANAGERS_MAP or {}).keys())
        if not _mgrs:
            import json as _jj
            _mgrs = list(_jj.loads((CONFIG_DIR / "managers.json").read_text(encoding="utf-8")).keys())

        # Продажи: sales_*.json → period_end == today ISO
        _today_iso = _today.isoformat()
        _sales_today: set[str] = set()
        for _sf in JSON_DIR.glob("sales_*.json"):
            try:
                _sd = _json.loads(_sf.read_text(encoding="utf-8"))
                if str(_sd.get("period_end", "")).startswith(_today_iso):
                    _period_str = str(_sd.get("period", ""))
                    _fname = _sf.stem.lower()
                    for _m in _mgrs:
                        if _m.lower() in _fname or _m.lower() in _period_str.lower():
                            _sales_today.add(_m)
            except Exception:
                pass

        for _m in _mgrs:
            if _m not in _sales_today:
                missing.append(f"❌ Продажи — {_m} ({_today_str})")

        # Дебиторка: debt_ext_*.json → period_max == today DD.MM.YYYY
        _debt_today: set[str] = set()
        for _df in JSON_DIR.glob("debt_ext_*.json"):
            try:
                _dd = _json.loads(_df.read_text(encoding="utf-8"))
                _pm = str(_dd.get("period_max", ""))
                if _pm == _today_str:
                    _mgr_in_file = str(_dd.get("manager", ""))
                    _fname_low = _df.stem.lower()
                    for _m in _mgrs:
                        if _m.lower() in _mgr_in_file.lower() or _m.lower() in _fname_low:
                            _debt_today.add(_m)
            except Exception:
                pass

        for _m in _mgrs:
            if _m not in _debt_today:
                missing.append(f"❌ Дебиторка — {_m} ({_today_str})")

    except Exception as _e:
        logger.debug("validate_reports sales/debt error: %s", _e)

    if not missing:
        log_event("daily_reports_validation_ok", date=_today_str)
        return

    _bullets = "\n".join(missing)
    text = (
        f"⚠️ Проверка отчётов за {_today_str}\n\n"
        f"Не хватает:\n{_bullets}\n\n"
        f"Пришли, пожалуйста."
    )
    _saida_cid = int(os.getenv("SAIDA_CHAT_ID", "920236287"))
    try:
        await context.bot.send_message(chat_id=_saida_cid, text=text)
        log_event("daily_reports_validation_sent", date=_today_str, missing_count=len(missing))
    except Exception as _e:
        logger.error("validate_daily_reports send error: %s", _e)


async def _remind_saida_mtd_if_missing(context) -> None:
    """После pipeline: если пришёл DAY gross/expenses, но MTD-пара не полная — напомнить Саиде.
    Срабатывает не чаще 1 раза в день (logs/saida_mtd_reminder.json).
    """
    import json as _json
    _remind_file = LOGS_DIR / "saida_mtd_reminder.json"
    _today = datetime.now(TZ).strftime("%Y-%m-%d")

    try:
        _rstate = _json.loads(_remind_file.read_text(encoding="utf-8")) if _remind_file.exists() else {}
    except Exception:
        _rstate = {}
    if _rstate.get("date") == _today:
        return  # уже напомнили сегодня

    try:
        import net_profit_report as _np
        from datetime import date as _date

        all_gross = _np.load_summary_gross_jsons()
        all_exp   = _np.load_all_jsons("expenses_*.json")

        # Ищем последний DAY gross
        latest_day: Optional[Any] = None
        for _g in all_gross:
            _gs, _ge = _np.extract_period_dates(_np.extract_period_from_json(_g))
            if _gs and _ge and _gs == _ge:
                latest_day = _gs
                break

        if not latest_day:
            return

        _month_start = latest_day.replace(day=1)
        _day_str     = latest_day.strftime("%d.%m.%Y")
        _start_str   = _month_start.strftime("%d.%m.%Y")

        def _has_mtd(items):
            for _item in items:
                _s, _e = _np.extract_period_dates(_np.extract_period_from_json(_item))
                if _s and _e and _s == _month_start and _e >= latest_day and _s != _e:
                    return True
            return False

        missing = []
        if not _has_mtd(all_gross):
            missing.append("Валовая прибыль (нарастающим)")
        if not _has_mtd(all_exp):
            missing.append("Затраты (нарастающим)")

        if not missing:
            return

        _bullets = "\n".join(f"• {m}" for m in missing)
        text = (
            f"📊 Получены дневные отчёты за {_day_str}.\n\n"
            f"Пришли, пожалуйста, ещё нарастающим за период {_start_str}–{_day_str}:\n"
            f"{_bullets}\n\n"
            f"Без них отчёт «Чистая прибыль за период» не обновится."
        )

        _saida_cid = int(os.getenv("SAIDA_CHAT_ID", "920236287"))
        await context.bot.send_message(chat_id=_saida_cid, text=text)

        _remind_file.write_text(
            _json.dumps({"date": _today, "day": _day_str, "missing": missing}, ensure_ascii=False),
            encoding="utf-8",
        )
        log_event("saida_mtd_reminder_sent", day=_day_str, missing=missing)

    except Exception as _e:
        logger.debug("_remind_saida_mtd_if_missing inner error: %s", _e)


async def pipeline_task(context: ContextTypes.DEFAULT_TYPE):
    log_event("pipeline_cycle_start")
    _imap_rc, _imap_out, _imap_err = await run_script_async("imap_fetcher.py", "--once")
    # v9.4.25: Уведомляем admin если почта не ответила (пропуск в выходной)
    from bot.workday_checker import is_holiday_today as _imap_is_holiday
    if _imap_rc != 0 and ADMIN_CHAT_ID and not _imap_is_holiday():
        try:
            _err_preview = (_imap_err or _imap_out or "")[:200].strip()
            _msg = await context.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text=(
                    f"⚠️ imap_fetcher завершился с ошибкой ({datetime.now(TZ).strftime('%H:%M')})\n"
                    f"Файлы могли быть скачаны частично или полностью.\n"
                    f"Проверь mailbox минбаракат.\n"
                    + (f"\nОшибка: {_err_preview}" if _err_preview else "")
                )
            )
            schedule_message_deletion(ADMIN_CHAT_ID, _msg.message_id, _msg.date.timestamp(), delay_hours=24)
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
    # v9.4.39: счётчик долговых файлов — silence_alerts запускается только по ним
    debt_files_processed = 0
    
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
                _this_file_is_debt = False  # v9.4.39

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
                elif "взаиморасч" in fname_lower:
                    # Именная Ведомость (Ергали/Алена/Магира/Оксана в имени) → debt_auto_report
                    # Сводная (без имени менеджера) → rejected/unknown
                    # v9.4.41: _this_file_is_debt НЕ устанавливаем — silence_alerts
                    #   использует только Детальный Дебиторы, Ведомость не тригерит silence.
                    _known = set(m.lower() for m in get_managers_list())
                    _is_named = any(m in fname_lower for m in _known)
                    if _is_named:
                        script_rc, _, _ = await run_script_async("debt_auto_report.py", str(file_path))
                        script_executed = True
                        # _this_file_is_debt остаётся False: не тригерит silence_alerts
                    else:
                        timestamp = datetime.now(TZ).strftime('%Y%m%d_%H%M%S')
                        rejected_path = REJECTED_UNKNOWN_DIR / f"{timestamp}_{file_path.name}"
                        shutil.move(file_path, rejected_path)
                        log_event("unknown_file_rejected", original=file_path.name,
                                  moved_to=rejected_path.name, reason="взаиморасчёты-сводный")
                        continue
                else:
                    script_rc, _, _ = await run_script_async("debt_auto_report.py", str(file_path))
                    script_executed = True
                    _this_file_is_debt = True  # v9.4.39

                # v9.4.8: AI генерация отключена (теперь еженедельная)
                if script_executed and script_rc == 0:
                    pass  # schedule_ai_generation отключена
                    # BUG FIX: логируем только 1 раз в день на менеджера (не при каждом файле)
                    try:
                        fname = file_path.name.lower()
                        today_key = datetime.now(TZ).strftime("%Y-%m-%d")
                        for manager in get_managers_list():
                            if manager.lower() in fname:
                                skip_key = f"{today_key}:{manager}"
                                if skip_key not in _AI_DAILY_SKIPPED_LOGGED:
                                    _AI_DAILY_SKIPPED_LOGGED.add(skip_key)
                                    log_event("ai_daily_skipped", manager=manager, reason="Weekly AI mode")
                                break
                    except Exception:
                        pass

                
                if script_executed and script_rc == 0:
                    processed_files += 1
                    if _this_file_is_debt:  # v9.4.39
                        debt_files_processed += 1
                
                if script_executed and script_rc == 0 and file_path.exists():
                    processed_path = PROCESSED_DIR / f"{datetime.now(TZ).strftime('%Y%m%d%H%M%S')}_{file_path.name}"
                    shutil.move(file_path, processed_path)
                    log_event("file_processed", original=file_path.name, moved_to=processed_path.name)
                elif not file_path.exists():
                    log_event("queue_file_already_removed", file=file_path.name, level="INFO")
                elif script_executed and script_rc != 0:
                    log_event("file_kept_for_retry", file=file_path.name, rc=script_rc, level="WARNING")
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
    
    # v9.4.39: silence_alerts только по долговым файлам (не продажи/затраты/остатки)
    if debt_files_processed > 0:
        logger.info("🔔 Обработано долговых файлов: %d — запускаю проверку молчания", debt_files_processed)
        try:
            await check_and_send_silence_alerts(context)
        except Exception as e:
            logger.error(f"❌ Ошибка проверки дней молчания: {e}", exc_info=True)
        # v9.4.24: Финальный рейтинг продаж — в конце цикла, когда все JSON готовы.
        # Используем аккумулятор _pipeline_managers_by_period (точные имена из filename).
        if _pipeline_managers_by_period and SalesSummary:
            try:
                _s = SalesSummary()

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
                                _mark_notified_today(_mgr_name, _period)
                                log_event("sales_pipeline_summary_sent",
                                          recipient=_mgr_name, period=_period)
                            except Exception as _e2:
                                logger.warning(f"Сводка менеджеру {_mgr_name}: {_e2}")

                    # 3. СУБАДМИНЫ: мини-рейтинг каждой команды (из roles.json)
                    for _sa_cid_str, _sa_scope in ROLES.get("subadmin_scopes", {}).items():
                        try:
                            _sa_cid = int(_sa_cid_str)
                        except (ValueError, TypeError):
                            continue
                        if not isinstance(_sa_scope, list):
                            continue
                        # Имя субадмина — менеджер чей chat_id совпадает
                        _sa_name = next(
                            (n for n, cid in (MANAGERS_MAP or {}).items() if cid == _sa_cid),
                            None,
                        )
                        if not _sa_name:
                            continue
                        _scope_data = [
                            m for m in _mgrs_sorted
                            if m["manager"] in [_sa_name] + _sa_scope
                        ]
                        if len(_scope_data) > 1:
                            _sub_txt = _s.format_subadmin_pipeline(
                                _sa_name, _scope_data, _period
                            )
                            try:
                                _msg = await context.bot.send_message(
                                    chat_id=_sa_cid, text=_sub_txt, parse_mode=None
                                )
                                schedule_message_deletion(_sa_cid, _msg.message_id,
                                    _msg.date.timestamp(), delay_hours=24)
                                log_event("sales_pipeline_subadmin_sent",
                                          manager=_sa_name, period=_period,
                                          scope_count=len(_scope_data))
                            except Exception as _e3:
                                logger.warning(f"Субадмин-сводка продаж ({_sa_name}): {_e3}")

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
    # CRM: обновляем базу клиентов после каждого цикла в котором были новые файлы.
    # Новые клиенты из debt/sales JSON сразу попадают в clients.json —
    # не ждём 18:00, добавляем по мере появления.
    if processed_files > 0:
        try:
            from bot.crm_clients import update_from_reports as _crm_pipeline_update
            _new = _crm_pipeline_update()
            _new_total = sum(len(v) for v in _new.values())
            if _new_total:
                logger.info("CRM: добавлено %d новых клиентов после пайплайна", _new_total)
        except Exception as _crm_e:
            logger.warning("CRM pipeline update error: %s", _crm_e)

    # Если обработаны новые файлы — проверяем нет ли пропущенного MTD, напоминаем Саиде
    if processed_files > 0:
        try:
            await _remind_saida_mtd_if_missing(context)
        except Exception as _ms_e:
            logger.debug("mtd_remind_saida error: %s", _ms_e)

    log_event("pipeline_cycle_finish")

async def _suggest_weekly_clients(context, chat_id: int, categorized: dict, weekly_clients: list) -> None:
    """
    Предлагает Алене добавить клиентов-кандидатов (Шапагат и аналогичные)
    в список еженедельных. Срабатывает только если клиент попал в overdue (7-9 дн)
    и ещё не добавлен в weekly_clients.json.
    """
    from telegram import InlineKeyboardMarkup, InlineKeyboardButton
    overdue = categorized.get('overdue', [])
    if not overdue:
        return
    weekly_set = set(weekly_clients)
    candidates = [
        c for c in overdue
        if c['client'] not in weekly_set
        and 'шапагат' in c['client'].lower()
    ]
    for c in candidates[:5]:  # не более 5 предложений за раз
        client_name = c['client']
        token = _weekly_token_add(client_name)
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Да, исключить", callback_data=f"weekly_suggest|{token}"),
            InlineKeyboardButton("❌ Нет",           callback_data=f"weekly_reject|{token}"),
        ]])
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"❓ <b>{client_name}</b> — долг {c.get('debt_str','')}, {c['silence_days']} дн.\n"
                    f"Это еженедельный клиент? Исключить из просрочки?"
                ),
                parse_mode="HTML",
                reply_markup=kb,
            )
        except Exception as e:
            logger.warning("_suggest_weekly_clients send error: %s", e)


def _silence_clients_flat(categorized: dict) -> List[Dict[str, Any]]:
    clients: List[Dict[str, Any]] = []
    for key in ("critical", "alarm", "silence", "overdue", "partial_payment", "on_stop"):
        clients.extend(categorized.get(key, []) or [])
    return clients


def _get_saida_chat_id() -> int:
    try:
        val = (ROLES.get("accountants") or {}).get("Саида")
        if val:
            return int(val)
    except Exception:
        pass
    try:
        return int(os.getenv("SAIDA_CHAT_ID", "920236287"))
    except Exception:
        return 0


# Полная инструкция для Саиды по работе с запросами на проверку оплат.
# Используется в двух местах: callback `payhold_help_full` и кнопке-подсказке
# в text-handler. Держим в module-scope, чтобы оба handler-а могли сослаться.
_SAIDA_PAYHOLD_HELP_TEXT = (
    "📖 <b>Как отвечать по оплатам — коротко</b>\n\n"
    "Когда менеджер заявил оплату, я присылаю тебе сообщение с кнопками. "
    "Тебе достаточно нажать одну из кнопок:\n"
    "  ✅ Да, оплата есть\n"
    "  🔸 Частично\n"
    "  ❌ Не вижу оплаты\n\n"
    "<b>Если кнопок не видно</b> (например, сообщение уехало далеко вверх) — "
    "просто напиши в этот чат:\n"
    "  • <code>&lt;имя клиента&gt; полная</code> — например, «Акжан полная»\n"
    "  • <code>&lt;имя клиента&gt; частично</code>\n"
    "  • <code>&lt;имя клиента&gt; нет</code>\n\n"
    "Имя нужно одно слово из названия клиента (Акжан, Шапагат, Petro). "
    "Если у двух клиентов похожие имена — добавь второе слово (улицу, точку).\n\n"
    "<b>Что я НЕ понимаю:</b> «ок», «хорошо», «посмотрю позже» — это не ответ. "
    "По таким сообщениям статус оплаты не меняется. "
    "Если запрос неактуален — нажми «❌ Не вижу оплаты»."
)


async def _send_payment_check_buttons(context, chat_id: int, manager: str, categorized: dict) -> None:
    """Send a compact action panel after the short debt alert."""
    clients = _silence_clients_flat(categorized)
    if not clients:
        return
    try:
        from collector.payment_hold import create_manager_payment_request
    except Exception as exc:
        logger.warning("payment hold module unavailable: %s", exc)
        return

    rows = []
    for c in clients[:12]:
        rec = create_manager_payment_request(
            manager=manager,
            client=c.get("client", ""),
            debt=float(c.get("debt", 0) or 0),
            debt_str=str(c.get("debt_str", "")),
            manager_chat_id=chat_id,
        )
        rows.append([
            InlineKeyboardButton(
                f"Проверить у Саиды: {str(c.get('client', ''))[:32]}",
                callback_data=f"payhold_req|{rec['token']}",
            )
        ])
    if not rows:
        return
    try:
        msg = await context.bot.send_message(
            chat_id=chat_id,
            text=(
                "Если по клиенту уже оплатили, но в 1С ещё не разнесено, "
                "отправьте запрос Саиде:"
            ),
            reply_markup=InlineKeyboardMarkup(rows),
            parse_mode=None,
        )
        schedule_message_deletion(chat_id, msg.message_id, msg.date.timestamp(), delay_hours=24)
    except Exception as exc:
        logger.warning("payment check buttons send error manager=%s: %s", manager, exc)


async def _silence_delete_prev(context, key: str, state: dict) -> None:
    """v9.4.40: Удаляет предыдущее silence-сообщение если оно отправлено сегодня."""
    entry = state.get(key)
    if not entry:
        return
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    if entry.get("date") != today:
        return  # старое (вчера и раньше) — не трогаем, auto-delete сам уберёт
    chat_id = entry.get("chat_id")
    ids = entry.get("ids", [])
    if not chat_id or not ids:
        return
    deleted = 0
    for mid in ids:
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=mid)
            deleted += 1
        except Exception:
            pass  # уже удалено или истёк срок
    if deleted:
        log_event("silence_outdated_deleted", key=key, chat_id=chat_id, deleted=deleted)
        logger.info("🗑️ Удалено %d устаревших silence-сообщений для %s", deleted, key)


async def check_and_send_silence_alerts(context=None):
    """Проверяет дни молчания у всех менеджеров и отправляет уведомления"""
    from bot.workday_checker import is_holiday_today
    if is_holiday_today():
        logger.info("check_and_send_silence_alerts: выходной — пропуск")
        return
    logger.info("🔔 Начинается проверка дней молчания...")
    alert = SilenceAlert()
    reports_dir = HTML_DIR
    all_managers_data = {}
    manager_dates = {}  # v9.4.23: дата отчёта по каждому менеджеру
    _silence_state = _silence_load()       # v9.4.40
    _today = datetime.now(TZ).strftime("%Y-%m-%d")  # v9.4.40
    
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
            clients_data = alert.apply_residual_debt_age(clients_data)
            clients_data = alert.apply_payment_holds(clients_data)
            # shipment_violation из debt_ext JSON
            _json_path = JSON_DIR / (latest_report.stem + ".json")
            if _json_path.exists():
                try:
                    import json as _json_mod
                    _ext = _json_mod.loads(_json_path.read_text(encoding='utf-8'))
                    _vmap = {r['client']: r.get('shipment_violation', False)
                             for r in _ext.get('clients', []) if isinstance(r, dict) and 'client' in r}
                    for _c in clients_data:
                        if _vmap.get(_c['client']):
                            _c['shipment_violation'] = True
                except Exception as _e:
                    logger.warning("Не удалось загрузить %s: %s", _json_path.name, _e)
            # v1.4: исторические дни молчания из предыдущего файла
            prev_report = alert.get_prev_debt_report(reports_dir, manager)
            hist_map = alert.build_historical_silence_map(prev_report) if prev_report else {}
            weekly = _load_weekly_clients()
            categorized = alert.categorize_by_silence(clients_data, historical_map=hist_map,
                                                      weekly_clients=weekly)
            all_managers_data[manager] = categorized
            manager_dates[manager] = alert.parse_report_date(latest_report)
            _SILENCE_CATS = ('critical', 'alarm', 'silence', 'overdue', 'partial_payment', 'on_stop')
            total_silent = sum(len(categorized.get(k, [])) for k in _SILENCE_CATS)
            if total_silent == 0:
                logger.info(f"✅ У {manager} нет молчащих клиентов")
            else:
                logger.info(f"📊 У {manager}: {total_silent} молчащих клиентов")
        except Exception as e:
            logger.error(f"❌ Ошибка обработки {manager}: {e}", exc_info=True)
    
    _SILENCE_CATS = ('critical', 'alarm', 'silence', 'overdue', 'partial_payment', 'on_stop')
    for manager, categorized in all_managers_data.items():
        total_silent = sum(len(categorized.get(k, [])) for k in _SILENCE_CATS)
        
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
                    sub_total = sum(len(sub_cat.get(k, [])) for k in _SILENCE_CATS)
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
                    sub_total = sum(len(sub_categorized.get(k, [])) for k in _SILENCE_CATS)
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
                    _key_sub = f"subadmin_{manager}"
                    await _silence_delete_prev(context, _key_sub, _silence_state)  # v9.4.40
                    _ids_sub: list[int] = []
                    await _tg_send_long(context, chat_id, message, parse_mode=None, delay_hours=24,
                                        _collect_ids=_ids_sub)
                    _silence_state[_key_sub] = {"chat_id": chat_id, "ids": _ids_sub, "date": _today}
                    await _send_payment_check_buttons(context, chat_id, manager, categorized)
                    logger.info(f"✅ Уведомление отправлено субадмину {manager} (свои: {total_silent}, подшефные: {'есть' if has_subordinate_alerts else 'нет'})")
                    await send_main_menu(context, chat_id, get_user_role(chat_id))  # Fix #MENU-SILENCE
                except Exception as e:
                    logger.error(f"❌ Ошибка отправки {manager}: {e}")
        else:
            message = alert.format_manager_alert(manager, categorized, report_date=report_date)
            if message and context:
                try:
                    await _silence_delete_prev(context, manager, _silence_state)  # v9.4.40
                    _ids_mgr: list[int] = []
                    await _tg_send_long(context, chat_id, message, parse_mode=None, delay_hours=24,
                                        _collect_ids=_ids_mgr)
                    _silence_state[manager] = {"chat_id": chat_id, "ids": _ids_mgr, "date": _today}
                    await _send_payment_check_buttons(context, chat_id, manager, categorized)
                    logger.info(f"✅ Уведомление отправлено: {manager} ({total_silent} клиентов)")
                    await send_main_menu(context, chat_id, get_user_role(chat_id))  # Fix #MENU-SILENCE
                except Exception as e:
                    logger.error(f"❌ Ошибка отправки {manager}: {e}")

            # Предложить Алене добавить Шапагат-клиентов как еженедельных
            if context and manager == "Алена":
                await _suggest_weekly_clients(context, chat_id, categorized, _load_weekly_clients())
    
    if all_managers_data and context:
        try:
            # Детальная сводка для админа (с именами клиентов)
            admin_summary = alert.format_admin_detailed(all_managers_data, manager_dates=manager_dates)  # v9.4.23
            if ADMIN_CHAT_ID:
                await _silence_delete_prev(context, "admin", _silence_state)  # v9.4.40
                _ids_admin: list[int] = []
                await _tg_send_long(context, ADMIN_CHAT_ID, admin_summary, parse_mode=None, delay_hours=24,
                                    _collect_ids=_ids_admin)
                _silence_state["admin"] = {"chat_id": ADMIN_CHAT_ID, "ids": _ids_admin, "date": _today}
                logger.info(f"✅ Детальная сводка отправлена админу")
                await send_main_menu(context, ADMIN_CHAT_ID, "admin")  # Fix #MENU-SILENCE
            else:
                logger.warning("⚠️ ADMIN_CHAT_ID не установлен")
        except Exception as e:
            logger.error(f"❌ Ошибка отправки сводки админу: {e}", exc_info=True)
    _silence_save(_silence_state)  # v9.4.40: сохранить все id после полного прохода
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
    from bot.workday_checker import is_holiday_today
    if is_holiday_today():
        logger.info("send_opportunity_loss_report: выходной — пропуск")
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
                    f"(🔴{len(data['zones']['red'])} "
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
            _msg = await context.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text="💸 Упущенная прибыль: нет данных — у всех менеджеров молчащих должников (≥15 дней) нет.",
                parse_mode=None
            )
            schedule_message_deletion(ADMIN_CHAT_ID, _msg.message_id, _msg.date.timestamp(), delay_hours=24)
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
    "silence":    "🔔 Дебиторка",
    "oploss":     "💸 Упущенная прибыль",
    "sales":      "🛒 Продажи",
    "gross":      "💰 Валовая",
    "inventory":  "📦 Остатки",
    "net_profit": "💰 Чистая прибыль",
    "ranking":    "📊 Рейтинг менеджеров",  # C9
}

def _net_profit_mtd_is_deliverable(candidate: Optional[Path]) -> bool:
    if not candidate or not candidate.exists():
        return False
    try:
        import net_profit_report as _np

        all_gross = _np.load_summary_gross_jsons()
        all_expenses = _np.load_all_jsons("expenses_*.json")
        best_range = None
        seen = set()
        for gross in all_gross:
            period = _np.extract_period_from_json(gross)
            start, end = _np.extract_period_dates(period)
            if start is None or start == end:
                continue
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            best_range = (start, end, period)
            break
        if not best_range:
            return False
        start, _, period = best_range
        if not _np.find_matching_expenses_strict(period, all_expenses):
            log_event("stale_net_profit_mtd_blocked", file=candidate.name, reason="no_exact_expenses", level="WARNING")
            return False
        expected_name = f"net_profit_mtd_{start.strftime('%Y%m%d')}.html"
        if candidate.name != expected_name:
            log_event("stale_net_profit_mtd_blocked", file=candidate.name, expected=expected_name, level="WARNING")
            return False
        return True
    except Exception as e:
        log_event("net_profit_mtd_freshness_error", error=str(e), level="WARNING")
        return False


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
                clients = alert.apply_residual_debt_age(clients)
                clients = alert.apply_payment_holds(clients)
                # v1.4: исторические дни молчания из предыдущего файла
                prev = alert.get_prev_debt_report(HTML_DIR, mgr)
                hist_map = alert.build_historical_silence_map(prev) if prev else {}
                weekly = _load_weekly_clients()
                all_managers_data[mgr] = alert.categorize_by_silence(clients, historical_map=hist_map,
                                                                      weekly_clients=weekly)
                manager_dates[mgr] = alert.parse_report_date(latest) or ""

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
            # v9.4.20: JSON-путь как в планировщике (HTML-fallback сохранён)
            latest_json = summary.get_latest_inventory_json(JSON_DIR)
            if latest_json:
                data = summary.parse_inventory_json(latest_json)
            else:
                latest_html = summary.get_latest_inventory_report(HTML_DIR)
                if not latest_html:
                    return "📦 Нет данных остатков."
                data = summary.parse_inventory_html(latest_html)
            msg_text = summary.format_summary(data)

        # ── ЧИСТАЯ ПРИБЫЛЬ (net_profit) ───────────────────────────────────────
        elif report_type == "net_profit":
            if role != "admin":
                return "⛔ Чистая прибыль доступна только администратору."

            def _parse_np_html(path: Path) -> Optional[str]:
                """Парсит net_profit HTML → краткое текстовое уведомление."""
                import re as _re
                try:
                    raw = path.read_text(encoding="utf-8")
                except Exception:
                    return None
                # Убираем style/script, затем все теги
                clean = _re.sub(r'<style[^>]*>.*?</style>', ' ', raw, flags=_re.S)
                clean = _re.sub(r'<script[^>]*>.*?</script>', ' ', clean, flags=_re.S)
                clean = _re.sub(r'<[^>]+>', ' ', clean)
                clean = _re.sub(r'\s+', ' ', clean).strip()

                def _get(pattern: str) -> str:
                    m = _re.search(pattern, clean, _re.I)
                    return m.group(1).strip() if m else "—"

                # Период
                ptype  = "За день" if "за день" in clean.lower() else "За период"
                period = _get(r'(?:За день|За период)\s+📅\s*([^\s|]+(?:\s*[–-]\s*[^\s|]+)?)')
                if period == "—":
                    period = _get(r'(\d{2}\.\d{2}\.\d{4}(?:\s*[–-]\s*\d{2}\.\d{2}\.\d{4})?)')

                revenue  = _get(r'Выручка\s+([\d\s\u202f,]+₸)')
                gross    = _get(r'Валовая прибыль\s+([\d\s\u202f,]+₸)')
                gross_m  = _get(r'Валовая прибыль[\s\S]{1,50}Маржа:\s*([\d.,\-]+%)')
                expenses = _get(r'Расходы\s+([\d\s\u202f,]+₸)')
                np_val   = _get(r'Чистая прибыль\s+([\-\d\s\u202f,]+₸)')
                np_m     = _get(r'Чистая прибыль[\s\S]{1,50}Маржа:\s*([\-\d.,]+%)')
                sign     = "✅" if "-" not in np_val else "❌"

                return (
                    f"💰 *ЧИСТАЯ ПРИБЫЛЬ — {ptype.upper()}*\n"
                    f"📅 {period}\n\n"
                    f"📈 Выручка:          {revenue}\n"
                    f"💹 Валовая прибыль: {gross} ({gross_m})\n"
                    f"💸 Расходы:          {expenses}\n"
                    f"{'—'*28}\n"
                    f"{sign} Чистая прибыль: {np_val} ({np_m})"
                )

            # Берём последние файлы за день И за период
            parts: list = []
            for subdir in ["net_profit_day", "net_profit_mtd"]:
                search = ANALYTICS_DIR / subdir
                files = sorted(search.glob("net_profit*.html"),
                               key=lambda p: p.stat().st_mtime, reverse=True)
                if files:
                    if subdir == "net_profit_mtd" and not _net_profit_mtd_is_deliverable(files[0]):
                        continue
                    parsed = _parse_np_html(files[0])
                    if parsed:
                        parts.append(parsed)

            if not parts:
                return "💰 Нет файлов чистой прибыли."
            msg_text = "\n\n".join(parts)

        # ── РЕЙТИНГ МЕНЕДЖЕРОВ (C9) ───────────────────────────────────────────
        elif report_type == "ranking":
            if role != "admin":
                return "⛔ Рейтинг менеджеров доступен только администратору."
            msg_text = _build_manager_ranking(JSON_DIR, ANALYTICS_DIR)
            if not msg_text:
                return "📊 Нет данных для рейтинга менеджеров."

        else:
            return f"❓ Неизвестный тип отчёта: {report_type}"

        # ── Отправка (с разбивкой на чанки — сообщение может превышать 4096 символов)
        await _tg_send_long(context, chat_id, msg_text, parse_mode=None, delay_hours=24)
        if report_type == "silence" and role != "admin":
            if manager_name and manager_name in all_managers_data:
                await _send_payment_check_buttons(context, chat_id, manager_name, all_managers_data[manager_name])
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
    from bot.workday_checker import is_holiday_today
    if is_holiday_today():
        return
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
            msg = await context.bot.send_message(chat_id=chat_id, text=message, reply_markup=kb_open_menu())
            schedule_message_deletion(chat_id, msg.message_id, msg.date.timestamp(), delay_hours=24)
            log_event("notification_sent", chat_id=chat_id, reports_count=len(reports))
        except Exception as e:
            log_event("notification_error", chat_id=chat_id, error=str(e), level="ERROR")
    
    # v9.4.6.1: Атомарная запись с merge для защиты от race condition.
    # Фильтруем current по existing_paths: мёртвые пути (старые диски E:/F:) не накапливаются.
    try:
        current = _load_json_safe(NOTIFY_STATE_PATH)
        if not isinstance(current, dict):
            current = {}
        existing_paths = set(new_state.keys())
        pruned = {k: v for k, v in current.items() if k in existing_paths}
        merged = {**pruned, **new_state}
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
    cutoff_time = time.time() - (hours * 3600)
    if report_type == "DEBT":
        detailed = [
            p for p in JSON_DIR.glob(f"debt_ext_*Детальный Дебиторы {manager}*.json")
            if p.stat().st_mtime >= cutoff_time
        ]
        if detailed:
            return max(detailed, key=lambda p: p.stat().st_mtime)
        fallback = [
            p for p in JSON_DIR.glob(f"debt_ext_*{manager}*.json")
            if p.stat().st_mtime >= cutoff_time
        ]
        if fallback:
            return max(fallback, key=lambda p: p.stat().st_mtime)
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


def _load_fresh_debt_totals_by_manager(json_dir: Path) -> tuple[dict, str]:
    """
    Для рейтинга менеджеров используем только актуальные manager-specific detailed debt JSON.
    Старые "Ведомость ..." не должны подменять свежие долги.
    """
    debt_by_mgr: dict = {}
    debt_date = ""
    best_debt_date = None
    manager_candidates: dict = {}

    for path in json_dir.glob("debt_ext_*.json"):
        if "детальный дебиторы" not in path.name.lower():
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        mgr = (data.get("manager") or "").strip()
        if not mgr or mgr in ("Не определён", "Неизвестно", "?", "-", "—") or len(mgr) < 2:
            continue
        prev = manager_candidates.get(mgr)
        if prev is None or path.stat().st_mtime > prev.stat().st_mtime:
            manager_candidates[mgr] = path

    for mgr, path in manager_candidates.items():
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        agg_close = float((data.get("aggregates") or {}).get("close", 0) or 0)
        if agg_close <= 0:
            continue
        debt_by_mgr[mgr] = agg_close
        pmax = (data.get("period_max") or "").strip()
        dm2 = re.findall(r'(\d{1,2})[./](\d{1,2})[./](\d{4})', pmax)
        if dm2:
            dd, mm, yyyy = dm2[-1]
            try:
                from datetime import date as _date2
                pd = _date2(int(yyyy), int(mm), int(dd))
                if best_debt_date is None or pd > best_debt_date:
                    best_debt_date = pd
                    debt_date = pmax
            except Exception:
                pass

    return debt_by_mgr, debt_date

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
    if not await _acl_gate(chat_id, context):
        return
    user_role = get_user_role(chat_id)
    
    # v2.0: Трекинг пользователя
    if track_user:
        user = update.effective_user
        track_user(user.id, user.first_name, user.username)
        track_action(user.id, "start")
    
    await send_main_menu(context, chat_id, user_role, text="📋 Выберите раздел:")


async def cmd_dev(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not await _acl_gate(chat_id, context):
        return
    _DEV_FEEDBACK_ARMED[chat_id] = datetime.now(TZ).isoformat()
    _dev_feedback_save()
    await context.bot.send_message(
        chat_id=chat_id,
        text=_dev_feedback_open_text(),
        parse_mode="HTML",
        reply_markup=_dev_feedback_menu_kb(),
    )


async def cmd_announce_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id != _get_developer_chat_id():
        await _send_auto(context, chat_id, "⛔ Доступ запрещён.")
        return
    sent, failed = await _broadcast_reset_notice(context.bot)
    await _send_auto(
        context,
        chat_id,
        f"✅ Информационная рассылка завершена.\n\nОтправлено: {sent}\nОшибок: {failed}"
    )


async def cmd_devhelp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id != _get_developer_chat_id():
        await _send_auto(context, chat_id, "⛔ Доступ запрещён.")
        return
    await context.bot.send_message(
        chat_id=chat_id,
        text=_developer_command_help_text(),
        parse_mode="HTML",
        reply_markup=_dev_feedback_menu_kb(),
    )

async def cmd_batch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Вызов из меню или командой /batch — открывает экран коллектора.
    Если есть pending_admin батч — сразу присылает сводку с кнопками утверждения.
    """
    chat_id = update.effective_chat.id
    if not is_admin(chat_id):
        await update.effective_chat.send_message("⛔ Доступ запрещён.")
        return
    pending = _get_pending_admin_batch()
    if pending:
        try:
            from collector.approval_flow import send_admin_summary
            await send_admin_summary(pending, context.bot)
            return
        except Exception as _e:
            logger.error("cmd_batch send_admin_summary error: %s", _e)
    # Нет pending_admin — показываем экран коллектора
    text = _format_collector_batch_text()
    kb   = _collector_batch_keyboard()
    await update.effective_chat.send_message(text, reply_markup=kb, parse_mode="HTML")


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
    await _send_auto(context, update.effective_chat.id, text)

async def cmd_logs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает последние ERROR/CRITICAL из runtime-лога. Только для admin."""
    chat_id = update.effective_chat.id
    if not is_admin(chat_id):
        await _send_auto(context, chat_id, "⛔ Доступ запрещён.")
        return

    args = (context.args or [])
    try:
        n = max(5, min(50, int(args[0]))) if args else 20
    except (ValueError, IndexError):
        n = 20

    log_path = LOGS_DIR / "send_reports.log"
    if not log_path.exists():
        await _send_auto(context, chat_id, "📭 Лог-файл не найден.")
        return

    try:
        all_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        await _send_auto(context, chat_id, f"❌ Ошибка чтения лога: {e}")
        return

    error_lines = [l for l in all_lines if " ERROR " in l or " CRITICAL " in l][-n:]

    if not error_lines:
        await _send_auto(context, chat_id, "✅ ERROR/CRITICAL записей не найдено.")
        return

    block = "\n".join(error_lines)
    # Telegram code-block лимит ~4096 символов, оставляем запас на обёртку
    if len(block) > 3800:
        block = "…\n" + block[-3800:]

    await _send_auto(
        context,
        chat_id,
        f"🔍 <b>Последние {len(error_lines)} ERROR/CRITICAL:</b>\n<code>{_html.escape(block)}</code>",
    )


async def cmd_timeline(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает единую CRM+Collector timeline по клиенту. Только для admin."""
    chat_id = update.effective_chat.id
    if not is_admin(chat_id):
        await _send_auto(context, chat_id, "⛔ Доступ запрещён.")
        return

    client_key = " ".join(context.args or []).strip()
    if not client_key:
        await _send_auto(context, chat_id, "Использование: /timeline <название клиента>")
        return

    records = read_client_timeline(
        client_key,
        crm_path=LOGS_DIR / "crm_audit.jsonl",
        collector_path=LOGS_DIR / "collector_audit.jsonl",
        limit=60,
    )
    await _send_auto(context, chat_id, format_client_timeline(client_key, records))


async def morning_error_digest_task(context: ContextTypes.DEFAULT_TYPE):
    """Отправляет админу краткий digest WARNING/ERROR/CRITICAL по доменам за ночь."""
    if not ADMIN_CHAT_ID:
        return
    try:
        now = datetime.now(TZ).replace(tzinfo=None)
        counts = summarize_errors_by_system(LOGS_DIR, now=now, hours=12)
        message = format_error_digest(counts, now=now, hours=12)
        await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=message, parse_mode="HTML")
        sched_logger.info("morning_error_digest: delivered to admin")
    except Exception as e:
        sched_logger.error("morning_error_digest failed: %s", e, exc_info=True)


async def _exit_after_reply(delay_sec: float = 1.0) -> None:
    await asyncio.sleep(delay_sec)
    _clear_pid()
    os._exit(0)

async def cmd_restart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id != ADMIN_CHAT_ID:
        await _send_auto(context, chat_id, "⛔ Доступ запрещён.")
        return
    try:
        if STOP_FILE.exists():
            STOP_FILE.unlink()
    except OSError as e:
        logger.warning("restart: cannot remove stop file: %s", e)
    log_event("bot_restart_requested", chat_id=chat_id)
    await _send_auto(context, chat_id, "🔄 Перезапускаю бота. Watchdog поднимет новый процесс через несколько секунд.")
    asyncio.create_task(_exit_after_reply())

async def cmd_shutdown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id != ADMIN_CHAT_ID:
        await _send_auto(context, chat_id, "⛔ Доступ запрещён.")
        return
    try:
        STOP_FILE.write_text(
            f"Stopped by Telegram /shutdown at {datetime.now(TZ).isoformat()} chat_id={chat_id}\n",
            encoding="utf-8",
        )
    except OSError as e:
        logger.error("shutdown: cannot write stop file: %s", e)
        await _send_auto(context, chat_id, f"❌ Не удалось создать стоп-файл: {e}")
        return
    log_event("bot_shutdown_requested", chat_id=chat_id, stop_file=str(STOP_FILE))
    await _send_auto(context, chat_id, "⏹️ Останавливаю бота. Watchdog увидит стоп-файл и не будет перезапускать.")
    asyncio.create_task(_exit_after_reply())

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """v2.0: Статистика использования бота"""
    chat_id = update.effective_chat.id

    if not is_admin(chat_id):
        await _send_auto(context, chat_id, "⛔ Доступ запрещён.")
        return

    if not get_stats:
        await _send_auto(context, chat_id, "⚠️ Модуль аналитики не загружен.")
        return

    try:
        stats = get_stats()
        message = format_stats_message(stats)
        await _send_auto(context, chat_id, message, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Ошибка в cmd_stats: {e}", exc_info=True)
        await _send_auto(context, chat_id, f"❌ Ошибка при получении статистики: {e}")

# Хранит временные данные выбора клиента: {chat_id: {"phone": str, "alias": str, "candidates": [str]}}
_CRM_PHONE_PENDING: Dict[int, Dict[str, Any]] = {}
CRM_DAILY_LIMIT = 15  # максимум клиентов за один сеанс в 18:00
CRM_PENDING_PATH = LOGS_DIR / "crm_pending_state.json"
CRM_PENDING_TTL_HOURS = float(os.getenv("CRM_PENDING_TTL_HOURS", "48"))
COLLECTOR_PENDING_TTL_DAYS = int(os.getenv("COLLECTOR_PENDING_TTL_DAYS", "2"))

# One-shot inbox "Для Разработчика":
# пользователь нажимает кнопку в меню, следующее текстовое/медиа сообщение
# уходит разработчику. Разработчик отвечает адресно простым reply.
DEVELOPER_FEEDBACK_PATH = LOGS_DIR / "developer_feedback_state.json"
DEVELOPER_FEEDBACK_ARM_TTL_MIN = int(os.getenv("DEVELOPER_FEEDBACK_ARM_TTL_MIN", "30"))
DEVELOPER_FEEDBACK_THREAD_TTL_HOURS = int(os.getenv("DEVELOPER_FEEDBACK_THREAD_TTL_HOURS", "168"))
_DEV_FEEDBACK_ARMED: Dict[int, str] = {}
_DEV_FEEDBACK_THREADS: Dict[str, Dict[str, Any]] = {}
_DEV_FEEDBACK_REPLY_INDEX: Dict[int, str] = {}


def _get_developer_chat_id() -> int:
    raw = os.getenv("DEVELOPER_CHAT_ID", "").strip()
    if raw:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    return int(ADMIN_CHAT_ID or 0)


def _dev_feedback_sender_name(chat_id: int, update: Optional[Update] = None) -> str:
    role = get_user_role(chat_id)
    if role == "admin":
        return "Вадим"
    if role == "manager":
        return get_my_manager_name(chat_id) or "Менеджер"
    if role == "subadmin":
        return get_my_manager_name(chat_id) or "Супервайзер"
    if role == "saida":
        return "Саида"
    if update and update.effective_user:
        return update.effective_user.full_name or update.effective_user.first_name or str(chat_id)
    return str(chat_id)


def _dev_feedback_sender_role(chat_id: int) -> str:
    role = get_user_role(chat_id)
    labels = {
        "admin": "admin",
        "subadmin": "subadmin",
        "manager": "manager",
        "saida": "saida",
    }
    return labels.get(role, "user")


def _dev_feedback_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📋 Открыть меню", callback_data="back_main")],
        [InlineKeyboardButton("📖 Инструкция", callback_data="show_help_doc")],
        [InlineKeyboardButton("🧾 Команды разработчика", callback_data="dev_commands")],
    ])


def _dev_feedback_open_text() -> str:
    return (
        "🛠️ <b>Связь с разработчиком</b>\n\n"
        "Следующим сообщением отправьте то, что нужно передать:\n"
        "• текст\n"
        "• скриншот как фото\n"
        "• файл или документ\n"
        "• пересланное сообщение\n\n"
        "Если отправляете скрин или файл, лучше добавьте подпись: что случилось и где именно.\n"
        "После отправки я сразу перешлю это разработчику. "
        "Он сможет ответить вам адресно прямо в этот чат."
    )


def _build_reset_broadcast_text() -> str:
    return (
        "ℹ️ <b>Важное обновление по работе бота</b>\n\n"
        "Сегодня выполнен полный технический сброс рабочих хвостов.\n"
        "Старые запросы, зависшие очереди и тестовые записи удалены.\n"
        "С этого момента бот работает <b>с чистого нуля</b>.\n\n"
        "<b>Что это значит для вас:</b>\n"
        "• старые запросы больше не актуальны;\n"
        "• новые сообщения от бота не игнорируем;\n"
        "• отвечаем вовремя и по кнопкам/инструкциям из сообщения;\n"
        "• если сообщение пришло повторно, считаем его рабочим и обрабатываем.\n\n"
        "<b>Если что-то непонятно:</b>\n"
        "1. Откройте меню.\n"
        "2. Нажмите <b>📖 Инструкция</b>.\n"
        "3. Выберите нужный сценарий и действуйте по подсказке.\n\n"
        "<b>Если есть ошибка, вопрос или неудобство:</b>\n"
        "1. Откройте меню.\n"
        "2. Нажмите <b>🛠️ Для Разработчика</b>.\n"
        "3. Следующим сообщением отправьте текст, скрин, фото, файл или пересланное сообщение.\n\n"
        "По скринам лучше писать подпись: что именно не так, у кого, в каком разделе и что ожидали увидеть.\n"
        "Разработчик получит это напрямую и сможет ответить вам адресно в этот же чат.\n\n"
        "Важно: теперь новые рабочие сообщения считаем актуальными и не откладываем без ответа."
    )


def _developer_command_help_text() -> str:
    return (
        "🧾 <b>Краткая справка по developer-командам</b>\n\n"
        "<b>Обратная связь от команды</b>\n"
        "• <code>/dev</code> — открыть режим «Для Разработчика» для себя.\n"
        "  Следующее сообщение, фото, скрин, файл или пересланный текст уйдёт разработчику.\n"
        "• В меню у сотрудников: <b>🛠️ Для Разработчика</b> — делает то же самое.\n"
        "• Чтобы ответить человеку адресно: просто сделайте <b>reply</b> на карточку тикета или на пересланное вложение.\n\n"
        "<b>Рассылка после технического сброса</b>\n"
        "• <code>/announce_reset</code> — отправить всем участникам инфо-сообщение,\n"
        "  что старые и тестовые запросы очищены, бот работает с нуля и новые сообщения игнорировать нельзя.\n\n"
        "<b>Технические команды бота</b>\n"
        "• <code>/start</code> — открыть главное меню.\n"
        "• <code>/guide</code> / <code>/help</code> — отправить инструкцию по роли.\n"
        "• <code>/logs</code> — последние ERROR/CRITICAL.\n"
        "• <code>/timeline &lt;клиент&gt;</code> — единая timeline по клиенту.\n"
        "• <code>/stats</code> — статистика использования бота.\n"
        "• <code>/restart</code> — перезапуск бота.\n"
        "• <code>/shutdown</code> — остановка бота.\n\n"
        "<b>CRM</b>\n"
        "• <code>/phone Имя клиента 87XXXXXXXXX</code> — записать номер вручную.\n"
        "• <code>/crmdupsend</code> — разовая отправка очереди конфликтных дублей CRM.\n\n"
        "<b>Практика по скринам и файлам</b>\n"
        "• Если шлёте скрин, добавляйте подпись: что произошло, у кого, где именно и что ожидали увидеть.\n"
        "• Если проблема в конкретном сообщении — можно просто переслать его через «Для Разработчика».\n"
    )


def _all_bot_participants() -> Dict[int, str]:
    participants: Dict[int, str] = {}
    if ADMIN_CHAT_ID:
        participants[int(ADMIN_CHAT_ID)] = "Вадим"
    for name, chat_id in (MANAGERS_MAP or {}).items():
        if not chat_id or name in _SYSTEM_ACCOUNTS:
            continue
        participants[int(chat_id)] = name
    for chat_str in (ROLES.get("subadmin_scopes", {}) or {}):
        try:
            cid = int(chat_str)
        except (TypeError, ValueError):
            continue
        participants.setdefault(cid, get_my_manager_name(cid) or "Супервайзер")
    saida_cid = _get_saida_chat_id()
    if saida_cid:
        participants.setdefault(int(saida_cid), "Саида")
    return participants


def _dev_feedback_cleanup(now_dt: Optional[datetime] = None) -> None:
    now_dt = now_dt or datetime.now(TZ)
    armed_cutoff = now_dt - timedelta(minutes=DEVELOPER_FEEDBACK_ARM_TTL_MIN)
    thread_cutoff = now_dt - timedelta(hours=DEVELOPER_FEEDBACK_THREAD_TTL_HOURS)

    stale_armed = []
    for chat_id, created_raw in list(_DEV_FEEDBACK_ARMED.items()):
        try:
            created_dt = datetime.fromisoformat(created_raw)
        except (TypeError, ValueError):
            stale_armed.append(chat_id)
            continue
        if created_dt < armed_cutoff:
            stale_armed.append(chat_id)
    for chat_id in stale_armed:
        _DEV_FEEDBACK_ARMED.pop(chat_id, None)

    stale_threads = []
    for ticket_id, payload in list(_DEV_FEEDBACK_THREADS.items()):
        try:
            created_dt = datetime.fromisoformat(str(payload.get("created_at") or ""))
        except (TypeError, ValueError):
            stale_threads.append(ticket_id)
            continue
        if created_dt < thread_cutoff:
            stale_threads.append(ticket_id)
    for ticket_id in stale_threads:
        _DEV_FEEDBACK_THREADS.pop(ticket_id, None)

    _DEV_FEEDBACK_REPLY_INDEX.clear()
    for ticket_id, payload in _DEV_FEEDBACK_THREADS.items():
        for msg_id in payload.get("reply_message_ids", []) or []:
            try:
                _DEV_FEEDBACK_REPLY_INDEX[int(msg_id)] = ticket_id
            except (TypeError, ValueError):
                continue


def _dev_feedback_save() -> None:
    DEVELOPER_FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    _dev_feedback_cleanup()
    tmp = DEVELOPER_FEEDBACK_PATH.with_suffix(".tmp")
    payload = {
        "armed": {str(k): v for k, v in _DEV_FEEDBACK_ARMED.items()},
        "threads": _DEV_FEEDBACK_THREADS,
    }
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, DEVELOPER_FEEDBACK_PATH)


def _dev_feedback_load() -> None:
    _DEV_FEEDBACK_ARMED.clear()
    _DEV_FEEDBACK_THREADS.clear()
    _DEV_FEEDBACK_REPLY_INDEX.clear()
    if not DEVELOPER_FEEDBACK_PATH.exists():
        return
    try:
        payload = json.loads(DEVELOPER_FEEDBACK_PATH.read_text(encoding="utf-8"))
        armed = payload.get("armed", {}) if isinstance(payload, dict) else {}
        threads = payload.get("threads", {}) if isinstance(payload, dict) else {}
        if isinstance(armed, dict):
            _DEV_FEEDBACK_ARMED.update({int(k): v for k, v in armed.items()})
        if isinstance(threads, dict):
            _DEV_FEEDBACK_THREADS.update(threads)
        _dev_feedback_cleanup()
        state_logger.info("developer_feedback restored: armed=%d threads=%d", len(_DEV_FEEDBACK_ARMED), len(_DEV_FEEDBACK_THREADS))
    except Exception as exc:
        state_logger.warning("_dev_feedback_load error: %s", exc)


async def _send_feedback_to_developer(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    content_type: str,
    text: str = "",
) -> bool:
    chat_id = update.effective_chat.id
    armed_at = _DEV_FEEDBACK_ARMED.pop(chat_id, None)
    if not armed_at:
        return False
    _dev_feedback_save()

    developer_chat_id = _get_developer_chat_id()
    if not developer_chat_id:
        await update.effective_message.reply_text("⚠️ Канал разработчика пока не настроен.")
        return True

    ticket_id = f"dev-{datetime.now(TZ).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
    sender_name = _dev_feedback_sender_name(chat_id, update)
    sender_role = _dev_feedback_sender_role(chat_id)
    username = getattr(update.effective_user, "username", "") or "—"
    header_lines = [
        "🛠️ <b>Новое сообщение для разработчика</b>",
        "",
        f"Тикет: <code>{ticket_id}</code>",
        f"От: <b>{_html.escape(sender_name)}</b> ({sender_role})",
        f"chat_id: <code>{chat_id}</code>",
        f"username: @{_html.escape(username)}" if username != "—" else "username: —",
        f"Тип: {content_type}",
        "",
        "Ответьте <b>reply</b> на это сообщение текстом, фото или файлом — бот доставит ответ отправителю.",
    ]
    if text:
        header_lines.extend(["", "<b>Комментарий:</b>", _html.escape(text[:3000])])

    header_msg = await context.bot.send_message(
        chat_id=developer_chat_id,
        text="\n".join(header_lines),
        parse_mode="HTML",
    )
    reply_ids = [header_msg.message_id]

    if content_type != "text":
        copied = await context.bot.copy_message(
            chat_id=developer_chat_id,
            from_chat_id=chat_id,
            message_id=update.effective_message.message_id,
        )
        reply_ids.append(copied.message_id)

    _DEV_FEEDBACK_THREADS[ticket_id] = {
        "source_chat_id": chat_id,
        "source_name": sender_name,
        "source_role": sender_role,
        "reply_message_ids": reply_ids,
        "created_at": datetime.now(TZ).isoformat(),
    }
    _dev_feedback_save()

    await update.effective_message.reply_text(
        "✅ Сообщение отправлено разработчику.\n"
        "Если нужен ещё один текст или скрин, снова нажмите «🛠️ Для Разработчика».",
        reply_markup=_dev_feedback_menu_kb(),
    )
    return True


async def _maybe_handle_developer_text_flow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    msg = update.effective_message
    if not msg:
        return False
    chat_id = update.effective_chat.id
    reply_to = msg.reply_to_message.message_id if msg.reply_to_message else 0
    developer_chat_id = _get_developer_chat_id()

    if developer_chat_id and chat_id == developer_chat_id and reply_to in _DEV_FEEDBACK_REPLY_INDEX:
        ticket_id = _DEV_FEEDBACK_REPLY_INDEX.get(reply_to, "")
        thread = _DEV_FEEDBACK_THREADS.get(ticket_id) or {}
        source_chat_id = int(thread.get("source_chat_id") or 0)
        if not source_chat_id:
            await msg.reply_text("⚠️ Тикет уже закрыт или не найден.")
            return True
        await context.bot.send_message(
            chat_id=source_chat_id,
            text=(
                "🛠️ <b>Ответ разработчика</b>\n\n"
                f"{_html.escape(msg.text or '')}"
            ),
            parse_mode="HTML",
        )
        await msg.reply_text(
            f"✅ Ответ доставлен: {_html.escape(str(thread.get('source_name') or source_chat_id))}.",
            parse_mode="HTML",
        )
        return True

    if chat_id not in _DEV_FEEDBACK_ARMED:
        return False
    return await _send_feedback_to_developer(
        update,
        context,
        content_type="text",
        text=(msg.text or "").strip(),
    )


async def _maybe_handle_developer_media_flow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    msg = update.effective_message
    if not msg:
        return False
    chat_id = update.effective_chat.id
    reply_to = msg.reply_to_message.message_id if msg.reply_to_message else 0
    developer_chat_id = _get_developer_chat_id()

    if developer_chat_id and chat_id == developer_chat_id and reply_to in _DEV_FEEDBACK_REPLY_INDEX:
        ticket_id = _DEV_FEEDBACK_REPLY_INDEX.get(reply_to, "")
        thread = _DEV_FEEDBACK_THREADS.get(ticket_id) or {}
        source_chat_id = int(thread.get("source_chat_id") or 0)
        if not source_chat_id:
            await msg.reply_text("⚠️ Тикет уже закрыт или не найден.")
            return True
        await context.bot.send_message(
            chat_id=source_chat_id,
            text="🛠️ Ответ разработчика:",
        )
        await context.bot.copy_message(
            chat_id=source_chat_id,
            from_chat_id=chat_id,
            message_id=msg.message_id,
        )
        await msg.reply_text(
            f"✅ Вложение доставлено: {_html.escape(str(thread.get('source_name') or source_chat_id))}.",
            parse_mode="HTML",
        )
        return True

    if chat_id not in _DEV_FEEDBACK_ARMED:
        return False

    media_type = "фото" if msg.photo else "документ" if msg.document else "вложение"
    return await _send_feedback_to_developer(
        update,
        context,
        content_type=media_type,
        text=(msg.caption or "").strip(),
    )


async def _broadcast_reset_notice(bot) -> Tuple[int, int]:
    text = _build_reset_broadcast_text()
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("📋 Открыть меню", callback_data="back_main")],
        [InlineKeyboardButton("📖 Инструкция", callback_data="show_help_doc")],
        [InlineKeyboardButton("🛠️ Для Разработчика", callback_data="dev_feedback_open")],
    ])
    sent = 0
    failed = 0
    for chat_id, _name in _all_bot_participants().items():
        try:
            await bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML", reply_markup=kb)
            sent += 1
        except Exception as exc:
            failed += 1
            logger.warning("reset broadcast failed chat_id=%s: %s", chat_id, exc)
    return sent, failed


_CRM_PREFIX_MAP: Dict[str, str] = {
    "А": "Алена",
    "Е": "Ергали",
    "М": "Магира",
    "О": "Оксана",
}
_CRM_SERVICE_KEYS = frozenset({"без клиента", "недостача"})


def _crm_manager_from_prefix(client_key: str) -> Optional[str]:
    """Возвращает имя менеджера если client_key начинается с 'Х ' (буква + пробел).

    Исключения: служебные записи (Без клиента, Недостача, *зп*) → None.
    """
    if not client_key or len(client_key) < 3:
        return None
    ck_lower = client_key.lower().strip()
    if ck_lower in _CRM_SERVICE_KEYS or "зп" in ck_lower:
        return None
    if client_key[1] == " ":
        return _CRM_PREFIX_MAP.get(client_key[0].upper())
    return None


def _crm_name_choice_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✏️ Ввести имя", callback_data="crm_name|edit")],
        [InlineKeyboardButton("✅ Оставить как в системе", callback_data="crm_name|keep")],
        [InlineKeyboardButton("⏳ Позже", callback_data="crm_name|later")],
        [InlineKeyboardButton("❓ Не понимаю, что ответить", callback_data="crm_help")],
    ])


def _crm_phone_suggestions(client_key: str) -> List[str]:
    try:
        from bot.crm_clients import extract_phones_from_client_name
        return extract_phones_from_client_name(client_key)
    except Exception as e:
        logger.warning("CRM phone suggestion extraction error: %s", e)
        return []


def _crm_phone_prompt_text(client_key: str, suggestions: Optional[List[str]] = None) -> str:
    suggestions = suggestions if suggestions is not None else _crm_phone_suggestions(client_key)
    if not suggestions:
        return "Введите телефон WhatsApp:\n<code>+7XXXXXXXXXX</code>"
    if len(suggestions) == 1:
        return (
            "У клиента нет WhatsApp в CRM.\n\n"
            f"Клиент:\n<b>{client_key}</b>\n\n"
            "В названии найден возможный номер:\n"
            f"<code>{suggestions[0]}</code>\n\n"
            "Подтвердите WhatsApp клиента."
        )
    rows = "\n".join(f"{idx}. <code>{phone}</code>" for idx, phone in enumerate(suggestions, start=1))
    return (
        "У клиента нет WhatsApp в CRM.\n\n"
        f"Клиент:\n<b>{client_key}</b>\n\n"
        "В названии найдено несколько возможных номеров:\n"
        f"{rows}\n\n"
        "Выберите актуальный номер или укажите другой."
    )


def _crm_key_token(client_key: str) -> str:
    """Короткий хэш client_key для верификации callback-кнопок (F-09)."""
    import hashlib
    return hashlib.sha1(client_key.encode("utf-8")).hexdigest()[:8]


def _crm_phone_choice_kb(client_key: str) -> Optional[InlineKeyboardMarkup]:
    suggestions = _crm_phone_suggestions(client_key)
    if not suggestions:
        return None
    # F-09: включаем token клиента в callback_data — при нажатии старой кнопки
    # можно обнаружить что pending уже для другого клиента
    tok = _crm_key_token(client_key)
    rows = []
    if len(suggestions) == 1:
        rows.append([InlineKeyboardButton("✅ Да, записать", callback_data=f"crm_phone|suggest|0|{tok}")])
    else:
        for idx, phone in enumerate(suggestions[:5]):
            rows.append([InlineKeyboardButton(f"✅ Записать {phone}", callback_data=f"crm_phone|suggest|{idx}|{tok}")])
    rows.append([InlineKeyboardButton("✏️ Указать другой номер", callback_data=f"crm_phone|edit|{tok}")])
    rows.append([InlineKeyboardButton("❓ Не понимаю, что ответить", callback_data="crm_help")])
    return InlineKeyboardMarkup(rows)


def _crm_phone_help_only_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("❓ Не понимаю, что ответить", callback_data="crm_help")]
    ])


def _crm_escalation_header(count: int, area: str = "CRM") -> str:
    if count <= 0:
        return ""
    if count == 1:
        return f"⏰ <b>Напоминание {area}</b> — ждём ответа:\n\n"
    if count == 2:
        return f"⚠️ <b>Повторное напоминание {area}</b> — запрос не закрыт:\n\n"
    if count == 3:
        return f"🚨 <b>Срочно {area}</b> — последнее предупреждение перед эскалацией:\n\n"
    return f"🔥 <b>{area}: просроченный запрос #{count}</b> — выполнить сейчас:\n\n"


def _crm_name_prompt_text(
    client_key: str,
    done_today: int,
    total: int,
    daily_limit: int,
    reminder: bool = False,
    remind_count: int = 0,
) -> str:
    header = _crm_escalation_header(remind_count or 1, "CRM") if reminder else ""
    prefix = "📋 Нужно внести контакты клиентов" if done_today == 0 else "📋 Продолжаем CRM-очередь"
    return (
        f"{header}"
        f"{prefix} — <b>{done_today + 1} из {min(daily_limit, done_today + total)}</b>\n\n"
        f"<b>{client_key}</b>\n\n"
        f"Как к нему обращаться?\n"
        f"Можно ввести удобное имя, оставить как в системе или вернуться к имени позже.\n\n"
        f"<i>Я вижу игнор. Каждый день без ответа фиксируется — "
        f"руководитель получит рекомендацию задержать зарплату на столько же дней. "
        f"Запрос повторяется каждые 30 минут.</i>"
    )


async def _crm_notify_admin_unresolved(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    pending: Dict[str, Any],
    remind_count: int,
) -> None:
    if not ADMIN_CHAT_ID:
        return
    if remind_count < 3:
        return
    if int(pending.get("admin_notice_count", 0) or 0) >= remind_count:
        return
    pending["admin_notice_count"] = remind_count
    manager = pending.get("manager") or _chat_to_manager(chat_id) or "?"
    client_key = pending.get("client_key", "?")
    state = pending.get("state", "?")
    manager_pending = [
        p for p in _CRM_PHONE_PENDING.values()
        if (p.get("manager") or "") == manager
    ]
    total_reminders = sum(int(p.get("remind_count", 0) or 0) for p in manager_pending)
    max_reminders = max([int(p.get("remind_count", 0) or 0) for p in manager_pending] or [0])
    ignored_lines = []
    for item in sorted(
        manager_pending,
        key=lambda p: int(p.get("remind_count", 0) or 0),
        reverse=True,
    )[:10]:
        ignored_lines.append(
            f"• {item.get('client_key', '?')} — "
            f"{int(item.get('remind_count', 0) or 0)} напомин."
        )
    ignored_details = "\n".join(ignored_lines) or "Нет открытых CRM-запросов"
    try:
        await context.bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text=(
                f"🚨 <b>Менеджер не выполняет CRM-запрос</b>\n\n"
                f"Менеджер: <b>{manager}</b>\n"
                f"Клиент: <b>{client_key}</b>\n"
                f"Этап: <b>{state}</b>\n\n"
                f"Напоминаний менеджеру уже: <b>{remind_count}</b>.\n"
                f"Данные всё ещё не заполнены.\n\n"
                f"📊 <b>Статистика игнора по менеджеру</b>\n"
                f"Открытых CRM-запросов: <b>{len(manager_pending)}</b>\n"
                f"Всего CRM-напоминаний: <b>{total_reminders}</b>\n"
                f"Максимум по одному клиенту: <b>{max_reminders}</b>\n\n"
                f"<b>По каждому открытому запросу:</b>\n"
                f"{ignored_details}"
            ),
            parse_mode="HTML",
        )
    except Exception as e:
        logger.warning("CRM admin unresolved notify error chat_id=%s client=%s: %s", chat_id, client_key, e)


def _crm_cleanup_pending() -> None:
    """Убирает просроченные CRM pending-кейсы с явной записью в лог."""
    now = datetime.now(TZ)
    stale_chat_ids: List[int] = []
    for chat_id, pending in list(_CRM_PHONE_PENDING.items()):
        # Служебные записи (зарплатные авансы и т.п.) — убираем из памяти
        ck = (pending.get("client_key") or "").lower()
        if "зп" in ck or ck in ("без клиента", "недостача"):
            state_logger.info("CRM cleanup: removing service entry '%s' (chat_id=%s)", ck, chat_id)
            stale_chat_ids.append(chat_id)
            continue
        # F-08: TTL считаем от created_at (неизменяем), а не last_sent
        # last_sent сбрасывается каждым напоминанием → pending никогда не истекал
        ts_raw = pending.get("created_at") or pending.get("last_sent")
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(ts_raw)
        except (TypeError, ValueError):
            state_logger.warning("CRM pending invalid timestamp: chat_id=%s raw=%r", chat_id, ts_raw)
            stale_chat_ids.append(chat_id)
            continue
        age_hours = (now - ts).total_seconds() / 3600
        if age_hours <= CRM_PENDING_TTL_HOURS:
            continue
        state_logger.warning(
            "CRM pending expired, removing: chat_id=%s client=%s state=%s age_hours=%.1f",
            chat_id,
            pending.get("client_key"),
            pending.get("state"),
            age_hours,
        )
        stale_chat_ids.append(chat_id)
    for chat_id in stale_chat_ids:
        _CRM_PHONE_PENDING.pop(chat_id, None)


def _cleanup_legacy_collector_pending_state() -> None:
    """Чистит просроченные __phone_pending__/__name_pending__ записи из collector_state.json."""
    try:
        from collector.collections_db import load_state, save_state
    except Exception as _e:
        logger.warning("legacy collector pending cleanup import error: %s", _e)
        return

    state = load_state()
    today = datetime.now(TZ).date()
    changed = False
    for key, value in list(state.items()):
        if not (key.startswith("__phone_pending__") or key.startswith("__name_pending__")):
            continue
        if not isinstance(value, dict):
            state_logger.warning("legacy pending malformed: %s=%r", key, value)
            del state[key]
            changed = True
            continue
        raw_date = value.get("date")
        try:
            record_date = datetime.fromisoformat(raw_date).date()
        except (TypeError, ValueError):
            state_logger.warning("legacy pending invalid date: %s=%r", key, raw_date)
            del state[key]
            changed = True
            continue
        age_days = (today - record_date).days
        if age_days < COLLECTOR_PENDING_TTL_DAYS:
            continue
        state_logger.warning("legacy pending expired: %s client=%s age_days=%d", key, value.get("client"), age_days)
        del state[key]
        changed = True
    if changed:
        save_state(state)


class CrmStateLockError(RuntimeError):
    """Exclusive lock not acquired within timeout — caller must not proceed with save."""


@contextmanager
def _crm_state_lock(path: Path):
    """Exclusive cross-process lock for CRM state read-modify-write.

    Hard-fail policy: при недоступности lock поднимает CrmStateLockError.
    Callers-save должны вернуть False. Callers-load — только логировать и продолжать
    (startup tolerant).
    """
    lock_file = path.with_suffix(".lock")
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with portalocker.Lock(
            str(lock_file),
            timeout=5,
            check_interval=0.1,
            flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
        ):
            yield
    except (portalocker.LockException, PermissionError, OSError) as _le:
        raise CrmStateLockError(
            f"crm_state_lock_timeout: {path.name} — {_le}"
        ) from _le


def _crm_save_pending() -> bool:
    """Сохраняет _CRM_PHONE_PENDING на диск. Возвращает False при ошибке lock или записи."""
    try:
        with _crm_state_lock(CRM_PENDING_PATH):
            _crm_cleanup_pending()
            tmp = CRM_PENDING_PATH.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as _f:
                json.dump({str(k): v for k, v in _CRM_PHONE_PENDING.items()},
                          _f, ensure_ascii=False, indent=2)
            tmp.replace(CRM_PENDING_PATH)
            state_logger.debug("CRM pending saved: %d", len(_CRM_PHONE_PENDING))
            return True
    except CrmStateLockError as _le:
        state_logger.error("crm_state_lock_timeout: pending — %s", _le)
        return False
    except Exception as _e:
        state_logger.warning("_crm_save_pending error: %s", _e)
        return False


def _crm_load_pending() -> None:
    """Восстанавливает _CRM_PHONE_PENDING из файла при старте."""
    if not CRM_PENDING_PATH.exists():
        _cleanup_legacy_collector_pending_state()
        return
    try:
        with _crm_state_lock(CRM_PENDING_PATH):
            with open(CRM_PENDING_PATH, encoding="utf-8") as _f:
                data = json.load(_f)
            _CRM_PHONE_PENDING.update({int(k): v for k, v in data.items()})
            _crm_cleanup_pending()
            state_logger.info("CRM pending restored: %d managers", len(_CRM_PHONE_PENDING))
    except CrmStateLockError as _le:
        state_logger.error("crm_state_lock_timeout: pending load — %s", _le)
    except Exception as _e:
        state_logger.warning("_crm_load_pending error: %s", _e)
    _cleanup_legacy_collector_pending_state()

# Рассылка "чей клиент": token → {client_key, notified_chat_ids, claimed}
CRM_CLAIM_PENDING_PATH = LOGS_DIR / "crm_claim_pending_state.json"
CRM_CLAIM_TTL_HOURS = int(os.getenv("CRM_CLAIM_TTL_HOURS", "72"))
_CRM_CLAIM_PENDING: Dict[str, Dict[str, Any]] = {}
CRM_DUP_REVIEW_PATH = LOGS_DIR / "crm_duplicate_review_state.json"
CRM_DUP_REVIEW_TTL_HOURS = int(os.getenv("CRM_DUP_REVIEW_TTL_HOURS", "336"))
_CRM_DUP_REVIEW_PENDING: Dict[str, Dict[str, Any]] = {}
_CRM_DUP_REVIEW_AWAITING_TEXT: Dict[int, str] = {}
CRM_AMBIGUOUS_PATH = LOGS_DIR / "crm_ambiguous_conflicts.json"
_CRM_AMBIGUOUS: Dict[str, Dict[str, Any]] = {}  # key = stable signature

# Имя администратора — участвует в CRM наравне с менеджерами
ADMIN_NAME = "Вадим"


def _all_crm_participants() -> Dict[str, int]:
    """Все участники CRM: менеджеры + admin (Вадим)."""
    result = dict(MANAGERS_MAP or {})
    if ADMIN_CHAT_ID and ADMIN_NAME not in result:
        result[ADMIN_NAME] = ADMIN_CHAT_ID
    return result


def _crmdup_cleanup_pending(now_dt: Optional[datetime] = None) -> None:
    now_dt = now_dt or datetime.now(TZ)
    stale_tokens = []
    stale_chats = []
    for token, review in list(_CRM_DUP_REVIEW_PENDING.items()):
        created_raw = review.get("created_at")
        if not created_raw:
            stale_tokens.append(token)
            continue
        try:
            created_dt = datetime.fromisoformat(created_raw)
        except (TypeError, ValueError):
            stale_tokens.append(token)
            continue
        if (now_dt - created_dt).total_seconds() > CRM_DUP_REVIEW_TTL_HOURS * 3600:
            stale_tokens.append(token)
    for token in stale_tokens:
        _CRM_DUP_REVIEW_PENDING.pop(token, None)
    for chat_id, token in list(_CRM_DUP_REVIEW_AWAITING_TEXT.items()):
        if token not in _CRM_DUP_REVIEW_PENDING:
            stale_chats.append(chat_id)
    for chat_id in stale_chats:
        _CRM_DUP_REVIEW_AWAITING_TEXT.pop(chat_id, None)


def _crmdup_save_pending() -> bool:
    """Возвращает False при ошибке lock или записи."""
    try:
        CRM_DUP_REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _crm_state_lock(CRM_DUP_REVIEW_PATH):
            _crmdup_cleanup_pending()
            tmp = CRM_DUP_REVIEW_PATH.with_suffix(".tmp")
            payload = {
                "reviews": _CRM_DUP_REVIEW_PENDING,
                "awaiting_text": {str(k): v for k, v in _CRM_DUP_REVIEW_AWAITING_TEXT.items()},
            }
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, CRM_DUP_REVIEW_PATH)
            return True
    except CrmStateLockError as _le:
        state_logger.error("crm_state_lock_timeout: dup_review — %s", _le)
        return False
    except Exception as _e:
        state_logger.warning("_crmdup_save_pending error: %s", _e)
        return False


def _crmdup_load_pending() -> None:
    _CRM_DUP_REVIEW_PENDING.clear()
    _CRM_DUP_REVIEW_AWAITING_TEXT.clear()
    if not CRM_DUP_REVIEW_PATH.exists():
        return
    try:
        with _crm_state_lock(CRM_DUP_REVIEW_PATH):
            payload = json.loads(CRM_DUP_REVIEW_PATH.read_text(encoding="utf-8"))
            reviews = payload.get("reviews", {}) if isinstance(payload, dict) else {}
            awaiting_text = payload.get("awaiting_text", {}) if isinstance(payload, dict) else {}
            if isinstance(reviews, dict):
                _CRM_DUP_REVIEW_PENDING.update(reviews)
            if isinstance(awaiting_text, dict):
                _CRM_DUP_REVIEW_AWAITING_TEXT.update({int(k): v for k, v in awaiting_text.items()})
            _crmdup_cleanup_pending()
            state_logger.info("CRM duplicate review restored: %d records", len(_CRM_DUP_REVIEW_PENDING))
    except CrmStateLockError as _le:
        state_logger.error("crm_state_lock_timeout: dup_review load — %s", _le)
    except Exception as _e:
        state_logger.warning("_crmdup_load_pending error: %s", _e)


def _crmdup_token() -> str:
    from uuid import uuid4
    stamp = datetime.now(TZ).strftime("%Y%m%d%H%M%S")
    return f"dup_{stamp}_{uuid4().hex[:8]}"


# ── Ambiguous (multi-manager) CRM conflicts ───────────────────────────────────

def _ambiguous_signature(items: List[Dict[str, Any]]) -> str:
    """Stable signature: (client_key, normalized_phone) parts отсортированы.

    Order-independent + sensitive к изменению телефонов: если у тех же пар клиентов
    реально изменился набор конфликтных телефонов — signature будет другой, и
    F-13 reopen сработает автоматически (новый ключ в _CRM_AMBIGUOUS, новая
    pending-запись попадёт в admin backlog и broadcast).
    """
    def _norm_phone(raw: Any) -> str:
        if not raw:
            return ""
        return "".join(ch for ch in str(raw) if ch.isdigit())

    parts = []
    for item in items[:4]:
        key = item.get("client_key", "")
        phone = _norm_phone(item.get("phone", ""))
        parts.append(f"{key}#{phone}")
    return "|".join(sorted(parts))


def _ambiguous_token() -> str:
    from uuid import uuid4
    return "a" + uuid4().hex[:12]


def _crmdup_ambiguous_count() -> int:
    return sum(1 for v in _CRM_AMBIGUOUS.values() if v.get("status") == "pending")


def _crmdup_save_ambiguous() -> bool:
    """Возвращает False при ошибке lock или записи."""
    try:
        CRM_AMBIGUOUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _crm_state_lock(CRM_AMBIGUOUS_PATH):
            tmp = CRM_AMBIGUOUS_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(_CRM_AMBIGUOUS, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, CRM_AMBIGUOUS_PATH)
            return True
    except CrmStateLockError as _le:
        state_logger.error("crm_state_lock_timeout: ambiguous — %s", _le)
        return False
    except Exception as _e:
        state_logger.warning("_crmdup_save_ambiguous error: %s", _e)
        return False


def _crmdup_load_ambiguous() -> None:
    _CRM_AMBIGUOUS.clear()
    if not CRM_AMBIGUOUS_PATH.exists():
        return
    try:
        with _crm_state_lock(CRM_AMBIGUOUS_PATH):
            data = json.loads(CRM_AMBIGUOUS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _CRM_AMBIGUOUS.update(data)
            state_logger.info("CRM ambiguous conflicts restored: %d records", len(_CRM_AMBIGUOUS))
    except CrmStateLockError as _le:
        state_logger.error("crm_state_lock_timeout: ambiguous load — %s", _le)
    except Exception as _e:
        state_logger.warning("_crmdup_load_ambiguous error: %s", _e)


def _crmdup_queue_ambiguous(conflict: Dict[str, Any]) -> None:
    """Save multi-manager conflict to ambiguous queue. Idempotent: no duplicates."""
    items = conflict.get("items", [])
    if len(items) < 2:
        return
    sig = _ambiguous_signature(items)
    existing = _CRM_AMBIGUOUS.get(sig)
    if existing:
        # Don't re-queue if pending or already resolved
        return
    managers = sorted({(item.get("manager") or "").strip() for item in items if (item.get("manager") or "").strip()})
    _CRM_AMBIGUOUS[sig] = {
        "token": _ambiguous_token(),
        "signature": sig,
        "group_key": conflict.get("group_key", ""),
        "items": items[:4],
        "managers": managers,
        "added_at": datetime.now(TZ).isoformat(),
        "status": "pending",
    }
    if not _crmdup_save_ambiguous():
        crm_logger.error("CRM ambiguous queue: save failed for sig=%s (in-memory only)", sig[:50])
    else:
        crm_logger.info("CRM ambiguous queued: %s managers=%s", sig[:50], managers)


def _crmdup_try_finalize_ambiguous(sig: str, *, ok: bool, reviewer: str, resolution: str) -> bool:
    """Mark ambiguous conflict resolved only after successful CRM write."""
    if not ok:
        return False
    entry = _CRM_AMBIGUOUS.get(sig)
    if not isinstance(entry, dict):
        return False
    entry.update(
        {
            "status": "resolved",
            "resolved_at": datetime.now(TZ).isoformat(),
            "resolved_by": reviewer,
            "resolution": resolution,
        }
    )
    if not _crmdup_save_ambiguous():
        state_logger.error("crm_state_lock_timeout: finalize_ambiguous failed, rolling back in-memory")
        entry.update({"status": "pending"})
        entry.pop("resolved_at", None)
        entry.pop("resolved_by", None)
        entry.pop("resolution", None)
        return False
    crm_audit("ambiguous_conflict_resolved", reviewer=reviewer, signature=sig, resolution=resolution)
    return True


def _format_ambiguous_text(entry: Dict[str, Any]) -> str:
    items = entry.get("items", [])
    n_pending = _crmdup_ambiguous_count()
    lines = []
    for i, item in enumerate(items[:4], 1):
        mgr = item.get("manager") or "?"
        phone = item.get("phone") or "—"
        key = item.get("display_name") or item.get("client_key", "?")
        lines.append(
            f"{i}. <b>{_html.escape(key)}</b>  [{_html.escape(mgr)}]\n"
            f"   тел. <code>{_html.escape(phone)}</code>"
        )
    return (
        f"⚠️ <b>Спорный CRM-конфликт</b>  (в очереди: {n_pending})\n\n"
        + "\n".join(lines)
        + "\n\nЧья карточка правильная?"
    )


def _ambiguous_keyboard(entry: Dict[str, Any]) -> InlineKeyboardMarkup:
    token = entry["token"]
    managers = entry.get("managers", [])
    rows = []
    for idx, mgr in enumerate(managers):
        rows.append([InlineKeyboardButton(f"👤 Это {mgr}", callback_data=f"crm_ambi|a|{token}|{idx}")])
    rows.append([InlineKeyboardButton("🔀 Разные клиенты", callback_data=f"crm_ambi|d|{token}")])
    rows.append([InlineKeyboardButton("⏭ Следующий", callback_data=f"crm_ambi|s|{token}")])
    return InlineKeyboardMarkup(rows)


def _crmdup_choice_kb(token: str, items: List[Dict[str, Any]]) -> InlineKeyboardMarkup:
    rows = []
    for idx, item in enumerate(items[:2]):
        label = item.get("phone") or f"Вариант {idx + 1}"
        rows.append([InlineKeyboardButton(f"✅ Оставить {label}", callback_data=f"crmdup|pick|{token}|{idx}")])
    rows.append([InlineKeyboardButton("✏️ Ввести другой номер", callback_data=f"crmdup|custom|{token}")])
    rows.append([InlineKeyboardButton("↔️ Это разные клиенты", callback_data=f"crmdup|distinct|{token}")])
    return InlineKeyboardMarkup(rows)


def _crmdup_prompt_text(review: Dict[str, Any]) -> str:
    items = review.get("items", [])
    lines = []
    for idx, item in enumerate(items[:2], start=1):
        lines.append(
            f"{idx}. <b>{_html.escape(item.get('client_key', '?'))}</b>\n"
            f"Телефон: <code>{_html.escape(item.get('phone', '-'))}</code>\n"
            f"Источник: <code>{_html.escape(','.join(item.get('sources', [])) or '-')}</code>"
        )
    manager = review.get("manager") or "не назначен"
    return (
        f"📞 <b>Разовая CRM-сверка дублей</b>\n\n"
        f"Менеджер: <b>{_html.escape(manager)}</b>\n"
        f"Ниже две карточки, которые похожи на одного клиента, но в CRM у них разные номера.\n\n"
        f"{chr(10).join(lines)}\n\n"
        f"Выберите действующий номер, введите новый или отметьте, что это разные клиенты."
    )


async def _crmdup_broadcast_once(context: ContextTypes.DEFAULT_TYPE, limit: int = 100) -> Dict[str, int]:
    from bot.crm_clients import get_phone_conflict_groups

    conflicts = get_phone_conflict_groups(limit=limit)
    sent = 0
    skipped = 0
    for conflict in conflicts:
        manager = (conflict.get("manager") or "").strip()
        if not manager:
            _crmdup_queue_ambiguous(conflict)
            skipped += 1
            continue
        chat_id = _all_crm_participants().get(manager)
        items = conflict.get("items", [])
        if not chat_id or len(items) < 2:
            skipped += 1
            continue
        signature = "|".join(sorted(item.get("client_key", "") for item in items[:2]))
        if any(review.get("signature") == signature and not review.get("resolved_at") for review in _CRM_DUP_REVIEW_PENDING.values()):
            skipped += 1
            continue
        token = _crmdup_token()
        review = {
            "token": token,
            "signature": signature,
            "manager": manager,
            "chat_id": chat_id,
            "items": items[:2],
            "created_at": datetime.now(TZ).isoformat(),
        }
        await context.bot.send_message(
            chat_id=chat_id,
            text=_crmdup_prompt_text(review),
            parse_mode="HTML",
            reply_markup=_crmdup_choice_kb(token, review["items"]),
        )
        _CRM_DUP_REVIEW_PENDING[token] = review
        crm_audit("duplicate_phone_conflict_sent", manager=manager, token=token, client_keys=[item.get("client_key", "") for item in review["items"]])
        sent += 1
    if not _crmdup_save_pending():
        crm_logger.error("crm_state_lock_timeout: dup_review broadcast save failed (in-memory only)")
    crm_logger.info("CRM duplicate review broadcast: sent=%d skipped=%d", sent, skipped)
    return {"sent": sent, "skipped": skipped, "total": len(conflicts)}


def _crm_claim_is_stale(claim: Dict[str, Any], now_dt: Optional[datetime] = None) -> bool:
    """True если claim-токен невалиден: без created_at, нечитаемая дата или TTL истёк."""
    now_dt = now_dt or datetime.now(TZ)
    created_raw = claim.get("created_at")
    if not created_raw:
        return True
    try:
        created_dt = datetime.fromisoformat(created_raw)
    except (TypeError, ValueError):
        return True
    return (now_dt - created_dt).total_seconds() > CRM_CLAIM_TTL_HOURS * 3600


def _crm_cleanup_claim_pending(now_dt: Optional[datetime] = None) -> None:
    now_dt = now_dt or datetime.now(TZ)
    stale_tokens = [
        token for token, claim in list(_CRM_CLAIM_PENDING.items())
        if _crm_claim_is_stale(claim, now_dt)
    ]
    for token in stale_tokens:
        _CRM_CLAIM_PENDING.pop(token, None)


def _crm_save_claim_pending() -> bool:
    """Возвращает False при ошибке lock или записи."""
    try:
        CRM_CLAIM_PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _crm_state_lock(CRM_CLAIM_PENDING_PATH):
            _crm_cleanup_claim_pending()
            tmp = CRM_CLAIM_PENDING_PATH.with_suffix(".tmp")
            payload = {k: v for k, v in _CRM_CLAIM_PENDING.items()}
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, CRM_CLAIM_PENDING_PATH)
            return True
    except CrmStateLockError as _le:
        state_logger.error("crm_state_lock_timeout: claim — %s", _le)
        return False
    except Exception as _e:
        state_logger.warning("_crm_save_claim_pending error: %s", _e)
        return False


def _crm_load_claim_pending() -> None:
    _CRM_CLAIM_PENDING.clear()
    if not CRM_CLAIM_PENDING_PATH.exists():
        return
    try:
        with _crm_state_lock(CRM_CLAIM_PENDING_PATH):
            data = json.loads(CRM_CLAIM_PENDING_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _CRM_CLAIM_PENDING.update(data)
            _crm_cleanup_claim_pending()
            state_logger.info("CRM claim pending restored: %d records", len(_CRM_CLAIM_PENDING))
    except CrmStateLockError as _le:
        state_logger.error("crm_state_lock_timeout: claim load — %s", _le)
    except Exception as _e:
        state_logger.warning("_crm_load_claim_pending error: %s", _e)


def _crm_claim_token() -> str:
    from uuid import uuid4
    stamp = datetime.now(TZ).strftime("%Y%m%d%H%M%S")
    return f"claim_{stamp}_{uuid4().hex[:8]}"


def _crm_collect_unowned_claim_clients(limit: int = 3) -> List[str]:
    # F-06: используем полный фильтр из crm_clients + проверяем is_vendor/do_not_call
    from bot.crm_clients import (
        load_clients as _crm_load,
        canonicalize_client_key,
        is_service_client_name as _is_svc,
    )
    _crm_data = _crm_load()
    grouped: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for key, value in _crm_data.get("clients", {}).items():
        if not isinstance(value, dict):
            continue
        if value.get("is_vendor") or value.get("do_not_call"):
            continue
        grouped.setdefault(canonicalize_client_key(key), []).append((key, value))

    result: List[str] = []
    for _canon, items in grouped.items():
        owned = any((item.get("manager") or "") not in ("", "Не определён", "?", "-", "—") for _, item in items)
        if owned:
            continue
        candidate = next((k for k, _ in items if not _is_svc(k)), None)
        if candidate:
            result.append(candidate)
        if len(result) >= limit:
            break
    return result


async def cmd_guide(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет HTML-инструкцию. Саиде — её инструкцию, остальным — общую."""
    chat_id = update.effective_chat.id
    if not await _acl_gate(chat_id, context):
        return

    # Для Саиды отдаём отдельный простой документ
    is_saida = (chat_id == _get_saida_chat_id())
    if is_saida:
        guide_path = ROOT_DIR / "docs" / "Инструкция Саида.html"
        filename = "Инструкция Саида.html"
        caption = "📖 Инструкция для тебя — как отвечать боту"
    else:
        guide_path = ROOT_DIR / "docs" / "Инструкция по работе с ботом.html"
        filename = "Инструкция по работе с ботом.html"
        caption = "📖 Инструкция по работе с ботом Минбаракат"

    if not guide_path.exists():
        await context.bot.send_message(chat_id=chat_id, text="❌ Файл инструкции не найден.")
        return
    try:
        with open(guide_path, "rb") as f:
            await context.bot.send_document(
                chat_id=chat_id,
                document=f,
                filename=filename,
                caption=caption,
            )
    except Exception as e:
        logger.error("cmd_guide error: %s", e)
        await context.bot.send_message(chat_id=chat_id, text="❌ Ошибка отправки инструкции. Попробуйте позже.")


# Алиас: /help → /guide. Чтобы менеджер мог писать привычной командой.
cmd_help = cmd_guide


async def cmd_phone(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """CRM: /phone <имя клиента> <номер> — прямая запись телефона если имя совпадает точно."""
    chat_id = update.effective_chat.id
    manager = _chat_to_manager(chat_id)
    if not manager and not is_admin(chat_id):
        await _send_auto(context, chat_id, "⛔ Доступ запрещён.")
        return

    args = context.args or []
    if len(args) < 2:
        await _send_auto(
            context, chat_id,
            "ℹ️ Использование: /phone <Имя клиента> <номер>\n\n"
            "Пример: /phone ТОО Асем +77771234567",
        )
        return

    import re as _re
    phone_raw = args[-1].strip()
    client_name = " ".join(args[:-1]).strip()

    phone_digits = _re.sub(r"\D", "", phone_raw)
    if _re.fullmatch(r"8\d{10}", phone_digits):
        phone_digits = "7" + phone_digits[1:]
    if not _re.fullmatch(r"7\d{10}", phone_digits):
        await _send_auto(
            context, chat_id,
            "⚠️ Неверный формат номера.\n"
            "Допустимо: <code>+7XXXXXXXXXX</code> / <code>7XXXXXXXXXX</code> / <code>8XXXXXXXXXX</code>",
            parse_mode="HTML",
        )
        return
    phone = "+" + phone_digits

    try:
        from bot.crm_clients import set_client_phone as _set_phone
        if _set_phone(client_name, phone, manager or "", phone_source="manager_command"):
            await _send_auto(
                context, chat_id,
                f"✅ Телефон <b>{phone}</b> записан для <b>{client_name}</b>.",
                parse_mode="HTML",
            )
            log_event("crm_phone_set", client=client_name, phone=phone, manager=manager)
        else:
            await _send_auto(
                context, chat_id,
                f"⚠️ Клиент <b>{client_name}</b> не найден.\n"
                f"Имя должно совпадать с названием в отчёте 1С.",
                parse_mode="HTML",
            )
    except Exception as e:
        logger.error("cmd_phone error: %s", e)
        await _send_auto(context, chat_id, f"❌ Ошибка: {e}")


def _set_client_phone_wrapper(client_name: str, phone: str, manager: str,
                               alias: str = "") -> bool:
    """Обёртка для set_client_phone без async."""
    from bot.crm_clients import set_client_phone as _set_phone
    return _set_phone(client_name, phone, manager, alias=alias, phone_source="manager_manual")


async def cmd_crmdupsend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin-only: one-off broadcast of CRM duplicate phone conflicts to managers."""
    chat_id = update.effective_chat.id
    if not is_admin(chat_id):
        await _send_auto(context, chat_id, "⛔ Доступно только администратору.")
        return
    try:
        result = await _crmdup_broadcast_once(context, limit=100)
        await _send_auto(
            context,
            chat_id,
            (
                "📞 Разовая CRM-сверка дублей запущена.\n\n"
                f"Отправлено менеджерам: <b>{result['sent']}</b>\n"
                f"Пропущено: <b>{result['skipped']}</b>\n"
                f"Всего конфликтов найдено: <b>{result['total']}</b>"
            ),
            parse_mode="HTML",
        )
    except Exception as e:
        crm_logger.error("cmd_crmdupsend error: %s", e)
        await _send_auto(context, chat_id, "❌ Не удалось запустить разовую CRM-сверку.")


def _chat_to_manager(chat_id: int) -> str:
    """Возвращает имя менеджера по chat_id или пустую строку."""
    for name, cid in (MANAGERS_MAP or {}).items():
        if cid == chat_id:
            return name
    return ""


async def _crm_save_phone_and_continue(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    pending: Dict[str, Any],
    phone: str,
    phone_source: str,
) -> None:
    from bot.crm_clients import set_client_details as _set_details

    client_key = pending.get("client_key", "")
    display_name = pending.get("display_name", "")
    original_name = pending.get("original_name", client_key)
    name_mode = pending.get("name_mode") or ("manual" if display_name else "later")
    name_review_needed = pending.get("name_review_needed")
    if name_review_needed is None:
        name_review_needed = not bool(display_name)

    ok = _set_details(
        client_key,
        display_name=display_name,
        phone=phone,
        original_name=original_name,
        name_mode=name_mode,
        name_review_needed=name_review_needed,
        phone_source=phone_source,
    )
    log_event(
        "crm_details_set",
        client=client_key,
        display_name=display_name,
        phone=phone,
        original_name=original_name,
        name_mode=name_mode,
        phone_source=phone_source,
        address="-",
    )

    manager_name = pending.get("manager", "")
    daily_limit = pending.get("daily_limit", CRM_DAILY_LIMIT)
    label = display_name or original_name

    if not ok:
        # F-04: при ошибке сохранения — оставляем клиента в очереди, не двигаем цепочку
        await context.bot.send_message(
            chat_id=chat_id,
            text="⚠️ Не удалось сохранить: клиент не найден в CRM. Введите номер ещё раз.",
        )
        return

    done_today = pending.get("done_today", 0) + 1
    old_pending = dict(pending)
    _CRM_PHONE_PENDING.pop(chat_id, None)
    if not _crm_save_pending():
        _CRM_PHONE_PENDING[chat_id] = old_pending
        await context.bot.send_message(
            chat_id=chat_id,
            text="⚠️ Временная ошибка сохранения, попробуйте ещё раз.",
        )
        return

    await context.bot.send_message(
        chat_id=chat_id,
        text=(
            f"✅ Сохранено: <b>{label}</b> · {phone}\n"
            f"Адрес можно заполнить позже."
        ),
        parse_mode="HTML",
    )

    if done_today < daily_limit:
        from bot.crm_clients import get_clients_without_phones as _crm_next
        next_list = _crm_next(manager_name, limit=1)
        remaining = len(_crm_next(manager_name, limit=500))
        if next_list:
            next_key = next_list[0]
            _next_has_prefix = _crm_manager_from_prefix(next_key) is not None
            _next_now_iso = datetime.now(TZ).isoformat()
            _CRM_PHONE_PENDING[chat_id] = {
                "state": "clarify_phone" if _next_has_prefix else "clarify_name",
                "client_key": next_key,
                "original_name": next_key,
                "display_name": next_key[2:] if _next_has_prefix else "",
                "name_mode": "system" if _next_has_prefix else None,
                "name_review_needed": False if _next_has_prefix else True,
                "done_today": done_today,
                "daily_limit": daily_limit,
                "manager": manager_name,
                "total_no_phone": remaining,
                "created_at": _next_now_iso,
                "last_sent": _next_now_iso,
            }
            if not _crm_save_pending():
                _CRM_PHONE_PENDING.pop(chat_id, None)
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="⚠️ Следующий клиент не сохранён в очереди. Откройте CRM снова.",
                )
                return
            if _next_has_prefix:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=_crm_phone_prompt_text(next_key),
                    parse_mode="HTML",
                    reply_markup=_crm_phone_choice_kb(next_key) or _crm_phone_help_only_kb(),
                )
            else:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=_crm_name_prompt_text(
                        client_key=next_key,
                        done_today=done_today,
                        total=remaining,
                        daily_limit=daily_limit,
                    ),
                    parse_mode="HTML",
                    reply_markup=_crm_name_choice_kb(),
                )
            return
        await context.bot.send_message(
            chat_id=chat_id,
            text="🎉 Все клиенты внесены! База полностью заполнена.",
            parse_mode="HTML",
        )
        return

    from bot.crm_clients import get_clients_without_phones as _crm_remain
    remaining = len(_crm_remain(manager_name, limit=500))
    if remaining:
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"✅ На сегодня готово — внесено {done_today} клиентов.\n\n"
                f"📋 Осталось без телефона: <b>{remaining}</b>\n"
                f"Завтра в 18:00 бот пришлёт ещё {min(daily_limit, remaining)}."
            ),
            parse_mode="HTML",
        )
    else:
        await context.bot.send_message(
            chat_id=chat_id,
            text="🎉 Все клиенты внесены! База полностью заполнена.",
        )


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
    json_file = find_recent_json_for_manager(manager, report_type="DEBT")
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
    new_trace_id()  # каждый callback получает свой trace_id для корреляции логов
    q = update.callback_query
    try:
        await q.answer()
    except Exception as e:
        # httpx.ConnectTimeout может пробивать telegram.error wrapper
        logger.warning("cb_data q.answer() error: %s: %s", type(e).__name__, e)
    data = q.data or ""
    chat_id = q.message.chat.id

    if get_user_role(chat_id) == "unknown":
        return

    # Выходной / рабочий день — ответ администратора
    if data.startswith("workday|"):
        if not is_admin(chat_id):
            return
        from bot.workday_checker import set_holiday, clear_holiday
        today = datetime.now(TZ).strftime("%Y-%m-%d")
        choice = data.split("|", 1)[1]
        if choice == "holiday":
            set_holiday(today)
            await q.edit_message_text(
                f"✅ Сегодня ({today}) — выходной.\n"
                f"Уведомления менеджерам не отправляются."
            )
            log_event("workday_set_holiday", date=today)
        else:
            clear_holiday(today)
            await q.edit_message_text(
                f"✅ Сегодня ({today}) — рабочий день.\n"
                f"Все уведомления активны."
            )
            log_event("workday_set_workday", date=today)
        return

    # Стоп-лист отгрузки (debt_stop_control)
    if data.startswith("dstop_") and _DEBT_STOP_AVAILABLE:
        try:
            result_text = await _dstop_callback(data, chat_id, context.bot)
            if result_text:
                try:
                    await q.edit_message_text(result_text, parse_mode="HTML")
                except Exception:
                    await q.answer("Принято.", show_alert=True)
        except Exception as e:
            logger.error("dstop callback error: %s", e)
        return

    if data.startswith("payhold_"):
        try:
            from collector.payment_hold import confirm_by_saida, get_request
            parts = data.split("|", 1)
            action = parts[0]
            token = parts[1] if len(parts) > 1 else ""

            # payhold_help_full — кнопка-подсказка для Саиды, токен не нужен
            if action == "payhold_help_full":
                if chat_id != _get_saida_chat_id():
                    await q.answer("Только для Саиды.")
                    return
                await q.answer()
                try:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=_SAIDA_PAYHOLD_HELP_TEXT,
                        parse_mode="HTML",
                    )
                except Exception as _he:
                    logger.warning("payhold_help_full send failed: %s", _he)
                return

            rec = get_request(token)
            if not rec:
                await q.answer("Запрос не найден или устарел.")
                return

            saida_chat_id = _get_saida_chat_id()
            admin_id = ADMIN_CHAT_ID

            if action == "payhold_req":
                if chat_id != int(rec.get("manager_chat_id") or 0):
                    await q.answer("Это запрос другого менеджера.")
                    return
                kb = InlineKeyboardMarkup([
                    [InlineKeyboardButton("✅ Да, оплата есть", callback_data=f"payhold_full|{token}")],
                    [InlineKeyboardButton("🔸 Частично",        callback_data=f"payhold_partial|{token}")],
                    [InlineKeyboardButton("❌ Не вижу оплаты",  callback_data=f"payhold_none|{token}")],
                    [InlineKeyboardButton("❓ Не понимаю",      callback_data=f"payhold_help|{token}")],
                ])
                if not saida_chat_id:
                    await q.answer("Не найден chat_id Саиды.")
                    return
                client_name = rec.get("client", "")
                manager_name = rec.get("manager", "")
                debt_str = rec.get("debt_str") or str(rec.get("debt", ""))
                await context.bot.send_message(
                    chat_id=saida_chat_id,
                    text=(
                        f"💬 <b>Запрос на проверку оплаты</b>\n\n"
                        f"Менеджер <b>{manager_name}</b> сообщил, что клиент оплатил.\n\n"
                        f"Клиент: <b>{client_name}</b>\n"
                        f"Долг в отчёте: {debt_str}\n\n"
                        f"Проверь в 1С и нажми нужную кнопку.\n"
                        f"<i>Ответь в течение 4 часов.</i>"
                    ),
                    reply_markup=kb,
                    parse_mode="HTML",
                )
                await q.answer("Запрос Саиде отправлен.")
                return

            if action == "payhold_help":
                if chat_id != saida_chat_id:
                    await q.answer("Только для Саиды.")
                    return
                await context.bot.send_message(
                    chat_id=saida_chat_id,
                    text=(
                        "❓ <b>Что означают кнопки:</b>\n\n"
                        "✅ <b>Да, оплата есть</b> — деньги пришли полностью.\n"
                        "   Клиент снимается со стопа автоматически. Менеджер и директор узнают.\n\n"
                        "🔸 <b>Частично</b> — пришла не вся сумма.\n"
                        "   Директор получит уведомление и решит сам.\n\n"
                        "❌ <b>Не вижу оплаты</b> — ничего не поступало.\n"
                        "   Клиент остаётся в стопе.\n\n"
                        "⏰ Если не ответишь в течение 8 часов — уведомления уйдут клиентам "
                        "автоматически, а менеджеры и директор узнают об этом."
                    ),
                    parse_mode="HTML",
                )
                return

            # Директор решает за Саиду (байпас после SAIDA_BYPASS_HOURS)
            if action in ("payhold_admin_full", "payhold_admin_none"):
                if chat_id != admin_id:
                    await q.answer("Только для директора.")
                    return
                status = "full" if action == "payhold_admin_full" else "none"
                updated = confirm_by_saida(token, status)
                if not updated:
                    await q.answer("Запрос уже закрыт.")
                    return
                client = updated.get("client", "")
                manager = updated.get("manager", "")
                mgr_cid = int(updated.get("manager_chat_id") or 0)
                result_label = "принял оплату" if status == "full" else "отклонил оплату"
                if mgr_cid:
                    try:
                        await context.bot.send_message(
                            chat_id=mgr_cid,
                            text=f"✅ Директор {result_label} по клиенту <b>{client}</b>.",
                            parse_mode="HTML",
                        )
                    except Exception:
                        pass
                try:
                    await q.edit_message_text(
                        f"Директор {result_label} по клиенту {client}. Менеджер уведомлён."
                    )
                except Exception:
                    await q.answer("Сохранено.")
                return

            if chat_id != saida_chat_id:
                await q.answer("Подтверждать может только Саида.")
                return
            status = {
                "payhold_full": "full",
                "payhold_partial": "partial",
                "payhold_none": "none",
            }.get(action)
            updated = confirm_by_saida(token, status or "")
            if not updated:
                await q.answer("Не удалось сохранить ответ.")
                return
            client = updated.get("client", "")
            manager = updated.get("manager", "")
            manager_chat_id = int(updated.get("manager_chat_id") or 0)
            if status in ("full", "partial"):
                status_label = "полная оплата" if status == "full" else "частичная оплата"
                answer_text = (
                    "✅ <b>Ответ сохранён</b>\n\n"
                    f"Клиент: <b>{client}</b>\n"
                    f"Статус: <b>{status_label}</b>\n\n"
                    "Менеджер и директор уведомлены.\n"
                    "Дальше ждём разноску в 1С."
                )
                notify_text = (
                    f"Саида подтвердила оплату по клиенту:\n\n"
                    f"{client}\n"
                    f"Менеджер: {manager}\n"
                    f"Статус: {'оплата есть' if status == 'full' else 'частичная оплата'}\n\n"
                    f"Клиент временно не будет попадать под давление до обновления 1С."
                )
            else:
                answer_text = (
                    "✅ <b>Ответ сохранён</b>\n\n"
                    f"Клиент: <b>{client}</b>\n"
                    "Статус: <b>оплаты нет</b>\n\n"
                    "Менеджер и директор уведомлены.\n"
                    "Клиент остаётся в обычной дебиторке."
                )
                notify_text = (
                    f"Саида не видит оплату по клиенту:\n\n"
                    f"{client}\n"
                    f"Менеджер: {manager}\n\n"
                    f"Клиент остаётся в обычной дебиторке."
                )
            for target in {manager_chat_id, admin_id}:
                if target:
                    try:
                        await context.bot.send_message(chat_id=target, text=notify_text, parse_mode=None)
                    except Exception as send_exc:
                        logger.warning("payhold notify error target=%s: %s", target, send_exc)
            logger.info(
                "payhold saved: client=%s manager=%s status=%s manager_chat_id=%s token=%s",
                client,
                manager,
                status,
                manager_chat_id,
                token,
            )
            try:
                await q.edit_message_text(answer_text, parse_mode="HTML")
            except Exception:
                await q.answer("Ответ сохранён.", show_alert=True)
            return
        except Exception as e:
            logger.error("payment hold callback error: %s", e, exc_info=True)
            await q.answer("Ошибка обработки оплаты.")
            return

    # Collector dialog callbacks
    if data.startswith("col_"):
        try:
            from collector.manager_dialog import handle_callback as _col_cb
            handled = await _col_cb(data, chat_id, q.message.message_id)
            if handled:
                return
        except Exception as e:
            logger.error("collector callback error: %s", e)
        return

    # Shipment control callbacks (cdlg_ship|action|phone_key)
    if data.startswith("cdlg_ship|"):
        await q.answer("Решения по отгрузке теперь принимает руководитель через стоп-лист.")
        return

    # WhatsApp approval flow callbacks (wa_appr_mgr_* / wa_appr_cli_* / wa_appr_adm_*)
    if data.startswith("wa_appr_"):
        try:
            await q.answer()
        except Exception:
            pass
        try:
            from collector.approval_flow import handle_callback as _wa_appr_cb
            handled = await _wa_appr_cb(data, chat_id, q.message.message_id)
            if handled:
                return
        except Exception as e:
            logger.error("wa_appr callback error: %s", e)
        return

    # No-movement Saida-first check callbacks (nm_paid/nm_nopay/nm_adm_send/nm_adm_skip)
    if data.startswith("nm_"):
        try:
            from collector.no_movement import handle_nm_callback as _nm_cb
            handled = await _nm_cb(data, chat_id)
            if handled:
                try:
                    await q.answer()
                except Exception:
                    pass
                return
        except Exception as e:
            logger.error("nm callback error: %s", e)
        return

    # [DISABLED v9.4.39] Старый flow: коллектор → reg_lang/reg_name/reg_phone.
    # Заменён на CRM 18:00 + /phone команда (crm_psel|).
    # Оставлен закомментированным на случай отката.
    #
    # if data in ("reg_lang_ru", "reg_lang_kz"):
    #     try:
    #         from collector.collections_db import get_phone_pending
    #         from collector.registry_manager import update_client_language
    #         pending_client = get_phone_pending(chat_id)
    #         if pending_client:
    #             lang = "ru" if data == "reg_lang_ru" else "kz"
    #             lang_label = "🇷🇺 Русский" if lang == "ru" else "🇰🇿 Қазақша"
    #             if update_client_language(pending_client, lang):
    #                 await context.bot.send_message(chat_id=chat_id,
    #                     text=f"✅ Язык сохранён: <b>{lang_label}</b>\nКлиент: <b>{pending_client}</b>",
    #                     parse_mode="HTML")
    #             else:
    #                 await context.bot.send_message(chat_id=chat_id, text="⚠️ Не удалось сохранить язык.")
    #         else:
    #             await context.bot.send_message(chat_id=chat_id, text="⚠️ Запрос устарел.")
    #     except Exception as e:
    #         logger.error("reg_lang callback error: %s", e)
    #     return
    #
    # if data == "reg_name":
    #     try:
    #         from collector.collections_db import get_name_pending, set_name_pending
    #         pending_client = get_name_pending(chat_id)
    #         if pending_client:
    #             await context.bot.send_message(chat_id=chat_id,
    #                 text=(f"✏️ <b>Введите правильное имя</b> для клиента:\n<b>{pending_client}</b>\n\n"
    #                       f"Имя из 1С останется как ключ для матчинга.\n"
    #                       f"Введённое имя будет использоваться в сообщениях должнику."),
    #                 parse_mode="HTML")
    #         else:
    #             await context.bot.send_message(chat_id=chat_id, text="⚠️ Запрос на исправление имени устарел.")
    #     except Exception as e:
    #         logger.error("reg_name callback error: %s", e)
    #     return
    #
    # if data == "reg_phone":
    #     try:
    #         from collector.collections_db import get_phone_pending
    #         pending_client = get_phone_pending(chat_id)
    #         if pending_client:
    #             await context.bot.send_message(chat_id=chat_id,
    #                 text=(f"📞 <b>Введите номер WhatsApp</b> для клиента:\n<b>{pending_client}</b>\n\n"
    #                       f"Принимается любой формат:\n• +77XXXXXXXXXX\n• 77XXXXXXXXXX\n• 87XXXXXXXXXX\n\n"
    #                       f"🔴 <b>Проверьте номер дважды!</b>"),
    #                 parse_mode="HTML")
    #         else:
    #             await context.bot.send_message(chat_id=chat_id, text="⚠️ Запрос на ввод телефона устарел.")
    #     except Exception as e:
    #         logger.error("reg_phone callback error: %s", e)
    #     return

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
            await _send_auto(context, chat_id, message, parse_mode="HTML")
            await q.answer("✅ Статистика отправлена")
        except Exception as e:
            logger.error(f"Ошибка в show_stats: {e}", exc_info=True)
            await q.answer(f"❌ Ошибка: {e}")
        return

    if data == "dev_feedback_open":
        _DEV_FEEDBACK_ARMED[chat_id] = datetime.now(TZ).isoformat()
        _dev_feedback_save()
        await q.answer("Следующим сообщением отправьте текст, скрин или файл.")
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=_dev_feedback_open_text(),
                parse_mode="HTML",
                reply_markup=_dev_feedback_menu_kb(),
            )
        except Exception as exc:
            logger.warning("dev_feedback_open send failed chat_id=%s: %s", chat_id, exc)
        return

    if data == "dev_commands":
        if chat_id != _get_developer_chat_id():
            await q.answer("⛔ Доступ запрещён")
            return
        await q.answer()
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=_developer_command_help_text(),
                parse_mode="HTML",
                reply_markup=_dev_feedback_menu_kb(),
            )
        except Exception as exc:
            logger.warning("dev_commands send failed chat_id=%s: %s", chat_id, exc)
        return

    # Коллектор: статус активного батча
    if data == "collector_batch":
        if user_role != "admin":
            await q.answer("⛔ Доступ запрещён")
            return
        await q.answer()
        text = _format_collector_batch_text()
        kb_back = _collector_batch_keyboard()
        await hide_main_menu(context, chat_id)
        try:
            msg = await context.bot.send_message(
                chat_id=chat_id, text=text, reply_markup=kb_back, parse_mode="HTML"
            )
            _menu_set(chat_id, msg.message_id)
        except Exception as _e:
            logger.error("collector_batch send error: %s", _e)
        return

    if data == "collector_actual_batch":
        if user_role != "admin":
            await q.answer("⛔ Доступ запрещён")
            return
        mode = _get_actual_collector_batch_mode()
        if mode == "pending_admin":
            await q.answer("Открываю батч на утверждение...")
            try:
                from collector.approval_flow import send_admin_summary
                batch = _get_pending_admin_batch()
                if not batch:
                    await _send_auto(context, chat_id, "⚠️ Актуальный батч уже изменился. Обновите экран.")
                    return
                await send_admin_summary(batch, context.bot)
            except Exception as _e:
                logger.error("collector_actual_batch pending_admin error: %s", _e)
                await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Ошибка: {_e}")
            return
        if mode == "send_ready":
            await q.answer("Открываю готовый список...")
            try:
                from collector.approval_flow import (
                    get_latest_send_ready_batch,
                    _format_admin_summary_text,
                    _admin_send_now_keyboard,
                )
                batch = get_latest_send_ready_batch()
                if not batch:
                    await _send_auto(context, chat_id, "⚠️ Нет готового списка для отправки.")
                    return
                batch_id = str(batch.get("batch_id") or "—")
                text = (
                    f"📤 <b>Готовый список ждёт отправки</b>\n"
                    f"Batch: <code>{batch_id}</code>\n\n"
                    f"{_format_admin_summary_text(batch)}"
                )
                await _send_auto(
                    context,
                    chat_id,
                    text,
                    parse_mode="HTML",
                    reply_markup=InlineKeyboardMarkup(_admin_send_now_keyboard(batch_id)["inline_keyboard"]),
                )
            except Exception as _e:
                logger.error("collector_actual_batch send_ready error: %s", _e, exc_info=True)
                await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Ошибка: {_e}")
            return
        await q.answer("Показываю текущий статус...")
        text = _format_collector_batch_text()
        kb_back = _collector_batch_keyboard()
        await hide_main_menu(context, chat_id)
        try:
            msg = await context.bot.send_message(
                chat_id=chat_id, text=text, reply_markup=kb_back, parse_mode="HTML"
            )
            _menu_set(chat_id, msg.message_id)
        except Exception as _e:
            logger.error("collector_actual_batch status error: %s", _e)
        return

    if data == "collector_resend_approval":
        if user_role != "admin":
            await q.answer("⛔ Доступ запрещён")
            return
        batch = _get_pending_admin_batch()
        if not batch:
            await q.answer("Нет батча, ожидающего утверждения")
            return
        await q.answer("Отправляю сводку для утверждения...")
        try:
            from collector.approval_flow import send_admin_summary
            await send_admin_summary(batch, context.bot)
        except Exception as _e:
            logger.error("collector_resend_approval error: %s", _e)
            await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Ошибка: {_e}")
        return

    if data == "collector_send_latest":
        if user_role != "admin":
            await q.answer("⛔ Доступ запрещён")
            return
        await q.answer("Открываю готовый список...")
        try:
            from collector.approval_flow import (
                get_latest_send_ready_batch,
                _format_admin_summary_text,
                _admin_send_now_keyboard,
            )
            batch = get_latest_send_ready_batch()
            if not batch:
                await _send_auto(context, chat_id, "⚠️ Нет готового списка для отправки.")
                return
            batch_id = str(batch.get("batch_id") or "—")
            text = (
                f"📤 <b>Готовый список ждёт отправки</b>\n"
                f"Batch: <code>{batch_id}</code>\n\n"
                f"{_format_admin_summary_text(batch)}"
            )
            await _send_auto(
                context,
                chat_id,
                text,
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(_admin_send_now_keyboard(batch_id)["inline_keyboard"]),
            )
        except Exception as _e:
            logger.error("collector_send_latest error: %s", _e, exc_info=True)
            await _send_auto(context, chat_id, "❌ Не удалось открыть готовый список.")
        return

    if data == "collector_agreed_stats":
        if user_role != "admin":
            await q.answer("⛔ Доступ запрещён")
            return
        await q.answer()
        text = _format_collector_agreed_stats_text()
        kb_back = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Обновить", callback_data="collector_agreed_stats")],
            [InlineKeyboardButton("↩️ К батчу", callback_data="collector_batch")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="back_main")],
        ])
        await hide_main_menu(context, chat_id)
        try:
            msg = await context.bot.send_message(
                chat_id=chat_id, text=text, reply_markup=kb_back, parse_mode="HTML"
            )
            _menu_set(chat_id, msg.message_id)
        except Exception as _e:
            logger.error("collector_agreed_stats send error: %s", _e)
        return

    if data == "collector_saida_stats":
        if user_role != "admin":
            await q.answer("⛔ Доступ запрещён")
            return
        await q.answer()
        text = _format_collector_saida_stats_text()
        kb_back = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Обновить", callback_data="collector_saida_stats")],
            [InlineKeyboardButton("↩️ К батчу", callback_data="collector_batch")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="back_main")],
        ])
        await hide_main_menu(context, chat_id)
        try:
            msg = await context.bot.send_message(
                chat_id=chat_id, text=text, reply_markup=kb_back, parse_mode="HTML"
            )
            _menu_set(chat_id, msg.message_id)
        except Exception as _e:
            logger.error("collector_saida_stats send error: %s", _e)
        return

    if data == "collector_partial_stats":
        if user_role != "admin":
            await q.answer("⛔ Доступ запрещён")
            return
        await q.answer()
        text = _format_collector_partial_stats_text()
        kb_back = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Обновить", callback_data="collector_partial_stats")],
            [InlineKeyboardButton("↩️ К батчу", callback_data="collector_batch")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="back_main")],
        ])
        await hide_main_menu(context, chat_id)
        try:
            msg = await context.bot.send_message(
                chat_id=chat_id, text=text, reply_markup=kb_back, parse_mode="HTML"
            )
            _menu_set(chat_id, msg.message_id)
        except Exception as _e:
            logger.error("collector_partial_stats send error: %s", _e)
        return

    if data == "collector_deferral_stats":
        if user_role != "admin":
            await q.answer("⛔ Доступ запрещён")
            return
        await q.answer()
        text = _format_collector_deferral_stats_text()
        kb_back = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔄 Обновить", callback_data="collector_deferral_stats")],
            [InlineKeyboardButton("↩️ К батчу", callback_data="collector_batch")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="back_main")],
        ])
        await hide_main_menu(context, chat_id)
        try:
            msg = await context.bot.send_message(
                chat_id=chat_id, text=text, reply_markup=kb_back, parse_mode="HTML"
            )
            _menu_set(chat_id, msg.message_id)
        except Exception as _e:
            logger.error("collector_deferral_stats send error: %s", _e)
        return

    # 🆕 v9.4.9: Аналитика
    if data.startswith("analytics|"):
        await handle_analytics(update, context, data)
        return

    if data.startswith("crmdup|"):
        parts = data.split("|")
        action = parts[1] if len(parts) > 1 else ""
        token = parts[2] if len(parts) > 2 else ""
        review = _CRM_DUP_REVIEW_PENDING.get(token)
        if not review:
            await q.answer("Запрос сверки устарел.")
            return
        if review.get("chat_id") != chat_id and not is_admin(chat_id):
            await q.answer("Это не ваш запрос.")
            return

        items = review.get("items", [])
        client_keys = [item.get("client_key", "") for item in items]
        reviewer = _chat_to_manager(chat_id) or ("Вадим" if is_admin(chat_id) else "")

        if action == "pick":
            try:
                idx = int(parts[3]) if len(parts) > 3 else -1
            except ValueError:
                idx = -1
            if idx < 0 or idx >= len(items):
                await q.answer("Вариант уже недоступен.")
                return
            chosen = items[idx]
            try:
                from bot.crm_clients import resolve_phone_conflict
                ok = resolve_phone_conflict(
                    client_keys=client_keys,
                    chosen_phone=chosen.get("phone", ""),
                    chosen_key=chosen.get("client_key", ""),
                    reviewer=reviewer,
                )
            except Exception as e:
                crm_logger.error("crmdup pick resolve error: %s", e)
                ok = False
            if not ok:
                await q.answer("Не удалось сохранить решение.")
                return
            review["resolved_at"] = datetime.now(TZ).isoformat()
            review["resolution"] = "pick"
            review["chosen_phone"] = chosen.get("phone", "")
            _CRM_DUP_REVIEW_AWAITING_TEXT.pop(chat_id, None)
            if not _crmdup_save_pending():
                review.pop("resolved_at", None)
                review.pop("resolution", None)
                review.pop("chosen_phone", None)
                _CRM_DUP_REVIEW_AWAITING_TEXT[chat_id] = token
                await q.answer("⚠️ Временная ошибка сохранения, попробуйте ещё раз.", show_alert=True)
                return
            await q.answer("Сохранено.")
            try:
                await q.message.edit_text(
                    (
                        "✅ <b>CRM-сверка закрыта</b>\n\n"
                        f"Выбран номер: <code>{_html.escape(chosen.get('phone', ''))}</code>\n"
                        f"Карточки объединены в CRM, alias сохранены."
                    ),
                    parse_mode="HTML",
                    reply_markup=None,
                )
            except Exception:
                pass
            return

        if action == "custom":
            _CRM_DUP_REVIEW_AWAITING_TEXT[chat_id] = token
            if not _crmdup_save_pending():
                _CRM_DUP_REVIEW_AWAITING_TEXT.pop(chat_id, None)
                await q.answer("⚠️ Временная ошибка сохранения, попробуйте ещё раз.", show_alert=True)
                return
            await q.answer("Жду новый номер.")
            await context.bot.send_message(
                chat_id=chat_id,
                text="Введите действующий номер WhatsApp:\n<code>+7XXXXXXXXXX</code>",
                parse_mode="HTML",
            )
            return

        if action == "distinct":
            try:
                from bot.crm_clients import mark_phone_conflict_distinct
                ok = mark_phone_conflict_distinct(client_keys=client_keys, reviewer=reviewer)
            except Exception as e:
                crm_logger.error("crmdup distinct error: %s", e)
                ok = False
            if not ok:
                await q.answer("Не удалось отметить различие.")
                return
            review["resolved_at"] = datetime.now(TZ).isoformat()
            review["resolution"] = "distinct"
            _CRM_DUP_REVIEW_AWAITING_TEXT.pop(chat_id, None)
            if not _crmdup_save_pending():
                review.pop("resolved_at", None)
                review.pop("resolution", None)
                _CRM_DUP_REVIEW_AWAITING_TEXT[chat_id] = token
                await q.answer("⚠️ Временная ошибка сохранения, попробуйте ещё раз.", show_alert=True)
                return
            await q.answer("Отмечено.")
            try:
                await q.message.edit_text(
                    (
                        "↔️ <b>CRM-сверка закрыта</b>\n\n"
                        "Пара отмечена как разные клиенты. "
                        "Автосверка больше не будет поднимать этот конфликт."
                    ),
                    parse_mode="HTML",
                    reply_markup=None,
                )
            except Exception:
                pass
            return

        await q.answer("Неизвестное действие CRM duplicate review.")
        return

    # ── CRM backlog: единый экран по всем CRM-очередям ────────────────────────
    if data == "crm_backlog":
        if not is_admin(chat_id):
            await q.answer("Только для администратора.")
            return
        await q.answer()
        text = _format_crm_backlog_text()
        kb   = _crm_backlog_keyboard()
        try:
            await q.edit_message_text(text=text, parse_mode="HTML", reply_markup=kb)
        except Exception as _e:
            crm_logger.error("crm_backlog show error: %s", _e)
            await context.bot.send_message(
                chat_id=chat_id, text=text, parse_mode="HTML", reply_markup=kb
            )
        return

    # ── Ambiguous (multi-manager) CRM conflict queue ──────────────────────────
    if data == "crm_ambiguous_queue":
        if not is_admin(chat_id):
            await q.answer("Только для администратора.")
            return
        pending = [v for v in _CRM_AMBIGUOUS.values() if v.get("status") == "pending"]
        if not pending:
            await q.answer("Спорных конфликтов нет.", show_alert=True)
            return
        entry = pending[0]
        try:
            await q.edit_message_text(
                text=_format_ambiguous_text(entry),
                parse_mode="HTML",
                reply_markup=_ambiguous_keyboard(entry),
            )
        except Exception as _e:
            crm_logger.error("crm_ambiguous_queue show error: %s", _e)
            await q.answer("Ошибка показа очереди.")
        return

    if data.startswith("crm_ambi|"):
        if not is_admin(chat_id):
            await q.answer("Только для администратора.")
            return
        parts = data.split("|")
        action = parts[1] if len(parts) > 1 else ""
        token  = parts[2] if len(parts) > 2 else ""
        entry  = next((v for v in _CRM_AMBIGUOUS.values() if v.get("token") == token), None)
        if not entry:
            await q.answer("Конфликт уже решён или не найден.")
            return
        sig   = entry["signature"]
        items = entry.get("items", [])
        client_keys = [item["client_key"] for item in items]
        reviewer = ADMIN_NAME

        result_text: Optional[str] = None

        if action == "a":  # assign to manager by index
            try:
                mgr_idx = int(parts[3]) if len(parts) > 3 else 0
            except ValueError:
                mgr_idx = 0
            managers = entry.get("managers", [])
            chosen_manager = managers[mgr_idx] if mgr_idx < len(managers) else ""
            chosen_item = next(
                (it for it in items if (it.get("manager") or "").strip() == chosen_manager),
                items[0],
            )
            try:
                from bot.crm_clients import resolve_phone_conflict
                ok = resolve_phone_conflict(
                    client_keys=client_keys,
                    chosen_phone=chosen_item.get("phone", ""),
                    chosen_key=chosen_item.get("client_key", ""),
                    reviewer=reviewer,
                    phone_source="ambiguous_conflict_admin_resolve",
                )
            except Exception as _e:
                crm_logger.error("crm_ambi assign error: %s", _e)
                ok = False
            _crmdup_try_finalize_ambiguous(
                sig,
                ok=ok,
                reviewer=reviewer,
                resolution=f"assigned:{chosen_manager}",
            )
            result_text = (
                f"✅ Назначено: клиент {_html.escape(chosen_manager)}."
                if ok else
                "⚠️ Запись в CRM не удалась. Конфликт оставлен в очереди."
            )

        elif action == "d":  # distinct
            try:
                from bot.crm_clients import mark_phone_conflict_distinct
                ok = mark_phone_conflict_distinct(client_keys=client_keys, reviewer=reviewer)
            except Exception as _e:
                crm_logger.error("crm_ambi distinct error: %s", _e)
                ok = False
            _crmdup_try_finalize_ambiguous(
                sig,
                ok=ok,
                reviewer=reviewer,
                resolution="distinct",
            )
            result_text = "✅ Отмечены как разные клиенты." if ok else "⚠️ Запись в CRM не удалась. Конфликт оставлен в очереди."

        elif action == "s":  # skip — просто перейти к следующему
            result_text = None

        else:
            await q.answer("Неизвестное действие.")
            return

        pending = [v for v in _CRM_AMBIGUOUS.values() if v.get("status") == "pending"]
        if not pending:
            final = (result_text + "\n\n" if result_text else "") + "✅ <b>Очередь CRM-конфликтов пуста.</b>"
            try:
                await q.edit_message_text(final, parse_mode="HTML", reply_markup=None)
            except Exception:
                await q.answer("Очередь пуста.")
            return

        next_entry = pending[0]
        prefix = result_text + "\n\n" if result_text else ""
        try:
            await q.edit_message_text(
                text=prefix + _format_ambiguous_text(next_entry),
                parse_mode="HTML",
                reply_markup=_ambiguous_keyboard(next_entry),
            )
        except Exception as _e:
            crm_logger.error("crm_ambi show next error: %s", _e)
            await q.answer("Следующий конфликт.")
        return
    # ── end ambiguous queue ───────────────────────────────────────────────────

    if data == "crm_help":
        pending = _CRM_PHONE_PENDING.get(chat_id)
        if not pending or not str(pending.get("state", "")).startswith("clarify_"):
            await q.answer("CRM-запрос не найден.")
            return
        await q.answer("Готовлю подсказку...")
        state = pending.get("state", "")
        client_key = pending.get("client_key", "?")
        buttons = (
            ["Ввести имя", "Оставить как в системе", "Позже"]
            if state == "clarify_name"
            else ["Записать найденный номер", "Указать другой номер"]
        )
        try:
            from collector.manager_help import build_manager_help
            help_text = await build_manager_help(
                area="CRM-заполнение базы",
                manager=pending.get("manager") or _chat_to_manager(chat_id) or "",
                client=client_key,
                state=state,
                buttons=buttons,
                context={
                    "done_today": pending.get("done_today"),
                    "daily_limit": pending.get("daily_limit"),
                    "total_no_phone": pending.get("total_no_phone"),
                    "remind_count": pending.get("remind_count", 0),
                },
            )
        except Exception as e:
            logger.warning("crm_help error: %s", e)
            help_text = _html.escape(
                "Нужно закрыть текущий CRM-запрос. Если не ответить, бот будет "
                "напоминать каждые 30 минут и передаст руководителю. "
                "Я вижу игнор. Каждый день без ответа фиксируется — "
                "руководитель получит рекомендацию задержать зарплату на столько же дней."
            )
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❓ <b>Подсказка по CRM-запросу</b>\n\n{help_text}",
            parse_mode="HTML",
        )
        return

    if data.startswith("crm_name|"):
        pending = _CRM_PHONE_PENDING.get(chat_id)
        if not pending or pending.get("state") != "clarify_name":
            await q.answer("CRM-запрос устарел.")
            return
        action = data.split("|", 1)[1]
        client_key = pending.get("client_key", "?")
        pending.setdefault("original_name", client_key)
        pending["last_sent"] = datetime.now(TZ).isoformat()
        pending.pop("awaiting_name_text", None)
        try:
            await q.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass
        if action == "edit":
            pending["awaiting_name_text"] = True
            if not _crm_save_pending():
                pending.pop("awaiting_name_text", None)
                await q.answer("⚠️ Временная ошибка сохранения, попробуйте ещё раз.", show_alert=True)
                return
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"✏️ Введите удобное имя для клиента:\n\n"
                    f"<b>{client_key}</b>\n\n"
                    f"Это имя будет использоваться в CRM и сообщениях."
                ),
                parse_mode="HTML",
            )
            return
        if action == "keep":
            pending["display_name"] = client_key[2:] if _crm_manager_from_prefix(client_key) else client_key
            pending["name_mode"] = "system"
            pending["name_review_needed"] = False
            pending["state"] = "clarify_phone"
            if not _crm_save_pending():
                await q.answer("⚠️ Временная ошибка сохранения, попробуйте ещё раз.", show_alert=True)
                return
            await context.bot.send_message(
                chat_id=chat_id,
                text=_crm_phone_prompt_text(client_key),
                parse_mode="HTML",
                reply_markup=_crm_phone_choice_kb(client_key) or _crm_phone_help_only_kb(),
            )
            return
        if action == "later":
            pending["name_mode"] = "later"
            pending["name_review_needed"] = True
            pending["state"] = "clarify_phone"
            pending["paused_until"] = (datetime.now(TZ) + timedelta(hours=24)).isoformat()
            if not _crm_save_pending():
                await q.answer("⚠️ Временная ошибка сохранения, попробуйте ещё раз.", show_alert=True)
                return
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    "Имя можно заполнить позже.\n\n"
                    f"{_crm_phone_prompt_text(client_key)}"
                ),
                parse_mode="HTML",
                reply_markup=_crm_phone_choice_kb(client_key) or _crm_phone_help_only_kb(),
            )
            return
        await q.answer("Неизвестное действие CRM")
        return

    if data.startswith("crm_phone|"):
        pending = _CRM_PHONE_PENDING.get(chat_id)
        if not pending or pending.get("state") != "clarify_phone":
            await q.answer("CRM-запрос устарел.")
            return
        parts = data.split("|")
        action = parts[1] if len(parts) > 1 else ""
        client_key = pending.get("client_key", "?")
        # F-09: если кнопка содержит token — верифицируем что pending не сменился
        cb_token = parts[-1] if len(parts) >= 4 or (action == "edit" and len(parts) == 3) else ""
        if cb_token and len(cb_token) == 8:
            expected = _crm_key_token(pending.get("client_key", ""))
            if cb_token != expected:
                await q.answer("Устаревшая кнопка — в очереди уже другой клиент.")
                return
        pending["last_sent"] = datetime.now(TZ).isoformat()
        try:
            await q.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass
        if action == "edit":
            if not _crm_save_pending():
                await q.answer("⚠️ Временная ошибка сохранения, попробуйте ещё раз.", show_alert=True)
                return
            await context.bot.send_message(
                chat_id=chat_id,
                text="Введите другой телефон WhatsApp:\n<code>+7XXXXXXXXXX</code>",
                parse_mode="HTML",
            )
            return
        if action == "suggest":
            suggestions = _crm_phone_suggestions(client_key)
            try:
                idx = int(parts[2]) if len(parts) > 2 else 0
            except ValueError:
                idx = -1
            if idx < 0 or idx >= len(suggestions):
                await q.answer("Номер уже недоступен. Введите другой номер.")
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="Введите телефон WhatsApp:\n<code>+7XXXXXXXXXX</code>",
                    parse_mode="HTML",
                )
                return
            await _crm_save_phone_and_continue(
                context,
                chat_id,
                pending,
                suggestions[idx],
                phone_source="client_name_confirmed",
            )
            await q.answer("Номер записан.")
            return
        await q.answer("Неизвестное действие CRM")
        return

    # v9.4.27: Принудительная отправка отчёта
    if data == "menu_notify":
        kb = kb_notify_menu(user_role)
        text_notify = "🔔 *Уведомления сейчас*\n\nВыберите отчёт для немедленной отправки:"
        if hasattr(q, "message") and q.message:
            try:
                await _safe_edit_text(q.message, text_notify, reply_markup=kb, parse_mode="Markdown")
            except Exception:
                await send_notify_menu(context, chat_id, user_role, text=text_notify)
        else:
            await send_notify_menu(context, chat_id, user_role, text=text_notify)
        return

    if data.startswith("force|"):
        report_type = data.split("|", 1)[1]
        await q.answer("⏳ Формирую отчёт...")
        status = await force_report_to_user(report_type, chat_id, context)
        try:
            await q.answer(status, show_alert=False)
        except Exception:
            pass
        # v9.4.33: Возвращаем в раздел УВЕДОМЛЕНИЙ (не в аналитику и не в главное)
        await send_notify_menu(context, chat_id, user_role)
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

    if data == "show_help_doc":
        # Отправляет HTML-инструкцию (то же что /help). Используется в подсказках.
        await q.answer()
        try:
            await cmd_guide(update, context)
        except Exception as _e:
            logger.warning("show_help_doc failed: %s", _e)
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="📖 Инструкция временно недоступна. Попробуйте команду /help.",
                )
            except Exception:
                pass
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
            await send_main_menu(context, chat_id, user_role)
            return
        if user_role == "manager":
            manager_name = get_my_manager_name(chat_id) or "Unknown"
            await _safe_edit_text(q.message, f"🤖 ИИ анализ запущен...", reply_markup=None)
            await handle_ai_only(manager_name, chat_id, context, user_role, scopes, report_type)
            await send_main_menu(context, chat_id, user_role)
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
        await send_main_menu(context, chat_id, user_role)
        return
    # ─────────────────────────────────────────────────────────────────

    if data.startswith("ai_only|"):
        manager = data.split("|")[1]
        if manager not in scopes:
            await q.answer("Нет доступа к этому менеджеру")
            return
        await handle_ai_only(manager, chat_id, context, user_role, scopes)
        await send_main_menu(context, chat_id, user_role)
        return

    if data.startswith("extended_ai|"):
        manager = data.split("|")[1]
        if manager not in scopes:
            await q.answer("Нет доступа к этому менеджеру")
            return
        await handle_extended_with_ai(manager, chat_id, context, user_role, scopes)
        await send_main_menu(context, chat_id, user_role)
        return

    if data == "back_submenu":
        await send_main_menu(context, chat_id, user_role, text="📋 Выберите раздел:")
        return

    # ── Еженедельные клиенты: Алена предлагает → Вадим подтверждает ──────────
    # weekly_suggest|<client_name>  — Алена нажала "Да, исключить"
    # weekly_reject|<client_name>   — Алена нажала "Нет"
    # weekly_confirm|<client_name>  — Вадим подтвердил
    # weekly_deny|<client_name>     — Вадим отклонил
    if data.startswith("weekly_suggest|"):
        token = data.split("|", 1)[1]
        client_name = _weekly_token_get(token)
        if not client_name:
            await q.answer("Запрос устарел — перезапустите бот")
            return
        admin_chat_id = ADMIN_CHAT_ID
        if not admin_chat_id:
            await q.answer("Ошибка: admin chat_id не настроен")
            return
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Подтвердить", callback_data=f"weekly_confirm|{token}"),
            InlineKeyboardButton("❌ Отклонить",   callback_data=f"weekly_deny|{token}"),
        ]])
        try:
            await context.bot.send_message(
                chat_id=admin_chat_id,
                text=(
                    f"📋 <b>Алена</b> предлагает добавить еженедельного клиента:\n\n"
                    f"<b>{client_name}</b>\n\n"
                    f"Такие клиенты исключаются из ПРОСРОЧКИ (7-9 дн).\n"
                    f"Подтвердить?"
                ),
                parse_mode="HTML",
                reply_markup=kb,
            )
            await q.answer("✅ Запрос отправлен Вадиму")
            await q.message.edit_text(
                f"⏳ Запрос на исключение <b>{client_name}</b> отправлен администратору.",
                parse_mode="HTML",
            )
        except Exception as e:
            logger.error("weekly_suggest send error: %s", e)
            await q.answer("❌ Ошибка отправки")
        return

    if data.startswith("weekly_reject|"):
        token = data.split("|", 1)[1]
        client_name = _weekly_token_get(token) or token
        await q.answer("Понятно, клиент остаётся в общем списке")
        try:
            await q.message.edit_text(f"❌ {client_name} — оставлен в общем списке.", parse_mode="HTML")
        except Exception:
            pass
        return

    if data.startswith("weekly_confirm|"):
        if user_role != "admin":
            await q.answer("Только администратор может подтверждать")
            return
        token = data.split("|", 1)[1]
        client_name = _weekly_token_get(token)
        if not client_name:
            await q.answer("Запрос устарел — перезапустите бот")
            return
        try:
            clients = _load_weekly_clients()
            if client_name not in clients:
                clients.append(client_name)
                _save_weekly_clients(clients)
            await q.answer("✅ Клиент добавлен в еженедельные")
            await q.message.edit_text(
                f"✅ <b>{client_name}</b> добавлен в список еженедельных клиентов.\n"
                f"Он не будет попадать в ПРОСРОЧКУ (7-9 дн).",
                parse_mode="HTML",
            )
            logger.info("weekly_clients: добавлен %s", client_name)
        except Exception as e:
            logger.error("weekly_confirm error: %s", e)
            await q.answer("❌ Ошибка сохранения")
        return

    if data.startswith("weekly_deny|"):
        if user_role != "admin":
            await q.answer("?????? ????????????? ????? ?????????")
            return
        token = data.split("|", 1)[1]
        client_name = _weekly_token_get(token)
        if not client_name:
            await q.answer("?????? ??????? ? ????????????? ???")
            return
        await q.answer("?????????")
        try:
            await q.message.edit_text(
                f"? ?????? ?? ?????????? <b>{client_name}</b> ????????.",
                parse_mode="HTML",
            )
        except Exception:
            pass
        return

    # ?? CRM: "??? ??????" ? ????????/admin ???????? ?????????? ??????? ??????
    if data.startswith("crm_claim|"):
        token = data.split("|", 1)[1]
        claim = _CRM_CLAIM_PENDING.get(token)
        # F-16: stale-токен (без created_at или просроченный) — снять кнопку и ответить как устаревший
        if claim and not claim.get("claimed") and _crm_claim_is_stale(claim):
            _CRM_CLAIM_PENDING.pop(token, None)
            if not _crm_save_claim_pending():
                crm_logger.error("crm_claim: stale cleanup save failed (in-memory only)")
            try:
                await q.message.edit_reply_markup(reply_markup=None)
            except Exception:
                pass
            claim = None
        if not claim:
            await q.answer("?????? ??????? ??? ??? ?????????.")
            return
        if claim.get("claimed"):
            await q.answer("???? ?????? ??? ???? ?????? ??????????.")
            try:
                await q.message.edit_reply_markup(reply_markup=None)
            except Exception:
                pass
            return

        claimer_chat_id = q.message.chat.id
        claimer_name = None
        for mgr, mid in _all_crm_participants().items():
            if mid == claimer_chat_id:
                claimer_name = mgr
                break
        if not claimer_name:
            await q.answer("?? ??????? ?????????? ?????????.")
            return

        client_key = claim["client_key"]
        claim["claimed"] = True
        claim["claimed_by"] = claimer_name
        claim["claimed_at"] = datetime.now(TZ).isoformat()
        if not _crm_save_claim_pending():
            claim["claimed"] = False
            claim.pop("claimed_by", None)
            claim.pop("claimed_at", None)
            await q.answer("⚠️ Временная ошибка сохранения, попробуйте ещё раз.", show_alert=True)
            return

        _crm_write_ok = False
        try:
            from bot.crm_clients import load_clients as _cc_load, save_clients as _cc_save, canonicalize_client_key
            _cc_data = _cc_load()
            _cc_clients = _cc_data.get("clients", {})
            _claim_canon = canonicalize_client_key(client_key)
            _updated_keys = []
            for _existing_key, _existing_value in list(_cc_clients.items()):
                if canonicalize_client_key(_existing_key) != _claim_canon:
                    continue
                if not isinstance(_existing_value, dict):
                    _existing_value = {}
                _existing_value["manager"] = claimer_name
                _existing_value["claimed_at"] = datetime.now(TZ).isoformat()
                _cc_clients[_existing_key] = _existing_value
                _updated_keys.append(_existing_key)
            _cc_data["clients"] = _cc_clients
            _cc_save(_cc_data)
            crm_logger.info("CRM claim: %s -> manager %s (aliases=%d)", client_key, claimer_name, len(_updated_keys))
            crm_audit("claim_taken", client_key=client_key, claimer=claimer_name, aliases=_updated_keys, token=token)
            _crm_write_ok = True
        except Exception as _e:
            crm_logger.error("crm_claim save error: %s", _e)
            # F-07: откат claim state — клиент снова доступен для broadcast
            claim["claimed"] = False
            claim.pop("claimed_by", None)
            claim.pop("claimed_at", None)
            if not _crm_save_claim_pending():
                crm_logger.error("crm_claim: rollback save also failed — in-memory rollback only")

        if not _crm_write_ok:
            await q.answer("⚠️ Не удалось сохранить — попробуйте ещё раз.")
            return

        await q.answer("✅ Взяли!")
        try:
            await q.message.edit_text(
                f"? <b>{client_key}</b>\n????: <b>{claimer_name}</b>",
                parse_mode="HTML",
                reply_markup=None,
            )
        except Exception:
            pass

        for other_chat_id in claim.get("notified", []):
            if other_chat_id == claimer_chat_id:
                continue
            try:
                await context.bot.send_message(
                    chat_id=other_chat_id,
                    text=f"?? <b>{client_key}</b> ? ???? {claimer_name}.",
                    parse_mode="HTML",
                )
            except Exception:
                pass

        try:
            from bot.crm_clients import get_clients_without_phones as _crm_next2
            remaining = len(_crm_next2(claimer_name, limit=500))
            _claimer_now_iso = datetime.now(TZ).isoformat()
            _CRM_PHONE_PENDING[claimer_chat_id] = {
                "state": "clarify_name",
                "client_key": client_key,
                "original_name": client_key,
                "done_today": 0,
                "daily_limit": 1,
                "manager": claimer_name,
                "total_no_phone": remaining,
                "created_at": _claimer_now_iso,
                "last_sent": _claimer_now_iso,
            }
            if not _crm_save_pending():
                _CRM_PHONE_PENDING.pop(claimer_chat_id, None)
                await context.bot.send_message(
                    chat_id=claimer_chat_id,
                    text="⚠️ Временная ошибка сохранения, откройте CRM снова.",
                )
                return
            crm_audit("claim_phone_chain_started", client_key=client_key, claimer=claimer_name, remaining=remaining)
            await context.bot.send_message(
                chat_id=claimer_chat_id,
                text=_crm_name_prompt_text(
                    client_key=client_key,
                    done_today=0,
                    total=remaining,
                    daily_limit=1,
                ),
                parse_mode="HTML",
                reply_markup=_crm_name_choice_kb(),
            )
        except Exception as _e:
            crm_logger.warning("crm_claim -> phone chain error: %s", _e)
        return
    # ─────────────────────────────────────────────────────────────────────────

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

    # Восстанавливаем CRM-очередь сбора телефонов
    _crm_load_pending()
    _crm_load_claim_pending()
    _crmdup_load_pending()
    _dev_feedback_load()
    _crmdup_load_ambiguous()

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
                f"· 10:00 — обещания коллектора\n"
                f"· 10:30 — срыв договорённостей\n"
                f"· 14:00 — контроль отгрузки / авто-стоп{_oploss_line}\n"
                f"· 16:30 — стоп-лист менеджерам\n"
                f"· 17:00 — коллектор (резерв)\n"
                f"· 18:00 — CRM: база + телефоны\n"
                f"· 18:30 — эскалация стоп-листа\n"
                f"· 20:00 — валовая\n"
                f"· 21:00 — продажи\n"
                f"· 22:00 — аналитика\n"
                f"· 22:15 — стоп-лист Саиде\n"
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
            if manager in _SYSTEM_ACCOUNTS or chat_id == ADMIN_CHAT_ID:
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
            log_event("analytics_startup_trigger", reason="Tuesday, no fresh analytics found")
            await weekly_analytics_job(app)


# ═══════════════════════════════════════════════════════════════
# 🆕 v9.4.9: БЛОК АНАЛИТИКИ
# ═══════════════════════════════════════════════════════════════

def _check_net_profit_negative(analytics_dir: Path) -> Optional[str]:
    """
    C1: Парсит последний net_profit_day HTML.
    Возвращает алёрт если чистая прибыль отрицательная, иначе None.
    """
    import re as _re
    try:
        search = analytics_dir / "net_profit_day"
        if not search.exists():
            return None
        files = sorted(search.glob("net_profit*.html"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return None
        raw   = files[0].read_text(encoding="utf-8")
        clean = _re.sub(r'<style[^>]*>.*?</style>', ' ', raw, flags=_re.S)
        clean = _re.sub(r'<script[^>]*>.*?</script>', ' ', clean, flags=_re.S)
        clean = _re.sub(r'<[^>]+>', ' ', clean)
        clean = _re.sub(r'\s+', ' ', clean).strip()
        # Ищем строку «Чистая прибыль  -123 456 ₸»
        m = _re.search(r'Чистая прибыль\s+([\-\−\d\s\u202f,]+₸)', clean, _re.I)
        if not m:
            return None
        val_str = m.group(1).strip()
        if not (val_str.startswith('-') or val_str.startswith('\u2212')):
            return None  # прибыль положительная — всё ок
        dm      = _re.search(r'(\d{2}\.\d{2}\.\d{4})', clean)
        date_s  = dm.group(1) if dm else "?"
        return (
            f"🚨 *ВНИМАНИЕ: Чистая прибыль в минусе*\n\n"
            f"📅 За день: {date_s}\n"
            f"❌ {val_str}\n\n"
            f"Проверьте расходы, валовую прибыль, затраты."
        )
    except Exception:
        return None


def _build_manager_ranking(json_dir: Path, analytics_dir: Path) -> Optional[str]:
    """
    C9: Строит таблицу-рейтинг менеджеров по трём показателям:
      • Продажи (выручка)  — из sales_*.json
      • Дебиторка (долг)   — из debt_ext_*.json
      • DSO (оценка)       — долг / (выручка / 30)
    Возвращает строку для Telegram или None если данных нет.
    """
    import re as _re

    # ── 1. Продажи по менеджерам ──────────────────────────────────────────────
    sales_by_mgr: dict = {}  # name → revenue
    sales_clients: dict = {}  # name → count
    best_sales_date = None
    best_sales_period = ""
    try:
        files = sorted(json_dir.glob("sales_*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        # Находим лучший период
        from datetime import date as _date
        for path in files:
            if "товар" in path.name.lower():
                continue
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
            mgr = (d.get("manager") or "").strip()
            if mgr in ("", "Не определён", "Неизвестно") or len(mgr) < 2:
                continue
            ps = (d.get("period") or "").strip()
            if not ps:
                continue
            # Простой парсер: берём последнюю дату из строки периода
            dm2 = _re.findall(r'(\d{1,2})[./](\d{1,2})[./](\d{4})', ps)
            if dm2:
                dd, mm, yyyy = dm2[-1]
                try:
                    pd = _date(int(yyyy), int(mm), int(dd))
                    if best_sales_date is None or pd > best_sales_date:
                        best_sales_date   = pd
                        best_sales_period = ps
                except Exception:
                    pass
        # Загружаем данные за лучший период
        if best_sales_period:
            for path in files:
                if "товар" in path.name.lower():
                    continue
                with open(path, encoding="utf-8") as fh:
                    d = json.load(fh)
                mgr = (d.get("manager") or "").strip()
                if mgr in ("", "Не определён", "Неизвестно") or len(mgr) < 2:
                    continue
                if (d.get("period") or "").strip() != best_sales_period:
                    # Нечёткое сравнение по дате
                    ps2 = (d.get("period") or "").strip()
                    dm3 = _re.findall(r'(\d{1,2})[./](\d{1,2})[./](\d{4})', ps2)
                    if not dm3:
                        continue
                    dd2, mm2, yyyy2 = dm3[-1]
                    try:
                        pd2 = _date(int(yyyy2), int(mm2), int(dd2))
                        if pd2 != best_sales_date:
                            continue
                    except Exception:
                        continue
                rev = float(d.get("total_revenue", 0))
                cnt = len(d.get("clients", []))
                sales_by_mgr[mgr]  = sales_by_mgr.get(mgr, 0.0) + rev
                sales_clients[mgr] = sales_clients.get(mgr, 0) + cnt
    except Exception:
        pass

    # ── 2. Дебиторка по менеджерам ────────────────────────────────────────────
    debt_by_mgr: dict = {}   # name → closing debt
    debt_date = ""
    debt_by_mgr, debt_date = _load_fresh_debt_totals_by_manager(json_dir)
    try:
        if debt_by_mgr:
            dfiles = []
        else:
            dfiles = sorted(json_dir.glob("debt_ext_*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        # Находим свежий period_max
        best_d = None
        best_dstr = ""
        for path in dfiles:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
            pmax = (d.get("period_max") or "").strip()
            dm2 = _re.findall(r'(\d{1,2})[./](\d{1,2})[./](\d{4})', pmax)
            if dm2:
                dd, mm, yyyy = dm2[-1]
                try:
                    from datetime import date as _date2
                    pd = _date2(int(yyyy), int(mm), int(dd))
                    if best_d is None or pd > best_d:
                        best_d    = pd
                        best_dstr = pmax
                except Exception:
                    pass
        # Суммируем долг каждого менеджера
        for path in dfiles:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
            pmax = (d.get("period_max") or "").strip()
            if pmax != best_dstr:
                continue
            mgr = (d.get("manager") or "").strip()
            if not mgr or mgr in ("Не определён", "Неизвестно") or len(mgr) < 2:
                continue
            agg_close = float((d.get("aggregates") or {}).get("close", 0) or 0)
            if agg_close <= 0:
                continue  # пропускаем кредитовые/сводные записи
            debt_by_mgr[mgr] = debt_by_mgr.get(mgr, 0.0) + agg_close
            debt_date = best_dstr
    except Exception:
        pass

    if not sales_by_mgr and not debt_by_mgr:
        return None

    # ── 3. Форматирование таблицы ─────────────────────────────────────────────
    all_mgrs = sorted(
        set(list(sales_by_mgr.keys()) + list(debt_by_mgr.keys()))
    )
    # Сортируем по выручке desc
    all_mgrs.sort(key=lambda n: sales_by_mgr.get(n, 0), reverse=True)

    SEP    = "─" * 34
    lines  = ["📊 *РЕЙТИНГ МЕНЕДЖЕРОВ*"]
    if best_sales_period:
        lines.append(f"🛒 Продажи: {best_sales_period}")
    if debt_date:
        lines.append(f"💳 Долг на: {debt_date}")
    lines.append(SEP)

    medals = ["🥇", "🥈", "🥉"]

    def _fmt(v: float) -> str:
        if v >= 1_000_000:
            return f"{v/1_000_000:.1f}М"
        if v >= 1_000:
            return f"{v/1_000:.0f}К"
        return f"{v:.0f}"

    for i, mgr in enumerate(all_mgrs):
        rev   = sales_by_mgr.get(mgr, 0.0)
        debt  = debt_by_mgr.get(mgr, 0.0)
        cnt   = sales_clients.get(mgr, 0)
        medal = medals[i] if i < 3 else "  "
        # DSO ≈ долг / (выручка / 30)
        dso_s = ""
        if rev > 0 and debt > 0:
            dso = debt / (rev / 30)
            dso_s = f"  Срок≈{dso:.0f}д"
        rev_s  = f"💰{_fmt(rev)}" if rev > 0 else "—"
        debt_s = f"💳{_fmt(debt)}" if debt > 0 else "—"
        lines.append(f"{medal} {mgr:<9}  {rev_s}  {debt_s}{dso_s}")
        if cnt > 0:
            lines[-1] += f"  ({cnt}кл)"

    lines.append(SEP)
    total_rev  = sum(sales_by_mgr.values())
    total_debt = sum(debt_by_mgr.values())
    if total_rev > 0:
        lines.append(f"ИТОГО:  💰{_fmt(total_rev)}  💳{_fmt(total_debt)}")

    return "\n".join(lines)


async def weekly_analytics_job(context):
    """v9.4.26: Генерация аналитических отчётов + уведомление admin/subadmin + alert если нет expenses"""
    from bot.workday_checker import is_holiday_today
    if is_holiday_today():
        logger.info("weekly_analytics_job: выходной — пропуск")
        return
    log_event("weekly_analytics_start")
    scripts = [
        "sales_profitability_report.py",
        "net_profit_report.py",
        "dead_stock_report.py",
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

                # C1: Alert если чистая прибыль отрицательная (по данным последнего отчёта)
                np_neg_alert = _check_net_profit_negative(ANALYTICS_DIR)
                if np_neg_alert:
                    try:
                        await bot.send_message(ADMIN_CHAT_ID, np_neg_alert, parse_mode="Markdown")
                        log_event("net_profit_negative_alert_sent")
                    except Exception as _e:
                        log_event("net_profit_negative_alert_error", error=str(_e), level="WARNING")

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
    """Строит клавиатуру меню аналитики — только HTML-отчёты, без force-кнопок."""
    if user_role == "admin":
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Продажи+Рентабельность", callback_data="analytics|sales_profit")],
            [InlineKeyboardButton("💰 Чистая прибыль",         callback_data="analytics|net_profit_submenu")],
            [InlineKeyboardButton("📦 Мёртвый запас",          callback_data="analytics|turnover")],
            [InlineKeyboardButton("👥 Активность клиентов",     callback_data="analytics|rfm")],
            [InlineKeyboardButton("🎯 Концентрация выручки",   callback_data="analytics|concentration")],
            [InlineKeyboardButton("💳 Сроки оплаты",            callback_data="analytics|dso")],
            [InlineKeyboardButton("🤖 ИИ анализ",             callback_data="ai_type_menu")],
            [InlineKeyboardButton("🔄 Обновить аналитику",     callback_data="analytics|refresh")],
            [InlineKeyboardButton("🔙 Главное меню",           callback_data="back_main")],
        ])
    elif user_role == "subadmin":
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("💳 Сроки оплаты", callback_data="analytics|dso")],
            [InlineKeyboardButton("🔙 Главное меню", callback_data="back_main")],
        ])
    elif user_role == "manager":
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 Продажи+Рентабельность", callback_data="analytics|sales_profit")],
            [InlineKeyboardButton("👥 Активность клиентов",     callback_data="analytics|rfm")],
            [InlineKeyboardButton("🎯 Концентрация",           callback_data="analytics|concentration")],
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

    if report_type == "net_profit_mtd":
        all_files = [p for p in all_files if _net_profit_mtd_is_deliverable(p)]
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


async def crm_phone_reminder_task(context: ContextTypes.DEFAULT_TYPE):
    """
    Каждый час: напоминает менеджерам у кого висит незаполненный клиент в CRM-очереди.
    Работает только в рабочие часы 09–19 в рабочие дни. Повторяет до получения ответа.
    """
    from bot.workday_checker import is_holiday_today
    if is_holiday_today():
        crm_logger.info("crm_phone_reminder_task: выходной день — пропуск")
        return
    _crm_cleanup_pending()
    if not _CRM_PHONE_PENDING:
        return
    now = datetime.now(TZ)
    if not (9 <= now.hour < 19):
        return
    for chat_id, pending in list(_CRM_PHONE_PENDING.items()):
        client_key  = pending.get("client_key", "?")
        state       = pending.get("state", "clarify_name")
        done_today  = pending.get("done_today", 0)
        total       = pending.get("total_no_phone", 0)
        daily_limit = pending.get("daily_limit", CRM_DAILY_LIMIT)
        paused_until_raw = pending.get("paused_until")
        if paused_until_raw:
            try:
                paused_until = datetime.fromisoformat(paused_until_raw)
                if paused_until.tzinfo is None:
                    paused_until = paused_until.replace(tzinfo=TZ)
                if now < paused_until:
                    continue
            except (TypeError, ValueError):
                pass

        if state == "clarify_name":
            prompt = None
        elif state == "clarify_phone":
            prompt = _crm_phone_prompt_text(client_key)
        else:
            prompt = "Введите адрес торговой точки"

        try:
            remind_count = int(pending.get("remind_count", 0) or 0) + 1
            if state == "clarify_name":
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=_crm_name_prompt_text(
                        client_key=client_key,
                        done_today=done_today,
                        total=total,
                        daily_limit=daily_limit,
                        reminder=True,
                        remind_count=remind_count,
                    ),
                    parse_mode="HTML",
                    reply_markup=_crm_name_choice_kb(),
                )
            else:
                header = _crm_escalation_header(remind_count, "CRM")
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"{header}"
                        f"<b>{client_key}</b>\n\n"
                        f"{prompt}\n\n"
                        f"Выполнено сегодня: {done_today} из {min(daily_limit, done_today + total)}\n\n"
                        f"<i>Запрос будет повторяться, пока данные не будут заполнены.</i>"
                    ),
                    parse_mode="HTML",
                reply_markup=(
                    (_crm_phone_choice_kb(client_key) or _crm_phone_help_only_kb())
                    if state == "clarify_phone"
                    else _crm_name_choice_kb()
                ),
                )
            pending["remind_count"] = remind_count
            pending["last_sent"] = now.isoformat()
            await _crm_notify_admin_unresolved(context, chat_id, pending, remind_count)
            if not _crm_save_pending():
                crm_logger.error("crm_state_lock_timeout: reminder save failed (in-memory only)")
            crm_logger.info("CRM escalating reminder → chat_id=%s client=%s count=%d", chat_id, client_key, remind_count)
        except Exception as e:
            crm_logger.warning("crm_phone_reminder_task chat_id=%s: %s", chat_id, e)


async def collector_reminder_task(context: ContextTypes.DEFAULT_TYPE):
    """Hourly: send reminders to managers with pending collector dialogs.
    Работает 09–19 (включительно): нужно успеть перевести батч к 19:00.
    """
    from bot.workday_checker import is_holiday_today
    if is_holiday_today():
        sched_logger.info("collector_reminder_task: выходной — пропуск")
        return
    now = datetime.now(TZ)
    if not (9 <= now.hour < 19):
        sched_logger.debug("collector_reminder_task: вне рабочих часов (%d:xx) — пропуск", now.hour)
        return
    try:
        from collector.manager_dialog import send_reminders as _collector_reminders
        await _collector_reminders()
    except Exception as e:
        sched_logger.error("collector_reminder_task error: %s", e)
    try:
        from collector.approval_flow import promote_silent_batches_to_admin
        promoted = await promote_silent_batches_to_admin()
        if promoted:
            sched_logger.info("collector_reminder_task: %d approval-батч(ей) передано администратору по таймауту", promoted)
    except Exception as e:
        sched_logger.error("collector approval escalation error: %s", e)



async def whatsapp_poller_task(context: ContextTypes.DEFAULT_TYPE):
    """Poll Green API for incoming WhatsApp messages every 30 seconds."""
    try:
        from collector.whatsapp_poller import poll_once
        # BUG FIX: poll_once может зависнуть (аудио Whisper до 90 сек).
        # Жёсткий таймаут 25 сек — не даём задержать следующие джобы.
        await asyncio.wait_for(poll_once(), timeout=25)
    except asyncio.TimeoutError:
        integration_logger.warning("whatsapp_poller_task: poll_once timeout (>25s) — пропуск итерации")
    except Exception as e:
        integration_logger.error("whatsapp_poller_task error: %s", e)


async def handle_voice_message_tg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик голосовых сообщений от менеджеров."""
    chat_id = update.effective_chat.id
    if get_user_role(chat_id) == "unknown":
        return
    voice = update.message.voice
    if voice:
        try:
            from collector.manager_dialog import handle_voice_message as _col_voice
            if await _col_voice(chat_id, voice.file_id):
                return
        except Exception as e:
            logger.error("voice handler error: %s", e)


async def handle_proof_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Перехватывает фото/документ от менеджера ожидающего подтверждения оплаты."""
    chat_id = update.effective_chat.id
    if get_user_role(chat_id) == "unknown":
        return
    try:
        if await _maybe_handle_developer_media_flow(update, context):
            return
    except Exception as e:
        logger.error("developer media flow error: %s", e)
    try:
        from collector.approval_flow import handle_manager_proof
        msg = update.message
        if msg.photo:
            file_id   = msg.photo[-1].file_id
            file_type = "photo"
        elif msg.document:
            file_id   = msg.document.file_id
            file_type = "document"
        else:
            return
        handled = await handle_manager_proof(
            chat_id=chat_id,
            file_id=file_id,
            file_type=file_type,
            admin_chat_id=ADMIN_CHAT_ID,
            bot=context.bot,
        )
        if handled:
            return
    except Exception as e:
        logger.error("handle_proof_document error: %s", e)


async def handle_agreed_details_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Перехватывает текст с деталями договорённости от менеджера."""
    chat_id = update.effective_chat.id
    if get_user_role(chat_id) == "unknown":
        return
    text = update.message.text or ""
    if not text.strip():
        return
    try:
        from collector.approval_flow import handle_manager_agreed_details
        handled = await handle_manager_agreed_details(chat_id=chat_id, text=text.strip())
        if handled:
            return
    except Exception as e:
        logger.error("handle_agreed_details_text error: %s", e)


# Антиспам: один раз в 6 часов на чат — чтобы не дёргать Саиду каждым «ок»
_SAIDA_HELP_LAST_SENT: Dict[int, float] = {}
_SAIDA_HELP_COOLDOWN_SEC = 6 * 3600


def _saida_help_due(chat_id: int) -> bool:
    import time as _t
    last = _SAIDA_HELP_LAST_SENT.get(chat_id, 0.0)
    return (_t.time() - last) >= _SAIDA_HELP_COOLDOWN_SEC


def _saida_help_mark_sent(chat_id: int) -> None:
    import time as _t
    _SAIDA_HELP_LAST_SENT[chat_id] = _t.time()


# Catch-all hint для менеджера/админа: если ни один text-handler не сработал.
_MGR_HINT_LAST_SENT: Dict[int, float] = {}
_MGR_HINT_COOLDOWN_SEC = 4 * 3600  # 4ч — мягче чем у Саиды, у менеджеров рабочий день


def _mgr_hint_due(chat_id: int) -> bool:
    import time as _t
    last = _MGR_HINT_LAST_SENT.get(chat_id, 0.0)
    return (_t.time() - last) >= _MGR_HINT_COOLDOWN_SEC


def _mgr_hint_mark_sent(chat_id: int) -> None:
    import time as _t
    _MGR_HINT_LAST_SENT[chat_id] = _t.time()


async def _maybe_send_manager_no_active_hint(update, context, chat_id: int, text: str) -> None:
    """Мягко подсказывает менеджеру: «у вас нет открытых запросов от меня».

    Срабатывает только если:
      - chat_id принадлежит менеджеру/админу (не Саиде, не неизвестному)
      - текст похож на «попытку ответа» (короткий, без команд)
      - cooldown 4ч прошёл
    """
    # Не для Саиды — там свой handler
    if chat_id == _get_saida_chat_id():
        return
    # Только для известных пользователей
    role = get_user_role(chat_id)
    if role == "unknown":
        return
    # Не на команды и не на длинные тексты (>120 симв — это не «ой я ответил»)
    t = (text or "").strip()
    if not t or t.startswith("/") or len(t) > 120:
        return
    # Только если текст похож на «ответ»
    looks_like_reply = any(kw in t.lower() for kw in (
        "оплат", "договор", "клиент", "не зна", "позже",
        "потом", "ок", "хорошо", "ладно", "понял", "поняла",
        "да", "нет", "сделаю", "сейчас", "завтра",
        "понедельник", "вторник", "среда", "четверг",
        "пятниц", "суббот", "воскресень",
    ))
    if not looks_like_reply:
        return
    if not _mgr_hint_due(chat_id):
        return
    try:
        kb_help = InlineKeyboardMarkup([[
            InlineKeyboardButton("🔄 Открыть меню", callback_data="back_main"),
            InlineKeyboardButton("📖 Инструкция", callback_data="show_help_doc"),
        ]])
        await update.message.reply_text(
            "🤔 Я не нашёл, к какому запросу относится ваш ответ.\n\n"
            "Возможные причины:\n"
            "• Запрос уже закрыт или истёк по времени.\n"
            "• Сообщение с кнопками уехало далеко вверх — пролистайте чат.\n"
            "• Открытых запросов от меня сейчас нет.\n\n"
            "Откройте меню кнопкой ниже или нажмите «📖 Инструкция».",
            reply_markup=kb_help,
        )
        _mgr_hint_mark_sent(chat_id)
    except Exception as _e:
        logger.warning("mgr no-active hint send failed: %s", _e)


async def _handle_saida_text_payhold_reply(update: Update, context, text: str) -> bool:
    """Распознаёт текстовый ответ Саиды по запросам на проверку оплат.

    Вызывается из handle_persistent_menu только если chat_id == SAIDA_CHAT_ID.
    Возвращает True, если сообщение распознано и обработано (handle_persistent_menu
    должен вернуться без дальнейшей обработки). False — пусть текст идёт обычным путём.
    """
    try:
        from collector.payment_hold import (
            parse_saida_text_reply,
            confirm_by_saida,
            list_pending,
        )
    except Exception as _e:
        logger.warning("payment_hold module unavailable for saida text: %s", _e)
        return False

    decision = parse_saida_text_reply(text)
    action = decision.get("action")
    chat_id = update.effective_chat.id

    if action == "no_match":
        # Это обычное сообщение Саиды, не про оплаты.
        # Если у неё есть pending запросы и текст похож на «ответ» — мягко
        # покажем мини-подсказку (раз в 6 часов, чтоб не спамить).
        try:
            pending = list_pending()
        except Exception:
            pending = []
        if not pending:
            return False
        # Триггер подсказки: текст короткий (<60 символов) — похоже на попытку ответа
        looks_like_attempt = len(text) <= 60 and any(
            kw in text.lower() for kw in (
                "оплат", "оплачен", "прошл", "поступ", "видн", "нашл",
                "част", "нет", "ок", "хорошо", "посмотр", "посмотрю",
                "позже", "потом", "разбер",
            )
        )
        if looks_like_attempt and _saida_help_due(chat_id):
            kb_help = InlineKeyboardMarkup([[
                InlineKeyboardButton(
                    "❓ Подробная инструкция", callback_data="payhold_help_full",
                ),
            ]])
            try:
                await update.message.reply_text(
                    f"🤔 У тебя сейчас открыто <b>{len(pending)}</b> "
                    f"запросов по оплатам — но я не понял, к какому относится "
                    f"твой ответ.\n\n"
                    "Ответь так:\n"
                    "  • Нажми кнопку под нужным запросом, или\n"
                    "  • Напиши: <code>&lt;имя клиента&gt; полная / частично / нет</code>\n\n"
                    "Например: <i>Акжан полная</i> или <i>Шапагат частично</i>.",
                    parse_mode="HTML",
                    reply_markup=kb_help,
                )
                _saida_help_mark_sent(chat_id)
            except Exception as _ne:
                logger.warning("saida soft hint send failed: %s", _ne)
            # True, чтобы не пускать дальше в legacy-меню
            return True
        return False

    matches = decision.get("matches") or []
    status = decision.get("status")
    chat_id = update.effective_chat.id

    if action == "confirm" and len(matches) == 1 and status:
        rec = matches[0]
        token = rec.get("token", "")
        client = rec.get("client", "")
        manager = rec.get("manager", "")
        manager_chat_id = int(rec.get("manager_chat_id") or 0)
        updated = confirm_by_saida(token, status)
        if not updated:
            await update.message.reply_text(
                f"⚠️ По «{client}» не удалось сохранить ответ — возможно, уже закрыт."
            )
            return True
        status_label = {
            "full": "полная оплата ✅",
            "partial": "частичная оплата 🔸",
            "none": "оплаты нет ❌",
        }.get(status, status)
        await update.message.reply_text(
            f"Принято по «{client}»: {status_label}.\nМенеджер и директор уведомлены."
        )
        # Уведомить менеджера и админа — повторяем UX из payhold-кнопок.
        notify_text_full = (
            "Саида подтвердила оплату по клиенту:\n\n"
            f"{client}\nМенеджер: {manager}\n"
            f"Статус: {'оплата есть' if status == 'full' else 'частичная оплата'}\n\n"
            "Клиент временно не будет попадать под давление до обновления 1С."
        )
        notify_text_none = (
            "Саида не видит оплату по клиенту:\n\n"
            f"{client}\nМенеджер: {manager}\n\n"
            "Клиент остаётся в обычной дебиторке."
        )
        notify_text = notify_text_full if status in ("full", "partial") else notify_text_none
        for target in {manager_chat_id, ADMIN_CHAT_ID}:
            if not target:
                continue
            try:
                await context.bot.send_message(
                    chat_id=target,
                    text=notify_text,
                    parse_mode="HTML",
                )
            except Exception as _ne:
                logger.warning("saida text confirm notify %s failed: %s", target, _ne)
        log_event(
            "saida_text_confirm",
            client=client,
            manager=manager,
            status=status,
        )
        return True

    if action == "ambiguous_client":
        # Несколько клиентов в pending матчатся — переспрашиваем
        names = []
        for r in matches[:5]:
            names.append(f"• {r.get('client', '—')} (менеджер {r.get('manager', '—')})")
        msg = (
            "🤔 Не понял, по какому клиенту ответ. Совпало несколько:\n\n"
            + "\n".join(names)
            + "\n\nНапишите имя клиента точнее (например, добавьте улицу) "
            "или нажмите кнопку под нужным запросом."
        )
        await update.message.reply_text(msg)
        return True

    if action == "unclear_status":
        rec = matches[0]
        client = rec.get("client", "—")
        msg = (
            f"🤔 По «{client}» — не понял статус.\n\n"
            "Напишите одно из: <b>полная</b> / <b>частично</b> / <b>нет</b>.\n"
            "Или нажмите кнопку под исходным запросом."
        )
        await update.message.reply_text(msg, parse_mode="HTML")
        return True

    return False


async def handle_persistent_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений (v9.4.12, cleanup v9.4.57)."""
    text = update.message.text
    chat_id = update.effective_chat.id

    if not await _acl_gate(chat_id, context):
        return

    try:
        if await _maybe_handle_developer_text_flow(update, context):
            return
    except Exception as _de:
        logger.error("developer text flow error: %s", _de)

    # ─── Текстовый ответ Саиды по pending-оплатам ────────────────────────
    # Why: Саида часто пишет «Акжан полная» вместо нажатия кнопок.
    # Без парсера её ответы пропадают, pending копится. Реагируем
    # ТОЛЬКО на её chat_id и ТОЛЬКО при найденном клиенте + ясном статусе.
    try:
        if text and chat_id == _get_saida_chat_id():
            handled = await _handle_saida_text_payhold_reply(update, context, text)
            if handled:
                return
    except Exception as _se:
        logger.error("saida text payhold handler error: %s", _se)

    # v9.4.57 (legacy reply-menu cleanup): снять "призрак" старой
    # reply-клавиатуры. Срабатывает при нажатии пользователем на любую
    # из устаревших кнопок — бот отправляет ReplyKeyboardRemove и
    # клавиатура исчезает у пользователя без необходимости /start.
    _LEGACY_REPLY_LABELS = {
        "Статус", "Отчёты", "Отчеты",
        "Последний debt", "Последний sales", "Последний gross",
        "Архив", "Меню",
    }
    if text and text.strip() in _LEGACY_REPLY_LABELS:
        try:
            await update.message.reply_text(
                "Меню обновлено. Откройте основное меню: /start",
                reply_markup=ReplyKeyboardRemove(),
            )
        except Exception as e:
            logger.warning("legacy reply-menu cleanup failed: %s", e)
        return

    dup_token = _CRM_DUP_REVIEW_AWAITING_TEXT.get(chat_id)
    if dup_token:
        review = _CRM_DUP_REVIEW_PENDING.get(dup_token)
        if not review:
            _CRM_DUP_REVIEW_AWAITING_TEXT.pop(chat_id, None)
            if not _crmdup_save_pending():
                crm_logger.error("crm_state_lock_timeout: dup cleanup save failed (in-memory only)")
        else:
            import re as _re
            # Безопасный выход для менеджера который не знает телефон
            _norm_text = text.strip().lower().replace("ё", "е")
            _give_up_phrases = (
                "не знаю", "не помню", "потом", "уточню", "позже",
                "позднее", "завтра", "не в курсе", "хз", "отмена",
            )
            if any(p in _norm_text for p in _give_up_phrases):
                _CRM_DUP_REVIEW_AWAITING_TEXT.pop(chat_id, None)
                if not _crmdup_save_pending():
                    crm_logger.error("crm_state_lock_timeout: dup give-up save failed (in-memory only)")
                await update.message.reply_text(
                    "Понял. Запрос закрыт без изменений — "
                    "телефон в CRM не обновлялся.\n\n"
                    "Когда уточните — напишите <code>/phone Имя клиента 87XXXXXXXXX</code>.",
                    parse_mode="HTML",
                )
                return
            phone_digits = _re.sub(r"\D", "", text.strip())
            if _re.fullmatch(r"8\d{10}", phone_digits):
                phone_digits = "7" + phone_digits[1:]
            if not _re.fullmatch(r"7\d{10}", phone_digits):
                await update.message.reply_text(
                    "❌ Это не похоже на телефон.\n\n"
                    "Введите номер в одном из форматов:\n"
                    "• <code>+77011234567</code>\n"
                    "• <code>87011234567</code>\n"
                    "• <code>77011234567</code>\n\n"
                    "Если не знаете телефон — напишите <b>«не знаю»</b> "
                    "или <b>«уточню позже»</b> и я закрою запрос.",
                    parse_mode="HTML",
                )
                return
            reviewer = _chat_to_manager(chat_id) or ("Вадим" if is_admin(chat_id) else "")
            client_keys = [item.get("client_key", "") for item in review.get("items", [])]
            # F-12: deterministic keep-key — alphabetical sort вместо случайного client_keys[0]
            chosen_key = sorted([k for k in client_keys if k])[0] if any(client_keys) else ""
            try:
                from bot.crm_clients import resolve_phone_conflict
                ok = resolve_phone_conflict(
                    client_keys=client_keys,
                    chosen_phone="+" + phone_digits,
                    chosen_key=chosen_key,
                    reviewer=reviewer,
                    phone_source="manager_duplicate_review_manual",
                )
            except Exception as e:
                crm_logger.error("crmdup manual resolve error: %s", e)
                ok = False
            if not ok:
                await update.message.reply_text("⚠️ Не удалось сохранить решение по дублю.")
                return
            review["resolved_at"] = datetime.now(TZ).isoformat()
            review["resolution"] = "custom"
            review["chosen_phone"] = "+" + phone_digits
            _CRM_DUP_REVIEW_AWAITING_TEXT.pop(chat_id, None)
            if not _crmdup_save_pending():
                review.pop("resolved_at", None)
                review.pop("resolution", None)
                review.pop("chosen_phone", None)
                _CRM_DUP_REVIEW_AWAITING_TEXT[chat_id] = dup_token
                await update.message.reply_text("⚠️ Временная ошибка сохранения, попробуйте ещё раз.")
                return
            await update.message.reply_text(
                (
                    "✅ Сохранено.\n\n"
                    f"Новый номер: <code>+{phone_digits}</code>\n"
                    "Карточки объединены в CRM, alias сохранены."
                ),
                parse_mode="HTML",
            )
            return

    # CRM: уточняющий диалог по шагам (clarify_name → clarify_phone → clarify_address)
    _crm_cleanup_pending()
    pending = _CRM_PHONE_PENDING.get(chat_id)
    if pending and pending.get("state", "").startswith("clarify_"):
        state = pending["state"]
        client_key = pending["client_key"]
        try:
            from bot.crm_clients import set_client_details as _set_details
            import re as _re

            if state == "clarify_name":
                display_name = text.strip()
                # Защита от «не знаю» / «потом» / «уточню» — это не имена.
                _norm = display_name.lower().replace("ё", "е")
                _bad_phrases = (
                    "не знаю", "не помню", "потом", "уточню", "позже",
                    "не в курсе", "хз", "?", "??", "???",
                )
                if any(p in _norm for p in _bad_phrases):
                    await update.message.reply_text(
                        "🤔 Это не похоже на имя клиента.\n\n"
                        "Если сейчас не знаете — нажмите кнопку <b>⏳ Позже</b> "
                        "под предыдущим сообщением. Я спрошу повторно завтра.\n\n"
                        "Или напишите имя как есть (например: <i>Аида</i>, <i>Серик</i>).",
                        parse_mode="HTML",
                    )
                    return
                if len(display_name) < 2:
                    await update.message.reply_text(
                        "❌ Слишком коротко. Введите имя или имя и отчество:"
                    )
                    return
                pending.setdefault("original_name", client_key)
                pending["display_name"] = display_name
                pending["name_mode"] = "manual"
                pending["name_review_needed"] = False
                pending.pop("awaiting_name_text", None)
                pending["state"] = "clarify_phone"
                pending["last_sent"] = datetime.now(TZ).isoformat()
                if not _crm_save_pending():
                    await update.message.reply_text("⚠️ Временная ошибка сохранения, попробуйте ещё раз.")
                    return
                await update.message.reply_text(
                    _crm_phone_prompt_text(client_key),
                    parse_mode="HTML",
                    reply_markup=_crm_phone_choice_kb(client_key) or _crm_phone_help_only_kb(),
                )
                return

            if state == "clarify_phone":
                phone_digits = _re.sub(r"\D", "", text.strip())
                if _re.fullmatch(r"8\d{10}", phone_digits):
                    phone_digits = "7" + phone_digits[1:]
                if not _re.fullmatch(r"7\d{10}", phone_digits):
                    await update.message.reply_text(
                        f"❌ Неверный формат.\n"
                        f"Введите: <code>+7XXXXXXXXXX</code>",
                        parse_mode="HTML",
                    )
                    return
                await _crm_save_phone_and_continue(
                    context,
                    chat_id,
                    pending,
                    "+" + phone_digits,
                    phone_source="manager_manual",
                )
                return

            if state == "clarify_address":
                address = text.strip()
                pending["address"] = address
                display_name = pending.get("display_name", "")
                phone = pending.get("phone", "")
                ok = _set_details(
                    client_key,
                    display_name=display_name,
                    phone=phone,
                    address=address,
                    original_name=pending.get("original_name", client_key),
                    name_mode=pending.get("name_mode", "manual" if display_name else "later"),
                    name_review_needed=pending.get("name_review_needed", not bool(display_name)),
                )
                log_event("crm_details_set", client=client_key,
                          display_name=display_name, phone=phone, address=address)

                done_today   = pending.get("done_today", 0) + 1
                daily_limit  = pending.get("daily_limit", CRM_DAILY_LIMIT)
                manager_name = pending.get("manager", "")
                old_pending = dict(pending)
                _CRM_PHONE_PENDING.pop(chat_id, None)
                if not _crm_save_pending():
                    _CRM_PHONE_PENDING[chat_id] = old_pending
                    await update.message.reply_text("⚠️ Временная ошибка сохранения, попробуйте ещё раз.")
                    return

                if ok:
                    await update.message.reply_text(
                        f"✅ Сохранено: <b>{display_name}</b> · {phone}",
                        parse_mode="HTML",
                    )
                else:
                    await update.message.reply_text("⚠️ Не удалось сохранить. Клиент не найден.")

                # Следующий клиент в цепочке или итог дня
                if done_today < daily_limit:
                    from bot.crm_clients import get_clients_without_phones as _crm_next
                    next_list = _crm_next(manager_name, limit=1)
                    remaining = len(_crm_next(manager_name, limit=500))
                    if next_list:
                        next_key = next_list[0]
                        _voice_next_now_iso = datetime.now(TZ).isoformat()
                        _CRM_PHONE_PENDING[chat_id] = {
                            "state": "clarify_name",
                            "client_key": next_key,
                            "original_name": next_key,
                            "done_today": done_today,
                            "daily_limit": daily_limit,
                            "manager": manager_name,
                            "total_no_phone": remaining,
                            "created_at": _voice_next_now_iso,
                            "last_sent": _voice_next_now_iso,
                        }
                        if not _crm_save_pending():
                            _CRM_PHONE_PENDING.pop(chat_id, None)
                            await update.message.reply_text("⚠️ Следующий клиент не сохранён в очереди. Откройте CRM снова.")
                            return
                        await update.message.reply_text(
                            _crm_name_prompt_text(
                                client_key=next_key,
                                done_today=done_today,
                                total=remaining,
                                daily_limit=daily_limit,
                            ),
                            parse_mode="HTML",
                            reply_markup=_crm_name_choice_kb(),
                        )
                    else:
                        await update.message.reply_text(
                            f"🎉 Все клиенты внесены! База полностью заполнена.",
                            parse_mode="HTML",
                        )
                else:
                    from bot.crm_clients import get_clients_without_phones as _crm_remain
                    remaining = len(_crm_remain(manager_name, limit=500))
                    if remaining:
                        await update.message.reply_text(
                            f"✅ На сегодня готово — внесено {done_today} клиентов.\n\n"
                            f"📋 Осталось без телефона: <b>{remaining}</b>\n"
                            f"Завтра в 18:00 бот пришлёт ещё {min(daily_limit, remaining)}.",
                            parse_mode="HTML",
                        )
                    else:
                        await update.message.reply_text(
                            f"🎉 Все клиенты внесены! База полностью заполнена."
                        )
                return

        except Exception as e:
            logger.error("crm clarify handler error: %s", e)
            _CRM_PHONE_PENDING.pop(chat_id, None)
        return

    # [DISABLED v9.4.39] Старый flow: коллектор → pending → ввод текстом.
    # Заменён на CRM /phone + crm_psel| callback.
    # Оставлен закомментированным на случай отката.
    #
    # try:
    #     from collector.collections_db import clear_name_pending, get_name_pending
    #     pending_name_client = get_name_pending(chat_id)
    #     if pending_name_client:
    #         new_name = text.strip()
    #         if len(new_name) >= 2:
    #             from collector.registry_manager import update_client_display_name
    #             if update_client_display_name(pending_name_client, new_name):
    #                 clear_name_pending(chat_id)
    #                 await update.message.reply_text(
    #                     f"✅ Имя сохранено.\nВ 1С (ключ): <b>{pending_name_client}</b>\n"
    #                     f"В сообщениях: <b>{new_name}</b>", parse_mode="HTML")
    #             else:
    #                 await update.message.reply_text("⚠️ Не удалось сохранить. Попробуйте ещё раз.")
    #         else:
    #             await update.message.reply_text("❌ Слишком короткое имя.")
    #         return
    # except Exception as e:
    #     logger.error("name input handler error: %s", e)
    #
    # try:
    #     from collector.collections_db import clear_phone_pending, get_phone_pending
    #     pending_client = get_phone_pending(chat_id)
    #     if pending_client:
    #         import re as _re
    #         phone_clean = _re.sub(r"\D", "", text.strip())
    #         if _re.fullmatch(r"8\d{10}", phone_clean):
    #             phone_clean = "7" + phone_clean[1:]
    #         if _re.fullmatch(r"7\d{10}", phone_clean):
    #             from collector.registry_manager import update_client_phone
    #             if update_client_phone(pending_client, phone_clean):
    #                 clear_phone_pending(chat_id)
    #                 await update.message.reply_text(
    #                     f"✅ Телефон <b>+{phone_clean}</b> сохранён для <b>{pending_client}</b>\n\n"
    #                     f"ИИ-помощник подключится при следующем цикле.", parse_mode="HTML")
    #             else:
    #                 await update.message.reply_text("⚠️ Не удалось сохранить. Попробуйте ещё раз.")
    #         else:
    #             await update.message.reply_text(
    #                 f"❌ Неверный формат: <code>{text.strip()}</code>\n"
    #                 f"Введите: +77XXXXXXXXXX / 77XXXXXXXXXX / 87XXXXXXXXXX", parse_mode="HTML")
    #         return
    # except Exception as e:
    #     logger.error("phone input handler error: %s", e)

    # Проверяем ожидание деталей "договорились" (debt_stop_control)
    try:
        from bot.debt_stop_control import handle_dstop_detail_message as _dstop_detail
        if await _dstop_detail(chat_id, text, context.bot):
            return
    except Exception as e:
        logger.error("dstop detail handler error: %s", e)

    # Check active collector dialog first
    try:
        from collector.manager_dialog import handle_text_message as _col_text
        if await _col_text(chat_id, text):
            return
    except Exception as e:
        logger.error("collector text handler error: %s", e)

    # ─── Catch-all мягкая подсказка для менеджера/админа ─────────────────────
    # Why: если менеджер пишет в чат, а ни один из flow выше не сматчился —
    # значит у него нет открытых запросов от бота. Раньше бот молчал, и менеджер
    # думал что бот сломан или его сообщение пропало. Теперь — мягкий хинт.
    try:
        await _maybe_send_manager_no_active_hint(update, context, chat_id, text)
    except Exception as _he:
        logger.warning("manager catch-all hint failed: %s", _he)

    # v9.4.57: elif-блок для мёртвых ярлыков kb_persistent()
    # (📊 Дебиторка / 🛒 Продажи / 💰 Валовая / 💸 Затраты / 📦 Остатки /
    # 📈 Аналитика / 🗄️ Архив) удалён. Функция kb_persistent() была
    # определена в v9.4.12, но никогда не подключалась как reply_markup,
    # поэтому эти ярлыки не мог прислать ни один пользователь. Основное
    # меню работает через inline-клавиатуру /start → callback_data.


class _PinnedTelegramRequest(HTTPXRequest):
    """
    PTB/httpx transport с явным public CA bundle и trust_env=False.
    Это не отключает TLS-проверку, а убирает скрытое влияние
    proxy/SSL env и фиксирует верификацию на certifi.
    """

    def __init__(self, *, client_label: str, **kwargs):
        self._client_label = client_label
        self._ca_bundle = certifi.where()
        super().__init__(**kwargs)

    def _build_client(self) -> httpx.AsyncClient:
        ssl_ctx = ssl.create_default_context(cafile=self._ca_bundle)
        return httpx.AsyncClient(
            verify=ssl_ctx,
            trust_env=False,
            **self._client_kwargs,
        )


def _build_telegram_requests() -> tuple[HTTPXRequest, HTTPXRequest]:
    main_request = _PinnedTelegramRequest(
        client_label="bot_api",
        connection_pool_size=8,
        read_timeout=20.0,
        write_timeout=20.0,
        connect_timeout=15.0,
        pool_timeout=5.0,
    )
    updates_request = _PinnedTelegramRequest(
        client_label="get_updates",
        connection_pool_size=2,
        read_timeout=35.0,
        write_timeout=20.0,
        connect_timeout=15.0,
        pool_timeout=5.0,
    )
    return main_request, updates_request


def _log_telegram_transport_settings() -> None:
    logger.info(
        "telegram_transport_tls: ca_bundle=%s trust_env=%s main_pool=%s updates_pool=%s",
        certifi.where(),
        False,
        8,
        2,
    )


def main():
    if STOP_FILE.exists():
        sched_logger.info("Stop file found: %s. Bot startup cancelled.", STOP_FILE)
        sys.exit(0)
    _check_single_instance()  # завершаем если уже запущен другой экземпляр
    if not BOT_TOKEN:
        integration_logger.critical("TG_BOT_TOKEN не найден в .env! Запуск невозможен.")
        sys.exit(1)

    # v9.4.6.2: Версия в логе
    log_event("bot_starting", version = __VERSION__)
    main_request, updates_request = _build_telegram_requests()
    _log_telegram_transport_settings()
    
    # v9.4.6.1: ПАТЧ - Правильная регистрация post_init через builder
    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .request(main_request)
        .get_updates_request(updates_request)
        .post_init(post_init)
        .build()
    )

    async def _runtime_alert_sender(text: str) -> None:
        if not ADMIN_CHAT_ID:
            return
        # Дренируем dead letters накопленные пока Telegram был недоступен
        if has_dead_letters():
            drained = pop_dead_letters()
            for dead_text in drained:
                try:
                    await application.bot.send_message(
                        chat_id=ADMIN_CHAT_ID,
                        text=dead_text,
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                except Exception:
                    push_dead_letter(dead_text)
                    break  # Telegram всё ещё недоступен — прекращаем дрейн
        try:
            await application.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except Exception as exc:
            integration_logger.warning("runtime alert send failed: %s", exc)
            push_dead_letter(text)

    set_telegram_alert_sender(_runtime_alert_sender)

    # BUG FIX: глушим "No error handlers are registered" для сетевых ошибок Telegram
    async def _tg_error_handler(update: object, context) -> None:
        err = context.error
        if isinstance(err, NetworkError):
            if "CERTIFICATE_VERIFY_FAILED" in str(err):
                integration_logger.error(
                    "Telegram TLS verify failed: %s | ca_bundle=%s | trust_env=%s",
                    err,
                    certifi.where(),
                    False,
                )
            integration_logger.warning("Telegram NetworkError (transient): %s", err)
        else:
            integration_logger.error("Telegram error: %s", err, exc_info=err)
    application.add_error_handler(_tg_error_handler)

    application.add_handler(CommandHandler("batch", cmd_batch))
    application.add_handler(CommandHandler("start", cmd_start))
    # v9.4.12: Обработчик текстовых команд от persistent menu
    from telegram.ext import MessageHandler, filters
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_persistent_menu))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message_tg))
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.ALL, handle_proof_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_agreed_details_text), group=1)
    application.add_handler(CommandHandler("health", cmd_health))
    application.add_handler(CommandHandler("restart", cmd_restart))
    application.add_handler(CommandHandler("shutdown", cmd_shutdown))
    application.add_handler(CommandHandler("stats", cmd_stats))
    application.add_handler(CommandHandler("analytics", cmd_analytics))  # 🆕 v9.4.9  # v2.0
    application.add_handler(CommandHandler("phone", cmd_phone))  # CRM: внести телефон клиента
    application.add_handler(CommandHandler("crmdupsend", cmd_crmdupsend))  # CRM: разовая сверка конфликтных дублей
    application.add_handler(CommandHandler("guide", cmd_guide))  # Инструкция (роль-зависимая)
    application.add_handler(CommandHandler("help", cmd_help))    # Алиас /guide
    application.add_handler(CommandHandler("dev", cmd_dev))      # Вход в one-shot inbox разработчика
    application.add_handler(CommandHandler("devhelp", cmd_devhelp))  # Краткая справка по dev-командам
    application.add_handler(CommandHandler("announce_reset", cmd_announce_reset))  # Инфо-рассылка после reset
    application.add_handler(CommandHandler("logs", cmd_logs))    # Последние ERROR/CRITICAL
    application.add_handler(CommandHandler("timeline", cmd_timeline))  # Единая timeline по клиенту
    application.add_handler(CallbackQueryHandler(cb_data))
    job_queue = application.job_queue
    if job_queue:
        job_queue.run_repeating(pipeline_task, interval=PIPELINE_INTERVAL_MIN * 60, first=60, name="pipeline")
        job_queue.run_repeating(new_reports_notifier, interval=SCAN_INTERVAL_MIN * 60, first=180, name="new_reports")
        
        # v9.4.39: silence_alerts убран из расписания — теперь только event-driven
        # (запускается после каждого пайплайн-цикла где обработаны долговые файлы)
        sched_logger.info("🔔 silence_alerts: event-driven по приходу долговых отчётов (без 14:00)")
        
        # v9.4.32: Упущенная прибыль — еженедельно в пятницу 14:05 (было: ежедневно 14:05 и 21:05)
        if _OPPORTUNITY_LOSS_AVAILABLE:
            job_queue.run_daily(
                send_opportunity_loss_report,
                time=dt_time(14, 5, tzinfo=TZ),
                name="opportunity_loss_weekly"
            )
            sched_logger.info("💸 Настроен еженедельный джоб: упущенная прибыль по пятницам 14:05")
        else:
            sched_logger.warning("⚠️ opportunity_loss не загружен — джобы 14:05/21:05 не запущены")
        
        # v9.4.6.1: Janitor каждые 60 минут (было 60 сек)
        job_queue.run_repeating(
            janitor_task,
            interval=JANITOR_INTERVAL_SEC,
            first=120,
            name="janitor"
        )
        sched_logger.info(f"🧹 Настроен janitor: проверка очереди удаления каждые {JANITOR_INTERVAL_SEC} сек ({JANITOR_INTERVAL_SEC//60} мин)")

        # v9.4.7: Обработка очереди автогенерации ИИ
        if AI_AUTO_GENERATION:
            job_queue.run_repeating(
                process_ai_generation_queue,
                interval=AI_GENERATION_INTERVAL_SEC,
                first=300,
                name="ai_queue_processor"
            )
            sched_logger.info(f"🤖 Настроен обработчик очереди ИИ: интервал {AI_GENERATION_INTERVAL_SEC} сек")
        
        # v9.4.7: Ежедневная сводка админу
        if ADMIN_ACTIVITY_LOG and ADMIN_CHAT_ID:
            job_queue.run_daily(
                send_daily_summary_to_admin,
                time=ADMIN_SUMMARY_TIME,
                name="daily_summary"
            )
            sched_logger.info(f"📊 Настроена ежедневная сводка админу в {ADMIN_SUMMARY_TIME_STR}")
        
        # v9.4.7: Сброс счётчика генераций в полночь
        job_queue.run_daily(
            reset_ai_generation_state,
            time=dt_time(0, 1, tzinfo=TZ),
            name="reset_ai_state"
        )
        sched_logger.info("🔄 Настроен сброс счётчика ИИ-генераций в 00:01")

        # ═══ v9.4.8: ЕЖЕНЕДЕЛЬНАЯ AI + КРАТКИЕ СВОДКИ ═══
        
        # Еженедельная AI генерация (вторник через run_daily + weekday guard)
        job_queue.run_daily(
            weekly_ai_generation,
            time=dt_time(10, 0, tzinfo=TZ),
            name="weekly_ai_generation"
        )
        sched_logger.info("🤖 Настроена еженедельная AI генерация: вторник 10:00")
        
        # v9.4.16: Ежедневная аналитика в 22:00 (было: только понедельник 10:00)
        job_queue.run_daily(
            weekly_analytics_wrapper,
            time=dt_time(22, 0, tzinfo=TZ),
            name="daily_analytics"
        )
        sched_logger.info("📊 Настроена ежедневная аналитика: каждый день 22:00")
        
        # Краткие сводки
        job_queue.run_daily(
            send_inventory_summary,
            time=dt_time(9, 0, tzinfo=TZ),
            name="inventory_summary"
        )
        sched_logger.info("📦 Настроена краткая сводка остатков: ежедневно 09:00")

        job_queue.run_daily(
            morning_error_digest_task,
            time=dt_time(9, 5, tzinfo=TZ),
            name="morning_error_digest",
        )
        sched_logger.info("🌅 Настроен утренний digest ошибок: ежедневно 09:05")
        
        job_queue.run_daily(
            send_gross_summary,
            time=dt_time(20, 0, tzinfo=TZ),
            name="gross_summary"
        )
        sched_logger.info("💰 Настроена краткая сводка валовой: ежедневно 20:00")
        
        job_queue.run_daily(
            send_sales_summary,
            time=dt_time(21, 0, tzinfo=TZ),
            name="sales_summary"
        )
        sched_logger.info("🛒 Настроена краткая сводка продаж: ежедневно 21:00")

        job_queue.run_daily(
            _validate_daily_reports_saida,
            time=dt_time(21, 0, tzinfo=TZ),
            name="validate_daily_reports",
        )
        sched_logger.info("📋 Настроена проверка отчётов Саиды: ежедневно 21:00")
        
        # v9.4.7.5: Автоочистка старых файлов в 03:00
        job_queue.run_daily(
            cleanup_old_files,
            time=dt_time(3, 0, tzinfo=TZ),
            name="cleanup_old_files"
        )
        sched_logger.info("🧹 Настроена автоочистка файлов: логи 2д, AI 7д, HTML 30д, JSON 7д, Excel 14д | Запуск в 03:00")

        job_queue.run_repeating(
            log_monitor_task,
            interval=2 * 60 * 60,
            first=10 * 60,
            name="log_monitor",
        )
        sched_logger.info("🩺 Настроен мониторинг логов: каждые 2 часа")

        # Проверка рабочего дня в 10:00 (если нет xlsx — спросить админа)
        job_queue.run_daily(
            check_workday_task,
            time=dt_time(10, 0, tzinfo=TZ),
            name="check_workday",
        )
        sched_logger.info("📅 Настроена проверка рабочего дня: ежедневно 10:00 (отчёты приходят 09:07–09:38)")

        # CRM: обновление базы клиентов + запрос телефонов в 18:00
        job_queue.run_daily(
            crm_daily_task,
            time=dt_time(18, 0, tzinfo=TZ),
            name="crm_daily",
        )
        sched_logger.info("👥 Настроена CRM: обновление базы + запрос телефонов ежедневно 18:00")

        # AI Debt Collector (17:00 — резервный запуск, если триггер не сработал)
        job_queue.run_daily(
            debt_collector_daily,
            time=dt_time(17, 0, tzinfo=TZ),
            name="debt_collector_daily",
        )
        sched_logger.info("💰 Настроен AI Debt Collector: ежедневно 17:00 (резервный)")

        job_queue.run_daily(
            debt_collector_promises,
            time=dt_time(10, 0, tzinfo=TZ),
            name="debt_collector_promises",
        )
        sched_logger.info("💰 Настроена проверка обещаний: ежедневно 10:00")

        async def _job_check_broken_agreed(ctx):
            from bot.workday_checker import is_holiday_today
            if is_holiday_today():
                return
            try:
                from collector.approval_flow import check_broken_agreed_deadlines
                broken = await check_broken_agreed_deadlines(bot=ctx.bot)
                if broken:
                    sched_logger.info("🤝 Нарушено обещаний: %d — менеджеры и директор уведомлены", broken)
            except Exception as e:
                sched_logger.error("check_broken_agreed_deadlines error: %s", e)

        job_queue.run_daily(
            _job_check_broken_agreed,
            time=dt_time(10, 30, tzinfo=TZ),
            name="wa_agreed_deadline_check",
        )
        sched_logger.info("🤝 Настроена проверка сорванных договорённостей: ежедневно 10:30")

        # Event-driven: --preview после появления свежих debt_ext файлов
        job_queue.run_repeating(
            debt_collector_trigger_check,
            interval=1800,   # каждые 30 минут
            first=120,       # первый check через 2 мин после старта
            name="collector_trigger_check",
        )
        sched_logger.info("⚡ Настроен event-driven триггер коллектора: проверка каждые 30 мин")

        # Контроль отгрузки: проверка allow_after / block_until после разноски оплат
        async def _job_shipment_check(ctx):
            from bot.workday_checker import is_holiday_today
            if is_holiday_today():
                return
            try:
                from collector.shipment_control import check_pending_decisions
                resolved = await check_pending_decisions(ctx.bot)
                if resolved:
                    sched_logger.info("shipment_check: закрыто %d решений об отгрузке", resolved)
            except Exception as e:
                sched_logger.error("shipment_check job error: %s", e)

        job_queue.run_daily(
            _job_shipment_check,
            time=dt_time(14, 0, tzinfo=TZ),
            name="shipment_check",
        )
        sched_logger.info("🚚 Настроен контроль отгрузки: проверка allow_after/block_until ежедневно 14:00")

        job_queue.run_repeating(
            crm_phone_reminder_task,
            interval=1800,
            first=600,
            name="crm_phone_reminders",
        )
        sched_logger.info("📋 Настроены CRM-напоминания о телефонах: каждые 30 мин (09–19)")

        job_queue.run_repeating(
            collector_reminder_task,
            interval=1800,
            first=300,
            name="collector_reminders",
        )
        sched_logger.info("📨 Настроен AI Коллектор: напоминания менеджерам каждые 30 мин")

        # ── Штрафные баллы за пропуск окна согласования ────────────────
        async def _job_approval_penalty_check(ctx):
            try:
                from collector.approval_penalty import check_recent_batches, check_crm_ignores
                await check_recent_batches(ctx.bot)
                await check_crm_ignores(ctx.bot)
            except Exception as e:
                sched_logger.error("approval_penalty_check error: %s", e)

        async def _job_approval_penalty_monthly(ctx):
            try:
                from collector.approval_penalty import send_monthly_penalty_report
                await send_monthly_penalty_report(ctx.bot)
            except Exception as e:
                sched_logger.error("approval_penalty_monthly error: %s", e)

        job_queue.run_repeating(
            _job_approval_penalty_check,
            interval=1800,
            first=120,
            name="approval_penalty_check",
        )
        job_queue.run_daily(
            _job_approval_penalty_monthly,
            time=__import__("datetime").time(23, 0, tzinfo=TZ),
            name="approval_penalty_monthly",
        )
        sched_logger.info("💰 Штрафные баллы: проверка каждые 30 мин + отчёт в конце месяца в 23:00")

        job_queue.run_repeating(
            whatsapp_poller_task,
            interval=30,
            first=60,
            name="whatsapp_poller",
        )
        sched_logger.info("📱 Настроен Green API поллер: каждые 30 сек")

        # ── Стоп-лист отгрузки (Саида) ─────────────────────────────
        if _DEBT_STOP_AVAILABLE:
            async def _job_dstop_monitor(ctx):
                from bot.workday_checker import is_holiday_today
                if is_holiday_today():
                    sched_logger.info("_job_dstop_monitor: выходной — пропуск")
                    return
                try:
                    await _dstop_monitor(ctx.bot)
                except Exception as e:
                    sched_logger.error("debt_stop monitor error: %s", e)

            async def _job_dstop_managers(ctx):
                from bot.workday_checker import is_holiday_today
                if is_holiday_today():
                    sched_logger.info("_job_dstop_managers: выходной — пропуск")
                    return
                try:
                    await _dstop_managers(ctx.bot)
                except Exception as e:
                    sched_logger.error("debt_stop managers error: %s", e)

            async def _job_dstop_manager_reminders(ctx):
                from bot.workday_checker import is_holiday_today
                if is_holiday_today():
                    sched_logger.info("_job_dstop_manager_reminders: выходной — пропуск")
                    return
                try:
                    await _dstop_manager_reminders(ctx.bot)
                except Exception as e:
                    sched_logger.error("debt_stop manager reminders error: %s", e)

            async def _job_dstop_escalate(ctx):
                from bot.workday_checker import is_holiday_today
                if is_holiday_today():
                    sched_logger.info("_job_dstop_escalate: выходной — пропуск")
                    return
                try:
                    await _dstop_escalate(ctx.bot)
                except Exception as e:
                    sched_logger.error("debt_stop escalate error: %s", e)

            async def _job_dstop_saida(ctx):
                from bot.workday_checker import is_holiday_today
                if is_holiday_today():
                    sched_logger.info("_job_dstop_saida: выходной — пропуск")
                    return
                try:
                    await _dstop_saida(ctx.bot)
                except Exception as e:
                    sched_logger.error("debt_stop saida error: %s", e)

            job_queue.run_daily(
                _job_dstop_monitor,
                time=dt_time(14, 0, tzinfo=TZ),
                name="debt_stop_monitor",
            )
            sched_logger.info("🚫 Настроен мониторинг авто-стопа: ежедневно 14:00")

            job_queue.run_daily(
                _job_dstop_managers,
                time=dt_time(16, 30, tzinfo=TZ),
                name="debt_stop_managers",
            )
            sched_logger.info("🚫 Настроен запрос менеджерам по стоп-листу: ежедневно 16:30 (до коллектора 17:00)")

            job_queue.run_repeating(
                _job_dstop_manager_reminders,
                interval=1800,
                first=1800,
                name="debt_stop_manager_reminders",
            )
            sched_logger.info("🚫 Настроены напоминания менеджерам по стоп-листу: каждые 30 мин")

            job_queue.run_daily(
                _job_dstop_escalate,
                time=dt_time(18, 30, tzinfo=TZ),
                name="debt_stop_escalate",
            )
            sched_logger.info("🚫 Настроена эскалация к руководителю: ежедневно 18:30 (до WA cutoff 19:30)")

            job_queue.run_daily(
                _job_dstop_saida,
                time=dt_time(22, 15, tzinfo=TZ),
                name="debt_stop_saida",
            )
            sched_logger.info("🚫 Настроено уведомление Саиды: ежедневно 22:15")

            if _dstop_saida_hold_reminders:
                async def _job_saida_hold_reminders(ctx):
                    from bot.workday_checker import is_holiday_today
                    if is_holiday_today():
                        return
                    try:
                        await _dstop_saida_hold_reminders(ctx.bot)
                    except Exception as e:
                        sched_logger.error("saida_hold_reminders error: %s", e)

                job_queue.run_repeating(
                    _job_saida_hold_reminders,
                    interval=3600,
                    first=600,
                    name="saida_hold_reminders",
                )
                sched_logger.info("🚫 Настроен SLA-контроль оплат Саиды: каждый час")

        sched_logger.info(f"🗑️ Автоудаление сообщений через {AUTO_DELETE_HOURS} часов")
    
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


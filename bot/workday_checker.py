#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot/workday_checker.py
Определение рабочего / выходного дня.

Версия: 1.2.0 (2026-04-09)

Логика:
  1. is_workday_today() — проверяет наличие xlsx/xlsx.work файлов от whitelist-отправителей
     за сегодня в reports/queue/ и reports/excel/processed/.
     Если файлы есть — рабочий день.
  2. set_holiday(date_str) / clear_holiday(date_str) — ручная установка флага
     выходного дня администратором (через кнопку в боте).
  3. is_holiday_today() — основная функция для всех планировщиков.
     True = выходной, уведомления не отправляем.

Умолчания:
  - Воскресенье всегда считается выходным, если администратор явно не установил
    флаг False (рабочий день) для этой конкретной даты.

Изменения 1.2.0:
  - Исправлен EXCEL_DIR: ROOT_DIR/excel → reports/excel (соответствует реальной структуре)
  - has_xlsx_today() теперь ищет и *.xlsx.work (файл в очереди = точно сегодня пришёл)
  - needs_admin_confirmation() теперь отсекает запросы после 13:00 и при уже отправленном вопросе
  - Новые функции: mark_asked_today() / was_asked_today() — защита от дублей при рестарте бота
"""

import json
import logging
import os
import tempfile
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from zoneinfo import ZoneInfo

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env",
            encoding="utf-8-sig", override=False)

TZ       = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))
ROOT_DIR = Path(__file__).resolve().parent.parent
# Корень excel-данных — reports/excel/ (не excel/ в корне проекта)
_REPORTS_EXCEL_DIR = ROOT_DIR / "reports" / "excel"
# Директории где могут лежать xlsx и xlsx.work
_XLSX_SEARCH_DIRS = [
    ROOT_DIR / "reports" / "queue",
    _REPORTS_EXCEL_DIR,
    _REPORTS_EXCEL_DIR / "clean",
]
FLAG_FILE = ROOT_DIR / "reports" / "holiday_flags.json"   # {date_str: true/false, date_str+"_asked": true}

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Флаги (JSON-файл)
# ─────────────────────────────────────────────

def _load_flags() -> dict:
    if not FLAG_FILE.exists():
        return {}
    try:
        with open(FLAG_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_flags(flags: dict) -> None:
    FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8",
            dir=FLAG_FILE.parent, suffix=".tmp", delete=False,
        )
        json.dump(flags, tmp, ensure_ascii=False, indent=2)
        tmp.close()
        os.replace(tmp.name, FLAG_FILE)
    except OSError as e:
        logger.error("Ошибка записи holiday_flags.json: %s", e)


def set_holiday(date_str: str) -> None:
    """Отмечает дату как выходной (admin подтвердил)."""
    flags = _load_flags()
    flags[date_str] = True
    _save_flags(flags)
    logger.info("Выходной установлен: %s", date_str)


def clear_holiday(date_str: str) -> None:
    """Снимает флаг выходного — рабочий день."""
    flags = _load_flags()
    flags[date_str] = False
    _save_flags(flags)
    logger.info("Рабочий день подтверждён: %s", date_str)


def mark_asked_today() -> None:
    """Записывает что вопрос о рабочем дне уже был отправлен сегодня.
    Предотвращает дублирование при рестарте бота."""
    today = _today_str()
    flags = _load_flags()
    flags[f"{today}_asked"] = True
    _save_flags(flags)
    logger.debug("Флаг asked_today установлен: %s", today)


def was_asked_today() -> bool:
    """True если вопрос о рабочем дне уже был отправлен сегодня."""
    today = _today_str()
    return bool(_load_flags().get(f"{today}_asked", False))


# ─────────────────────────────────────────────
# Проверка xlsx-активности
# ─────────────────────────────────────────────

def _today_str() -> str:
    from datetime import datetime
    return datetime.now(TZ).strftime("%Y-%m-%d")


def has_xlsx_today() -> bool:
    """
    Возвращает True если сегодня в reports/queue/ или reports/excel/ появился
    хотя бы один xlsx или xlsx.work файл (= от whitelist-отправителя через imap_fetcher).

    Ищет *.xlsx и *.xlsx.work — последний означает файл ещё в обработке,
    но точно пришёл сегодня.
    """
    today = _today_str()
    from datetime import datetime

    patterns = ["*.xlsx", "*.xlsx.work"]
    for search_dir in _XLSX_SEARCH_DIRS:
        if not search_dir.exists():
            continue
        for pattern in patterns:
            for p in search_dir.rglob(pattern):
                try:
                    mtime = datetime.fromtimestamp(p.stat().st_mtime, TZ)
                    if mtime.strftime("%Y-%m-%d") == today:
                        logger.debug("xlsx сегодня найден: %s", p.name)
                        return True
                except OSError:
                    continue
    return False


# ─────────────────────────────────────────────
# Основная функция
# ─────────────────────────────────────────────

def is_holiday_today() -> bool:
    """
    True = сегодня выходной, уведомления менеджерам не отправлять.

    Приоритет:
      1. Флаг установлен администратором (holiday_flags.json) → используем его
         (в т.ч. если admin явно пометил воскресенье как рабочий день → False)
      2. Флага нет, и сегодня воскресенье → выходной (True) по умолчанию
      3. Флага нет, не воскресенье → смотрим на xlsx-активность:
           xlsx/xlsx.work есть → рабочий день (False)
           нет файлов         → предполагаем рабочий (False), check_workday_task спросит в 09:30
    """
    from datetime import datetime
    today = _today_str()
    flags = _load_flags()

    if today in flags:
        result = bool(flags[today])
        logger.debug("Флаг дня %s: %s", today, "выходной" if result else "рабочий")
        return result

    # Воскресенье — выходной по умолчанию (суббота — рабочий день)
    if datetime.now(TZ).weekday() == 6:  # 6 = Sunday
        logger.debug("Воскресенье %s — выходной по умолчанию", today)
        return True

    # Флага нет — смотрим xlsx/xlsx.work
    if has_xlsx_today():
        return False   # точно рабочий

    # Xlsx не пришли, флага нет — неизвестно (возможно ещё рано)
    return False        # по умолчанию не блокируем; check_workday_task спросит в 09:30


def needs_admin_confirmation() -> bool:
    """
    True если к 09:30 не пришло xlsx и администратор ещё не подтвердил статус дня.
    Используется в check_workday_task чтобы не спрашивать дважды.

    Возвращает False если:
      - Флаг уже установлен администратором (ответил на вопрос)
      - Файлы xlsx/xlsx.work уже есть сегодня (точно рабочий день)
      - Вопрос уже был отправлен сегодня (защита от дублей при рестарте)
      - Текущее время после 13:00 (слишком поздно для такого вопроса)
    """
    from datetime import datetime
    today = _today_str()
    flags = _load_flags()

    # Администратор уже ответил
    if today in flags:
        return False

    # Файлы уже пришли — рабочий день очевиден
    if has_xlsx_today():
        return False

    # Вопрос уже был отправлен сегодня (защита от дублей при рестарте бота)
    if flags.get(f"{today}_asked", False):
        return False

    # После 13:00 спрашивать бессмысленно
    if datetime.now(TZ).hour >= 13:
        logger.debug("needs_admin_confirmation: пропуск — уже %d:xx, слишком поздно",
                     datetime.now(TZ).hour)
        return False

    return True

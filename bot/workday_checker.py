#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot/workday_checker.py
Определение рабочего / выходного дня.

Версия: 1.1.0 (2026-03-25)

Логика:
  1. is_workday_today() — проверяет наличие xlsx-файлов от whitelist-отправителей
     за сегодня в директории excel/ (входящие) или excel/processed/ (обработанные).
     Если файлы есть — рабочий день.
  2. set_holiday(date_str) / clear_holiday(date_str) — ручная установка флага
     выходного дня администратором (через кнопку в боте).
  3. is_holiday_today() — основная функция для всех планировщиков.
     True = выходной, уведомления не отправляем.

Умолчания:
  - Воскресенье всегда считается выходным, если администратор явно не установил
    флаг False (рабочий день) для этой конкретной даты.
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
EXCEL_DIR = ROOT_DIR / "excel"
FLAG_FILE = ROOT_DIR / "reports" / "holiday_flags.json"   # {date_str: true/false}

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


# ─────────────────────────────────────────────
# Проверка xlsx-активности
# ─────────────────────────────────────────────

def _today_str() -> str:
    from datetime import datetime
    return datetime.now(TZ).strftime("%Y-%m-%d")


def has_xlsx_today() -> bool:
    """
    Возвращает True если сегодня в excel/ или excel/processed/ появился
    хотя бы один xlsx-файл (= от whitelist-отправителя через imap_fetcher).
    """
    today = _today_str()
    from datetime import datetime

    for search_dir in (EXCEL_DIR, EXCEL_DIR / "processed"):
        if not search_dir.exists():
            continue
        for p in search_dir.rglob("*.xlsx"):
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
           xlsx есть → рабочий день (False)
           xlsx нет  → предполагаем рабочий (False), check_workday_task спросит в 12:00
    """
    from datetime import datetime
    today = _today_str()
    flags = _load_flags()

    if today in flags:
        result = bool(flags[today])
        logger.debug("Флаг дня %s: %s", today, "выходной" if result else "рабочий")
        return result

    # Воскресенье — выходной по умолчанию
    if datetime.now(TZ).weekday() == 6:  # 6 = Sunday
        logger.debug("Воскресенье %s — выходной по умолчанию", today)
        return True

    # Флага нет — смотрим xlsx
    if has_xlsx_today():
        return False   # точно рабочий

    # Xlsx не пришли, флага нет — неизвестно (возможно ещё рано)
    return False        # по умолчанию не блокируем; check_workday_task спросит в 12:00


def needs_admin_confirmation() -> bool:
    """
    True если к 12:00 не пришло xlsx и администратор ещё не подтвердил статус дня.
    Используется в check_workday_task чтобы не спрашивать дважды.
    """
    today = _today_str()
    flags = _load_flags()
    if today in flags:
        return False      # уже подтверждено
    return not has_xlsx_today()

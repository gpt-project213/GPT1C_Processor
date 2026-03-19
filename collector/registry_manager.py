#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/registry_manager.py
Управление реестром должников: авторегистрация + Excel-экспорт.

Версия: 1.0.0 (2026-03-18)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from zoneinfo import ZoneInfo

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env",
            encoding="utf-8-sig", override=False)

TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))

ROOT_DIR = Path(__file__).resolve().parent.parent
CONTACTS_PATH = ROOT_DIR / "config" / "debtors_contacts.json"
REGISTRY_XLSX = ROOT_DIR / "config" / "debtors_registry.xlsx"

logger = logging.getLogger(__name__)


def _load_contacts() -> Dict[str, Any]:
    if not CONTACTS_PATH.exists():
        return {}
    try:
        with open(CONTACTS_PATH, encoding="utf-8") as f:
            d = json.load(f)
        d.pop("_comment", None)
        return d
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Ошибка чтения debtors_contacts.json: %s", e)
        return {}


def _save_contacts(contacts: Dict[str, Any]) -> bool:
    CONTACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".tmp",
            dir=CONTACTS_PATH.parent, delete=False,
        ) as tmp:
            json.dump(contacts, tmp, ensure_ascii=False, indent=2)
            tmp_path = Path(tmp.name)
        tmp_path.replace(CONTACTS_PATH)
        return True
    except OSError as e:
        logger.error("Ошибка записи debtors_contacts.json: %s", e)
        return False


def auto_register_client(
    name: str,
    manager: str = "",
    amount: float = 0.0,
    days: int = 0,
    violation: bool = False,
) -> bool:
    """Автоматически добавляет клиента в реестр без контактных данных.

    Телефон и Telegram ID остаются пустыми — менеджер должен заполнить через бот.
    Возвращает True если клиент был добавлен (новый), False если уже существует.
    """
    contacts = _load_contacts()

    # Проверяем — может уже есть (точное совпадение или нечёткое)
    if name in contacts:
        return False
    # Нечёткая проверка по lower
    name_lower = name.lower().strip()
    for key in contacts:
        if key.lower().strip() == name_lower:
            return False

    contacts[name] = {
        "whatsapp": "",
        "telegram_id": "",
        "manager": manager,
        "language": "ru",
        "do_not_call": False,
        "_auto_registered": True,
        "_needs_phone": True,
        "_debt_amount": round(amount, 2),
        "_days_overdue": days,
        "_violation": violation,
        "_registered_date": datetime.now(tz=TZ).date().isoformat(),
    }

    if not _save_contacts(contacts):
        return False

    logger.info("Авторегистрация: %s (менеджер=%s, долг=%.0f, дней=%d)", name, manager, amount, days)
    export_registry_excel(contacts)
    return True


def update_client_phone(name: str, phone: str) -> bool:
    """Обновляет номер WhatsApp клиента в реестре и сбрасывает флаг needs_phone.

    Возвращает True при успехе.
    """
    contacts = _load_contacts()
    if name not in contacts:
        logger.warning("update_client_phone: клиент не найден: %s", name)
        return False

    contacts[name]["whatsapp"] = phone
    contacts[name]["_needs_phone"] = False
    if not _save_contacts(contacts):
        return False

    logger.info("Телефон обновлён для %s: %s", name, phone)
    export_registry_excel(contacts)
    return True


def update_client_display_name(name: str, display_name: str) -> bool:
    """Сохраняет откорректированное отображаемое имя клиента (псевдоним).

    Ключ в базе (имя из 1С) остаётся неизменным — меняется только display_name,
    которое используется в сообщениях должнику.
    Возвращает True при успехе.
    """
    contacts = _load_contacts()
    if name not in contacts:
        logger.warning("update_client_display_name: клиент не найден: %s", name)
        return False

    contacts[name]["display_name"] = display_name.strip()
    if not _save_contacts(contacts):
        return False

    logger.info("display_name обновлён для %s: %s", name, display_name.strip())
    export_registry_excel(contacts)
    return True


def export_registry_excel(contacts: Optional[Dict[str, Any]] = None) -> bool:
    """Экспортирует реестр должников в Excel для просмотра администратором.

    Файл: config/debtors_registry.xlsx
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        logger.warning("openpyxl не установлен — Excel экспорт недоступен")
        return False

    if contacts is None:
        contacts = _load_contacts()

    wb = Workbook()
    ws = wb.active
    ws.title = "Реестр должников"

    # --- Заголовки ---
    headers = [
        "Статус", "Клиент (1С)", "Имя для сообщений", "WhatsApp", "Telegram ID",
        "Менеджер", "Язык", "Не звонить",
        "Долг (тг)", "Дней просрочки", "Нарушение", "Дата добавления",
    ]
    header_fill = PatternFill("solid", fgColor="1A3A5C")
    header_font = Font(bold=True, color="FFFFFF")
    for col_idx, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx, value=h)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    ws.row_dimensions[1].height = 28

    # --- Данные ---
    green_fill = PatternFill("solid", fgColor="E2EFDA")
    orange_fill = PatternFill("solid", fgColor="FCE4D6")
    bold_font = Font(bold=True)

    for row_idx, (name, info) in enumerate(contacts.items(), 2):
        if not isinstance(info, dict):
            continue
        needs_phone = info.get("_needs_phone", False) or not info.get("whatsapp")
        status = "⚠️ Нет телефона" if needs_phone else "✅ Готов"
        row_fill = orange_fill if needs_phone else green_fill

        values = [
            status,
            name,
            info.get("display_name", ""),
            info.get("whatsapp", ""),
            info.get("telegram_id", ""),
            info.get("manager", ""),
            info.get("language", "ru"),
            "Да" if info.get("do_not_call") else "Нет",
            info.get("_debt_amount", ""),
            info.get("_days_overdue", ""),
            "Да" if info.get("_violation") else "",
            info.get("_registered_date", ""),
        ]
        for col_idx, val in enumerate(values, 1):
            cell = ws.cell(row=row_idx, column=col_idx, value=val)
            cell.fill = row_fill
            if col_idx == 1:
                cell.font = bold_font
            cell.alignment = Alignment(vertical="center")

    # --- Ширина столбцов ---
    col_widths = [16, 45, 30, 16, 14, 14, 8, 12, 14, 14, 12, 16]
    for col_idx, width in enumerate(col_widths, 1):
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    ws.freeze_panes = "A2"

    try:
        wb.save(REGISTRY_XLSX)
        logger.info("Excel реестр сохранён: %s (%d записей)", REGISTRY_XLSX.name, len(contacts))
        return True
    except OSError as e:
        logger.error("Ошибка сохранения Excel реестра: %s", e)
        return False

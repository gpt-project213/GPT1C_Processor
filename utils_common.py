"""
utils_common.py · v1.1.0 (2026-03-11)
──────────────────────────────────────────────────────────────────────────────
Общие утилиты для аналитических модулей GPT1C_Processor.

Содержит функции, которые ранее дублировались между модулями:
  - normalize_client_name  (была в revenue_concentration_report.py)
  - slugify_safe           (безопасный slug для HTML-id и имён файлов)
  - clean_number           (строка/число → float, 0.0 при ошибке)

Fix #RC-1: вынесено из revenue_concentration_report.py.
v1.1.0: добавлены slugify_safe и clean_number — требуются utils.py.
"""
from __future__ import annotations

import re
from typing import Any

__VERSION__ = "1.1.0"


def normalize_client_name(name: str) -> str:
    """
    Нормализует имя клиента для дедупликации при мёрже нескольких JSON.
    Убирает префиксы менеджера, скобки, спецсимволы, схлопывает пробелы.

    Примеры:
      «О ТД Асем (холодильник № 4)» → «тд асем холодильник 4»
      «М ИП Иванов»                 → «ип иванов»
    """
    s = str(name).lower().strip()
    s = re.sub(r"^[оаемOАЕМ]\s+", "", s)   # убираем префикс менеджера
    s = re.sub(r"[^\w\s]", " ", s)          # убираем спецсимволы
    s = re.sub(r"\s+", " ", s)              # схлопываем пробелы
    return s.strip()


def slugify_safe(txt: str) -> str:
    """
    Преобразует строку в безопасный slug для HTML-идентификаторов и имён файлов.
    Кириллица сохраняется (транслитерация не выполняется), допустимы [0-9a-z_-а-я].

    Примеры:
      «Валовая прибыль / Ергали» → «валовая_прибыль_ергали»
      «ТОО Асем (2025)»          → «тоо_асем_2025»
    """
    s = str(txt or "").strip().lower()
    s = s.replace("ё", "е")
    s = re.sub(r"[\s/\\]+", "_", s)           # пробелы и слеши → подчёркивание
    s = re.sub(r"[^0-9a-zа-я_\-]", "_", s)   # всё остальное → подчёркивание
    s = re.sub(r"__+", "_", s)
    return s.strip("_") or "row"


def clean_number(value: Any) -> float:
    """
    Преобразует строку или число в float.
    Убирает пробелы (в т.ч. NBSP), меняет запятую на точку.
    Возвращает 0.0 при ошибке или пустом значении.

    Примеры:
      «1 234 567,89» → 1234567.89
      «—»            → 0.0
      None           → 0.0
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    s = re.sub(r"[\s\u00A0\u202F\u2007\u200B]", "", s)  # пробелы и NBSP
    s = s.replace(",", ".")
    s = re.sub(r"[^\d.\-]", "", s)
    if not s or s in ("-", "."):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0
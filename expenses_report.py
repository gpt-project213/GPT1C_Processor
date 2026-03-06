#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expenses_report.py · v2.0.0 (2026-03-06)
──────────────────────────────────────────
ИСПРАВЛЕНИЕ B1: Устранён дубль класса UnifiedExpensesParser.

Ранее этот файл содержал полную копию класса UnifiedExpensesParser,
идентичную expenses_parser.py. При независимом изменении одного из файлов
второй оставался устаревшим — поведение становилось непредсказуемым.

Теперь expenses_report.py является тонкой обёрткой над expenses_parser.py:
- весь код класса живёт только в expenses_parser.py (v1.1.0+)
- этот файл реэкспортирует всё необходимое для обратной совместимости
- любой код, импортирующий из expenses_report, продолжает работать без изменений

Импорт для pipeline и бота:
    from expenses_report import UnifiedExpensesParser   # работает
    from expenses_report import parse_file              # работает
    from expenses_parser import UnifiedExpensesParser   # канонический вариант
"""

import warnings as _warnings

# Canonical source — all logic lives here
from expenses_parser import (
    UnifiedExpensesParser,
    PeriodInfo,
    MONTHS_RU,
    NBSP_NARROW,
    _safe_isna,
    parse_file,
)

# Deprecation notice for direct imports from this module
_warnings.warn(
    "expenses_report.py is a compatibility shim. "
    "Import directly from expenses_parser instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "UnifiedExpensesParser",
    "PeriodInfo",
    "MONTHS_RU",
    "NBSP_NARROW",
    "_safe_isna",
    "parse_file",
]
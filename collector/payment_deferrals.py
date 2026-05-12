#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/payment_deferrals.py
Договорные сроки отсрочки платежа.

v1.0.0 (2026-05-12)

Клиенты из config/payment_deferrals.json имеют N дней отсрочки по договору.
Просрочка считается только после истечения этого срока.

effective_days = max(0, actual_days - deferral_days)

Если клиент не в списке — отсрочки нет (дефолт: предоплата / 0 дней).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ROOT          = Path(__file__).resolve().parent.parent
_DEFERRALS_PATH = _ROOT / "config" / "payment_deferrals.json"

_cache: dict = {}


def _load() -> dict:
    global _cache
    if _cache:
        return _cache
    try:
        with open(_DEFERRALS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        _cache = data.get("deferrals", {})
        logger.info("payment_deferrals: загружено %d записей", len(_cache))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("payment_deferrals: ошибка загрузки: %s", e)
        _cache = {}
    return _cache


def get_deferral_days(client_name: str) -> int:
    """Возвращает количество дней отсрочки для клиента (0 если нет договора)."""
    return _load().get(client_name, {}).get("days", 0)


def effective_overdue_days(client_name: str, actual_days: int) -> int:
    """Эффективные дни просрочки с учётом отсрочки по договору."""
    deferral = get_deferral_days(client_name)
    return max(0, actual_days - deferral)


def has_deferral(client_name: str) -> bool:
    return client_name in _load()


def all_deferred_clients() -> dict:
    """Все клиенты с договорными отсрочками."""
    return dict(_load())


# Шкала давления для клиентов с договорной отсрочкой.
# Считается от effective_days (фактические дни минус отсрочка).
# Пороги тighter: effective=2 уже первое напоминание, effective=10 — критично.
_DEFERRAL_LEVEL_THRESHOLDS = [
    (10, 5),   # effective 10+ → критично
    (7,  4),   # effective 7–9 → серьёзно
    (5,  3),   # effective 5–6 → настойчиво
    (3,  2),   # effective 3–4 → умеренное давление
    (2,  1),   # effective 2   → первое напоминание
    (0,  0),
]


def deferral_level(effective_days: int) -> int:
    """Уровень давления для клиента с отсрочкой (по effective_days)."""
    for threshold, level in _DEFERRAL_LEVEL_THRESHOLDS:
        if effective_days >= threshold:
            return level
    return 0


def reload() -> None:
    """Сбросить кэш (вызывается после изменения конфига)."""
    global _cache
    _cache = {}

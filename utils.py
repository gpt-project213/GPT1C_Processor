#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils.py · v2.0.1 (29 Авг 2025)
────────────────────────────────────────────────────────────────────
Общие утилиты, используемые отчётными скриптами.

• money(x)          – формат «1 234 567,89»
• qty(x)            – целое / с одним знаком после запятой
• price(x)          – цена (три знака после запятой)
• slugify_safe(txt) – прокси из utils_common
• generated_at_tz() – прокси из config
• build_jinja_env() – единый Jinja-Environment с фильтрами
"""
from __future__ import annotations

import locale
import math
from pathlib import Path
from typing import Any

import jinja2
from babel.numbers import format_decimal

import config                       # ROOT/пути/TZ/логи, generated_at_tz()
from utils_common import slugify_safe, clean_number  # лёгкие хелперы

# ── версия и логи ──────────────────────────────────────────────────
__VERSION__ = "2.0.1"
log = config.setup_logging(Path(__file__).stem)
config.register_version(__name__, __VERSION__)

# ── прокси для generated_at_tz (важно для extended_debt_report) ───
generated_at_tz = config.generated_at_tz  # ← ключевая добавка

# ── локаль для числового форматирования (ru_KZ) ───────────────────
try:
    locale.setlocale(locale.LC_ALL, "ru_KZ.UTF-8")
except locale.Error:
    locale.setlocale(locale.LC_ALL, "")   # fallback к системной

NBSP_THIN = "\u202F"  # узкий неразрывный пробел


# ── форматирование чисел ──────────────────────────────────────────
def _fmt(num: float | int, frac: int = 2) -> str:
    if num is None or (isinstance(num, float) and math.isnan(num)):
        return ""
    return (
        format_decimal(num, format=f"#,##0.{ '0'*frac }", locale="ru_KZ")
        .replace("\u00A0", NBSP_THIN)
    )


def money(x: Any) -> str:
    """Деньги «1 234 567,89»"""
    try:
        return _fmt(float(x), 2)
    except (ValueError, TypeError):
        return str(x)


def qty(x: Any) -> str:
    """Количество: целое или с одной дробной цифрой"""
    try:
        val = float(x)
        return _fmt(val, 1 if val % 1 else 0)
    except (ValueError, TypeError):
        return str(x)


def price(x: Any) -> str:
    """Цена: три знака после запятой"""
    try:
        return _fmt(float(x), 3)
    except (ValueError, TypeError):
        return str(x)


# ── Jinja-environment ─────────────────────────────────────────────
def build_jinja_env() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(config.TEMPLATES_DIR)),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters.update(
        {
            "money": money,
            "qty": qty,
            "price": price,
            "slug": slugify_safe,
        }
    )
    # глобальные функции доступны прямо в шаблоне: {{ generated_at_tz() }}
    env.globals["generated_at_tz"] = generated_at_tz
    return env


# ── smoke-тест ────────────────────────────────────────────────────
if __name__ == "__main__":  # запуск: python utils.py
    print("generated_at:", generated_at_tz())
    for val in (1_234_567.891, 10, 0.5):
        print("money:", money(val), "qty:", qty(val), "price:", price(val))
    env = build_jinja_env()
    print("Jinja OK, filters:", list(env.filters.keys())[:4])

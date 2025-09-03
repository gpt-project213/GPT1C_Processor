"""
utils_common.py · baseline 2025-08-29
──────────────────────────────────────────────────────────────
Лёгкие утилиты без тяжёлых зависимостей.
"""

from __future__ import annotations
import re, unicodedata
from typing import Any

NBSP       = "\u00A0"
THIN_NBSP  = "\u202F"
_SPACES_RE = re.compile(f"[{NBSP}{THIN_NBSP}\\s]+")

def clean_number(value: Any) -> str:
    """
    '800 123,50'  → '800123.50'
    NBSP/тонкие NBSP/обычные пробелы → удалить; запятую → точку.
    """
    s = str(value)
    s = _SPACES_RE.sub("", s)
    return s.replace(",", ".").strip()

def slugify_safe(text: str, allow_dot: bool = False) -> str:
    """
    'Продажи Оксана (38).xlsx'  → 'prodazhi_oksana_38'
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\-.]" if allow_dot else r"[^\w\-]", "_", text.lower())
    text = re.sub(r"__+", "_", text).strip("_")
    return text[:255]      # безопасная длина для Windows/FAT

__all__ = ["clean_number", "slugify_safe"]

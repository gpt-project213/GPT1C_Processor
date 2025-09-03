# -*- coding: utf-8 -*-
"""
detectors.py — безопасный «сниффер» без парсинга таблиц.
Правила:
  • Тип "дебиторка": префикс 'debt*' / слово 'дебитор*' в имени файла
    ИЛИ точная фраза "Ведомость по взаиморасчётам с контрагентами"
    в первых килобайтах HTML/шапки XLSX.
  • «Чей отчёт»: ТОЛЬКО из ИМЕНИ ФАЙЛА. Нет имени — сводный (только админу).
"""

from __future__ import annotations
from pathlib import Path
import re
from typing import Dict, List, Optional

_DEBT_KEY = "ведомость по взаиморасчетам с контрагентами"   # без 'ё' для надёжности

RE_NAME_FROM_STEM = re.compile(
    r"""(?ix)
    ^
    (?:debt(?:_ext|_ai)?)         # префикс
    (?:[-_. ]\d{4}[-_.]\d{2}(?:[-_.]\d{2})?)? # опц. дата
    [-_. ](?P<name>[^.]+)$        # имя менеджера до расширения
    """
)

def _norm(s: str) -> str:
    return " ".join(str(s or "").strip().lower().replace("ё","е").split())

def is_debt_by_filename(path: Path) -> bool:
    s = path.stem.lower().replace("ё","е")
    return s.startswith("debt") or ("дебитор" in s)

def is_debt_by_html_head(html_text: str) -> bool:
    head = _norm(html_text[:4000])
    return ("дебитор" in head) or (_DEBT_KEY in head)

def detect_report_type(path: Path) -> str:
    if is_debt_by_filename(path):
        return "debt"
    if path.suffix.lower() == ".html":
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
            if is_debt_by_html_head(txt):
                return "debt"
        except Exception:
            pass
    return "unknown"

def detect_manager_from_filename(path: Path, managers_synonyms: Dict[str, List[str]]) -> Optional[str]:
    m = RE_NAME_FROM_STEM.match(path.stem)
    if not m:
        return None
    raw = m.group("name").strip().lower()
    for tg, variants in managers_synonyms.items():
        for v in variants:
            if raw == v.lower():
                return tg
    return None

def compute_recipients(
    *,
    path: Path,
    managers_synonyms: Dict[str, List[str]],
    admins: List[str],
    watchers: Dict[str, List[str]] | None = None,
) -> List[str]:
    watchers = watchers or {}
    recips = set(admins)
    manager_tg = detect_manager_from_filename(path, managers_synonyms)
    if manager_tg:
        recips.add(manager_tg)
        for watcher, scope in watchers.items():
            if manager_tg in scope:
                recips.add(watcher)
    return sorted(recips)

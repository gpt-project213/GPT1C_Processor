#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
config.py · v3.2 · 2025-09-02
Инварианты путей/логов/TZ/футера по ТЗ. Подгрузка managers.yaml и pattern_config.yaml.
Никакой бизнес-логики парсинга не меняет.

ТЗ (ключевые пункты):
• HTML → reports/html; JSON (extended) → reports/json; PDF → reports/pdf.  [§3, §5, §8] 
• Логи: logs/<module>_YYYYMMDD_HHMMSS.log, формат "%(asctime)s, %(levelname)s %(message)s"; TZ=Asia/Almaty.  [§3, §6]
• Имена: простой {stem}_debt.html; расширенный debt_ext_{stem}.html/.json.  [§3]
• Футер: "Сформировано: DD.MM.YYYY HH:MM (Asia/Almaty) | Версия: …".  [§3, §4]
• Стиль HTML единый «как в продажах».  [§3, §4]
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import logging
from logging import Logger
from typing import Any, Dict, Optional

try:
    from zoneinfo import ZoneInfo  # py3.11+
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore

import json
import yaml

# ──────────────────────────────────────────────────────────────
# БАЗОВЫЕ ПУТИ
# ──────────────────────────────────────────────────────────────
ROOT: Path = Path(__file__).resolve().parent

# reports/*
REPORTS_DIR: Path       = ROOT / "reports"
HTML_DIR: Path          = REPORTS_DIR / "html"    # HTML → reports/html  [ТЗ §3, §5]
JSON_DIR: Path          = REPORTS_DIR / "json"    # JSON (extended) → reports/json  [ТЗ §5]
PDF_DIR: Path           = REPORTS_DIR / "pdf"     # PDF → reports/pdf (нижний регистр)  [Твои требования]
EXCEL_CLEAN_DIR: Path   = REPORTS_DIR / "excel"   # чистые .__clean.xlsx (TTL 90д)

QUEUE_DIR: Path         = REPORTS_DIR / "queue"   # входящие «грязные» xlsx  [ТЗ §5, §6]
TEMPLATES_DIR: Path     = ROOT / "templates"
LOGS_DIR: Path          = ROOT / "logs"
CACHE_DIR: Path         = ROOT / "cache"
STATE_DIR: Path         = CACHE_DIR / "state"     # служебные индексы/состояния (TTL 90д)

CONFIG_DIR: Path        = ROOT / "config"
MANAGERS_YAML: Path     = CONFIG_DIR / "managers.yaml"
PATTERN_YAML: Path      = CONFIG_DIR / "pattern_config.yaml"

DEFAULT_TZ = "Asia/Almaty"                        # [ТЗ §3, §4, §6]

def _ensure_dirs() -> None:
    for p in (REPORTS_DIR, HTML_DIR, JSON_DIR, PDF_DIR,
              EXCEL_CLEAN_DIR, QUEUE_DIR, LOGS_DIR, TEMPLATES_DIR,
              CACHE_DIR, STATE_DIR, CONFIG_DIR):
        p.mkdir(parents=True, exist_ok=True)

_ensure_dirs()

# ──────────────────────────────────────────────────────────────
# ВЕРСИИ МОДУЛЕЙ (для футера/диагностики)
# ──────────────────────────────────────────────────────────────
__VERSIONS: Dict[str, str] = {}

def register_version(module_key: str, version: str) -> None:
    """Регистрирует версию модуля для последующего вывода в футере."""
    __VERSIONS[module_key] = str(version)

def get_versions_line() -> str:
    """Собирает компактную строку версий, например: debt_simple=v93; debt_ext=v41"""
    if not __VERSIONS:
        return ""
    parts = [f"{k}={v}" for k, v in __VERSIONS.items()]
    return "; ".join(parts)

# ──────────────────────────────────────────────────────────────
# ФОРМАТ ВРЕМЕНИ/ФУТЕР
# ──────────────────────────────────────────────────────────────
_TZ = ZoneInfo(DEFAULT_TZ)

def generated_at_tz(version: Optional[str] = None) -> str:
    """
    Продуцирует строку футера:
    «Сформировано: DD.MM.YYYY HH:MM (Asia/Almaty) | Версия: <...>»
    """
    now = datetime.now(_TZ).strftime("%d.%m.%Y %H:%M")
    ver = version or get_versions_line()
    base = f"Сформировано: {now} ({DEFAULT_TZ})"
    return f"{base} | Версия: {ver}" if ver else base

# ──────────────────────────────────────────────────────────────
# ЛОГИРОВАНИЕ (формат/ТЗ с TZ=Asia/Almaty)
# ──────────────────────────────────────────────────────────────
class _TzFormatter(logging.Formatter):
    def __init__(self, fmt: str, datefmt: str, tz: ZoneInfo):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._tz = tz
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, self._tz)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()

_LOG_FORMAT = "%(asctime)s, %(levelname)s %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

def setup_logging(module_name: str, level: int = logging.INFO) -> Logger:
    """
    Создаёт логгер <module_name> с файлом logs/<module>_YYYYMMDD_HHMMSS.log
    и выводом в консоль. Формат из ТЗ. TZ=Asia/Almaty.  [ТЗ §3, §6]
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    # не плодим дубликаты хендлеров при повторном импорте
    if logger.handlers:
        return logger

    ts = datetime.now(_TZ).strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{module_name}_{ts}.log"

    fh = logging.FileHandler(log_path, encoding="utf-8")
    sh = logging.StreamHandler()

    fmt = _TzFormatter(_LOG_FORMAT, _LOG_DATEFMT, _TZ)
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger

# ──────────────────────────────────────────────────────────────
# ЗАГРУЗКА ВНЕШНИХ КОНФИГОВ (без изменения бизнес-логики)
# ──────────────────────────────────────────────────────────────
def _read_yaml(p: Path) -> Dict[str, Any]:
    try:
        if p.exists():
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        pass
    return {}

def load_managers_config() -> Dict[str, Any]:
    """
    managers.yaml:
      timezone: Asia/Almaty
      admins: [<chat_id>...]
      managers: {Имя: "<chat_id>"}
      synonyms: {Имя: [варианты...]}
    Используется для:
      • фильтр «менеджер ≠ клиент» (ТОЛЬКО на выдачу; расчёты не трогаем)
      • рассылка в боте (если chat_id заполнены)
    """
    cfg = _read_yaml(MANAGERS_YAML)
    # подстрахуем структуру
    cfg.setdefault("timezone", DEFAULT_TZ)
    cfg.setdefault("admins", [])
    cfg.setdefault("managers", {})
    cfg.setdefault("synonyms", {})
    return cfg

def load_pattern_config() -> Dict[str, Any]:
    """
    pattern_config.yaml:
      regex: { total, meta, client }
    Подключаем без изменения бизнес-логики; пригодится детекторам/диагностике.
    """
    cfg = _read_yaml(PATTERN_YAML)
    cfg.setdefault("regex", {})
    return cfg

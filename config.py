#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
config.py · v3.6.1 · 2026-03-10

Совместимость с вашим кодом:
• utils_excel.py → EXCEL_CLEAN_DIR, QUEUE_DIR, setup_logging
• utils.py       → TEMPLATES_DIR, generated_at_tz, register_version, setup_logging
• debt_auto_report.py → import config, generated_at_tz, MANAGERS_CFG

Инварианты из ТЗ:
• HTML → reports/html; JSON → reports/json
• Логи: logs/<module>_YYYYMMDD_HHMMSS.log; "%(asctime)s, %(levelname)s %(message)s"; TZ из .env (по умолчанию Asia/Almaty)
• Футер: "Сформировано: DD.MM.YYYY HH:MM (Asia/Almaty) | Версия: …"
• reports_state.json — в корне проекта
• Источник истины менеджеров — config/managers.json; managers.yaml/ pattern_config.yaml — опциональны
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import logging
from logging import Logger
import os, json

from dotenv import load_dotenv
from zoneinfo import ZoneInfo

# ── .env ──────────────────────────────────────────────────────
ROOT: Path = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT / ".env", encoding="utf-8-sig", override=True)

# ── TZ ────────────────────────────────────────────────────────
DEFAULT_TZ = "Asia/Almaty"
TZ = ZoneInfo(os.getenv("TZ", DEFAULT_TZ) or DEFAULT_TZ)

# ── Пути ──────────────────────────────────────────────────────
REPORTS_DIR: Path       = ROOT / "reports"
HTML_DIR: Path          = REPORTS_DIR / "html"
JSON_DIR: Path          = REPORTS_DIR / "json"
EXCEL_CLEAN_DIR: Path   = REPORTS_DIR / "excel"
QUEUE_DIR: Path         = REPORTS_DIR / "queue"
TEMPLATES_DIR: Path     = ROOT / "templates"
LOGS_DIR: Path          = ROOT / "logs"
CACHE_DIR: Path         = ROOT / "cache"

REPORTS_STATE_PATH: Path = ROOT / "reports_state.json"  # по ТЗ — в корне
STATE_FILE: str = str(REPORTS_STATE_PATH)               # алиас для совместимости

CONFIG_DIR: Path        = ROOT / "config"
MANAGERS_JSON: Path     = CONFIG_DIR / "managers.json"
MANAGERS_YAML: Path     = CONFIG_DIR / "managers.yaml"          # опционально
PATTERN_YAML: Path      = CONFIG_DIR / "pattern_config.yaml"     # опционально


def ensure_dirs() -> None:
    for p in (REPORTS_DIR, HTML_DIR, JSON_DIR, EXCEL_CLEAN_DIR, QUEUE_DIR, LOGS_DIR, CACHE_DIR, CONFIG_DIR):
        p.mkdir(parents=True, exist_ok=True)

ensure_dirs()

# ── Версии модулей для футера ─────────────────────────────────
__VERSIONS: Dict[str, str] = {}

def register_version(module_key: str, version: str) -> None:
    __VERSIONS[module_key] = str(version)

def get_versions_line() -> str:
    return "; ".join(f"{k}={v}" for k, v in __VERSIONS.items()) if __VERSIONS else ""

def generated_at_tz(version: Optional[str] = None) -> str:
    now = datetime.now(TZ).strftime("%d.%m.%Y %H:%M")
    ver = version or get_versions_line()
    base = f"Сформировано: {now} ({TZ.key})"
    return f"{base} | Версия: {ver}" if ver else base

# ── Логирование (файл+консоль) ────────────────────────────────
class _TzFormatter(logging.Formatter):
    def __init__(self, fmt: str, datefmt: str, tz: ZoneInfo):
        super().__init__(fmt=fmt, datefmt=datefmt); self._tz = tz
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, self._tz)
        return dt.strftime(datefmt) if datefmt else dt.isoformat()

_LOG_FORMAT = "%(asctime)s, %(levelname)s %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

def setup_logging(module_name: str, level: int = logging.INFO) -> Logger:
    """
    Логгер: logs/<module>_YYYYMMDD_HHMMSS.log + stdout. Формат из ТЗ. TZ из .env.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    ts = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{module_name}_{ts}.log"

    fh = logging.FileHandler(log_path, encoding="utf-8")
    sh = logging.StreamHandler()

    fmt = _TzFormatter(_LOG_FORMAT, _LOG_DATEFMT, TZ)
    fh.setFormatter(fmt); sh.setFormatter(fmt)

    logger.addHandler(fh); logger.addHandler(sh)
    logger.propagate = False
    return logger

# ── YAML (опционально) ────────────────────────────────────────
try:
    import yaml  # PyYAML может отсутствовать
except Exception:
    yaml = None  # type: ignore

def _read_yaml(p: Path) -> Dict[str, Any]:
    if not (p.exists() and yaml is not None):
        return {}
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

# ── Менеджеры и синонимы ──────────────────────────────────────
def _load_managers_json() -> Dict[str, Optional[int]]:
    if not MANAGERS_JSON.exists():
        import logging as _log
        _log.getLogger("config").warning("managers.json not found at %s", MANAGERS_JSON)
        return {}
    try:
        raw = json.loads(MANAGERS_JSON.read_text(encoding="utf-8")) or {}
    except (json.JSONDecodeError, OSError) as e:
        import logging as _log
        _log.getLogger("config").error("managers.json read error: %s", e)
        return {}
    result: Dict[str, Optional[int]] = {}
    if isinstance(raw, dict):
        for name, val in raw.items():
            result[str(name)] = int(val) if isinstance(val, int) else None
    return result

def _load_yaml_overrides() -> Dict[str, Any]:
    """
    managers.yaml (если есть):
      admins: [chat_id, ...]
      managers: {Имя: chat_id}         # НЕ переопределяем JSON-источник
      synonyms: {Имя: [синонимы...]}
    Используем только admins/synonyms как дополнение.
    """
    cfg = _read_yaml(MANAGERS_YAML)
    return {
        "admins": cfg.get("admins") or [],
        "synonyms": cfg.get("synonyms") or {},
    }

_json_mgr = _load_managers_json()
_yaml_extra = _load_yaml_overrides()

MANAGERS_CFG: Dict[str, Any] = {
    "admins": _yaml_extra.get("admins") or [],
    "managers": {name: {"chat_id": chat_id} for name, chat_id in _json_mgr.items()},
    "synonyms": _yaml_extra.get("synonyms") or {},
}

_SYSTEM_MANAGER_NAMES: frozenset = frozenset(
    n.strip().lower() for n in os.getenv("SYSTEM_MANAGERS", "Минай").split(",") if n.strip()
)

_PREFIX_MAP: Dict[str, str] = {}
for _name in _json_mgr:
    _n = _name.strip()
    if _n.lower() in _SYSTEM_MANAGER_NAMES or not _n:
        continue
    _letter = _n[0].upper()
    _PREFIX_MAP.setdefault(_letter, _n)


def get_manager_by_client_prefix(client_name: str) -> str:
    """Маппинг клиента на менеджера по первой букве (соглашение 1С)."""
    if not client_name or not client_name.strip():
        return "Неизвестно"
    letter = client_name.strip()[0].upper()
    return _PREFIX_MAP.get(letter, "Неизвестно")


def get_manager_names(*, exclude_system: bool = True) -> list[str]:
    """Список имён менеджеров из managers.json."""
    return [
        n for n in _json_mgr
        if not exclude_system or n.strip().lower() not in _SYSTEM_MANAGER_NAMES
    ]


# ── pattern_config.yaml (опционально) ─────────────────────────
def load_pattern_config() -> Dict[str, Any]:
    cfg = _read_yaml(PATTERN_YAML)
    cfg.setdefault("regex", {})
    return cfg

# ── AI / PDF (если нужны) ─────────────────────────────────────
AI_PROVIDER: str       = os.getenv("AI_PROVIDER", "deepseek")
AI_MODEL: str          = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_KEY: str  = os.getenv("DEEPSEEK_API_KEY", "") or ""
OPENAI_API_KEY: str    = os.getenv("OPENAI_API_KEY", "") or ""
AI_PROMPT_PATH: str    = os.getenv("AI_PROMPT_PATH", "ПРОМТ ДЛЯ ДЕБИТОРКИ.txt")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
config.py · v3.6 · 2025-09-08

Совместимость с вашим кодом:
• utils_excel.py → EXCEL_CLEAN_DIR, QUEUE_DIR, setup_logging
• utils.py       → TEMPLATES_DIR, generated_at_tz, register_version, setup_logging
• debt_auto_report.py → import config, generated_at_tz, MANAGERS_CFG

Инварианты из ТЗ:
• HTML → reports/html; JSON → reports/json; PDF → reports/pdf
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
PDF_DIR: Path           = REPORTS_DIR / "pdf"
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

TOOLS_WKHTMLTOPDF_DIR: Path = ROOT / "tools" / "wkhtmltopdf" / "bin"
WKHTMLTOPDF_BIN: Optional[str] = (
    os.getenv("WKHTMLTOPDF_BIN")
    or (str(TOOLS_WKHTMLTOPDF_DIR / "wkhtmltopdf.exe")
        if (TOOLS_WKHTMLTOPDF_DIR / "wkhtmltopdf.exe").exists()
        else None)
)

def ensure_dirs() -> None:
    for p in (REPORTS_DIR, HTML_DIR, JSON_DIR, PDF_DIR, EXCEL_CLEAN_DIR, QUEUE_DIR, LOGS_DIR, CACHE_DIR, CONFIG_DIR):
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
        raise FileNotFoundError(f"config/managers.json is missing at {MANAGERS_JSON}")
    raw = json.loads(MANAGERS_JSON.read_text(encoding="utf-8")) or {}
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
PDF_AUTO: bool         = os.getenv("PDF_AUTO", "true").lower() == "true"

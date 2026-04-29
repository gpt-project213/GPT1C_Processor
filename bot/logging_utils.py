#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot/logging_utils.py
Shared runtime logging helpers for the bot/orchestrator layer.
"""

from __future__ import annotations

import asyncio
import html
import io
import logging
import os
import sys
import time
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from zoneinfo import ZoneInfo

__VERSION__ = "1.0.0"

_ALERT_SENDER: Optional[Callable[[str], Awaitable[None]]] = None
_ALERT_HANDLER: Optional["TelegramErrorAlertHandler"] = None


class RuntimeFormatter(logging.Formatter):
    def __init__(self, tz: ZoneInfo):
        super().__init__("%(asctime)s, %(levelname)s %(message)s")
        self._tz = tz

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, self._tz)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        system = getattr(record, "system", "") or ""
        component = getattr(record, "component", "") or ""
        job = getattr(record, "job", "") or ""

        parts = []
        if system:
            parts.append(f"[{system}]")
        if component:
            parts.append(f"[{component}]")
        if job:
            parts.append(f"[{job}]")

        original_msg = record.msg
        original_args = record.args
        prefix = "".join(parts)
        if prefix:
            record.msg = f"{prefix} {record.getMessage()}"
            record.args = ()
        try:
            return super().format(record)
        finally:
            record.msg = original_msg
            record.args = original_args


class ContextAdapter(logging.LoggerAdapter):
    def __init__(
        self,
        base_logger: logging.Logger,
        *,
        system: str,
        component: str = "",
        job: str = "",
    ) -> None:
        super().__init__(base_logger, {"system": system, "component": component, "job": job})

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = dict(self.extra)
        user_extra = kwargs.get("extra") or {}
        extra.update(user_extra)
        kwargs["extra"] = extra
        return msg, kwargs


class TelegramErrorAlertHandler(logging.Handler):
    def __init__(
        self,
        *,
        level: int = logging.ERROR,
        cooldown_sec: int = 300,
        max_message_len: int = 1800,
    ) -> None:
        super().__init__(level=level)
        self._cooldown_sec = cooldown_sec
        self._max_message_len = max_message_len
        self._recent: dict[str, float] = {}

    def emit(self, record: logging.LogRecord) -> None:
        if _ALERT_SENDER is None:
            return
        system = getattr(record, "system", "") or "BOT"
        component = getattr(record, "component", "") or ""
        event = getattr(record, "event", "") or ""
        message = record.getMessage()
        if not message:
            return

        signature = f"{record.levelname}|{system}|{component}|{event}|{message[:300]}"
        now = time.monotonic()
        last_ts = self._recent.get(signature, 0.0)
        if now - last_ts < self._cooldown_sec:
            return
        self._recent[signature] = now

        text = self._format_alert_text(record, system, component, event, message)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        try:
            loop.create_task(_ALERT_SENDER(text))
        except Exception:
            return

    def _format_alert_text(
        self,
        record: logging.LogRecord,
        system: str,
        component: str,
        event: str,
        message: str,
    ) -> str:
        source = record.name
        lines = [
            "🚨 <b>Runtime error</b>",
            f"level: <code>{html.escape(record.levelname)}</code>",
            f"system: <code>{html.escape(system)}</code>",
        ]
        if component:
            lines.append(f"component: <code>{html.escape(component)}</code>")
        if event:
            lines.append(f"event: <code>{html.escape(event)}</code>")
        lines.append(f"source: <code>{html.escape(source)}</code>")
        lines.append("")
        lines.append(html.escape(message[: self._max_message_len]))
        return "\n".join(lines)


def configure_runtime_logging(
    *,
    logs_dir: Path,
    tz: ZoneInfo,
    app_name: str = "send_reports",
    level: int = logging.INFO,
    retention_days: int = 14,
    error_alert_level: int = logging.ERROR,
    alert_cooldown_sec: int = 300,
) -> TelegramErrorAlertHandler:
    global _ALERT_HANDLER

    logs_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    formatter = RuntimeFormatter(tz)

    log_path = logs_dir / f"{app_name}.log"
    stream_handler = logging.StreamHandler(
        io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    )
    stream_handler.setFormatter(formatter)

    alert_handler = TelegramErrorAlertHandler(
        level=error_alert_level,
        cooldown_sec=alert_cooldown_sec,
    )
    alert_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(alert_handler)

    try:
        file_handler = TimedRotatingFileHandler(
            log_path,
            when="midnight",
            interval=1,
            backupCount=retention_days,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except Exception:
        pass

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.INFO)

    _ALERT_HANDLER = alert_handler
    return alert_handler


def get_runtime_logger(
    name: str,
    *,
    system: str,
    component: str = "",
    job: str = "",
) -> ContextAdapter:
    return ContextAdapter(logging.getLogger(name), system=system, component=component, job=job)


def derive_system_for_module(module_name: str) -> tuple[str, str]:
    name = (module_name or "").lower()
    if any(token in name for token in ("parser", "report", "pipeline", "imap", "inventory", "sales", "gross", "expenses", "rfm", "dso", "aging", "turnover", "concentration")):
        return "PIPELINE", "MODULE"
    if "collector" in name:
        return "COLLECTOR", "MODULE"
    if "crm" in name:
        return "CRM", "MODULE"
    if "stop" in name or "shipment" in name:
        return "STOP_CONTROL", "MODULE"
    return "BOT", "MODULE"


def configure_module_logger(
    module_name: str,
    *,
    logs_dir: Path,
    tz: ZoneInfo,
    level: int = logging.INFO,
    retention_days: int = 14,
) -> logging.Logger:
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    if getattr(logger, "_gpt1c_configured", False):
        return logger

    system, component = derive_system_for_module(module_name)
    formatter = RuntimeFormatter(tz)
    log_path = logs_dir / f"{module_name}.log"

    try:
        file_handler = TimedRotatingFileHandler(
            log_path,
            when="midnight",
            interval=1,
            backupCount=retention_days,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_ContextDefaultsFilter(system=system, component=component))
        logger.addHandler(file_handler)
    except Exception:
        pass

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(_ContextDefaultsFilter(system=system, component=component))
    logger.addHandler(stream_handler)

    logger.propagate = False
    setattr(logger, "_gpt1c_configured", True)
    return logger


class _ContextDefaultsFilter(logging.Filter):
    def __init__(self, *, system: str, component: str) -> None:
        super().__init__()
        self._system = system
        self._component = component

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "system", None):
            record.system = self._system
        if not getattr(record, "component", None):
            record.component = self._component
        if not getattr(record, "job", None):
            record.job = ""
        return True


def install_filter_on_root_handlers(logging_filter: logging.Filter) -> None:
    for handler in logging.getLogger().handlers:
        handler.addFilter(logging_filter)


def set_telegram_alert_sender(sender: Optional[Callable[[str], Awaitable[None]]]) -> None:
    global _ALERT_SENDER
    _ALERT_SENDER = sender


def get_log_retention_days() -> int:
    raw = os.getenv("LOG_RETENTION_DAYS", "14")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 14

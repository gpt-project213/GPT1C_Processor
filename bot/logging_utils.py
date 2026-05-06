#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot/logging_utils.py
Shared runtime logging helpers for the bot/orchestrator layer.
"""

from __future__ import annotations

import asyncio
import contextvars
import html
import io
import json
import logging
import os
import sys
import time
import traceback
import uuid
from collections import deque
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from zoneinfo import ZoneInfo

__VERSION__ = "1.1.1"

_ALERT_SENDER: Optional[Callable[[str], Awaitable[None]]] = None
_ALERT_HANDLER: Optional["TelegramErrorAlertHandler"] = None

# ── Correlation ID ─────────────────────────────────────────────
_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")


def new_trace_id() -> str:
    """Генерирует новый trace_id и устанавливает его в текущий async-контекст."""
    tid = uuid.uuid4().hex[:12]
    _trace_id_var.set(tid)
    return tid


def get_trace_id() -> str:
    return _trace_id_var.get()


def bind_trace_id(tid: str) -> None:
    _trace_id_var.set(tid)


# ── Formatter ──────────────────────────────────────────────────

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
        trace_id = getattr(record, "trace_id", "") or _trace_id_var.get()

        parts = []
        if system:
            parts.append(f"[{system}]")
        if component:
            parts.append(f"[{component}]")
        if job:
            parts.append(f"[{job}]")
        if trace_id:
            parts.append(f"[{trace_id[:8]}]")

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


# ── ContextAdapter ─────────────────────────────────────────────

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
        # Прокидываем trace_id из ContextVar если не задан явно
        if "trace_id" not in extra:
            tid = _trace_id_var.get()
            if tid:
                extra["trace_id"] = tid
        kwargs["extra"] = extra
        return msg, kwargs


# ── TelegramErrorAlertHandler ──────────────────────────────────

class TelegramErrorAlertHandler(logging.Handler):
    def __init__(
        self,
        *,
        level: int = logging.ERROR,
        cooldown_sec: int = 300,
        max_message_len: int = 1800,
        dead_letter_path: Optional[Path] = None,
        dead_letter_maxlen: int = 20,
    ) -> None:
        super().__init__(level=level)
        self._cooldown_sec = cooldown_sec
        self._max_message_len = max_message_len
        self._dead_letter_path = dead_letter_path
        self._dead_letters: deque[str] = deque(maxlen=dead_letter_maxlen)
        self._recent: dict[str, float] = {}
        # Загружаем накопленные dead letters с диска при старте
        if dead_letter_path:
            self._load_spool()

    # ── dead-letter spool ──────────────────────────────────────

    def push_dead_letter(self, text: str) -> None:
        self._dead_letters.append(text)
        self._append_spool(text)

    def pop_dead_letters(self) -> list[str]:
        items = list(self._dead_letters)
        self._dead_letters.clear()
        self._clear_spool()
        return items

    def has_dead_letters(self) -> bool:
        return bool(self._dead_letters)

    def _append_spool(self, text: str) -> None:
        if not self._dead_letter_path:
            return
        try:
            self._dead_letter_path.parent.mkdir(parents=True, exist_ok=True)
            with self._dead_letter_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), "text": text}, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _clear_spool(self) -> None:
        if not self._dead_letter_path:
            return
        try:
            self._dead_letter_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _load_spool(self) -> None:
        if not self._dead_letter_path or not self._dead_letter_path.exists():
            return
        try:
            for line in self._dead_letter_path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    self._dead_letters.append(rec["text"])
                except Exception:
                    pass
        except Exception:
            pass

    # ── emit ───────────────────────────────────────────────────

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
        if len(self._recent) > 500:
            cutoff = now - self._cooldown_sec * 2
            self._recent = {k: v for k, v in self._recent.items() if v > cutoff}

        text = self._format_alert_text(record, system, component, event, message)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Нет event loop — кладём в dead-letter чтобы не потерять
            self.push_dead_letter(text)
            return
        try:
            loop.create_task(_ALERT_SENDER(text))
        except Exception:
            self.push_dead_letter(text)

    def _format_alert_text(
        self,
        record: logging.LogRecord,
        system: str,
        component: str,
        event: str,
        message: str,
    ) -> str:
        source = record.name
        trace_id = getattr(record, "trace_id", "") or _trace_id_var.get()
        lines = [
            "🚨 <b>Runtime error</b>",
            f"level: <code>{html.escape(record.levelname)}</code>",
            f"system: <code>{html.escape(system)}</code>",
        ]
        if component:
            lines.append(f"component: <code>{html.escape(component)}</code>")
        if event:
            lines.append(f"event: <code>{html.escape(event)}</code>")
        if trace_id:
            lines.append(f"trace_id: <code>{html.escape(trace_id)}</code>")
        lines.append(f"source: <code>{html.escape(source)}</code>")
        lines.append("")
        lines.append(html.escape(message[: self._max_message_len]))

        # Traceback если есть
        if record.exc_info:
            try:
                tb_lines = traceback.format_exception(*record.exc_info)
                tb_text = "".join(tb_lines).strip()
                # Оставляем последние 800 символов чтобы не упереться в лимит Telegram
                if len(tb_text) > 800:
                    tb_text = "…" + tb_text[-800:]
                lines.append("")
                lines.append("<code>" + html.escape(tb_text) + "</code>")
            except Exception:
                pass

        return "\n".join(lines)


# ── configure_runtime_logging ──────────────────────────────────

def configure_runtime_logging(
    *,
    logs_dir: Path,
    tz: ZoneInfo,
    app_name: str = "send_reports",
    level: int = logging.INFO,
    retention_days: int = 14,
    error_alert_level: int = logging.ERROR,
    alert_cooldown_sec: int = 300,
    test_mode: bool = False,
) -> TelegramErrorAlertHandler:
    global _ALERT_HANDLER

    logs_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    for _h in root.handlers[:]:
        _h.close()
    root.handlers.clear()

    formatter = RuntimeFormatter(tz)

    log_path = logs_dir / f"{app_name}.log"
    stream_handler = logging.StreamHandler(
        io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    )
    stream_handler.setFormatter(formatter)

    dead_letter_path = logs_dir / "alert_dead_letters.jsonl"
    alert_handler = TelegramErrorAlertHandler(
        level=error_alert_level,
        cooldown_sec=alert_cooldown_sec,
        dead_letter_path=dead_letter_path,
    )
    alert_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(alert_handler)

    if not test_mode:
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


# ── configure_module_logger ────────────────────────────────────

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

    # Подключаем alert handler чтобы ERROR из модулей доходили до Telegram
    # несмотря на propagate=False
    if _ALERT_HANDLER is not None:
        logger.addHandler(_ALERT_HANDLER)
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


# ── Helpers ────────────────────────────────────────────────────

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


def push_dead_letter(text: str) -> None:
    """Кладёт алерт в dead-letter spool если handler доступен."""
    if _ALERT_HANDLER is not None:
        _ALERT_HANDLER.push_dead_letter(text)


def pop_dead_letters() -> list[str]:
    """Извлекает все накопленные dead letters и очищает spool."""
    if _ALERT_HANDLER is None:
        return []
    return _ALERT_HANDLER.pop_dead_letters()


def has_dead_letters() -> bool:
    return _ALERT_HANDLER is not None and _ALERT_HANDLER.has_dead_letters()

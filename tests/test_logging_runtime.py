#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
import bot.logging_utils as blu
from bot.logging_utils import (
    RuntimeFormatter,
    TelegramErrorAlertHandler,
    configure_module_logger,
    configure_runtime_logging,
    get_runtime_logger,
    new_trace_id,
    get_trace_id,
    bind_trace_id,
    push_dead_letter,
    pop_dead_letters,
    has_dead_letters,
)
from collector.logging_utils import get_stop_logger


class RuntimeLoggingTests(unittest.TestCase):

    def setUp(self):
        # Сбрасываем trace_id между тестами чтобы ContextVar не утекал
        bind_trace_id("")

    def test_runtime_logger_formats_domain_prefixes(self):
        logger_name = "tests.runtime.crm.v2"
        base = logging.getLogger(logger_name)
        base.handlers.clear()
        base.propagate = False
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(RuntimeFormatter(ZoneInfo("Asia/Almaty")))
        base.addHandler(handler)
        base.setLevel(logging.INFO)

        log = get_runtime_logger(logger_name, system="CRM", component="FLOW")
        log.info("hello")
        # trace_id пуст, поэтому его нет в выводе
        self.assertIn("[CRM][FLOW] hello", stream.getvalue())

    def test_stop_logger_uses_stop_control_domain(self):
        logger_name = "tests.runtime.stop.v2"
        base = logging.getLogger(logger_name)
        base.handlers.clear()
        base.propagate = False
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(RuntimeFormatter(ZoneInfo("Asia/Almaty")))
        base.addHandler(handler)
        base.setLevel(logging.INFO)

        log = get_stop_logger(logger_name)
        log.warning("blocked")
        self.assertIn("[STOP_CONTROL][FLOW] blocked", stream.getvalue())

    def test_config_setup_logging_writes_module_log_with_domain(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_logs = Path(td)
            logger_name = "sales_parser_test_v2"
            patched_logger = logging.getLogger(logger_name)
            patched_logger.handlers.clear()
            patched_logger.propagate = False
            if hasattr(patched_logger, "_gpt1c_configured"):
                delattr(patched_logger, "_gpt1c_configured")

            with patch.object(config, "LOGS_DIR", tmp_logs):
                log = config.setup_logging(logger_name)
                log.info("pipeline test line")
                for handler in log.handlers:
                    if hasattr(handler, "flush"):
                        handler.flush()

            files = list(tmp_logs.glob(f"{logger_name}*"))
            self.assertTrue(files, "module log file was not created")
            content = files[0].read_text(encoding="utf-8")
            self.assertIn("[PIPELINE][MODULE]", content)
            self.assertIn("pipeline test line", content)
            for handler in list(log.handlers):
                try:
                    handler.close()
                finally:
                    log.removeHandler(handler)

    def test_module_logger_alert_fires_despite_propagate_false(self):
        """ERROR из модульного логгера доходит до TelegramErrorAlertHandler несмотря на propagate=False."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            test_handler = TelegramErrorAlertHandler(level=logging.ERROR, cooldown_sec=0)
            original = blu._ALERT_HANDLER
            blu._ALERT_HANDLER = test_handler
            try:
                sentinel_name = "test.pipeline.module.unique456"
                sentinel = logging.getLogger(sentinel_name)
                sentinel.handlers.clear()
                if hasattr(sentinel, "_gpt1c_configured"):
                    delattr(sentinel, "_gpt1c_configured")
                mod_log = configure_module_logger(sentinel_name, logs_dir=tmp, tz=ZoneInfo("Asia/Almaty"))
                self.assertFalse(mod_log.propagate)
                found_alert = any(h is test_handler for h in mod_log.handlers)
                self.assertTrue(found_alert, "TelegramErrorAlertHandler не подключён к module logger")
            finally:
                blu._ALERT_HANDLER = original
                for h in list(sentinel.handlers):
                    h.close()
                    sentinel.removeHandler(h)

    # ── Traceback в alert ──────────────────────────────────────

    def test_alert_includes_traceback(self):
        """Алерт содержит traceback если запись логировалась с exc_info."""
        handler = TelegramErrorAlertHandler(level=logging.ERROR, cooldown_sec=0)
        try:
            raise ValueError("тестовая ошибка")
        except ValueError:
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname="", lineno=0,
                msg="что-то сломалось", args=(), exc_info=sys.exc_info(),
            )
        text = handler._format_alert_text(record, "BOT", "CORE", "", "что-то сломалось")
        self.assertIn("ValueError", text)
        self.assertIn("тестовая ошибка", text)
        # Traceback завёрнут в <code>...</code>
        self.assertIn("Traceback", text)

    def test_alert_without_exception_has_no_traceback(self):
        """Без exc_info Traceback-блок не добавляется."""
        handler = TelegramErrorAlertHandler(level=logging.ERROR, cooldown_sec=0)
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0,
            msg="просто ошибка", args=(), exc_info=None,
        )
        text = handler._format_alert_text(record, "CRM", "FLOW", "", "просто ошибка")
        self.assertNotIn("Traceback", text)

    # ── Dead-letter spool ──────────────────────────────────────

    def test_dead_letters_persist_to_disk_and_reload(self):
        """Dead letters записываются в JSONL и восстанавливаются при создании нового handler."""
        with tempfile.TemporaryDirectory() as td:
            spool = Path(td) / "alert_dead_letters.jsonl"
            h1 = TelegramErrorAlertHandler(dead_letter_path=spool)
            h1.push_dead_letter("ошибка 1")
            h1.push_dead_letter("ошибка 2")
            self.assertTrue(spool.exists())

            # Новый handler загружает с диска
            h2 = TelegramErrorAlertHandler(dead_letter_path=spool)
            self.assertTrue(h2.has_dead_letters())
            letters = h2.pop_dead_letters()
            self.assertIn("ошибка 1", letters)
            self.assertIn("ошибка 2", letters)
            # После pop spool очищен
            self.assertFalse(spool.exists())

    def test_dead_letters_module_level_helpers(self):
        """Модульные функции push/pop/has работают через _ALERT_HANDLER."""
        original = blu._ALERT_HANDLER
        try:
            with tempfile.TemporaryDirectory() as td:
                test_handler = TelegramErrorAlertHandler(
                    dead_letter_path=Path(td) / "dl.jsonl"
                )
                blu._ALERT_HANDLER = test_handler
                push_dead_letter("тест алерт")
                self.assertTrue(has_dead_letters())
                items = pop_dead_letters()
                self.assertIn("тест алерт", items)
                self.assertFalse(has_dead_letters())
        finally:
            blu._ALERT_HANDLER = original

    # ── Correlation ID ─────────────────────────────────────────

    def test_new_trace_id_returns_unique_ids(self):
        tid1 = new_trace_id()
        tid2 = new_trace_id()
        self.assertNotEqual(tid1, tid2)
        self.assertEqual(len(tid1), 12)

    def test_trace_id_appears_in_log_output(self):
        """trace_id отображается в форматированном выводе."""
        logger_name = "tests.trace.check.v2"
        base = logging.getLogger(logger_name)
        base.handlers.clear()
        base.propagate = False
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(RuntimeFormatter(ZoneInfo("Asia/Almaty")))
        base.addHandler(handler)
        base.setLevel(logging.INFO)

        tid = new_trace_id()
        log = get_runtime_logger(logger_name, system="CRM", component="FLOW")
        log.info("trace test")
        self.assertIn(tid[:8], stream.getvalue())

    def test_bind_trace_id_and_get_trace_id(self):
        bind_trace_id("abc123def456")
        self.assertEqual(get_trace_id(), "abc123def456")

    def test_alert_includes_trace_id(self):
        """trace_id виден в теле Telegram-алерта."""
        bind_trace_id("tracetest99")
        handler = TelegramErrorAlertHandler(level=logging.ERROR, cooldown_sec=0)
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0,
            msg="ошибка с trace", args=(), exc_info=None,
        )
        text = handler._format_alert_text(record, "BOT", "CORE", "", "ошибка с trace")
        self.assertIn("tracetest99", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)

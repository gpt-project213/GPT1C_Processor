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
from bot.logging_utils import RuntimeFormatter, get_runtime_logger
from collector.logging_utils import get_stop_logger


class RuntimeLoggingTests(unittest.TestCase):
    def test_runtime_logger_formats_domain_prefixes(self):
        logger_name = "tests.runtime.crm"
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
        output = stream.getvalue()
        self.assertIn("[CRM][FLOW] hello", output)

    def test_stop_logger_uses_stop_control_domain(self):
        logger_name = "tests.runtime.stop"
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
        output = stream.getvalue()
        self.assertIn("[STOP_CONTROL][FLOW] blocked", output)

    def test_config_setup_logging_writes_module_log_with_domain(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_logs = Path(td)
            logger_name = "sales_parser_test"
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
            self.assertIn("[PIPELINE][MODULE] pipeline test line", content)
            for handler in list(log.handlers):
                try:
                    handler.close()
                finally:
                    log.removeHandler(handler)


    def test_module_logger_alert_fires_despite_propagate_false(self):
        """ERROR из модульного логгера доходит до TelegramErrorAlertHandler несмотря на propagate=False."""
        import tempfile
        from bot.logging_utils import (
            TelegramErrorAlertHandler,
            configure_module_logger,
            configure_runtime_logging,
        )
        captured: list[str] = []

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            configure_runtime_logging(logs_dir=tmp, tz=ZoneInfo("Asia/Almaty"))

            test_handler = TelegramErrorAlertHandler(level=logging.ERROR, cooldown_sec=0)
            import bot.logging_utils as blu
            original = blu._ALERT_HANDLER
            blu._ALERT_HANDLER = test_handler

            sentinel_name = "test.pipeline.module.unique123"
            sentinel = logging.getLogger(sentinel_name)
            sentinel.handlers.clear()
            if hasattr(sentinel, "_gpt1c_configured"):
                delattr(sentinel, "_gpt1c_configured")

            mod_log = configure_module_logger(sentinel_name, logs_dir=tmp, tz=ZoneInfo("Asia/Almaty"))
            self.assertFalse(mod_log.propagate)

            found_alert = any(h is test_handler for h in mod_log.handlers)
            blu._ALERT_HANDLER = original
            self.assertTrue(found_alert, "TelegramErrorAlertHandler не подключён к module logger")


if __name__ == "__main__":
    unittest.main(verbosity=2)

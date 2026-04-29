#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/logging_utils.py
Единый helper для системного логирования collector-контура.

Версия: 1.0.0 (2026-04-29)
"""

from __future__ import annotations

import logging
from typing import Any


class PrefixAdapter(logging.LoggerAdapter):
    def __init__(self, base_logger: logging.Logger, prefix: str) -> None:
        super().__init__(base_logger, {})
        self._prefix = prefix

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return f"{self._prefix} {msg}", kwargs


def get_prefixed_logger(name: str, prefix: str = "[COLLECTOR]") -> logging.LoggerAdapter:
    return PrefixAdapter(logging.getLogger(name), prefix)


def get_collector_logger(name: str) -> logging.LoggerAdapter:
    return get_prefixed_logger(name, "[COLLECTOR]")


def get_stop_logger(name: str) -> logging.LoggerAdapter:
    return get_prefixed_logger(name, "[STOP]")

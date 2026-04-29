#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/logging_utils.py
Единый helper для системного логирования collector-контура.

Версия: 1.0.0 (2026-04-29)
"""

from __future__ import annotations

from bot.logging_utils import get_runtime_logger


def get_collector_logger(name: str):
    return get_runtime_logger(name, system="COLLECTOR", component="FLOW")


def get_stop_logger(name: str):
    return get_runtime_logger(name, system="STOP_CONTROL", component="FLOW")

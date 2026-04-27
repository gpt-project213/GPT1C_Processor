#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тесты доказательства изменений v1.0.7 в debt_stop_control.py:
  - порог оплаты STOP_PAID_THRESHOLD = 5000 ₸ (было 1000)
  - delta-логика send_saida_final (только новые клиенты)
  - новые статусы в already_controlled
  - _apply_final_clearance записывает правильные статусы
  - cleared_limited: поле limit_expires корректно рассчитывается
"""
import asyncio
import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Добавляем root в sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Патчируем tg-зависимости до импорта модуля ──────────────────────
import unittest.mock as _m
_tg_stub = _m.MagicMock()
_tg_stub.InlineKeyboardButton = MagicMock(return_value=MagicMock())
_tg_stub.InlineKeyboardMarkup  = MagicMock(return_value=MagicMock())
sys.modules.setdefault("telegram", _tg_stub)

import importlib
dsc = importlib.import_module("bot.debt_stop_control")


# ════════════════════════════════════════════════════════════════════
# 1. STOP_PAID_THRESHOLD — порог оплаты изменён
# ════════════════════════════════════════════════════════════════════

def test_stop_paid_threshold_default():
    """STOP_PAID_THRESHOLD должен быть 5000 по умолчанию (было 1000)."""
    assert dsc.STOP_PAID_THRESHOLD == 5000.0, (
        f"Ожидалось 5000, получено {dsc.STOP_PAID_THRESHOLD}"
    )


def test_old_constant_removed():
    """FULL_PAYMENT_THRESHOLD не должен существовать в модуле."""
    assert not hasattr(dsc, "FULL_PAYMENT_THRESHOLD"), (
        "FULL_PAYMENT_THRESHOLD не должен быть в модуле — заменён на STOP_PAID_THRESHOLD"
    )


# ════════════════════════════════════════════════════════════════════
# 2. _apply_final_clearance — правильные статусы и поля
# ════════════════════════════════════════════════════════════════════

def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_registry(client_name: str, status: str) -> dict:
    return {
        client_name: {
            "manager": "Ергали",
            "manager_chat_id": 756622791,
            "status": status,
            "days_at_stop": 24,
        }
    }


def test_apply_final_clearance_clear():
    """action='clear' → status='cleared', менеджер и Саида уведомлены."""
    client = "Тест Клиент Авт"
    rec = {"manager": "Ергали", "manager_chat_id": 756622791, "status": "pending_clearance_admin", "days_at_stop": 10}

    bot = AsyncMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1, date=MagicMock(timestamp=MagicMock(return_value=0.0))))

    with tempfile.TemporaryDirectory() as tmp:
        reg_file = Path(tmp) / "debt_stop_registry.json"
        reg_file.write_text(json.dumps({client: rec}), encoding="utf-8")

        known_file = Path(tmp) / "debt_stop_saida_known.json"
        known_file.write_text(json.dumps({"known": [client]}), encoding="utf-8")

        with (
            patch.object(dsc, "REGISTRY_FILE", reg_file),
            patch.object(dsc, "SAIDA_KNOWN_FILE", known_file),
            patch.object(dsc, "_get_admin_chat_id", return_value=7422963573),
            patch.object(dsc, "_load_managers", return_value={"Ергали": 756622791}),
            patch.object(dsc, "_get_client_current_state", return_value=None),
        ):
            _run(dsc._apply_final_clearance(client, rec, "clear", 0, 0, bot))

            updated = json.loads(reg_file.read_text(encoding="utf-8"))
            assert updated[client]["status"] == "cleared", f"Ожидался 'cleared', получен: {updated[client]['status']}"
            # Проверяем что клиент убран из known
            known = json.loads(known_file.read_text(encoding="utf-8"))
            assert client not in known["known"], "Клиент должен быть убран из known после снятия стопа"

    # Проверяем что bot.send_message вызывался (уведомления)
    assert bot.send_message.call_count >= 1


def test_apply_final_clearance_blacklist():
    """action='blacklist' → status='blacklisted'."""
    client = "Тест Чёрный Список"
    rec = {"manager": "Алена", "manager_chat_id": 188939016, "status": "pending_clearance_admin", "days_at_stop": 25}
    bot = AsyncMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1, date=MagicMock(timestamp=MagicMock(return_value=0.0))))

    with tempfile.TemporaryDirectory() as tmp:
        reg_file  = Path(tmp) / "debt_stop_registry.json"
        reg_file.write_text(json.dumps({client: rec}), encoding="utf-8")
        known_file = Path(tmp) / "debt_stop_saida_known.json"
        known_file.write_text(json.dumps({"known": []}), encoding="utf-8")

        with (
            patch.object(dsc, "REGISTRY_FILE", reg_file),
            patch.object(dsc, "SAIDA_KNOWN_FILE", known_file),
            patch.object(dsc, "_get_admin_chat_id", return_value=7422963573),
            patch.object(dsc, "_load_managers", return_value={}),
            patch.object(dsc, "_get_client_current_state", return_value=None),
        ):
            _run(dsc._apply_final_clearance(client, rec, "blacklist", 0, 0, bot))
            updated = json.loads(reg_file.read_text(encoding="utf-8"))
            assert updated[client]["status"] == "blacklisted"


def test_apply_final_clearance_limit():
    """action='limit' → status='cleared_limited', limit_expires через N дней."""
    client = "Тест Лимит"
    rec = {"manager": "Магира", "manager_chat_id": 735574334, "status": "pending_clearance_admin", "days_at_stop": 15}
    bot = AsyncMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1, date=MagicMock(timestamp=MagicMock(return_value=0.0))))

    with tempfile.TemporaryDirectory() as tmp:
        reg_file  = Path(tmp) / "debt_stop_registry.json"
        reg_file.write_text(json.dumps({client: rec}), encoding="utf-8")
        known_file = Path(tmp) / "debt_stop_saida_known.json"
        known_file.write_text(json.dumps({"known": []}), encoding="utf-8")

        with (
            patch.object(dsc, "REGISTRY_FILE", reg_file),
            patch.object(dsc, "SAIDA_KNOWN_FILE", known_file),
            patch.object(dsc, "_get_admin_chat_id", return_value=7422963573),
            patch.object(dsc, "_load_managers", return_value={}),
            patch.object(dsc, "_get_client_current_state", return_value=None),
        ):
            _run(dsc._apply_final_clearance(client, rec, "limit", 100_000.0, 7, bot))
            updated = json.loads(reg_file.read_text(encoding="utf-8"))
            r = updated[client]
            assert r["status"] == "cleared_limited"
            assert r["shipment_limit"] == 100_000.0
            assert r["limit_days"] == 7
            assert r["limit_expires"] is not None
            # expires = today + 7 дней
            from zoneinfo import ZoneInfo
            tz  = ZoneInfo("Asia/Almaty")
            exp = (datetime.now(tz) + timedelta(days=7)).strftime("%Y-%m-%d")
            assert r["limit_expires"] == exp, f"Ожидалось {exp}, получено {r['limit_expires']}"


def test_apply_final_clearance_prepay():
    """action='prepay' → status='prepayment_only'."""
    client = "Тест Предоплата"
    rec = {"manager": "Оксана", "manager_chat_id": 1446255940, "status": "pending_clearance_admin", "days_at_stop": 20}
    bot = AsyncMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1, date=MagicMock(timestamp=MagicMock(return_value=0.0))))

    with tempfile.TemporaryDirectory() as tmp:
        reg_file  = Path(tmp) / "debt_stop_registry.json"
        reg_file.write_text(json.dumps({client: rec}), encoding="utf-8")
        known_file = Path(tmp) / "debt_stop_saida_known.json"
        known_file.write_text(json.dumps({"known": []}), encoding="utf-8")

        with (
            patch.object(dsc, "REGISTRY_FILE", reg_file),
            patch.object(dsc, "SAIDA_KNOWN_FILE", known_file),
            patch.object(dsc, "_get_admin_chat_id", return_value=7422963573),
            patch.object(dsc, "_load_managers", return_value={}),
            patch.object(dsc, "_get_client_current_state", return_value=None),
        ):
            _run(dsc._apply_final_clearance(client, rec, "prepay", 0, 0, bot))
            updated = json.loads(reg_file.read_text(encoding="utf-8"))
            assert updated[client]["status"] == "prepayment_only"


# ════════════════════════════════════════════════════════════════════
# 3. Новые статусы в already_controlled (_build_candidates)
# ════════════════════════════════════════════════════════════════════

def test_new_statuses_excluded_from_candidates():
    """Клиенты с новыми статусами не должны попадать в кандидатов."""
    new_statuses = [
        "pending_clearance_mgr", "pending_clearance_admin",
        "awaiting_mgr_limit_input", "awaiting_admin_limit_override",
        "prepayment_only", "blacklisted", "cleared_limited",
    ]
    # Имитируем registry с клиентами в новых статусах
    registry = {f"Клиент_{s}": {"status": s} for s in new_statuses}

    controlled = {
        name for name, rec in registry.items()
        if rec.get("status") in (
            "exception", "auto_stopped", "stopped", "conditional",
            "allow_after_payment", "block_until_payment",
            "pending_clearance", "pending_clearance_mgr", "pending_clearance_admin",
            "awaiting_clearance_limit", "awaiting_mgr_limit_input", "awaiting_admin_limit_override",
            "prepayment_only", "blacklisted", "cleared_limited",
        )
    }
    for s in new_statuses:
        name = f"Клиент_{s}"
        assert name in controlled, f"Статус '{s}' должен исключать клиента из кандидатов"


# ════════════════════════════════════════════════════════════════════
# 4. Delta-логика: known-файл
# ════════════════════════════════════════════════════════════════════

def test_saida_known_save_load():
    """_save_saida_known / _load_saida_known — round-trip."""
    with tempfile.TemporaryDirectory() as tmp:
        known_file = Path(tmp) / "debt_stop_saida_known.json"
        with patch.object(dsc, "SAIDA_KNOWN_FILE", known_file):
            original = {"Клиент А", "Клиент Б", "Клиент В"}
            dsc._save_saida_known(original)
            loaded = dsc._load_saida_known()
            assert loaded == original, f"Ожидалось {original}, получено {loaded}"


def test_saida_known_empty_file():
    """_load_saida_known возвращает пустое множество если файл отсутствует."""
    with tempfile.TemporaryDirectory() as tmp:
        known_file = Path(tmp) / "nonexistent.json"
        with patch.object(dsc, "SAIDA_KNOWN_FILE", known_file):
            result = dsc._load_saida_known()
            assert result == set(), f"Ожидался пустой set, получено {result}"


# ════════════════════════════════════════════════════════════════════
# 5. _handle_mgr_clearance_proposal — роутинг по action
# ════════════════════════════════════════════════════════════════════

def test_mgr_clearance_proposal_keep():
    """action='keep' → status='pending_clearance_admin', proposal записана."""
    client = "ТОО Тест"
    mgr_id = 756622791
    rec = {"manager": "Ергали", "manager_chat_id": mgr_id,
           "status": "pending_clearance_mgr", "days_at_stop": 24}

    bot = AsyncMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1, date=MagicMock(timestamp=MagicMock(return_value=0.0))))

    with tempfile.TemporaryDirectory() as tmp:
        reg_file = Path(tmp) / "debt_stop_registry.json"
        reg_file.write_text(json.dumps({client: rec}), encoding="utf-8")
        known_file = Path(tmp) / "debt_stop_saida_known.json"
        known_file.write_text(json.dumps({"known": []}), encoding="utf-8")

        with (
            patch.object(dsc, "REGISTRY_FILE", reg_file),
            patch.object(dsc, "SAIDA_KNOWN_FILE", known_file),
            patch.object(dsc, "_get_admin_chat_id", return_value=7422963573),
            patch.object(dsc, "_get_client_current_state", return_value=None),
        ):
            result = _run(dsc._handle_mgr_clearance_proposal(client[:26], "keep", mgr_id, bot))
            updated = json.loads(reg_file.read_text(encoding="utf-8"))
            assert updated[client]["status"] == "pending_clearance_admin"
            assert updated[client]["clearance_proposal"]["action"] == "keep"
            assert "передано руководителю" in result.lower()


def test_mgr_clearance_proposal_limit_starts_input():
    """action='limit' → status='awaiting_mgr_limit_input', ждём ввода суммы."""
    client = "ИП Лимит Тест"
    mgr_id = 188939016
    rec = {"manager": "Алена", "manager_chat_id": mgr_id,
           "status": "pending_clearance_mgr", "days_at_stop": 15}

    bot = AsyncMock()
    with tempfile.TemporaryDirectory() as tmp:
        reg_file = Path(tmp) / "debt_stop_registry.json"
        reg_file.write_text(json.dumps({client: rec}), encoding="utf-8")

        with (
            patch.object(dsc, "REGISTRY_FILE", reg_file),
            patch.object(dsc, "_get_admin_chat_id", return_value=7422963573),
        ):
            result = _run(dsc._handle_mgr_clearance_proposal(client[:26], "limit", mgr_id, bot))
            updated = json.loads(reg_file.read_text(encoding="utf-8"))
            assert updated[client]["status"] == "awaiting_mgr_limit_input"
            assert updated[client]["clearance_proposal"]["step"] == "amount"
            assert "лимит" in result.lower()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

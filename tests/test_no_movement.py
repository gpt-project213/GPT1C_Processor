#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тесты no_movement.py — Saida-first check перед WA.
"""
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import unittest.mock as _m
_tg_stub = _m.MagicMock()
_tg_stub.InlineKeyboardButton = MagicMock(return_value=MagicMock())
_tg_stub.InlineKeyboardMarkup  = MagicMock(return_value=MagicMock())
sys.modules.setdefault("telegram", _tg_stub)

import importlib
nm = importlib.import_module("collector.no_movement")


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ════════════════════════════════════════════════════════════════════
# 1. State helpers
# ════════════════════════════════════════════════════════════════════

def test_was_saida_asked_today_false_when_empty():
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(nm, "NM_STATE_FILE", Path(tmp) / "nm.json"):
            assert nm.was_saida_asked_today("ТОО Тест") is False


def test_was_saida_asked_today_true_after_save():
    with tempfile.TemporaryDirectory() as tmp:
        state_file = Path(tmp) / "nm.json"
        today = nm._today()
        state_file.write_text(
            json.dumps({"тоо тест": {"date": today, "name": "ТОО Тест", "status": "pending_saida",
                                     "amount": 100000.0, "days": 15, "mgr_name": "Ергали"}}),
            encoding="utf-8",
        )
        with patch.object(nm, "NM_STATE_FILE", state_file):
            assert nm.was_saida_asked_today("ТОО Тест") is True


def test_get_nm_state_returns_none_for_yesterday():
    with tempfile.TemporaryDirectory() as tmp:
        state_file = Path(tmp) / "nm.json"
        state_file.write_text(
            json.dumps({"тоо тест": {"date": "2000-01-01", "name": "ТОО Тест",
                                     "status": "pending_saida", "amount": 50000.0, "days": 10}}),
            encoding="utf-8",
        )
        with patch.object(nm, "NM_STATE_FILE", state_file):
            assert nm.get_nm_state("ТОО Тест") is None


# ════════════════════════════════════════════════════════════════════
# 2. ask_saida_about_no_movement
# ════════════════════════════════════════════════════════════════════

def test_ask_saida_saves_state():
    with tempfile.TemporaryDirectory() as tmp:
        state_file = Path(tmp) / "nm.json"
        with (
            patch.object(nm, "NM_STATE_FILE", state_file),
            patch("collector.communications.send_telegram_with_markup", new=AsyncMock(return_value=True)),
        ):
            ok = _run(nm.ask_saida_about_no_movement("ТОО Клиент", 75000.0, 12, "Алена", 188939016))
            assert ok is True
            state = json.loads(state_file.read_text(encoding="utf-8"))
            assert "тоо клиент" in state
            assert state["тоо клиент"]["status"] == "pending_saida"
            assert state["тоо клиент"]["days"] == 12


def test_ask_saida_does_not_save_on_failure():
    with tempfile.TemporaryDirectory() as tmp:
        state_file = Path(tmp) / "nm.json"
        with (
            patch.object(nm, "NM_STATE_FILE", state_file),
            patch("collector.communications.send_telegram_with_markup", new=AsyncMock(return_value=False)),
        ):
            ok = _run(nm.ask_saida_about_no_movement("ТОО Клиент", 75000.0, 12, "Алена"))
            assert ok is False
            assert not state_file.exists()


# ════════════════════════════════════════════════════════════════════
# 3. handle_nm_callback — nm_paid
# ════════════════════════════════════════════════════════════════════

def _write_nm_state(state_file, name, status="pending_saida", amount=100000.0, days=15):
    today = nm._today()
    data = {nm._normalize(name): {"date": today, "name": name, "status": status,
                                   "amount": amount, "days": days, "mgr_name": "Ергали", "mgr_chat_id": 756622791}}
    state_file.write_text(json.dumps(data), encoding="utf-8")


def test_nm_paid_creates_hold_and_saves_state():
    with tempfile.TemporaryDirectory() as tmp:
        state_file = Path(tmp) / "nm.json"
        client = "ТОО Оплатил"
        key = client[:26]
        _write_nm_state(state_file, client)

        mock_send = AsyncMock(return_value=True)
        mock_hold = MagicMock()
        mock_notify = AsyncMock(return_value=True)

        with (
            patch.object(nm, "NM_STATE_FILE", state_file),
            patch.object(nm, "SAIDA_CHAT_ID", 999),
            patch("collector.communications.send_telegram", new=mock_send),
            patch("collector.communications.notify_admin", new=mock_notify),
            patch("collector.payment_hold.create_manager_payment_request", return_value={}),
        ):
            result = _run(nm.handle_nm_callback(f"nm_paid|{key}", 999))
            assert result is True
            state = json.loads(state_file.read_text(encoding="utf-8"))
            assert state[nm._normalize(client)]["status"] == "paid"


# ════════════════════════════════════════════════════════════════════
# 4. handle_nm_callback — nm_nopay → admin notified
# ════════════════════════════════════════════════════════════════════

def test_nm_nopay_sets_status_and_notifies_admin():
    with tempfile.TemporaryDirectory() as tmp:
        state_file = Path(tmp) / "nm.json"
        client = "ТОО Задолжал"
        key = client[:26]
        _write_nm_state(state_file, client)

        mock_send = AsyncMock(return_value=True)
        mock_markup = AsyncMock(return_value=True)

        with (
            patch.object(nm, "NM_STATE_FILE", state_file),
            patch.object(nm, "SAIDA_CHAT_ID", 999),
            patch.object(nm, "ADMIN_CHAT_ID_STR", "7422963573"),
            patch("collector.communications.send_telegram", new=mock_send),
            patch("collector.communications.send_telegram_with_markup", new=mock_markup),
            patch("collector.communications.notify_admin", new=mock_send),
        ):
            result = _run(nm.handle_nm_callback(f"nm_nopay|{key}", 999))
            assert result is True
            state = json.loads(state_file.read_text(encoding="utf-8"))
            assert state[nm._normalize(client)]["status"] == "nopay_notified_admin"


# ════════════════════════════════════════════════════════════════════
# 5. handle_nm_callback — nm_adm_skip
# ════════════════════════════════════════════════════════════════════

def test_nm_adm_skip_sets_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        state_file = Path(tmp) / "nm.json"
        client = "ТОО Пропустить"
        key = client[:26]
        _write_nm_state(state_file, client, status="nopay_notified_admin")

        mock_send = AsyncMock(return_value=True)

        with (
            patch.object(nm, "NM_STATE_FILE", state_file),
            patch.object(nm, "ADMIN_CHAT_ID_STR", "7422963573"),
            patch("collector.communications.send_telegram", new=mock_send),
        ):
            result = _run(nm.handle_nm_callback(f"nm_adm_skip|{key}", 7422963573))
            assert result is True
            state = json.loads(state_file.read_text(encoding="utf-8"))
            assert state[nm._normalize(client)]["status"] == "skipped"


# ════════════════════════════════════════════════════════════════════
# 6. handle_nm_callback — access control
# ════════════════════════════════════════════════════════════════════

def test_nm_paid_rejected_for_wrong_user():
    with tempfile.TemporaryDirectory() as tmp:
        state_file = Path(tmp) / "nm.json"
        _write_nm_state(state_file, "ТОО Тест")
        mock_send = AsyncMock()
        with (
            patch.object(nm, "NM_STATE_FILE", state_file),
            patch.object(nm, "SAIDA_CHAT_ID", 999),
            patch("collector.communications.send_telegram", new=mock_send),
        ):
            result = _run(nm.handle_nm_callback("nm_paid|ТОО Тест", 12345))
            assert result is True
            # Should have sent a warning
            assert mock_send.call_count >= 1


# ════════════════════════════════════════════════════════════════════
# 7. new msg_type templates in collection_agent
# ════════════════════════════════════════════════════════════════════

def test_no_movement_reminder_template_exists():
    from collector.collection_agent import _FALLBACK_TEMPLATES_DEFAULT, _get_fallback_template
    assert "no_movement_reminder" in _FALLBACK_TEMPLATES_DEFAULT
    t = _get_fallback_template(
        "no_movement_reminder",
        company="Test", client_name="ТОО", manager_name="Иван",
        amount="50 000", days=15, report_date="",
    )
    assert "тоо" in t.lower() or "Test" in t


def test_promise_broken_reminder_template_exists():
    from collector.collection_agent import _FALLBACK_TEMPLATES_DEFAULT, _get_fallback_template
    assert "promise_broken_reminder" in _FALLBACK_TEMPLATES_DEFAULT
    t = _get_fallback_template(
        "promise_broken_reminder",
        company="Test", client_name="ТОО", manager_name="Иван",
        amount="100 000", days=20, report_date="",
    )
    assert "обещ" in t.lower()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

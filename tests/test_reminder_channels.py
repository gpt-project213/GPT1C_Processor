#!/usr/bin/env python
# coding: utf-8
import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bot"))

from bot import darya_reminders as darya
from bot import minai_reminders as minai


class ReminderChannelFlowTests(unittest.TestCase):
    def test_darya_audio_yes_routes_pending_audio_directly_to_ai(self):
        state = {"__pending_audio_text__": {"_ts": "2026-05-19T10:00:00+05:00", "_val": "Напомню позвонить Руслану"}}
        with patch.object(darya, "_load_state", return_value=state.copy()), \
             patch.object(darya, "_save_state"), \
             patch.object(darya, "_process_add_text", new=AsyncMock()) as process_add:
            handled = asyncio.run(darya.handle_darya_response("Да"))
        self.assertTrue(handled)
        process_add.assert_awaited_once()
        self.assertEqual(process_add.await_args.args[0], "Напомню позвонить Руслану")

    def test_darya_free_text_routes_to_ai_without_trigger_word(self):
        with patch.object(darya, "_load_state", return_value={}), \
             patch.object(darya, "_save_state"), \
             patch.object(darya, "_process_add_text", new=AsyncMock()) as process_add, \
             patch.object(darya, "_send_buttons", return_value=True) as send_buttons:
            handled = asyncio.run(darya.handle_darya_response("Напомню сегодня в 15:00 позвонить Руслану"))
        self.assertTrue(handled)
        process_add.assert_awaited_once()
        send_buttons.assert_not_called()

    def test_darya_short_digit_keeps_catchall(self):
        with patch.object(darya, "_load_state", return_value={}), \
             patch.object(darya, "_save_state"), \
             patch.object(darya, "_process_add_text", new=AsyncMock()) as process_add, \
             patch.object(darya, "_send_buttons", return_value=True) as send_buttons:
            handled = asyncio.run(darya.handle_darya_response("1"))
        self.assertTrue(handled)
        process_add.assert_not_awaited()
        send_buttons.assert_called_once()

    def test_minai_audio_yes_routes_pending_audio_directly_to_ai(self):
        state = {"__pending_audio_text__": {"_ts": "2026-05-19T10:00:00+05:00", "_val": "Напомню оплатить интернет"}}
        with patch.object(minai, "_load_state", return_value=state.copy()), \
             patch.object(minai, "_save_state"), \
             patch.object(minai, "_process_add_text", new=AsyncMock()) as process_add:
            handled = asyncio.run(minai.handle_minai_response("Да"))
        self.assertTrue(handled)
        process_add.assert_awaited_once()
        self.assertEqual(process_add.await_args.args[0], "Напомню оплатить интернет")

    def test_minai_free_text_routes_to_ai_without_trigger_word(self):
        with patch.object(minai, "_load_state", return_value={}), \
             patch.object(minai, "_save_state"), \
             patch.object(minai, "_process_add_text", new=AsyncMock()) as process_add, \
             patch.object(minai, "_send_buttons", return_value=True) as send_buttons:
            handled = asyncio.run(minai.handle_minai_response("Напомню завтра оплатить интернет"))
        self.assertTrue(handled)
        process_add.assert_awaited_once()
        send_buttons.assert_not_called()

    def test_minai_pending_audio_unknown_reply_prompts_retry(self):
        state = {"__pending_audio_text__": {"_ts": "2026-05-19T10:00:00+05:00", "_val": "Напомню оплатить интернет"}}
        with patch.object(minai, "_load_state", return_value=state.copy()), \
             patch.object(minai, "_save_state"), \
             patch.object(minai, "_process_add_text", new=AsyncMock()) as process_add, \
             patch.object(minai, "_send_plain") as send_plain:
            handled = asyncio.run(minai.handle_minai_response("Ок"))
        self.assertTrue(handled)
        process_add.assert_not_awaited()
        send_plain.assert_called_once()
        self.assertIn("Если всё верно", send_plain.call_args.args[0])

    def test_minai_pending_confirm_unknown_reply_keeps_work_choice(self):
        state = {
            "__pending_confirm__": {
                "text": "Оплатить интернет",
                "schedule": "monthly:10",
                "hour": 9,
                "_ts": "2026-05-19T10:00:00+05:00",
            }
        }
        with patch.object(minai, "_load_state", return_value=state.copy()), \
             patch.object(minai, "_save_state"), \
             patch.object(minai, "_send_buttons", return_value=True) as send_buttons, \
             patch.object(minai, "_send_plain") as send_plain:
            handled = asyncio.run(minai.handle_minai_response("А зачем переспрашивать?"))
        self.assertTrue(handled)
        send_buttons.assert_called_once()
        send_plain.assert_not_called()

    def test_minai_pending_confirm_after_work_choice_prompts_yes_no(self):
        state = {
            "__pending_confirm__": {
                "text": "Оплатить интернет",
                "schedule": "monthly:10",
                "hour": 9,
                "work": True,
                "_ts": "2026-05-19T10:00:00+05:00",
            }
        }
        with patch.object(minai, "_load_state", return_value=state.copy()), \
             patch.object(minai, "_save_state"), \
             patch.object(minai, "_send_buttons", return_value=True) as send_buttons, \
             patch.object(minai, "_send_plain") as send_plain:
            handled = asyncio.run(minai.handle_minai_response("А зачем переспрашивать?"))
        self.assertTrue(handled)
        send_plain.assert_called_once()
        self.assertIn("Если всё верно", send_plain.call_args.args[0])
        send_buttons.assert_not_called()

    def test_minai_pending_confirm_ai_routes_work_choice(self):
        state = {
            "__pending_confirm__": {
                "text": "Оплатить интернет",
                "schedule": "monthly:10",
                "hour": 9,
                "_ts": "2026-05-19T10:00:00+05:00",
            }
        }
        with patch.object(minai, "_load_state", return_value=state.copy()), \
             patch.object(minai, "_save_state"), \
             patch.object(minai, "_deepseek_reply_intent", new=AsyncMock(return_value="work_yes")), \
             patch.object(minai, "_send_buttons", return_value=True) as send_buttons, \
             patch.object(minai, "_send_plain") as send_plain:
            handled = asyncio.run(minai.handle_minai_response("Ну, это для всех"))
        self.assertTrue(handled)
        send_buttons.assert_called_once()
        send_plain.assert_not_called()

    def test_minai_pending_confirm_ai_routes_final_confirmation(self):
        state = {
            "__pending_confirm__": {
                "text": "Оплатить интернет",
                "schedule": "monthly:10",
                "hour": 9,
                "work": True,
                "_ts": "2026-05-19T10:00:00+05:00",
            }
        }
        with patch.object(minai, "_load_state", return_value=state.copy()), \
             patch.object(minai, "_save_state"), \
             patch.object(minai, "_deepseek_reply_intent", new=AsyncMock(return_value="confirm_yes")), \
             patch.object(minai, "_save_custom_reminder") as save_custom, \
             patch.object(minai, "_send_buttons", return_value=True) as send_buttons:
            handled = asyncio.run(minai.handle_minai_response("Ну да, всё верно"))
        self.assertTrue(handled)
        save_custom.assert_called_once()
        send_buttons.assert_called_once()


if __name__ == "__main__":
    unittest.main()

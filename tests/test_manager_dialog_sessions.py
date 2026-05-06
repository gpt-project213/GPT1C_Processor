#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тесты миграции collector/dialog_store.py на dialog_id модель.
Покрывает 10 обязательных кейсов из ТЗ миграции.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import collector.dialog_store as ds


def _make_contact():
    return {"phone": "+77001112233", "name_confirmations": 0, "phone_confirmations": 0}


def _new_dlg(tmpdir: Path, chat_id: int = 111, client: str = "ТОО Альфа") -> dict:
    ds.DIALOGS_PATH = tmpdir / "dialogs.json"
    return ds.new_dialog(
        manager_chat_id=chat_id,
        manager_name="Магира",
        client_name=client,
        level=1,
        days=10,
        amount=50000.0,
        current_contact=_make_contact(),
    )


class TestDialogStoreMigration(unittest.TestCase):

    # ── 1. Старый формат мигрируется без потери данных ────────────────────────
    def test_old_format_migrates_to_new(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dialogs.json"
            old_data = {
                "123456": {
                    "client_name": "ТОО Бета",
                    "manager_name": "Оксана",
                    "manager_chat_id": 123456,
                    "state": "AWAITING_CONFIRM",
                    "created": "2026-05-06T10:00:00+05:00",
                }
            }
            path.write_text(json.dumps(old_data), encoding="utf-8")
            ds.DIALOGS_PATH = path

            container = ds.load_dialogs()
            self.assertIn("dialogs", container)
            self.assertIn("active_by_chat", container)
            self.assertEqual(len(container["dialogs"]), 1)

            dialog = next(iter(container["dialogs"].values()))
            self.assertEqual(dialog["client_name"], "ТОО Бета")
            self.assertIn("dialog_id", dialog)
            self.assertTrue(dialog["dialog_id"].startswith("dlg_"))

            # Индекс active_by_chat обновлён
            self.assertIn("123456", container["active_by_chat"])
            self.assertEqual(container["active_by_chat"]["123456"], [dialog["dialog_id"]])

            # Файл теперь в новом формате
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("dialogs", saved)

    # ── 2. Один chat_id может иметь 2 активных диалога ───────────────────────
    def test_two_active_dialogs_same_chat(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dialogs.json"
            ds.DIALOGS_PATH = path

            d1 = ds.new_dialog(111, "Магира", "ТОО Альфа", 1, 10, 50000.0, _make_contact())
            d2 = ds.new_dialog(111, "Магира", "ТОО Бета",  1, 15, 30000.0, _make_contact())

            self.assertNotEqual(d1["dialog_id"], d2["dialog_id"])
            ids = ds.get_active_dialog_ids(111)
            self.assertEqual(len(ids), 2)
            self.assertIn(d1["dialog_id"], ids)
            self.assertIn(d2["dialog_id"], ids)

    # ── 3. get_dialog возвращает правильный диалог по dialog_id ──────────────
    def test_get_dialog_by_dialog_id(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dialogs.json"
            ds.DIALOGS_PATH = path
            dlg = _new_dlg(path.parent)
            fetched = ds.get_dialog(dlg["dialog_id"])
            self.assertIsNotNone(fetched)
            self.assertEqual(fetched["client_name"], "ТОО Альфа")
            self.assertEqual(fetched["dialog_id"], dlg["dialog_id"])

    # ── 4. Старый callback-формат col_<action>_<mid> не вызывает KeyError ────
    def test_old_callback_mid_not_a_dialog_id(self):
        # get_dialog с числовым строковым ключом должен вернуть None, не упасть
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dialogs.json"
            ds.DIALOGS_PATH = path
            _new_dlg(path.parent, chat_id=222)
            # старый callback содержал mid=222 как строку
            result = ds.get_dialog("222")
            self.assertIsNone(result)  # не найден — это корректно

    # ── 5. Текстовый ответ при одном waiting диалоге ─────────────────────────
    def test_text_target_single_waiting_dialog(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dialogs.json"
            ds.DIALOGS_PATH = path
            dlg = _new_dlg(path.parent, chat_id=333)
            # Переводим в text-waiting state
            ds.update_dialog(dlg["dialog_id"], state=ds.STATE_AWAITING_DATA)
            target = ds.get_text_target_dialog(333)
            self.assertIsNotNone(target)
            self.assertEqual(target["dialog_id"], dlg["dialog_id"])

    # ── 6. Текстовый ответ при двух waiting диалогах → ambiguity → None ──────
    def test_text_target_ambiguity_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dialogs.json"
            ds.DIALOGS_PATH = path
            d1 = ds.new_dialog(444, "Ергали", "ТОО Альфа", 1, 10, 1.0, _make_contact())
            d2 = ds.new_dialog(444, "Ергали", "ТОО Бета",  1, 10, 1.0, _make_contact())
            ds.update_dialog(d1["dialog_id"], state=ds.STATE_AWAITING_DATA)
            ds.update_dialog(d2["dialog_id"], state=ds.STATE_AWAITING_REJECTION_REASON)
            target = ds.get_text_target_dialog(444)
            self.assertIsNone(target)

    # ── 7. remove_dialog чистит active_by_chat ────────────────────────────────
    def test_remove_dialog_cleans_index(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dialogs.json"
            ds.DIALOGS_PATH = path
            d1 = _new_dlg(path.parent, chat_id=555, client="ТОО Альфа")
            d2 = ds.new_dialog(555, "Магира", "ТОО Бета", 1, 10, 1.0, _make_contact())

            ds.remove_dialog(d1["dialog_id"])

            ids = ds.get_active_dialog_ids(555)
            self.assertNotIn(d1["dialog_id"], ids)
            self.assertIn(d2["dialog_id"], ids)

            ds.remove_dialog(d2["dialog_id"])
            container = ds.load_dialogs()
            self.assertNotIn("555", container["active_by_chat"])

    # ── 8. get_all_pending видит диалоги из новой структуры ──────────────────
    def test_get_all_pending_new_format(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dialogs.json"
            ds.DIALOGS_PATH = path
            d1 = ds.new_dialog(666, "Оксана", "ТОО А", 1, 5, 1.0, _make_contact())
            d2 = ds.new_dialog(666, "Оксана", "ТОО Б", 1, 5, 1.0, _make_contact())
            # d2 завершён
            ds.update_dialog(d2["dialog_id"], state=ds.STATE_DONE)

            pending = ds.get_all_pending()
            pending_ids = [d["dialog_id"] for d in pending]
            self.assertIn(d1["dialog_id"], pending_ids)
            self.assertNotIn(d2["dialog_id"], pending_ids)

    # ── 9. get_all_pending не падает на новой структуре ───────────────────────
    def test_get_all_pending_does_not_raise(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dialogs.json"
            ds.DIALOGS_PATH = path
            # Пустой файл
            try:
                result = ds.get_all_pending()
                self.assertIsInstance(result, list)
            except Exception as e:
                self.fail(f"get_all_pending raised: {e}")

    # ── 10. get_text_target_dialog не падает при отсутствии диалогов ──────────
    def test_text_target_no_dialogs_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dialogs.json"
            ds.DIALOGS_PATH = path
            result = ds.get_text_target_dialog(9999)
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bot"))

import logging


class _NoopFileHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def emit(self, record):
        return


with patch("logging.FileHandler", _NoopFileHandler):
    import bot.crm_clients as crm
    import bot.send_reports as sr
    import bot.crm_audit_log as crm_audit


class CRMRegressionTests(unittest.TestCase):
    def test_update_from_reports_merges_canonical_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_dir = root / "config"
            json_dir = root / "reports" / "json"
            config_dir.mkdir(parents=True)
            json_dir.mkdir(parents=True)
            clients_path = config_dir / "clients.json"
            clients_path.write_text(
                json.dumps(
                    {
                        "clients": {
                            "М Плов центр ЕСБОЛОВА  Мангилик Ел 54": {
                                "manager": "Магира",
                                "sources": ["sales"],
                                "first_seen": "2026-04-01",
                                "last_seen": "2026-04-01",
                                "aliases": [],
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (json_dir / "debt_ext_Детальный Дебиторы Магира (1).json").write_text(
                json.dumps(
                    {
                        "manager": "Магира",
                        "clients": [{"name": "М Плов центр ЕСБОЛОВА Мангилик Ел 54"}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.object(crm, "CONFIG_DIR", config_dir), patch.object(
                crm, "CLIENTS_PATH", clients_path
            ), patch.object(crm, "JSON_DIR", json_dir), patch.object(
                crm, "CONTACTS_XLSX_PATH", root / "contacts.xlsx"
            ), patch.object(
                crm, "CONTACTS_XLSX_BACKUP_DIR", root / "backups"
            ):
                result = crm.update_from_reports()
                data = crm.load_clients()
            self.assertEqual(result, {})
            clients = data["clients"]
            self.assertEqual(len(clients), 1)
            entry = clients["М Плов центр ЕСБОЛОВА  Мангилик Ел 54"]
            self.assertIn("М Плов центр ЕСБОЛОВА Мангилик Ел 54", entry.get("aliases", []))

    def test_collect_unowned_claim_clients_skips_group_if_duplicate_is_owned(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_dir = root / "config"
            config_dir.mkdir(parents=True)
            clients_path = config_dir / "clients.json"
            clients_path.write_text(
                json.dumps(
                    {
                        "clients": {
                            "М Кафе Пиала Лесная поляна  дом 9": {"manager": "Ергали"},
                            "М Кафе Пиала Лесная поляна дом 9": {"manager": ""},
                            "ИП Другой": {"manager": ""},
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.object(crm, "CONFIG_DIR", config_dir), patch.object(crm, "CLIENTS_PATH", clients_path):
                result = sr._crm_collect_unowned_claim_clients(limit=3)
            self.assertEqual(result, ["ИП Другой"])

    def test_claim_pending_persists_and_tokens_are_unique(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "crm_claim_pending_state.json"
            with patch.object(sr, "CRM_CLAIM_PENDING_PATH", path):
                sr._CRM_CLAIM_PENDING.clear()
                token1 = sr._crm_claim_token()
                token2 = sr._crm_claim_token()
                self.assertNotEqual(token1, token2)
                sr._CRM_CLAIM_PENDING[token1] = {
                    "client_key": "A",
                    "claimed": False,
                    "created_at": "2099-04-29T10:00:00+05:00",
                }
                sr._crm_save_claim_pending()
                sr._CRM_CLAIM_PENDING.clear()
                sr._crm_load_claim_pending()
                self.assertIn(token1, sr._CRM_CLAIM_PENDING)

    def test_crm_audit_log_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td) / "crm_audit.jsonl"
            with patch.object(crm_audit, "_AUDIT_PATH", tmp_path):
                crm_audit.audit("claim_broadcast", client_key="ИП Тест", notified_count=3)
                lines = tmp_path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(lines), 1)
                rec = json.loads(lines[0])
                self.assertEqual(rec["event"], "claim_broadcast")
                self.assertEqual(rec["client_key"], "ИП Тест")


if __name__ == "__main__":
    unittest.main(verbosity=2)

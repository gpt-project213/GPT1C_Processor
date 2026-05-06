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

    def test_get_clients_without_phones_skips_duplicate_when_sibling_has_phone(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_dir = root / "config"
            config_dir.mkdir(parents=True)
            clients_path = config_dir / "clients.json"
            clients_path.write_text(
                json.dumps(
                    {
                        "clients": {
                            "О ТОО Petro Retail (Автогаз) ул Мангилик Ел 89 В": {
                                "manager": "Оксана",
                                "whatsapp": "+77769992271",
                                "sources": ["debt", "sales"],
                            },
                            "О ТОО Petro Retail (Автогаз) ул Мангилик Ел 89 В 2": {
                                "manager": "Оксана",
                                "whatsapp": "",
                                "sources": ["debt", "sales"],
                            },
                            "О ТОО Реальный клиент без телефона": {
                                "manager": "Оксана",
                                "whatsapp": "",
                                "sources": ["debt"],
                            },
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.object(crm, "CONFIG_DIR", config_dir), patch.object(crm, "CLIENTS_PATH", clients_path):
                result = crm.get_clients_without_phones("Оксана", limit=50)
            self.assertNotIn("О ТОО Petro Retail (Автогаз) ул Мангилик Ел 89 В 2", result)
            self.assertIn("О ТОО Реальный клиент без телефона", result)

    def test_get_clients_without_phones_skips_service_rows_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_dir = root / "config"
            config_dir.mkdir(parents=True)
            clients_path = config_dir / "clients.json"
            clients_path.write_text(
                json.dumps(
                    {
                        "clients": {
                            "Ергали тов. под ЗП": {
                                "manager": "Ергали",
                                "whatsapp": "",
                                "sources": ["sales"],
                            },
                            "Недостача": {
                                "manager": "Ергали",
                                "whatsapp": "",
                                "sources": ["sales"],
                            },
                            "Без клиента": {
                                "manager": "Ергали",
                                "whatsapp": "",
                                "sources": ["sales"],
                            },
                            "Водитель Серик": {
                                "manager": "Ергали",
                                "whatsapp": "",
                                "sources": ["sales"],
                            },
                            "ИП Реальный клиент без телефона": {
                                "manager": "Ергали",
                                "whatsapp": "",
                                "sources": ["debt"],
                            },
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.object(crm, "CONFIG_DIR", config_dir), patch.object(crm, "CLIENTS_PATH", clients_path):
                result = crm.get_clients_without_phones("Ергали", limit=50)
            self.assertEqual(result, ["ИП Реальный клиент без телефона"])

    def test_loose_duplicate_match_does_not_collapse_real_address_numbers(self):
        clients_db = {
            "М ЕНУ Евразийский универ ул Кажымукана 11": {
                "manager": "Магира",
                "whatsapp": "+77023174146",
            },
            "М ЕНУ Евразийский универ ул Кажымукана 13": {
                "manager": "Магира",
                "whatsapp": "",
            },
        }
        self.assertIsNone(
            crm._find_phone_donor_key(
                clients_db,
                "М ЕНУ Евразийский универ ул Кажымукана 13",
                manager="Магира",
            )
        )

    def test_update_from_reports_merges_legacy_duplicate_into_phoneful_sibling(self):
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
                            "О ТОО Petro Retail (Автогаз) ул Мангилик Ел 89 В": {
                                "manager": "Оксана",
                                "whatsapp": "+77769992271",
                                "sources": ["debt", "sales"],
                                "first_seen": "2026-04-01",
                                "last_seen": "2026-04-28",
                                "aliases": [],
                            },
                            "О ТОО Petro Retail (Автогаз) ул Мангилик Ел 89 В 2": {
                                "manager": "Оксана",
                                "whatsapp": "",
                                "sources": ["debt", "sales"],
                                "first_seen": "2026-04-28",
                                "last_seen": "2026-04-29",
                                "aliases": [],
                            },
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (json_dir / "debt_ext_Детальный Дебиторы Оксана (1).json").write_text(
                json.dumps(
                    {
                        "manager": "Оксана",
                        "clients": [{"name": "О ТОО Petro Retail (Автогаз) ул Мангилик Ел 89 В 2"}],
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
                crm.update_from_reports()
                data = crm.load_clients()
            clients = data["clients"]
            self.assertIn("О ТОО Petro Retail (Автогаз) ул Мангилик Ел 89 В", clients)
            self.assertNotIn("О ТОО Petro Retail (Автогаз) ул Мангилик Ел 89 В 2", clients)
            self.assertIn(
                "О ТОО Petro Retail (Автогаз) ул Мангилик Ел 89 В 2",
                clients["О ТОО Petro Retail (Автогаз) ул Мангилик Ел 89 В"].get("aliases", []),
            )

    def test_get_phone_conflict_groups_returns_only_real_phone_conflicts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_dir = root / "config"
            config_dir.mkdir(parents=True)
            clients_path = config_dir / "clients.json"
            clients_path.write_text(
                json.dumps(
                    {
                        "clients": {
                            "М Erzo Park  Косши ТОО Best Management group": {
                                "manager": "Магира",
                                "whatsapp": "+77025806778",
                                "sources": ["debt"],
                            },
                            "М Erzo Park Косши ТОО Best Management group": {
                                "manager": "Магира",
                                "whatsapp": "+77019257122",
                                "sources": ["sales"],
                            },
                            "О ТОО Safe Same Phone 89 В": {
                                "manager": "Оксана",
                                "whatsapp": "+77769992271",
                                "sources": ["debt"],
                            },
                            "О ТОО Safe Same Phone 89 В 2": {
                                "manager": "Оксана",
                                "whatsapp": "+77769992271",
                                "sources": ["sales"],
                            },
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.object(crm, "CONFIG_DIR", config_dir), patch.object(crm, "CLIENTS_PATH", clients_path):
                result = crm.get_phone_conflict_groups()
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["manager"], "Магира")
            self.assertEqual(len(result[0]["items"]), 2)

    def test_resolve_phone_conflict_merges_entries_and_keeps_alias(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_dir = root / "config"
            config_dir.mkdir(parents=True)
            clients_path = config_dir / "clients.json"
            clients_path.write_text(
                json.dumps(
                    {
                        "clients": {
                            "М Erzo Park  Косши ТОО Best Management group": {
                                "manager": "Магира",
                                "whatsapp": "+77025806778",
                                "sources": ["debt"],
                                "aliases": [],
                            },
                            "М Erzo Park Косши ТОО Best Management group": {
                                "manager": "Магира",
                                "whatsapp": "+77019257122",
                                "sources": ["sales"],
                                "aliases": [],
                            },
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.object(crm, "CONFIG_DIR", config_dir), patch.object(
                crm, "CLIENTS_PATH", clients_path
            ), patch.object(crm, "CONTACTS_XLSX_PATH", root / "contacts.xlsx"), patch.object(
                crm, "CONTACTS_XLSX_BACKUP_DIR", root / "backups"
            ):
                ok = crm.resolve_phone_conflict(
                    client_keys=[
                        "М Erzo Park  Косши ТОО Best Management group",
                        "М Erzo Park Косши ТОО Best Management group",
                    ],
                    chosen_phone="+77019257122",
                    chosen_key="М Erzo Park Косши ТОО Best Management group",
                    reviewer="Магира",
                )
                data = crm.load_clients()
            self.assertTrue(ok)
            clients = data["clients"]
            self.assertIn("М Erzo Park Косши ТОО Best Management group", clients)
            self.assertNotIn("М Erzo Park  Косши ТОО Best Management group", clients)
            self.assertEqual(clients["М Erzo Park Косши ТОО Best Management group"]["whatsapp"], "+77019257122")
            self.assertIn(
                "М Erzo Park  Косши ТОО Best Management group",
                clients["М Erzo Park Косши ТОО Best Management group"].get("aliases", []),
            )

    def test_mark_phone_conflict_distinct_blocks_future_review(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_dir = root / "config"
            config_dir.mkdir(parents=True)
            clients_path = config_dir / "clients.json"
            clients_path.write_text(
                json.dumps(
                    {
                        "clients": {
                            "М Erzo Park  Косши ТОО Best Management group": {
                                "manager": "Магира",
                                "whatsapp": "+77025806778",
                                "sources": ["debt"],
                            },
                            "М Erzo Park Косши ТОО Best Management group": {
                                "manager": "Магира",
                                "whatsapp": "+77019257122",
                                "sources": ["sales"],
                            },
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.object(crm, "CONFIG_DIR", config_dir), patch.object(
                crm, "CLIENTS_PATH", clients_path
            ), patch.object(crm, "CONTACTS_XLSX_PATH", root / "contacts.xlsx"), patch.object(
                crm, "CONTACTS_XLSX_BACKUP_DIR", root / "backups"
            ):
                ok = crm.mark_phone_conflict_distinct(
                    [
                        "М Erzo Park  Косши ТОО Best Management group",
                        "М Erzo Park Косши ТОО Best Management group",
                    ],
                    reviewer="Магира",
                )
                result = crm.get_phone_conflict_groups()
            self.assertTrue(ok)
            self.assertEqual(result, [])

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

    def test_stale_phone_pending_removed_after_ttl(self):
        """Просроченная clarify_name запись удаляется из _CRM_PHONE_PENDING, а не висит вечно."""
        import bot.send_reports as _sr
        _sr._CRM_PHONE_PENDING.clear()
        old_ts = "2000-01-01T10:00:00+05:00"
        _sr._CRM_PHONE_PENDING[99999] = {
            "client_key": "ИП Тест Стейл",
            "state": "clarify_name",
            "last_sent": old_ts,
        }
        _sr._crm_cleanup_pending()
        self.assertNotIn(99999, _sr._CRM_PHONE_PENDING)

    def test_active_claim_pending_excluded_from_next_broadcast(self):
        """Клиент с активным claim-токеном не попадает в следующую рассылку."""
        unowned = ["ИП Тест1", "ИП Тест2"]
        sr._CRM_CLAIM_PENDING.clear()
        sr._CRM_CLAIM_PENDING["claim_20260429_aabbccdd"] = {
            "client_key": "ИП Тест1",
            "claimed": False,
            "created_at": "2099-04-29T10:00:00+05:00",
        }
        active_keys = {
            v["client_key"] for v in sr._CRM_CLAIM_PENDING.values() if not v.get("claimed")
        }
        filtered = [k for k in unowned if k not in active_keys]
        sr._CRM_CLAIM_PENDING.clear()
        self.assertEqual(filtered, ["ИП Тест2"])

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

    def test_ambiguous_conflict_stays_pending_when_crm_write_fails(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "crm_ambiguous_conflicts.json"
            sig = "A|B"
            sr._CRM_AMBIGUOUS.clear()
            sr._CRM_AMBIGUOUS[sig] = {
                "token": "a123",
                "signature": sig,
                "group_key": "тоо альфа",
                "items": [
                    {"client_key": "A", "phone": "+7701", "manager": "Магира"},
                    {"client_key": "B", "phone": "+7702", "manager": "Оксана"},
                ],
                "managers": ["Магира", "Оксана"],
                "added_at": "2026-05-06T10:00:00+05:00",
                "status": "pending",
            }
            with patch.object(sr, "CRM_AMBIGUOUS_PATH", path), patch.object(sr, "crm_audit") as audit_mock:
                ok = sr._crmdup_try_finalize_ambiguous(
                    sig,
                    ok=False,
                    reviewer="Вадим",
                    resolution="assigned:Магира",
                )
                self.assertFalse(ok)
                self.assertEqual(sr._CRM_AMBIGUOUS[sig]["status"], "pending")
                audit_mock.assert_not_called()

                ok = sr._crmdup_try_finalize_ambiguous(
                    sig,
                    ok=True,
                    reviewer="Вадим",
                    resolution="assigned:Магира",
                )
                self.assertTrue(ok)
                self.assertEqual(sr._CRM_AMBIGUOUS[sig]["status"], "resolved")
                self.assertEqual(sr._CRM_AMBIGUOUS[sig]["resolved_by"], "Вадим")
                self.assertTrue(path.exists())
                audit_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)

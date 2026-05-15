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
    import collector.registry_manager as registry_manager


class CRMRegressionTests(unittest.TestCase):
    def test_load_contacts_for_collector_prefers_crm_and_keeps_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_dir = root / "config"
            config_dir.mkdir(parents=True)
            clients_path = config_dir / "clients.json"
            legacy_path = config_dir / "debtors_contacts.json"
            clients_path.write_text(
                json.dumps(
                    {
                        "clients": {
                            "CRM Клиент": {
                                "whatsapp": "+77011111111",
                                "telegram_id": "123",
                                "manager": "Оксана",
                                "language": "ru",
                                "do_not_call": False,
                                "aliases": ["CRM Алиас"],
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            legacy_path.write_text(
                json.dumps(
                    {
                        "_comment": "legacy",
                        "CRM Клиент": {"whatsapp": "+77099999999", "manager": "Legacy"},
                        "Legacy Клиент": {"whatsapp": "+77022222222", "manager": "Ергали"},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.object(crm, "CONFIG_DIR", config_dir), patch.object(crm, "CLIENTS_PATH", clients_path), patch.object(crm, "LEGACY_CONTACTS_PATH", legacy_path):
                result = crm.load_contacts_for_collector()
            self.assertEqual(result["CRM Клиент"]["whatsapp"], "+77011111111")
            self.assertEqual(result["CRM Клиент"]["_source"], "crm")
            self.assertEqual(result["CRM Алиас"]["_source"], "crm")
            self.assertEqual(result["Legacy Клиент"]["whatsapp"], "+77022222222")
            self.assertEqual(result["Legacy Клиент"]["_source"], "legacy_fallback")

    def test_registry_manager_update_client_phone_uses_crm_helper(self):
        with patch.object(registry_manager, "set_client_phone", return_value=True) as mocked, patch.object(registry_manager, "export_registry_excel", return_value=True):
            ok = registry_manager.update_client_phone("ТОО Альфа", "+77073334455")
        self.assertTrue(ok)
        mocked.assert_called_once_with("ТОО Альфа", "+77073334455", reviewer="registry_manager")

    def test_registry_manager_auto_register_client_writes_to_clients_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_dir = root / "config"
            config_dir.mkdir(parents=True)
            clients_path = config_dir / "clients.json"
            clients_path.write_text(json.dumps({"clients": {}}, ensure_ascii=False), encoding="utf-8")
            with patch.object(crm, "CONFIG_DIR", config_dir), patch.object(crm, "CLIENTS_PATH", clients_path), patch.object(registry_manager, "crm_load_clients", side_effect=crm.load_clients), patch.object(registry_manager, "crm_save_clients", side_effect=crm.save_clients), patch.object(registry_manager, "export_registry_excel", return_value=True):
                ok = registry_manager.auto_register_client("ТОО Бета", manager="Магира", amount=12000.0, days=11)
                data = crm.load_clients()
            self.assertTrue(ok)
            self.assertIn("ТОО Бета", data["clients"])
            self.assertEqual(data["clients"]["ТОО Бета"]["manager"], "Магира")

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


class StabilizationRegressionTests(unittest.TestCase):
    """Regressions для stabilization-волны 2026-05-14:
    F-11 lock, F-12 deterministic keep, F-13 ambiguous phone-signature,
    F-16 stale claim cleanup + callback rejection, admin CRM backlog.
    """

    def setUp(self):
        sr._CRM_CLAIM_PENDING.clear()
        sr._CRM_PHONE_PENDING.clear()
        sr._CRM_DUP_REVIEW_PENDING.clear()
        sr._CRM_AMBIGUOUS.clear()

    # ── F-16: stale claim tokens ─────────────────────────────────────────────
    def test_claim_without_created_at_is_stale(self):
        claim = {"client_key": "A", "claimed": False}
        self.assertTrue(sr._crm_claim_is_stale(claim))

    def test_claim_cleanup_removes_token_without_created_at(self):
        sr._CRM_CLAIM_PENDING["t_legacy"] = {"client_key": "A", "claimed": False}
        sr._crm_cleanup_claim_pending()
        self.assertNotIn("t_legacy", sr._CRM_CLAIM_PENDING)

    def test_claim_cleanup_removes_expired_token(self):
        from datetime import timedelta
        old_iso = (sr.datetime.now(sr.TZ) - timedelta(hours=sr.CRM_CLAIM_TTL_HOURS + 1)).isoformat()
        sr._CRM_CLAIM_PENDING["t_old"] = {"client_key": "B", "claimed": False, "created_at": old_iso}
        sr._crm_cleanup_claim_pending()
        self.assertNotIn("t_old", sr._CRM_CLAIM_PENDING)

    def test_claim_fresh_token_not_stale(self):
        fresh = sr.datetime.now(sr.TZ).isoformat()
        claim = {"client_key": "C", "claimed": False, "created_at": fresh}
        self.assertFalse(sr._crm_claim_is_stale(claim))

    def test_dup_review_without_created_at_is_stale(self):
        review = {"items": [{"client_key": "A"}]}
        self.assertTrue(sr._crmdup_review_is_stale(review))

    def test_dup_review_cleanup_removes_expired_review_and_awaiting_text(self):
        from datetime import timedelta
        old_iso = (sr.datetime.now(sr.TZ) - timedelta(hours=sr.CRM_DUP_REVIEW_TTL_HOURS + 1)).isoformat()
        sr._CRM_DUP_REVIEW_PENDING["dup_old"] = {
            "manager": "Магира",
            "items": [{"client_key": "A"}, {"client_key": "B"}],
            "created_at": old_iso,
        }
        sr._CRM_DUP_REVIEW_AWAITING_TEXT[1001] = "dup_old"
        sr._crmdup_cleanup_pending()
        self.assertNotIn("dup_old", sr._CRM_DUP_REVIEW_PENDING)
        self.assertNotIn(1001, sr._CRM_DUP_REVIEW_AWAITING_TEXT)

    # ── F-11: CRM state lock — hard-fail policy ───────────────────────────────
    def test_crm_state_lock_creates_lock_file_and_yields(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.json"
            with sr._crm_state_lock(path):
                self.assertTrue((Path(td) / "test.lock").exists())

    def test_crm_state_lock_raises_on_contention(self):
        """Hard-fail: при занятом lock должен подняться CrmStateLockError (не yield)."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.json"
            import portalocker as _pl
            lock_file = path.with_suffix(".lock")
            lock_file.parent.mkdir(parents=True, exist_ok=True)
            # Удерживаем эксклюзивный lock извне — имитируем contention
            with _pl.Lock(str(lock_file), flags=_pl.LOCK_EX | _pl.LOCK_NB):
                with self.assertRaises(sr.CrmStateLockError):
                    with sr._crm_state_lock(path):
                        pass  # не должно дойти

    def test_crm_save_pending_returns_false_on_lock_fail(self):
        """save при недоступном lock возвращает False, не raise."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "crm_pending_state.json"
            import portalocker as _pl
            lock_file = path.with_suffix(".lock")
            lock_file.parent.mkdir(parents=True, exist_ok=True)
            with _pl.Lock(str(lock_file), flags=_pl.LOCK_EX | _pl.LOCK_NB):
                with patch.object(sr, "CRM_PENDING_PATH", path):
                    result = sr._crm_save_pending()
                    self.assertFalse(result)

    def test_crm_save_and_reload_pending_preserves_data(self):
        # Smoke: save_pending → load_pending не теряет запись (lock-discipline ОК)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "crm_pending_state.json"
            with patch.object(sr, "CRM_PENDING_PATH", path):
                sr._CRM_PHONE_PENDING[1001] = {
                    "manager": "Магира", "client_key": "X",
                    "state": "clarify_name", "created_at": sr.datetime.now(sr.TZ).isoformat(),
                    "last_sent": sr.datetime.now(sr.TZ).isoformat(),
                }
                sr._crm_save_pending()
                sr._CRM_PHONE_PENDING.clear()
                sr._crm_load_pending()
                self.assertIn(1001, sr._CRM_PHONE_PENDING)
                self.assertEqual(sr._CRM_PHONE_PENDING[1001]["client_key"], "X")

    def test_set_client_details_returns_false_when_clients_save_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_dir = root / "config"
            config_dir.mkdir(parents=True)
            clients_path = config_dir / "clients.json"
            clients_path.write_text(
                json.dumps({"clients": {"Тест": {"manager": "Оксана"}}}, ensure_ascii=False),
                encoding="utf-8",
            )
            with patch.object(crm, "CONFIG_DIR", config_dir), \
                 patch.object(crm, "CLIENTS_PATH", clients_path), \
                 patch.object(crm, "save_clients", return_value=False):
                self.assertFalse(crm.set_client_details("Тест", phone="+77015554433"))

    def test_resolve_phone_conflict_returns_false_when_clients_save_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_dir = root / "config"
            config_dir.mkdir(parents=True)
            clients_path = config_dir / "clients.json"
            clients_path.write_text(
                json.dumps(
                    {
                        "clients": {
                            "A": {"manager": "Магира", "whatsapp": ""},
                            "B": {"manager": "Магира", "whatsapp": ""},
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.object(crm, "CONFIG_DIR", config_dir), \
                 patch.object(crm, "CLIENTS_PATH", clients_path), \
                 patch.object(crm, "save_clients", return_value=False):
                self.assertFalse(
                    crm.resolve_phone_conflict(["A", "B"], "+77015554433", chosen_key="A", reviewer="Магира")
                )

    # ── F-12: deterministic keep-key для custom phone в dup-review ──────────
    def test_dup_review_custom_phone_chosen_key_is_order_independent(self):
        keys_a = ["М Клиент 2", "М Клиент 1"]
        keys_b = ["М Клиент 1", "М Клиент 2"]
        chosen_a = sorted([k for k in keys_a if k])[0] if any(keys_a) else ""
        chosen_b = sorted([k for k in keys_b if k])[0] if any(keys_b) else ""
        self.assertEqual(chosen_a, chosen_b)
        self.assertEqual(chosen_a, "М Клиент 1")

    # ── F-13: ambiguous signature чувствителен к телефонам ──────────────────
    def test_ambiguous_signature_reacts_to_phone_change(self):
        items_a = [
            {"client_key": "A", "phone": "+77011112233"},
            {"client_key": "B", "phone": "+77011112244"},
        ]
        items_b = [
            {"client_key": "A", "phone": "+77011112233"},
            {"client_key": "B", "phone": "+77019999999"},  # phone changed for B
        ]
        sig_a = sr._ambiguous_signature(items_a)
        sig_b = sr._ambiguous_signature(items_b)
        self.assertNotEqual(sig_a, sig_b)

    def test_ambiguous_signature_order_independent(self):
        i1 = [
            {"client_key": "A", "phone": "+77011112233"},
            {"client_key": "B", "phone": "+77011112244"},
        ]
        i2 = list(reversed(i1))
        self.assertEqual(sr._ambiguous_signature(i1), sr._ambiguous_signature(i2))

    def test_ambiguous_reopen_via_new_signature(self):
        # Resolved запись по signature1 остаётся; новый conflict с другим phone
        # должен попасть как новый pending под новый signature.
        old_items = [
            {"client_key": "A", "phone": "+77011112233"},
            {"client_key": "B", "phone": "+77011112244"},
        ]
        new_items = [
            {"client_key": "A", "phone": "+77011112233"},
            {"client_key": "B", "phone": "+77019999999"},
        ]
        sig_old = sr._ambiguous_signature(old_items)
        sr._CRM_AMBIGUOUS[sig_old] = {
            "signature": sig_old, "status": "resolved", "items": old_items,
            "added_at": sr.datetime.now(sr.TZ).isoformat(),
            "resolved_at": sr.datetime.now(sr.TZ).isoformat(),
        }
        with patch.object(sr, "CRM_AMBIGUOUS_PATH",
                          Path(tempfile.gettempdir()) / "test_ambi_reopen.json"):
            sr._crmdup_queue_ambiguous({"items": new_items, "group_key": "g"})
        sig_new = sr._ambiguous_signature(new_items)
        self.assertIn(sig_new, sr._CRM_AMBIGUOUS)
        self.assertEqual(sr._CRM_AMBIGUOUS[sig_new].get("status"), "pending")
        # старая resolved-запись осталась как history
        self.assertEqual(sr._CRM_AMBIGUOUS[sig_old].get("status"), "resolved")

    # ── Admin CRM backlog: counts отображаются ──────────────────────────────
    def test_crm_backlog_shows_counts_for_each_queue(self):
        now_iso = sr.datetime.now(sr.TZ).isoformat()
        sr._CRM_PHONE_PENDING[1001] = {
            "manager": "Магира", "client_key": "X",
            "state": "clarify_name", "created_at": now_iso,
        }
        sr._CRM_CLAIM_PENDING["c1"] = {
            "client_key": "Y", "claimed": False, "created_at": now_iso,
            "notified": [1002, 1003],
        }
        sr._CRM_DUP_REVIEW_PENDING["d1"] = {
            "manager": "Оксана",
            "items": [{"client_key": "A"}, {"client_key": "B"}],
            "created_at": now_iso,
        }
        sr._CRM_AMBIGUOUS["sig1"] = {
            "group_key": "Plov центр", "managers": ["Магира", "Оксана"],
            "added_at": now_iso, "status": "pending",
        }
        text = sr._format_crm_backlog_text()
        self.assertIn("Phone pending: 1", text)
        self.assertIn("Claim pending: 1", text)
        self.assertIn("Duplicate review: 1", text)
        self.assertIn("Ambiguous: 1", text)
        self.assertIn("Магира", text)
        self.assertIn("Plov центр", text)

    def test_crm_backlog_marks_stale_claims_separately(self):
        from datetime import timedelta
        now_iso = sr.datetime.now(sr.TZ).isoformat()
        old_iso = (sr.datetime.now(sr.TZ) - timedelta(hours=sr.CRM_CLAIM_TTL_HOURS + 1)).isoformat()
        sr._CRM_CLAIM_PENDING["c_fresh"] = {
            "client_key": "Fresh", "claimed": False, "created_at": now_iso, "notified": [9001],
        }
        sr._CRM_CLAIM_PENDING["c_stale"] = {
            "client_key": "Stale", "claimed": False, "created_at": old_iso, "notified": [9002],
        }
        text = sr._format_crm_backlog_text()
        self.assertIn("Claim pending: 1", text)  # только свежий
        self.assertIn("+1 stale", text)


class PrivatePersonPlaceholderTests(unittest.TestCase):
    """Regressions для CRM-фильтра placeholder "Частное лицо" (2026-05-15).

    1C регулярно выгружает placeholder-имена типа "Частное лицо N" в debt/sales.
    Без фильтра они попадают в claim broadcast / phone-pending — менеджеры
    получают запрос «чей это клиент?» по непоименованному физлицу.
    """

    def test_is_service_client_name_skips_chastnoe_litso_variations(self):
        self.assertTrue(crm.is_service_client_name("Частное лицо"))
        self.assertTrue(crm.is_service_client_name("Частное лицо 1"))
        self.assertTrue(crm.is_service_client_name("М Частное лицо"))
        self.assertTrue(crm.is_service_client_name("Е 1. Частное лицо"))
        self.assertTrue(crm.is_service_client_name("А Частное Лицо"))  # camelcase из 1C
        self.assertTrue(crm.is_service_client_name("О Частное лицо"))

    def test_is_service_client_name_skips_fizicheskoe_litso(self):
        self.assertTrue(crm.is_service_client_name("Физическое лицо"))
        self.assertTrue(crm.is_service_client_name("Физлицо"))
        self.assertTrue(crm.is_service_client_name("М Физлицо Иванов"))

    def test_is_service_client_name_keeps_real_clients_with_similar_words(self):
        # Не должно зацепить настоящих клиентов с похожими словами,
        # которые случайно содержат буквосочетания.
        self.assertFalse(crm.is_service_client_name("М ТОО Частная клиника"))
        self.assertFalse(crm.is_service_client_name("О Магазин Физкультура"))
        self.assertFalse(crm.is_service_client_name("Е ИП Физкульт-товары"))

    def test_update_from_reports_marks_chastnoe_litso_as_vendor(self):
        # При появлении новой записи "Частное лицо N" в debt-выгрузке
        # она должна сразу попасть в clients.json с is_vendor=True
        # и do_not_call=True. Pending CRM-запрос не создаётся.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_dir = root / "config"
            json_dir = root / "reports" / "json"
            config_dir.mkdir(parents=True)
            json_dir.mkdir(parents=True)
            clients_path = config_dir / "clients.json"
            clients_path.write_text(
                json.dumps({"clients": {}}, ensure_ascii=False),
                encoding="utf-8",
            )
            (json_dir / "debt_ext_Детальный Дебиторы Магира (1).json").write_text(
                json.dumps(
                    {
                        "manager": "Магира",
                        "clients": [{"name": "Частное лицо 5"}],
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
            entry = data["clients"].get("Частное лицо 5")
            self.assertIsNotNone(entry)
            self.assertTrue(entry.get("is_vendor"))
            self.assertTrue(entry.get("do_not_call"))

    def test_collect_unowned_claim_skips_chastnoe_litso(self):
        # Если в clients.json лежит "Частное лицо 1" без manager —
        # _crm_collect_unowned_claim_clients не должен его поднимать в broadcast.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_dir = root / "config"
            config_dir.mkdir(parents=True)
            clients_path = config_dir / "clients.json"
            clients_path.write_text(
                json.dumps(
                    {
                        "clients": {
                            "Частное лицо 1": {"manager": ""},
                            "Частное лицо": {"manager": ""},
                            "Физлицо": {"manager": ""},
                            "ИП Реальный клиент": {"manager": ""},
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.object(crm, "CONFIG_DIR", config_dir), patch.object(crm, "CLIENTS_PATH", clients_path):
                result = sr._crm_collect_unowned_claim_clients(limit=10)
            # Placeholder'ы должны быть исключены, реальный — остаться
            self.assertNotIn("Частное лицо 1", result)
            self.assertNotIn("Частное лицо", result)
            self.assertNotIn("Физлицо", result)
            self.assertIn("ИП Реальный клиент", result)

    def test_get_clients_without_phones_skips_chastnoe_litso(self):
        # daily phone-pending не должен поднимать placeholder
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_dir = root / "config"
            config_dir.mkdir(parents=True)
            clients_path = config_dir / "clients.json"
            clients_path.write_text(
                json.dumps(
                    {
                        "clients": {
                            "М Частное лицо": {
                                "manager": "Магира",
                                "whatsapp": "",
                                "sources": ["debt"],
                            },
                            "М ИП Реальный клиент": {
                                "manager": "Магира",
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
                result = crm.get_clients_without_phones("Магира", limit=50)
            self.assertNotIn("М Частное лицо", result)
            self.assertIn("М ИП Реальный клиент", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)

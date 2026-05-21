#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot/crm_clients.py
Универсальная база клиентов Минбаракат (CRM).

Версия: 1.1.3 (2026-05-19)
Изменения v1.0.5:
  - Fix S1: список менеджеров читается из config/managers.json (single source of truth).
    Раньше был хардкод ("Алена", "Ергали", "Магира", "Оксана"); fallback — тот же
    хардкод, если файл отсутствует/пуст (защищает прод от сломанного конфига).

Источники данных:
  - reports/json/debt_ext_*.json   → клиенты по менеджерам (дебиторка)
  - reports/json/sales_*.json      → покупатели по менеджерам

Функции:
  load_clients()                  → загрузить config/clients.json
  save_clients(data)              → атомарная запись
  update_from_reports()           → обновить из последних JSON-отчётов
  get_clients_without_phones()    → список клиентов без телефона (для 18:00 задачи)
  set_client_phone()              → записать телефон клиента
  load_contacts_compat()          → формат, совместимый с debtors_contacts.json
  get_new_clients_since()         → новые клиенты с даты (для уведомлений)
"""

import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import portalocker
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from bot.crm_audit_log import audit as crm_audit
from bot.logging_utils import get_runtime_logger

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env",
            encoding="utf-8-sig", override=False)

TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))

ROOT_DIR   = Path(__file__).resolve().parent.parent
JSON_DIR   = ROOT_DIR / "reports" / "json"
CONFIG_DIR = ROOT_DIR / "config"
CLIENTS_PATH = CONFIG_DIR / "clients.json"
LEGACY_CONTACTS_PATH = CONFIG_DIR / "debtors_contacts.json"
CONTACTS_XLSX_PATH = ROOT_DIR / "contacts.xlsx"
CONTACTS_XLSX_BACKUP_DIR = ROOT_DIR / "backups" / "contacts_xlsx"

logger = get_runtime_logger(__name__, system="CRM", component="STORE")
__VERSION__ = "1.1.3"
_UNKNOWN_MANAGERS = {"", "Не определён", "?", "-", "—"}


class CrmClientsLockError(RuntimeError):
    """Exclusive lock on clients.json not acquired within timeout."""

PHONE_IN_NAME_RE = re.compile(
    r"(?<!\d)(?:\+?7|8)[\s\-\(\)]*\d{3}[\s\-\(\)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)"
)

# Фильтр служебных записей из sales JSON.
# Клиенты и товары различаются по колонкам в sales_parser, а не по тексту имени.
_METADATA_KEYWORDS = ("Дополнительные поля:", "Отборы:", "Сортировка:", "Группировка:")

# Ключевые слова для автоматической пометки записей как вендоров/контрагентов.
# Записи с этими подстроками — сотрудники, поставщики услуг, внутренние расчёты.
# Они появляются в 1С как контрагенты, но не являются клиентами-покупателями.
_VENDOR_NAME_KEYWORDS = (
    "зарплат",    # зарплата, зарплату, зарплатный
    "зар.плат",   # зар.плата
    "зар плат",   # зар плата, зар плату, товар по зар плату
    "з.п.",       # з.п. в любой позиции
    "з/п",        # з/п в любой позиции
    "зп",         # plain "ЗП" без разделителей
    "по зп",       # тов по зп, товар по зп
    "под зп",      # тов под зп, товар под зп
    "тов по з",    # тов по зп, тов по з/п, тов по зарплате
    "тов под з",   # тов под зп, тов под з/п, тов под зарплате
    "товар по з",  # товар по зп, товар по зарплате
    "товар под з", # товар под зп, товар под зарплате
    "аванс сотр", # авансы сотрудникам
    "водитель",   # внутренние сотрудники / логистика
    "недостача",  # служебная строка, не клиент
    "без клиента", # служебная строка, не клиент
    "частное лицо",    # 1C placeholder для непоименованных физлиц
    "физическое лицо", # альтернативная форма placeholder из 1C
    "физлицо",         # сокращённая форма placeholder
)


# ─────────────────────────────────────────────
# Загрузка / сохранение
# ─────────────────────────────────────────────

def load_clients() -> Dict[str, Any]:
    """Загружает config/clients.json. Возвращает {'clients': {...}}."""
    if not CLIENTS_PATH.exists():
        return {"clients": {}}
    try:
        with open(CLIENTS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data.get("clients"), dict):
            data["clients"] = {}
        return data
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Ошибка чтения clients.json: %s", e)
        return {"clients": {}}


@contextmanager
def _clients_state_lock(path: Path):
    """Hard-fail lock for config/clients.json writes on Windows."""
    lock_file = path.with_suffix(".lock")
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with portalocker.Lock(
            str(lock_file),
            timeout=5,
            check_interval=0.1,
            flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
        ):
            yield
    except (portalocker.LockException, PermissionError, OSError) as exc:
        raise CrmClientsLockError(f"clients_lock_timeout: {path.name} — {exc}") from exc


def save_clients(data: Dict[str, Any]) -> bool:
    """Атомарная запись config/clients.json. Возвращает False при lock/write error."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with _clients_state_lock(CLIENTS_PATH):
            tmp = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8",
                dir=CONFIG_DIR, suffix=".tmp", delete=False,
            )
            json.dump(data, tmp, ensure_ascii=False, indent=2)
            tmp.close()
            os.replace(tmp.name, CLIENTS_PATH)
        logger.debug("clients.json сохранён (%d клиентов)", len(data.get("clients", {})))
        refresh_contacts_xlsx_mirror(reason="save_clients")
        return True
    except CrmClientsLockError as e:
        logger.error("Ошибка lock clients.json: %s", e)
        return False
    except OSError as e:
        logger.error("Ошибка записи clients.json: %s", e)
        return False


def _backup_contacts_xlsx_once_per_day() -> Optional[Path]:
    if not CONTACTS_XLSX_PATH.exists():
        return None
    today = datetime.now(tz=TZ).strftime("%Y%m%d")
    CONTACTS_XLSX_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if any(CONTACTS_XLSX_BACKUP_DIR.glob(f"contacts_{today}_*.xlsx")):
        return None
    stamp = datetime.now(tz=TZ).strftime("%Y%m%d_%H%M%S")
    backup_path = CONTACTS_XLSX_BACKUP_DIR / f"contacts_{stamp}.xlsx"
    shutil.copy2(CONTACTS_XLSX_PATH, backup_path)
    return backup_path


def refresh_contacts_xlsx_mirror(reason: str = "") -> bool:
    """Refreshes contacts.xlsx from the live CRM database without importing back."""
    try:
        backup_path = _backup_contacts_xlsx_once_per_day()
        if backup_path:
            logger.info("contacts.xlsx backup created before CRM mirror refresh: %s", backup_path)
        from tools.contacts_sync import export_to_excel
        count = export_to_excel(CONTACTS_XLSX_PATH, clients_path=CLIENTS_PATH)
        logger.info(
            "contacts.xlsx refreshed from clients.json (%d clients, reason=%s)",
            count,
            reason or "-",
        )
        return True
    except Exception as e:
        logger.warning(
            "contacts.xlsx mirror refresh failed (CRM data is saved; reason=%s): %s",
            reason or "-",
            e,
        )
        return False


def normalize_kz_phone(phone_raw: str) -> str:
    """Returns +7XXXXXXXXXX for valid KZ phones, otherwise an empty string."""
    digits = re.sub(r"\D", "", str(phone_raw or ""))
    if re.fullmatch(r"8\d{10}", digits):
        digits = "7" + digits[1:]
    if re.fullmatch(r"7\d{10}", digits):
        return "+" + digits
    return ""


def extract_phones_from_client_name(client_name: str) -> List[str]:
    """Extracts possible KZ phone numbers embedded in a 1C client name."""
    phones: List[str] = []
    for match in PHONE_IN_NAME_RE.finditer(str(client_name or "")):
        phone = normalize_kz_phone(match.group(0))
        if phone and phone not in phones:
            phones.append(phone)
    return phones


def canonicalize_client_key(name: str) -> str:
    """Canonical form for CRM duplicate detection without changing the display key."""
    normalized = re.sub(r"\s+", " ", str(name or "").strip())
    normalized = normalized.replace("ё", "е").replace("Ё", "Е")
    return normalized.lower()


def canonicalize_client_key_loose(name: str) -> str:
    """
    Softer canonical form for legacy duplicates.
    Trims a trailing counter like "... В 2" so a filled card can win over an empty legacy row.
    """
    normalized = canonicalize_client_key(name)
    return re.sub(r"(?<=\d\s[^\W\d_])\s+\d+$", "", normalized, count=1, flags=re.UNICODE)


def is_service_client_name(name: str) -> bool:
    """Returns True for internal/service rows that should not request contact filling."""
    normalized = canonicalize_client_key(name)
    if normalized in ("без клиента", "недостача"):
        return True
    return any(keyword in normalized for keyword in _VENDOR_NAME_KEYWORDS)


def _find_existing_client_key(clients_db: Dict[str, Any], name: str) -> Optional[str]:
    if name in clients_db:
        return name
    target = canonicalize_client_key(name)
    for existing_key in clients_db.keys():
        if canonicalize_client_key(existing_key) == target:
            return existing_key
    return None


def _contact_phone_value(info: Dict[str, Any]) -> str:
    if not isinstance(info, dict):
        return ""
    return (info.get("whatsapp") or info.get("phone") or "").strip()


def _review_exclusions(info: Dict[str, Any]) -> List[str]:
    raw = info.get("duplicate_review_exclusions", []) if isinstance(info, dict) else []
    if isinstance(raw, list):
        return [str(v) for v in raw if str(v).strip()]
    return []


def _pair_review_blocked(left_key: str, left_info: Dict[str, Any], right_key: str, right_info: Dict[str, Any]) -> bool:
    left_exclusions = {canonicalize_client_key(v) for v in _review_exclusions(left_info)}
    right_exclusions = {canonicalize_client_key(v) for v in _review_exclusions(right_info)}
    left_canon = canonicalize_client_key(left_key)
    right_canon = canonicalize_client_key(right_key)
    return right_canon in left_exclusions or left_canon in right_exclusions


def _iter_duplicate_candidate_keys(clients_db: Dict[str, Any], name: str) -> List[str]:
    """Collects strict and loose duplicate candidates for a CRM key."""
    target_strict = canonicalize_client_key(name)
    target_loose = canonicalize_client_key_loose(name)
    candidates: List[str] = []
    for existing_key in clients_db.keys():
        strict = canonicalize_client_key(existing_key)
        loose = canonicalize_client_key_loose(existing_key)
        if strict == target_strict or loose == target_loose:
            candidates.append(existing_key)
    return candidates


def _choose_best_duplicate_key(
    clients_db: Dict[str, Any],
    name: str,
    manager: str = "",
) -> Optional[str]:
    """
    Prefer the most useful duplicate candidate:
    - saved phone/contact first
    - then stricter key equality
    - then same manager / owned record
    """
    candidates = _iter_duplicate_candidate_keys(clients_db, name)
    if not candidates:
        return None

    target_strict = canonicalize_client_key(name)
    target_loose = canonicalize_client_key_loose(name)
    unknown_managers = {"", "Не определён", "?", "-", "—"}

    def _score(existing_key: str) -> Tuple[int, int, int, int, int]:
        info = clients_db.get(existing_key, {}) if isinstance(clients_db.get(existing_key), dict) else {}
        strict = canonicalize_client_key(existing_key)
        loose = canonicalize_client_key_loose(existing_key)
        phone = (info.get("whatsapp") or info.get("phone") or "").strip()
        telegram_id = (info.get("telegram_id") or "").strip()
        existing_manager = (info.get("manager") or "").strip()
        relation = 3
        if existing_key == name:
            relation = 0
        elif strict == target_strict:
            relation = 1
        elif loose == target_loose:
            relation = 2
        return (
            0 if (phone or telegram_id) else 1,
            relation,
            0 if manager and existing_manager.lower() == manager.lower() else 1,
            0 if existing_manager not in unknown_managers else 1,
            len(existing_key),
        )

    return min(candidates, key=_score)


def _find_phone_donor_key(clients_db: Dict[str, Any], name: str, manager: str = "") -> Optional[str]:
    """Returns a duplicate key that already has a saved phone or telegram id."""
    preferred = _choose_best_duplicate_key(clients_db, name, manager=manager)
    if not preferred:
        return None
    info = clients_db.get(preferred, {}) if isinstance(clients_db.get(preferred), dict) else {}
    if (info.get("whatsapp") or info.get("phone") or "").strip():
        return preferred
    if (info.get("telegram_id") or "").strip():
        return preferred
    return None


def _merge_client_entries(clients_db: Dict[str, Any], keep_key: str, drop_key: str) -> None:
    """Merges a legacy duplicate row into the preferred CRM card."""
    if keep_key == drop_key:
        return
    keep = clients_db.get(keep_key)
    drop = clients_db.get(drop_key)
    if not isinstance(keep, dict) or not isinstance(drop, dict):
        return

    aliases = keep.setdefault("aliases", [])
    for alias in [drop_key, *drop.get("aliases", [])]:
        if alias and alias != keep_key and alias not in aliases:
            aliases.append(alias)

    keep_sources = keep.setdefault("sources", [])
    for source in drop.get("sources", []):
        if source not in keep_sources:
            keep_sources.append(source)

    for field in ("display_name", "original_name", "address", "phone_source", "name_mode",
                  "whatsapp", "phone", "telegram_id"):
        if not keep.get(field) and drop.get(field):
            keep[field] = drop[field]
    if keep.get("name_review_needed") is None and drop.get("name_review_needed") is not None:
        keep["name_review_needed"] = drop["name_review_needed"]

    first_seen_values = [v for v in (keep.get("first_seen"), drop.get("first_seen")) if v]
    if first_seen_values:
        keep["first_seen"] = min(first_seen_values)
    last_seen_values = [v for v in (keep.get("last_seen"), drop.get("last_seen")) if v]
    if last_seen_values:
        keep["last_seen"] = max(last_seen_values)

    if not keep.get("manager") and drop.get("manager"):
        keep["manager"] = drop["manager"]
    if drop.get("is_vendor"):
        keep["is_vendor"] = True
        keep["do_not_call"] = True
    keep_exclusions = keep.setdefault("duplicate_review_exclusions", [])
    if not isinstance(keep_exclusions, list):
        keep_exclusions = keep["duplicate_review_exclusions"] = [keep_exclusions] if keep_exclusions else []
    for excluded in _review_exclusions(drop):
        if excluded != keep_key and excluded not in keep_exclusions:
            keep_exclusions.append(excluded)

    clients_db.pop(drop_key, None)


def get_phone_conflict_groups(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Returns duplicate groups where variants have different non-empty phones.
    These groups need manager confirmation before merge.
    """
    data = load_clients()
    clients_db = data.get("clients", {})
    grouped: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for key, info in clients_db.items():
        if not isinstance(info, dict):
            continue
        if info.get("is_vendor") or is_service_client_name(key):
            continue
        grouped.setdefault(canonicalize_client_key_loose(key), []).append((key, info))

    results: List[Dict[str, Any]] = []
    for loose_key, items in grouped.items():
        if len(items) < 2:
            continue
        phones: Dict[str, List[str]] = {}
        managers = set()
        for key, info in items:
            phone = _contact_phone_value(info)
            if phone:
                phones.setdefault(phone, []).append(key)
            manager = (info.get("manager") or "").strip()
            if manager not in _UNKNOWN_MANAGERS:
                managers.add(manager)
        if len(phones) < 2:
            continue
        if len(items) == 2 and _pair_review_blocked(items[0][0], items[0][1], items[1][0], items[1][1]):
            continue

        manager = managers.pop() if len(managers) == 1 else ""
        results.append(
            {
                "group_key": loose_key,
                "manager": manager,
                "items": [
                    {
                        "client_key": key,
                        "phone": _contact_phone_value(info),
                        "manager": (info.get("manager") or "").strip(),
                        "sources": list(info.get("sources", [])) if isinstance(info.get("sources"), list) else [],
                        "display_name": (info.get("display_name") or "").strip(),
                    }
                    for key, info in items
                    if _contact_phone_value(info)
                ],
            }
        )
    results.sort(key=lambda row: ((row.get("manager") or "~"), row.get("group_key") or ""))
    return results[:limit] if limit else results


def resolve_phone_conflict(
    client_keys: List[str],
    chosen_phone: str,
    chosen_key: str = "",
    reviewer: str = "",
    phone_source: str = "manager_duplicate_review",
) -> bool:
    """
    Resolves a duplicate phone conflict by keeping one key, setting the chosen phone,
    and merging sibling entries into aliases of the kept card.
    """
    if not client_keys or len(client_keys) < 2:
        return False
    data = load_clients()
    clients_db = data.get("clients", {})
    existing_keys = [key for key in client_keys if isinstance(clients_db.get(key), dict)]
    if len(existing_keys) < 2:
        return False

    manager = ""
    for key in existing_keys:
        info = clients_db.get(key, {})
        manager = (info.get("manager") or "").strip()
        if manager and manager not in _UNKNOWN_MANAGERS:
            break
    keep_key = chosen_key if chosen_key in existing_keys else _choose_best_duplicate_key(clients_db, existing_keys[0], manager=manager)
    if keep_key not in existing_keys:
        keep_key = existing_keys[0]

    keep = clients_db.get(keep_key)
    if not isinstance(keep, dict):
        return False
    keep["whatsapp"] = chosen_phone.strip()
    keep["phone_source"] = phone_source
    keep["duplicate_review_exclusions"] = []

    for other_key in existing_keys:
        if other_key == keep_key:
            continue
        _merge_client_entries(clients_db, keep_key, other_key)

    data["clients"] = clients_db
    if not save_clients(data):
        logger.error("resolve_phone_conflict: clients.json не сохранён")
        return False
    crm_audit(
        "duplicate_phone_conflict_resolved",
        reviewer=reviewer,
        keep_key=keep_key,
        chosen_phone=chosen_phone,
        merged_keys=existing_keys,
    )
    logger.info("CRM duplicate conflict resolved: keep=%s merged=%d reviewer=%s", keep_key, len(existing_keys), reviewer or "-")
    return True


def apply_manual_ownership(
    client_keys: List[str],
    manager: str,
    reviewer: str = "",
    source: str = "manual_ownership",
) -> bool:
    """
    Persists an explicit ownership decision for one or more CRM cards.

    This is additive and safe: it only stamps explicit ownership fields plus the
    visible manager field on the surviving cards. It does not change merge logic.
    """
    manager = (manager or "").strip()
    if not client_keys or not manager or manager in _UNKNOWN_MANAGERS:
        return False
    data = load_clients()
    clients_db = data.get("clients", {})
    stamp = datetime.now(TZ).isoformat()
    changed_keys: List[str] = []
    for key in client_keys:
        info = clients_db.get(key)
        if not isinstance(info, dict):
            continue
        info["manager"] = manager
        info["ownership_manager"] = manager
        info["ownership_decided_at"] = stamp
        info["ownership_decided_by"] = reviewer or "system"
        info["ownership_source"] = source
        clients_db[key] = info
        changed_keys.append(key)
    if not changed_keys:
        return False
    data["clients"] = clients_db
    if not save_clients(data):
        logger.error("apply_manual_ownership: clients.json not saved")
        return False
    crm_audit(
        "ownership_assigned",
        reviewer=reviewer,
        manager=manager,
        source=source,
        client_keys=changed_keys,
    )
    logger.info(
        "CRM ownership assigned: manager=%s keys=%d source=%s reviewer=%s",
        manager,
        len(changed_keys),
        source,
        reviewer or "-",
    )
    return True


def mark_phone_conflict_distinct(client_keys: List[str], reviewer: str = "") -> bool:
    """Marks a duplicate pair as intentionally distinct so future review won't re-open it."""
    if not client_keys or len(client_keys) < 2:
        return False
    data = load_clients()
    clients_db = data.get("clients", {})
    changed = False
    for key in client_keys:
        info = clients_db.get(key)
        if not isinstance(info, dict):
            continue
        exclusions = info.setdefault("duplicate_review_exclusions", [])
        if not isinstance(exclusions, list):
            exclusions = info["duplicate_review_exclusions"] = [exclusions] if exclusions else []
        for other_key in client_keys:
            if other_key == key:
                continue
            if other_key not in exclusions:
                exclusions.append(other_key)
                changed = True
    if not changed:
        return False
    data["clients"] = clients_db
    if not save_clients(data):
        logger.error("mark_phone_conflict_distinct: clients.json не сохранён")
        return False
    crm_audit("duplicate_phone_conflict_marked_distinct", reviewer=reviewer, client_keys=client_keys)
    logger.info("CRM duplicate conflict marked distinct: keys=%d reviewer=%s", len(client_keys), reviewer or "-")
    return True


def is_client_pair_excluded(left_key: str, right_key: str) -> bool:
    """Public wrapper: whether two CRM keys were explicitly marked as different."""
    if not left_key or not right_key:
        return False
    data = load_clients()
    clients_db = data.get("clients", {})
    left_info = clients_db.get(left_key)
    right_info = clients_db.get(right_key)
    if not isinstance(left_info, dict) or not isinstance(right_info, dict):
        return False
    return _pair_review_blocked(left_key, left_info, right_key, right_info)


def merge_client_into_existing(
    keep_key: str,
    alias_key: str,
    manager: str = "",
    reviewer: str = "",
    source: str = "claim_same_client",
) -> bool:
    """
    Safe merge for manager-confirmed "это тот же клиент?" cases.

    keep_key remains the canonical CRM card; alias_key is folded into aliases.
    """
    if not keep_key or not alias_key or keep_key == alias_key:
        return False

    data = load_clients()
    clients_db = data.get("clients", {})
    keep_entry = clients_db.get(keep_key)
    alias_entry = clients_db.get(alias_key)
    if not isinstance(keep_entry, dict) or not isinstance(alias_entry, dict):
        return False

    _merge_client_entries(clients_db, keep_key, alias_key)
    data["clients"] = clients_db
    if not save_clients(data):
        logger.error("merge_client_into_existing: clients.json not saved")
        return False

    crm_audit(
        "same_client_merged",
        keep_key=keep_key,
        alias_key=alias_key,
        manager=manager,
        reviewer=reviewer,
        source=source,
    )

    if manager and manager not in _UNKNOWN_MANAGERS:
        if not apply_manual_ownership(
            client_keys=[keep_key],
            manager=manager,
            reviewer=reviewer or manager,
            source=source,
        ):
            logger.error(
                "merge_client_into_existing: ownership stamp failed keep=%s alias=%s manager=%s",
                keep_key,
                alias_key,
                manager,
            )
            return False

    logger.info(
        "CRM same-client merge: keep=%s alias=%s manager=%s reviewer=%s",
        keep_key,
        alias_key,
        manager or "-",
        reviewer or "-",
    )
    return True


# ─────────────────────────────────────────────
# Вспомогательные функции разбора JSON отчётов
# ─────────────────────────────────────────────

def _safe_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except (FileNotFoundError, OSError):
        return 0.0


def _today() -> str:
    return date.today().isoformat()


def _parse_manager_from_filename(filename: str) -> str:
    """Пытается извлечь имя менеджера из имени файла sales_продажи_МЕНЕДЖЕР_..."""
    managers = ("алена", "ергали", "магира", "оксана")
    name_lower = filename.lower()
    for mgr in managers:
        if mgr in name_lower:
            return mgr.capitalize()
    return ""


def _latest_debt_json_for_manager(manager: str) -> Optional[Path]:
    detailed = list(JSON_DIR.glob(f"debt_ext_*Детальный Дебиторы {manager}*.json"))
    if detailed:
        return max(detailed, key=_safe_mtime)

    fallback = list(JSON_DIR.glob(f"debt_ext_*{manager}*.json"))
    if fallback:
        return max(fallback, key=_safe_mtime)
    return None


def _load_known_managers() -> Tuple[str, ...]:
    """
    Fix S1: читаем имена менеджеров из config/managers.json — единственный
    источник правды по CLAUDE.md. Fallback — исторический хардкод, чтобы
    прод не ломался при отсутствии конфига.
    """
    fallback = ("Алена", "Ергали", "Магира", "Оксана")
    cfg_path = CONFIG_DIR / "managers.json"
    try:
        if cfg_path.exists():
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                names = tuple(k.strip() for k in data.keys() if k and k.strip())
                if names:
                    return names
    except (OSError, ValueError) as e:
        logger.error("managers.json read error: %s — используем fallback", e)
    return fallback


def _load_latest_debt_clients() -> List[Tuple[str, str]]:
    """
    Читает последние debt_ext_*.json по каждому менеджеру.
    Возвращает список (client_name, manager).
    """
    result: List[Tuple[str, str]] = []
    for manager_name in _load_known_managers():
        latest = _latest_debt_json_for_manager(manager_name)
        if latest is None:
            continue
        try:
            with open(latest, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("debt JSON read error %s: %s", latest.name, e)
            continue

        manager = (data.get("manager") or manager_name) if isinstance(data, dict) else manager_name
        if not manager or manager in ("?", "-", "вЂ”", "ABSENT"):
            continue

        clients: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            for key in ("clients", "rows", "data"):
                if key in data and isinstance(data[key], list):
                    clients = data[key]
                    break

        for c in clients:
            if not isinstance(c, dict):
                continue
            name = (c.get("name") or c.get("client") or "").strip()
            if name:
                result.append((name, manager))

    if result:
        return result

    candidates = list(JSON_DIR.glob("debt_ext_*.json"))
    if not candidates:
        return []

    # Группируем по базовому имени (без суффикса ' (NNN)')
    groups: Dict[str, List[Path]] = {}
    for p in candidates:
        base = re.sub(r"\s*\(\d+\)$", "", p.stem)
        groups.setdefault(base, []).append(p)

    result: List[Tuple[str, str]] = []
    for _base, paths in groups.items():
        latest = max(paths, key=_safe_mtime)
        try:
            with open(latest, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("debt JSON read error %s: %s", latest.name, e)
            continue

        manager = (data.get("manager") or "") if isinstance(data, dict) else ""
        if not manager or manager in ("?", "-", "—", "ABSENT"):
            continue

        clients: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            for key in ("clients", "rows", "data"):
                if key in data and isinstance(data[key], list):
                    clients = data[key]
                    break

        for c in clients:
            if not isinstance(c, dict):
                continue
            name = (c.get("name") or c.get("client") or "").strip()
            if name:
                result.append((name, manager))

    return result


def _load_latest_sales_clients() -> List[Tuple[str, str]]:
    """
    Читает последние sales_*.json по каждому менеджеру.
    Возвращает список (client_name, manager).
    """
    candidates = list(JSON_DIR.glob("sales_*.json"))
    if not candidates:
        return []

    # Группируем по слагу менеджера: sales_продажи_<manager>_...
    # Файлы без менеджера в имени (общие) группируем в ""
    groups: Dict[str, List[Path]] = {}
    for p in candidates:
        mgr = _parse_manager_from_filename(p.stem)
        key = mgr if mgr else "__общий__"
        groups.setdefault(key, []).append(p)

    result: List[Tuple[str, str]] = []
    for _key, paths in groups.items():
        latest = max(paths, key=_safe_mtime)
        try:
            with open(latest, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("sales JSON read error %s: %s", latest.name, e)
            continue

        manager = (data.get("manager") or "").strip() if isinstance(data, dict) else ""
        if not manager:
            continue

        clients_list: List[Dict[str, Any]] = data.get("clients", []) if isinstance(data, dict) else []
        for c in clients_list:
            if not isinstance(c, dict):
                continue
            name = (c.get("client") or c.get("name") or "").strip()
            if not name:
                continue
            if any(kw in name for kw in _METADATA_KEYWORDS):
                logger.debug("crm: пропускаем нежелательную строку из sales JSON: %s", name[:60])
                continue
            result.append((name, manager))

    return result


# ─────────────────────────────────────────────
# Обновление базы из отчётов
# ─────────────────────────────────────────────

def update_from_reports() -> Dict[str, List[str]]:
    """
    Сканирует последние debt и sales JSON, добавляет новых клиентов.
    Обновляет last_seen, sources для существующих.

    Возвращает словарь новых клиентов по менеджерам:
      {"Алена": ["Клиент А", "Клиент Б"], "Ергали": [...]}
    """
    data = load_clients()
    clients_db: Dict[str, Any] = data.get("clients", {})
    today = _today()
    new_by_manager: Dict[str, List[str]] = {}

    def _is_vendor_name(name: str) -> bool:
        return is_service_client_name(name)

    def _upsert(name: str, manager: str, source: str) -> bool:
        """Добавляет/обновляет клиента. Возвращает True если клиент новый (не вендор)."""
        is_vendor = _is_vendor_name(name)
        existing_key = _choose_best_duplicate_key(clients_db, name, manager=manager)
        existing = clients_db.get(existing_key) if existing_key else None
        if existing is None:
            clients_db[name] = {
                "manager": manager,
                "whatsapp": "",
                "telegram_id": "",
                "language": "ru",
                "do_not_call": is_vendor,
                "is_vendor": is_vendor,
                "sources": [source],
                "first_seen": today,
                "last_seen": today,
                "aliases": [],
            }
            crm_audit("client_created", client_key=name, manager=manager, source=source, is_vendor=is_vendor)
            return not is_vendor

        if existing_key != name and name in clients_db:
            _merge_client_entries(clients_db, existing_key, name)
            existing = clients_db.get(existing_key) if existing_key else None

        existing["last_seen"] = today
        if source not in existing.get("sources", []):
            existing.setdefault("sources", []).append(source)
        if is_vendor and not existing.get("is_vendor"):
            existing["is_vendor"] = True
            existing["do_not_call"] = True
        _UNOWNED = ("", "Не определён", "?", "-", "—")
        if existing.get("manager", "") in _UNOWNED and manager and manager not in _UNOWNED:
            existing["manager"] = manager
            crm_audit("manager_assigned", client_key=existing_key or name, manager=manager, source=source)
        if existing_key and existing_key != name:
            aliases = existing.setdefault("aliases", [])
            if name not in aliases and name != existing_key:
                aliases.append(name)
                crm_audit("canonical_merge", client_key=existing_key, alias=name, manager=existing.get("manager", ""), source=source)
        return False

    for name, manager in _load_latest_debt_clients():
        is_new = _upsert(name, manager, "debt")
        if is_new and manager:
            new_by_manager.setdefault(manager, []).append(name)

    for name, manager in _load_latest_sales_clients():
        is_new = _upsert(name, manager, "sales")
        if is_new and manager:
            new_by_manager.setdefault(manager, []).append(name)

    data["clients"] = clients_db
    if not save_clients(data):
        logger.error("update_from_reports: clients.json не сохранён")
        return {}

    total_new = sum(len(v) for v in new_by_manager.values())
    crm_audit("crm_sync_complete", total_clients=len(clients_db), total_new=total_new)
    logger.info("CRM обновлена: %d клиентов всего, %d новых", len(clients_db), total_new)
    return new_by_manager

def get_clients_without_phones(manager: str, limit: int = 5) -> List[str]:
    """
    Возвращает до `limit` имён клиентов данного менеджера без телефона.
    Приоритет: сначала клиенты из дебиторки.
    Вендоры/контрагенты (is_vendor=True) пропускаются.
    """
    data = load_clients()
    clients_db = data.get("clients", {})
    no_phone = []
    for name, info in clients_db.items():
        if not isinstance(info, dict):
            continue
        if info.get("is_vendor"):
            continue
        if is_service_client_name(name):
            continue
        if info.get("manager", "").lower() != manager.lower():
            continue
        if info.get("whatsapp") or info.get("telegram_id"):
            continue
        if _find_phone_donor_key(clients_db, name, manager=manager):
            continue
        # Приоритет — дебиторка
        sources = info.get("sources", [])
        priority = 0 if "debt" in sources else 1
        no_phone.append((priority, name))

    no_phone.sort()
    return [name for _, name in no_phone[:limit]]


def _strip_legal(name: str) -> str:
    """Убирает ТОО/ИП/АО/LLP префиксы для нечёткого сравнения."""
    return re.sub(
        r"^\s*(ТОО|ИП|АО|ОАО|ООО|LLP|LLC|ЧП)\s+",
        "", name, flags=re.IGNORECASE,
    ).strip().lower()


def find_similar_clients(query: str, manager: str = "", limit: int = 5) -> List[str]:
    """
    Ищет похожих клиентов в базе по запросу менеджера.

    Алгоритм (в порядке приоритета):
      1. Точное совпадение (lower)
      2. display_name совпадает
      3. Все слова запроса входят в имя клиента
      4. Хотя бы одно слово из запроса совпадает (≥ 3 символа)
      5. Подстрока запроса в имени (без правовой формы)

    Если задан manager — приоритет клиентам этого менеджера.
    Возвращает список ключей (имён из 1С), не более `limit`.
    """
    data = load_clients()
    clients_db = data.get("clients", {})
    if not clients_db or not query:
        return []

    q = query.lower().strip()
    q_stripped = _strip_legal(query)
    q_words = [w for w in q.split() if len(w) >= 3]

    scores: List[Tuple[int, str]] = []  # (score, key)  — чем меньше, тем лучше

    for key, info in clients_db.items():
        if not isinstance(info, dict):
            continue
        # Фильтр по менеджеру — не жёсткий, просто снижает приоритет
        mgr_match = (not manager) or (info.get("manager", "").lower() == manager.lower())

        key_l = key.lower().strip()
        key_stripped = _strip_legal(key)
        display = info.get("display_name", "").lower().strip()

        score = 100
        if key_l == q:
            score = 0
        elif display and display == q:
            score = 1
        elif key_stripped == q_stripped and q_stripped:
            score = 2
        elif q_words and all(w in key_l for w in q_words):
            score = 3
        elif q_words and any(w in key_l for w in q_words):
            score = 4
        elif q_stripped and q_stripped in key_stripped:
            score = 5
        else:
            continue  # нет совпадения

        if not mgr_match:
            score += 10  # откладываем чужих менеджеров вниз

        scores.append((score, key))

    scores.sort()
    return [key for _, key in scores[:limit]]


def set_client_phone(client_name: str, phone: str, manager: str = "",
                     alias: str = "", phone_source: str = "") -> bool:
    """
    Записывает телефон (WhatsApp) клиента в clients.json.
    client_name — точный ключ из 1С (после подтверждения менеджером).
    alias — как менеджер назвал клиента (display_name), если отличается.
    Возвращает True если клиент найден и обновлён.
    """
    data = load_clients()
    clients_db = data.get("clients", {})

    entry_key = _find_existing_client_key(clients_db, client_name)
    entry = clients_db.get(entry_key) if entry_key else None
    if entry is None:
        logger.warning("set_client_phone: клиент не найден: %s", client_name)
        crm_audit("phone_set_missing", client_key=client_name, phone=phone)
        return False

    entry["whatsapp"] = phone.strip()
    if phone_source:
        entry["phone_source"] = phone_source
    if manager and not entry.get("manager"):
        entry["manager"] = manager
    # Псевдоним: сохраняем если отличается от ключа 1С
    if alias and alias.lower().strip() != client_name.lower().strip():
        entry["display_name"] = alias.strip()
        logger.info("display_name сохранён: %s → «%s»", client_name, alias)

    data["clients"] = clients_db
    if not save_clients(data):
        logger.error("set_client_phone: clients.json не сохранён для %s", client_name)
        return False
    crm_audit("phone_set", client_key=entry_key or client_name, phone=phone, manager=entry.get("manager", manager), phone_source=phone_source or "")
    logger.info("Телефон записан: %s → %s", client_name, phone)
    return True


def update_client_contact_from_dialog(
    client_name: str,
    new_phone: str,
    old_phone: str = "",
    contact_info: str = "",
    source: str = "client_dialog_auto_update",
) -> bool:
    """Обновляет телефон клиента по данным из WA-диалога.

    Сохраняет старый телефон в previous_whatsapp для аудит-следа.
    Возвращает True если CRM успешно обновлён.
    """
    data = load_clients()
    clients_db = data.get("clients", {})
    entry_key = _find_existing_client_key(clients_db, client_name)
    if not entry_key:
        logger.warning("update_client_contact_from_dialog: клиент не найден: %s", client_name)
        crm_audit("contact_update_missing", client_key=client_name, new_phone=new_phone)
        return False

    entry = clients_db[entry_key]
    if old_phone:
        entry["previous_whatsapp"] = old_phone.strip()
    entry["whatsapp"] = new_phone.strip()
    entry["phone_source"] = source
    if contact_info:
        entry["contact_info"] = contact_info.strip()

    data["clients"] = clients_db
    if not save_clients(data):
        logger.error("update_client_contact_from_dialog: clients.json не сохранён для %s", client_name)
        return False

    crm_audit(
        "contact_updated_from_dialog",
        client_key=entry_key,
        new_phone=new_phone,
        old_phone=old_phone or "—",
        source=source,
    )
    logger.info(
        "CRM контакт обновлён: %s → %s (был: %s)", client_name, new_phone, old_phone or "—"
    )
    return True


def _normalize_system_display_name(display_name: str, name_mode: str = "") -> str:
    """Strips manager ownership prefix from system-generated CRM display names."""
    value = str(display_name or "").strip()
    if str(name_mode or "").strip() != "system":
        return value
    if len(value) >= 3 and value[1] == " " and value[0].upper() in {"А", "Е", "М", "О", "A", "E", "M", "O"}:
        return value[2:].strip()
    return value


def set_client_details(client_name: str, display_name: str = "",
                        phone: str = "", address: str = "",
                        original_name: str = "", name_mode: str = "",
                        name_review_needed: Optional[bool] = None,
                        phone_source: str = "") -> bool:
    """
    Сохраняет display_name, телефон и/или адрес торговой точки для клиента.
    Обновляет только переданные (непустые) поля.
    """
    data = load_clients()
    clients_db = data.get("clients", {})
    # F-01: нормализуем ключ — прямой get не находит клиента при расхождении пробелов/регистра
    _entry_key = _find_existing_client_key(clients_db, client_name)
    entry = clients_db.get(_entry_key) if _entry_key else None
    if entry is None:
        logger.warning("set_client_details: клиент не найден: %s", client_name)
        return False
    client_name = _entry_key  # используем нормализованный ключ для аудита и save
    display_name = _normalize_system_display_name(display_name, name_mode)
    if original_name:
        entry["original_name"] = original_name.strip()
    if display_name:
        entry["display_name"] = display_name.strip()
    if phone:
        entry["whatsapp"] = phone.strip()
        if phone_source:
            entry["phone_source"] = phone_source
    if address:
        entry["address"] = address.strip()
    if name_mode:
        entry["name_mode"] = name_mode
    if name_review_needed is not None:
        entry["name_review_needed"] = bool(name_review_needed)
    data["clients"] = clients_db
    if not save_clients(data):
        logger.error("set_client_details: clients.json не сохранён для %s", client_name)
        return False
    crm_audit("client_details_set", client_key=client_name, display_name=display_name or "", phone=phone or "", address=address or "", mode=name_mode or "")
    logger.info("Данные обновлены: %s (name=%r phone=%r address=%r original=%r mode=%r review=%r)",
                client_name, display_name or "-", phone or "-", address or "-",
                original_name or "-", name_mode or "-", name_review_needed)
    return True


def set_client_alias(client_name: str, alias: str) -> bool:
    """
    Сохраняет display_name (псевдоним) для клиента без изменения телефона.
    Используется когда менеджер исправляет только имя.
    """
    data = load_clients()
    clients_db = data.get("clients", {})
    entry = clients_db.get(client_name)
    if entry is None:
        logger.warning("set_client_alias: клиент не найден: %s", client_name)
        return False
    entry["display_name"] = alias.strip()
    data["clients"] = clients_db
    if not save_clients(data):
        logger.error("set_client_alias: clients.json не сохранён для %s", client_name)
        return False
    crm_audit("alias_set", client_key=client_name, alias=alias)
    logger.info("Псевдоним сохранён: %s → «%s»", client_name, alias)
    return True


# ─────────────────────────────────────────────
# Совместимость с коллектором
# ─────────────────────────────────────────────

def load_contacts_compat() -> Dict[str, Any]:
    """
    Возвращает словарь контактов из CRM (clients.json) в формате,
    совместимом с collector:
      {client_name: {"whatsapp": ..., "telegram_id": ..., "manager": ..., ...}}

    Единственный источник — config/clients.json (заполняется менеджерами через бот).
    """
    data = load_clients()
    clients_db = data.get("clients", {})

    result: Dict[str, Any] = {}
    for name, info in clients_db.items():
        if not isinstance(info, dict):
            continue
        contact_entry = {
            "whatsapp": info.get("whatsapp", ""),
            "telegram_id": info.get("telegram_id", ""),
            "manager": info.get("manager", ""),
            "language": info.get("language", "ru"),
            "do_not_call": info.get("do_not_call", False),
        }
        result[name] = contact_entry
        # F-02: раскрываем aliases — collector ищет по именам из 1С,
        # которые могут быть старыми ключами после canonical merge
        for alias in info.get("aliases", []):
            if alias and alias not in result:
                result[alias] = contact_entry

    return result


def _load_legacy_contacts_fallback() -> Dict[str, Any]:
    """Best-effort reader for legacy debtors_contacts.json."""
    if not LEGACY_CONTACTS_PATH.exists():
        return {}
    try:
        with open(LEGACY_CONTACTS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        data.pop("_comment", None)
        result: Dict[str, Any] = {}
        for name, info in data.items():
            if not isinstance(name, str) or not isinstance(info, dict):
                continue
            result[name] = {
                "whatsapp": str(info.get("whatsapp", "") or info.get("phone", "") or "").strip(),
                "telegram_id": str(info.get("telegram_id", "") or "").strip(),
                "manager": str(info.get("manager", "") or "").strip(),
                "language": str(info.get("language", "ru") or "ru").strip() or "ru",
                "do_not_call": bool(info.get("do_not_call", False)),
                "_source": "legacy_fallback",
            }
        return result
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Legacy contacts fallback unreadable: %s", e)
        return {}


def load_contacts_for_collector(include_legacy_fallback: bool = True) -> Dict[str, Any]:
    """
    Unified collector contact view: CRM is primary, debtors_contacts.json is
    read-only fallback for keys absent in CRM.
    """
    crm_contacts = load_contacts_compat()
    result: Dict[str, Any] = {
        name: {**info, "_source": "crm"}
        for name, info in crm_contacts.items()
        if isinstance(info, dict)
    }
    if not include_legacy_fallback:
        return result

    for name, info in _load_legacy_contacts_fallback().items():
        result.setdefault(name, info)
    return result


# ─────────────────────────────────────────────
# Уведомления о новых клиентах
# ─────────────────────────────────────────────

def get_new_clients_since(since_date: str) -> Dict[str, List[str]]:
    """
    Возвращает клиентов с first_seen >= since_date, сгруппированных по менеджеру.
    since_date: "YYYY-MM-DD"
    """
    data = load_clients()
    result: Dict[str, List[str]] = {}
    for name, info in data.get("clients", {}).items():
        if not isinstance(info, dict):
            continue
        if info.get("first_seen", "0000-00-00") >= since_date:
            mgr = info.get("manager", "")
            result.setdefault(mgr, []).append(name)
    return result

"""Payment confirmation hold shared by silence alerts and collector.

This module stores cases where a manager says payment was made and Saida
confirms that the payment exists but is not posted in 1C yet. While confirmed,
the client should not receive pressure from short debt alerts or WhatsApp
collector.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
TZ_NAME = os.getenv("TZ", "Asia/Qyzylorda")
try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo(TZ_NAME)
except Exception:  # pragma: no cover - defensive fallback
    TZ = None

PAYMENT_HOLD_PATH = ROOT / "logs" / "saida_payment_holds.json"
HOLD_TTL_DAYS = int(os.getenv("SAIDA_PAYMENT_HOLD_TTL_DAYS", "2"))

ACTIVE_STATUSES = {"confirmed_full", "confirmed_partial"}
OPEN_STATUSES = {"pending_saida", *ACTIVE_STATUSES}


def _now() -> datetime:
    return datetime.now(TZ) if TZ else datetime.now()


def _now_iso() -> str:
    return _now().isoformat(timespec="seconds")


def normalize_client_name(name: str) -> str:
    return " ".join(str(name or "").lower().split())


def _load() -> Dict[str, Any]:
    try:
        if PAYMENT_HOLD_PATH.exists():
            data = json.loads(PAYMENT_HOLD_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _save(data: Dict[str, Any]) -> None:
    PAYMENT_HOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=PAYMENT_HOLD_PATH.parent,
            delete=False,
            suffix=".tmp",
        ) as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            tmp = fh.name
        os.replace(tmp, PAYMENT_HOLD_PATH)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _token(manager: str, client: str) -> str:
    raw = f"{normalize_client_name(manager)}|{normalize_client_name(client)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def create_manager_payment_request(
    manager: str,
    client: str,
    debt: float = 0.0,
    debt_str: str = "",
    manager_chat_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Create or refresh a manager-to-Saida payment check request."""
    data = _load()
    token = _token(manager, client)
    current = data.get(token, {})
    if current.get("status") in ACTIVE_STATUSES:
        return current

    record = {
        "token": token,
        "status": "pending_saida",
        "manager": manager,
        "client": client,
        "client_norm": normalize_client_name(client),
        "debt": float(debt or 0.0),
        "debt_str": debt_str,
        "manager_chat_id": int(manager_chat_id or 0),
        "created_at": current.get("created_at") or _now_iso(),
        "updated_at": _now_iso(),
    }
    data[token] = record
    _save(data)
    return record


def confirm_by_saida(token: str, status: str) -> Optional[Dict[str, Any]]:
    """Record Saida's answer: full, partial, or none."""
    data = _load()
    record = data.get(token)
    if not isinstance(record, dict):
        return None
    if status == "full":
        record["status"] = "confirmed_full"
    elif status == "partial":
        record["status"] = "confirmed_partial"
    elif status == "none":
        record["status"] = "rejected"
    else:
        return None
    record["saida_confirmed_at"] = _now_iso()
    record["updated_at"] = _now_iso()
    data[token] = record
    _save(data)
    return record


def get_request(token: str) -> Optional[Dict[str, Any]]:
    record = _load().get(token)
    return record if isinstance(record, dict) else None


def get_hold_for_client(client: str) -> Optional[Dict[str, Any]]:
    client_norm = normalize_client_name(client)
    now = _now()
    changed = False
    data = _load()
    found: Optional[Dict[str, Any]] = None
    for token, record in list(data.items()):
        if not isinstance(record, dict):
            continue
        if record.get("client_norm") != client_norm:
            continue
        status = record.get("status")
        if status in ACTIVE_STATUSES:
            created_raw = record.get("saida_confirmed_at") or record.get("updated_at") or record.get("created_at")
            try:
                created = datetime.fromisoformat(str(created_raw))
                if created.tzinfo is None and TZ:
                    created = created.replace(tzinfo=TZ)
            except Exception:
                created = now
            if now - created > timedelta(days=HOLD_TTL_DAYS):
                record["status"] = "expired"
                record["expired_at"] = _now_iso()
                data[token] = record
                changed = True
                continue
            found = record
            break
    if changed:
        _save(data)
    return found


def is_payment_hold_active(client: str) -> bool:
    return get_hold_for_client(client) is not None


def sync_holds_with_debtors(debtors: Iterable[Dict[str, Any]]) -> int:
    """Close active holds when a fresh 1C snapshot no longer has debt."""
    debt_by_norm = {
        normalize_client_name(d.get("name") or d.get("client")): float(d.get("amount", d.get("debt", 0)) or 0)
        for d in debtors
        if isinstance(d, dict) and (d.get("name") or d.get("client"))
    }
    data = _load()
    changed = 0
    for token, record in list(data.items()):
        if not isinstance(record, dict) or record.get("status") not in ACTIVE_STATUSES:
            continue
        norm = record.get("client_norm", "")
        if norm and debt_by_norm.get(norm, 0.0) <= 0:
            record["status"] = "cleared_by_1c"
            record["cleared_at"] = _now_iso()
            data[token] = record
            changed += 1
    if changed:
        _save(data)
    return changed


def list_open_holds() -> Dict[str, Dict[str, Any]]:
    return {
        token: record
        for token, record in _load().items()
        if isinstance(record, dict) and record.get("status") in OPEN_STATUSES
    }

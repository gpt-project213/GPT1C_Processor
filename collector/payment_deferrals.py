#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/payment_deferrals.py
Договорные сроки отсрочки платежа.

v1.0.1 (2026-05-13)

Клиенты из config/payment_deferrals.json имеют N дней отсрочки по договору.
Просрочка считается только после истечения этого срока.

effective_days = max(0, actual_days - deferral_days)

Если клиент не в списке — отсрочки нет (дефолт: предоплата / 0 дней).

v1.0.1: добавлен read-only мониторинг финдисциплины по отсрочникам
  в logs/deferral_violations.json. Считаются закрытые циклы "в срок" и
  циклы с нарушением, а также средняя/максимальная задержка по просрочке.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

_ROOT          = Path(__file__).resolve().parent.parent
_DEFERRALS_PATH = _ROOT / "config" / "payment_deferrals.json"
_VIOLATIONS_PATH = _ROOT / "logs" / "deferral_violations.json"
_TZ_NAME = os.getenv("TZ", "Asia/Almaty")
try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo(_TZ_NAME)
except Exception:  # pragma: no cover - defensive fallback
    TZ = None

_cache: dict = {}


def _load() -> dict:
    global _cache
    if _cache:
        return _cache
    try:
        with open(_DEFERRALS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        _cache = data.get("deferrals", {})
        logger.info("payment_deferrals: загружено %d записей", len(_cache))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("payment_deferrals: ошибка загрузки: %s", e)
        _cache = {}
    return _cache


def get_deferral_days(client_name: str) -> int:
    """Возвращает количество дней отсрочки для клиента (0 если нет договора)."""
    return _load().get(client_name, {}).get("days", 0)


def effective_overdue_days(client_name: str, actual_days: int) -> int:
    """Эффективные дни просрочки с учётом отсрочки по договору."""
    deferral = get_deferral_days(client_name)
    return max(0, actual_days - deferral)


def has_deferral(client_name: str) -> bool:
    return client_name in _load()


def all_deferred_clients() -> dict:
    """Все клиенты с договорными отсрочками."""
    return dict(_load())


# Шкала давления для клиентов с договорной отсрочкой.
# Считается от effective_days (фактические дни минус отсрочка).
# Пороги тighter: effective=2 уже первое напоминание, effective=10 — критично.
_DEFERRAL_LEVEL_THRESHOLDS = [
    (10, 5),   # effective 10+ → критично
    (7,  4),   # effective 7–9 → серьёзно
    (5,  3),   # effective 5–6 → настойчиво
    (3,  2),   # effective 3–4 → умеренное давление
    (2,  1),   # effective 2   → первое напоминание
    (0,  0),
]


def deferral_level(effective_days: int) -> int:
    """Уровень давления для клиента с отсрочкой (по effective_days)."""
    for threshold, level in _DEFERRAL_LEVEL_THRESHOLDS:
        if effective_days >= threshold:
            return level
    return 0


def reload() -> None:
    """Сбросить кэш (вызывается после изменения конфига)."""
    global _cache
    _cache = {}


def get_deferral_manager(client_name: str) -> str:
    """Менеджер из конфига отсрочек, если указан."""
    return str(_load().get(client_name, {}).get("manager", "") or "")


def _now() -> datetime:
    return datetime.now(TZ) if TZ else datetime.now()


def _now_iso() -> str:
    return _now().isoformat(timespec="seconds")


def _norm(name: str) -> str:
    return " ".join(str(name or "").lower().split())


def _load_violation_state() -> Dict[str, Any]:
    try:
        if _VIOLATIONS_PATH.exists():
            data = json.loads(_VIOLATIONS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                clients = data.get("clients")
                if isinstance(clients, dict):
                    return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"version": 1, "updated_at": "", "clients": {}}


def _save_violation_state(data: Dict[str, Any]) -> None:
    _VIOLATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=_VIOLATIONS_PATH.parent,
            delete=False,
            suffix=".tmp",
        ) as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            tmp = fh.name
        os.replace(tmp, _VIOLATIONS_PATH)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def sync_deferral_discipline(debtors: Iterable[Dict[str, Any]]) -> int:
    """Синхронизирует состояние финдисциплины по клиентам с отсрочкой.

    Логика:
    - клиент с отсрочкой и долгом > 0 открывает/обновляет текущий цикл
    - effective_days > 0 помечает цикл как нарушение
    - если клиент исчез из текущего debt snapshot, цикл считается закрытым:
      либо "в срок", либо "с нарушением"
    """
    state = _load_violation_state()
    clients_state = state.setdefault("clients", {})
    now_iso = _now_iso()
    current_open: Dict[str, Dict[str, Any]] = {}

    for debtor in debtors:
        if not isinstance(debtor, dict):
            continue
        client_name = str(debtor.get("name") or debtor.get("client") or "").strip()
        if not client_name:
            continue
        if float(debtor.get("amount", 0) or 0) <= 0:
            continue
        deferral_days = get_deferral_days(client_name)
        if deferral_days <= 0:
            continue
        actual_days = int(debtor.get("days", 0) or 0)
        effective_days = effective_overdue_days(client_name, actual_days)
        manager = str(debtor.get("manager") or get_deferral_manager(client_name) or "—")
        norm = _norm(client_name)
        current_open[norm] = {
            "client_name": client_name,
            "manager": manager,
            "deferral_days": deferral_days,
            "actual_days": actual_days,
            "effective_days": effective_days,
            "debt": float(debtor.get("amount", 0) or 0),
        }

    changed = 0

    for norm, item in current_open.items():
        rec = clients_state.get(norm)
        if not isinstance(rec, dict):
            rec = {
                "client_name": item["client_name"],
                "manager": item["manager"],
                "deferral_days": item["deferral_days"],
                "cycles_closed_in_term": 0,
                "cycles_closed_with_violation": 0,
                "violation_count": 0,
                "total_violation_days": 0,
                "last_closed_status": "",
                "last_closed_at": "",
                "current_cycle": None,
            }
            clients_state[norm] = rec
            changed += 1

        rec["client_name"] = item["client_name"]
        rec["manager"] = item["manager"]
        rec["deferral_days"] = item["deferral_days"]
        cycle = rec.get("current_cycle")
        if not isinstance(cycle, dict):
            cycle = {
                "opened_at": now_iso,
                "last_seen_at": now_iso,
                "actual_days": item["actual_days"],
                "effective_days": item["effective_days"],
                "max_effective_days": item["effective_days"],
                "violated": item["effective_days"] > 0,
            }
            if item["effective_days"] > 0:
                rec["violation_count"] = int(rec.get("violation_count", 0) or 0) + 1
            rec["current_cycle"] = cycle
            changed += 1
        else:
            cycle["last_seen_at"] = now_iso
            cycle["actual_days"] = item["actual_days"]
            cycle["effective_days"] = item["effective_days"]
            cycle["max_effective_days"] = max(int(cycle.get("max_effective_days", 0) or 0), item["effective_days"])
            if item["effective_days"] > 0 and not cycle.get("violated"):
                cycle["violated"] = True
                rec["violation_count"] = int(rec.get("violation_count", 0) or 0) + 1
            rec["current_cycle"] = cycle
            changed += 1

    for norm, rec in list(clients_state.items()):
        if norm in current_open:
            continue
        cycle = rec.get("current_cycle")
        if not isinstance(cycle, dict):
            continue
        if cycle.get("violated"):
            rec["cycles_closed_with_violation"] = int(rec.get("cycles_closed_with_violation", 0) or 0) + 1
            rec["total_violation_days"] = int(rec.get("total_violation_days", 0) or 0) + int(cycle.get("max_effective_days", 0) or 0)
            rec["last_closed_status"] = "violated"
        else:
            rec["cycles_closed_in_term"] = int(rec.get("cycles_closed_in_term", 0) or 0) + 1
            rec["last_closed_status"] = "on_time"
        rec["last_closed_at"] = now_iso
        rec["current_cycle"] = None
        changed += 1

    if changed:
        state["updated_at"] = now_iso
        _save_violation_state(state)
    return changed


def get_deferral_discipline_stats() -> Dict[str, Any]:
    """Read-only сводка финдисциплины по клиентам с договорной отсрочкой."""
    state = _load_violation_state()
    clients_state = state.get("clients", {})
    totals = {
        "tracked_clients": 0,
        "open_now": 0,
        "open_violations": 0,
        "closed_in_term": 0,
        "closed_with_violation": 0,
        "violation_count": 0,
    }
    managers: Dict[str, Dict[str, Any]] = {}
    rows = []

    for rec in clients_state.values():
        if not isinstance(rec, dict):
            continue
        client_name = str(rec.get("client_name") or "—")
        manager = str(rec.get("manager") or "—")
        cycle = rec.get("current_cycle") if isinstance(rec.get("current_cycle"), dict) else None
        cycles_closed_in_term = int(rec.get("cycles_closed_in_term", 0) or 0)
        cycles_closed_with_violation = int(rec.get("cycles_closed_with_violation", 0) or 0)
        violation_count = int(rec.get("violation_count", 0) or 0)
        total_violation_days = int(rec.get("total_violation_days", 0) or 0)
        avg_delay = (
            total_violation_days / cycles_closed_with_violation
            if cycles_closed_with_violation > 0 else 0.0
        )
        current_eff = int(cycle.get("effective_days", 0) or 0) if cycle else 0
        max_eff = int(cycle.get("max_effective_days", 0) or 0) if cycle else 0
        if cycle:
            if cycle.get("violated") and max_eff >= 10:
                current_status = "злостный нарушитель"
            elif cycle.get("violated"):
                current_status = "нарушил"
            else:
                current_status = "в норме"
        else:
            current_status = {
                "violated": "закрыт после нарушения",
                "on_time": "закрыт в срок",
            }.get(str(rec.get("last_closed_status") or ""), "нет активного цикла")

        totals["tracked_clients"] += 1
        totals["closed_in_term"] += cycles_closed_in_term
        totals["closed_with_violation"] += cycles_closed_with_violation
        totals["violation_count"] += violation_count
        if cycle:
            totals["open_now"] += 1
            if cycle.get("violated"):
                totals["open_violations"] += 1

        row = {
            "client_name": client_name,
            "manager": manager,
            "deferral_days": int(rec.get("deferral_days", 0) or 0),
            "cycles_closed_in_term": cycles_closed_in_term,
            "cycles_closed_with_violation": cycles_closed_with_violation,
            "violation_count": violation_count,
            "avg_delay_days": avg_delay,
            "current_effective_days": current_eff,
            "max_effective_days": max_eff,
            "current_status": current_status,
            "is_open": bool(cycle),
        }
        rows.append(row)

        mgr = managers.setdefault(
            manager,
            {
                "manager": manager,
                "tracked_clients": 0,
                "open_now": 0,
                "open_violations": 0,
                "closed_in_term": 0,
                "closed_with_violation": 0,
                "violation_count": 0,
                "clients": [],
            },
        )
        mgr["tracked_clients"] += 1
        mgr["closed_in_term"] += cycles_closed_in_term
        mgr["closed_with_violation"] += cycles_closed_with_violation
        mgr["violation_count"] += violation_count
        if cycle:
            mgr["open_now"] += 1
            if cycle.get("violated"):
                mgr["open_violations"] += 1
        mgr["clients"].append(row)

    managers_list = sorted(
        managers.values(),
        key=lambda item: (-item["open_violations"], -item["violation_count"], item["manager"].lower()),
    )
    for mgr in managers_list:
        mgr["clients"].sort(
            key=lambda item: (
                0 if item["current_status"] == "злостный нарушитель" else
                1 if item["current_status"] == "нарушил" else
                2,
                -item["current_effective_days"],
                item["client_name"].lower(),
            )
        )

    return {
        "generated_at": _now_iso(),
        "totals": totals,
        "managers": managers_list,
        "clients": rows,
    }


def format_deferral_discipline_stats_text() -> str:
    """Форматирует read-only сводку финдисциплины по отсрочникам."""
    stats = get_deferral_discipline_stats()
    totals = stats["totals"]
    lines = [
        "⏱ <b>Финдисциплина по отсрочкам</b>",
        "",
        f"Клиентов на мониторинге: <b>{totals['tracked_clients']}</b>",
        f"Активных циклов сейчас: <b>{totals['open_now']}</b>",
        f"Текущих нарушений: <b>{totals['open_violations']}</b>",
        f"Закрыто в срок: <b>{totals['closed_in_term']}</b>",
        f"Закрыто с нарушением: <b>{totals['closed_with_violation']}</b>",
        "",
    ]
    managers = stats["managers"]
    if not managers:
        lines.append("Данных по отсрочкам пока нет.")
        return "\n".join(lines)

    lines.append("<b>По менеджерам:</b>")
    for mgr in managers:
        lines.append(
            f"• <b>{mgr['manager']}</b>: "
            f"открыто {mgr['open_now']}, "
            f"нарушений сейчас {mgr['open_violations']}, "
            f"закрыто в срок {mgr['closed_in_term']}, "
            f"закрыто с нарушением {mgr['closed_with_violation']}"
        )
        for client in mgr["clients"][:5]:
            avg_delay = f"{client['avg_delay_days']:.1f}".rstrip("0").rstrip(".")
            lines.append(
                f"  - {client['client_name']}: "
                f"{client['current_status']}, "
                f"в срок {client['cycles_closed_in_term']}, "
                f"нарушений {client['cycles_closed_with_violation']}, "
                f"ср. задержка {avg_delay} дн"
            )
        if len(mgr["clients"]) > 5:
            lines.append(f"  - … ещё {len(mgr['clients']) - 5}")

    risky_clients = [
        client
        for mgr in managers
        for client in mgr["clients"]
        if client["current_status"] in {"нарушил", "злостный нарушитель"}
    ]
    if risky_clients:
        lines.extend(["", "<b>Текущие нарушители:</b>"])
        for client in risky_clients[:10]:
            avg_delay = f"{client['avg_delay_days']:.1f}".rstrip("0").rstrip(".")
            lines.append(
                f"• {client['manager']}: <b>{client['client_name']}</b> — "
                f"{client['current_status']}, eff={client['current_effective_days']} дн, "
                f"средняя задержка {avg_delay} дн"
            )
        if len(risky_clients) > 10:
            lines.append(f"… ещё {len(risky_clients) - 10}")
    return "\n".join(lines)

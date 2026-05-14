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
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional

import portalocker

ROOT = Path(__file__).resolve().parents[1]
TZ_NAME = os.getenv("TZ", "Asia/Almaty")
try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo(TZ_NAME)
except Exception:  # pragma: no cover - defensive fallback
    TZ = None

PAYMENT_HOLD_PATH = ROOT / "logs" / "saida_payment_holds.json"
HOLD_TTL_DAYS = int(os.getenv("SAIDA_PAYMENT_HOLD_TTL_DAYS", "2"))
SAIDA_WARN_HOURS = int(os.getenv("SAIDA_WARN_HOURS", "1"))
SAIDA_BYPASS_HOURS = int(os.getenv("SAIDA_BYPASS_HOURS", "2"))

ACTIVE_STATUSES = {"confirmed_full", "confirmed_partial"}
OPEN_STATUSES = {"pending_saida", *ACTIVE_STATUSES}

_LOCK_FILE = PAYMENT_HOLD_PATH.with_suffix(".lock")


@contextmanager
def _hold_lock():
    """Exclusive cross-process lock for payment holds read-modify-write.

    LOCK_EX|LOCK_NB: non-blocking attempt with retry loop every 100 ms up to
    5 seconds — the only mode where portalocker's timeout parameter is effective
    on Windows (pure LOCK_EX blocking mode ignores timeout entirely).
    """
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with portalocker.Lock(
            str(_LOCK_FILE),
            timeout=5,
            check_interval=0.1,
            flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
        ):
            yield
    except portalocker.LockException:
        import logging as _lock_log
        _lock_log.getLogger(__name__).warning(
            "payment_hold: lock timeout (5 s) — proceeding without exclusive lock"
        )
        yield


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
    claimed_by_manager: bool = False,  # True когда менеджер заявил оплату в WA approval
) -> Dict[str, Any]:
    """Create or refresh a manager-to-Saida payment check request."""
    token = _token(manager, client)
    with _hold_lock():
        data = _load()
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
            "claimed_by_manager": claimed_by_manager,
            "created_at": current.get("created_at") or _now_iso(),
            "updated_at": _now_iso(),
        }
        data[token] = record
        _save(data)
        return record


_CONFIRMED_STATUSES = {"confirmed_full", "confirmed_partial", "rejected"}


def confirm_by_saida(token: str, status: str) -> Optional[Dict[str, Any]]:
    """Record Saida's answer: full, partial, or none.

    Returns None if already confirmed (idempotent — prevents double notifications
    when Saida both presses a button and writes a text reply).
    """
    with _hold_lock():
        data = _load()
        record = data.get(token)
        if not isinstance(record, dict):
            return None
        if record.get("status") in _CONFIRMED_STATUSES:
            return None  # already answered — caller must not send notifications again
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


def list_pending() -> list:
    """Возвращает все записи в статусе pending_saida (без TTL-чистки)."""
    data = _load()
    out = []
    for token, rec in data.items():
        if not isinstance(rec, dict):
            continue
        if rec.get("status") == "pending_saida":
            out.append(rec)
    return out


# ─── Парсер текстовых ответов Саиды ──────────────────────────────────────────
#
# Why: Саида часто отвечает текстом «Акжан полная», «Шапагат частично»,
# а не нажимает кнопки. Без парсера её ответы пропадают, а pending копится.
# Алгоритм безопасный: реагируем только при найденном клиенте + ясном статусе,
# при неоднозначности возвращаем структуру для уточнения вместо записи.

_FULL_KEYWORDS = (
    "полн", "оплачен", "оплата есть", "прошл", "пришл", "поступил",
    "получен", "видн", " есть", "+",
)
_PARTIAL_KEYWORDS = (
    "частич", "часть", "часть оплат", "не вся", "не весь",
)
_NONE_KEYWORDS = (
    "не вижу", "не нашл", "нет оплат", " нет", " ноль", " 0", " -",
    "не пришл", "не поступил",
)
# Стоп-слова — не считаем их «значимым токеном» имени клиента.
_NAME_STOP_TOKENS = frozenset({
    "тоо", "ип", "ао", "уп", "кх", "пкф", "тд", "пом",
    "ул", "пр", "пл", "пер", "д", "дом", "кв",
    "магазин", "кафе", "ресторан", "склад", "база", "точка",
    "клиент", "оплата", "оплат", "чек", "сумма",
})


def _client_tokens(name: str) -> list:
    """Значимые токены имени клиента — слова >=4 букв, без стоп-слов.

    Для дефисных слов отдаём И целое, И каждую часть отдельно (если >=4 букв).
    Например `акжан-лк` → ["акжан-лк", "акжан"]. Это нужно, чтобы Саида
    могла написать просто «Акжан» без хвоста.
    """
    norm = normalize_client_name(name).replace("ё", "е")
    # Грубое разбиение на слова (включая дефис как часть слова)
    parts = []
    cur = []
    for ch in norm:
        if ch.isalnum() or ch == "-":
            cur.append(ch)
        else:
            if cur:
                parts.append("".join(cur))
                cur = []
    if cur:
        parts.append("".join(cur))
    out = []
    seen = set()
    for p in parts:
        candidates = [p]
        if "-" in p:
            candidates.extend(p.split("-"))
        for cand in candidates:
            if len(cand) < 4:
                continue
            if cand in _NAME_STOP_TOKENS:
                continue
            if cand in seen:
                continue
            seen.add(cand)
            out.append(cand)
    return out


def _detect_status_from_text(text: str) -> Optional[str]:
    """Возвращает 'full' / 'partial' / 'none' / None по ключевым словам."""
    t = " " + (text or "").lower().replace("ё", "е") + " "
    # partial проверяем ПЕРВЫМ — иначе «частичная оплата» сматчит на full ("оплачен")
    for kw in _PARTIAL_KEYWORDS:
        if kw in t:
            return "partial"
    for kw in _NONE_KEYWORDS:
        if kw in t:
            return "none"
    for kw in _FULL_KEYWORDS:
        if kw in t:
            return "full"
    return None


def parse_saida_text_reply(text: str) -> Dict[str, Any]:
    """Разбирает свободный текст Саиды и возвращает решение.

    Возвращает dict:
      {
        "action": "confirm" | "ambiguous_client" | "unclear_status" | "no_match",
        "status": "full"|"partial"|"none"|None,
        "matches": [record, ...],  # совпавшие pending-записи
      }

    Никогда не пишет в файл — только определяет, что делать.
    Запись делается через confirm_by_saida(token, status) на стороне вызывающего.
    """
    t = (text or "").strip()
    result = {"action": "no_match", "status": None, "matches": []}
    if not t:
        return result

    pending = list_pending()
    if not pending:
        return result

    # 1. Найти все pending-записи, у которых хотя бы один значимый токен имени
    #    встречается как подстрока в тексте Саиды.
    norm_text = " " + normalize_client_name(t).replace("ё", "е") + " "
    matches: list = []
    for rec in pending:
        tokens = _client_tokens(rec.get("client", ""))
        if not tokens:
            continue
        for tok in tokens:
            if f" {tok}" in norm_text or f"{tok} " in norm_text or tok in norm_text:
                matches.append(rec)
                break

    if not matches:
        return result

    status = _detect_status_from_text(t)

    if len(matches) == 1:
        if status:
            return {"action": "confirm", "status": status, "matches": matches}
        return {"action": "unclear_status", "status": None, "matches": matches}

    # Несколько совпадений — нужно уточнение
    return {"action": "ambiguous_client", "status": status, "matches": matches}


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
        if status not in OPEN_STATUSES:
            continue
        # TTL только для подтверждённых холдов; pending_saida блокирует без TTL
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
    with _hold_lock():
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


def get_saida_hold_stats() -> Dict[str, Any]:
    """Read-only сводка backlog Саиды по saida_payment_holds.json."""
    data = _load()
    now = _now()
    today = now.date().isoformat()
    pending: list[Dict[str, Any]] = []
    totals = {
        "pending_total": 0,
        "warn_total": 0,
        "bypass_total": 0,
        "claimed_by_manager_total": 0,
        "closed_today": 0,
        "oldest_age_hours": 0.0,
    }
    managers: Dict[str, Dict[str, Any]] = {}

    for token, record in data.items():
        if not isinstance(record, dict):
            continue
        status = str(record.get("status") or "")
        updated_raw = str(record.get("updated_at") or record.get("saida_confirmed_at") or "")
        if status != "pending_saida":
            if updated_raw[:10] == today:
                totals["closed_today"] += 1
            continue

        created_raw = str(record.get("created_at") or record.get("updated_at") or "")
        try:
            created_at = datetime.fromisoformat(created_raw)
            if created_at.tzinfo is None and TZ:
                created_at = created_at.replace(tzinfo=TZ)
        except Exception:
            created_at = now
        age_hours = max((now - created_at).total_seconds() / 3600.0, 0.0)
        manager = str(record.get("manager") or "—")
        claimed_by_manager = bool(record.get("claimed_by_manager"))

        item = {
            "token": str(token),
            "manager": manager,
            "client": str(record.get("client") or "—"),
            "debt_str": str(record.get("debt_str") or ""),
            "status": status,
            "age_hours": age_hours,
            "created_at": created_raw,
            "claimed_by_manager": claimed_by_manager,
        }
        pending.append(item)
        totals["pending_total"] += 1
        totals["oldest_age_hours"] = max(totals["oldest_age_hours"], age_hours)
        if claimed_by_manager:
            totals["claimed_by_manager_total"] += 1
        if age_hours >= SAIDA_WARN_HOURS:
            totals["warn_total"] += 1
        if age_hours >= SAIDA_BYPASS_HOURS:
            totals["bypass_total"] += 1

        mgr = managers.setdefault(
            manager,
            {
                "manager": manager,
                "pending_total": 0,
                "warn_total": 0,
                "bypass_total": 0,
                "claimed_by_manager_total": 0,
                "oldest_age_hours": 0.0,
            },
        )
        mgr["pending_total"] += 1
        mgr["oldest_age_hours"] = max(mgr["oldest_age_hours"], age_hours)
        if claimed_by_manager:
            mgr["claimed_by_manager_total"] += 1
        if age_hours >= SAIDA_WARN_HOURS:
            mgr["warn_total"] += 1
        if age_hours >= SAIDA_BYPASS_HOURS:
            mgr["bypass_total"] += 1

    pending.sort(key=lambda item: (-item["age_hours"], item["manager"].lower(), item["client"].lower()))
    managers_list = sorted(
        managers.values(),
        key=lambda item: (-item["bypass_total"], -item["pending_total"], item["manager"].lower()),
    )

    return {
        "generated_at": _now_iso(),
        "totals": totals,
        "managers": managers_list,
        "oldest_pending": pending[:5],
    }


def get_partial_payment_stats() -> Dict[str, Any]:
    """Read-only сводка активных частичных оплат по saida_payment_holds.json."""
    data = _load()
    now = _now()
    partials: list[Dict[str, Any]] = []
    totals = {
        "partial_total": 0,
        "claimed_by_manager_total": 0,
        "oldest_age_hours": 0.0,
    }
    managers: Dict[str, Dict[str, Any]] = {}
    changed = False

    for token, record in list(data.items()):
        if not isinstance(record, dict):
            continue
        if str(record.get("status") or "") != "confirmed_partial":
            continue

        created_raw = record.get("saida_confirmed_at") or record.get("updated_at") or record.get("created_at")
        try:
            created_at = datetime.fromisoformat(str(created_raw))
            if created_at.tzinfo is None and TZ:
                created_at = created_at.replace(tzinfo=TZ)
        except Exception:
            created_at = now

        age_delta = now - created_at
        if age_delta > timedelta(days=HOLD_TTL_DAYS):
            record["status"] = "expired"
            record["expired_at"] = _now_iso()
            data[token] = record
            changed = True
            continue

        age_hours = max(age_delta.total_seconds() / 3600.0, 0.0)
        manager = str(record.get("manager") or "—")
        claimed_by_manager = bool(record.get("claimed_by_manager"))
        item = {
            "token": str(token),
            "manager": manager,
            "client": str(record.get("client") or "—"),
            "debt_str": str(record.get("debt_str") or ""),
            "age_hours": age_hours,
            "claimed_by_manager": claimed_by_manager,
            "created_at": str(record.get("created_at") or ""),
            "confirmed_at": str(record.get("saida_confirmed_at") or record.get("updated_at") or ""),
        }
        partials.append(item)
        totals["partial_total"] += 1
        totals["oldest_age_hours"] = max(totals["oldest_age_hours"], age_hours)
        if claimed_by_manager:
            totals["claimed_by_manager_total"] += 1

        mgr = managers.setdefault(
            manager,
            {
                "manager": manager,
                "partial_total": 0,
                "claimed_by_manager_total": 0,
                "oldest_age_hours": 0.0,
            },
        )
        mgr["partial_total"] += 1
        mgr["oldest_age_hours"] = max(mgr["oldest_age_hours"], age_hours)
        if claimed_by_manager:
            mgr["claimed_by_manager_total"] += 1

    if changed:
        _save(data)

    partials.sort(key=lambda item: (-item["age_hours"], item["manager"].lower(), item["client"].lower()))
    managers_list = sorted(
        managers.values(),
        key=lambda item: (-item["partial_total"], item["manager"].lower()),
    )
    return {
        "generated_at": _now_iso(),
        "totals": totals,
        "managers": managers_list,
        "oldest_partials": partials[:10],
    }


def _fmt_age_human(hours: float) -> str:
    """1.7 дн / 8 ч / 35 мин — человеческий формат возраста запроса."""
    h = float(hours or 0.0)
    if h < 1:
        mins = int(round(h * 60))
        return f"{mins} мин"
    if h < 24:
        return f"{h:.0f} ч"
    days = h / 24.0
    return f"{days:.1f} дн"


def format_saida_hold_stats_text() -> str:
    """Текстовая сводка backlog Саиды для директора (человеческий язык)."""
    stats = get_saida_hold_stats()
    totals = stats.get("totals", {})
    if not totals.get("pending_total"):
        return (
            "📋 <b>Запросы оплат у Саиды</b>\n\n"
            "Сейчас всё закрыто — новых запросов нет."
        )

    oldest_h = float(totals.get("oldest_age_hours") or 0.0)
    pending = totals.get("pending_total", 0)
    warn_total = totals.get("warn_total", 0)
    bypass_total = totals.get("bypass_total", 0)
    closed_today = totals.get("closed_today", 0)

    lines = [
        "📋 <b>Запросы оплат у Саиды</b>",
        "",
        f"Менеджеры заявили оплату — Саида ещё не подтвердила: <b>{pending}</b>",
        f"Самый старый ждёт уже <b>{_fmt_age_human(oldest_h)}</b>",
    ]
    # Просрочка по 4-часовому SLA — мягкое напоминание
    if warn_total:
        lines.append(
            f"Без ответа дольше {SAIDA_WARN_HOURS} ч: <b>{warn_total}</b> "
            f"(пора отвечать)"
        )
    # Просрочка по 8-часовому байпасу — критично, директор может ответить сам
    if bypass_total:
        lines.append(
            f"Без ответа дольше {SAIDA_BYPASS_HOURS} ч: <b>{bypass_total}</b> "
            f"(критично — директор может ответить вместо неё)"
        )
    if closed_today:
        lines.append(f"Сегодня закрыто: <b>{closed_today}</b>")
    else:
        lines.append("Сегодня закрыто: <b>0</b> — ни одного ответа за сегодня")

    if totals.get("claimed_by_manager_total", 0):
        lines.append(
            f"Из них без приложенного документа от менеджера: "
            f"<b>{totals['claimed_by_manager_total']}</b>"
        )

    managers = stats.get("managers", [])
    if managers:
        lines.append("")
        lines.append("<b>По менеджерам:</b>")
        for item in managers:
            mgr_open = item['pending_total']
            mgr_oldest = _fmt_age_human(item['oldest_age_hours'])
            line = f"• <b>{item['manager']}</b>: {mgr_open} ждут ответа, самый старый {mgr_oldest}"
            if item.get('bypass_total'):
                line += f" · {item['bypass_total']} критично"
            if item.get("claimed_by_manager_total"):
                line += f" · {item['claimed_by_manager_total']} без документа"
            lines.append(line)

    oldest_pending = stats.get("oldest_pending", [])
    if oldest_pending:
        lines.append("")
        lines.append("<b>Самые давние без ответа:</b>")
        for item in oldest_pending:
            debt_part = f" · {item['debt_str']} ₸" if item.get("debt_str") else ""
            wa_mark = " · из WA" if item.get("claimed_by_manager") else ""
            lines.append(
                f"• {item['manager']}: {item['client']} — "
                f"{_fmt_age_human(item['age_hours'])}{debt_part}{wa_mark}"
            )

    lines.append("")
    lines.append(
        "<i>Саида может ответить кнопками под каждым запросом или "
        "написать в свой чат «&lt;имя клиента&gt; полная/частично/нет».</i>"
    )

    return "\n".join(lines)


def format_partial_payment_stats_text() -> str:
    """Текстовая сводка частичных оплат для директора."""
    stats = get_partial_payment_stats()
    totals = stats.get("totals", {})
    if not totals.get("partial_total"):
        return "🔸 <b>Частичные оплаты</b>\n\nСейчас нет активных кейсов со статусом confirmed_partial."

    oldest_h = float(totals.get("oldest_age_hours") or 0.0)
    oldest_days = oldest_h / 24.0 if oldest_h else 0.0
    lines = [
        "🔸 <b>Частичные оплаты</b>",
        "",
        f"Активных кейсов: <b>{totals.get('partial_total', 0)}</b>",
        f"Старейший возраст: <b>{oldest_days:.1f} дн</b> ({oldest_h:.0f} ч)",
    ]
    if totals.get("claimed_by_manager_total", 0):
        lines.append(f"Из WA approval без документа: <b>{totals['claimed_by_manager_total']}</b>")

    managers = stats.get("managers", [])
    if managers:
        lines.append("")
        lines.append("<b>По менеджерам:</b>")
        for item in managers:
            line = (
                f"• <b>{item['manager']}</b> — частичных {item['partial_total']}, "
                f"старейший {item['oldest_age_hours'] / 24.0:.1f} дн"
            )
            if item.get("claimed_by_manager_total"):
                line += f", из WA {item['claimed_by_manager_total']}"
            lines.append(line)

    oldest_partials = stats.get("oldest_partials", [])
    if oldest_partials:
        lines.append("")
        lines.append("<b>Текущие кейсы:</b>")
        for item in oldest_partials:
            debt_part = f" · {item['debt_str']} ₸" if item.get("debt_str") else ""
            source_part = " · WA" if item.get("claimed_by_manager") else ""
            lines.append(
                f"• {item['manager']}: {item['client']} — {item['age_hours'] / 24.0:.1f} дн{debt_part}{source_part}"
            )

    return "\n".join(lines)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collections/debt_monitor.py
Анализ дебиторки, классификация должников по уровням давления.

Версия: 1.0.5 (2026-04-11)

Уровни:
  0–9 дней   → level 0 (пропустить)
  10–14 дней → level 1 (мягкое напоминание)
  15–19 дней → level 2 (среднее давление)
  20–24 дней → level 3 (настойчиво)
  25–29 дней → level 4 (строго + звонок)
  30+ дней   → level 5 (жёстко + эскалация директору)
"""

import json
import logging
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from zoneinfo import ZoneInfo

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env",
            encoding="utf-8-sig", override=False)

TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))

ROOT_DIR = Path(__file__).resolve().parent.parent
JSON_DIR = ROOT_DIR / "reports" / "json"
CONTACTS_PATH = ROOT_DIR / "config" / "debtors_contacts.json"
LOGS_DIR = ROOT_DIR / "logs"

logger = logging.getLogger(__name__)


def _safe_float(val: Any) -> float:
    """Парсит число из строки 1C безопасно: '1 234 567,89' → 1234567.89.
    Убирает пробелы/nbsp, заменяет ТОЛЬКО ПЕРВУЮ запятую на точку."""
    if val is None:
        return 0.0
    s = str(val).replace(" ", "").replace("\xa0", "").replace("\u202f", "")
    # Если запятая — десятичный разделитель (европейский формат):
    # "1234567,89" → "1234567.89"
    # "1,234,567" → "1.234.567" ОШИБКА — поэтому берём только первую запятую
    s = s.replace(",", ".", 1)
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


# Порог уровней (минимальное кол-во дней просрочки)
_LEVEL_THRESHOLDS = [
    (30, 5),
    (25, 4),
    (20, 3),
    (15, 2),
    (10, 1),
    (0,  0),
]


def _level_for_days(days: int) -> int:
    """Определяет уровень давления по кол-ву дней просрочки."""
    for threshold, level in _LEVEL_THRESHOLDS:
        if days >= threshold:
            return level
    return 0


def load_latest_debt_json() -> Dict[str, Any]:
    """Загружает последний по дате debt_ext_*.json для каждой группы (менеджера).

    Группировка по базовому имени файла (без ' (NNN)').
    Клиенты всех групп объединяются; при дублях берётся запись с большим days_silence.
    """
    candidates = list(JSON_DIR.glob("debt_ext_*.json"))
    if not candidates:
        logger.warning("Нет debt_ext_*.json в %s", JSON_DIR)
        return {}

    # Группируем по базовому имени (убираем суффикс ' (NNN)' и timestamp-префикс YYYYMMDDHHMMSS_)
    groups: Dict[str, List[Path]] = {}
    for p in candidates:
        base = re.sub(r"\s*\(\d+\)$", "", p.stem)
        base = re.sub(r"^(debt_ext_)\d{14}_", r"\1", base)
        groups.setdefault(base, []).append(p)

    merged_clients: Dict[str, Dict[str, Any]] = {}
    loaded = 0
    for base, paths in groups.items():
        latest = max(paths, key=_safe_mtime)
        try:
            with open(latest, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Ошибка чтения %s: %s", latest.name, e)
            continue
        # Пропускаем общие файлы без привязки к менеджеру —
        # их клиенты уже есть в per-менеджерных файлах.
        # manager='?' означает «не определён» (общий файл).
        file_mgr = (data.get("manager") or "") if isinstance(data, dict) else ""
        if not file_mgr or file_mgr in ("?", "-", "—", "ABSENT"):
            logger.debug("Пропускаем общий файл (нет менеджера): %s", latest.name)
            continue
        logger.info("Загружаем debt JSON: %s", latest.name)

        clients: List[Dict[str, Any]] = []
        blocks_by_client: Dict[str, Dict[str, Any]] = {}
        if isinstance(data, dict):
            for key in ("clients", "rows", "data"):
                if key in data and isinstance(data[key], list):
                    clients = data[key]
                    break
            else:
                for name, val in data.items():
                    if isinstance(val, dict):
                        clients.append({"name": name, **val})
            blocks = data.get("blocks")
            if isinstance(blocks, list):
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    block_name = (block.get("name") or block.get("client") or "").strip()
                    if block_name:
                        blocks_by_client[block_name] = block
        elif isinstance(data, list):
            clients = data

        file_manager = data.get("manager", "") if isinstance(data, dict) else ""
        period_min = data.get("period_min", "") if isinstance(data, dict) else ""
        period_max = data.get("period_max", "") if isinstance(data, dict) else ""
        for c in clients:
            if not isinstance(c, dict):
                continue
            name = (c.get("name") or c.get("client") or "").strip()
            if not name:
                continue
            # Сохраняем привязку к менеджеру из корня файла
            c_stamped = dict(c)
            block = blocks_by_client.get(name)
            if block:
                movements = block.get("movements")
                if isinstance(movements, list):
                    c_stamped["_movements"] = movements
                    c_stamped["_movement_block"] = block
            if period_min and not c_stamped.get("_period_min"):
                c_stamped["_period_min"] = period_min
            if period_max and not c_stamped.get("_period_max"):
                c_stamped["_period_max"] = period_max
            if file_manager and not c_stamped.get("_manager"):
                c_stamped["_manager"] = file_manager
            existing = merged_clients.get(name)
            if existing is None:
                merged_clients[name] = c_stamped
            else:
                if _extract_days_silence(c_stamped) > _extract_days_silence(existing):
                    merged_clients[name] = c_stamped
        loaded += 1

    logger.info("Загружено %d файлов, объединено %d клиентов", loaded, len(merged_clients))
    return {"clients": list(merged_clients.values())}


def _extract_days_silence(c: Dict[str, Any]) -> int:
    """Возвращает days_silence из записи клиента (вспомогательная функция)."""
    for field in ("days_silence", "max_days", "days", "overdue_days", "max_overdue_days"):
        val = c.get(field)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                continue
    return 0


def _safe_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except (FileNotFoundError, OSError):
        return 0.0


def get_overdue_days(client_data: Dict[str, Any]) -> int:
    """Извлекает максимальное кол-во дней просрочки клиента из данных debt JSON.

    Поддерживаемые поля: max_days, days, overdue_days, max_overdue_days.
    Возвращает 0 если поле не найдено.
    """
    for field in ("days_silence", "max_days", "days", "overdue_days", "max_overdue_days"):
        val = client_data.get(field)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                continue
    # Попробуем вычислить из invoice_date если есть
    invoice_date_str = client_data.get("invoice_date") or client_data.get("date")
    if invoice_date_str:
        try:
            from datetime import datetime as _dt
            d = _dt.strptime(str(invoice_date_str)[:10], "%Y-%m-%d").date()
            return (date.today() - d).days
        except (ValueError, TypeError):
            pass
    return 0


def _parse_movement_date(value: Any) -> Optional[date]:
    raw = str(value or "").strip()
    if not raw:
        return None
    raw = raw[:10]
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _fmt_iso(d: Optional[date]) -> Optional[str]:
    return d.isoformat() if d else None


def _movement_entries(movement: Dict[str, Any]) -> List[Tuple[str, date, float]]:
    d = _parse_movement_date(movement.get("date"))
    if not d:
        return []

    entries: List[Tuple[str, date, float]] = []
    kind = str(movement.get("type") or movement.get("kind") or "").lower()
    amount = _safe_float(movement.get("amount") or movement.get("sum") or 0)
    if kind:
        if "оплат" in kind or "credit" in kind or kind == "payment":
            if amount > 0:
                entries.append(("payment", d, amount))
        else:
            if amount > 0:
                entries.append(("shipment", d, amount))
        return entries

    debit = _safe_float(movement.get("debit") or 0)
    credit = _safe_float(movement.get("credit") or 0)
    if debit > 0:
        entries.append(("shipment", d, debit))
    if credit > 0:
        entries.append(("payment", d, credit))
    return entries


def compute_residual_debt_profile(
    client_data: Dict[str, Any],
    as_of_date: Optional[date] = None,
) -> Dict[str, Any]:
    """Calculates current debt age by matching payments to oldest shipments.

    Returns a profile, not just a number, so manager/admin previews can explain
    why the collector level was chosen. If detailed movements are unavailable,
    falls back to the legacy payment-silence metric.
    """
    as_of = (
        as_of_date
        or _parse_movement_date(client_data.get("_as_of_date"))
        or _parse_movement_date(client_data.get("_period_max"))
        or date.today()
    )
    payment_silence_days = get_overdue_days(client_data)
    movements = client_data.get("_movements")
    if not isinstance(movements, list) or not movements:
        return {
            "residual_debt_age_days": payment_silence_days,
            "payment_silence_days": payment_silence_days,
            "oldest_unpaid_date": None,
            "unpaid_parts": [],
            "basis": "fallback_days_silence",
            "confidence": "low",
            "active_turnover": False,
            "explanation": "movements unavailable; using payment silence",
        }

    period_min = _parse_movement_date(client_data.get("_period_min")) or as_of
    opening = _safe_float(client_data.get("opening") or 0)
    queue: List[Tuple[date, float, str]] = []
    unapplied_payment = 0.0

    if opening > 0:
        queue.append((period_min, opening, "opening"))
    elif opening < 0:
        unapplied_payment = abs(opening)

    parsed_entries: List[Tuple[str, date, float]] = []
    for movement in movements:
        if isinstance(movement, dict):
            parsed_entries.extend(_movement_entries(movement))
    parsed_entries.sort(key=lambda item: (item[1], 0 if item[0] == "shipment" else 1))

    recent_from = as_of - timedelta(days=14)
    recent_shipment = False
    recent_payment = False

    for kind, movement_date, amount in parsed_entries:
        if amount <= 0:
            continue
        if movement_date >= recent_from:
            if kind == "shipment":
                recent_shipment = True
            elif kind == "payment":
                recent_payment = True

        if kind == "shipment":
            if unapplied_payment > 0:
                covered = min(unapplied_payment, amount)
                amount -= covered
                unapplied_payment -= covered
            if amount > 0:
                queue.append((movement_date, amount, "shipment"))
            continue

        remaining = amount
        while queue and remaining > 0:
            item_date, item_amount, source = queue[0]
            if remaining + 0.005 >= item_amount:
                remaining -= item_amount
                queue.pop(0)
            else:
                queue[0] = (item_date, item_amount - remaining, source)
                remaining = 0.0
        if remaining > 0:
            unapplied_payment += remaining

    debt_amount = _safe_float(
        client_data.get("amount")
        or client_data.get("closing")
        or client_data.get("debt")
        or client_data.get("balance")
        or 0
    )
    if debt_amount <= 0:
        return {
            "residual_debt_age_days": 0,
            "payment_silence_days": payment_silence_days,
            "oldest_unpaid_date": None,
            "unpaid_parts": [],
            "basis": "no_debt",
            "confidence": "high",
            "active_turnover": recent_shipment and recent_payment,
            "explanation": "debt is closed",
        }

    queue_total = sum(amount for _, amount, _ in queue)
    if queue_total > debt_amount + 1.0:
        overage = queue_total - debt_amount
        while queue and overage > 0:
            item_date, item_amount, source = queue[-1]
            if overage + 0.005 >= item_amount:
                overage -= item_amount
                queue.pop()
            else:
                queue[-1] = (item_date, item_amount - overage, source)
                overage = 0.0

    if not queue:
        return {
            "residual_debt_age_days": payment_silence_days,
            "payment_silence_days": payment_silence_days,
            "oldest_unpaid_date": None,
            "unpaid_parts": [],
            "basis": "fallback_days_silence",
            "confidence": "low",
            "active_turnover": recent_shipment and recent_payment,
            "explanation": "positive debt but FIFO queue is empty; using payment silence",
        }

    oldest_date = queue[0][0]
    age_days = max((as_of - oldest_date).days, 0)
    basis = "opening_fallback" if queue[0][2] == "opening" else "movements_fifo"
    confidence = "medium" if basis == "opening_fallback" else "high"
    unpaid_parts = [
        {"date": _fmt_iso(item_date), "amount": round(amount, 2), "source": source}
        for item_date, amount, source in queue
        if amount > 0.005
    ]

    return {
        "residual_debt_age_days": age_days,
        "payment_silence_days": payment_silence_days,
        "oldest_unpaid_date": _fmt_iso(oldest_date),
        "unpaid_parts": unpaid_parts,
        "basis": basis,
        "confidence": confidence,
        "active_turnover": recent_shipment and recent_payment,
        "explanation": (
            "oldest unpaid part is from "
            f"{_fmt_iso(oldest_date)}; payment silence is {payment_silence_days} days"
        ),
    }


def classify_debtors(debt_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Классифицирует должников по уровням давления.

    Принимает словарь debt JSON (ключ = имя клиента или список clients/rows).
    Возвращает список dict с полями:
      name, amount, days, level, raw
    """
    results: List[Dict[str, Any]] = []

    # Поддерживаем разные форматы debt JSON
    clients: List[Dict[str, Any]] = []
    if isinstance(debt_data, dict):
        if "clients" in debt_data:
            clients = debt_data["clients"]
        elif "rows" in debt_data:
            clients = debt_data["rows"]
        elif "data" in debt_data:
            clients = debt_data["data"]
        else:
            # Плоский словарь {name: {...}}
            for name, val in debt_data.items():
                if isinstance(val, dict):
                    clients.append({"name": name, **val})
    elif isinstance(debt_data, list):
        clients = debt_data

    for client in clients:
        if not isinstance(client, dict):
            continue
        name = (
            client.get("name") or
            client.get("client") or
            client.get("контрагент") or
            client.get("клиент") or
            ""
        ).strip()
        if not name:
            continue

        debt_age_profile = compute_residual_debt_profile(client)
        days = int(debt_age_profile.get("residual_debt_age_days", get_overdue_days(client)) or 0)
        payment_silence_days = int(debt_age_profile.get("payment_silence_days", get_overdue_days(client)) or 0)
        amount = 0.0
        for field in ("amount", "closing", "debt", "balance", "сумма", "остаток"):
            val = client.get(field)
            if val is not None:
                v = _safe_float(val)
                if v != 0.0 or str(val).strip() not in ("", "0", "0.0"):
                    amount = v
                    break

        # Клиент с нулевым/отрицательным или ниже минимального порога долгом — пропуск
        if amount < 5000:
            continue

        # Нарушение: была отгрузка при наличии предыдущего долга
        opening = _safe_float(client.get("opening") or 0)
        debit   = _safe_float(client.get("debit") or 0)
        # Нарушение: отгрузка при наличии предыдущего долга И просрочка >= 7 дней.
        # Флаг violation_shipment используется ТОЛЬКО для уведомления менеджера ("Внимание!").
        # Повышение level производится только по стандартным порогам (_level_for_days).
        violation_shipment = opening >= 100 and debit > 0 and payment_silence_days >= 7

        level = _level_for_days(days)
        # Уровень определяется только по дням (стандартные пороги):
        # 0–9 дней → level 0 (не трогаем), 10+ → level 1 и выше.
        # violation_shipment НЕ повышает level — иначе 7–9-дневные клиенты
        # попадали бы в коллектор как должники, хотя они ещё активны.

        # Извлекаем credit (платежи) для статистики
        credit = _safe_float(client.get("credit") or 0)

        results.append({
            "name": name,
            "amount": amount,
            "days": days,
            "level": level,
            "residual_debt_age_days": days,
            "payment_silence_days": payment_silence_days,
            "oldest_unpaid_date": debt_age_profile.get("oldest_unpaid_date"),
            "unpaid_parts": debt_age_profile.get("unpaid_parts", []),
            "debt_age_basis": debt_age_profile.get("basis", ""),
            "debt_age_confidence": debt_age_profile.get("confidence", ""),
            "active_turnover": bool(debt_age_profile.get("active_turnover", False)),
            "classification_explanation": debt_age_profile.get("explanation", ""),
            "opening": opening,
            "violation_shipment": violation_shipment,
            "debit": debit,    # текущие отгрузки (>0 = клиент активно покупает)
            "credit": credit,  # платежи за период (>0 = клиент что-то платит)
            "manager": client.get("_manager", ""),
            "raw": client,
        })

    results.sort(key=lambda x: (-x["level"], -x["days"]))
    logger.info("Классифицировано %d должников (уровни 1–5: %d)",
                len(results),
                sum(1 for r in results if r["level"] > 0))
    return results


def load_contacts() -> Dict[str, Any]:
    """Загружает справочник контактов из config/debtors_contacts.json."""
    if not CONTACTS_PATH.exists():
        logger.warning("Файл контактов не найден: %s", CONTACTS_PATH)
        return {}
    try:
        with open(CONTACTS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        # Убираем служебный ключ _comment
        data.pop("_comment", None)
        return data
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Ошибка чтения контактов: %s", e)
        return {}


def _strip_prefix(name: str) -> str:
    """Удаляет ТОО/ИП/АО/LLP/ОАО префиксы для нечёткого сравнения."""
    return re.sub(
        r"^\s*(ТОО|ИП|АО|ОАО|ООО|LLP|LLC|ЧП)\s+",
        "", name, flags=re.IGNORECASE
    ).strip()


def match_client(debt_name: str, contacts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Находит контакт должника в справочнике.

    Порядок:
      1. Прямое совпадение
      2. lower().strip()
      3. Без ТОО/ИП/АО/LLP префиксов
      4. Совпадение по первым 3 словам
    """
    if not debt_name or not contacts:
        return None

    # 1. Прямое совпадение
    if debt_name in contacts:
        return contacts[debt_name]

    # 2. lower().strip()
    debt_lower = debt_name.lower().strip()
    for key, val in contacts.items():
        if key.lower().strip() == debt_lower:
            return val

    # 3. Без префиксов
    debt_stripped = _strip_prefix(debt_name).lower()
    if debt_stripped:
        for key, val in contacts.items():
            if _strip_prefix(key).lower() == debt_stripped:
                return val

    # 4. Первые 3 слова
    debt_words = debt_lower.split()[:3]
    if len(debt_words) >= 2:
        for key, val in contacts.items():
            key_words = key.lower().split()[:3]
            if key_words[:len(debt_words)] == debt_words:
                return val

    return None

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collections/debt_monitor.py
Анализ дебиторки, классификация должников по уровням давления.

Версия: 1.0.0 (2026-03-16)

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
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    # Группируем по базовому имени (убираем суффикс ' (NNN)')
    groups: Dict[str, List[Path]] = {}
    for p in candidates:
        base = re.sub(r"\s*\(\d+\)$", "", p.stem)
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
        if isinstance(data, dict):
            for key in ("clients", "rows", "data"):
                if key in data and isinstance(data[key], list):
                    clients = data[key]
                    break
            else:
                for name, val in data.items():
                    if isinstance(val, dict):
                        clients.append({"name": name, **val})
        elif isinstance(data, list):
            clients = data

        file_manager = data.get("manager", "") if isinstance(data, dict) else ""
        for c in clients:
            if not isinstance(c, dict):
                continue
            name = (c.get("name") or c.get("client") or "").strip()
            if not name:
                continue
            # Сохраняем привязку к менеджеру из корня файла
            c_stamped = dict(c)
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

        days = get_overdue_days(client)
        amount = 0.0
        for field in ("amount", "closing", "debt", "balance", "сумма", "остаток"):
            val = client.get(field)
            if val is not None:
                try:
                    amount = float(str(val).replace(" ", "").replace(",", "."))
                    break
                except (ValueError, TypeError):
                    continue

        # Клиент с нулевым/отрицательным или ниже минимального порога долгом — пропуск
        if amount < 5000:
            continue

        # Нарушение: была отгрузка при наличии предыдущего долга
        opening = 0.0
        debit = 0.0
        try:
            opening = float(str(client.get("opening") or 0).replace(" ", "").replace(",", "."))
            debit = float(str(client.get("debit") or 0).replace(" ", "").replace(",", "."))
        except (ValueError, TypeError):
            pass
        violation_shipment = opening > 0 and debit > 0

        level = _level_for_days(days)
        # Нарушение → минимум уровень 1, даже если дней молчания < 10
        if violation_shipment and level == 0:
            level = 1

        results.append({
            "name": name,
            "amount": amount,
            "days": days,
            "level": level,
            "violation_shipment": violation_shipment,
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

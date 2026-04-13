#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/shipment_control.py
Контроль решений об отгрузке должникам после эскалации WhatsApp-диалога.

Версия: 1.0.0 (2026-04-13)

Четыре состояния (менеджер/админ выбирает кнопкой при эскалации):
  allow         — разрешить сейчас (снять вопрос)
  allow_after   — разрешить ПОСЛЕ полной оплаты (долг ≤ 1000 ₸)
  block_until   — запретить ДО полной оплаты, уведомить когда долг закрыт
  block         — запретить безусловно

Хранилище: logs/collector_shipment_decisions.json
Ключ: нормализованное имя клиента (строчные, одиночные пробелы)

Ежедневный job (14:00) вызывает check_pending_decisions(bot) —
читает свежий debt_ext JSON и авто-закрывает allow_after / block_until
когда долг клиента ≤ FULL_PAYMENT_THRESHOLD.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from zoneinfo import ZoneInfo

load_dotenv(
    dotenv_path=Path(__file__).resolve().parent.parent / ".env",
    encoding="utf-8-sig",
    override=False,
)

TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))
_ROOT = Path(__file__).resolve().parent.parent
_DECISIONS_PATH = _ROOT / "logs" / "collector_shipment_decisions.json"

# Порог «полная оплата» — долг ≤ 1000 ₸ считается закрытым (±1С округление)
FULL_PAYMENT_THRESHOLD = float(os.getenv("SHIPMENT_FULL_PAYMENT_THRESHOLD", "1000"))

logger = logging.getLogger(__name__)

VALID_DECISIONS = {"allow", "allow_after", "block_until", "block"}


def _now_iso() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def _normalize(name: str) -> str:
    return " ".join(str(name or "").lower().split())


# ─── I/O ─────────────────────────────────────────────────────────────────────

def _load() -> Dict[str, Any]:
    try:
        if _DECISIONS_PATH.exists():
            data = json.loads(_DECISIONS_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save(data: Dict[str, Any]) -> None:
    _DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(_DECISIONS_PATH.parent), suffix=".tmp", prefix="ship_dec_"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, str(_DECISIONS_PATH))
    except (OSError, TypeError, ValueError):
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─── Public API ───────────────────────────────────────────────────────────────

def set_decision(
    phone: str,
    client_name: str,
    decision: str,
    manager_name: str = "",
    manager_chat_id: int = 0,
    amount: float = 0.0,
) -> Dict[str, Any]:
    """Сохраняет решение об отгрузке для клиента.

    Args:
        phone:           Номер телефона (ключ диалога).
        client_name:     Имя клиента из 1С.
        decision:        allow | allow_after | block_until | block
        manager_name:    Кто принял решение.
        manager_chat_id: Telegram chat_id менеджера для обратной связи.
        amount:          Текущий долг на момент решения.

    Returns:
        Сохранённая запись.
    """
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Неверное решение: {decision!r}. Допустимые: {VALID_DECISIONS}")

    data = _load()
    norm = _normalize(client_name)
    record: Dict[str, Any] = {
        "phone":           phone,
        "client_name":     client_name,
        "client_norm":     norm,
        "decision":        decision,
        "manager_name":    manager_name,
        "manager_chat_id": manager_chat_id,
        "amount_at_decision": amount,
        "decided_at":      _now_iso(),
        "resolved_at":     None,
        "resolved_reason": None,
    }
    data[norm] = record
    _save(data)
    logger.info(
        "Решение об отгрузке: %s → %s (менеджер: %s, долг: %.0f ₸)",
        client_name, decision, manager_name, amount,
    )
    return record


def get_decision(client_name: str) -> Optional[Dict[str, Any]]:
    """Возвращает активное решение по клиенту или None."""
    norm = _normalize(client_name)
    rec = _load().get(norm)
    if not isinstance(rec, dict):
        return None
    # Закрытые решения не возвращаем
    if rec.get("resolved_at"):
        return None
    return rec


def resolve_decision(client_name: str, reason: str) -> bool:
    """Закрывает решение (долг погашен / отменено вручную)."""
    norm = _normalize(client_name)
    data = _load()
    rec = data.get(norm)
    if not isinstance(rec, dict) or rec.get("resolved_at"):
        return False
    rec["resolved_at"] = _now_iso()
    rec["resolved_reason"] = reason
    data[norm] = rec
    _save(data)
    logger.info("Решение об отгрузке закрыто: %s (%s)", client_name, reason)
    return True


def list_pending() -> List[Dict[str, Any]]:
    """Возвращает все незакрытые решения allow_after / block_until."""
    data = _load()
    return [
        rec for rec in data.values()
        if isinstance(rec, dict)
        and not rec.get("resolved_at")
        and rec.get("decision") in ("allow_after", "block_until")
    ]


# ─── Daily check ──────────────────────────────────────────────────────────────

async def check_pending_decisions(bot: Any) -> int:
    """Проверяет allow_after / block_until клиентов против свежего debt JSON.

    Если долг ≤ FULL_PAYMENT_THRESHOLD (1000 ₸) — считается полной оплатой:
      allow_after  → авто-снятие + уведомление менеджеру «можно отгружать»
      block_until  → уведомление менеджеру «оплата пришла, проверьте»

    Возвращает количество обработанных записей.
    """
    pending = list_pending()
    if not pending:
        return 0

    # Загружаем свежий debt JSON
    try:
        from collector.debt_monitor import load_latest_debt_json
        debt_data = load_latest_debt_json()
        clients_list = (
            debt_data.get("clients") or debt_data.get("rows") or []
            if isinstance(debt_data, dict) else []
        )
    except Exception as e:
        logger.error("check_pending_decisions: ошибка загрузки debt JSON: %s", e)
        return 0

    # Строим словарь долгов: нормализованное_имя → сумма
    debt_by_norm: Dict[str, float] = {}
    for c in clients_list:
        if not isinstance(c, dict):
            continue
        name = c.get("name") or c.get("client") or ""
        if not name:
            continue
        amount = float(c.get("amount") or c.get("debt") or 0)
        debt_by_norm[_normalize(name)] = amount

    resolved = 0
    for rec in pending:
        norm = rec.get("client_norm", "")
        client_name = rec.get("client_name", norm)
        decision = rec.get("decision")
        mgr_id = rec.get("manager_chat_id", 0)

        current_debt = debt_by_norm.get(norm)
        if current_debt is None:
            # Клиент пропал из дебиторки — долг закрыт
            current_debt = 0.0

        if current_debt <= FULL_PAYMENT_THRESHOLD:
            if decision == "allow_after":
                resolve_decision(client_name, "debt_cleared")
                msg = (
                    f"✅ <b>{client_name}</b> — долг погашен.\n"
                    f"Остаток: <b>{current_debt:,.0f} ₸</b>\n"
                    f"Отгрузка <b>разрешена</b> (авто-снятие условия)."
                )
            else:  # block_until
                resolve_decision(client_name, "debt_cleared_notify")
                msg = (
                    f"💰 <b>{client_name}</b> — долг погашен.\n"
                    f"Остаток: <b>{current_debt:,.0f} ₸</b>\n"
                    f"Условие «запрет до оплаты» снято. Проверьте и разрешите отгрузку."
                )

            if mgr_id and bot:
                try:
                    await bot.send_message(chat_id=mgr_id, text=msg, parse_mode="HTML")
                except Exception as e:
                    logger.warning(
                        "check_pending_decisions: не удалось уведомить менеджера %s: %s",
                        mgr_id, e,
                    )
            resolved += 1
            logger.info(
                "check_pending_decisions: %s → %s (долг=%.0f ₸)",
                client_name, decision, current_debt,
            )

    return resolved

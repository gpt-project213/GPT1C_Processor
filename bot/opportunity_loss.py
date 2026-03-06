"""
opportunity_loss.py · v1.5.0 (28.02.2026)
──────────────────────────────────────────────────────────────────────────────
Модуль "Упущенная прибыль" — считает потенциальные потери по должникам.

Формула: loss = долг × маржа_менеджера (% последнего gross отчёта)

Фильтры:
  - молчание >= 15 дней (ALARM_DAYS из silence_alerts)
  - долг >= 10 000 ₸ (MIN_DEBT_AMOUNT из silence_alerts)

Зоны риска:
  15–60 дней   → ⚡ рабочая просрочка
  60–120 дней  → 🔴 красная зона
  120+ дней    → ☠️  мёртвая дебиторка (коэффициент 100%)

Источники данных:
  - debt HTML  → silence_alerts.SilenceAlert (parse_html_silence_days, get_latest_debt_report)
  - gross HTML → silence_alerts + GrossSummary (для маржи)

Используется в send_reports.py: отправка admin + subadmin после silence_alerts (14:05 и 21:05).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Зоны риска ──────────────────────────────────────────────────────────────
ZONE_YELLOW_MIN  = 8     # >= 8 дней  → просрочка (договорной срок 7 дней)
ZONE_RED_MIN     = 15    # >= 15 дней → красная зона
# Зона 30+ убрана — в месячных отчётах не накапливается

# Оборачиваемость: 2 раза в месяц = период 15 дней
# Реальные потери = долг × маржа% × (дней_молчания / TURNOVER_DAYS)
TURNOVER_DAYS = 15  # дней на один оборот капитала

# Маржа по умолчанию если gross отчёт не найден
DEFAULT_MARGIN_PCT = 10.0

# Минимальная сумма долга для расчёта (берём из silence_alerts.MIN_DEBT_AMOUNT)
MIN_DEBT_AMOUNT = 10_000.0

# ── Утилиты форматирования ───────────────────────────────────────────────────

def _fmt_money(amount: float) -> str:
    """Форматирует сумму: 1500000.0 → '1 500 000 ₸'"""
    return f"{amount:,.0f}".replace(",", " ") + " ₸"


def _fmt_pct(pct: float) -> str:
    return f"{pct:.1f}%"


# ── Поиск gross HTML для менеджера ────────────────────────────────────────────

def _find_latest_gross_html(html_dir: Path, manager_name: str) -> Optional[Path]:
    """
    Ищет последний файл *{manager_name}*_gross_sum.html в html_dir.
    Если менеджер не найден — берёт любой последний gross_sum.html.
    Сортировка по mtime (быстро, не парсим HTML).
    """
    # Пытаемся найти по имени менеджера
    manager_lower = manager_name.lower()
    candidates = [
        p for p in html_dir.glob("*_gross_sum.html")
        if manager_lower in p.name.lower()
    ]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    # Fallback: любой gross_sum.html
    all_gross = list(html_dir.glob("*_gross_sum.html"))
    if all_gross:
        logger.warning(
            f"opportunity_loss: gross HTML для '{manager_name}' не найден, "
            f"используется общий файл"
        )
        return max(all_gross, key=lambda p: p.stat().st_mtime)

    logger.warning(f"opportunity_loss: нет gross HTML файлов в {html_dir}")
    return None


def _get_manager_margin(html_dir: Path, manager_name: str) -> Tuple[float, str]:
    """
    Извлекает % маржи для менеджера из последнего gross_sum.html.
    Возвращает: (margin_pct, source_description)
    """
    try:
        from gross_summary import GrossSummary  # type: ignore
    except ImportError:
        logger.warning("opportunity_loss: GrossSummary не импортирован, используется маржа по умолчанию")
        return DEFAULT_MARGIN_PCT, f"по умолчанию ({DEFAULT_MARGIN_PCT}%)"

    gross_html = _find_latest_gross_html(html_dir, manager_name)
    if not gross_html:
        return DEFAULT_MARGIN_PCT, f"по умолчанию ({DEFAULT_MARGIN_PCT}%)"

    try:
        gs = GrossSummary()
        data = gs.parse_gross_html(gross_html)
        margin = float(data.get("margin", 0.0))
        if margin <= 0:
            logger.warning(
                f"opportunity_loss: маржа 0% в {gross_html.name}, используется {DEFAULT_MARGIN_PCT}%"
            )
            return DEFAULT_MARGIN_PCT, f"по умолчанию ({DEFAULT_MARGIN_PCT}%, файл пустой)"
        return margin, gross_html.name
    except Exception as e:
        logger.warning(f"opportunity_loss: ошибка парсинга {gross_html.name}: {e}")
        return DEFAULT_MARGIN_PCT, f"по умолчанию ({DEFAULT_MARGIN_PCT}%)"


# ── Основная логика ──────────────────────────────────────────────────────────

def calculate_opportunity_loss(
    html_dir: Path,
    manager_name: str
) -> Optional[Dict]:
    """
    Рассчитывает упущенную прибыль по долгам менеджера.

    Возвращает dict:
    {
      'manager': str,
      'margin_pct': float,
      'margin_source': str,
      'report_date': str,
      'zones': {
          'dead':   [{'client': str, 'debt': float, 'debt_str': str, 'days': int, 'loss': float}],
          'red':    [...],
          'yellow': [...]
      },
      'total_loss': float,
      'dead_loss': float,
      'red_loss': float,
      'yellow_loss': float,
      'total_clients': int
    }
    Возвращает None если данных нет.
    """
    try:
        from silence_alerts import SilenceAlert  # type: ignore
    except ImportError:
        logger.error("opportunity_loss: SilenceAlert не импортирован")
        return None

    alert = SilenceAlert()

    # 1. Найти последний debt HTML
    debt_html = alert.get_latest_debt_report(html_dir, manager_name)
    if not debt_html:
        logger.warning(f"opportunity_loss: нет debt HTML для {manager_name}")
        return None

    # 2. Парсим клиентов
    clients_data = alert.parse_html_silence_days(debt_html)
    if not clients_data:
        logger.warning(f"opportunity_loss: нет данных клиентов для {manager_name}")
        return None

    # 3. Дата отчёта
    report_date = alert.parse_report_date(debt_html) or ""

    # 4. Маржа менеджера
    margin_pct, margin_source = _get_manager_margin(html_dir, manager_name)

    # 5. Фильтруем и группируем по зонам
    zones: Dict[str, List[Dict]] = {"dead": [], "red": [], "yellow": []}

    for client in clients_data:
        days  = client.get("silence_days", 0)
        debt  = client.get("debt", 0.0)
        name  = client.get("client", "—")
        d_str = client.get("debt_str", "")

        # Фильтры
        if days < ZONE_YELLOW_MIN:
            continue
        if debt < MIN_DEBT_AMOUNT:
            continue

        loss = debt * (margin_pct / 100.0)
        turns = max(1.0, days / TURNOVER_DAYS)  # минимум 1 оборот
        real_loss = loss * turns  # с учётом оборачиваемости (2 раза/мес)

        entry = {
            "client":    name,
            "debt":      debt,
            "debt_str":  d_str,
            "days":      days,
            "loss":      loss,       # простые потери (1 оборот)
            "real_loss": real_loss,  # реальные потери с оборачиваемостью
            "turns":     turns,
        }

        if days >= ZONE_RED_MIN:
            zones["red"].append(entry)
        else:
            zones["yellow"].append(entry)

    # Сортируем внутри каждой зоны по сумме долга (больший долг выше)
    for key in zones:
        zones[key].sort(key=lambda x: x["debt"], reverse=True)

    # 6. Итоги
    red_loss       = sum(e["loss"] for e in zones["red"])
    yellow_loss    = sum(e["loss"] for e in zones["yellow"])
    total_loss     = red_loss + yellow_loss

    red_real       = sum(e["real_loss"] for e in zones["red"])
    yellow_real    = sum(e["real_loss"] for e in zones["yellow"])
    total_real     = red_real + yellow_real

    total_clients = len(zones["red"]) + len(zones["yellow"])

    if total_clients == 0:
        logger.info(f"opportunity_loss: у {manager_name} нет клиентов с просрочкой >= {ZONE_YELLOW_MIN} дней")
        return None

    return {
        "manager":       manager_name,
        "margin_pct":    margin_pct,
        "margin_source": margin_source,
        "report_date":   report_date,
        "zones":         zones,
        "total_loss":    total_loss,
        "red_loss":      red_loss,
        "yellow_loss":   yellow_loss,
        "total_real":    total_real,
        "red_real":      red_real,
        "yellow_real":   yellow_real,
        "total_clients": total_clients,
    }


# ── Форматирование сообщений ──────────────────────────────────────────────────

MAX_CLIENTS_PER_ZONE = 10  # Не показываем больше N клиентов в зоне

# Пояснения по зонам — для менеджеров
_ZONE_TIPS = {
    "red":    "",
    "yellow": "",
}

def format_opportunity_loss_message(data: Dict) -> str:
    """
    Форматирует сообщение об упущенной прибыли для менеджера.
    Включает: период, откуда данные, формулу, каждого клиента, ответственность.
    """
    mgr          = data["manager"]
    margin       = data["margin_pct"]
    margin_src   = data.get("margin_source", "")
    date_str     = data["report_date"]
    zones        = data["zones"]
    total_clients= data["total_clients"]

    # Заголовок
    lines = [
        f"💸 УПУЩЕННАЯ ПРИБЫЛЬ — {mgr}",
        f"📅 {date_str}" if date_str else "",
        f"Маржа: {_fmt_pct(margin)}  |  Договорной срок: 7 дней",
        f"Клиентов с просрочкой (8+ дней): {total_clients}",
    ]
    lines = [l for l in lines if l]
    lines.append("")

    def _render_zone(zone_clients: List[Dict], header: str, tip: str,
                     limit: int = MAX_CLIENTS_PER_ZONE) -> List[str]:
        if not zone_clients:
            return []
        result = [header, ""]
        for c in zone_clients[:limit]:
            turns = c.get("turns", 1.0)
            result.append(
                f"  • {c['client']}\n"
                f"    Долг: {_fmt_money(c['debt'])}  |  {c['days']} дн.\n"
                f"    Упущено: {_fmt_money(c['real_loss'])}"
            )
        extra = len(zone_clients) - limit
        if extra > 0:
            result.append(f"  ... и ещё {extra} клиент{'ов' if extra >= 5 else 'а'}")
        result.append("")
        return result

    lines += _render_zone(zones["red"],    "🔴 Красная зона (15–30 дней):",  _ZONE_TIPS["red"])
    lines += _render_zone(zones["yellow"], "⚡ Просрочка (8–15 дней):",      _ZONE_TIPS["yellow"])

    # Итоги
    lines += [
        "─" * 36,
        f"💰 Потери из-за просрочки:  {_fmt_money(data['total_real'])}",
        f"   (маржа {_fmt_pct(data['margin_pct'])} × оборачиваемость 2 раза/мес)",
        "",
    ]
    if data["red_real"] > 0:
        lines.append(f"   🔴 15–30 дней: {_fmt_money(data['red_real'])}")
    if data["yellow_real"] > 0:
        lines.append(f"   ⚡ 8–15 дней:  {_fmt_money(data['yellow_real'])}")

    lines += [
        "",
        "Свяжитесь с каждым клиентом из этого списка.",
        "Фиксируйте договорённости о сроках оплаты.",
    ]

    return "\n".join(lines)


def format_opportunity_loss_admin(all_data: List[Dict]) -> str:
    """
    Форматирует сводную таблицу по всем менеджерам для admin.
    Включает: период, формулу, топ-клиентов по каждому менеджеру.
    """
    if not all_data:
        return "💸 Упущенная прибыль: нет данных по молчащим должникам."

    sorted_data = sorted(all_data, key=lambda d: d["total_real"], reverse=True)

    # Период — берём из первого доступного
    periods = [d["report_date"] for d in sorted_data if d.get("report_date")]
    period_line = f"📅 Период: {periods[0]}" if periods else ""

    lines = [
        "💸 УПУЩЕННАЯ ПРИБЫЛЬ — СВОДКА",
        period_line,
        "",
        f"Расчёт: долг × маржа × обороты (2 раза/мес, период {TURNOVER_DAYS} дн.)",
        f"Договорной срок: 7 дней.  Зоны: ⚡ 8-15д / 🔴 15-30д",
        "",
    ]

    for d in sorted_data:
        mgr          = d["manager"]
        total_real   = d["total_real"]
        red_count    = len(d["zones"]["red"])
        yellow_count = len(d["zones"]["yellow"])
        margin       = d["margin_pct"]
        date_str     = d.get("report_date", "")

        icon = "🔴" if red_count > 0 else "⚡"

        lines.append(
            f"{icon} {mgr} — {_fmt_money(total_real)}"
            f"  (маржа {_fmt_pct(margin)} | 🔴 {red_count} ⚡ {yellow_count})"
        )
        if date_str:
            lines.append(f"   Отчёт за: {date_str}")

        # Топ-3 клиента
        all_clients = d["zones"]["red"] + d["zones"]["yellow"]
        top3 = sorted(all_clients, key=lambda c: c["debt"], reverse=True)[:3]
        for c in top3:
            lines.append(
                f"   • {c['client']} — долг {_fmt_money(c['debt'])}, "
                f"{c['days']} дн., упущено {_fmt_money(c['real_loss'])}"
            )
        lines.append("")

    now_total_real = sum(d["total_real"] for d in sorted_data)
    champion = sorted_data[0]  # уже отсортировано по total_real desc

    lines += [
        "─" * 36,
        f"🏆 Чемпион по заморозке: {champion['manager']} — {_fmt_money(champion['total_real'])}",
        "",
        f"💰 Потери из-за просрочки:  {_fmt_money(now_total_real)}",
        f"   ⚡ 8–15 дней:  {_fmt_money(sum(d['yellow_real'] for d in sorted_data))}",
        f"   🔴 15–30 дней: {_fmt_money(sum(d['red_real'] for d in sorted_data))}",
    ]

    return "\n".join(lines)


def format_opportunity_loss_subadmin(all_data: List[Dict], subadmin_name: str, subordinates: List[str]) -> str:
    """
    Форматирует сводку для subadmin (себя + подчинённых).
    """
    relevant = [d for d in all_data if d["manager"] in ([subadmin_name] + subordinates)]
    if not relevant:
        return ""
    return format_opportunity_loss_admin(relevant)
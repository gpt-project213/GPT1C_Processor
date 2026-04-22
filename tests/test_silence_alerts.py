#!/usr/bin/env python
# coding: utf-8
"""
Регрессионные тесты для bot/silence_alerts.py
Запуск: python -X utf8 tests/test_silence_alerts.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bot"))

PASS = "✅"
FAIL = "❌"
results = []


def check(name: str, ok: bool, detail: str = ""):
    icon = PASS if ok else FAIL
    msg = f"  {icon} {name}"
    if detail and not ok:
        msg += f"\n       > {detail}"
    print(msg)
    results.append((name, ok))


def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


from bot.silence_alerts import SilenceAlert


def _write(path: Path, text: str, ts: int) -> Path:
    path.write_text(text, encoding="utf-8")
    os.utime(path, (ts, ts))
    return path


def _debt_html(period_text: str, rows: list[tuple[str, str, str, str, str, str]]) -> str:
    body_rows = []
    for client, debt, opening, debit, paid, silence in rows:
        body_rows.append(
            f"<tr>"
            f"<td>{client}</td><td>{debt}</td><td>{opening}</td>"
            f"<td>{debit}</td><td>{paid}</td><td>1</td><td>{silence}</td>"
            f"</tr>"
        )
    body = "".join(body_rows)
    return f"""<!DOCTYPE html>
<html lang="ru">
<body>
  <div class="stat-label">Период</div><div class="stat-value">{period_text}</div>
  <div id="t_all">
    <table>
      <thead>
        <tr>
          <th>Клиент</th><th>Долг</th><th>Нач.остаток</th>
          <th>Отгрузка</th><th>Оплата</th><th>Операций</th><th>Дни молчания</th>
        </tr>
      </thead>
      <tbody>{body}</tbody>
    </table>
  </div>
</body>
</html>"""


section("1. parse_debt_amount")
alert = SilenceAlert()
check("T1a: '3 098 966,13' -> 3098966.13",
      abs(alert.parse_debt_amount("3 098 966,13") - 3098966.13) < 0.01)
check("T1b: em-dash -> 0.0", alert.parse_debt_amount("—") == 0.0)
check("T1c: None -> 0.0", alert.parse_debt_amount(None) == 0.0)
check("T1d: empty -> 0.0", alert.parse_debt_amount("") == 0.0)


section("2. parse_report_date")
_tmp = Path(tempfile.mkdtemp())
try:
    p1 = _write(_tmp / "small.html", "<small><span class='key'>Период:</span> 22 апреля 2026 г.</small>", 1)
    p2 = _write(_tmp / "stat.html", "<div class='stat-label'>Период</div><div class='stat-value'>01.04.2026 — 22.04.2026</div>", 2)
    p3 = _write(_tmp / "regex.html", "<div>Период: 21.04.2026</div>", 3)
    check("T2a: strategy small", alert.parse_report_date(p1) == "22 апреля 2026 г.")
    check("T2b: strategy stat-label/stat-value normalizes dash",
          alert.parse_report_date(p2) == "01.04.2026 - 22.04.2026",
          alert.parse_report_date(p2))
    check("T2c: regex fallback", alert.parse_report_date(p3) == "21.04.2026")
finally:
    shutil.rmtree(_tmp, ignore_errors=True)


section("3. categorize_by_silence")
clients = [
    {"client": "Critical", "debt": 100000.0, "silence_days": 35, "debit_amount": 0.0, "paid_amount": 0.0},
    {"client": "Alarm", "debt": 100000.0, "silence_days": 20, "debit_amount": 0.0, "paid_amount": 0.0},
    {"client": "Silence", "debt": 100000.0, "silence_days": 12, "debit_amount": 0.0, "paid_amount": 0.0},
    {"client": "Overdue", "debt": 100000.0, "silence_days": 8, "debit_amount": 0.0, "paid_amount": 0.0},
    {"client": "Partial", "debt": 100000.0, "silence_days": 3, "debit_amount": 0.0, "paid_amount": 15000.0},
    {"client": "OnStop", "debt": 100000.0, "silence_days": 4, "debit_amount": 0.0, "paid_amount": 5000.0},
]
cat = alert.categorize_by_silence(clients)
check("T3a: critical=1", len(cat["critical"]) == 1 and cat["critical"][0]["client"] == "Critical", str(cat["critical"]))
check("T3b: alarm=1", len(cat["alarm"]) == 1 and cat["alarm"][0]["client"] == "Alarm", str(cat["alarm"]))
check("T3c: silence=1", len(cat["silence"]) == 1 and cat["silence"][0]["client"] == "Silence", str(cat["silence"]))
check("T3d: overdue=1", len(cat["overdue"]) == 1 and cat["overdue"][0]["client"] == "Overdue", str(cat["overdue"]))
check("T3e: partial_payment=1", len(cat["partial_payment"]) == 1 and cat["partial_payment"][0]["client"] == "Partial", str(cat["partial_payment"]))
check("T3f: on_stop=1", len(cat["on_stop"]) == 1 and cat["on_stop"][0]["client"] == "OnStop", str(cat["on_stop"]))


section("4. weekly clients / min debt / imitation")
weekly_clients = [
    {"client": "Weekly 8d", "debt": 100000.0, "silence_days": 8, "debit_amount": 0.0, "paid_amount": 0.0},
    {"client": "Weekly 12d", "debt": 100000.0, "silence_days": 12, "debit_amount": 0.0, "paid_amount": 0.0},
]
cat_weekly = alert.categorize_by_silence(weekly_clients, weekly_clients=["Weekly 8d", "Weekly 12d"])
check("T4a: weekly client 8d excluded from overdue",
      all(c["client"] != "Weekly 8d" for c in cat_weekly["overdue"]),
      str(cat_weekly["overdue"]))
check("T4b: weekly client 12d still in silence",
      any(c["client"] == "Weekly 12d" for c in cat_weekly["silence"]),
      str(cat_weekly["silence"]))

small_debt = [{"client": "Small", "debt": 3000.0, "silence_days": 20, "debit_amount": 0.0, "paid_amount": 0.0}]
cat_small = alert.categorize_by_silence(small_debt)
check("T5: debt 3000 ignored in all categories",
      sum(len(v) for v in cat_small.values()) == 0,
      str(cat_small))

imitation = [{"client": "Imitation", "debt": 100000.0, "silence_days": 4, "debit_amount": 0.0, "paid_amount": 5000.0}]
cat_imit = alert.categorize_by_silence(imitation)
check("T8a: imitation goes to on_stop",
      len(cat_imit["on_stop"]) == 1 and cat_imit["on_stop"][0]["client"] == "Imitation",
      str(cat_imit["on_stop"]))
check("T8b: imitation gets is_imitation=True",
      bool(cat_imit["on_stop"][0].get("is_imitation")) is True,
      str(cat_imit["on_stop"][0]))


section("5. freshness / period sort")
_tmp = Path(tempfile.mkdtemp())
try:
    reports_dir = _tmp / "html"
    reports_dir.mkdir(parents=True, exist_ok=True)
    old_v = _write(
        reports_dir / "debt_ext_Ведомость_по_взаиморасчетам_с_контрагентами_X_01.04.html",
        _debt_html("01.04.2026", [("Client", "100000", "0", "0", "0", "7")]),
        100,
    )
    det_10 = _write(
        reports_dir / "debt_ext_Детальный_Дебиторы_X_10.04.html",
        _debt_html("10.04.2026", [("Client", "100000", "0", "0", "0", "7")]),
        200,
    )
    det_20 = _write(
        reports_dir / "debt_ext_Детальный Дебиторы X 20.04.html",
        _debt_html("20.04.2026", [("Client", "100000", "0", "0", "0", "7")]),
        300,
    )

    picked = alert.get_latest_debt_report(reports_dir, "X")
    check("T6a: latest debt report is detailed 20.04, not ledger",
          picked is not None and picked.name == det_20.name,
          str(picked))

    det_10.unlink()
    det_20.unlink()
    picked_none = alert.get_latest_debt_report(reports_dir, "X")
    check("T6b: if only ledger remains, returns None (ledger must not win)",
          picked_none is None,
          str(picked_none))

    key_range = SilenceAlert._period_sort_key(old_v, "01.04.2026-22.04.2026")
    key_ru = SilenceAlert._period_sort_key(old_v, "22 апреля 2026")
    key_bad = SilenceAlert._period_sort_key(old_v, "мусор")
    check("T7a: range period key ends with 22.04.2026", key_range == (2026, 4, 22), str(key_range))
    check("T7b: ru month key ends with 22.04.2026", key_ru == (2026, 4, 22), str(key_ru))
    check("T7c: bad period falls back to (0,0,mtime)", key_bad == (0, 0, 100.0), str(key_bad))
finally:
    shutil.rmtree(_tmp, ignore_errors=True)


print("\n" + "=" * 60)
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"{PASS} Пройдено: {passed}/{total}")
if passed != total:
    sys.exit(1)

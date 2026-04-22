#!/usr/bin/env python
# coding: utf-8
"""
Регрессионные тесты для bot/opportunity_loss.py
Запуск: python -X utf8 tests/test_opportunity_loss.py
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


from bot.opportunity_loss import (
    DEFAULT_MARGIN_PCT,
    TURNOVER_DAYS,
    ZONE_RED_MIN,
    ZONE_YELLOW_MIN,
    _find_latest_gross_html,
    _get_manager_margin,
    calculate_opportunity_loss,
)


def _write(path: Path, text: str, ts: int) -> Path:
    path.write_text(text, encoding="utf-8")
    os.utime(path, (ts, ts))
    return path


def _gross_html(period_text: str, margin: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="ru">
<body>
  <small><span class="key">Период:</span> {period_text}</small>
  <div class="summary">
    <div class="kv">
      <div><span class="k">Выручка</span><span class="v">100 000</span></div>
      <div><span class="k">Себестоимость</span><span class="v">90 000</span></div>
      <div><span class="k">Валовая прибыль</span><span class="v">10 000</span></div>
      <div><span class="k">Рентабельность</span><span class="v">{margin}</span></div>
    </div>
  </div>
</body>
</html>"""


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


section("1. gross file selection")
_tmp = Path(tempfile.mkdtemp())
try:
    html_dir = _tmp / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    _write(html_dir / "Чужой_1_gross_sum.html", _gross_html("20 апреля 2026 г.", "15,0 %"), 100)
    _write(html_dir / "Чужой_2_gross_sum.html", _gross_html("21 апреля 2026 г.", "20,0 %"), 200)
    picked_none = _find_latest_gross_html(html_dir, "Алена")
    check("T1: only foreign gross files -> None", picked_none is None, str(picked_none))

    own = _write(html_dir / "Алена_gross_sum.html", _gross_html("22 апреля 2026 г.", "12,5 %"), 300)
    picked_own = _find_latest_gross_html(html_dir, "Алена")
    check("T2: own gross file wins among foreign ones",
          picked_own is not None and picked_own.name == own.name,
          str(picked_own))
finally:
    shutil.rmtree(_tmp, ignore_errors=True)


section("2. manager margin")
_tmp = Path(tempfile.mkdtemp())
try:
    html_dir = _tmp / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    margin, source = _get_manager_margin(html_dir, "Алена")
    check("T3: no gross file -> default margin",
          margin == DEFAULT_MARGIN_PCT and "по умолчанию" in source,
          f"{margin} | {source}")
finally:
    shutil.rmtree(_tmp, ignore_errors=True)


section("3. calculate_opportunity_loss")
_tmp = Path(tempfile.mkdtemp())
try:
    html_dir = _tmp / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    _write(
        html_dir / "debt_ext_Детальный Дебиторы Алена 22.04.html",
        _debt_html(
            "22.04.2026",
            [
                ("Red Client", "50000", "0", "0", "0", "20"),
                ("Yellow Client", "100000", "0", "0", "0", "10"),
                ("Too Fresh", "30000", "0", "0", "0", "5"),
                ("Too Small", "3000", "0", "0", "0", "10"),
            ],
        ),
        300,
    )
    _write(
        html_dir / "Алена_gross_sum.html",
        _gross_html("22 апреля 2026 г.", "10,0 %"),
        400,
    )

    data = calculate_opportunity_loss(html_dir, "Алена")
    check("T4a: calculate_opportunity_loss returns data", data is not None)
    check("T4b: red zone has 1 client", len(data["zones"]["red"]) == 1, str(data["zones"]["red"]))
    check("T4c: yellow zone has 1 client", len(data["zones"]["yellow"]) == 1, str(data["zones"]["yellow"]))
    check("T4d: 5-day client filtered out", all(c["client"] != "Too Fresh" for c in data["zones"]["yellow"] + data["zones"]["red"]))
    check("T5: debt 3000 filtered out", all(c["client"] != "Too Small" for c in data["zones"]["yellow"] + data["zones"]["red"]))

    yellow = data["zones"]["yellow"][0]
    red = data["zones"]["red"][0]
    check("T6a: turns for 10 days -> 1.0", abs(yellow["turns"] - 1.0) < 0.001, str(yellow))
    check("T6b: turns for 20 days -> 20/15", abs(red["turns"] - (20 / TURNOVER_DAYS)) < 0.001, str(red))
finally:
    shutil.rmtree(_tmp, ignore_errors=True)


print("\n" + "=" * 60)
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"{PASS} Пройдено: {passed}/{total}")
if passed != total:
    sys.exit(1)

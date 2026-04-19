from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


ROOT = Path(__file__).resolve().parents[1]


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(ROOT / "templates"),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["money"] = lambda value: f"{float(value or 0):,.2f}".replace(",", " ").replace(".", ",")
    env.filters["qty"] = lambda value: str(value)
    env.filters["price"] = lambda value: str(value)
    env.filters["pct"] = lambda value: str(value)
    return env


def test_debt_template_renders_mobile_cards() -> None:
    env = _env()
    row = {
        "client": "\u0414\u043b\u0438\u043d\u043d\u043e\u0435 \u0438\u043c\u044f \u043a\u043b\u0438\u0435\u043d\u0442\u0430 \u0415\u0440\u0433\u0430\u043b\u0438 \u0422\u041e\u041e",
        "client_slug": "client-long",
        "debt": 1234567.89,
        "pct": 12.3,
        "days": 44,
        "ship": 200000,
        "pay": 150000,
        "overpay": -5000,
        "opening": 1000,
        "debit": 2000,
        "credit": 1500,
        "movements": 2,
        "days_silence": 9,
    }
    client = dict(
        row,
        sum_debit=2000,
        sum_credit=1500,
        closing=1500,
        movements=[{"date": datetime(2026, 4, 18), "debit": 2000, "credit": 0}],
    )

    html = env.get_template("debt_auto.html").render(
        title="\u0414\u0435\u0431\u0438\u0442\u043e\u0440\u043a\u0430 \u0415\u0440\u0433\u0430\u043b\u0438",
        period="01.04.2026-18.04.2026",
        manager="\u0415\u0440\u0433\u0430\u043b\u0438",
        client_count=1,
        total_debt=1234567.89,
        report_type="extended",
        top_debtors=[row],
        silent_rows=[row],
        closed_rows=[row],
        overpay_rows=[row],
        all_rows=[row],
        by_subgroup={"\u0413\u0440\u0443\u043f\u043f\u0430": [client]},
        tech_info={"file": "long_file_name_without_spaces.xlsx"},
        generated="2026-04-19 06:30",
        open_total=1000,
        debit_total=2000,
        credit_total=1500,
        delta_label="\u0423\u0432\u0435\u043b\u0438\u0447\u0435\u043d\u0438\u0435",
        delta_abs=500,
    )

    assert '<table class="report-table"' in html
    assert '@media(max-width:640px)' in html
    assert 'content:attr(data-label)' in html
    assert 'data-label="&#1050;&#1083;&#1080;&#1077;&#1085;&#1090;"' in html
    assert 'data-label="&#1044;&#1086;&#1083;&#1075;, &#8376;"' in html
    assert "movement-stats" in html
    assert "overflow-wrap:anywhere" in html


def test_common_report_templates_keep_mobile_overflow_controls() -> None:
    env = _env()
    product = {
        "category": "\u0413\u0440\u0443\u043f\u043f\u0430",
        "product": "LongProductNameWithoutSpacesForMobileOverflowCheck",
        "qty_raw": 1,
        "price_raw": 100,
        "sum_raw": 100,
        "sale_raw": 100,
        "cost_raw": 70,
        "profit_raw": 30,
        "margin_raw": 30,
    }
    context = {
        "title": "Test",
        "period": "Period",
        "manager": "Manager",
        "generated": "2026-04-19",
        "generated_at": "2026-04-19",
        "summary": '<div class="table-wrap"><table><tr><td>LongSummary</td></tr></table></div>',
        "abc_table": '<div class="table-wrap"><table><tr><td>LongABC</td></tr></table></div>',
        "loss_table": '<div class="table-wrap"><table><tr><td>LongLoss</td></tr></table></div>',
        "top10_profit": '<div class="table-wrap"><table><tr><td>LongProfit</td></tr></table></div>',
        "tech_block": "tech",
        "file_name": "long.xlsx",
        "products": [product],
        "rows": [product],
        "groups": [{"client": "Client", "products": [product], "total_sum": 100}],
        "grand_total_qty": "1",
        "grand_total_sum": "100",
    }

    for template_name in [
        "gross.html",
        "gross_percent.html",
        "sales_report_grouped.html",
        "sales_by_product.html",
        "inventory_simple.html",
        "base.html",
    ]:
        html = env.get_template(template_name).render(**context)
        assert "-webkit-overflow-scrolling:touch" in html, template_name
        assert "overflow-wrap:anywhere" in html, template_name


def test_telegram_admin_control_commands_are_registered() -> None:
    send_reports = (ROOT / "bot" / "send_reports.py").read_text(encoding="utf-8")
    watchdog = (ROOT / "start_bot_watchdog.bat").read_text(encoding="utf-8")

    assert 'CommandHandler("restart", cmd_restart)' in send_reports
    assert 'CommandHandler("shutdown", cmd_shutdown)' in send_reports
    assert "if chat_id != ADMIN_CHAT_ID:" in send_reports
    assert "if STOP_FILE.exists():" in send_reports
    assert "Bot startup cancelled" in send_reports
    assert "STOP_FILE=logs\\bot.stop" in watchdog
    assert 'if exist "%STOP_FILE%"' in watchdog

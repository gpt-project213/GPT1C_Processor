#!/usr/bin/env python
# coding: utf-8
"""
tests/test_parsers.py — интеграционные тесты парсеров на реальных xlsx
Запуск: python -X utf8 tests/test_parsers.py
"""
import sys, os, json, re as _re, tempfile
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bot"))

PROCESSED = ROOT / "reports" / "excel" / "processed"
PASS, FAIL, SKIP = "✅", "❌", "⏭️"
results = []

def check(name, ok, detail=""):
    msg = f"  {PASS if ok else FAIL} {name}"
    if detail and not ok: msg += f"\n       > {detail}"
    print(msg); results.append((name, ok))

def skip(name, reason=""): print(f"  {SKIP} {name}  [{reason}]")

def section(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")

def find_xlsx(*keywords):
    for kw in keywords:
        for f in sorted(PROCESSED.glob("*.xlsx"), reverse=True):
            if kw.lower() in f.name.lower(): return f
    return None

def read_json(p):
    if p is None: return None
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except: return None

def is_pos(v):
    try: return float(v) >= 0
    except: return False

def is_str(v): return isinstance(v, str) and len(str(v).strip()) > 0

# ═══════════════════════════════════════════════════════════════
# 1. gross_parser — build_gross_json(xlsx: Path) -> Path(JSON)
#    JSON структура: {source_file, report_type, period, total_revenue,
#                     total_cost, gross_profit, margin_pct, product_count,
#                     products: [{product, revenue, cost, profit, margin_pct}],
#                     metadata}
# ═══════════════════════════════════════════════════════════════
section("1. gross_parser — Валовая прибыль")

f = find_xlsx("Валовая прибыль Ергали", "Валовая прибыль Алена", "Валовая прибыль")
if not f:
    skip("gross_parser", "xlsx не найден в processed/")
else:
    print(f"  Файл: {f.name}")
    try:
        from gross_parser import build_gross_json
        json_path = build_gross_json(Path(f))
        check("gross_parser — возвращает Path", isinstance(json_path, Path))
        check("gross_parser — JSON создан", json_path is not None and json_path.exists())
        data = read_json(json_path)
        check("gross_parser — JSON валиден", data is not None)
        if data:
            for k in ["period", "total_revenue", "gross_profit", "products"]:
                check(f"gross_parser — ключ '{k}'", k in data, f"keys={list(data.keys())}")
            products = data.get("products", [])
            check("gross_parser — products не пустые", len(products) > 0)
            if products:
                p0 = products[0]
                for k in ["product", "revenue", "cost", "profit"]:
                    check(f"gross_parser — product.{k}", k in p0, f"keys={list(p0.keys())}")
                check("gross_parser — revenue >= 0", is_pos(p0.get("revenue")))
            check("gross_parser — total_revenue >= 0", is_pos(data.get("total_revenue")))
            check("gross_parser — product_count > 0",
                  (data.get("product_count", 0) or 0) > 0)
    except Exception as e:
        check("gross_parser — без исключений", False, str(e)[:120])

# ═══════════════════════════════════════════════════════════════
# 2. sales_parser — parse_file(xlsx) -> Path(JSON)
#    JSON структура: {source_file, report_type, period, manager,
#                     total_revenue, client_count,
#                     clients: [{client, total, products:[]}],
#                     metadata}
# ═══════════════════════════════════════════════════════════════
section("2. sales_parser — Продажи")

f = find_xlsx("Продажи Ергали", "Продажи Алена", "Продажи")
if not f:
    skip("sales_parser", "xlsx не найден")
else:
    print(f"  Файл: {f.name}")
    try:
        from sales_parser import parse_file
        with tempfile.TemporaryDirectory() as td:
            json_path = parse_file(f, out_dir=Path(td))
            check("sales_parser — возвращает Path или None",
                  json_path is None or isinstance(json_path, Path))
            if json_path and json_path.exists():
                data = read_json(json_path)
                check("sales_parser — JSON валиден", data is not None)
                if data:
                    for k in ["manager", "period", "clients", "total_revenue"]:
                        check(f"sales_parser — ключ '{k}'", k in data,
                              f"keys={list(data.keys())}")
                    clients = data.get("clients", [])
                    check("sales_parser — clients не пустые", len(clients) > 0)
                    if clients:
                        c = clients[0]
                        check("sales_parser — client.client непустой", is_str(c.get("client")))
                        # Поле называется 'total', не 'revenue'
                        check("sales_parser — client.total >= 0", is_pos(c.get("total", 0)))
                    check("sales_parser — total_revenue >= 0", is_pos(data.get("total_revenue", 0)))
            else:
                skip("sales_parser json", "parse_file вернул None")
    except Exception as e:
        check("sales_parser — без исключений", False, str(e)[:120])

# ═══════════════════════════════════════════════════════════════
# 3. analyze_debt_excel — parse_debt_report
#    Возвращает tuple(DataFrame, list[(client_name, amount)])
#    Файлы требуют ensure_clean_xlsx перед парсингом
# ═══════════════════════════════════════════════════════════════
section("3. analyze_debt_excel — Детальный Дебиторы")

f = find_xlsx("Детальный Дебиторы Ергали", "Детальный Дебиторы")
if not f:
    skip("debt_parser", "xlsx не найден")
else:
    print(f"  Файл: {f.name}")
    try:
        from utils_excel import ensure_clean_xlsx
        from analyze_debt_excel import parse_debt_report
        import pandas as pd

        clean = ensure_clean_xlsx(f, force_fix=True)
        check("debt — ensure_clean_xlsx создаёт файл",
              clean is not None and clean.exists())

        if clean and clean.exists():
            result = parse_debt_report(str(clean))
            # Возвращает tuple(DataFrame, list)
            check("debt_parser — возвращает tuple", isinstance(result, tuple),
                  f"got {type(result)}")
            if isinstance(result, tuple) and len(result) == 2:
                df, closing_list = result
                check("debt_parser — df является DataFrame",
                      isinstance(df, pd.DataFrame))
                check("debt_parser — DataFrame не пустой",
                      len(df) > 0, f"rows={len(df)}")
                check("debt_parser — closing_list является list",
                      isinstance(closing_list, list))
                if len(df) > 0:
                    # ИНВАРИАНТ ПРОЕКТА: ключ 'debt' используется в debt_auto_report
                    # parse_debt_report возвращает сырой DataFrame — ключ проверяется
                    # уже в debt_auto_report который строит структуру с 'debt'
                    check("debt_parser — DataFrame содержит числовые колонки",
                          len(df.select_dtypes(include='number').columns) > 0)
    except Exception as e:
        check("debt_parser — без исключений", False, str(e)[:120])

# ═══════════════════════════════════════════════════════════════
# 4. inventory_cost_parser — build_inventory_cost_report(xlsx) -> {json: Path, html: Path}
# ═══════════════════════════════════════════════════════════════
section("4. inventory_cost_parser — Ведомость по партиям")

f = find_xlsx("партиям_товаров", "партиям товаров", "партиям")
if not f:
    skip("inv_cost_parser", "xlsx не найден")
else:
    print(f"  Файл: {f.name}")
    try:
        from inventory_cost_parser import build_inventory_cost_report
        result = build_inventory_cost_report(Path(f))
        check("inv_cost_parser — возвращает dict", isinstance(result, dict))
        if isinstance(result, dict):
            check("inv_cost_parser — содержит 'json'", "json" in result)
            check("inv_cost_parser — содержит 'html'", "html" in result)
            for k in ("json", "html"):
                if k in result:
                    p = result[k]
                    check(f"inv_cost_parser — {k} файл существует",
                          isinstance(p, Path) and p.exists(), f"{k}={p}")
            # Проверяем содержимое JSON
            if "json" in result and result["json"].exists():
                data = read_json(result["json"])
                check("inv_cost_parser — JSON валиден", data is not None)
                if data:
                    check("inv_cost_parser — содержит 'period'", "period" in data,
                          f"keys={list(data.keys())[:8]}")
    except Exception as e:
        check("inv_cost_parser — без исключений", False, str(e)[:120])

# ═══════════════════════════════════════════════════════════════
# 5. inventory.py — build_report(xlsx: Path) -> Path
# ═══════════════════════════════════════════════════════════════
section("5. inventory.py — Ведомость по товарам")

f = find_xlsx("Ведомость по товарам на складах")
if not f:
    skip("inventory.build_report", "xlsx не найден")
else:
    print(f"  Файл: {f.name}")
    try:
        import inventory as inv_mod
        result = inv_mod.build_report(Path(f))
        check("inventory.build_report — не падает", True)
        check("inventory.build_report — возвращает Path",
              isinstance(result, Path), f"got {type(result)}")
        if isinstance(result, Path):
            check("inventory.build_report — HTML файл создан", result.exists())
    except Exception as e:
        check("inventory.build_report — без исключений", False, str(e)[:120])

# ═══════════════════════════════════════════════════════════════
# 6. expenses_parser — требует ensure_clean_xlsx
# ═══════════════════════════════════════════════════════════════
section("6. expenses_parser — Затраты")

f = find_xlsx("Затрат", "затрат", "расход", "expense")
if not f:
    skip("expenses_parser", "xlsx расходов не найден — файлы не поступали (нормально)")
else:
    print(f"  Файл: {f.name}")
    try:
        from utils_excel import ensure_clean_xlsx
        from expenses_parser import parse_file as parse_expenses

        clean = ensure_clean_xlsx(f, force_fix=True)
        check("expenses — ensure_clean_xlsx создаёт файл",
              clean is not None and clean.exists())

        if clean and clean.exists():
            result = parse_expenses(clean)
            check("expenses_parser — не падает", True)
            check("expenses_parser — возвращает Path или dict",
                  isinstance(result, (Path, dict)))
            if isinstance(result, Path) and result.exists():
                data = read_json(result)
                check("expenses_parser — JSON валиден", data is not None)
                if data:
                    check("expenses_parser — содержит числовые данные",
                          any(isinstance(v, (int, float)) for v in data.values() if not isinstance(v, dict)),
                          f"keys={list(data.keys())[:8]}")
    except Exception as e:
        check("expenses_parser — без исключений", False, str(e)[:120])

# ═══════════════════════════════════════════════════════════════
# 7. Устойчивость — несуществующий файл
# ═══════════════════════════════════════════════════════════════
section("7. Устойчивость — несуществующий файл")

from analyze_debt_excel import parse_debt_report
from gross_parser import build_gross_json

for name, fn, arg in [
    ("debt_parser(nonexistent)",  parse_debt_report,  "/nonexistent/x.xlsx"),
    ("gross_parser(nonexistent)", build_gross_json,   Path("/nonexistent/x.xlsx")),
]:
    try:
        fn(arg); check(f"{name} — без краша", True)
    except (FileNotFoundError, OSError, ValueError, KeyError):
        check(f"{name} — ожидаемое исключение", True)
    except Exception as e:
        check(f"{name} — неожиданное исключение", False,
              f"{type(e).__name__}: {str(e)[:80]}")

# ═══════════════════════════════════════════════════════════════
# 8. send_reports helpers
# ═══════════════════════════════════════════════════════════════
section("8. send_reports.py — helper-функции")

src = (ROOT / "bot" / "send_reports.py").read_text(encoding="utf-8")

# normalize_manager_name — strips, replaces Ё→Е, collapses spaces, preserves case
m = _re.search(r"def normalize_manager_name\([^)]*\)[^:]*:\n((?:    [^\n]+\n)+)", src)
if m:
    env = {"re": _re, "__builtins__": __builtins__}
    exec("def normalize_manager_name(name: str) -> str:\n" + m.group(1), env)
    norm = env["normalize_manager_name"]
    for inp, expected in [
        ("Ергали",   "Ергали"),    # не приводит к нижнему регистру
        (" Алена  ", "Алена"),     # стрипает пробелы
        ("МагиЁра",  "МагиЕра"),  # Ё→Е
        ("",         ""),
    ]:
        got = norm(inp)
        check(f"normalize_manager_name({inp!r})", got == expected, f"got={got!r}")
else:
    skip("normalize_manager_name", "не найдена")

# _classify_type — извлекаем полное тело функции (включая пустые строки внутри)
m2_start = _re.search(r"^def _classify_type\b", src, _re.MULTILINE)
if m2_start:
    after = src[m2_start.start():]
    # Найти следующую top-level def/class
    m2_end = _re.search(r"^(?:def |class |\Z)", after[1:], _re.MULTILINE)
    func_src = after[:m2_end.start() + 1] if m2_end else after
    env2 = {"re": _re, "Optional": Optional, "Path": Path, "__builtins__": __builtins__}
    try:
        exec(func_src, env2)
        clf = env2["_classify_type"]
        # Ожидаемые значения соответствуют РЕАЛЬНЫМ константам функции
        for fname, expected in [
            ("sales_grouped_Ергали_20260312.html",        "SALES_SIMPLE"),
            ("sales_products_Алена_20260312.html",        "SALES_EXTENDED"),
            ("gross_Ергали_gross.html",                   "GROSS_SUM"),   # _gross.html → GROSS_SUM
            ("gross_Ергали_gross_pct.html",               "GROSS_PCT"),   # _gross_pct.html → GROSS_PCT
            ("debt_ext_Детальный_Дебиторы_Магира.html",   "DEBT_EXTENDED"),
            ("inventory_simple_Склад_20260312.html",      "INVENTORY_SIMPLE"),
            ("expenses_Расход_20260312.html",             "EXPENSES"),
            ("random_file.html",                          "UNKNOWN"),
        ]:
            got = clf(fname)
            check(f"_classify_type({fname})", got == expected,
                  f"got={got!r} expected={expected!r}")
    except Exception as e:
        check("_classify_type — выполнение", False, str(e)[:80])
else:
    skip("_classify_type", "не найдена")

# ═══════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
total  = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed
print(f"  ИТОГ: {passed}/{total} тестов прошло  |  {failed} упало")
print(f"{'═'*60}")
if failed:
    print("\nУпавшие:")
    for name, ok in results:
        if not ok: print(f"  {FAIL} {name}")
    sys.exit(1)
else:
    print(f"\n  {PASS} Все тесты прошли!")
    sys.exit(0)

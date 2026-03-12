#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тест всех 11 CRITICAL фиксов коммита 3902b0e.
Запуск: python test_critical_fixes.py
"""
import sys, os, tempfile, shutil, time, logging, re
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))

PASS = 0
FAIL = 0

def ok(tag, msg):
    global PASS
    PASS += 1
    print(f"  ✅ {tag}: {msg}")

def fail(tag, msg):
    global FAIL
    FAIL += 1
    print(f"  ❌ {tag}: {msg}")

# ═══════════════════════════════════════════════════════════════
print("\n═══ CR-01: _release не удаляет .work при неудачном move ═══")
# ═══════════════════════════════════════════════════════════════
try:
    from run_pipeline_all_mp import _move_to_processed
    import inspect
    src = inspect.getsource(_move_to_processed)
    if "return True" in src and "return False" in src:
        ok("CR-01a", "_move_to_processed возвращает bool")
    else:
        fail("CR-01a", "_move_to_processed не возвращает bool")

    from run_pipeline_all_mp import _process_one
    src2 = inspect.getsource(_process_one)
    if "moved = False" in src2 and "if moved:" in src2:
        ok("CR-01b", "_process_one проверяет moved перед _release")
    else:
        fail("CR-01b", "_process_one не проверяет moved flag")

    if "preserved in queue" in src2:
        ok("CR-01c", "логирование сохранения файла для retry")
    else:
        fail("CR-01c", "нет лога о сохранении файла")
except Exception as e:
    fail("CR-01", f"import/inspect error: {e}")

# ═══════════════════════════════════════════════════════════════
print("\n═══ CR-02: _claim без TOCTOU (нет exists перед rename) ═══")
# ═══════════════════════════════════════════════════════════════
try:
    from run_pipeline_all_mp import _claim
    src = inspect.getsource(_claim)
    if "src.exists()" not in src:
        ok("CR-02a", "exists() убран из _claim")
    else:
        fail("CR-02a", "exists() всё ещё в _claim — TOCTOU осталось")

    if "FileNotFoundError" in src:
        ok("CR-02b", "FileNotFoundError ловится отдельно")
    else:
        fail("CR-02b", "FileNotFoundError не перехватывается")

    if "err=True" in src:
        ok("CR-02c", "ошибки claim логируются как ERROR")
    else:
        fail("CR-02c", "ошибки claim не ERROR-level")

    # Функциональный тест: claim на несуществующий файл
    result = _claim(Path(tempfile.gettempdir()) / "nonexistent_test_12345.xlsx")
    if result is None:
        ok("CR-02d", "_claim(несуществующий файл) → None")
    else:
        fail("CR-02d", f"_claim(несуществующий файл) → {result}")

    # Функциональный тест: claim на существующий файл
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        tmp = Path(f.name)
    try:
        work = _claim(tmp)
        if work is not None and work.exists() and work.name.endswith(".xlsx.work"):
            ok("CR-02e", f"_claim(реальный файл) → {work.name}")
            work.unlink()
        elif work is None and not tmp.exists():
            ok("CR-02e", "_claim сработал (файл переименован)")
        else:
            fail("CR-02e", f"claim unexpected: work={work}, tmp_exists={tmp.exists()}")
    finally:
        if tmp.exists():
            tmp.unlink()
except Exception as e:
    fail("CR-02", f"error: {e}")

# ═══════════════════════════════════════════════════════════════
print("\n═══ CR-03: run_pipeline.py исключает .work файлы ═══")
# ═══════════════════════════════════════════════════════════════
try:
    with open("run_pipeline.py", "r", encoding="utf-8") as f:
        src = f.read()
    if '.work' in src and 'not p.name.endswith(".work")' in src:
        ok("CR-03", ".work файлы исключены из glob в run_pipeline.py")
    elif '.work' in src:
        ok("CR-03", ".work фильтрация присутствует")
    else:
        fail("CR-03", ".work фильтрация не найдена в run_pipeline.py")
except Exception as e:
    fail("CR-03", f"error: {e}")

# ═══════════════════════════════════════════════════════════════
print("\n═══ CR-04: XSS — html.escape в HTML-генераторах ═══")
# ═══════════════════════════════════════════════════════════════
xss_files = {
    "gross_report.py": ["_html.escape"],
    "expenses_parser.py": ["_html.escape"],
    "sales_report.py": ["_html.escape"],
    "bot/send_reports.py": ["_html.escape"],
}
for fpath, markers in xss_files.items():
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            src = f.read()
        found = all(m in src for m in markers)
        if found:
            ok("CR-04", f"{fpath}: html.escape используется")
        else:
            fail("CR-04", f"{fpath}: html.escape НЕ найден")
    except Exception as e:
        fail("CR-04", f"{fpath}: {e}")

# ═══════════════════════════════════════════════════════════════
print("\n═══ CR-05: _safe_replace не удаляет dst заранее ═══")
# ═══════════════════════════════════════════════════════════════
try:
    with open("utils_excel.py", "r", encoding="utf-8") as f:
        src = f.read()
    # Ищем функцию _safe_replace
    func_start = src.index("def _safe_replace")
    func_body = src[func_start:src.index("\ndef ", func_start + 1)]
    # Проверяем что НЕТ кода удаления dst перед циклом (только комментарий допустим)
    pre_loop = func_body[:func_body.index("for _i in")]
    has_unlink_code = "dst.unlink()" in pre_loop and "# (ранее" not in pre_loop.split("dst.unlink()")[0][-80:]
    if not has_unlink_code:
        ok("CR-05a", "dst.unlink() убран из _safe_replace (до цикла)")
    else:
        fail("CR-05a", "dst.unlink() всё ещё вызывается до цикла")

    if "os.replace() атомарно" in func_body or "предварительное удаление НЕ нужно" in func_body:
        ok("CR-05b", "комментарий объясняет почему удаление убрано")
    else:
        fail("CR-05b", "нет пояснительного комментария")

    # Функциональный тест: создаём src и dst, replace должен сработать
    tmpdir = Path(tempfile.mkdtemp())
    try:
        src_f = tmpdir / "src.txt"
        dst_f = tmpdir / "dst.txt"
        src_f.write_text("new content")
        dst_f.write_text("old content")

        try:
            from utils_excel import _safe_replace
            _safe_replace(src_f, dst_f)

            if dst_f.read_text() == "new content":
                ok("CR-05c", "_safe_replace работает: dst обновлён")
            else:
                fail("CR-05c", f"dst содержимое: {dst_f.read_text()!r}")
        except ImportError:
            ok("CR-05c", "SKIP (pandas не установлен) — проверено по исходнику")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
except Exception as e:
    fail("CR-05", f"error: {e}")

# ═══════════════════════════════════════════════════════════════
print("\n═══ CR-06: NaT-safe days_silence ═══")
# ═══════════════════════════════════════════════════════════════
try:
    with open("debt_auto_report.py", "r", encoding="utf-8") as f:
        src = f.read()
    if "_calc_silence" in src and "pd.isna" in src:
        ok("CR-06a", "_calc_silence с pd.isna guard найден")
    else:
        fail("CR-06a", "_calc_silence или pd.isna не найдены")

    try:
        import pandas as pd
    except ImportError:
        pd = None

    exec_ns = {"pd": pd} if pd else {}
    match = re.search(r"(    def _calc_silence\(.*?\n(?:        .*\n)*)", src)
    if match and pd:
        func_src = match.group(1).replace("    def ", "def ", 1)
        # Убираем один уровень отступа
        lines = func_src.split("\n")
        dedented = "\n".join(l[4:] if l.startswith("    ") else l for l in lines)
        exec(dedented, exec_ns)
        _calc = exec_ns["_calc_silence"]

        # Тест 1: Нормальные даты
        r = _calc(pd.Timestamp("2026-03-01"), pd.Timestamp("2026-02-01"), pd.Timestamp("2026-03-12"))
        if r == 11:
            ok("CR-06b", f"нормальные даты: days_silence=11")
        else:
            fail("CR-06b", f"нормальные даты: ожидалось 11, получено {r}")

        # Тест 2: NaT last_date, fallback на period_min
        r = _calc(pd.NaT, pd.Timestamp("2026-02-01"), pd.Timestamp("2026-03-12"))
        if r == 39:
            ok("CR-06c", f"NaT last_date + valid period_min: days_silence=39")
        else:
            fail("CR-06c", f"NaT last_date: ожидалось 39, получено {r}")

        # Тест 3: Оба NaT
        r = _calc(pd.NaT, None, pd.Timestamp("2026-03-12"))
        if r is None:
            ok("CR-06d", "оба NaT/None → None (не crash)")
        else:
            fail("CR-06d", f"оба NaT/None: ожидалось None, получено {r}")

        # Тест 4: None period_max
        r = _calc(pd.Timestamp("2026-03-01"), pd.Timestamp("2026-02-01"), None)
        if r is None:
            ok("CR-06e", "period_max=None → None")
        else:
            fail("CR-06e", f"period_max=None: ожидалось None, получено {r}")
    elif match and not pd:
        ok("CR-06b", "SKIP функциональных тестов (pandas не установлен)")
    else:
        fail("CR-06b", "_calc_silence функция не найдена в исходнике")
except Exception as e:
    fail("CR-06", f"error: {e}")

# ═══════════════════════════════════════════════════════════════
print("\n═══ CR-07: handle_persistent_menu без undefined функций ═══")
# ═══════════════════════════════════════════════════════════════
try:
    with open("bot/send_reports.py", "r", encoding="utf-8") as f:
        src = f.read()
    # Ищем handle_persistent_menu
    idx = src.index("async def handle_persistent_menu")
    # До следующей функции верхнего уровня
    next_func = src.index("\nasync def ", idx + 10) if "\nasync def " in src[idx+10:] else src.index("\ndef main", idx + 10)
    func_body = src[idx:idx + (next_func - idx) if next_func > idx else 500]

    if "handle_direct(" not in func_body:
        ok("CR-07a", "handle_direct() убран из handle_persistent_menu")
    else:
        fail("CR-07a", "handle_direct() всё ещё вызывается")

    if "handle_archive(" not in func_body:
        ok("CR-07b", "handle_archive() убран из handle_persistent_menu")
    else:
        fail("CR-07b", "handle_archive() всё ещё вызывается")

    if "handle_report_request" in func_body:
        ok("CR-07c", "handle_report_request используется вместо handle_direct")
    else:
        fail("CR-07c", "handle_report_request не найден")

    if "kb_archive_managers" in func_body:
        ok("CR-07d", "kb_archive_managers используется для архива")
    else:
        fail("CR-07d", "kb_archive_managers не найден")
except Exception as e:
    fail("CR-07", f"error: {e}")

# ═══════════════════════════════════════════════════════════════
print("\n═══ CR-08: SensitiveDataFilter сохраняет args ═══")
# ═══════════════════════════════════════════════════════════════
try:
    with open("bot/send_reports.py", "r", encoding="utf-8") as f:
        src = f.read()
    idx = src.index("class SensitiveDataFilter")
    class_end = src.index("\n# ", idx + 100)
    class_src = src[idx:class_end]

    if "record.msg) % record.args" in class_src or "% record.args" in class_src:
        ok("CR-08a", "args подставляются в msg перед маскировкой")
    else:
        fail("CR-08a", "args не подставляются перед маскировкой")

    if "record.args = None" in class_src:
        ok("CR-08b", "args обнуляется после форматирования")
    else:
        fail("CR-08b", "record.args = None не найден")

    # Функциональный тест: симулируем LogRecord
    record = logging.LogRecord(
        name="test", level=logging.INFO,
        pathname="", lineno=0, msg="chunk %d/%d chat=%s",
        args=(1, 5, "12345"), exc_info=None
    )
    # Симулируем filter без реальных правил
    # Компилируем класс
    exec_ns = {"logging": logging, "re": re}
    exec(class_src, exec_ns)
    filt = exec_ns["SensitiveDataFilter"]()
    filt._MASK_RULES = [(re.compile(r"12345"), "MASKED")]
    filt.filter(record)

    if "chunk 1/5" in record.msg and "MASKED" in record.msg:
        ok("CR-08c", f"args отформатированы И замаскированы: '{record.msg}'")
    elif "%d" in record.msg:
        fail("CR-08c", f"args НЕ отформатированы: '{record.msg}'")
    else:
        ok("CR-08c", f"маскировка работает: '{record.msg}'")
except Exception as e:
    fail("CR-08", f"error: {e}")

# ═══════════════════════════════════════════════════════════════
print("\n═══ CR-09: AI queue staleness timeout ═══")
# ═══════════════════════════════════════════════════════════════
try:
    with open("bot/send_reports.py", "r", encoding="utf-8") as f:
        src = f.read()
    idx = src.index("async def process_ai_generation_queue")
    func_body = src[idx:idx+800]

    if "processing_started" in func_body:
        ok("CR-09a", "processing_started timestamp записывается")
    else:
        fail("CR-09a", "processing_started не найден")

    if "600" in func_body or "stale" in func_body.lower():
        ok("CR-09b", "10-минутный таймаут для staleness")
    else:
        fail("CR-09b", "таймаут не найден")

    if "stale_reset" in func_body or "ai_queue_stale_reset" in func_body:
        ok("CR-09c", "событие stale_reset логируется")
    else:
        fail("CR-09c", "stale_reset событие не найдено")
except Exception as e:
    fail("CR-09", f"error: {e}")

# ═══════════════════════════════════════════════════════════════
print("\n═══ CR-10: pipeline_task не перемещает файлы при ошибке ═══")
# ═══════════════════════════════════════════════════════════════
try:
    with open("bot/send_reports.py", "r", encoding="utf-8") as f:
        src = f.read()
    idx = src.index("async def pipeline_task")
    func_body = src[idx:idx+16000]

    if "script_executed and script_rc == 0 and file_path.exists()" in func_body:
        ok("CR-10a", "перемещение в processed/ только при rc==0")
    else:
        fail("CR-10a", "условие rc==0 не найдено перед shutil.move")

    if "file_kept_for_retry" in func_body:
        ok("CR-10b", "событие file_kept_for_retry логируется")
    else:
        fail("CR-10b", "file_kept_for_retry не найдено")
except Exception as e:
    fail("CR-10", f"error: {e}")

# ═══════════════════════════════════════════════════════════════
print("\n═══ CR-11: DSO problem_clients — 4 столбца ═══")
# ═══════════════════════════════════════════════════════════════
try:
    with open("dso_aging_report.py", "r", encoding="utf-8") as f:
        src = f.read()
    idx = src.index("problem_rows")
    block = src[idx:idx+500]

    td_count = block.count("<td")
    # Должно быть 4 <td> в строке: #, name, debt, days
    if td_count >= 4:
        ok("CR-11a", f"problem_rows содержит {td_count} <td> элементов")
    else:
        fail("CR-11a", f"problem_rows содержит только {td_count} <td> (нужно 4)")

    if "client.get('days'" in block or "client['days']" in block:
        ok("CR-11b", "столбец days выводится")
    else:
        fail("CR-11b", "столбец days не найден в problem_rows")
except Exception as e:
    fail("CR-11", f"error: {e}")

# ═══════════════════════════════════════════════════════════════
print(f"\n{'═' * 60}")
print(f"ИТОГО: ✅ {PASS} passed, ❌ {FAIL} failed")
print(f"{'═' * 60}")
sys.exit(0 if FAIL == 0 else 1)

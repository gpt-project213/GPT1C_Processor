#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
smoke_test_all.py · v4
— Автосмок: создаём clean-копию, детектируем тип, запускаем нужный отчёт.
"""
from __future__ import annotations
from pathlib import Path
import subprocess, sys

from utils_excel import ensure_clean_xlsx
from analyze_debt_excel import detect_has_movements

def run(cmd): return subprocess.call(cmd)

def main():
    if len(sys.argv) != 2:
        print("Usage: python smoke_test_all.py <folder>")
        sys.exit(2)
    folder = Path(sys.argv[1])
    files = sorted(folder.glob("*.xlsx"))
    errs = 0

    for f in files:
        clean = ensure_clean_xlsx(f)
        has_mov, uniq, total = detect_has_movements(clean)
        print(f"{f.name}: movements={has_mov} (uniq={uniq}, total={total})")
        if has_mov:
            ret = run([sys.executable, "extended_debt_report.py", str(clean)])
        else:
            ret = run([sys.executable, "debt_report.py", str(clean)])
        if ret: errs += 1

    print(f"Total {len(files)}  Errors {errs}")

if __name__ == "__main__":
    main()

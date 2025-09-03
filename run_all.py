#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_all.py · v2  (01 Sep 2025)
— Пакетная обработка XLSX: при --force-fix создаёт чистую копию; автодетект расширенного/простого.
— Не меняет ваши отчёты: только orchestration.
"""
from __future__ import annotations

import argparse, subprocess, sys, time
from pathlib import Path

from utils_excel import ensure_clean_xlsx
from analyze_debt_excel import detect_has_movements

ROOT = Path(__file__).resolve().parent
REPORTS_DIR_DEFAULT = ROOT / "reports" / "отчеты для теста"
HTML_DIR = ROOT / "reports" / "html"
JSON_DIR = ROOT / "reports" / "json"
PDF_DIR  = ROOT / "reports" / "pdf"

def run(cmd: list[str]) -> int:
    print("→", " ".join(cmd))
    return subprocess.call(cmd)

def build_one(xlsx: Path, force_fix: bool = False) -> int:
    src = Path(xlsx)
    if force_fix:
        clean = ensure_clean_xlsx(src)  # вернёт путь к .__clean.xlsx
    else:
        cand = src.with_name(src.name + ".__clean.xlsx")
        clean = cand if cand.exists() else src

    has_mov, uniq, span = detect_has_movements(clean)
    print(f"[{src.name}] movements={has_mov} (uniq={uniq}, span={span})")

    if has_mov:
        return run([sys.executable, str(ROOT / "extended_debt_report.py"), str(clean)])
    else:
        return run([sys.executable, str(ROOT / "debt_report.py"), str(clean)])

def main():
    ap = argparse.ArgumentParser(description="Пакетная обработка XLSX в отчёты")
    ap.add_argument("--queue", default=str(REPORTS_DIR_DEFAULT), help="Папка с XLSX (по умолчанию reports/отчеты для теста)")
    ap.add_argument("--once", action="store_true", help="Один проход и выход (без демона)")
    ap.add_argument("--interval", type=int, default=20, help="Интервал сканирования папки, сек (для демона)")
    ap.add_argument("--force-fix", action="store_true", help="Перед обработкой делать .__clean.xlsx")
    args = ap.parse_args()

    queue = Path(args.queue).resolve()
    queue.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in queue.rglob("*.xlsx") if not p.name.endswith(".__clean.xlsx"))
    errors = 0

    for f in files:
        try:
            ret = build_one(f, force_fix=args.force_fix)
            if ret:
                errors += 1
        except Exception as e:
            print(f"  ✗ Ошибка: {f.name}: {e}")
            errors += 1

    print(f"Done. Processed {len(files)} file(s). Errors {errors}.")

    if not args.once:
        print("Daemon mode. Press Ctrl+C to stop.")
        known = set(files)
        try:
            while True:
                current = set(p for p in queue.rglob("*.xlsx") if not p.name.endswith(".__clean.xlsx"))
                new = sorted(current - known)
                for f in new:
                    try:
                        ret = build_one(f, force_fix=args.force_fix)
                        if ret:
                            errors += 1
                    except Exception as e:
                        print(f"  ✗ Ошибка: {f.name}: {e}")
                known = current
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("Stopped by user.")

if __name__ == "__main__":
    main()

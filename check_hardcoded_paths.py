#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_hardcoded_paths.py -- линтер жёстких путей
Запуск: python check_hardcoded_paths.py [--fix-hint] [файл1.py файл2.py ...]
Без аргументов -- проверяет все .py в текущей папке рекурсивно.
Exit code: 0 = OK, 1 = найдены нарушения.

Используется как pre-commit hook и в sync_project_to_github.ps1.
"""
from __future__ import annotations
import re, sys, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# -- Что считается нарушением --------------------------------------------------

# 1. Жёсткий диск: "E:\...", 'C:\...', r"D:\..."  (включая r"..." raw-строки)
#    X:\ исключён -- стандартный placeholder в документации/примерах ("X:\...\folder")
DRIVE_RE = re.compile(
    r"""[rR]?["']([A-WYZ]:[/\\][^"']{2,80})["']""",
    re.IGNORECASE
)

# 2. Голый относительный Path без ROOT-якоря -- Path("logs"), Path("reports/html")
#    Только если в файле нет ROOT = Path(__file__) и нет from config import
BARE_PATH_RE = re.compile(
    r'''Path\(\s*["'](?!https?://)([a-zA-Z0-9_/\\.-]{3,60})["']\s*\)'''
)

# Исключения -- допустимые строки в Path()
BARE_PATH_WHITELIST = {
    "utf-8", "utf-8-sig", "r", "w", "rb", "wb",
    "html", "xml", "json", "yaml", ".", "..",
    "utf8", "cp1251",
}

# -- Паттерны "файл уже правильный" -------------------------------------------
HAS_ROOT     = re.compile(r'ROOT\s*=\s*Path\s*\(\s*__file__\s*\)')
HAS_CFG      = re.compile(r'from\s+config\s+import|import\s+config\b')
HAS_ROOT_DIR = re.compile(r'ROOT_DIR\s*=\s*Path\s*\(\s*__file__\s*\)')

# -- Исключённые файлы (сам линтер, конфиги и т.д.) ---------------------------
SKIP_FILES = {
    "check_hardcoded_paths.py",
    "setup.py", "conf.py",
    "txt_to_html.py",        # вспомогательный инструмент tools/
    "patch_mobile_click.py", # вспомогательный инструмент tools/
    "inject_local.py",       # вспомогательный инструмент tools/
}

# -- Цвета для терминала -------------------------------------------------------
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def check_file(path: Path) -> list[tuple[int, str, str]]:
    """Возвращает список (line_num, code, message)."""
    issues = []

    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return [(0, "READ_ERROR", str(e))]

    lines = src.splitlines()
    has_root = bool(HAS_ROOT.search(src) or HAS_ROOT_DIR.search(src))
    has_cfg  = bool(HAS_CFG.search(src))
    anchored = has_root or has_cfg

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Пропускаем комментарии
        if stripped.startswith("#"):
            continue

        # Пропускаем строки в docstring (упрощённо -- строки только с кавычками)
        if stripped.startswith('"""') or stripped.startswith("'''"):
            continue

        # -- Проверка 1: жёсткий путь на диск ---------------------------------
        for m in DRIVE_RE.finditer(line):
            path_val = m.group(1).replace("\\", "/")
            parts = [p for p in path_val.split("/") if p and ":" not in p]
            hint_rel = "/".join(parts) if parts else path_val
            issues.append((
                i,
                "HARDCODED_DRIVE",
                f'Жёсткий путь: {m.group(0)!r}  ->  замени на ROOT / "{hint_rel}"'
            ))

        # -- Проверка 2: голый Path() без ROOT --------------------------------
        if not anchored:
            for m in BARE_PATH_RE.finditer(line):
                val = m.group(1)
                if val.lower() in BARE_PATH_WHITELIST:
                    continue
                if val.startswith("."):
                    continue
                issues.append((
                    i,
                    "BARE_PATH",
                    f'Path("{val}") без ROOT -- добавь ROOT = Path(__file__).resolve().parent  в начало файла'
                ))

    return issues


def main(argv=None):
    # Фикс Windows cp1251: принудительно UTF-8 для вывода в консоль
    import sys as _sys
    if hasattr(_sys.stdout, "reconfigure"):
        try:
            _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="Линтер жёстких путей")
    ap.add_argument("files", nargs="*", help="Файлы для проверки (по умолчанию -- все .py рекурсивно)")
    ap.add_argument("--strict", action="store_true", help="BARE_PATH тоже считать ошибкой (exit 1)")
    ap.add_argument("--quiet",  action="store_true", help="Только итог, без деталей")
    args = ap.parse_args(argv)

    if args.files:
        py_files = [Path(f) for f in args.files if f.endswith(".py")]
    else:
        py_files = sorted(ROOT.rglob("*.py"))
        # Исключаем .venv и __pycache__
        py_files = [
            f for f in py_files
            if ".venv" not in f.parts
            and "__pycache__" not in f.parts
            and f.name not in SKIP_FILES
        ]

    total_errors   = 0
    total_warnings = 0
    files_with_issues = 0

    for f in py_files:
        issues = check_file(f)
        if not issues:
            continue

        errors   = [(n, c, m) for n, c, m in issues if c == "HARDCODED_DRIVE"]
        warnings = [(n, c, m) for n, c, m in issues if c == "BARE_PATH"]

        total_errors   += len(errors)
        total_warnings += len(warnings)

        if errors or (warnings and args.strict):
            files_with_issues += 1

        if not args.quiet and (errors or warnings):
            rel = f.relative_to(ROOT) if f.is_relative_to(ROOT) else f
            print(f"\n{BOLD}{rel}{RESET}")

            for line_num, code, msg in errors:
                print(f"  {RED}L{line_num:4d} [{code}]{RESET} {msg}")

            for line_num, code, msg in warnings:
                print(f"  {YELLOW}L{line_num:4d} [{code}]{RESET} {msg}")

    # -- Итог -----------------------------------------------------------------
    print()
    if total_errors == 0 and total_warnings == 0:
        print(f"{GREEN}[OK] Zhyostkikh putey ne naydeno -- proekt portativny!{RESET}")
        return 0

    if total_errors > 0:
        print(f"{RED}[ERROR] Zhyostkie puti: {total_errors} v {files_with_issues} faylakh{RESET}")

    if total_warnings > 0:
        print(f"{YELLOW}[WARN] Bare Path() bez ROOT: {total_warnings}{RESET}")

    if total_errors > 0:
        print(f"\n{BOLD}Правило:{RESET}")
        print("  Замени Path(r\"X:\\...\\folder\") на ROOT / \"folder\"")
        print("  Где ROOT = Path(__file__).resolve().parent  (в начале файла)")
        return 1

    if args.strict and total_warnings > 0:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
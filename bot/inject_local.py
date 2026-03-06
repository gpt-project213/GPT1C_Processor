#!/usr/bin/env python
# coding: utf-8
"""
inject_local.py · v1.0.0 (2026-02-19)
──────────────────────────────────────────────────────
ТЕСТОВЫЙ ИНЖЕКТОР: замена imap_fetcher для локального тестирования.

Читает Excel-файлы из папки test_inbox/ (или любой другой через --inbox),
копирует их в reports/queue/ и создаёт clean-копии в reports/excel/clean/
— точно так же, как imap_fetcher, но БЕЗ почты.

ИСПОЛЬЗОВАНИЕ:
    # 1. Положи Excel-файлы в папку test_inbox/
    # 2. В send_reports.py временно замени строку:
    #        script_rc, _, _ = await run_script_async("imap_fetcher.py", "--once")
    #    на:
    #        script_rc, _, _ = await run_script_async("inject_local.py", "--once")
    # 3. Или запусти вручную:
    python inject_local.py

КЛЮЧИ:
    --inbox PATH    папка с входящими Excel (по умолч: test_inbox/)
    --once          режим однократного запуска (совместим с pipeline)
    --keep          НЕ удалять файлы из inbox после обработки (по умолч: удаляет)
    --dry-run       показать что будет сделано, без реального копирования
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

TZ = timezone(timedelta(hours=5))
ROOT = Path(__file__).resolve().parent

QUEUE_DIR = ROOT / "reports" / "queue"
CLEAN_DIR = ROOT / "reports" / "excel" / "clean"
LOGS_DIR  = ROOT / "logs"

for d in (QUEUE_DIR, CLEAN_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s, INFO %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
LOG = logging.getLogger("inject_local")

__VERSION__ = "v1.0.0"


def get_clean_xlsx():
    """Импортируем ensure_clean_xlsx из utils_excel если доступен."""
    try:
        from utils_excel import ensure_clean_xlsx
        return ensure_clean_xlsx
    except ImportError:
        LOG.warning("utils_excel не найден — clean-копии будут простыми копиями")
        return None


def process_inbox(inbox: Path, keep: bool = False, dry_run: bool = False) -> int:
    """
    Копирует Excel-файлы из inbox в queue/ и создаёт clean-копии.
    Возвращает количество обработанных файлов.
    """
    ensure_clean_xlsx = get_clean_xlsx()

    xlsx_files = sorted([
        f for f in inbox.iterdir()
        if f.suffix.lower() == ".xlsx"
        and not f.name.startswith("~$")  # пропускаем lock-файлы Excel
        and f.is_file()
    ])

    if not xlsx_files:
        LOG.info("INBOX пустой: %s", inbox)
        LOG.info("CYCLE DONE: saved=0")
        return 0

    saved = 0
    for src in xlsx_files:
        dst = QUEUE_DIR / src.name
        clean_dst = CLEAN_DIR / f"{src.name}.__clean.xlsx"

        LOG.info("INJECT detected: %s", src.name)

        if dry_run:
            LOG.info("  [DRY-RUN] WOULD COPY → %s", dst)
            LOG.info("  [DRY-RUN] WOULD CLEAN → %s", clean_dst)
            saved += 1
            continue

        # Копируем в queue
        shutil.copy2(src, dst)
        LOG.info("SAVED: %s", dst.relative_to(ROOT))

        # Создаём clean-копию
        if ensure_clean_xlsx:
            try:
                clean_path = ensure_clean_xlsx(dst, force_fix=True)
                LOG.info("Clean copy: %s", clean_path.relative_to(ROOT))
            except Exception as e:
                LOG.warning("Clean failed (%s), делаю простую копию: %s", e, clean_dst)
                shutil.copy2(dst, clean_dst)
                LOG.info("Clean copy (fallback): %s", clean_dst.relative_to(ROOT))
        else:
            shutil.copy2(dst, clean_dst)
            LOG.info("Clean copy: %s", clean_dst.relative_to(ROOT))

        # Удаляем из inbox если не --keep
        if not keep:
            src.unlink()
            LOG.info("INBOX removed: %s", src.name)

        saved += 1

    LOG.info("CYCLE DONE: saved=%d", saved)
    return saved


def main(argv=None):
    ap = argparse.ArgumentParser(description="Локальный инжектор файлов в pipeline (замена imap_fetcher для тестов)")
    ap.add_argument("--inbox",   default="test_inbox",  help="Папка с входящими Excel-файлами")
    ap.add_argument("--once",    action="store_true",   help="Однократный запуск (флаг совместимости с pipeline)")
    ap.add_argument("--keep",    action="store_true",   help="Не удалять файлы из inbox после обработки")
    ap.add_argument("--dry-run", action="store_true",   help="Показать что будет сделано без реального копирования")
    args = ap.parse_args(argv)

    inbox = Path(args.inbox)
    if not inbox.is_absolute():
        inbox = ROOT / inbox

    LOG.info("=== inject_local %s ===", __VERSION__)
    LOG.info("Inbox: %s", inbox)
    LOG.info("Keep: %s | Dry-run: %s", args.keep, args.dry_run)

    if not inbox.exists():
        inbox.mkdir(parents=True)
        LOG.info("Создана папка inbox: %s", inbox)
        LOG.info("Положи Excel-файлы в %s и запусти снова", inbox)
        LOG.info("CYCLE DONE: saved=0")
        return 0

    saved = process_inbox(inbox, keep=args.keep, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
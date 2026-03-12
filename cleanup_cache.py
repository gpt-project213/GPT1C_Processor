# cleanup_cache.py — автоочистка кэша/отчётов (TTL и лимиты по объёму)
# ТЗ: HTML/JSON/ИИ — 90 дней, общий объём ≤ 5 ГБ; Оригиналы XLSX — до 2 ГБ.
# Логи: logs/cleanup_YYYYMMDD_HHMMSS.log
#
# v1.0.1 fix (06.03.2026): Все пути переведены на ROOT-relative (Path(__file__)).
#   Было: Path("logs"), Path("reports/html") — зависели от CWD.
#   Стало: ROOT / "logs", ROOT / "reports" / "html" — работают из любой директории.

from __future__ import annotations
import os, sys, argparse, logging
from pathlib import Path
from datetime import datetime, timedelta

# ── Якорь проекта: папка, где лежит этот скрипт ──────────────────────────────
ROOT = Path(__file__).resolve().parent

DATEFMT = "%Y-%m-%d %H:%M:%S"
LOGDIR = ROOT / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)
now = datetime.now()
logpath = LOGDIR / f"cleanup_{now.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=str(logpath), level=logging.INFO,
                    format="%(asctime)s, %(levelname)s %(message)s", datefmt=DATEFMT)

# директории и политики
ARTIFACTS  = [ROOT / "reports/html", ROOT / "reports/json",
              ROOT / "reports/pdf",  ROOT / "reports/ai"]
XLSX_STORES = [ROOT / "reports/excel", ROOT / "reports/queue"]

def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except (FileNotFoundError, OSError):
        return 0.0

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except Exception:
        return default

def _gb_to_bytes(gb: int) -> int:
    return gb * 1024**3

TTL_DAYS = _env_int("CLEANUP_TTL_DAYS", 90)
ART_CAP_BYTES = _gb_to_bytes(_env_int("CLEANUP_ARTIFACTS_CAP_GB", 5))   # 5 ГБ
XLSX_CAP_BYTES = _gb_to_bytes(_env_int("CLEANUP_XLSX_CAP_GB", 2))       # 2 ГБ

def ls_files(dirs):
    files = []
    for d in dirs:
        if not d.exists(): 
            continue
        for p in d.rglob("*"):
            if p.is_file():
                files.append(p)
    return files

def total_size(files):
    s = 0
    for f in files:
        try:
            s += f.stat().st_size
        except Exception:
            pass
    return s

def purge_older_than(files, cutoff, dry_run=False):
    removed = []
    for f in files:
        try:
            if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
                removed.append(f)
        except Exception:
            continue
    removed.sort(key=_mtime)
    for f in removed:
        logging.info("TTL REMOVE: %s", f)
        if not dry_run:
            try:
                f.unlink(missing_ok=True)
            except Exception as e:
                logging.warning("TTL remove failed: %s (%s)", f, e)
    return len(removed)

def purge_over_cap(files, cap_bytes, dry_run=False):
    files = [f for f in files if f.exists()]
    files.sort(key=_mtime)  # старые сначала
    current = total_size(files)
    removed = 0
    for f in files:
        if current <= cap_bytes:
            break
        try:
            sz = f.stat().st_size
            logging.info("CAP REMOVE: %s (%d bytes)", f, sz)
            if not dry_run:
                f.unlink(missing_ok=True)
            current -= sz
            removed += 1
        except Exception as e:
            logging.warning("CAP remove failed: %s (%s)", f, e)
    return removed

def main(argv):
    ap = argparse.ArgumentParser("cleanup_cache")
    ap.add_argument("--dry-run", action="store_true", help="только показать, что будет удалено")
    ap.add_argument("--ttl-days", type=int, default=TTL_DAYS)
    args = ap.parse_args(argv)

    cutoff = datetime.now() - timedelta(days=args.ttl_days)
    art_files = ls_files(ARTIFACTS)
    xlsx_files = ls_files(XLSX_STORES)

    logging.info("START CLEANUP (dry=%s, ttl_days=%d)", args.dry_run, args.ttl_days)
    logging.info("Artifacts files: %d, XLSX files: %d", len(art_files), len(xlsx_files))

    n1 = purge_older_than(art_files, cutoff, dry_run=args.dry_run)
    n2 = purge_older_than(xlsx_files, cutoff, dry_run=args.dry_run)
    logging.info("TTL REMOVED: artifacts=%d, xlsx=%d", n1, n2)

    # лимиты
    art_files = ls_files(ARTIFACTS)
    xlsx_files = ls_files(XLSX_STORES)
    removed_cap_art = purge_over_cap(art_files, ART_CAP_BYTES, dry_run=args.dry_run)
    removed_cap_xlsx = purge_over_cap(xlsx_files, XLSX_CAP_BYTES, dry_run=args.dry_run)
    logging.info("CAP REMOVED: artifacts=%d, xlsx=%d", removed_cap_art, removed_cap_xlsx)

    logging.info("DONE")
    print(f"Log: {logpath}")

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
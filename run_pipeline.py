#!/usr/bin/env python
# coding: utf-8
"""
run_pipeline.py · v2.2.1 · Asia/Almaty

Изменения относительно v2.1:
• [BEHAVIOR] Пайплайн больше НЕ отправляет ничего в Telegram по умолчанию.
• [ENV] Добавлен флаг PIPELINE_TG_SEND=true/false (по умолчанию false).
• [LOG] Явно логируем подавление отправки (pipeline_tg_suppressed).

v2.2.1: TZ timezone(timedelta(hours=5)) → ZoneInfo("Asia/Almaty") (Bug TZ)
"""

from __future__ import annotations

import os
import sys
import shutil
import traceback
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from importlib import import_module

# .env
try:
    import dotenv  # type: ignore
    dotenv.load_dotenv()
except Exception:
    pass

TZ = ZoneInfo("Asia/Almaty")

ROOT = Path(__file__).resolve().parent
LOG_DIR         = ROOT / "logs"
QUEUE_DIR       = ROOT / "reports" / "queue"
HTML_DIR        = ROOT / "reports" / "html"
JSON_DIR        = ROOT / "reports" / "json"
EXCEL_PROCESSED = ROOT / "reports" / "excel" / "processed"

for _d in (LOG_DIR, QUEUE_DIR, HTML_DIR, JSON_DIR, EXCEL_PROCESSED):
    _d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"pipeline_{datetime.now(TZ).strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE.touch(exist_ok=True)

def _log(msg: str, err: bool = False) -> None:
    tag  = "ERROR" if err else "INFO"
    line = f"{datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')}, {tag} {msg}"
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    finally:
        print(line)

# «святая» сборка
try:
    rep = import_module("debt_auto_report")
    build_report = getattr(rep, "build_report")
except Exception as e:
    _log(f"Не удалось импортировать debt_auto_report.build_report: {e}", err=True)
    sys.exit(2)

# TG (оставляем импорт, но используем по флагу)
try:
    from send_tg import send_file
except Exception as e:
    _log(f"Не удалось импортировать send_tg.send_file: {e}", err=True)
    sys.exit(2)

# mobile patch
try:
    from tools.patch_mobile_click import patch as mobile_patch
except Exception as e:
    _log(f"Не удалось импортировать tools.patch_mobile_click: {e}", err=True)
    sys.exit(2)

PIPELINE_TG_SEND = os.getenv("PIPELINE_TG_SEND", "false").lower() == "true"

def _run_sub(cmd: list[str]) -> int:
    _log("RUN: " + " ".join(cmd))
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
        if out.strip():
            _log(out.strip())
        return 0
    except subprocess.CalledProcessError as e:
        if e.output:
            _log(e.output.strip(), err=True)
        return e.returncode

def _is_simple_html(p: Path) -> bool:
    name = p.name.lower()
    return name.endswith("_debt.html") and not name.startswith("debt_ext_")

def _is_extended_html(p: Path) -> bool:
    return p.name.lower().startswith("debt_ext_")

def imap_cycle_once() -> None:
    _log("== IMAP fetch_once ==")
    rc = _run_sub([sys.executable, "imap_fetcher.py", "--once"])
    if rc != 0:
        _log("IMAP завершился с ошибкой (см. логи выше). Продолжаю обработку локальной очереди.", err=True)

def process_queue() -> None:
    ok = fail = 0
    files = sorted(
        p for p in QUEUE_DIR.glob("*.[xX][lL][sS]*")
        if not p.name.endswith(".work")
    )
    if not files:
        _log("QUEUE is empty")

    for x in files:
        try:
            _log(f"START {x.name}")
            out_html = Path(build_report(x))
            if not out_html.exists():
                raise RuntimeError("HTML не создан (build_report не вернул существующий путь)")

            try:
                mobile_patch(out_html)
            except Exception as e:
                _log(f"Mobile-patch warning: {e}", err=False)

            if _is_simple_html(out_html):
                if PIPELINE_TG_SEND:
                    # старое поведение (не рекомендуется): отправить «Простой отчёт» и меню
                    send_file(out_html, caption="Простой отчёт", with_menu=True)
                else:
                    _log("pipeline_tg_suppressed: simple html готов, отправка в TG отключена по флагу")
            else:
                _log(f"Extended HTML готов (локально): {out_html.name}")

            try:
                if x.exists():
                    shutil.move(str(x), str(EXCEL_PROCESSED / x.name))
                else:
                    _log(f"Источник уже удалён (clean) → {x.name}")
            except Exception as e:
                _log(f"Move to processed warning: {x.name}: {e}", err=False)

            _log(f"FINISH OK {x.name}")
            ok += 1

        except Exception as e:
            _log(f"FAIL {x.name}: {e}\n{traceback.format_exc()}", err=True)
            fail += 1

    _log(f"QUEUE DONE: processed={ok}, failed={fail}")

def main() -> int:
    imap_cycle_once()
    process_queue()
    _log("PIPELINE FINISHED")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

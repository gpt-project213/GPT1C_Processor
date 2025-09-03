# tools/pdf_export.py
# version: v1.1 (2025-09-02)
# Назначение: конвертирует готовые HTML-отчёты в PDF с помощью wkhtmltopdf.
# Инвариант путей (по ТЗ, согласовано): выход по умолчанию в reports/pdf (нижний регистр).
# Совместимость: если reports/pdf отсутствует, а reports/PDF существует — используем reports/PDF.
# Зависимости: wkhtmltopdf(.exe) установлен (portable) и доступен по пути или через .env (WKHTMLTOPDF_BIN)

from __future__ import annotations
import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

try:
    # .env не обязателен, но если есть — считаем
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # допустимо, просто пропустим

__VERSION__ = "pdf_export.py v1.1 — 2025-09-02"

# ── базовые пути ──────────────────────────────────────────────────────────────
# Файл лежит в tools/, корень проекта — на уровень выше
ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
HTML_DIR = ROOT / "reports" / "html"

# Новый стандарт: reports/pdf (нижний)
PDF_DIR_NEW = ROOT / "reports" / "pdf"
# Старый путь для совместимости: reports/PDF (верхний регистр)
PDF_DIR_OLD = ROOT / "reports" / "PDF"

WKHTML_DEFAULT = ROOT / "tools" / "wkhtmltopdf" / "bin" / ("wkhtmltopdf.exe" if os.name == "nt" else "wkhtmltopdf")

for d in (LOGS_DIR, HTML_DIR, PDF_DIR_NEW):
    d.mkdir(parents=True, exist_ok=True)
# старую папку не создаём специально — только если уже есть

# ── логирование ───────────────────────────────────────────────────────────────
def _make_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"pdf_export_{ts}.log"
    logger = logging.getLogger("pdf_export")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s, %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    # не дублировать хендлеры при повторном импорте
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

log = _make_logger()
log.info("Старт %s", __VERSION__)

# ── env загрузка (опционально) ───────────────────────────────────────────────
if load_dotenv is not None:
    try:
        load_dotenv()
        log.info(".env загружен (если присутствовал)")
    except Exception as e:
        log.warning("Не удалось загрузить .env: %s", e)

# ── выбор каталога вывода с учётом совместимости ─────────────────────────────
def _resolve_pdf_out_dir(user_out: str | None) -> Path:
    """
    Приоритет:
      1) --out <dir> от пользователя
      2) новый стандарт: reports/pdf (нижний)
      3) если новый отсутствует, а старый reports/PDF есть — использовать старый
    """
    if user_out:
        p = Path(user_out)
        p.mkdir(parents=True, exist_ok=True)
        return p
    if PDF_DIR_NEW.exists():
        PDF_DIR_NEW.mkdir(parents=True, exist_ok=True)
        return PDF_DIR_NEW
    if PDF_DIR_OLD.exists() and not PDF_DIR_NEW.exists():
        # совместимость со старой структурой до миграции
        PDF_DIR_OLD.mkdir(parents=True, exist_ok=True)
        log.warning("Использую legacy каталог вывода PDF: %s (создай reports/pdf для перехода)", PDF_DIR_OLD)
        return PDF_DIR_OLD
    # по умолчанию создаём новый стандарт
    PDF_DIR_NEW.mkdir(parents=True, exist_ok=True)
    return PDF_DIR_NEW

# ── вспомогательные ──────────────────────────────────────────────────────────
def _detect_wkhtmltopdf(cli_path: str | None) -> Path:
    """
    Приоритет:
      1) аргумент --wkhtml
      2) переменная окружения WKHTMLTOPDF_BIN (.env)
      3) дефолтный tools/wkhtmltopdf/bin/wkhtmltopdf(.exe)
    """
    candidates: List[Path] = []
    if cli_path:
        candidates.append(Path(cli_path))
    env_path = os.getenv("WKHTMLTOPDF_BIN")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(WKHTML_DEFAULT)

    for p in candidates:
        if p and p.exists() and p.is_file():
            return p

    tried = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Не найден wkhtmltopdf. Проверьте:\n"
        "  — ключ --wkhtml\n"
        "  — или переменную окружения WKHTMLTOPDF_BIN в .env\n"
        "Пробовали пути:\n" + tried
    )

def export_to_pdf(html_path: Path | str,
                  wkhtml_path: Path | None = None,
                  out_dir: Path | None = None,
                  force: bool = False) -> Path:
    """
    Конвертирует один HTML в PDF.
    :param html_path: путь к HTML-файлу
    :param wkhtml_path: путь к wkhtmltopdf(.exe) (если None — autodetect)
    :param out_dir: каталог для PDF (если None — auto resolve с учётом совместимости)
    :param force: перезаписывать готовый PDF (False — по умолчанию)
    :return: путь к PDF
    """
    html_path = Path(html_path).resolve()
    if not html_path.exists() or html_path.suffix.lower() != ".html":
        raise FileNotFoundError(f"HTML не найден или неверное расширение: {html_path}")

    wkhtml = wkhtml_path or _detect_wkhtmltopdf(None)
    out_dir = out_dir or _resolve_pdf_out_dir(user_out=None)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = html_path.stem + ".pdf"
    pdf_path = out_dir / pdf_name

    if pdf_path.exists() and not force:
        log.info("↪ Пропуск: уже существует (use --force) — %s", pdf_path)
        return pdf_path

    # Включаем доступ к локальным ресурсам на диске (css, картинки)
    cmd = [
        str(wkhtml),
        "--enable-local-file-access",
        "--encoding", "utf-8",
        str(html_path),
        str(pdf_path),
    ]

    log.info("PDF: %s -> %s", html_path.name, pdf_path)
    try:
        # Под Windows скрываем окно процесса
        creationflags = 0x08000000 if os.name == "nt" else 0
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            creationflags=creationflags,
        )
        if res.returncode != 0:
            log.error("wkhtmltopdf завершился с кодом %s", res.returncode)
            if res.stdout:
                log.error("stdout:\n%s", res.stdout.strip())
            if res.stderr:
                log.error("stderr:\n%s", res.stderr.strip())
            raise RuntimeError(f"wkhtmltopdf error code {res.returncode}")
        else:
            if res.stderr:
                log.info("wkhtmltopdf stderr (инфо):\n%s", res.stderr.strip())
    except Exception as e:
        log.exception("Ошибка конвертации %s: %s", html_path.name, e)
        raise

    if not pdf_path.exists():
        raise RuntimeError(f"wkhtmltopdf отработал без ошибок, но файл не найден: {pdf_path}")
    log.info("✔ PDF готов: %s", pdf_path)
    return pdf_path

# ── CLI ──────────────────────────────────────────────────────────────────────
def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"{__VERSION__}\nКонвертер HTML → PDF (выход: reports/pdf по умолчанию)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("inputs", nargs="+",
                   help="Путь(и) к HTML-файлу(ам) или папке(ам) с HTML.")
    p.add_argument("--wkhtml", dest="wkhtml", default=None,
                   help="Путь к wkhtmltopdf(.exe) (иначе .env WKHTMLTOPDF_BIN, иначе tools/.../wkhtmltopdf(.exe)).")
    p.add_argument("--glob", dest="glob", default="*.html",
                   help="Маска поиска HTML в режиме каталога (по умолчанию: *.html).")
    p.add_argument("--recursive", action="store_true",
                   help="Рекурсивный поиск по папкам.")
    p.add_argument("--out", dest="out", default=None,
                   help="Каталог вывода PDF (перекрывает дефолт reports/pdf).")
    p.add_argument("--force", action="store_true",
                   help="Перезаписать готовые PDF.")
    return p.parse_args(argv)

def _collect_html_files(paths: Iterable[str], pattern: str, recursive: bool) -> list[Path]:
    htmls: list[Path] = []
    for raw in paths:
        p = Path(raw).resolve()
        if p.is_file() and p.suffix.lower() == ".html":
            htmls.append(p)
        elif p.is_dir():
            htmls.extend(sorted(p.rglob(pattern) if recursive else p.glob(pattern)))
        else:
            log.warning("Пропускаю: %s (ни файл .html, ни папка)", p)
    # уникализируем и фильтруем
    uniq: list[Path] = []
    seen = set()
    for f in htmls:
        if f.exists() and f.suffix.lower() == ".html" and str(f) not in seen:
            uniq.append(f)
            seen.add(str(f))
    return uniq

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    try:
        wkhtml = _detect_wkhtmltopdf(args.wkhtml)
        log.info("wkhtmltopdf: %s", wkhtml)
    except Exception as e:
        log.error("Не удалось определить путь к wkhtmltopdf: %s", e)
        return 2

    out_dir = _resolve_pdf_out_dir(args.out)
    log.info("Каталог вывода: %s", out_dir)

    html_files = _collect_html_files(args.inputs, args.glob, args.recursive)
    if not html_files:
        log.error("HTML-файлы не найдены.")
        return 3

    ok, fail = 0, 0
    for f in html_files:
        try:
            export_to_pdf(f, wkhtml_path=wkhtml, out_dir=out_dir, force=args.force)
            ok += 1
        except Exception:
            fail += 1

    log.info("Итого: успех=%s, ошибки=%s", ok, fail)
    return 0 if fail == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())

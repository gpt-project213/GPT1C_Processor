# -*- coding: utf-8 -*-
"""
send_reports.py
version: v1.3 (2025-09-02)

Рассылка HTML-отчётов дебиторки в Telegram (строго по ТЗ):
• HTML → reports/html (инвариант).
• PDF → reports/pdf (нижний регистр). Кнопка «PDF (iPhone)» добавляется только при наличии *.pdf.
• Логи: logs/send_reports_YYYYMMDD_HHMMSS.log (формат "%(asctime)s, %(levelname)s %(message)s", TZ=Asia/Almaty) + дублирование в консоль.
• Кнопки: Простой / Расширенный / Анализ ИИ / PDF (iPhone) / 📦 Архив / Период.
• Команды: /start, /help, /period YYYY-MM.
• Антиспам: задержка TG_RATE_DELAY (сек) из .env, по умолчанию 0.7.
• --only-latest: шлём самый свежий файл на менеджера.
• --admin-only: рассылаем ТОЛЬКО админу(ам).
• Если запрошен debt_ai_*.html и его нет — создаётся лёгкая заглушка (без ИИ-аналитики).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

# ──────────────────────────── проектные инварианты / логи ────────────────────────────
import config
from config import HTML_DIR, PDF_DIR, LOGS_DIR, setup_logging

__VERSION__ = "send_reports.py v1.3 — 2025-09-02"
log = setup_logging("send_reports")

# ──────────────────────────── пути ────────────────────────────
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
CONFIG_DIR = ROOT / "config"

# По умолчанию сканируем ИМЕННО инвариантную папку HTML-отчётов
REPORTS_DIR_DEFAULT = HTML_DIR

CACHE_DIR = ROOT / "cache" / "bot"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# локальный модуль (бизнес-логика детекции — не трогаем)
from detectors import (  # noqa: E402
    detect_report_type,
    compute_recipients,
    detect_manager_from_filename,
)

SENT_INDEX_PATH = CACHE_DIR / "sent_index.json"
CHATS_PATH = CACHE_DIR / "chats.json"   # username -> chat_id

# ──────────────────────────── утилиты ─────────────────────────

def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}

def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def now_iso_tz() -> str:
    # TZ задаётся общесистемно (Asia/Almaty)
    tzname = os.getenv("TZ", "Asia/Almaty")
    return datetime.now(ZoneInfo(tzname)).isoformat(timespec="seconds")

def find_html_files(reports_dir: Path) -> List[Path]:
    return sorted(p for p in reports_dir.rglob("*.html") if p.is_file())

DATE_RGX = re.compile(r"(\d{4}[-_.]\d{2}(?:[-_.]\d{2})?)")

def caption_for(path: Path) -> str:
    who = path.stem.rsplit("_", 1)[-1]
    m = DATE_RGX.search(path.stem)
    period = m.group(1).replace("_", "-").replace(".", "-") if m else ""
    title = "Дебиторка"
    stem_low = path.stem.lower()
    if "debt_ext" in stem_low:
        title += " — расширенный"
    if "debt_ai" in stem_low:
        title += " — аналитика"
    first = f"📄 {title}"
    parts = []
    if period:
        parts.append(f"Период: {period}")
    if who and who.lower() not in {"summary", "ext", "ai"}:
        parts.append(who)
    second = " • ".join(parts)
    return f"{first}\n{second}" if second else first

def _pdf_path_for_html(html_path: Path) -> Path:
    """Путь к PDF в инвариантной папке reports/pdf по stem HTML."""
    pdf_name = html_path.stem + ".pdf"
    return PDF_DIR / pdf_name

def _pdf_exists_for(html_path: Path) -> bool:
    return _pdf_path_for_html(html_path).exists()

def build_keyboard_for(path: Path) -> InlineKeyboardMarkup:
    buttons = []
    # Кнопки «Простой/Расширенный/Анализ ИИ» — только если совпадает шаблон имени
    m = re.match(r"(?i)debt(?:_ext|_ai)?[-_.](\d{4}[-_.]\d{2}(?:[-_.]\d{2})?)[-_.]([^.]+)$", path.stem)
    if m:
        date, name = m.group(1), m.group(2)
        for label, fname in (
            ("Простой",     f"debt_{date}_{name}.html"),
            ("Расширенный", f"debt_ext_{date}_{name}.html"),
            ("Анализ ИИ",   f"debt_ai_{date}_{name}.html"),
        ):
            buttons.append([InlineKeyboardButton(label, callback_data=f"send|{fname}")])

    # Кнопка PDF (iPhone) — всегда проверяем наличие PDF в reports/pdf по stem
    pdf_path = _pdf_path_for_html(path)
    if pdf_path.exists():
        buttons.append([InlineKeyboardButton("PDF (iPhone)", callback_data=f"send|{pdf_path.name}")])

    # Архив — всегда в конце
    buttons.append([InlineKeyboardButton("📦 Архив / Период", callback_data="archive")])
    return InlineKeyboardMarkup(buttons)

# ───────────────────── AI-заглушка (создание файла) ───────────────

def ensure_ai_variant(any_related_path: Path) -> Optional[Path]:
    """
    Пытаемся сделать debt_ai_*.html на базе ближайшего отчёта.
    Если debt_ai_* нет — ищем рядом debt_ext_* или debt_* с той же датой и именем,
    копируем HTML как есть и добавляем баннер в <body>.
    """
    stem = any_related_path.stem
    base_dir = any_related_path.parent

    m = re.match(r"(?i)debt(?:_ext|_ai)?[-_.](\d{4}[-_.]\d{2}(?:[-_.]\d{2})?)[-_.]([^.]+)$", stem)
    if not m:
        return None
    date, name = m.group(1), m.group(2)
    ai_path = base_dir / f"debt_ai_{date}_{name}.html"
    if ai_path.exists():
        return ai_path

    src = base_dir / f"debt_ext_{date}_{name}.html"
    if not src.exists():
        src = base_dir / f"debt_{date}_{name}.html"
    if not src.exists():
        return None

    try:
        html = src.read_text(encoding="utf-8", errors="ignore")
        inject = (
            "<div style='position:sticky;top:0;z-index:9999;"
            "background:#111;color:#fff;padding:8px 12px;"
            "font-family:system-ui,Segoe UI,Arial;font-size:14px;"
            "border-bottom:2px solid #0af;'>"
            "🤖 Аналитика ИИ — подготовлено по запросу (тестовая заглушка)</div>"
        )
        html = re.sub(r"(?i)<body[^>]*>", lambda m: m.group(0) + inject, html, count=1)
        ai_path.write_text(html, encoding="utf-8")
        return ai_path
    except Exception:
        return None

# ──────────────────────────── чаты ────────────────────────────

def chats_map() -> Dict[str, int]:
    return load_json(CHATS_PATH, {})

def chats_save(d: Dict[str, int]) -> None:
    save_json(CHATS_PATH, d)

# ──────────────────────────── команды ─────────────────────────

async def cmd_start(update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not user:
        return
    uname = f"@{user.username}" if user.username else str(user.id)
    cm = chats_map()
    cm[uname] = update.effective_chat.id
    chats_save(cm)
    await update.message.reply_text(
        "✅ Привязка установлена.\n"
        "Команды:\n"
        "• /period YYYY-MM — прислать файлы за период из архива\n"
        "• /help — краткая справка"
    )

async def cmd_help(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Команды:\n"
        "• /period YYYY-MM — прислать файлы за период из архива\n"
        "• /help — это сообщение\n"
        "Кнопка «📦 Архив / Период» под каждым отчётом."
    )

async def cmd_period(update, context: ContextTypes.DEFAULT_TYPE):
    arg = " ".join(context.args).strip() if context.args else ""
    if not arg or not re.fullmatch(r"\d{4}-\d{2}", arg):
        await update.message.reply_text("Укажите период: /period 2025-08")
        return
    period = arg
    repo = Path(context.bot_data.get("reports_dir") or REPORTS_DIR_DEFAULT)
    paths = [p for p in find_html_files(repo) if period in p.stem]
    if not paths:
        await update.message.reply_text("В архиве нет файлов за указанный период.")
        return
    for p in paths:
        await update.message.reply_document(
            document=p.open("rb"),
            filename=p.name,
            caption=caption_for(p),
            reply_markup=build_keyboard_for(p),
        )

async def cb_handler(update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""
    repo = Path(context.bot_data.get("reports_dir") or REPORTS_DIR_DEFAULT)

    if data == "archive":
        await q.message.reply_text("Выберите период: отправьте команду, например:\n/period 2025-08")
        return

    if data.startswith("send|"):
        fname = data.split("|", 1)[1]

        # Ветвление: PDF или HTML
        if fname.lower().endswith(".pdf"):
            pdf_path = PDF_DIR / fname
            if not pdf_path.exists():
                await q.message.reply_text("PDF не найден.")
                return
            await q.message.reply_document(
                document=pdf_path.open("rb"),
                filename=pdf_path.name,
                caption=caption_for(pdf_path),
                reply_markup=build_keyboard_for(pdf_path),
            )
            return

        # HTML — ищем в архиве отчётов
        path = next((p for p in find_html_files(repo) if p.name == fname), None)

        # если не нашли файл «Аналитика ИИ» — создадим заглушку
        if path is None and fname.startswith("debt_ai_"):
            probe = repo / fname
            created = ensure_ai_variant(probe)
            if created and created.exists():
                path = created

        if not path:
            await q.message.reply_text("Файл не найден в архиве.")
            return

        await q.message.reply_document(
            document=path.open("rb"),
            filename=path.name,
            caption=caption_for(path),
            reply_markup=build_keyboard_for(path),
        )

# ───────────────────── скан и рассылка «новых» ────────────────────

def scan_new_debt_files(
    reports_dir: Path,
    managers_synonyms: Dict[str, List[str]],
    admins: List[str],
    watchers: Dict[str, List[str]],
    only_latest: bool = False,
    admin_only: bool = False,
) -> List[Tuple[Path, List[str]]]:
    """
    Возвращает список (файл, получатели) только для 'новых' файлов.
    • 'Новый' = его SHA-1 нет в индексе отправленных.
    • only_latest=True: берём только самый свежий файл per manager.
    • admin_only=True: рассылаем только админам (для тестов).
    """
    sent_index = load_json(SENT_INDEX_PATH, {})
    seen_sha: set[str] = set()
    candidates: List[Tuple[Path, List[str], str]] = []

    for p in find_html_files(reports_dir):
        if detect_report_type(p) != "debt":
            continue
        sha = sha1_of_file(p)
        if sha in sent_index or sha in seen_sha:
            continue

        if admin_only:
            recips = admins[:]  # только админу(ам)
        else:
            recips = compute_recipients(
                path=p,
                managers_synonyms=managers_synonyms,
                admins=admins,
                watchers=watchers,
            )

        mgr = detect_manager_from_filename(p, managers_synonyms) or "summary"
        candidates.append((p, recips, mgr))
        seen_sha.add(sha)

    if not only_latest:
        return [(p, r) for (p, r, _mgr) in candidates]

    latest_by_mgr: Dict[str, Tuple[float, Path, List[str]]] = {}
    for p, recips, mgr in candidates:
        ts = p.stat().st_mtime
        cur = latest_by_mgr.get(mgr)
        if cur is None or ts > cur[0]:
            latest_by_mgr[mgr] = (ts, p, recips)

    return [(p, r) for (_ts, p, r) in latest_by_mgr.values()]

def mark_sent(path: Path, recipients: List[str]) -> None:
    idx = load_json(SENT_INDEX_PATH, {})
    sha = sha1_of_file(path)
    idx[sha] = {
        "file": str(path.name),
        "first_seen": now_iso_tz(),
        "sent_to": recipients,
    }
    save_json(SENT_INDEX_PATH, idx)

async def send_batch(paths_and_recipients: List[Tuple[Path, List[str]]], app: Application, chats: Dict[str, int], verbose: bool, rate_delay: float):
    for path, recips in paths_and_recipients:
        kb = build_keyboard_for(path)
        cap = caption_for(path)
        if verbose:
            print(f"→ {path.name}: получателей {len(recips)}")
        for user in recips:
            # user может быть "@username" ИЛИ числовой chat_id (строкой)
            chat_id = None
            if isinstance(user, str) and user.startswith("@"):
                chat_id = chats.get(user)  # нужна привязка /start
            else:
                try:
                    chat_id = int(user)     # шлём напрямую по chat_id
                except Exception:
                    chat_id = chats.get(user)

            if not chat_id:
                if verbose:
                    print(f"   ! нет chat_id для {user}")
                continue

            try:
                await app.bot.send_document(
                    chat_id=chat_id,
                    document=path.open("rb"),
                    filename=path.name,
                    caption=cap,
                    reply_markup=kb,
                )
                if verbose:
                    print(f"   ✓ {user}")
            except Exception as e:
                with (LOGS_DIR / "telegram_errors.log").open("a", encoding="utf-8") as fh:
                    fh.write(f"{now_iso_tz()} SEND ERR {user} {path.name}: {e}\n")
                if verbose:
                    print(f"   ✗ {user}: {e}")
            await asyncio.sleep(rate_delay)
        mark_sent(path, recips)

# ───────────────────────────── main ───────────────────────────────

def load_config():
    # managers.yaml — используем как есть (совместимость)
    cfg = load_yaml(CONFIG_DIR / "managers.yaml")
    tz = cfg.get("timezone", "Asia/Almaty")
    admins = cfg.get("admins", [])
    managers: Dict[str, List[str]] = cfg.get("managers", {})
    watchers: Dict[str, List[str]] = cfg.get("watchers", {})
    return tz, admins, managers, watchers

def parse_args():
    ap = argparse.ArgumentParser(description=f"Рассылка HTML дебиторки в Telegram | {__VERSION__}")
    ap.add_argument("--reports-dir", default=str(REPORTS_DIR_DEFAULT), help="Где искать HTML отчёты (по умолчанию reports/html)")
    ap.add_argument("--send-all-now", action="store_true", help="Отправить новые версии сейчас")
    ap.add_argument("--only-latest", action="store_true", help="Только самый свежий файл на менеджера")
    ap.add_argument("--admin-only", action="store_true", help="Только админам (режим теста)")
    ap.add_argument("--run-bot", action="store_true", help="Запустить бота (/start, /period)")
    ap.add_argument("--verbose", action="store_true", help="Логировать процесс отправки (stdout)")
    return ap.parse_args()

async def _send_all_now_async(reports_dir: Path, token: str, to_send, chats, verbose: bool, rate_delay: float):
    app = Application.builder().token(token).build()
    app.bot_data["reports_dir"] = reports_dir
    await app.initialize()
    await send_batch(to_send, app, chats, verbose=verbose, rate_delay=rate_delay)
    await app.shutdown()

def main():
    load_dotenv(ROOT / ".env")  # TZ, TELEGRAM_BOT_TOKEN, TG_RATE_DELAY
    args = parse_args()

    reports_dir = Path(args.reports_dir).expanduser().resolve()
    _tz, admins, managers, watchers = load_config()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    rate_delay = float(os.getenv("TG_RATE_DELAY", "0.7"))
    verbose = bool(args.verbose)

    # разовая отправка (под планировщик)
    if args.send_all_now:
        if not token:
            print("❌ Нет TELEGRAM_BOT_TOKEN в .env")
            return
        to_send = scan_new_debt_files(
            reports_dir, managers, admins, watchers,
            only_latest=args.only_latest, admin_only=args.admin_only
        )
        if not to_send:
            print("Нет новых файлов дебиторки для отправки.")
        else:
            chats = chats_map()
            if verbose:
                print(f"Найдено к отправке: {len(to_send)} файл(ов)")
            asyncio.run(_send_all_now_async(
                reports_dir, token, to_send, chats, verbose=verbose, rate_delay=rate_delay
            ))

    # бот (для /start, /period и кнопок)
    if args.run_bot:
        if not token:
            print("❌ Нет TELEGRAM_BOT_TOKEN в .env")
            return
        app = Application.builder().token(token).build()
        app.bot_data["reports_dir"] = reports_dir
        app.add_handler(CommandHandler("start", cmd_start))
        app.add_handler(CommandHandler("help", cmd_help))
        app.add_handler(CommandHandler("period", cmd_period))
        app.add_handler(CallbackQueryHandler(cb_handler))
        print("🤖 Bot is running. Отправьте /start в Telegram…")
        app.run_polling()

if __name__ == "__main__":
    main()

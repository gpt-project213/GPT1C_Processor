#!/usr/bin/env python
# coding: utf-8
"""
imap_fetcher.py
Version: v4.4.4 (2026-03-10, Asia/Almaty) - ИСПРАВЛЕН ВЫЗОВ utils_excel

Назначение:
- Разовый цикл IMAP: скачать вложения .xlsx/.xls из белого списка отправителей,
  сохранить в reports/queue/ под ИСХОДНЫМ именем (не нормализуем),
  создать clean-копию через utils_excel.ensure_clean_xlsx в reports/excel/clean/,
  удалить письма и очистить корзину.
- Единая загрузка .env + config/imap.json во всех режимах, с BOM и override.
- Супрелог: host/port/ssl/user/pass_set/tz, источник .env/config.
- Таймауты/ретраи при подключении/логине (до 5 попыток).
- Селектор корзины: перебор ["INBOX.Trash", "Trash", "Корзина", "Deleted Items"] + из конфигов.
- ★ ИСПРАВЛЕНО v4.4.3: Правильный вызов utils_excel.ensure_clean_xlsx() - 
  функция принимает 1 аргумент и возвращает путь к очищенному файлу.
- Фильтр "REQUIRE_MANAGER_IN_NAME=true" имеет исключения для СВОДНЫХ отчетов.
  Файлы с ключевыми словами ("Продажи", "Остатки", "Валовая", "Gross", "Sales", "Inventory") 
  скачиваются НЕЗАВИСИМО от наличия имени менеджера.

ВАЖНО:
- Имена вложений НЕ меняем (по согласованию).
- «Святые» модули/шаблоны не трогаем.
- TZ берём из .env TZ=Asia/Almaty (IANA); по умолчанию Asia/Almaty.

CLI:
  --once [--since YYYY-MM-DD]   : один цикл, скачивание/очистка
  --list-mailboxes              : вывести список ящиков
  --debug {0..4}                : 0=WARNING, 1=INFO, 2=DEBUG, 3..4=TRACE
"""

from __future__ import annotations

import argparse
import email
import imaplib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from email.header import decode_header
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv, dotenv_values

# Импорт для XML-очистки битых файлов 1С
import utils_excel

__version__ = "v4.4.3"

# ─────────────────────────────────────────────────────────────────────
# Пути/каталоги

ROOT = Path(__file__).resolve().parent
PRJ  = ROOT
LOGS = PRJ / "logs"
LOGS.mkdir(exist_ok=True)

REPORTS = PRJ / "reports"
QUEUE   = REPORTS / "queue"
CLEAN   = REPORTS / "excel" / "clean"
for p in (REPORTS, QUEUE, CLEAN):
    p.mkdir(parents=True, exist_ok=True)

# TZ из .env (если нет — дефолт Asia/Almaty)
TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))

# Логгер
logger = logging.getLogger("imap_fetcher")
logger.setLevel(logging.INFO)

_ts = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
_file_handler = logging.FileHandler(LOGS / f"email_{_ts}.log", encoding="utf-8")
_stream_handler = logging.StreamHandler(sys.stdout)

class AlmatyFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, ZoneInfo("Asia/Almaty"))
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")

# было:
# fmt = logging.Formatter("%(asctime)s, %(levelname)s %(message)s")
# стало:
fmt = AlmatyFormatter("%(asctime)s, %(levelname)s %(message)s")
_file_handler.setFormatter(fmt)
_stream_handler.setFormatter(fmt)
logger.addHandler(_file_handler)
logger.addHandler(_stream_handler)

# ─────────────────────────────────────────────────────────────────────
# Утилиты

def _trace_level(debug: int) -> int:
    if debug >= 2:
        return logging.DEBUG
    if debug == 1:
        return logging.INFO
    return logging.WARNING

def _load_env_and_cfg() -> Dict:
    """
    Загружаем .env (учитываем BOM через utf-8-sig, override=True) и config/imap.json.
    Возвращаем словарь cfg: host, port, ssl, username, password, whitelist, trash_mailboxes, tz, sources,
    require_manager_in_name, manager_names.
    """
    env_path = PRJ / ".env"
    env_exists = env_path.exists()

    # 1) Основная загрузка .env (override=True, BOM-safe)
    if env_exists:
        load_dotenv(dotenv_path=env_path, override=True, encoding="utf-8-sig")
    else:
        load_dotenv(override=True, encoding="utf-8-sig")

    # 2) Прямое чтение пар key->val (на случай, если переменная окружения кем-то перетёрта позже)
    env_vals = {}
    try:
        if env_exists:
            env_vals = dotenv_values(dotenv_path=env_path, encoding="utf-8-sig") or {}
    except Exception as e:
        logger.warning("dotenv_values read error: %s", e)
        env_vals = {}

    def _get_env(name: str, default: str = "") -> str:
        val = os.getenv(name, "")
        if val is None or val == "":
            val = env_vals.get(name, env_vals.get("\ufeff" + name, default))
        return (val if val is not None else default)

    def _get_bool(name: str, default: bool) -> bool:
        raw = _get_env(name, "")
        if raw == "":
            return default
        return str(raw).strip().lower() in ("1", "true", "yes", "on")

    cfg: Dict = {
        "host": _get_env("IMAP_HOST", "").strip(),
        "port": int(_get_env("IMAP_PORT", "993") or "993"),
        "ssl":  _get_bool("IMAP_SSL", True),
        "username": _get_env("IMAP_USERNAME", "").strip(),
        "password": _get_env("IMAP_PASSWORD", "").strip(),
        "whitelist": [],
        "trash_mailboxes": [],
        "tz": _get_env("TZ", "Asia/Almaty").strip() or "Asia/Almaty",
        "sources": {"env": str(env_path if env_exists else "(not found)"), "config_json": None},
        # ★ NEW
        "require_manager_in_name": _get_bool("REQUIRE_MANAGER_IN_NAME", False),
        "manager_names": [],
    }

    # 3) config/imap.json (если есть — дополняет/переопределяет)
    cfg_path = PRJ / "config" / "imap.json"
    if cfg_path.exists():
        try:
            j = json.loads(cfg_path.read_text(encoding="utf-8"))
            if "host" in j:      cfg["host"] = (j["host"] or cfg["host"])
            if "port" in j:      cfg["port"] = int(j["port"] or cfg["port"])
            if "ssl" in j:       cfg["ssl"]  = bool(j["ssl"])
            if "username" in j:  cfg["username"] = (j["username"] or cfg["username"])
            # пароль из json применяем только если в .env он пуст
            if not cfg["password"] and "password" in j:
                cfg["password"] = (j["password"] or cfg["password"])
            wl = j.get("whitelist") or []
            if isinstance(wl, list):
                cfg["whitelist"] = [s.strip().lower() for s in wl if s and isinstance(s, str)]
            tm = j.get("trash_mailboxes") or []
            if isinstance(tm, list):
                cfg["trash_mailboxes"] = [s.strip() for s in tm if s and isinstance(s, str)]
            cfg["sources"]["config_json"] = str(cfg_path)
        except Exception as e:
            logger.error("CONFIG read error: %s", e)

    # 4) Дополнение из .env CSV
    wl_env = _get_env("IMAP_WHITELIST", "")
    if wl_env:
        for s in wl_env.split(","):
            s = s.strip().lower()
            if s and s not in cfg["whitelist"]:
                cfg["whitelist"].append(s)

    tr_env = _get_env("IMAP_TRASH_MAILBOXES", "")
    if tr_env:
        for s in tr_env.split(","):
            s = s.strip()
            if s and s not in cfg["trash_mailboxes"]:
                cfg["trash_mailboxes"].append(s)

    # 5) Дефолтные корзины — добавим в начало для перебора
    defaults = ["INBOX.Trash", "Trash", "Корзина", "Deleted Items"]
    for s in reversed(defaults):
        if s not in cfg["trash_mailboxes"]:
            cfg["trash_mailboxes"].insert(0, s)

    # ★ NEW: загрузим список имён менеджеров
    cfg["manager_names"] = _load_manager_names()
    if cfg["require_manager_in_name"] and not cfg["manager_names"]:
        logger.warning("REQUIRE_MANAGER_IN_NAME=True, но список менеджеров пуст (config/managers.json). Фильтр будет проигнорирован.")

    return cfg

def _set_log_level(debug: int) -> None:
    logger.setLevel(_trace_level(debug))

def _fix_mojibake(s: str) -> str:
    """
    Исправляет «мойку» (mojibake): UTF-8 байты, декодированные как CP1251.

    Пример: «РіРѕРґ» (UTF-8 `год` прочитанный как CP1251) → «год».

    Алгоритм: re-encode строки как CP1251 и decode обратно как UTF-8.
    Если оба шага проходят без ошибок — результат, иначе исходная строка.
    """
    try:
        fixed = s.encode("cp1251").decode("utf-8")
        return fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def _decode_h(value) -> str:
    """
    Декодирует заголовок email (Subject, From, filename и т.п.)
    с поддержкой RFC 2047, а также исправлением мойки CP1251/UTF-8.
    """
    if not value:
        return ""
    parts = decode_header(value)
    out = ""
    for s, enc in parts:
        if isinstance(s, bytes):
            # Пробуем: заданная кодировка → UTF-8 → CP1251 → ignore
            decoded = None
            for charset in filter(None, [enc, "utf-8", "cp1251"]):
                try:
                    decoded = s.decode(charset)
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            out += decoded if decoded is not None else s.decode("utf-8", errors="replace")
        else:
            # str: может быть мойка (UTF-8 байты, декодированные как CP1251)
            out += _fix_mojibake(s)
    return out

def _sender_from(msg) -> str:
    frm = _decode_h(msg.get("From", ""))
    m = re.search(r"<([^>]+)>", frm)
    if m:
        return m.group(1).strip().lower()
    m2 = re.search(r"[\w.\-+]+@[\w.\-]+", frm)
    return (m2.group(0).lower() if m2 else frm.strip().lower())

def _select_mailbox(M: imaplib.IMAP4, name: str) -> bool:
    try:
        typ, _ = M.select(name)
        return (typ == "OK")
    except Exception:
        return False

def _expunge_mailbox(M: imaplib.IMAP4, mailbox: str) -> None:
    if _select_mailbox(M, mailbox):
        try:
            M.expunge()
            logger.info("TRASH cleaned: %s", mailbox)
        except Exception as e:
            logger.error("TRASH expunge error on %s: %s", mailbox, e)
    else:
        logger.info("TRASH skip %s: cannot SELECT", mailbox)

def _save_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def _ensure_clean_copy(src: Path) -> Optional[Path]:
    """
    Создаём clean-копию через utils_excel.ensure_clean_xlsx.
    Файл очищается в QUEUE, затем перемещается в CLEAN.
    """
    name = src.name
    if name.endswith(".__clean.xlsx"):
        return src
    
    source_path = QUEUE / name
    clean_name = f"{name}.__clean.xlsx"
    
    # защита от «двойного хвоста»
    if clean_name.endswith(".__clean.xlsx.__clean.xlsx"):
        clean_name = f"{name}"
    
    try:
        # ИСПРАВЛЕНО v4.4.3: utils_excel.ensure_clean_xlsx() возвращает путь
        cleaned = utils_excel.ensure_clean_xlsx(source_path)
        
        # Переместить из QUEUE в CLEAN
        final_path = CLEAN / clean_name
        if cleaned.exists():
            if final_path.exists():
                final_path.unlink()
            # FIX B4: WinError 17 кросс-диск — shutil.move вместо .rename()
            import shutil as _shutil
            _shutil.move(str(cleaned), str(final_path))
            logger.info("Clean copy: %s", final_path.relative_to(PRJ))
            return final_path
        else:
            logger.error("Clean copy not created for %s", name)
            return None
    except Exception as e:
        logger.error("Clean copy FAIL for %s: %s", name, e)
        return None

# ★ NEW: загрузка имён менеджеров и проверка вхождения в имя файла
def _load_manager_names() -> List[str]:
    """
    Читаем config/managers.json и берём ключи (имена менеджеров).
    Если файла нет/пусто — возвращаем [].
    """
    try:
        cfg_path = PRJ / "config" / "managers.json"
        if cfg_path.exists():
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                out = []
                for k in data.keys():
                    k = (k or "").strip()
                    if k:
                        out.append(k)
                return out
    except Exception as e:
        logger.error("MANAGERS read error: %s", e)
    return []

def _is_summary_report(name: str) -> bool:
    """
    ИСПРАВЛЕНО: Проверка, является ли файл сводным отчетом.
    Сводные отчеты скачиваются НЕЗАВИСИМО от фильтра менеджеров.
    """
    base = (name or "").lower()
    
    # Ключевые слова для сводных отчетов
    summary_keywords = [
        "остатки", "остаток", "inventory", 
        "продажи", "продаж", "sales", 
        "валовая", "валов", "gross", "рентабельность",
        "затраты", "затрат", "расход", "расходы", "expenses",  # ← v9.4.13: ЗАТРАТЫ БЕЗ МЕНЕДЖЕРА
        "сводный", "сводная", "summary",
        "все", "всем", "all",
        "общий", "общая", "total",
        "ведомость", "партии", "партиям", "складах", "складе",
        "товарам", "взаиморасчетам", "контрагентами", "детальный"
    ]
    
    # Специальные фразы для комплексной проверки
    special_phrases = [
        "ведомость по партиям товара",
        "ведомость по партиям", 
        "партии товара на складах",
        "партии товара на складе",
        "ведомость по товарам на складах",
        "ведомость по товарам",
        "ведомость_по_взаиморасчетам_с_контрагентами",
        "ведомость по взаиморасчетам с контрагентами",
        "взаиморасчеты с контрагентами",
        "детальный_по_взаиморасчетам_с_контрагентами",
        "детальный по взаиморасчетам с контрагентами"
    ]
    
    # Проверяем специальные фразы
    for phrase in special_phrases:
        if phrase in base:
            logger.info("SUMMARY REPORT detected: %s (phrase: %s)", name, phrase)
            return True
    
    # Проверяем ключевые слова
    for keyword in summary_keywords:
        if keyword in base:
            logger.info("SUMMARY REPORT detected: %s (keyword: %s)", name, keyword)
            return True
    
    return False

def _filename_has_manager(name: str, managers: List[str]) -> bool:
    """
    ИСПРАВЛЕНО: Проверка наличия имени менеджера в имени файла.
    Теперь с исключением для сводных отчетов.
    """
    # Сначала проверяем, является ли файл сводным отчетом
    if _is_summary_report(name):
        return True
    
    # Если не сводный отчет - проверяем наличие имени менеджера
    base = (name or "").lower()
    for m in managers:
        if (m or "").lower() in base:
            return True
    return False

def _imap_connect(cfg: Dict, max_retries: int = 5, login_timeout: int = 15) -> imaplib.IMAP4:
    """
    Подключаемся с ретраями и LOGIN. Возвращаем IMAP4/IMAP4_SSL.
    """
    host, port, use_ssl = cfg["host"], cfg["port"], cfg["ssl"]
    user, pwd = cfg["username"], cfg["password"]

    logger.info(
        "IMAP cfg: host=%s port=%s ssl=%s user=%s pass_set=%s tz=%s env=%s cfg=%s",
        host, port, use_ssl, user, bool(pwd), cfg.get("tz"),
        cfg["sources"].get("env"), cfg["sources"].get("config_json"),
    )

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            if use_ssl:
                M = imaplib.IMAP4_SSL(host=host, port=port, timeout=login_timeout)
            else:
                M = imaplib.IMAP4(host=host, port=port, timeout=login_timeout)
            M.login(user, pwd)
            logger.info("IMAP LOGIN OK (attempt %s/%s)", attempt, max_retries)
            return M
        except Exception as e:
            last_exc = e
            logger.error("IMAP connect/login failed (attempt %s/%s): %s", attempt, max_retries, e)
            time.sleep(min(2 * attempt, 5))
    raise last_exc or RuntimeError("IMAP connect/login failed")

def _search_since(M: imaplib.IMAP4, since: Optional[str]) -> List[bytes]:
    if not _select_mailbox(M, "INBOX"):
        raise RuntimeError("Cannot SELECT INBOX")
    if since:
        dt = datetime.strptime(since, "%Y-%m-%d")
        _MONTHS_IMAP = ("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec")
        imap_date = f"{dt.day:02d}-{_MONTHS_IMAP[dt.month-1]}-{dt.year}"
        criteria = f'(SINCE {imap_date})'
    else:
        criteria = "ALL"
    typ, data = M.search(None, criteria)
    if typ != "OK":
        return []
    return data[0].split() if data and data[0] else []

# ─────────────────────────────────────────────────────────────────────
# Режимы

def list_mailboxes(debug: int = 1) -> None:
    cfg = _load_env_and_cfg()
    _set_log_level(debug)
    try:
        M = _imap_connect(cfg)
    except Exception as e:
        logger.error("LIST: connect error: %s", e)
        return
    try:
        typ, mboxes = M.list()
        logger.info("LIST mailboxes typ=%s count=%s", typ, 0 if mboxes is None else len(mboxes))
        if mboxes:
            for raw in mboxes:
                try:
                    line = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
                except Exception:
                    line = str(raw)
                logger.info("BOX: %s", line)
    finally:
        try:
            M.logout()
        except Exception:
            pass

def run_once(since: Optional[str] = None, debug: int = 1) -> None:
    cfg = _load_env_and_cfg()
    _set_log_level(debug)

    whitelist = set(cfg.get("whitelist") or [])
    trash_list = cfg.get("trash_mailboxes") or []

    # Подключение
    try:
        M = _imap_connect(cfg)
    except Exception as e:
        logger.error("IMAP error: %s", e)
        return

    saved_count = 0
    try:
        ids = _search_since(M, since)
        logger.info("SEARCH found %s msgs", len(ids))
        for num in ids:
            try:
                typ, data = M.fetch(num, "(RFC822)")
                if typ != "OK" or not data:
                    logger.info("FETCH skip #%s: typ=%s", num, typ)
                    continue

                raw = data[0][1] if isinstance(data[0], tuple) else data[0]
                msg = email.message_from_bytes(raw)
                sender = _sender_from(msg)

                # белый список (если задан)
                if whitelist and sender not in whitelist:
                    logger.info("SKIP sender not whitelisted: %s", sender)
                    try:
                        M.store(num, "+FLAGS", "\\Seen")
                    except Exception:
                        pass
                    continue

                found_any = False
                for part in msg.walk():
                    if part.get_content_maintype() == "multipart":
                        continue
                    cd = (part.get("Content-Disposition") or "")
                    if "attachment" not in cd.lower():
                        continue
                    fname_raw = part.get_filename()
                    if not fname_raw:
                        continue
                    fname = _decode_h(fname_raw).strip()  # Только декод, без нормализации!
                    low = fname.lower()
                    if not (low.endswith(".xlsx") or low.endswith(".xls")):
                        continue

                    # ★ ИСПРАВЛЕНО: фильтр по имени менеджера в названии файла с исключением для сводных отчетов
                    if (cfg.get("require_manager_in_name") and cfg.get("manager_names")):
                        if not _filename_has_manager(fname, cfg["manager_names"]):
                            logger.info("SKIP no-manager-in-name: %s", fname)
                            try:
                                M.store(num, "+FLAGS", "\\Seen")
                            except Exception:
                                pass
                            continue

                    payload = part.get_payload(decode=True) or b""
                    if not payload:
                        logger.info("ATTACH empty: %s", fname)
                        continue

                    dst = QUEUE / fname
                    # если файл существует — добавим (n)
                    if dst.exists():
                        base = Path(fname)
                        root = base.stem
                        ext  = base.suffix
                        n = 1
                        while dst.exists():
                            n += 1
                            dst = QUEUE / f"{root} ({n}){ext}"

                    _save_bytes(dst, payload)
                    logger.info("SAVED: %s (from %s)", dst.relative_to(PRJ), sender)
                    found_any = True

                    # clean-копия (без двойных суффиксов)
                    try:
                        _ensure_clean_copy(dst)
                    except Exception as e:
                        logger.error("Clean copy error for %s: %s", dst.name, e)

                if found_any:
                    try:
                        M.store(num, "+FLAGS", "\\Deleted")
                        saved_count += 1
                        logger.info("MAIL flagged \\Deleted, msg #%s", num)
                    except Exception as e:
                        logger.error("MAIL delete flag error #%s: %s", num, e)
                else:
                    try:
                        M.store(num, "+FLAGS", "\\Seen")
                    except Exception:
                        pass

            except Exception as e:
                logger.error("MSG error #%s: %s", num, e)

        # INBOX EXPUNGE
        try:
            if _select_mailbox(M, "INBOX"):
                M.expunge()
                logger.info("INBOX EXPUNGE OK")
        except Exception as e:
            logger.error("INBOX expunge error: %s", e)

        # Очистка корзины: перебор вариаций
        for mb in trash_list:
            _expunge_mailbox(M, mb)

        logger.info("CYCLE DONE: saved=%s", saved_count)

    finally:
        try:
            M.logout()
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────
# CLI

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="imap_fetcher", description="IMAP fetch cycle")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--once", action="store_true", help="run once (download attachments)")
    g.add_argument("--list-mailboxes", action="store_true", help="list mailboxes and exit")

    p.add_argument("--since", type=str, help="YYYY-MM-DD, search SINCE date (with --once)")
    p.add_argument("--debug", type=int, default=1, choices=[0,1,2,3,4], help="verbosity: 0=warn,1=info,2=debug")
    return p

def main() -> None:
    ap = _build_argparser()
    args = ap.parse_args()

    logger.info("=== imap_fetcher v%s ===", __version__)
    if args.list_mailboxes:
        list_mailboxes(debug=args.debug)
        return
    if args.once:
        run_once(since=args.since, debug=args.debug)
        return

if __name__ == "__main__":
    main()
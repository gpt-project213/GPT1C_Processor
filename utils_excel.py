#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils_excel.py
version: v2.0 (2025-09-02)

Назначение: создать «чистую» копию XLSX с корректным xl/sharedStrings.xml
Инварианты ТЗ:
• Clean XLSX → reports/excel/<имя>.xlsx.__clean.xlsx
• После успешной подготовки удаляется исходник из reports/queue
• Логи: logs/<module>_YYYYMMDD_HHMMSS.log (формат '%(asctime)s, %(levelname)s %(message)s', TZ=Asia/Almaty)
• Бизнес-логика файла: менять только sharedStrings.xml, остальное byte-to-byte

Публичный API:
    ensure_clean_xlsx(path: Path, *, force_fix: bool = False) -> Path
"""

from __future__ import annotations

from pathlib import Path
import io
import logging
import time
import zipfile
import tempfile
import xml.etree.ElementTree as ET

# Проектные инварианты / логи по ТЗ
import config
from config import EXCEL_CLEAN_DIR, QUEUE_DIR, setup_logging

__VERSION__ = "utils_excel.py v2.0 — 2025-09-02"
log = setup_logging("utils_excel", level=logging.INFO)

# ── XML константы ─────────────────────────────────────────────
NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
SST_TAG = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sst"
SI_TAG  = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"
T_TAG   = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"

def _find_existing_sst_name(zf: zipfile.ZipFile) -> str | None:
    """Имя sharedStrings в архиве с учётом регистра (xl/sharedStrings.xml)."""
    for name in zf.namelist():
        if name.replace("\\", "/").lower() == "xl/sharedstrings.xml":
            return name
    return None

def _scan_max_string_index(zf: zipfile.ZipFile) -> int:
    """Макс. индекс <c t="s"><v>IDX</v></c> по всем xl/worksheets/*.xml."""
    max_idx = -1
    for zinfo in zf.infolist():
        name = zinfo.filename.replace("\\", "/")
        if not (name.startswith("xl/worksheets/") and name.endswith(".xml")):
            continue
        try:
            data = zf.read(zinfo)
            root = ET.fromstring(data)
            for c in root.iterfind(".//a:c[@t='s']", NS):
                v = c.find("a:v", NS)
                if v is not None and v.text is not None:
                    try:
                        idx = int(v.text)
                        if idx > max_idx:
                            max_idx = idx
                    except ValueError:
                        pass
        except Exception as e:
            log.warning("Не удалось разобрать лист %s: %s", name, e)
    return max_idx

def _read_or_create_sst(zf: zipfile.ZipFile) -> ET.Element:
    """Читает существующий sharedStrings.xml или создаёт пустой <sst/>."""
    name = _find_existing_sst_name(zf)
    if not name:
        return ET.Element(SST_TAG)
    try:
        data = zf.read(name)
        return ET.fromstring(data)
    except Exception as e:
        log.warning("sharedStrings.xml повреждён (%s) — создаю заново", e)
        return ET.Element(SST_TAG)

def _serialize_xml(elem: ET.Element) -> bytes:
    buf = io.BytesIO()
    ET.ElementTree(elem).write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue()

def _build_fixed_shared_strings(src_zip: zipfile.ZipFile) -> bytes:
    """
    Собрать корректный sharedStrings.xml:
      — дополнить <si> до (max_index + 1);
      — проставить count/uniqueCount = фактической длине.
    """
    max_idx  = _scan_max_string_index(src_zip)
    need_len = max_idx + 1 if max_idx >= 0 else 0

    sst_root = _read_or_create_sst(src_zip)
    cur_len  = len(list(sst_root.findall("a:si", NS)))

    if need_len > cur_len:
        for _ in range(need_len - cur_len):
            si = ET.Element(SI_TAG)
            t  = ET.SubElement(si, T_TAG)
            t.text = ""  # пустая строка
            sst_root.append(si)

    final_len = len(list(sst_root.findall("a:si", NS)))
    sst_root.set("count",       str(final_len))
    sst_root.set("uniqueCount", str(final_len))
    return _serialize_xml(sst_root)

def _safe_replace(src: Path, dst: Path, attempts: int = 10, delay: float = 0.25) -> None:
    """Надёжная замена файла под Windows: ретраи при PermissionError."""
    try:
        if dst.exists():
            dst.unlink()
    except Exception as e:
        log.warning("Удаление %s перед заменой: %s", dst, e)

    for i in range(1, attempts + 1):
        try:
            src.replace(dst)
            return
        except PermissionError:
            time.sleep(delay)
            # последняя попытка бросит исключение автоматически

def ensure_clean_xlsx(path: Path, *, force_fix: bool = False) -> Path:
    """
    Создать «чистую» копию XLSX с корректным sharedStrings.xml.
    Выход: reports/excel/<имя>.xlsx.__clean.xlsx
    После успешной подготовки удаляется исходник из reports/queue.
    """
    src = Path(path).resolve()
    if not src.exists() or src.suffix.lower() != ".xlsx":
        raise FileNotFoundError(src)

    EXCEL_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    clean = EXCEL_CLEAN_DIR / f"{src.name}.__clean.xlsx"

    if clean.exists() and not force_fix:
        log.info("Clean copy уже существует → %s", clean.name)
        return clean

    with zipfile.ZipFile(src, "r") as zin, tempfile.TemporaryDirectory() as td:
        temp_path = Path(td) / (src.stem + ".__tmp_clean.xlsx")
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            fixed_sst = _build_fixed_shared_strings(zin)
            existing_sst_name = _find_existing_sst_name(zin)

            for zinfo in zin.infolist():
                name = zinfo.filename.replace("\\", "/")
                # пропускаю оригинальный sharedStrings.xml (любой регистр)
                if existing_sst_name and name == existing_sst_name:
                    continue
                data = zin.read(zinfo)
                zout.writestr(zinfo, data)

            # пишу исправленный sharedStrings.xml
            zout.writestr("xl/sharedStrings.xml", fixed_sst)

        _safe_replace(temp_path, clean)

    log.info("Clean copy → %s", clean.name)

    # Удаление исходника из очереди (инвариант ТЗ)
    try:
        src_res = src.resolve()
        in_queue = any(parent.samefile(QUEUE_DIR) for parent in src_res.parents)
        if in_queue:
            src.unlink(missing_ok=True)
            log.info("Удалён оригинал из очереди → %s", src.name)
    except Exception as e:
        # не критично для пайплайна, оставляю в логе
        log.warning("Не удалось удалить оригинал из очереди: %s (%s)", src, e)

    return clean

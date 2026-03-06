#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils_excel.py
version: v2.3.3 (2026-03-01)

ИЗМЕНЕНИЯ v2.3.3:
• Bug #15 fix: _safe_replace теперь обрабатывает WinError 17 (cross-drive move C: → F:).
  Реализация: shutil.copy2 во временный файл рядом с dst (тот же диск) → атомарный replace.
  Старая PermissionError-retry логика (10 попыток) сохранена.

ИЗМЕНЕНИЯ v2.3.2:
• НЕ удаляем оригинал из queue после clean-копии.
  Причина: pipeline_task() ищет файлы в queue/ для запуска парсеров.
  Если оригинал удалён — queue всегда пуста, парсеры никогда не запускаются,
  аналитика не получает свежие JSON. После обработки pipeline сам перемещает
  файл в reports/excel/processed/.

ИЗМЕНЕНИЯ v2.3.1:
• ensure_clean_xlsx() теперь поддерживает 2-й позиционный аргумент out_path (совместимость с imap_fetcher)

ИЗМЕНЕНИЯ v2.3:
• Добавлена функция read_excel_as_text_rows() для совместимости с парсерами

ИЗМЕНЕНИЯ v2.2:
• Добавлена защита от двойного clean: если файл уже .__clean.xlsx → вернуть как есть
• Исправлена проблема KeyError: xl/sharedStrings.xml при повторной обработке

Назначение: создать «чистую» копию XLSX с корректным xl/sharedStrings.xml
Инварианты ТЗ:
• Clean XLSX → reports/excel/<имя>.xlsx.__clean.xlsx
• После успешной подготовки удаляется исходник из reports/queue
• Логи: logs/<module>_YYYYMMDD_HHMMSS.log (формат '%(asctime)s, %(levelname)s %(message)s', TZ=Asia/Almaty)
• Бизнес-логика файла: менять только sharedStrings.xml, остальное byte-to-byte

Публичный API:
    ensure_clean_xlsx(path: Path, out_path: Path | None = None, *, force_fix: bool = False) -> Path
    read_excel_as_text_rows(xlsx_path: Path, sheet: int = 0) -> List[List[str]]
"""

from __future__ import annotations

from pathlib import Path
import io
import logging
import time
import zipfile
import tempfile
import xml.etree.ElementTree as ET
from typing import List, Union, Optional

import pandas as pd

# Проектные инварианты / логи по ТЗ
import config
from config import EXCEL_CLEAN_DIR, QUEUE_DIR, setup_logging

__VERSION__ = "utils_excel.py v2.3.1 — 2026-02-18"
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
    """
    Надёжная замена файла под Windows.
    v2.3.3: ретраи при PermissionError (WinError 32) + cross-drive fix (WinError 17).

    Bug #15 fix: os.replace() не умеет cross-drive (C: → F:).
    Решение: shutil.copy2 во временный файл РЯДОМ с dst (тот же диск),
    затем атомарный os.replace(tmp, dst). Оригинал src удаляется после.
    """
    import shutil  # stdlib, всегда доступен

    # Предварительно удаляем dst если существует
    try:
        if dst.exists():
            dst.unlink()
    except Exception as e:
        log.warning("Удаление %s перед заменой: %s", dst, e)

    for _i in range(1, attempts + 1):
        try:
            src.replace(dst)
            return
        except PermissionError:
            # WinError 32: файл занят другим процессом — ждём и повторяем
            time.sleep(delay)
        except OSError as e:
            if hasattr(e, "winerror") and e.winerror == 17:
                # WinError 17: нельзя переместить на другой диск (C: → F:)
                # Копируем во временный файл НА ТОМ ЖЕ ДИСКЕ что и dst,
                # затем атомарно переименовываем (tmp и dst — один диск).
                tmp = dst.with_suffix(".__tmp_move")
                try:
                    shutil.copy2(str(src), str(tmp))   # C: → F: (копирование OK)
                    tmp.replace(dst)                    # F: → F: (атомарно)
                    try:
                        src.unlink()                    # удаляем оригинал из temp
                    except OSError:
                        pass  # temp-файл удалит ОС при следующем старте
                    return
                except Exception as copy_err:
                    log.error("Cross-drive copy fallback failed: %s", copy_err)
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
                    raise
            else:
                raise

    # Последняя попытка без перехвата (бросит если не получилось)
    src.replace(dst)


def ensure_clean_xlsx(path: Union[str, Path], out_path: Optional[Union[str, Path]] = None, *, force_fix: bool = False) -> Path:
    """
    Создать «чистую» копию XLSX с корректным sharedStrings.xml.

    По умолчанию выход: reports/excel/<имя>.xlsx.__clean.xlsx
    Если задан out_path — пишем ровно туда (без автоматического добавления .__clean.xlsx).

    После успешной подготовки удаляется исходник из reports/queue.

    v2.2: Если файл уже .__clean.xlsx → вернуть как есть (защита от двойного clean)
    v2.3.1: Поддержка out_path (2-й позиционный аргумент) для совместимости
    """
    src = Path(path).resolve()

    # v2.2: защита от двойного clean
    if src.name.endswith(".__clean.xlsx"):
        log.info("Файл уже очищен, возвращаем как есть → %s", src.name)
        return src

    if not src.exists() or src.suffix.lower() != ".xlsx":
        raise FileNotFoundError(src)

    EXCEL_CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    if out_path is not None:
        clean = Path(out_path).resolve()
        clean.parent.mkdir(parents=True, exist_ok=True)
    else:
        clean = (EXCEL_CLEAN_DIR / f"{src.name}.__clean.xlsx").resolve()

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

    # v2.3.2: Оригинал из queue НЕ удаляем — pipeline_task должен его найти и обработать.
    # После обработки pipeline сам перемещает файл в processed/.
    # (Ранее удаление приводило к тому, что queue всегда пустая и парсеры не запускались.)

    return clean


# ──────────────────────────────────────────────────────────────────
# read_excel_as_text_rows (v2.3)
# ──────────────────────────────────────────────────────────────────

def read_excel_as_text_rows(
    xlsx_path: Union[str, Path],
    sheet: int = 0,
    max_rows: int = 5000,
    max_cols: int = 50
) -> List[List[str]]:
    """
    Прочитать Excel файл как список строк со строковыми значениями.

    Args:
        xlsx_path: Путь к XLSX файлу
        sheet: Номер листа (0 = первый)
        max_rows: Максимум строк для чтения
        max_cols: Максимум колонок для чтения

    Returns:
        List[List[str]]: Список строк, где каждая строка - список строковых значений
    """
    p = Path(xlsx_path) if not isinstance(xlsx_path, Path) else xlsx_path

    df = pd.read_excel(
        p,
        sheet_name=sheet,
        header=None,
        dtype=object,
        engine="openpyxl"
    )

    df = df.iloc[:max_rows, :max_cols]

    rows: List[List[str]] = []
    for _, row in df.iterrows():
        out_row = []
        for v in row.tolist():
            if v is None or (isinstance(v, float) and pd.isna(v)):
                out_row.append("")
            else:
                out_row.append(str(v))
        rows.append(out_row)

    return rows
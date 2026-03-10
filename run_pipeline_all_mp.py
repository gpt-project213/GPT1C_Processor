# run_pipeline_all_mp.py · v1.5 · Asia/Almaty · 09.03.2026
# Оркестратор всех типов отчётов: DEBT / SALES / GROSS / INVENTORY / EXPENSE
# Fix #PIPE-1: добавлен тип EXPENSE (RE_EXPENSE_NAME, _classify_by_content, process_expenses_file)
# Fix #PIPE-1: else-ветка в _process_one теперь SKIP вместо fallback на DEBT
# - Берёт файлы из reports/queue
# - Claim через переименование *.xlsx -> *.xlsx.work (исключает двойной захват)
# - Копия в reports/excel/active для устойчивой обработки
# - Маршрутизация по имени и содержимому (peek первых ~50 строк)
# - Вызов соответствующих билдеров
# - Перенос исходников в reports/excel/processed
# - Никакого fallback в DEBT для SALES/GROSS/INVENTORY
# - «Тихий skip» при недоступности модулей/ошибках билдеров
# - Опциональный вызов imap_fetcher.py --once перед запуском цикла

from __future__ import annotations

import os
import re
import sys
import time
import json
import shutil
import queue as pyqueue
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional, List, Tuple, Iterable, Dict, Any, Union

# ─────────────────────────────────────────────────────────────────────
# Пути проекта
PRJ = Path(__file__).resolve().parent
REPORTS = PRJ / "reports"
QUEUE_DIR = REPORTS / "queue"
HTML_DIR = REPORTS / "html"
AI_DIR = REPORTS / "ai"
JSON_DIR = REPORTS / "json"
EXCEL_DIR = REPORTS / "excel"
ACTIVE_DIR = EXCEL_DIR / "active"
PROCESSED_DIR = EXCEL_DIR / "processed"
CLEAN_DIR = EXCEL_DIR / "clean"
LOGS_DIR = PRJ / "logs"

for d in (REPORTS, QUEUE_DIR, HTML_DIR, AI_DIR, JSON_DIR, EXCEL_DIR, ACTIVE_DIR, PROCESSED_DIR, CLEAN_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# Логирование
def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Almaty")).strftime("%Y-%m-%d %H:%M:%S")

def _log(msg: str, *, err: bool = False, extra: Dict[str, Any] | None = None) -> None:
    line = f"{_now()} {'ERROR' if err else 'INFO'} {msg}"
    print(line)
    try:
        with (LOGS_DIR / "run_pipeline_all_mp.log").open("a", encoding="utf-8") as _f:
            _f.write(line + "\n")
    except Exception:
        pass

def _json_event(event: str, **kwargs) -> None:
    payload = {"event": event, **kwargs}
    print(f"{_now()}, INFO {json.dumps(payload, ensure_ascii=False)}")

# ─────────────────────────────────────────────────────────────────────
# Импорты билдера с «тихим skip»
# ВАЖНО: ничего не ломаем, только расширяем.

try:
    from debt_auto_report import build_report as build_report_debt  # -> Path|str
except Exception as e:
    build_report_debt = None  # type: ignore
    _log(f"[IMPORT] debt_auto_report not available: {e}")

try:
    from sales_report import build_report as build_report_sales  # -> Path | (Path, Path)
except Exception as e:
    build_report_sales = None  # type: ignore
    _log(f"[IMPORT] sales_report not available (skip SALES): {e}")

try:
    from gross_report import build_gross_report  # -> Path
except Exception as e:
    build_gross_report = None  # type: ignore
    _log(f"[IMPORT] gross_report not available (skip GROSS sums): {e}")

try:
    # в вашем модуле функция называется build_gross_report_percent
    from gross_report_pct import build_gross_report_percent as build_gross_pct_report  # -> Path
except Exception as e:
    build_gross_pct_report = None  # type: ignore
    _log(f"[IMPORT] gross_report_pct not available (skip GROSS pct): {e}")

try:
    # в вашем модуле INVENTORY функция называется build_report
    from inventory import build_report as build_inventory_report  # -> Path | (Path, Path)
except Exception as e:
    build_inventory_report = None  # type: ignore
    _log(f"[IMPORT] inventory not available (skip INVENTORY): {e}")

try:
    from expenses_parser import build_report as build_expenses_report  # -> Path
except Exception as e:
    build_expenses_report = None  # type: ignore
    _log(f"[IMPORT] expenses_parser not available (skip EXPENSE): {e}")

# ─────────────────────────────────────────────────────────────────────
# Регэкспы маршрутизации
RE_SALES_NAME = re.compile(r"(sales|продаж|выруч|реализац)", re.I)
RE_GROSS_NAME = re.compile(r"(gross|валов|маржа|рентаб|прибыл)", re.I)
# Расширено: «Ведомость по товарам на складах», «остатки всем», «склад»
RE_INV_NAME = re.compile(r"(остат|склад|товарам?\s+на\s+складах|ведомост[ьи]\s+по\s+товарам|inventory|stock|резерв|остатки\s+всем)", re.I)
RE_DEBT_NAME = re.compile(r"(debt|дебит|взаиморасч|контрагент|дебитор)", re.I)
RE_EXPENSE_NAME = re.compile(r"(затрат|расход|expense|costs?)", re.I)  # Fix #PIPE-1

# ─────────────────────────────────────────────────────────────────────
# Утилиты ФС
def _claim(src: Path) -> Optional[Path]:
    """
    Переименовывает файл очереди -> .work, чтобы единовременно обрабатывал только один воркер.
    """
    if not src.exists():
        return None
    work = src.with_suffix(src.suffix + ".work")
    try:
        src.rename(work)
        return work
    except Exception as e:
        _log(f"[CLAIM] fail {src.name}: {e}", err=False)
        return None

def _release(work: Path) -> None:
    """Удаляет .work после завершения."""
    try:
        if work.exists():
            work.unlink()
    except Exception:
        pass

def _copy_to_active(work: Path) -> Path:
    """
    Стабильная копия источника в excel/active.
    ВАЖНО: снимаем хвост '.work', чтобы билдеры получили валидный *.xlsx.
    """
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    # если имя заканчивается на '.xlsx.work' — обрежем '.work'
    if work.name.lower().endswith(".xlsx.work"):
        dst_name = work.name[:-5]  # убираем суффикс ".work"
    else:
        dst_name = work.name
    dst = ACTIVE_DIR / dst_name
    shutil.copy2(work, dst)
    return dst

def _cleanup_active(active_copy: Path) -> None:
    try:
        if active_copy.exists():
            active_copy.unlink()
    except Exception:
        pass

def _move_to_processed(work: Path, original_name: str) -> None:
    """
    Перемещает .work файл в processed/ под оригинальным именем + временна́я метка.
    Fix #1: раньше искал queue/foo.xlsx, которого уже нет (переименован в foo.xlsx.work).
    Теперь принимает work-путь напрямую и перемещает его.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not work.exists():
        return
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = PROCESSED_DIR / f"{original_name}__{ts}"
        shutil.move(str(work), str(dst))
    except Exception as e:
        _log(f"[FS] move to processed failed: {original_name}: {e}", err=False)
# ─────────────────────────────────────────────────────────────────────
# Пик содержимого для эвристик (до 50 строк/ячеек)
def _peek_excel_text(path: Path, max_rows: int = 50) -> str:
    """
    Пытается прочитать первые max_rows строк первого листа и склеить в текст для эвристик.
    Тихо падает в пустоту при ошибках.
    """
    try:
        import pandas as pd
        df = pd.read_excel(path, nrows=max_rows, header=None, engine=None)
        # склеиваем в одну строку
        vals: List[str] = []
        for _, row in df.iterrows():
            for v in row.tolist():
                if isinstance(v, str):
                    vals.append(v)
        return "\n".join(vals).lower()
    except Exception:
        return ""

def _classify_by_name(name: str) -> Optional[str]:
    n = name.lower()
    if RE_INV_NAME.search(n):
        return "INVENTORY"
    if RE_EXPENSE_NAME.search(n):  # Fix #PIPE-1: проверяем до SALES/GROSS (затраты ≠ продажи)
        return "EXPENSE"
    if RE_SALES_NAME.search(n):
        return "SALES"
    if RE_GROSS_NAME.search(n):
        return "GROSS"
    if RE_DEBT_NAME.search(n):
        return "DEBT"
    return None

def _classify_by_content(path: Path) -> Optional[str]:
    """
    Порядок важен: DEBT → INVENTORY → EXPENSE → SALES → GROSS.
    Это исключает увод «Остатков» в SALES на слове «товар».
    """
    low = _peek_excel_text(path, max_rows=50)
    if not low:
        return None
    # Cash/Bank filter: skip routing for cash-like ledgers
    if any(k in low for k in (
        "ведомость по денежным средствам", "по денежным средствам", "касс", "банк", "выписка", "платеж", "платёж"
    )):
        return "SKIP"
    if any(k in low for k in ("дебитор", "debt", "accounts receivable", "кредитор", "задолж", "взаиморасч", "контрагент")):
        return "DEBT"
    if any(k in low for k in ("остат", "склад", "товарам на складах", "ведомость по товарам", "резерв", "на складе")):
        return "INVENTORY"
    if any(k in low for k in ("затрат", "расход", "expense", "статья затрат", "вид расходов")):  # Fix #PIPE-1
        return "EXPENSE"
    if any(k in low for k in ("товар", "номенк", "покупател", "клиент", "колич", "сумма продаж", "выруч", "реализац")):
        return "SALES"
    if any(k in low for k in ("себестоим", "валовая прибыль", "маржа", "рентаб")):
        return "GROSS"
    return None

def _route_file(path: Path) -> str:
    """
    Классификация: по имени, затем по содержимому, иначе DEBT по умолчанию.
    """
    typ = _classify_by_name(path.name)
    if typ:
        return typ
    typ = _classify_by_content(path)
    if typ:
        return typ
    return "SKIP"

# ────────────────────────────────────────────────────────────
# Процессоры типов
def _coerce_path(res: Union[str, Path, Tuple[Union[str, Path], Union[str, Path]], None]) -> List[Path]:
    out: List[Path] = []
    if res is None:
        return out
    if isinstance(res, (str, Path)):
        p = Path(res)
        if p.exists():
            out.append(p)
        return out
    if isinstance(res, tuple) and len(res) == 2:
        for r in res:
            if r:
                p = Path(r)
                if p.exists():
                    out.append(p)
        return out
    return out

def process_debt_file(work: Path) -> List[Path]:
    outs: List[Path] = []
    if build_report_debt is None:
        _log(f"[DEBT] skip (builder not available): {work.name}")
        return outs
    try:
        a = _copy_to_active(work)
    except FileNotFoundError:
        _log(f"[DEBT] skip (source vanished during copy): {work.name}")
        return outs
    try:
        res = build_report_debt(a)  # type: ignore
        outs.extend(_coerce_path(res))
        if outs:
            _log(f"[DEBT] ok {work.name}: {', '.join(p.name for p in outs)}")
    except Exception as e:
        _log(f"[DEBT] fail {work.name}: {e}", err=False)
    finally:
        _cleanup_active(a)
    return outs

def process_sales_file(work: Path) -> List[Path]:
    outs: List[Path] = []
    if build_report_sales is None:
        _log(f"[SALES] skip (builder not available): {work.name}")
        return outs
    try:
        a = _copy_to_active(work)
    except FileNotFoundError:
        _log(f"[SALES] skip (source vanished during copy): {work.name}")
        return outs
    try:
        res = build_report_sales(a)  # type: ignore
        new = _coerce_path(res)
        if new:
            outs.extend(new)
            _log(f"[SALES] ok {work.name}: {', '.join(p.name for p in new)}")
    except Exception as e:
        _log(f"[SALES] fail {work.name}: {e}", err=False)
    finally:
        _cleanup_active(a)
    return outs

def process_gross_file(work: Path) -> List[Path]:
    outs: List[Path] = []
    try:
        a = _copy_to_active(work)
    except FileNotFoundError:
        _log(f"[GROSS] skip (source vanished during copy): {work.name}")
        return outs
    try:
        if build_gross_report is not None:
            p1 = build_gross_report(a)  # type: ignore
            outs.extend(_coerce_path(p1))
        else:
            _log(f"[GROSS] sums builder not available", err=False)
        if build_gross_pct_report is not None:
            p2 = build_gross_pct_report(a)  # type: ignore
            outs.extend(_coerce_path(p2))
        else:
            _log(f"[GROSS] pct builder not available", err=False)
        if outs:
            _log(f"[GROSS] ok {work.name}: {', '.join(p.name for p in outs)}")
        else:
            _log(f"[GROSS] skip (no outputs): {work.name}")
    except Exception as e:
        _log(f"[GROSS] fail {work.name}: {e}", err=False)
    finally:
        _cleanup_active(a)
    return outs

def process_inventory_file(work: Path) -> List[Path]:
    outs: List[Path] = []
    if build_inventory_report is None:
        _log(f"[INVENTORY] skip (not implemented): {work.name}")
        return outs
    try:
        a = _copy_to_active(work)
    except FileNotFoundError:
        _log(f"[INVENTORY] skip (source vanished during copy): {work.name}")
        return outs
    try:
        res = build_inventory_report(a)  # type: ignore
        new = _coerce_path(res)
        if new:
            outs.extend(new)
            _log(f"[INVENTORY] ok {work.name}: {', '.join(p.name for p in new)}")
        else:
            _log(f"[INVENTORY] skip (no outputs): {work.name}")
    except Exception as e:
        _log(f"[INVENTORY] fail {work.name}: {e}", err=False)
    finally:
        _cleanup_active(a)
    return outs

def process_expenses_file(work: Path) -> List[Path]:  # Fix #PIPE-1
    outs: List[Path] = []
    if build_expenses_report is None:
        _log(f"[EXPENSE] skip (builder not available): {work.name}")
        return outs
    try:
        a = _copy_to_active(work)
    except FileNotFoundError:
        _log(f"[EXPENSE] skip (source vanished during copy): {work.name}")
        return outs
    try:
        res = build_expenses_report(a)  # type: ignore
        new = _coerce_path(res)
        if new:
            outs.extend(new)
            _log(f"[EXPENSE] ok {work.name}: {', '.join(p.name for p in new)}")
        else:
            _log(f"[EXPENSE] skip (no outputs): {work.name}")
    except Exception as e:
        _log(f"[EXPENSE] fail {work.name}: {e}", err=False)
    finally:
        _cleanup_active(a)
    return outs

# ─────────────────────────────────────────────────────────────────────
# Основной цикл
def _iter_queue() -> List[Path]:
    """Список *.xlsx из очереди (без *.work).
    Fix #6: p.stat() обёрнут в try/except — файл может быть захвачен (.work)
    между glob() и stat(), что роняло весь цикл с FileNotFoundError."""
    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except FileNotFoundError:
            return 0.0
    candidates = [p for p in QUEUE_DIR.glob("*.xlsx") if not p.name.endswith(".work")]
    return sorted(candidates, key=_mtime)

def _process_one(src: Path) -> Tuple[str, List[Path]]:
    """
    Обработка одного файла: claim -> route -> process -> move to processed
    Возвращает (тип, список output-путей)
    """
    work = _claim(src)
    if work is None:
        return ("SKIP", [])
    routed_to = _route_file(work)
    _json_event("ROUTING", file=src.name, to=routed_to)
    try:
        if routed_to == "DEBT":
            outs = process_debt_file(work)
        elif routed_to == "SALES":
            outs = process_sales_file(work)
        elif routed_to == "GROSS":
            outs = process_gross_file(work)
        elif routed_to == "INVENTORY":
            outs = process_inventory_file(work)
        elif routed_to == "EXPENSE":  # Fix #PIPE-1
            outs = process_expenses_file(work)
        else:
            # SKIP или неизвестный тип — не обрабатываем
            _log(f"[SKIP] {src.name} (route={routed_to})")
            outs = []
        # перенос .work файла в processed/
        _move_to_processed(work, src.name)
        return (routed_to, outs)
    finally:
        _release(work)

def _imap_once() -> None:
    """
    Опционально: забрать письма разово перед циклом.
    Управляется переменной окружения RUN_IMAP_ONCE=1.
    """
    if os.environ.get("RUN_IMAP_ONCE", "1") not in ("1", "true", "True"):
        return
    try:
        _log("== IMAP fetch_once ==")
        exe = sys.executable
        cmd = [exe, "imap_fetcher.py", "--once"]
        _log(f"RUN: {exe} imap_fetcher.py --once")
        out = subprocess.run(cmd, cwd=str(PRJ), capture_output=True, text=True, timeout=600)
        # Логируем stdout частями как INFO
        for line in (out.stdout or "").splitlines():
            _log(line)
    except Exception as e:
        _log(f"IMAP once failed: {e}", err=False)

def run_once() -> Tuple[int, int]:
    """
    Один проход очереди. Возвращает (processed, failed)
    """
    files = _iter_queue()
    if not files:
        _log("QUEUE is empty")
        return (0, 0)

    processed = 0
    failed = 0
    for src in files:
        _log(f"START {src.name}")
        typ, outs = _process_one(src)
        if outs:
            processed += 1
        else:
            # отсутствие выходов не считаем фатальной ошибкой для SALES/GROSS/INVENTORY
            # но для DEBT это чаще всего ошибка парсинга, считаем failed
            if typ == "DEBT":
                failed += 1
        _log(f"FINISH {src.name}: type={typ}, outs={len(outs)}")
    return (processed, failed)

# ─────────────────────────────────────────────────────────────────────
# CLI
def main(argv: List[str]) -> int:
    _json_event("boot", prj=str(PRJ), reports=str(REPORTS), html=str(HTML_DIR))
    _imap_once()
    processed, failed = run_once()
    _log(f"QUEUE DONE: processed={processed}, failed={failed}")
    _json_event("pipeline_finished", processed=processed, failed=failed)
    return 0

if __name__ == "__main__":
    try:
        rc = main(sys.argv[1:])
    except KeyboardInterrupt:
        rc = 130
    except Exception as e:
        _log(f"PIPELINE CRASH: {e}", err=True)
        rc = 1
    print(f"{_now()}, INFO {{\"event\": \"pipeline_rc\", \"rc\": {rc}}}")
    sys.exit(rc)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expenses_parser_final.py – Универсальный парсер отчётов 1С "Затраты" (расходы)
Особенности:
- Поддержка отчётов за день ("18 февраля 2026 г.") и за период ("01.02.2026 - 17.02.2026")
- Автоопределение типа периода (DAY, MTD, MONTH, RANGE)
- Группировка расходов по подразделениям
- Сортировка подразделений по убыванию итоговой суммы
- Сортировка детальных строк внутри подразделения по убыванию суммы
- Генерация JSON и детализированного HTML-отчёта
- Подробное логирование в папку logs/
"""

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

# Константы
NBSP_NARROW = "\u202F"  # узкий неразрывный пробел (используется в 1С)

# Словарь месяцев (именительный и родительный падежи)
MONTHS_RU = {
    "январь": 1, "января": 1,
    "февраль": 2, "февраля": 2,
    "март": 3, "марта": 3,
    "апрель": 4, "апреля": 4,
    "май": 5, "мая": 5,
    "июнь": 6, "июня": 6,
    "июль": 7, "июля": 7,
    "август": 8, "августа": 8,
    "сентябрь": 9, "сентября": 9,
    "октябрь": 10, "октября": 10,
    "ноябрь": 11, "ноября": 11,
    "декабрь": 12, "декабря": 12,
}


def _safe_isna(x: Any) -> bool:
    """Безопасная проверка на NaN/None."""
    try:
        return pd.isna(x)
    except Exception:
        return x is None


@dataclass
class PeriodInfo:
    """Информация о периоде отчёта."""
    period_str: str
    period_start: Optional[date]
    period_end: Optional[date]
    report_type: str  # DAY, MTD, MONTH, RANGE


class UnifiedExpensesParser:
    """
    Основной класс парсера.
    """

    def __init__(self, out_root: Optional[Path] = None, timezone: str = "Asia/Qyzylorda"):
        self.out_root = Path(out_root) if out_root else Path.cwd()
        self.timezone = timezone

        self.html_dir = self.out_root / "reports" / "html"
        self.json_dir = self.out_root / "reports" / "json"
        self.logs_dir = self.out_root / "logs"

        for d in (self.html_dir, self.json_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logging()

        # Ключевые слова для строки общего итога
        self.grand_total_keywords = ("итог", "итого", "всего", "total", "result")

    # ------------------------------------------------------------------
    # Вспомогательные методы (время, логирование, нормализация)
    # ------------------------------------------------------------------

    def _now_tz(self) -> datetime:
        """Текущее время с учётом часового пояса."""
        if ZoneInfo is None:
            return datetime.now()
        try:
            return datetime.now(ZoneInfo(self.timezone))
        except Exception:
            return datetime.now()

    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования в файл и на консоль."""
        logger = logging.getLogger(f"expenses.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        ts = self._now_tz().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"expenses_{ts}.log"

        fmt = logging.Formatter("%(asctime)s, %(levelname)s %(message)s")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)

        logger.handlers = [fh, sh]
        return logger

    def _normalize_text(self, text: Any) -> str:
        """Очистка текста: удаление неразрывных пробелов, лишних пробелов."""
        if _safe_isna(text):
            return ""
        s = str(text)
        s = re.sub(r"[\u00A0\u202F\u2007\u200B]", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def _normalize_number(self, value: Any) -> Optional[float]:
        """Преобразование строки в число (с учётом российского формата)."""
        if _safe_isna(value):
            return None
        s = str(value)
        s = re.sub(r"[\s\u00A0\u202F\u2007\u200B]", "", s)
        if not s or s.lower() == "nan":
            return None
        s = s.replace(",", ".")
        s = re.sub(r"[^\d.\-]", "", s)
        if not s or s == "-":
            return None
        try:
            return float(s)
        except Exception:
            return None

    def _format_money(self, num: Optional[float]) -> str:
        """Форматирование числа в денежный вид: 1 234,56 ₸."""
        if num is None:
            return "—"
        is_neg = num < 0
        v = abs(float(num))
        s = f"{v:,.2f}".replace(",", NBSP_NARROW).replace(".", ",")
        return f"-{s}" if is_neg else s

    # ------------------------------------------------------------------
    # Парсинг периода
    # ------------------------------------------------------------------

    def _parse_date_dmy(self, date_str: str) -> date:
        """Парсинг даты в формате DD.MM.YYYY или DD/MM/YYYY."""
        ds = date_str.replace("/", ".")
        parts = ds.split(".")
        if len(parts) != 3:
            raise ValueError(f"Невозможно распарсить дату: {date_str}")
        d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
        return date(y, m, d)

    def _extract_period_str(self, header_rows: List[str]) -> str:
        """
        Извлечение строки периода из первых строк отчёта.
        Ищет подстроку после "Период:".
        """
        header_text = " ".join(header_rows)

        m = re.search(r"Период:\s*(.+?)(?:\s*[|;]|\s*$)", header_text, re.IGNORECASE)
        if m:
            s = m.group(1).strip()
        else:
            # fallback: первая строка, содержащая дату
            s = ""
            for row in header_rows:
                r = row.strip()
                if re.search(r"\d{1,2}[./]\d{1,2}[./]\d{4}", r):
                    s = r
                    break
                if any(mon in r.lower() for mon in MONTHS_RU.keys()):
                    s = r
                    break

        # Отсекаем хвосты после "Показатели:" или ";"
        if s:
            s = re.split(r"\bПоказател[ьи]\b\s*:", s, flags=re.IGNORECASE)[0].strip()
            s = s.split(";")[0].strip()
            s = s.strip(" ,.")
        return s

    def _parse_period(self, header_rows: List[str]) -> PeriodInfo:
        """
        Основная функция определения периода.
        Возвращает структуру PeriodInfo.
        """
        p = self._extract_period_str(header_rows)
        if not p:
            self.logger.warning("Период не найден в заголовке")
            return PeriodInfo("—", None, None, "RANGE")

        p_norm = p.lower().replace("ё", "е")

        # 1) Диапазон DD.MM.YYYY - DD.MM.YYYY
        range_match = re.search(
            r"(\d{1,2}[./]\d{1,2}[./]\d{4})\s*[-–—]\s*(\d{1,2}[./]\d{1,2}[./]\d{4})",
            p_norm,
        )
        if range_match:
            try:
                d1 = self._parse_date_dmy(range_match.group(1))
                d2 = self._parse_date_dmy(range_match.group(2))
                # Определяем тип: если начало месяца и конец в том же месяце -> MTD
                if d1.day == 1 and d1.month == d2.month and d1.year == d2.year:
                    rtype = "MTD"
                else:
                    rtype = "RANGE"
                self.logger.info(f"Период: {d1} - {d2}, тип: {rtype}")
                return PeriodInfo(p, d1, d2, rtype)
            except Exception as e:
                self.logger.warning(f"Ошибка парсинга диапазона: {e}")

        # 2) Одиночная дата DD.MM.YYYY
        single_match = re.search(r"(\d{1,2}[./]\d{1,2}[./]\d{4})", p_norm)
        if single_match:
            try:
                d = self._parse_date_dmy(single_match.group(1))
                self.logger.info(f"Период: {d}, тип: DAY")
                return PeriodInfo(p, d, d, "DAY")
            except Exception as e:
                self.logger.warning(f"Ошибка парсинга одиночной даты: {e}")

        # 3) Текстовая дата "31 января 2026 г."
        text_date_match = re.search(
            r"(\d{1,2})\s+([а-яa-z]+)\s+(\d{4})\s*(?:г\.?)?",
            p_norm, re.IGNORECASE
        )
        if text_date_match:
            try:
                dd = int(text_date_match.group(1))
                mon_name = text_date_match.group(2).lower().replace("ё", "е")
                yy = int(text_date_match.group(3))
                if mon_name in MONTHS_RU:
                    d = date(yy, MONTHS_RU[mon_name], dd)
                    self.logger.info(f"Период: {d}, тип: DAY (текстовый месяц)")
                    return PeriodInfo(p, d, d, "DAY")
            except Exception as e:
                self.logger.warning(f"Ошибка парсинга текстовой даты: {e}")

        # 4) Месяц + год ("январь 2026", "февраля 2026 г.")
        month_match = re.search(
            r"([а-яa-z]+)\s+(\d{4})\s*(?:г\.?)?",
            p_norm, re.IGNORECASE
        )
        if month_match:
            mon_name = month_match.group(1).lower().replace("ё", "е")
            yy = int(month_match.group(2))
            if mon_name in MONTHS_RU:
                mnum = MONTHS_RU[mon_name]
                start = date(yy, mnum, 1)
                # последний день месяца
                if mnum == 12:
                    end = date(yy, 12, 31)
                else:
                    end = date(yy, mnum + 1, 1) - timedelta(days=1)
                self.logger.info(f"Период: {mon_name} {yy}, тип: MONTH")
                return PeriodInfo(p, start, end, "MONTH")

        self.logger.warning(f"Период не распознан: {p}")
        return PeriodInfo("—", None, None, "RANGE")

    # ------------------------------------------------------------------
    # Поиск таблицы в данных
    # ------------------------------------------------------------------

    def _find_table_start(self, df: pd.DataFrame) -> Tuple[int, int, int]:
        """
        Ищет строку, содержащую заголовки "Подразделение" и "Сумма".
        Возвращает кортеж:
          - индекс первой строки данных (сразу после заголовков)
          - индекс колонки с названием подразделения/статьи
          - индекс колонки с суммой
        """
        for i in range(min(80, len(df))):
            row_norm = [self._normalize_text(x).lower() for x in df.iloc[i].tolist()]
            joined = " ".join(row_norm)

            if "подраздел" in joined and "сумм" in joined:
                name_col = None
                amt_col = None
                for j, cell in enumerate(row_norm):
                    if name_col is None and "подраздел" in cell:
                        name_col = j
                    if amt_col is None and "сумм" in cell:
                        amt_col = j

                if name_col is not None and amt_col is not None:
                    start = i + 1
                    # если следующая строка содержит "статья затрат", пропускаем её
                    if i + 1 < len(df):
                        nxt = " ".join(self._normalize_text(x).lower() for x in df.iloc[i + 1].tolist())
                        if "статья затрат" in nxt:
                            start = i + 2
                    return start, name_col, amt_col

        raise ValueError("Не найдена строка заголовка таблицы ('Подразделение' и 'Сумма').")

    # ------------------------------------------------------------------
    # Основной парсинг
    # ------------------------------------------------------------------

    def _is_grand_total_row(self, name: str) -> bool:
        """Проверяет, является ли строка строкой общего итога."""
        low = name.lower().strip()
        if not low:
            return False
        return any(low == kw or low.startswith(kw + " ") for kw in self.grand_total_keywords)

    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Главный метод: читает Excel, извлекает данные, возвращает структуру.
        """
        self.logger.info(f"Парсинг файла: {file_path.name}")
        df = pd.read_excel(file_path, header=None, dtype=str, engine="openpyxl")

        # --- Извлечение периода из шапки ---
        header_rows: List[str] = []
        for i in range(min(25, len(df))):
            parts = [self._normalize_text(x) for x in df.iloc[i].tolist() if not _safe_isna(x)]
            if parts:
                header_rows.append(" ".join(parts))
        period = self._parse_period(header_rows)

        # --- Поиск начала таблицы ---
        start_row, name_col, amt_col = self._find_table_start(df)
        self.logger.info(f"Таблица: start_row={start_row}, name_col={name_col}, amount_col={amt_col}")

        # --- Сбор данных с группировкой по подразделениям ---
        subdivisions: Dict[str, List[Dict[str, Any]]] = {}
        current_sub = None
        grand_total = None

        for idx in range(start_row, len(df)):
            name = self._normalize_text(df.iat[idx, name_col] if name_col < df.shape[1] else "")
            amount = self._normalize_number(df.iat[idx, amt_col] if amt_col < df.shape[1] else None)

            # Полностью пустая строка – разделитель блоков
            if not name and amount is None:
                current_sub = None
                continue

            # Если нет суммы, такую строку игнорируем (это может быть просто текст)
            if amount is None:
                continue

            # Проверка на общий итог
            if self._is_grand_total_row(name):
                grand_total = float(amount)
                current_sub = None
                continue

            # Если это новое подразделение (нет текущего)
            if current_sub is None:
                current_sub = name
                subdivisions.setdefault(current_sub, [])
                # Сохраняем итоговую строку подразделения (первая строка после заголовка блока)
                subdivisions[current_sub].append({
                    "subdivision": current_sub,
                    "category": "ИТОГО ПО ПОДРАЗДЕЛЕНИЮ",
                    "amount": float(amount),
                    "is_subtotal": True
                })
            else:
                # Детальная строка (статья затрат)
                subdivisions[current_sub].append({
                    "subdivision": current_sub,
                    "category": name or "—",
                    "amount": float(amount),
                    "is_subtotal": False
                })

        # --- Формирование итогового списка строк с сортировкой ---
        rows = []
        # Вычислим сумму по каждому подразделению (по детальным строкам)
        sub_totals = {}
        for sub, items in subdivisions.items():
            detail_sum = sum(it["amount"] for it in items if not it["is_subtotal"])
            sub_totals[sub] = detail_sum

        # Сортируем подразделения по убыванию суммы
        sorted_subs = sorted(subdivisions.keys(), key=lambda s: sub_totals[s], reverse=True)

        for sub in sorted_subs:
            items = subdivisions[sub]
            # Отделяем подытог (он должен быть первым)
            subtotal_item = next((it for it in items if it["is_subtotal"]), None)
            detail_items = [it for it in items if not it["is_subtotal"]]
            # Сортируем детальные строки по убыванию суммы
            detail_items.sort(key=lambda x: x["amount"], reverse=True)

            if subtotal_item:
                rows.append(subtotal_item)
            rows.extend(detail_items)

        # --- Определение общей суммы ---
        if grand_total is not None:
            total_expenses = grand_total
        else:
            total_expenses = sum(it["amount"] for it in rows if it.get("is_subtotal"))

        # --- ТОП-5 статей расходов (только детальные строки) ---
        cat_sums = {}
        for it in rows:
            if not it.get("is_subtotal"):
                cat = it["category"]
                cat_sums[cat] = cat_sums.get(cat, 0.0) + it["amount"]

        top5 = [
            {"category": k, "amount": v}
            for k, v in sorted(cat_sums.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        result = {
            "source_file": file_path.name,
            "report_type": period.report_type,
            "period": period.period_str,
            "period_start": period.period_start.isoformat() if period.period_start else None,
            "period_end": period.period_end.isoformat() if period.period_end else None,
            "total_expenses": total_expenses,
            "top5_categories": top5,
            "rows": rows,
            "metadata": {
                "subdivisions_count": len(subdivisions),
                "detail_count": sum(1 for r in rows if not r.get("is_subtotal")),
                "grand_total": grand_total,
                "parsed_at": self._now_tz().strftime("%Y-%m-%d %H:%M:%S"),
                "timezone": self.timezone,
                "version": "1.1.0-final",
            },
        }

        self.logger.info(
            f"Готово: подразделений={len(subdivisions)}, деталей={result['metadata']['detail_count']}, "
            f"total={total_expenses:.2f}" + (f", grand_total={grand_total:.2f}" if grand_total else "")
        )
        return result

    # ------------------------------------------------------------------
    # Сохранение результатов
    # ------------------------------------------------------------------

    def generate_slug(self, file_path: Path, data: Dict[str, Any]) -> str:
        """Генерирует уникальный идентификатор для файлов отчёта."""
        base_name = file_path.stem
        # берём дату начала периода или текущую
        date_part = (data.get("period_start") or datetime.now().strftime("%Y-%m-%d")).replace("-", "")
        clean_name = re.sub(r"[^\w\-]+", "_", base_name).strip("_")
        return f"{clean_name}_{date_part}"

    def save_json(self, data: Dict[str, Any], slug: str) -> Path:
        """Сохраняет данные в JSON."""
        json_path = self.json_dir / f"expenses_{slug}.json"
        json_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        self.logger.info(f"JSON сохранён: {json_path}")
        return json_path

    def save_html(self, data: Dict[str, Any], slug: str) -> Path:
        """Генерирует и сохраняет HTML-отчёт."""
        html_path = self.html_dir / f"expenses_{slug}.html"

        period_start = data.get("period_start") or "—"
        period_end = data.get("period_end") or "—"
        rt = (data.get("report_type") or "range").lower()
        if rt not in {"day", "mtd", "month", "range"}:
            rt = "range"

        # ТОП-5
        top5_rows = ""
        for i, item in enumerate(data.get("top5_categories", []), 1):
            top5_rows += f"""
            <tr>
                <td>{i}</td>
                <td>{item.get('category', '')}</td>
                <td class="amount">{self._format_money(item.get('amount'))} ₸</td>
            </tr>"""

        # Детализация (уже отсортирована)
        details_rows = ""
        for row in data.get("rows", []):
            is_sub = row.get("is_subtotal", False)
            cls = "subtotal" if is_sub else ""
            sub = row.get("subdivision", "")
            cat = row.get("category", "")
            details_rows += f"""
            <tr class="{cls}">
                <td>{sub}</td>
                <td>{cat}</td>
                <td class="amount">{self._format_money(row.get('amount'))} ₸</td>
            </tr>"""

        footer_time = self._now_tz().strftime("%d.%m.%Y %H:%M")

        html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Затраты (расходы) – {slug}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: #f5f5f5;
            margin: 0;
            padding: 20px;
            color: #111;
        }}
        .wrap {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50, #4a6491);
            color: #fff;
            padding: 18px;
            border-radius: 10px;
        }}
        .meta {{
            opacity: 0.9;
            font-size: 13px;
            margin-top: 8px;
        }}
        .kpi {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 12px;
            margin: 16px 0;
        }}
        .card {{
            background: #fff;
            border-radius: 10px;
            padding: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .kpi-val {{
            font-size: 22px;
            font-weight: 700;
            margin-top: 6px;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .type-day {{ background: #d4edda; color: #155724; }}
        .type-mtd {{ background: #d1ecf1; color: #0c5460; }}
        .type-month {{ background: #fff3cd; color: #856404; }}
        .type-range {{ background: #f8d7da; color: #721c24; }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px 10px;
            border-bottom: 1px solid #eee;
            font-size: 13px;
        }}
        th {{
            background: #fafafa;
            text-align: left;
        }}
        .amount {{
            text-align: right;
            font-family: 'Consolas', monospace;
        }}
        .subtotal td {{
            font-weight: 700;
            background: #e8f4fd;
        }}
        .footer {{
            text-align: center;
            color: #666;
            font-size: 12px;
            margin: 18px 0;
        }}
    </style>
</head>
<body>
<div class="wrap">
    <div class="header">
        <div style="font-size:20px; font-weight:800;">📊 Затраты (расходы)</div>
        <div class="meta">
            <div><b>Источник:</b> {data.get('source_file','')}</div>
            <div><b>Период:</b> {data.get('period','—')}</div>
            <div><b>Тип отчёта:</b> <span class="badge type-{rt}">{data.get('report_type','')}</span></div>
        </div>
    </div>

    <div class="kpi">
        <div class="card">
            <div style="color:#666; font-size:12px; text-transform:uppercase;">Итого расходов</div>
            <div class="kpi-val">{self._format_money(data.get('total_expenses'))} ₸</div>
        </div>
        <div class="card">
            <div style="color:#666; font-size:12px; text-transform:uppercase;">Начало периода</div>
            <div class="kpi-val">{period_start}</div>
        </div>
        <div class="card">
            <div style="color:#666; font-size:12px; text-transform:uppercase;">Конец периода</div>
            <div class="kpi-val">{period_end}</div>
        </div>
    </div>

    <div class="card">
        <div style="font-weight:800; margin-bottom:10px;">🏆 ТОП-5 статей расходов</div>
        <table>
            <thead>
                <tr>
                    <th style="width:40px">#</th>
                    <th>Статья расходов</th>
                    <th style="text-align:right; width:180px">Сумма</th>
                </tr>
            </thead>
            <tbody>
                {top5_rows}
            </tbody>
        </table>
    </div>

    <div class="card">
        <div style="font-weight:800; margin-bottom:10px;">📋 Детализация расходов (по убыванию суммы)</div>
        <table>
            <thead>
                <tr>
                    <th style="width:260px">Подразделение</th>
                    <th>Статья затрат</th>
                    <th style="text-align:right; width:180px">Сумма</th>
                </tr>
            </thead>
            <tbody>
                {details_rows}
            </tbody>
        </table>
    </div>

    <div class="footer">
        Сформировано: {footer_time} ({self.timezone}) | expenses_parser_final v1.1.0
    </div>
</div>
</body>
</html>"""
        html_path.write_text(html_content, encoding="utf-8")
        self.logger.info(f"HTML сохранён: {html_path}")
        return html_path


# ------------------------------------------------------------------
# Внешний интерфейс
# ------------------------------------------------------------------

def parse_expenses_file(xlsx_path: Union[str, Path], out_root: Optional[Path] = None) -> Path:
    """
    Удобная функция для вызова из других скриптов.
    Возвращает путь к сгенерированному HTML-файлу.
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Файл не найден: {xlsx_path}")

    parser = UnifiedExpensesParser(out_root=out_root)
    data = parser.parse_file(xlsx_path)
    slug = parser.generate_slug(xlsx_path, data)
    parser.save_json(data, slug)
    return parser.save_html(data, slug)


def main() -> None:
    """Точка входа для командной строки."""
    ap = argparse.ArgumentParser(
        description="Универсальный парсер отчетов 1С «Затраты» (расходы) с сортировкой по убыванию"
    )
    ap.add_argument("files", nargs="+", help="Один или несколько Excel-файлов для обработки")
    ap.add_argument("--out-root", default=".", help="Корневая директория для reports/ и logs/")
    args = ap.parse_args()

    out_root = Path(args.out_root)

    for fp in args.files:
        try:
            html_path = parse_expenses_file(fp, out_root=out_root)
            print(f"[OK] {fp} -> {html_path}")
        except Exception as e:
            print(f"[ERROR] {fp}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
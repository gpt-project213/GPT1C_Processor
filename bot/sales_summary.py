"""
Модуль для генерации кратких сводок по продажам

Версия: 1.3.1 (2026-03-10)
─────────────────────────────────────────────────
v1.3.1: bare except: → except (ValueError, TypeError): в parse_sales_html (Bug S6)
v1.3: load_all_managers_json сравнивает периоды по ДАТАМ, не по строкам
  - _normalize_period_dates() — парсит любой формат периода в (start, end)
  - Нечувствительно к пробелам, разделителям ("11.02-20.02" = "11.02 - 20.02")
  - Fallback: точное сравнение строк если парсинг не удался
v1.2: Pipeline-сводки после обработки файла:
  - detect_period_type()     — день / декада / месяц по диапазону дат
  - period_label()           — человекочитаемый заголовок периода
  - load_all_managers_json() — все sales JSON одного периода → рейтинг
  - format_admin_pipeline()  — admin: итог + рейтинг всех менеджеров
  - format_manager_pipeline()— менеджер: только свои данные + топ-3
v1.1: get_latest_sales_report сортирует по периоду данных, не по mtime
"""

import json
import logging
import re
from datetime import date as date_type
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup

# ── Типы периода ─────────────────────────────────────────────────────────────
PERIOD_DAY    = "day"     # один день
PERIOD_DECADE = "decade"  # декада (1–10, 11–20, 21–конец)
PERIOD_MONTH  = "month"   # полный месяц

# ── Утилита: парсинг строки периода → сравнимая дата ────────────────────────
_MONTHS_RU = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}

def _period_to_date(period_str: str) -> date_type:
    """Парсит строку периода в date для сортировки. Диапазон → конечная дата."""
    if not period_str:
        return date_type.min
    s = period_str.strip()
    m = re.search(r'(\d{1,2})[./](\d{1,2})[./](\d{4})\s*[-–—]\s*(\d{1,2})[./](\d{1,2})[./](\d{4})', s)
    if m:
        try:
            return date_type(int(m.group(6)), int(m.group(5)), int(m.group(4)))
        except Exception:
            pass
    m = re.match(r'(\d{1,2})[./](\d{1,2})[./](\d{4})', s)
    if m:
        try:
            return date_type(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except Exception:
            pass
    m = re.match(r'(\d{1,2})\s+([а-яё]+)\s+(\d{4})', s.lower())
    if m:
        mon = _MONTHS_RU.get(m.group(2))
        if mon:
            try:
                return date_type(int(m.group(3)), mon, int(m.group(1)))
            except Exception:
                pass
    return date_type.min


# ── v1.2: определение типа периода ───────────────────────────────────────────

def _period_range(period_str: str) -> Tuple[Optional[date_type], Optional[date_type]]:
    """Извлекает (start, end) из строки периода."""
    s = (period_str or "").strip()
    # Диапазон: 01.02.2026 - 24.02.2026
    m = re.search(
        r'(\d{1,2})[./](\d{1,2})[./](\d{4})\s*[-–—]\s*(\d{1,2})[./](\d{1,2})[./](\d{4})', s
    )
    if m:
        try:
            return (date_type(int(m.group(3)), int(m.group(2)), int(m.group(1))),
                    date_type(int(m.group(6)), int(m.group(5)), int(m.group(4))))
        except Exception:
            pass
    # Один день: 24.02.2026
    m2 = re.match(r'(\d{1,2})[./](\d{1,2})[./](\d{4})$', s)
    if m2:
        try:
            d = date_type(int(m2.group(3)), int(m2.group(2)), int(m2.group(1)))
            return d, d
        except Exception:
            pass
    # Один день русский: 24 февраля 2026 г.
    m3 = re.match(r'(\d{1,2})\s+([а-яё]+)\s+(\d{4})', s.lower())
    if m3:
        mon = _MONTHS_RU.get(m3.group(2))
        if mon:
            try:
                d = date_type(int(m3.group(3)), mon, int(m3.group(1)))
                return d, d
            except Exception:
                pass
    return None, None


def detect_period_type(period_str: str) -> str:
    """
    Определяет тип периода:
      PERIOD_DAY    — один день
      PERIOD_DECADE — декада (2–27 дней в рамках одной декады месяца)
      PERIOD_MONTH  — полный месяц (≥28 дней, начинается с 1-го числа)
    """
    start, end = _period_range(period_str)
    if start is None or end is None:
        return PERIOD_DAY
    delta = (end - start).days
    if delta == 0:
        return PERIOD_DAY
    if delta >= 27 and start.day == 1:
        return PERIOD_MONTH
    return PERIOD_DECADE


def period_label(period_str: str) -> str:
    """Человекочитаемый заголовок для уведомлений: 📅 День / 📊 1-я декада / 📆 Месяц."""
    _MONTH_NAMES = {
        1:"январь", 2:"февраль", 3:"март", 4:"апрель",
        5:"май", 6:"июнь", 7:"июль", 8:"август",
        9:"сентябрь", 10:"октябрь", 11:"ноябрь", 12:"декабрь",
    }
    ptype = detect_period_type(period_str)
    start, _ = _period_range(period_str)
    s = period_str.strip()
    if ptype == PERIOD_DAY:
        return f"📅 День: {s}"
    if ptype == PERIOD_MONTH:
        if start:
            return f"📆 Месяц: {_MONTH_NAMES.get(start.month, '')} {start.year}"
        return f"📆 Месяц: {s}"
    # Декада
    if start:
        n = 1 if start.day <= 10 else (2 if start.day <= 20 else 3)
        return f"📊 {n}-я декада {_MONTH_NAMES.get(start.month, '')} {start.year}"
    return f"📊 Декада: {s}"


logger = logging.getLogger(__name__)


class SalesSummary:
    """Краткие сводки по продажам"""
    
    @staticmethod
    def parse_amount(amount_str: str) -> float:
        """Парсит сумму: '1 000 000' -> 1000000.0"""
        if not amount_str:
            return 0.0
        
        cleaned = ''.join(amount_str.split()).replace(',', '.').replace('₸', '').strip()
        try:
            return float(cleaned)
        except ValueError:
            logger.warning(f"Не удалось распарсить сумму: '{amount_str}'")
            return 0.0
    
    def parse_sales_html(self, html_path: Path) -> Dict:
        """
        Парсит HTML продаж
        
        Возвращает:
        {
            'date': '8 января 2026 г.',
            'total_amount': 6173090.0,
            'clients_count': 37,
            'clients': [...],
            'products': [...]
        }
        """
        try:
            html_content = html_path.read_text(encoding='utf-8')
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Извлекаем метаданные
            small_tag = soup.find('small')
            
            date_str = ""
            clients_count = 0
            total_amount = 0.0
            
            if small_tag:
                text = small_tag.get_text()
                
                # Период: 8 января 2026 г.
                if 'Период:' in text or 'период:' in text.lower():
                    parts = text.split('Период:') if 'Период:' in text else text.split('период:')
                    if len(parts) > 1:
                        date_str = parts[1].split('\n')[0].strip()
                
                # Клиентов: 37
                if 'Клиентов:' in text or 'клиентов:' in text.lower():
                    parts = text.split('Клиентов:') if 'Клиентов:' in text else text.split('клиентов:')
                    if len(parts) > 1:
                        clients_str = parts[1].split('\n')[0].strip()
                        try:
                            clients_count = int(clients_str)
                        except (ValueError, TypeError):
                            pass
                
                # Итого (расч.): 6 173 090
                if 'Итого' in text or 'итого' in text.lower():
                    parts = text.lower().split('итого')
                    if len(parts) > 1:
                        total_str = parts[1].split(':')[-1].split('\n')[0].strip()
                        total_amount = self.parse_amount(total_str)
            
            # Парсим клиентов и товары
            clients = []
            products_dict = {}  # Аггрегация товаров
            
            tables = soup.find_all('table')
            
            for table in tables:
                # Найти заголовок клиента
                client_head = table.find('tr', class_='client-head')
                
                if not client_head:
                    continue
                
                client_text = client_head.get_text(strip=True)
                
                # Парсим имя клиента и итог
                # "В ИП Глушков — Итого: 1 000 000 тг; Кол-во: 800 шт/кг"
                client_name = ""
                client_amount = 0.0
                client_qty = 0.0
                
                if '—' in client_text:
                    client_name = client_text.split('—')[0].strip()
                    
                    if 'Итого:' in client_text:
                        amount_part = client_text.split('Итого:')[1].split('тг')[0].strip()
                        client_amount = self.parse_amount(amount_part)
                    
                    if 'Кол-во:' in client_text:
                        qty_part = client_text.split('Кол-во:')[1].split('шт')[0].strip()
                        client_qty = self.parse_amount(qty_part)
                
                clients.append({
                    'name': client_name,
                    'amount': client_amount,
                    'quantity': client_qty
                })
                
                # Парсим товары в этой таблице
                rows = table.find_all('tr')
                
                for row in rows:
                    # Пропускаем заголовки
                    if row.find('th') or row == client_head:
                        continue
                    
                    cells = row.find_all('td')
                    
                    if len(cells) == 4:
                        product = cells[0].get_text(strip=True)
                        qty_str = cells[1].get_text(strip=True)
                        price_str = cells[2].get_text(strip=True)
                        sum_str = cells[3].get_text(strip=True)
                        
                        qty = self.parse_amount(qty_str)
                        amount = self.parse_amount(sum_str)
                        
                        # Аггрегируем товары
                        if product in products_dict:
                            products_dict[product]['total_amount'] += amount
                            products_dict[product]['total_quantity'] += qty
                        else:
                            products_dict[product] = {
                                'product': product,
                                'total_amount': amount,
                                'total_quantity': qty
                            }
            
            products = list(products_dict.values())
            
            logger.info(f"📊 Распарсено {len(clients)} клиентов и {len(products)} товаров из {html_path.name}")
            
            return {
                'date': date_str,
                'total_amount': total_amount,
                'clients_count': clients_count,
                'clients': clients,
                'products': products
            }
            
        except Exception as e:
            logger.error(f"Ошибка при парсинге {html_path}: {e}", exc_info=True)
            return {
                'date': '',
                'total_amount': 0.0,
                'clients_count': 0,
                'clients': [],
                'products': []
            }
    
    def format_summary(self, data: Dict) -> str:
        """
        Форматирует краткую сводку
        
        🛒 ПРОДАЖИ за 8 января 2026 г.
        
        Итого: 6 173 090 ₸
        Клиентов: 37
        Средний чек: 166 840 ₸
        
        🏆 ТОП-3 клиента:
          1. В ИП Глушков — 1 000 000 ₸
          2. Е ИП Алина — 643 457 ₸
        
        🔥 ТОП-3 товара:
          1. Утиные Окорочка — 1 000 000 ₸
        """
        msg_lines = [
            f"🛒 ПРОДАЖИ за {data['date']}",
            ""
        ]
        
        # Метрики
        total = data['total_amount']
        clients = data['clients_count']
        avg_check = total / clients if clients > 0 else 0.0
        
        msg_lines.append(f"Итого: {self.format_amount(total)} ₸")
        msg_lines.append(f"Клиентов: {clients}")
        msg_lines.append(f"Средний чек: {self.format_amount(avg_check)} ₸")
        msg_lines.append("")
        
        # ТОП-3 клиента
        top_clients = sorted(data['clients'], key=lambda x: x['amount'], reverse=True)[:3]
        
        if top_clients:
            msg_lines.append("🏆 ТОП-3 клиента:")
            for i, client in enumerate(top_clients, 1):
                client_short = client['name'][:30]
                msg_lines.append(f"  {i}. {client_short} — {self.format_amount(client['amount'])} ₸")
            msg_lines.append("")
        
        # ТОП-3 товара
        top_products = sorted(data['products'], key=lambda x: x['total_amount'], reverse=True)[:3]
        
        if top_products:
            msg_lines.append("🔥 ТОП-3 товара:")
            for i, product in enumerate(top_products, 1):
                product_short = product['product'][:30]
                msg_lines.append(f"  {i}. {product_short} — {self.format_amount(product['total_amount'])} ₸")
        
        return "\n".join(msg_lines)
    
    @staticmethod
    def format_amount(amount: float) -> str:
        """Форматирует сумму: 1000000.0 -> '1 000 000'"""
        return f"{amount:,.0f}".replace(',', ' ')
    
    def get_latest_sales_report(self, reports_dir: Path, manager: Optional[str] = None) -> Optional[Path]:
        """
        v1.1: Находит отчёт продаж с наиболее свежим ПЕРИОДОМ ДАННЫХ.
        Сортирует по дате из HTML (не по mtime файла).
        """
        if manager:
            pattern = f"sales_grouped*{manager}*.html"
        else:
            pattern = "sales_grouped_*.html"

        matching_files = list(reports_dir.glob(pattern))

        if not matching_files:
            logger.warning(f"Не найдены отчёты продаж для {manager or 'всех'}")
            return None

        def _get_period(p: Path) -> date_type:
            try:
                html = p.read_text(encoding='utf-8', errors='ignore')
                soup = BeautifulSoup(html, 'html.parser')
                small = soup.find('small')
                if small:
                    text = small.get_text()
                    if 'Период:' in text:
                        period_str = text.split('Период:')[1].split('\n')[0].strip()
                        return _period_to_date(period_str)
            except Exception:
                pass
            return date_type.min

        latest = max(matching_files, key=lambda p: (_get_period(p), p.stat().st_mtime))
        pd = _get_period(latest)
        logger.info(f"📄 Найден отчёт продаж: {latest.name} (период={pd})")
        return latest


    # ── v1.2: Pipeline-методы ──────────────────────────────────────────────────

    @staticmethod
    def _normalize_period_dates(period_str: str) -> Tuple[Optional[date_type], Optional[date_type]]:
        """
        v1.3: Нормализует строку периода в (start_date, end_date) для сравнения.
        Снимает зависимость от точного формата строки.
        """
        if not period_str:
            return None, None
        s = period_str.strip()
        # Диапазон DD.MM.YYYY - DD.MM.YYYY (любые разделители)
        m = re.search(
            r'(\d{1,2})[./](\d{1,2})[./](\d{4})\s*[-–—]\s*(\d{1,2})[./](\d{1,2})[./](\d{4})', s
        )
        if m:
            try:
                start = date_type(int(m.group(3)), int(m.group(2)), int(m.group(1)))
                end   = date_type(int(m.group(6)), int(m.group(5)), int(m.group(4)))
                return start, end
            except Exception:
                pass
        # Одна дата DD.MM.YYYY
        m2 = re.search(r'(\d{1,2})[./](\d{1,2})[./](\d{4})', s)
        if m2:
            try:
                d = date_type(int(m2.group(3)), int(m2.group(2)), int(m2.group(1)))
                return d, d
            except Exception:
                pass
        # Русские месяцы: DD месяц YYYY [г.]
        m3 = re.search(r'(\d{1,2})\s+([а-яё]+)\s+(\d{4})', s.lower())
        if m3:
            _MONTHS_RU = {"января":1,"февраля":2,"марта":3,"апреля":4,"мая":5,"июня":6,"июля":7,"августа":8,"сентября":9,"октября":10,"ноября":11,"декабря":12}
            mon = _MONTHS_RU.get(m3.group(2))
            if mon:
                try:
                    d = date_type(int(m3.group(3)), mon, int(m3.group(1)))
                    return d, d
                except Exception:
                    pass
        return None, None

    def load_all_managers_json(self, json_dir: Path, reference_period: str) -> List[Dict]:
        """
        v1.3: Загружает ВСЕ sales_*.json с тем же периодом что reference_period.
        Сравнение по ДАТАМ (не по строке) — нечувствительно к формату.
        Возвращает список [{manager, total_revenue, clients}] — отсортированный по выручке.
        """
        results = []
        ref_start, ref_end = self._normalize_period_dates(reference_period)
        if ref_start is None:
            logger.warning(f"load_all_managers_json: не удалось распарсить период '{reference_period}'")
            # Fallback: точное сравнение строк
            ref_start = ref_end = None

        try:
            files = sorted(json_dir.glob("sales_*.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            return results

        for path in files:
            if "товару" in path.name.lower():
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                file_period = data.get("period", "").strip()
                # v1.3: сравниваем по датам
                if ref_start is not None:
                    f_start, f_end = self._normalize_period_dates(file_period)
                    if (f_start, f_end) != (ref_start, ref_end):
                        continue
                else:
                    # fallback: точная строка
                    if file_period != reference_period.strip():
                        continue
                manager = data.get("manager", "") or path.stem
                results.append({
                    "manager":       manager,
                    "total_revenue": float(data.get("total_revenue", 0)),
                    "clients":       data.get("clients", []),
                })
            except Exception as e:
                logger.warning(f"Ошибка чтения {path.name}: {e}")

        results.sort(key=lambda x: x["total_revenue"], reverse=True)
        return results

    def format_admin_pipeline(self, data: Dict, all_managers: List[Dict]) -> str:
        """
        Сводка для ADMIN после pipeline.
        Рейтинг всех менеджеров за период + итог.

        🛒 ПРОДАЖИ · 📅 День: 24.02.2026
        ────────────────────────────────
        🥇 Оксана       12 500 000 ₸  (38 кл.)
        🥈 Магира        9 800 000 ₸  (29 кл.)
        🥉 Ергали        7 200 000 ₸  (21 кл.)
           Алена         5 100 000 ₸  (17 кл.)
        ────────────────────────────────
        ИТОГО           34 600 000 ₸ · 105 кл.
        Средний чек: 329 524 ₸
        """
        period_str = data.get("date", "")
        ptype      = detect_period_type(period_str)
        plabel     = period_label(period_str)
        SEP        = "─" * 32

        lines = [f"🛒 ПРОДАЖИ · {plabel}", SEP]

        total_revenue = 0.0
        total_clients = 0
        medals = ["🥇", "🥈", "🥉"]

        for i, mgr in enumerate(all_managers):
            rev   = mgr["total_revenue"]
            cnt   = len(mgr["clients"])
            name  = mgr["manager"]
            medal = medals[i] if i < 3 else "  "
            lines.append(f"{medal} {name:<10} {self.format_amount(rev):>13} ₸  ({cnt} кл.)")
            total_revenue += rev
            total_clients += cnt

        lines.append(SEP)
        lines.append(f"{'ИТОГО':<13} {self.format_amount(total_revenue):>13} ₸ · {total_clients} кл.")

        if ptype == PERIOD_DAY and total_clients > 0:
            avg = total_revenue / total_clients
            lines.append(f"Средний чек: {self.format_amount(avg)} ₸")

        return "\n".join(lines)

    def format_manager_pipeline(self, manager_name: str, data: Dict,
                                all_managers: Optional[List[Dict]] = None) -> str:
        """
        v1.4: Сводка для МЕНЕДЖЕРА — свои данные + рейтинг среди коллег (конкуренция).

        📋 Ваши продажи · 📅 День: 27.02.2026
        Итого: 2 150 000 ₸
        Клиентов: 12 · Средний чек: 179 167 ₸

        🏆 Топ-3 клиента:
          1. ТОО Омега                   — 500 000 ₸
          ...

        📊 Рейтинг дня:
        🥇 Магира     2 800 000 ₸  (18 кл.)
        🥈 Вы         2 150 000 ₸  (12 кл.) ←
        🥉 Ергали     1 450 000 ₸   (7 кл.)
           Оксана       501 511 ₸   (2 кл.)
        """
        period_str = data.get("date", "")
        plabel     = period_label(period_str)
        rev        = data.get("total_amount", 0.0)
        clients    = data.get("clients", [])
        cnt        = data.get("clients_count", 0) or len(clients)
        avg        = rev / cnt if cnt > 0 else 0.0

        lines = [
            f"📋 Ваши продажи · {plabel}",
            f"Итого: {self.format_amount(rev)} ₸",
            f"Клиентов: {cnt} · Средний чек: {self.format_amount(avg)} ₸",
        ]

        top3 = sorted(clients, key=lambda x: x.get("amount", 0), reverse=True)[:3]
        if top3:
            lines.append("")
            lines.append("🏆 Топ-3 клиента:")
            for i, c in enumerate(top3, 1):
                name   = (c.get("name", "?"))[:28]
                amount = c.get("amount", 0)
                lines.append(f"  {i}. {name:<28} — {self.format_amount(amount)} ₸")

        # v1.4: Рейтинг среди всех менеджеров (розжиг конкуренции)
        if all_managers and len(all_managers) > 1:
            lines.append("")
            lines.append("📊 Рейтинг:")
            medals = ["🥇", "🥈", "🥉"]
            SEP_R  = "─" * 32
            lines.append(SEP_R)
            for i, mgr in enumerate(all_managers):
                m_name  = mgr["manager"]
                m_rev   = mgr["total_revenue"]
                m_cnt   = len(mgr.get("clients", []))
                medal   = medals[i] if i < 3 else "  "
                is_me   = m_name.lower() == manager_name.lower()
                you_tag = " ←" if is_me else ""
                display = "Вы" if is_me else m_name
                lines.append(
                    f"{medal} {display:<10} {self.format_amount(m_rev):>13} ₸  ({m_cnt} кл.){you_tag}"
                )
            lines.append(SEP_R)

        return "\n".join(lines)

    def format_subadmin_pipeline(self, subadmin_name: str,
                                 scope_data: List[Dict], period_str: str) -> str:
        """
        v1.4: Мини-рейтинг для субадмина (Алена) по её команде:
        Алена + Магира + Оксана.

        👥 Твоя команда · 📅 День: 27.02.2026
        ────────────────────────────────
        🥇 Магира     2 800 000 ₸  (18 кл.)
        🥈 Ты         2 150 000 ₸  (12 кл.) ←
        🥉 Оксана       501 511 ₸   (2 кл.)
        ────────────────────────────────
        ИТОГО         5 451 511 ₸ · 32 кл.
        """
        ptype  = detect_period_type(period_str)
        plabel = period_label(period_str)
        SEP    = "─" * 32
        medals = ["🥇", "🥈", "🥉"]

        lines = [f"👥 Твоя команда · {plabel}", SEP]

        total_rev = 0.0
        total_cnt = 0
        for i, mgr in enumerate(scope_data):
            m_name  = mgr["manager"]
            m_rev   = mgr["total_revenue"]
            m_cnt   = len(mgr.get("clients", []))
            medal   = medals[i] if i < 3 else "  "
            is_me   = m_name.lower() == subadmin_name.lower()
            you_tag = " ←" if is_me else ""
            display = "Ты" if is_me else m_name
            lines.append(
                f"{medal} {display:<10} {self.format_amount(m_rev):>13} ₸  ({m_cnt} кл.){you_tag}"
            )
            total_rev += m_rev
            total_cnt += m_cnt

        lines.append(SEP)
        lines.append(f"{'ИТОГО':<13} {self.format_amount(total_rev):>13} ₸ · {total_cnt} кл.")
        if ptype == PERIOD_DAY and total_cnt > 0:
            avg = total_rev / total_cnt
            lines.append(f"Средний чек: {self.format_amount(avg)} ₸")

        return "\n".join(lines)


# Тестирование
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    summary = SalesSummary()
    test_html = Path("/mnt/user-data/uploads/sales_grouped_Продажи__77_.html")
    
    if test_html.exists():
        data = summary.parse_sales_html(test_html)
        message = summary.format_summary(data)
        
        print("\n" + "="*60)
        print(message)
        print("="*60)
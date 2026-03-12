"""
Модуль для генерации кратких сводок по валовой прибыли

Версия: 1.1
Дата: 22.02.2026
Изменения v1.1: get_latest_gross_report сортирует по периоду данных, не по mtime
"""

import logging
import re
from datetime import date as date_type
from pathlib import Path
from typing import Dict, List, Optional
from bs4 import BeautifulSoup

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
        except (ValueError, TypeError):
            pass
    m = re.match(r'(\d{1,2})[./](\d{1,2})[./](\d{4})', s)
    if m:
        try:
            return date_type(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except (ValueError, TypeError):
            pass
    m = re.match(r'(\d{1,2})\s+([а-яё]+)\s+(\d{4})', s.lower())
    if m:
        mon = _MONTHS_RU.get(m.group(2))
        if mon:
            try:
                return date_type(int(m.group(3)), mon, int(m.group(1)))
            except (ValueError, TypeError):
                pass
    return date_type.min

logger = logging.getLogger(__name__)


class GrossSummary:
    """Краткие сводки по валовой прибыли"""
    
    # Порог низкой маржи
    LOW_MARGIN_THRESHOLD = 8.0  # %
    
    @staticmethod
    def parse_amount(amount_str: str) -> float:
        """Парсит сумму: '6 173 090,49' -> 6173090.49"""
        if not amount_str:
            return 0.0
        
        cleaned = ''.join(amount_str.split()).replace(',', '.').replace('₸', '').strip()
        try:
            return float(cleaned)
        except ValueError:
            logger.warning(f"Не удалось распарсить сумму: '{amount_str}'")
            return 0.0
    
    @staticmethod
    def parse_percent(pct_str: str) -> float:
        """Парсит процент: '11,76 %' -> 11.76"""
        if not pct_str:
            return 0.0
        
        cleaned = pct_str.replace('%', '').replace(',', '.').strip()
        try:
            return float(cleaned)
        except ValueError:
            logger.warning(f"Не удалось распарсить процент: '{pct_str}'")
            return 0.0
    
    def parse_gross_html(self, html_path: Path) -> Dict:
        """
        Парсит HTML валовой прибыли
        
        Возвращает:
        {
            'date': '8 января 2026 г.',
            'revenue': 6173090.49,
            'gross_profit': 725778.88,
            'margin': 11.76,
            'top_profitable': [...],
            'low_margin': [...]
        }
        """
        try:
            html_content = html_path.read_text(encoding='utf-8')
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Извлекаем дату
            small_tag = soup.find('small')
            date_str = ""
            
            if small_tag:
                text = small_tag.get_text()
                if 'Период:' in text or 'период:' in text.lower():
                    parts = text.split('Период:') if 'Период:' in text else text.split('период:')
                    if len(parts) > 1:
                        date_str = parts[1].split('\n')[0].strip()
            
            # Извлекаем метрики из блока summary
            revenue = 0.0
            gross_profit = 0.0
            margin = 0.0
            
            summary_div = soup.find('div', class_='summary')
            
            if summary_div:
                kv_divs = summary_div.find_all('div', class_='kv')
                
                for kv in kv_divs:
                    # Ищем внутренние div с классами k и v
                    inner_divs = kv.find_all('div', recursive=False)
                    
                    for inner_div in inner_divs:
                        k_span = inner_div.find('span', class_='k')
                        v_span = inner_div.find('span', class_='v')
                        
                        if not k_span or not v_span:
                            continue
                        
                        key = k_span.get_text(strip=True)
                        value = v_span.get_text(strip=True)
                        
                        if 'Выручка' in key or 'выручка' in key.lower():
                            revenue = self.parse_amount(value)
                        elif 'Валовая' in key or 'валовая' in key.lower():
                            gross_profit = self.parse_amount(value)
                        elif 'Рентабельность' in key or 'рентабельность' in key.lower():
                            margin = self.parse_percent(value)
            
            # Парсим таблицы
            top_profitable = []
            low_margin = []
            
            sections = soup.find_all('div', class_='section')
            
            for section in sections:
                section_text = section.get_text(strip=True)
                
                # Найти следующую таблицу после этого заголовка
                table_wrap = section.find_next_sibling('div', class_='table-wrap')
                
                if not table_wrap:
                    continue
                
                table_tag = table_wrap.find('table')
                
                if not table_tag:
                    continue
                
                tbody = table_tag.find('tbody')
                
                if not tbody:
                    continue
                
                rows = tbody.find_all('tr')
                
                for row in rows:
                    cells = row.find_all('td')
                    
                    if len(cells) < 6:
                        continue
                    
                    # Пропускаем первую колонку (#)
                    product = cells[1].get_text(strip=True)
                    profit_str = cells[4].get_text(strip=True)
                    margin_str = cells[5].get_text(strip=True)
                    
                    profit = self.parse_amount(profit_str)
                    margin_pct = self.parse_percent(margin_str)
                    
                    item = {
                        'product': product,
                        'profit': profit,
                        'margin': margin_pct
                    }
                    
                    # Определяем в какой список добавить
                    if 'ТОП' in section_text and 'прибыл' in section_text.lower():
                        top_profitable.append(item)
                    elif 'Низкая' in section_text or 'низк' in section_text.lower():
                        low_margin.append(item)
            
            logger.info(f"📊 Распарсено: выручка={revenue:.0f}, прибыль={gross_profit:.0f}, маржа={margin:.2f}%")
            logger.info(f"    Топ прибыльных: {len(top_profitable)}, низкая маржа: {len(low_margin)}")
            
            return {
                'date': date_str,
                'revenue': revenue,
                'gross_profit': gross_profit,
                'margin': margin,
                'top_profitable': top_profitable,
                'low_margin': low_margin
            }
            
        except Exception as e:
            logger.error(f"Ошибка при парсинге {html_path}: {e}", exc_info=True)
            return {
                'date': '',
                'revenue': 0.0,
                'gross_profit': 0.0,
                'margin': 0.0,
                'top_profitable': [],
                'low_margin': []
            }
    
    def format_summary(self, data: Dict) -> str:
        """
        Форматирует краткую сводку
        
        💰 ВАЛОВАЯ ПРИБЫЛЬ за 8 января 2026 г.
        
        Выручка: 6 173 090 ₸
        Прибыль: 725 779 ₸
        Маржа: 11.76% ⚠️
        
        🏆 ТОП-3 прибыльных:
          1. УКПФ Четвертина — 177 781 ₸ (13.2%)
        
        ⚠️ Низкая маржа (<8%):
          1. Говяжий язык — 3.70%
        """
        msg_lines = [
            f"💰 ВАЛОВАЯ ПРИБЫЛЬ за {data['date']}",
            ""
        ]
        
        # Метрики
        revenue = data['revenue']
        profit = data['gross_profit']
        margin = data['margin']
        
        # Индикатор маржи
        margin_indicator = "✅" if margin >= 12.0 else ("⚠️" if margin >= 10.0 else "🔴")
        
        msg_lines.append(f"Выручка: {self.format_amount(revenue)} ₸")
        msg_lines.append(f"Прибыль: {self.format_amount(profit)} ₸")
        msg_lines.append(f"Маржа: {margin:.2f}% {margin_indicator}")
        msg_lines.append("")
        
        # ТОП-3 прибыльных
        top3 = data['top_profitable'][:3]
        
        if top3:
            msg_lines.append("🏆 ТОП-3 прибыльных:")
            for i, item in enumerate(top3, 1):
                product_short = item['product'][:30]
                msg_lines.append(
                    f"  {i}. {product_short} — {self.format_amount(item['profit'])} ₸ ({item['margin']:.1f}%)"
                )
            msg_lines.append("")
        
        # Низкая маржа
        low = [item for item in data['low_margin'] if item['margin'] < self.LOW_MARGIN_THRESHOLD]
        low.sort(key=lambda x: x['margin'])  # От меньшего к большему
        
        if low:
            msg_lines.append(f"⚠️ Низкая маржа (<{self.LOW_MARGIN_THRESHOLD}%):")
            for i, item in enumerate(low[:5], 1):  # Топ-5
                product_short = item['product'][:30]
                msg_lines.append(f"  {i}. {product_short} — {item['margin']:.2f}%")
        
        return "\n".join(msg_lines)
    
    @staticmethod
    def format_amount(amount: float) -> str:
        """Форматирует сумму: 1000000.0 -> '1 000 000'"""
        return f"{amount:,.0f}".replace(',', ' ')
    
    def get_latest_gross_report(self, reports_dir: Path) -> Optional[Path]:
        """
        v1.1: Находит отчёт валовой прибыли с наиболее свежим ПЕРИОДОМ ДАННЫХ.
        Сортирует по дате из HTML (не по mtime файла).
        """
        pattern = "*_gross_sum.html"
        matching_files = list(reports_dir.glob(pattern))

        if not matching_files:
            logger.warning("Не найдены отчёты валовой прибыли")
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
        logger.info(f"📄 Найден отчёт валовой: {latest.name} (период={pd})")
        return latest


# Тестирование
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    summary = GrossSummary()
    test_html = Path("/mnt/user-data/uploads/Валовая_прибыль__79__gross_sum.html")
    
    if test_html.exists():
        data = summary.parse_gross_html(test_html)
        message = summary.format_summary(data)
        
        print("\n" + "="*60)
        print(message)
        print("="*60)
"""
Модуль для генерации кратких сводок по остаткам

Версия: 1.2
Дата: 09.03.2026
Изменения v1.2:
  - Fix #INV-1: исправлен glob-паттерн inventory_simple_*.html → inventory_*.html
    (inventory.py генерирует файлы как inventory_{slug}.html без _simple_ в имени)
Изменения v1.1:
  - get_latest_inventory_report() сортирует по периоду данных из HTML (не по mtime)
  - Добавлен _parse_period_date_from_html() — лёгкий парсер даты
"""

import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Русские месяцы для парсинга дат
_MONTHS_RU = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}


class InventorySummary:
    """Краткие сводки по остаткам"""
    
    LOW_STOCK_THRESHOLD = 50.0
    
    @staticmethod
    def parse_quantity(qty_str: str) -> float:
        """Парсит количество: '2 330.77' -> 2330.77"""
        if not qty_str:
            return 0.0
        cleaned = ''.join(qty_str.split()).replace(',', '.')
        try:
            return float(cleaned)
        except ValueError:
            logger.warning(f"Не удалось распарсить количество: '{qty_str}'")
            return 0.0

    @staticmethod
    def _parse_period_date_from_html(path: Path) -> datetime:
        """
        v1.1: Извлекает дату периода из HTML для сортировки.
        Для диапазона берёт конечную дату. Fallback: mtime файла.
        """
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
            m = re.search(r'Период[:\s<>/\w"=]+?>?\s*([^\n<]+)', text)
            if not m:
                m = re.search(r'Период:\s*([^\n<]+)', text)
            if m:
                period_str = m.group(1).strip().rstrip('.')
                # Диапазон DD.MM.YYYY - DD.MM.YYYY → берём конечную дату
                range_m = re.search(
                    r'(\d{1,2})[./](\d{1,2})[./](\d{4})\s*[-–—]\s*(\d{1,2})[./](\d{1,2})[./](\d{4})',
                    period_str
                )
                if range_m:
                    return datetime(int(range_m.group(6)), int(range_m.group(5)), int(range_m.group(4)))
                # Одна дата DD.MM.YYYY
                date_m = re.search(r'(\d{1,2})[./](\d{1,2})[./](\d{4})', period_str)
                if date_m:
                    return datetime(int(date_m.group(3)), int(date_m.group(2)), int(date_m.group(1)))
                # Русские месяцы: DD месяц YYYY
                ru_m = re.search(r'(\d{1,2})\s+([а-яё]+)\s+(\d{4})', period_str.lower())
                if ru_m:
                    month = _MONTHS_RU.get(ru_m.group(2))
                    if month:
                        return datetime(int(ru_m.group(3)), month, int(ru_m.group(1)))
        except Exception as e:
            logger.debug(f"_parse_period_date_from_html({path.name}): {e}")
        # Fallback: mtime
        return datetime.fromtimestamp(path.stat().st_mtime)

    def parse_inventory_html(self, html_path: Path) -> Dict:
        """Парсит HTML остатков"""
        try:
            html_content = html_path.read_text(encoding='utf-8')
            soup = BeautifulSoup(html_content, 'html.parser')
            
            small_tag = soup.find('small')
            date_str = ""
            total_qty = 0.0
            
            if small_tag:
                text = small_tag.get_text()
                if 'Период:' in text or 'период:' in text.lower():
                    parts = text.split('Период:') if 'Период:' in text else text.split('период:')
                    if len(parts) > 1:
                        date_str = parts[1].split('\n')[0].strip()
                
                if 'количество:' in text.lower():
                    parts = text.lower().split('количество:')
                    if len(parts) > 1:
                        qty_str = parts[1].split('\n')[0].strip()
                        total_qty = self.parse_quantity(qty_str)
            
            table = soup.find('table')
            if not table:
                logger.error(f"Не найдена таблица в {html_path}")
                return {'date': date_str, 'total_quantity': total_qty, 'items': []}
            
            items = []
            current_category = ""
            
            for row in table.find_all('tr'):
                if row.find('th'):
                    continue
                
                if 'class' in row.attrs and 'category' in row.attrs['class']:
                    strong_tag = row.find('strong')
                    if strong_tag:
                        current_category = strong_tag.get_text(strip=True)
                    continue
                
                cells = row.find_all('td')
                if len(cells) == 2:
                    product = cells[0].get_text(strip=True)
                    qty_str = cells[1].get_text(strip=True)
                    qty = self.parse_quantity(qty_str)
                    items.append({
                        'category': current_category,
                        'product': product,
                        'quantity': qty
                    })
            
            logger.info(f"📊 Распарсено {len(items)} товаров из {html_path.name}")
            return {'date': date_str, 'total_quantity': total_qty, 'items': items}
            
        except Exception as e:
            logger.error(f"Ошибка при парсинге {html_path}: {e}", exc_info=True)
            return {'date': '', 'total_quantity': 0.0, 'items': []}
    
    def format_summary(self, data: Dict) -> str:
        """Форматирует краткую сводку"""
        msg_lines = [f"📦 ОСТАТКИ на {data['date']}", ""]
        
        total = data['total_quantity']
        msg_lines.append(f"Общее количество: {self.format_number(total)} ед")
        msg_lines.append("")
        
        low_stock = [item for item in data['items'] if item['quantity'] < self.LOW_STOCK_THRESHOLD]
        low_stock.sort(key=lambda x: x['quantity'])
        
        if low_stock:
            msg_lines.append("⚠️ Товаров менее 50 ед:")
            for item in low_stock[:10]:
                product_short = item['product'][:40]
                msg_lines.append(f"  • {product_short} — {self.format_number(item['quantity'])} кг/шт")
            if len(low_stock) > 10:
                msg_lines.append(f"  ... и ещё {len(low_stock) - 10} товаров")
            msg_lines.append("")
        
        category_totals = {}
        for item in data['items']:
            cat = item['category']
            if cat:
                category_totals[cat] = category_totals.get(cat, 0.0) + item['quantity']
        
        sorted_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
        if sorted_categories:
            msg_lines.append("📊 По категориям (топ-5):")
            for i, (cat, qty) in enumerate(sorted_categories[:5], 1):
                msg_lines.append(f"  {i}. {cat} — {self.format_number(qty)} кг/шт")
        
        return "\n".join(msg_lines)
    
    @staticmethod
    def format_number(num: float) -> str:
        return f"{num:,.0f}".replace(',', ' ')
    
    def get_latest_inventory_report(self, reports_dir: Path) -> Optional[Path]:
        """
        v1.1: Находит отчёт остатков с НОВЕЙШИМ ПЕРИОДОМ ДАННЫХ (не mtime).
        """
        pattern = "inventory_*.html"
        matching_files = list(reports_dir.glob(pattern))
        
        if not matching_files:
            logger.warning("Не найдены отчёты остатков")
            return None
        
        # Сортируем по периоду данных из HTML (не по mtime файла)
        latest = max(matching_files, key=lambda p: self._parse_period_date_from_html(p))
        logger.info(f"📄 Найден отчёт остатков: {latest.name} "
                    f"(период: {self._parse_period_date_from_html(latest).strftime('%d.%m.%Y')})")
        return latest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    summary = InventorySummary()
    test_html = Path("/mnt/user-data/uploads/inventory_simple_Остатки_всем__54_.html")
    if test_html.exists():
        data = summary.parse_inventory_html(test_html)
        message = summary.format_summary(data)
        print("\n" + "="*60)
        print(message)
        print("="*60)
"""
Модуль для генерации кратких сводок по остаткам

Версия: 1.5
Дата: 2026-04-13
Изменения v1.4:
  - Fix #INV-2: parse_inventory_json() читает новый JSON-формат inventory.py
    (total_qty / categories[].item_list[].qty вместо total_quantity / items[].quantity)
  - get_latest_inventory_json() — поиск JSON по mtime
  - Используется JSON-первый путь: JSON → HTML-fallback
Изменения v1.2:
  - Fix #INV-1: исправлен glob-паттерн inventory_simple_*.html → inventory_*.html
"""

import json
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
                    try:
                        return datetime(int(range_m.group(6)), int(range_m.group(5)), int(range_m.group(4)))
                    except ValueError:
                        pass
                # Одна дата DD.MM.YYYY
                date_m = re.search(r'(\d{1,2})[./](\d{1,2})[./](\d{4})', period_str)
                if date_m:
                    try:
                        return datetime(int(date_m.group(3)), int(date_m.group(2)), int(date_m.group(1)))
                    except ValueError:
                        pass
                # Русские месяцы: DD месяц YYYY
                ru_m = re.search(r'(\d{1,2})\s+([а-яё]+)\s+(\d{4})', period_str.lower())
                if ru_m:
                    month = _MONTHS_RU.get(ru_m.group(2))
                    if month:
                        try:
                            return datetime(int(ru_m.group(3)), month, int(ru_m.group(1)))
                        except ValueError:
                            pass
        except Exception as e:
            logger.debug(f"_parse_period_date_from_html({path.name}): {e}")
        # Fallback: mtime (защита от FileNotFoundError при конкурентном pipeline)
        try:
            return datetime.fromtimestamp(path.stat().st_mtime)
        except (FileNotFoundError, OSError):
            return datetime.min

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
                # v1.4: regex вместо хрупкого split('\n') / split('количество:')
                _dm = re.search(r'[Пп]ериод[:\s]+(.+?)(?:\n|$)', text)
                if _dm:
                    date_str = _dm.group(1).strip()
                _qm = re.search(r'[Вв]сего\s+количество[:\s]+([\d\s,.\u202f]+)', text)
                if _qm:
                    total_qty = self.parse_quantity(_qm.group(1))
            
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
            
        except (OSError, AttributeError, TypeError, ValueError) as e:
            logger.error(f"Ошибка при парсинге {html_path}: {e}", exc_info=True)
            return {'date': '', 'total_quantity': 0.0, 'items': []}
    
    def format_summary(self, data: Dict) -> str:
        """Форматирует краткую сводку"""
        msg_lines = [f"📦 ОСТАТКИ на {data['date']}", ""]

        total = data['total_quantity']
        msg_lines.append(f"Общее количество: {self.format_number(total)} кг")
        msg_lines.append("")

        low_stock = [item for item in data['items'] if item['quantity'] < self.LOW_STOCK_THRESHOLD]
        low_stock.sort(key=lambda x: x['quantity'])

        if low_stock:
            msg_lines.append("⚠️ Товаров менее 50 кг:")
            for item in low_stock[:10]:
                product_short = item['product'][:40]
                msg_lines.append(f"  • {product_short} — {self.format_number(item['quantity'])} кг")
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
                msg_lines.append(f"  {i}. {cat} — {self.format_number(qty)} кг")

        return "\n".join(msg_lines)
    
    @staticmethod
    def format_number(num: float) -> str:
        return f"{num:,.0f}".replace(',', ' ')
    
    def get_latest_inventory_json(self, json_dir: Path) -> Optional[Path]:
        """v1.4: Находит свежий JSON остатков по mtime."""
        files = list(json_dir.glob("inventory_*.json"))
        if not files:
            return None
        latest = max(files, key=lambda p: p.stat().st_mtime)
        logger.info(f"📄 Найден JSON остатков: {latest.name}")
        return latest

    def parse_inventory_json(self, json_path: Path) -> Dict:
        """v1.5: Читает JSON-формат inventory.py v1.1.6+. Дата берётся из поля period (1C)."""
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            total_qty = float(data.get("total_qty") or 0)
            items: List[Dict] = []
            for cat in data.get("categories", []):
                cat_name = cat.get("category", "")
                for item in cat.get("item_list", []):
                    items.append({
                        "category": cat_name,
                        "product": item.get("product", ""),
                        "quantity": float(item.get("qty") or 0),
                    })
            # Дата периода из 1C (поле period сохраняется с v1.1.6)
            # Fallback: mtime файла
            date_str = ""
            period_raw = data.get("period", "")
            if period_raw and period_raw != "Не указан":
                date_str = period_raw
            if not date_str:
                try:
                    mtime = json_path.stat().st_mtime
                    date_str = datetime.fromtimestamp(mtime).strftime("%d.%m.%Y")
                except Exception:
                    date_str = ""
            logger.info(f"📊 JSON: {len(items)} товаров, итого {total_qty:.0f} кг")
            return {"date": date_str, "total_quantity": total_qty, "items": items}
        except Exception as e:
            logger.error(f"Ошибка при разборе JSON {json_path}: {e}", exc_info=True)
            return {"date": "", "total_quantity": 0.0, "items": []}

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
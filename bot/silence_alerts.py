"""
Модуль для мониторинга дней молчания клиентов в дебиторке
и отправки уведомлений менеджерам

Версия: 1.2
Дата: 25.02.2026
Изменения v1.2:
  - parse_report_date(): добавлена стратегия 2 для debt_ext (stat-label/stat-value)
  - Нормализация em-dash "—" → "-" в датах дебиторки
Изменения v1.1:
  - Добавлен parse_report_date() — извлекает дату отчёта из HTML
  - format_manager_alert() принимает report_date: str = ""
  - format_admin_detailed() принимает manager_dates: dict = None
  - Дата отчёта теперь видна в каждом уведомлении
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class SilenceAlert:
    """Класс для работы с уведомлениями о днях молчания"""
    
    # Пороги дней молчания
    WARNING_DAYS = 7   # 🟡 Внимание
    ALARM_DAYS = 15    # 🟠 Тревога
    CRITICAL_DAYS = 30 # 🔴 Критично
    
    # Минимальный долг для уведомлений (в тенге)
    MIN_DEBT_AMOUNT = 10000.0  # Только клиенты с долгом >= 10,000 ₸
    
    def __init__(self):
        self.stats = {
            'total_checked': 0,
            'managers_with_issues': 0,
            'total_silent_clients': 0,
            'total_silent_debt': 0.0
        }
    
    @staticmethod
    def parse_debt_amount(debt_str: str) -> float:
        """
        Парсит сумму долга из строки
        
        Примеры:
        "3 098 966,13" -> 3098966.13
        "1 671 345,20" -> 1671345.20
        """
        if not debt_str:
            return 0.0
        
        import unicodedata
        cleaned = ''.join(debt_str.split())
        cleaned = cleaned.replace('₸', '').replace('₽', '').strip()
        cleaned = cleaned.replace(',', '.')
        
        try:
            return float(cleaned)
        except ValueError:
            logger.warning(f"Не удалось распарсить сумму: '{debt_str}' (cleaned: '{cleaned}')")
            return 0.0

    @staticmethod
    def parse_report_date(html_path: Path) -> str:
        """
        v1.2: Извлекает дату/период отчёта из HTML.
        Поддерживает три формата:
        1. <small><span class="key">Период:</span> дата</small>  — inventory/gross/sales
        2. <div class="stat-label">Период</div><div class="stat-value">дата</div>  — debt_ext
        3. Regex fallback: Период: дата  — expenses и прочие
        """
        try:
            text = html_path.read_text(encoding='utf-8', errors='ignore')
            soup = BeautifulSoup(text, 'html.parser')

            # Стратегия 1: <small> тег (inventory, gross, sales)
            small_tag = soup.find('small')
            if small_tag:
                small_text = small_tag.get_text()
                if 'Период:' in small_text:
                    parts = small_text.split('Период:')
                    if len(parts) > 1:
                        date_str = parts[1].split('\n')[0].strip()
                        if date_str:
                            return date_str

            # Стратегия 2: stat-label/stat-value (debt_ext — Детальный, Ведомость)
            for label_tag in soup.find_all(class_='stat-label'):
                if 'Период' in label_tag.get_text(strip=True):
                    value_tag = label_tag.find_next_sibling(class_='stat-value')
                    if value_tag:
                        date_str = value_tag.get_text(strip=True)
                        # Нормализуем em-dash "—" → "-"
                        date_str = date_str.replace('\u2014', '-').replace('\u2013', '-')
                        if date_str:
                            return date_str

            # Стратегия 3: Regex fallback (expenses, прочие)
            m = re.search(r'Период[:\s]+([^\n<]+)', text)
            if m:
                return m.group(1).strip().rstrip('.')

        except Exception as e:
            logger.warning(f"Не удалось извлечь дату из {html_path.name}: {e}")
        return ""

    def parse_html_silence_days(self, html_path: Path) -> List[Dict]:
        """
        Парсит HTML файл детальной дебиторки и извлекает данные о днях молчания
        
        Возвращает список словарей:
        [
            {
                'client': 'Е Олжас',
                'debt': 3098966.13,
                'debt_str': '3 098 966,13 ₸',
                'silence_days': 0
            },
            ...
        ]
        """
        try:
            html_content = html_path.read_text(encoding='utf-8')
            soup = BeautifulSoup(html_content, 'html.parser')
            
            all_clients_panel = soup.find('div', id='t_all')
            if not all_clients_panel:
                logger.error(f"Не найдена вкладка 't_all' в {html_path}")
                return []
            
            table = all_clients_panel.find('table')
            if not table:
                logger.error(f"Не найдена таблица в вкладке 't_all' в {html_path}")
                return []
            
            tbody = table.find('tbody')
            if not tbody:
                logger.error(f"Не найден tbody в таблице {html_path}")
                return []
            
            clients_data = []
            
            for row in tbody.find_all('tr'):
                cells = row.find_all('td')
                
                if len(cells) < 7:
                    continue
                
                client_name = cells[0].get_text(strip=True)
                debt_str = cells[1].get_text(strip=True)
                silence_days_str = cells[6].get_text(strip=True)
                
                try:
                    silence_days = int(silence_days_str)
                except ValueError:
                    logger.warning(f"Не удалось распарсить дни молчания: '{silence_days_str}' для {client_name}")
                    continue
                
                debt_amount = self.parse_debt_amount(debt_str)
                
                clients_data.append({
                    'client': client_name,
                    'debt': debt_amount,
                    'debt_str': debt_str,
                    'silence_days': silence_days
                })
            
            logger.info(f"📊 Распарсено {len(clients_data)} клиентов из {html_path.name}")
            return clients_data
            
        except Exception as e:
            logger.error(f"Ошибка при парсинге {html_path}: {e}", exc_info=True)
            return []
    
    def categorize_by_silence(self, clients_data: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Группирует клиентов по критичности дней молчания.
        Учитываются только клиенты с долгом >= MIN_DEBT_AMOUNT.
        """
        categorized = {
            'critical': [],
            'alarm': [],
            'warning': []
        }
        
        for client in clients_data:
            days = client['silence_days']
            debt = client['debt']
            
            if debt < self.MIN_DEBT_AMOUNT:
                continue
            
            if days >= self.CRITICAL_DAYS:
                categorized['critical'].append(client)
            elif days >= self.ALARM_DAYS:
                categorized['alarm'].append(client)
            elif days >= self.WARNING_DAYS:
                categorized['warning'].append(client)
        
        return categorized
    
    def format_manager_alert(self, manager_name: str, categorized: Dict[str, List[Dict]], report_date: str = "") -> str:
        """
        v1.1: Формирует текст уведомления для менеджера.
        report_date — строка с датой отчёта (опционально).
        """
        total_silent = len(categorized['critical']) + len(categorized['alarm']) + len(categorized['warning'])
        
        if total_silent == 0:
            return None
        
        date_line = f"\n📅 Отчёт за: {report_date}" if report_date else ""
        
        msg_lines = [
            f"⚠️ ОТЧЁТ ПО ДНЯМ МОЛЧАНИЯ{date_line}",
            "",
            f"👨‍💼 {manager_name}, у вас клиенты молчат:",
            ""
        ]
        
        if categorized['critical']:
            msg_lines.append("🔴 КРИТИЧНО (30+ дней):")
            total_critical_debt = 0.0
            for client in categorized['critical'][:10]:
                msg_lines.append(f"  • {client['client']} — {client['debt_str']} ({client['silence_days']} дн)")
                total_critical_debt += client['debt']
            
            if len(categorized['critical']) > 10:
                msg_lines.append(f"  ... и ещё {len(categorized['critical']) - 10} клиентов")
            
            msg_lines.append(f"  💰 Сумма: {self.format_amount(total_critical_debt)} ₸")
            msg_lines.append("")
        
        if categorized['alarm']:
            msg_lines.append("🟠 ТРЕВОГА (15-29 дней):")
            total_alarm_debt = 0.0
            for client in categorized['alarm'][:5]:
                msg_lines.append(f"  • {client['client']} — {client['debt_str']} ({client['silence_days']} дн)")
                total_alarm_debt += client['debt']
            
            if len(categorized['alarm']) > 5:
                msg_lines.append(f"  ... и ещё {len(categorized['alarm']) - 5} клиентов")
            
            msg_lines.append(f"  💰 Сумма: {self.format_amount(total_alarm_debt)} ₸")
            msg_lines.append("")
        
        if categorized['warning']:
            msg_lines.append("🟡 ВНИМАНИЕ (7-14 дней):")
            total_warning_debt = 0.0
            for client in categorized['warning'][:5]:
                msg_lines.append(f"  • {client['client']} — {client['debt_str']} ({client['silence_days']} дн)")
                total_warning_debt += client['debt']
            
            if len(categorized['warning']) > 5:
                msg_lines.append(f"  ... и ещё {len(categorized['warning']) - 5} клиентов")
            
            msg_lines.append(f"  💰 Сумма: {self.format_amount(total_warning_debt)} ₸")
            msg_lines.append("")
        
        total_debt = sum(c['debt'] for cats in categorized.values() for c in cats)
        msg_lines.append(f"💰 Общий долг молчащих: {self.format_amount(total_debt)} ₸")
        msg_lines.append("")
        msg_lines.append("📊 Открыть детальный отчёт → /debt")
        
        return "\n".join(msg_lines)
    
    def format_admin_summary(self, all_managers_data: Dict[str, Dict]) -> str:
        """Формирует сводку для админа по всем менеджерам"""
        msg_lines = [
            "⚠️ СВОДКА: ДНИ МОЛЧАНИЯ ПО ВСЕМ МЕНЕДЖЕРАМ",
            ""
        ]
        
        total_overall_debt = 0.0
        managers_with_issues = 0
        
        for manager_name, categorized in sorted(all_managers_data.items()):
            critical_count = len(categorized['critical'])
            alarm_count = len(categorized['alarm'])
            warning_count = len(categorized['warning'])
            total_count = critical_count + alarm_count + warning_count
            
            if total_count == 0:
                continue
            
            managers_with_issues += 1
            msg_lines.append(f"👨‍💼 {manager_name}:")
            
            if critical_count > 0:
                critical_debt = sum(c['debt'] for c in categorized['critical'])
                msg_lines.append(f"  🔴 {critical_count} клиент(ов) (30+ дн) — {self.format_amount(critical_debt)} ₸")
                total_overall_debt += critical_debt
            
            if alarm_count > 0:
                alarm_debt = sum(c['debt'] for c in categorized['alarm'])
                msg_lines.append(f"  🟠 {alarm_count} клиент(ов) (15-29 дн) — {self.format_amount(alarm_debt)} ₸")
                total_overall_debt += alarm_debt
            
            if warning_count > 0:
                warning_debt = sum(c['debt'] for c in categorized['warning'])
                msg_lines.append(f"  🟡 {warning_count} клиент(ов) (7-14 дн) — {self.format_amount(warning_debt)} ₸")
                total_overall_debt += warning_debt
            
            msg_lines.append("")
        
        if managers_with_issues == 0:
            return "✅ У всех менеджеров нет критичных дней молчания!"
        
        msg_lines.append(f"💰 Всего молчащих: {self.format_amount(total_overall_debt)} ₸")
        msg_lines.append(f"📊 Менеджеров с проблемами: {managers_with_issues}")
        
        return "\n".join(msg_lines)

    def format_admin_detailed(self, all_managers_data: Dict[str, Dict], manager_dates: Dict[str, str] = None) -> str:
        """
        v1.1: Формирует ДЕТАЛЬНУЮ сводку для админа с именами клиентов, суммами и днями.
        manager_dates — словарь {manager_name: report_date_str} (опционально).
        """
        msg_lines = [
            "⚠️ ДЕТАЛЬНАЯ СВОДКА: ДНИ МОЛЧАНИЯ",
            ""
        ]
        
        total_overall_debt = 0.0
        managers_with_issues = 0
        total_overall_clients = 0
        
        for manager_name, categorized in sorted(all_managers_data.items()):
            critical_count = len(categorized['critical'])
            alarm_count = len(categorized['alarm'])
            warning_count = len(categorized['warning'])
            total_count = critical_count + alarm_count + warning_count
            
            if total_count == 0:
                continue
            
            managers_with_issues += 1
            total_overall_clients += total_count
            
            # v1.1: дата отчёта по менеджеру
            report_date = (manager_dates or {}).get(manager_name, "")
            date_suffix = f" | 📅 {report_date}" if report_date else ""
            
            msg_lines.append(f"👨‍💼 {manager_name.upper()}{date_suffix}")
            msg_lines.append("━" * 50)
            
            if critical_count > 0:
                msg_lines.append("🔴 КРИТИЧНО (30+ дней):")
                critical_debt_total = 0.0
                
                show_count = min(20, critical_count)
                for client in categorized['critical'][:show_count]:
                    msg_lines.append(f"  • {client['client']} — {client['debt_str']} ({client['silence_days']} дн)")
                    critical_debt_total += client['debt']
                
                if critical_count > 20:
                    remaining = critical_count - 20
                    remaining_debt = sum(c['debt'] for c in categorized['critical'][20:])
                    msg_lines.append(f"  ... и ещё {remaining} клиент(ов) на {self.format_amount(remaining_debt)} ₸")
                    critical_debt_total += remaining_debt
                
                msg_lines.append(f"  💰 Итого критично: {self.format_amount(critical_debt_total)} ₸")
                msg_lines.append("")
                total_overall_debt += critical_debt_total
            
            if alarm_count > 0:
                msg_lines.append("🟠 ТРЕВОГА (15-29 дней):")
                alarm_debt_total = 0.0
                
                show_count = min(10, alarm_count)
                for client in categorized['alarm'][:show_count]:
                    msg_lines.append(f"  • {client['client']} — {client['debt_str']} ({client['silence_days']} дн)")
                    alarm_debt_total += client['debt']
                
                if alarm_count > 10:
                    remaining = alarm_count - 10
                    remaining_debt = sum(c['debt'] for c in categorized['alarm'][10:])
                    msg_lines.append(f"  ... и ещё {remaining} клиент(ов) на {self.format_amount(remaining_debt)} ₸")
                    alarm_debt_total += remaining_debt
                
                msg_lines.append(f"  💰 Итого тревога: {self.format_amount(alarm_debt_total)} ₸")
                msg_lines.append("")
                total_overall_debt += alarm_debt_total
            
            if warning_count > 0:
                msg_lines.append("🟡 ВНИМАНИЕ (7-14 дней):")
                warning_debt_total = 0.0
                
                show_count = min(10, warning_count)
                for client in categorized['warning'][:show_count]:
                    msg_lines.append(f"  • {client['client']} — {client['debt_str']} ({client['silence_days']} дн)")
                    warning_debt_total += client['debt']
                
                if warning_count > 10:
                    remaining = warning_count - 10
                    remaining_debt = sum(c['debt'] for c in categorized['warning'][10:])
                    msg_lines.append(f"  ... и ещё {remaining} клиент(ов) на {self.format_amount(remaining_debt)} ₸")
                    warning_debt_total += remaining_debt
                
                msg_lines.append(f"  💰 Итого внимание: {self.format_amount(warning_debt_total)} ₸")
                msg_lines.append("")
                total_overall_debt += warning_debt_total
            
            msg_lines.append("")
        
        if managers_with_issues == 0:
            return "✅ У всех менеджеров нет критичных дней молчания!"
        
        msg_lines.append("━" * 50)
        msg_lines.append(f"💰 ВСЕГО МОЛЧАЩИХ: {self.format_amount(total_overall_debt)} ₸")
        msg_lines.append(f"📊 Менеджеров с проблемами: {managers_with_issues}")
        msg_lines.append(f"👥 Всего молчащих клиентов: {total_overall_clients}")
        
        return "\n".join(msg_lines)
    
    @staticmethod
    def format_amount(amount: float) -> str:
        """Форматирует сумму с пробелами между тысячами"""
        return f"{amount:,.2f}".replace(',', ' ').replace('.', ',')
    
    def get_latest_debt_report(self, reports_dir: Path, manager_name: str) -> Optional[Path]:
        """
        v1.1: Находит отчёт дебиторки с наиболее свежим ПЕРИОДОМ ДАННЫХ.
        Сортирует по дате из HTML (не по mtime файла).
        """
        pattern = f"debt_ext_Детальный_Дебиторы_{manager_name}*.html"
        matching_files = list(reports_dir.glob(pattern))

        if not matching_files:
            pattern2 = f"debt_ext*{manager_name}*.html"
            matching_files = list(reports_dir.glob(pattern2))

        if not matching_files:
            logger.warning(f"Не найдены отчёты дебиторки для {manager_name}")
            return None

        def _get_period_date(p: Path) -> tuple:
            date_str = self.parse_report_date(p)
            # Сортируем по конечной дате диапазона
            m = re.search(r'(\d{1,2})[./](\d{1,2})[./](\d{4})\s*[-–—]\s*(\d{1,2})[./](\d{1,2})[./](\d{4})', date_str)
            if m:
                try:
                    return (int(m.group(6)), int(m.group(5)), int(m.group(4)))
                except Exception:
                    pass
            m2 = re.search(r'(\d{1,2})[./](\d{1,2})[./](\d{4})', date_str)
            if m2:
                try:
                    return (int(m2.group(3)), int(m2.group(2)), int(m2.group(1)))
                except Exception:
                    pass
            # Русский формат
            MONTHS = {"января":1,"февраля":2,"марта":3,"апреля":4,"мая":5,"июня":6,
                      "июля":7,"августа":8,"сентября":9,"октября":10,"ноября":11,"декабря":12}
            m3 = re.search(r'(\d{1,2})\s+([а-яё]+)\s+(\d{4})', date_str.lower())
            if m3:
                mon = MONTHS.get(m3.group(2))
                if mon:
                    try:
                        return (int(m3.group(3)), mon, int(m3.group(1)))
                    except Exception:
                        pass
            return (0, 0, 0)

        latest = max(matching_files, key=lambda p: (_get_period_date(p), p.stat().st_mtime))
        date_str = self.parse_report_date(latest)
        logger.info(f"📄 Найден отчёт для {manager_name}: {latest.name} (период={date_str})")
        return latest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    alert = SilenceAlert()
    test_html = Path("/mnt/user-data/uploads/debt_ext_Детальный_Дебиторы_Ергали__30_.html")
    if test_html.exists():
        date_str = alert.parse_report_date(test_html)
        print(f"Дата отчёта: {date_str}")
        clients = alert.parse_html_silence_days(test_html)
        print(f"Найдено клиентов: {len(clients)}")
        categorized = alert.categorize_by_silence(clients)
        message = alert.format_manager_alert("Ергали", categorized, report_date=date_str)
        print(message)
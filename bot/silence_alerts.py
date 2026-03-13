"""
Модуль для мониторинга дней молчания клиентов в дебиторке
и отправки уведомлений менеджерам

Версия: 1.4
Дата: 2026-03-11
Изменения v1.4:
  - parse_html_silence_days(): добавлен парсинг cells[3] (Отгрузка/debit).
  - categorize_by_silence(): исправлена логика partial_payment:
      Основной фильтр — days_silence >= 7 (как и раньше, именно он определяет молчуна).
      Флаг partial_payment ставится только клиентам с debit == 0 (нет заказов)
      И paid > 0 (что-то платит) — это паттерн "имитация оплаты для сброса счётчика".
      Клиенты с debit > 0 и любым days — показываются без флага (нормальная задолженность).
      debit == 0 AND days < 7 AND paid > 0 AND debt > 0
        → "подозрительная оплата": счётчик сброшен платежом, но товар не брали
        → категория partial_payment (отдельный блок внизу).
  - Убрана ошибка v1.3: активные клиенты (debit > 0, days 0-6) больше не
    попадают в partial_payment (там были нормальные торговые клиенты).
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

    # Минимальный долг для стандартных категорий (7+ дней)
    MIN_DEBT_AMOUNT = 10_000.0

    # Минимальный долг для категории "частичная оплата" (days < 7).
    # Клиент с days < 7, любой отгрузкой, но ДОЛГОМ >= этого порога
    # остаётся виден — ситуация "берут и платят, но большой долг не гасится".
    PARTIAL_PAYMENT_MIN_DEBT = 100_000.0
    
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

                # v1.4: col0=Клиент, col1=Долг, col2=Нач.остаток,
                #        col3=Отгрузка, col4=Оплата, col5=Операций, col6=Дни молчания
                debit_str  = cells[3].get_text(strip=True) if len(cells) > 3 else ""
                paid_str   = cells[4].get_text(strip=True) if len(cells) > 4 else ""
                debit_amount = self.parse_debt_amount(debit_str)
                paid_amount  = self.parse_debt_amount(paid_str)

                clients_data.append({
                    'client':        client_name,
                    'debt':          debt_amount,
                    'debt_str':      debt_str,
                    'silence_days':  silence_days,
                    'debit_amount':  debit_amount,   # отгрузка в периоде
                    'debit_str':     debit_str,
                    'paid_amount':   paid_amount,    # оплата в периоде
                    'paid_str':      paid_str,
                })
            
            logger.info(f"📊 Распарсено {len(clients_data)} клиентов из {html_path.name}")
            return clients_data
            
        except Exception as e:
            logger.error(f"Ошибка при парсинге {html_path}: {e}", exc_info=True)
            return []
    
    def categorize_by_silence(self, clients_data: List[Dict],
                              historical_map: Optional[Dict[str, int]] = None) -> Dict[str, List[Dict]]:
        """
        v1.4: Группирует клиентов по критичности дней молчания.
        Учитываются только клиенты с долгом >= MIN_DEBT_AMOUNT.

        Основной критерий молчания — days_silence:
          days >= 30 → 'critical'
          days >= 15 → 'alarm'
          days >= 7  → 'warning'

        Флаг partial_payment (⚠️ в строке клиента) ставится когда:
          debit_amount == 0 AND paid_amount > 0
          → клиент НЕ БРАЛ товар в периоде, но что-то оплатил
          → это паттерн "имитация активности": платит мало, чтобы сбросить
            счётчик days_silence в 1С и исчезнуть из контроля.

        Отдельная категория 'partial_payment' (days < 7):
          debit == 0 AND days < 7 AND paid > 0 AND debt > 0
          → счётчик сброшен платежом (days стал 0-6), товар не брали
          → клиент выжил из стандартной выборки, показываем отдельно.

        Клиенты с debit > 0: берут товар → нормальная торговая задолженность,
          флаг НЕ ставим (они не "имитируют", а реально работают).
        """
        categorized: Dict[str, List[Dict]] = {
            'critical':        [],   # 30+ дней
            'alarm':           [],   # 15-29 дней
            'warning':         [],   # 7-14 дней
            'partial_payment': [],   # days < 7, нет заказов, но платит (имитация)
        }

        for client in clients_data:
            days         = client['silence_days']
            debt         = client['debt']
            debit_amount = client.get('debit_amount', 0.0)
            paid_amount  = client.get('paid_amount', 0.0)

            if debt < self.MIN_DEBT_AMOUNT:
                continue

            # Флаг "имитация оплаты": нет заказов, но что-то заплатил
            is_fake_payment = (debit_amount == 0 and paid_amount > 0 and debt > 0)

            # Крупный должник: берёт товар И платит, но долг >= PARTIAL_PAYMENT_MIN_DEBT
            # и days < 7 (счётчик обнулён оплатой или недавней отгрузкой).
            # Такие клиенты должны оставаться видимыми, пока долг не погашен полностью.
            is_large_partial = (
                paid_amount > 0
                and debt >= self.PARTIAL_PAYMENT_MIN_DEBT
            )

            if days >= self.CRITICAL_DAYS:
                categorized['critical'].append(dict(client, partial_payment=is_fake_payment))
            elif days >= self.ALARM_DAYS:
                categorized['alarm'].append(dict(client, partial_payment=is_fake_payment))
            elif days >= self.WARNING_DAYS:
                categorized['warning'].append(dict(client, partial_payment=is_fake_payment))
            elif is_fake_payment or is_large_partial:
                # days < 7: либо нет заказов + платит (имитация),
                # либо крупный долг >= 100K + оплата (не гасится полностью).
                # Ищем исторические дни молчания в предыдущем отчёте:
                # показываем "оплачено (N дн)" — сколько дней клиент числился
                # молчащим до того, как оплата сбросила счётчик в 1С.
                _raw_hist = (historical_map or {}).get(client['client']) if historical_map else None
                # Показываем "было N дн молчания" только когда счётчик был сброшен
                # оплатой: исторические дни ЗНАЧИТЕЛЬНО выше текущих.
                # Если дни растут или стоят на месте — оплаты не было, метка вводит в заблуждение.
                hist_days = _raw_hist if (_raw_hist and _raw_hist > client['silence_days'] + 2) else None
                categorized['partial_payment'].append(dict(
                    client,
                    partial_payment=True,
                    historical_days=hist_days,  # None если нет сброса счётчика
                ))
            # else: days < 7, маленький долг или нет оплаты → пропускаем

        return categorized
    
    def format_manager_alert(self, manager_name: str, categorized: Dict[str, List[Dict]], report_date: str = "") -> str:
        """
        v1.3: Формирует текст уведомления для менеджера.
        report_date — строка с датой отчёта (опционально).
        Включает блок "💛 ЧАСТИЧНАЯ ОПЛАТА" для клиентов с долгом и частичной оплатой.
        """
        total_silent = (
            len(categorized.get('critical', []))
            + len(categorized.get('alarm', []))
            + len(categorized.get('warning', []))
            + len(categorized.get('partial_payment', []))
        )

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
        
        if categorized.get('warning'):
            msg_lines.append("🟡 ВНИМАНИЕ (7-14 дней):")
            total_warning_debt = 0.0
            for client in categorized['warning'][:5]:
                msg_lines.append(f"  • {client['client']} — {client['debt_str']} ({client['silence_days']} дн)")
                total_warning_debt += client['debt']

            if len(categorized['warning']) > 5:
                msg_lines.append(f"  ... и ещё {len(categorized['warning']) - 5} клиентов")

            msg_lines.append(f"  💰 Сумма: {self.format_amount(total_warning_debt)} ₸")
            msg_lines.append("")

        # v1.5: блок частичных оплат с расшифровкой скобок.
        # "было N дн молчания" = из предыдущего отчёта (historical_days): сколько дней
        # клиент числился молчащим ДО того, как оплата сбросила счётчик в 1С.
        # "сейчас Y дн" = текущие дни молчания после оплаты.
        if categorized.get('partial_payment'):
            msg_lines.append("💛 ЧАСТИЧНАЯ ОПЛАТА (долг не закрыт):")
            total_partial_debt = 0.0
            for client in categorized['partial_payment'][:5]:
                paid_str = client.get('paid_str', '') or self.format_amount(client.get('paid_amount', 0))
                hist = client.get('historical_days')
                hist_part = f"было {hist} дн молчания, " if hist else ""
                msg_lines.append(
                    f"  • {client['client']} — долг: {client['debt_str']}, "
                    f"оплачено: {paid_str} ({hist_part}сейчас {client['silence_days']} дн)"
                )
                total_partial_debt += client['debt']

            if len(categorized['partial_payment']) > 5:
                msg_lines.append(f"  ... и ещё {len(categorized['partial_payment']) - 5} клиентов")

            msg_lines.append(f"  💰 Остаток долга: {self.format_amount(total_partial_debt)} ₸")
            msg_lines.append("")

        total_debt = sum(c['debt'] for cats in categorized.values() for c in cats)
        msg_lines.append(f"💰 Общий долг молчащих: {self.format_amount(total_debt)} ₸")
        msg_lines.append("")
        msg_lines.append("📊 Открыть детальный отчёт → /debt")

        return "\n".join(msg_lines)
    
    def format_admin_summary(self, all_managers_data: Dict[str, Dict]) -> str:
        """Формирует сводку для админа по всем менеджерам.
        v1.3: учитывает категорию partial_payment."""
        msg_lines = [
            "⚠️ СВОДКА: ДНИ МОЛЧАНИЯ ПО ВСЕМ МЕНЕДЖЕРАМ",
            ""
        ]

        total_overall_debt = 0.0
        managers_with_issues = 0

        for manager_name, categorized in sorted(all_managers_data.items()):
            critical_count = len(categorized.get('critical', []))
            alarm_count = len(categorized.get('alarm', []))
            warning_count = len(categorized.get('warning', []))
            partial_count = len(categorized.get('partial_payment', []))
            total_count = critical_count + alarm_count + warning_count + partial_count

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

            # v1.3: частичная оплата
            if partial_count > 0:
                partial_debt = sum(c['debt'] for c in categorized['partial_payment'])
                msg_lines.append(f"  💛 {partial_count} клиент(ов) частичная оплата — {self.format_amount(partial_debt)} ₸")
                total_overall_debt += partial_debt

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
            critical_count = len(categorized.get('critical', []))
            alarm_count = len(categorized.get('alarm', []))
            warning_count = len(categorized.get('warning', []))
            partial_count = len(categorized.get('partial_payment', []))
            total_count = critical_count + alarm_count + warning_count + partial_count

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
                    suffix = " ⚠️ частичная оплата" if client.get('partial_payment') else ""
                    msg_lines.append(f"  • {client['client']} — {client['debt_str']} ({client['silence_days']} дн){suffix}")
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
                    suffix = " ⚠️ частичная оплата" if client.get('partial_payment') else ""
                    msg_lines.append(f"  • {client['client']} — {client['debt_str']} ({client['silence_days']} дн){suffix}")
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
                    suffix = " ⚠️ частичная оплата" if client.get('partial_payment') else ""
                    msg_lines.append(f"  • {client['client']} — {client['debt_str']} ({client['silence_days']} дн){suffix}")
                    warning_debt_total += client['debt']

                if warning_count > 10:
                    remaining = warning_count - 10
                    remaining_debt = sum(c['debt'] for c in categorized['warning'][10:])
                    msg_lines.append(f"  ... и ещё {remaining} клиент(ов) на {self.format_amount(remaining_debt)} ₸")
                    warning_debt_total += remaining_debt

                msg_lines.append(f"  💰 Итого внимание: {self.format_amount(warning_debt_total)} ₸")
                msg_lines.append("")
                total_overall_debt += warning_debt_total

            # v1.5: блок частичных оплат с расшифровкой скобок
            if partial_count > 0:
                msg_lines.append("💛 ЧАСТИЧНАЯ ОПЛАТА (долг не закрыт):")
                partial_debt_total = 0.0

                show_count = min(10, partial_count)
                for client in categorized['partial_payment'][:show_count]:
                    paid_str = client.get('paid_str', '') or self.format_amount(client.get('paid_amount', 0))
                    hist = client.get('historical_days')
                    hist_part = f"было {hist} дн молчания, " if hist else ""
                    msg_lines.append(
                        f"  • {client['client']} — долг: {client['debt_str']}, "
                        f"оплачено: {paid_str} ({hist_part}сейчас {client['silence_days']} дн)"
                    )
                    partial_debt_total += client['debt']

                if partial_count > 10:
                    remaining = partial_count - 10
                    remaining_debt = sum(c['debt'] for c in categorized['partial_payment'][10:])
                    msg_lines.append(f"  ... и ещё {remaining} клиент(ов) на {self.format_amount(remaining_debt)} ₸")
                    partial_debt_total += remaining_debt

                msg_lines.append(f"  💰 Остаток долга: {self.format_amount(partial_debt_total)} ₸")
                msg_lines.append("")
                total_overall_debt += partial_debt_total

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
    
    # ──────────────────────────────────────────────────────────────────────
    # Вспомогательный метод сортировки файлов отчётов по периоду
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _period_sort_key(p: Path, date_str: str) -> tuple:
        """Возвращает (year, month, day) конечной даты периода для сортировки."""
        m = re.search(
            r'(\d{1,2})[./](\d{1,2})[./](\d{4})\s*[-–—]\s*(\d{1,2})[./](\d{1,2})[./](\d{4})',
            date_str
        )
        if m:
            try:
                return (int(m.group(6)), int(m.group(5)), int(m.group(4)))
            except (ValueError, TypeError):
                pass
        m2 = re.search(r'(\d{1,2})[./](\d{1,2})[./](\d{4})', date_str)
        if m2:
            try:
                return (int(m2.group(3)), int(m2.group(2)), int(m2.group(1)))
            except (ValueError, TypeError):
                pass
        MONTHS = {"января":1,"февраля":2,"марта":3,"апреля":4,"мая":5,"июня":6,
                  "июля":7,"августа":8,"сентября":9,"октября":10,"ноября":11,"декабря":12}
        m3 = re.search(r'(\d{1,2})\s+([а-яё]+)\s+(\d{4})', date_str.lower())
        if m3:
            mon = MONTHS.get(m3.group(2))
            if mon:
                try:
                    return (int(m3.group(3)), mon, int(m3.group(1)))
                except (ValueError, TypeError):
                    pass
        return (0, 0, p.stat().st_mtime)

    def _get_all_debt_reports(self, reports_dir: Path, manager_name: str) -> List[Path]:
        """
        Возвращает все файлы типа 'Детальный Дебиторы' для менеджера,
        отсортированные по периоду данных (старые → новые).

        Намеренно ограничивается только этим типом отчёта — не смешивает
        с 'Ведомость по взаиморасчетам' и другими форматами, у которых
        другая HTML-структура и другие номера в имени файла.
        """
        # Ищем с пробелами и с подчёркиваниями — 1С генерирует оба варианта
        seen: dict = {}
        for pattern in [
            f"debt_ext_Детальный Дебиторы {manager_name}*.html",
            f"debt_ext_Детальный_Дебиторы_{manager_name}*.html",
            f"debt_ext_Детальный Дебиторы_{manager_name}*.html",
            f"debt_ext_Детальный_Дебиторы {manager_name}*.html",
        ]:
            for f in reports_dir.glob(pattern):
                seen[f.name] = f

        files = list(seen.values())

        if not files:
            # Резерв: берём только файлы с "детальный" в имени
            files = [
                f for f in reports_dir.glob(f"debt_ext*{manager_name}*.html")
                if "детальный" in f.name.lower()
            ]

        if not files:
            return []

        return sorted(files, key=lambda p: (
            self._period_sort_key(p, self.parse_report_date(p)),
            p.stat().st_mtime
        ))

    def get_latest_debt_report(self, reports_dir: Path, manager_name: str) -> Optional[Path]:
        """
        v1.4: Находит отчёт дебиторки с наиболее свежим периодом данных.
        Делегирует _get_all_debt_reports и берёт последний элемент.
        """
        files = self._get_all_debt_reports(reports_dir, manager_name)
        if not files:
            logger.warning(f"Не найдены отчёты дебиторки для {manager_name}")
            return None
        latest = files[-1]
        logger.info(f"📄 Найден отчёт для {manager_name}: {latest.name} "
                    f"(период={self.parse_report_date(latest)})")
        return latest

    def get_prev_debt_report(self, reports_dir: Path, manager_name: str) -> Optional[Path]:
        """
        v1.4: Возвращает предпоследний файл дебиторки менеджера.
        Используется для восстановления исторических дней молчания
        клиентов, у которых счётчик был сброшен оплатой.
        """
        files = self._get_all_debt_reports(reports_dir, manager_name)
        if len(files) < 2:
            return None
        prev = files[-2]
        logger.debug(f"📄 Предыдущий отчёт для {manager_name}: {prev.name}")
        return prev

    def build_historical_silence_map(self, html_path: Path) -> Dict[str, int]:
        """
        v1.4: Парсит предыдущий HTML-файл и возвращает словарь
        {client_name: days_silence} для всех клиентов с days > 0.
        Используется для отображения "оплачено (N дн)" в partial_payment:
        показывает, сколько дней клиент числился молчащим до оплаты.
        """
        try:
            clients = self.parse_html_silence_days(html_path)
            return {c['client']: c['silence_days'] for c in clients if c['silence_days'] > 0}
        except Exception as e:
            logger.warning(f"build_historical_silence_map: не удалось прочитать {html_path.name}: {e}")
            return {}


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
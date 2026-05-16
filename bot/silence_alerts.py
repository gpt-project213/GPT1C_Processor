"""
Модуль для мониторинга дней молчания клиентов в дебиторке
и отправки уведомлений менеджерам

Версия: 1.7
Дата: 2026-03-25
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
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def _safe_mtime(p: Path) -> float:
    """p.stat().st_mtime с защитой от FileNotFoundError при конкурентном pipeline."""
    try:
        return p.stat().st_mtime
    except (FileNotFoundError, OSError):
        return 0.0


class SilenceAlert:
    """Класс для работы с уведомлениями о днях молчания"""
    
    # Пороги дней молчания
    OVERDUE_DAYS  = 7   # ⚡ Просрочка (7-9 дн) — не рассчитался в срок
    SILENCE_DAYS  = 10  # 🟡 Молчание (10-14 дн) — не реагирует
    ALARM_DAYS    = 15  # 🟠 Тревога (15-29 дн)
    CRITICAL_DAYS = 30  # 🔴 Критично (30+ дн)

    # Минимальный долг для отображения (все категории)
    MIN_DEBT_AMOUNT = 5_000.0

    # Порог "имитации оплаты": оплата < IMITATION_THRESHOLD * долг
    IMITATION_THRESHOLD = 0.10  # 10%
    
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

            # v1.6: читаем заголовки из <thead> вместо жёстких индексов
            # Ожидаемые колонки: Клиент, Долг, Нач.остаток, Отгрузка, Оплата, Операций, Дни молчания
            _FALLBACK = {
                "client": 0, "debt": 1, "initial": 2,
                "debit": 3, "paid": 4, "silence": 6,
            }
            col_idx = dict(_FALLBACK)  # начинаем с fallback
            thead = table.find('thead')
            if thead:
                ths = thead.find_all('th')
                if ths:
                    _header_map: Dict[str, int] = {}
                    for idx, th in enumerate(ths):
                        _header_map[th.get_text(strip=True).lower()] = idx
                    # Сопоставляем ключевые заголовки (без учёта регистра/пробелов)
                    _KNOWN = {
                        "client":  ("клиент",),
                        "debt":    ("долг",),
                        "initial": ("нач.остаток", "нач. остаток", "начостаток"),
                        "debit":   ("отгрузка",),
                        "paid":    ("оплата",),
                        "silence": ("дни молчания", "дни_молчания", "silence"),
                    }
                    for key, variants in _KNOWN.items():
                        for variant in variants:
                            if variant in _header_map:
                                col_idx[key] = _header_map[variant]
                                break
                    logger.debug("Карта колонок из thead: %s", col_idx)
                else:
                    logger.warning("thead найден, но <th> отсутствуют — используются fallback индексы")
            else:
                logger.warning("thead не найден в таблице %s — используются fallback индексы", html_path.name)

            min_cols = max(col_idx["client"], col_idx["debt"], col_idx["silence"]) + 1

            clients_data = []

            for row in tbody.find_all('tr'):
                cells = row.find_all('td')

                if len(cells) < min_cols:
                    continue

                def _cell(key: str) -> str:
                    j = col_idx[key]
                    return cells[j].get_text(strip=True) if j < len(cells) else ""

                client_name = _cell("client")
                debt_str = _cell("debt")
                silence_days_str = _cell("silence")

                try:
                    silence_days = int(silence_days_str)
                except ValueError:
                    logger.warning(f"Не удалось распарсить дни молчания: '{silence_days_str}' для {client_name}")
                    continue

                debt_amount = self.parse_debt_amount(debt_str)

                initial_str = _cell("initial")
                debit_str   = _cell("debit")
                paid_str    = _cell("paid")
                initial_amount = self.parse_debt_amount(initial_str)
                debit_amount   = self.parse_debt_amount(debit_str)
                paid_amount    = self.parse_debt_amount(paid_str)

                clients_data.append({
                    'client':          client_name,
                    'debt':            debt_amount,
                    'debt_str':        debt_str,
                    'silence_days':    silence_days,
                    'initial_amount':  initial_amount,  # начальный остаток периода
                    'initial_str':     initial_str,
                    'debit_amount':    debit_amount,    # отгрузка в периоде
                    'debit_str':       debit_str,
                    'paid_amount':     paid_amount,     # оплата в периоде
                    'paid_str':        paid_str,
                })
            
            logger.info(f"📊 Распарсено {len(clients_data)} клиентов из {html_path.name}")
            return clients_data
            
        except Exception as e:
            logger.error(f"Ошибка при парсинге {html_path}: {e}", exc_info=True)
            return []

    @staticmethod
    def _norm_client_name(name: str) -> str:
        return " ".join(str(name or "").lower().split())

    # BUG-5 (2026-05-16): окно Саиды скользящее (на 15-е число reset).
    # Без persistence долг с апреля показывается как "14 дн" вместо реального возраста.
    # Решение: храним собственный журнал oldest_unpaid_date в logs/debt_age_history.json
    # и используем min(saved, current) — bot помнит истинный возраст независимо от окна.
    DEBT_AGE_HISTORY_PATH = Path(__file__).resolve().parent.parent / "logs" / "debt_age_history.json"

    def apply_residual_debt_age(self, clients_data: List[Dict]) -> List[Dict]:
        """Adds Phase 5 residual debt age to short debt notifications.

        BUG-5: после получения oldest_unpaid_date от Саиды сравниваем с сохранённой
        историей. Если в истории есть более ранняя дата для этого клиента — используем
        её. Так бот не теряет реальный возраст долга при reset окна Саиды.
        """
        try:
            from collector.debt_monitor import classify_debtors, load_latest_debt_json
            classified = classify_debtors(load_latest_debt_json())
            try:
                from collector.payment_hold import sync_holds_with_debtors
                sync_holds_with_debtors(classified)
            except Exception as hold_exc:
                logger.warning("sync payment holds failed: %s", hold_exc)
        except Exception as exc:
            logger.warning("apply_residual_debt_age: fallback to silence_days: %s", exc)
            return clients_data

        by_name = {
            self._norm_client_name(c.get("name")): c
            for c in classified
            if c.get("name")
        }
        history = self._load_debt_age_history()
        from datetime import datetime as _dt, date as _date
        from zoneinfo import ZoneInfo as _ZI
        tz = _ZI(os.getenv("TZ", "Asia/Almaty"))
        today = _dt.now(tz).date()
        matched = 0
        history_updated = False
        for client in clients_data:
            client_name = client.get("client", "")
            profile = by_name.get(self._norm_client_name(client_name))
            if not profile:
                continue
            current_oldest = profile.get("oldest_unpaid_date") or ""
            # BUG-5: проверяем сохранённую историю
            effective_oldest = current_oldest
            saved_oldest = history.get(client_name, {}).get("oldest_unpaid_date")
            if saved_oldest and current_oldest:
                # Берём более раннюю дату — окно Саиды не должно "омолаживать" долг
                if saved_oldest < current_oldest:
                    effective_oldest = saved_oldest
            elif saved_oldest and not current_oldest:
                effective_oldest = saved_oldest
            # Обновляем историю если есть новая oldest_unpaid_date или она раньше
            if effective_oldest and effective_oldest != saved_oldest:
                history[client_name] = {
                    "oldest_unpaid_date": effective_oldest,
                    "last_updated": _dt.now(tz).isoformat(),
                    "last_debt": float(profile.get("debt") or 0),
                }
                history_updated = True

            # Пересчитываем age если effective_oldest отличается от current
            effective_age_days = int(
                profile.get("residual_debt_age_days", profile.get("days", client.get("silence_days", 0))) or 0
            )
            if effective_oldest and effective_oldest != current_oldest:
                try:
                    eff_date = _date.fromisoformat(effective_oldest)
                    effective_age_days = max(effective_age_days, (today - eff_date).days)
                except (ValueError, TypeError):
                    pass

            client["residual_debt_age_days"] = effective_age_days
            client["oldest_unpaid_date"] = effective_oldest
            client["debt_age_basis"] = profile.get("debt_age_basis", "")
            client["payment_silence_days"] = profile.get("payment_silence_days", client.get("silence_days", 0))
            client["active_turnover"] = bool(profile.get("active_turnover", False))
            matched += 1

        # Cleanup: убираем из истории клиентов с нулевым долгом (debt=0 в текущей выгрузке)
        for c in classified:
            cname = c.get("name", "")
            if cname in history and (float(c.get("debt") or 0) <= 0):
                history.pop(cname, None)
                history_updated = True

        if history_updated:
            self._save_debt_age_history(history)

        logger.info("apply_residual_debt_age: matched %d/%d clients", matched, len(clients_data))
        return clients_data

    @classmethod
    def _load_debt_age_history(cls) -> Dict[str, Dict[str, str]]:
        """Загружает logs/debt_age_history.json."""
        path = cls.DEBT_AGE_HISTORY_PATH
        if not path.exists():
            return {}
        try:
            import json as _json
            data = _json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning("_load_debt_age_history error: %s", exc)
            return {}

    @classmethod
    def _save_debt_age_history(cls, history: Dict[str, Dict[str, str]]) -> bool:
        """Атомарная запись logs/debt_age_history.json.

        TEST_MODE guard: при COLLECTOR_TEST_MODE=1 запись пропускается, если путь
        не замокан явно (защита от контаминации боевого файла из тестов которые
        вызывают apply_residual_debt_age без mock на DEBT_AGE_HISTORY_PATH).
        """
        path = cls.DEBT_AGE_HISTORY_PATH
        # Защита: если test mode + путь не переопределён в tempdir → не писать
        test_mode = os.getenv("COLLECTOR_TEST_MODE", "0").lower() in ("1", "true", "yes")
        if test_mode:
            # Проверяем что путь действительно мокается (не указывает на боевой logs/)
            real_default = Path(__file__).resolve().parent.parent / "logs" / "debt_age_history.json"
            if path == real_default:
                logger.warning(
                    "_save_debt_age_history: TEST_MODE без mock DEBT_AGE_HISTORY_PATH — запись пропущена"
                )
                return False
        try:
            import json as _json
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(_json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
            return True
        except Exception as exc:
            logger.warning("_save_debt_age_history error: %s", exc)
            return False

    # BUG-saida (2026-05-16): только ПОЛНАЯ подтверждённая оплата снимает молчание.
    # Частичная (confirmed_partial), pending_saida, rejected — оставляют клиента в молчании.
    # TTL: 7 дней с момента подтверждения Саидой (достаточно для разноски в 1С).
    PAYMENT_FULL_GRACE_DAYS = 7

    def apply_payment_holds(self, clients_data: List[Dict]) -> List[Dict]:
        """Marks clients confirmed by Saida as waiting for 1C posting.

        BUG-saida: payment_hold=True ставится ТОЛЬКО для confirmed_full в течение
        PAYMENT_FULL_GRACE_DAYS дней. confirmed_partial / pending_saida / rejected
        НЕ снимают молчание (по бизнес-правилу: молчание = молчание В ОПЛАТЕ,
        частичная оплата = всё ещё молчание).
        """
        full_payments = self._load_recent_full_payments()
        marked = 0
        for client in clients_data:
            name = (client.get("client") or "").strip()
            confirmed = self._lookup_full_payment(full_payments, name)
            if not confirmed:
                continue
            client["payment_hold"] = True
            client["payment_hold_status"] = "confirmed_full"
            client["payment_hold_since"] = confirmed
            marked += 1
        if marked:
            logger.info(
                "apply_payment_holds: marked %d clients as waiting for posting (confirmed_full only)",
                marked,
            )
        return clients_data

    def _load_recent_full_payments(self) -> Dict[str, str]:
        """Загружает все confirmed_full holds за последние PAYMENT_FULL_GRACE_DAYS.

        Returns: {normalized_client_name: saida_confirmed_at_iso}
        """
        try:
            from collector.payment_hold import _load as _hold_load
            from datetime import timedelta
            holds = _hold_load()
        except Exception as exc:
            logger.warning("_load_recent_full_payments: unavailable: %s", exc)
            return {}

        from datetime import datetime as _dt
        from zoneinfo import ZoneInfo as _ZI
        tz = _ZI(os.getenv("TZ", "Asia/Almaty"))
        cutoff = _dt.now(tz) - timedelta(days=self.PAYMENT_FULL_GRACE_DAYS)
        result: Dict[str, str] = {}
        for hold in (holds or {}).values():
            if not isinstance(hold, dict):
                continue
            if hold.get("status") != "confirmed_full":
                continue
            confirmed_at = hold.get("saida_confirmed_at") or hold.get("updated_at") or ""
            if not confirmed_at:
                continue
            try:
                ts = _dt.fromisoformat(confirmed_at)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=tz)
            except (ValueError, TypeError):
                continue
            if ts < cutoff:
                continue
            name = (hold.get("client") or "").strip()
            if not name:
                continue
            # Берём самую свежую дату если несколько hold по одному клиенту
            prev = result.get(name)
            if prev is None or confirmed_at > prev:
                result[name] = confirmed_at
        return result

    @staticmethod
    def _lookup_full_payment(full_payments: Dict[str, str], client_name: str) -> str:
        """Ищет confirmed_full hold по имени с учётом legacy обрезки name[:26]."""
        if not client_name:
            return ""
        if client_name in full_payments:
            return full_payments[client_name]
        # Legacy: hold мог быть записан с обрезанным именем (BUG-2 до фикса)
        short = client_name[:26]
        for hold_name, ts in full_payments.items():
            if hold_name == short or hold_name[:26] == short:
                return ts
        return ""

    @staticmethod
    def _age_days(client: Dict) -> int:
        value = client.get("residual_debt_age_days")
        if value is None or value == "":
            value = client.get("effective_days", client.get("silence_days", 0))
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _age_text(cls, client: Dict) -> str:
        text = f"остаток {cls._age_days(client)} дн"
        oldest = client.get("oldest_unpaid_date")
        if oldest:
            text += f", с {oldest}"
        return text
    
    def categorize_by_silence(self, clients_data: List[Dict],
                              historical_map: Optional[Dict[str, int]] = None,
                              weekly_clients: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """
        v1.7: Классификация клиентов по дням молчания с анализом
        ОБОИХ направлений: debit (отгрузки) и credit (оплаты).

        Категории:
          'critical'        — 30+ дней
          'alarm'           — 15-29 дней
          'silence'         — 10-14 дней (молчание)
          'overdue'         — 7-9 дней (просрочка, не рассчитался в срок)
          'partial_payment' — days < 7, debit==0, credit >= 10% долга
                              (платит значимо, но не закрыл — следим)
          'on_stop'         — debit==0, credit < 10% долга ИЛИ credit==0,
                              debt > 0, days < 7 (имитация / заморожен)
                              Если days >= 7 — уже попал в overdue/silence/
                              alarm/critical с флагом is_imitation.

        Флаг is_imitation добавляется клиентам в любой категории когда:
          debit == 0 AND credit > 0 AND credit < debt * IMITATION_THRESHOLD
          → нет отгрузок, платит что-то, но < 10% долга (сброс счётчика).
          credit == 0 — НЕ имитация: просто не платит (обычная задолженность).

        Еженедельные клиенты (weekly_clients): исключаются из 'overdue'
        (7-9 дн) — у них нормальный недельный цикл оплаты.
        При days >= SILENCE_DAYS (10+) исключение не действует.
        """
        categorized: Dict[str, List[Dict]] = {
            'critical':        [],
            'alarm':           [],
            'silence':         [],
            'overdue':         [],
            'partial_payment': [],
            'on_stop':         [],
        }

        weekly_set = set(weekly_clients or [])

        for client in clients_data:
            days   = self._age_days(client)
            debt   = client['debt']
            debit  = client.get('debit_amount', 0.0)
            credit = client.get('paid_amount', 0.0)

            if debt < self.MIN_DEBT_AMOUNT:
                continue
            if client.get("payment_hold"):
                logger.info(
                    "silence skip: [%s] оплата подтверждена Саидой, ждём разноски",
                    client.get("client", ""),
                )
                continue

            # Флаг имитации: нет отгрузок + платит, но < 10% долга (сброс счётчика).
            # credit == 0 — НЕ имитация: клиент просто не платит (обычная задолженность).
            is_imitation = (
                debit == 0
                and credit > 0
                and credit < debt * self.IMITATION_THRESHOLD
                and debt > 0
            )

            # Флаг реальной частичной оплаты: нет отгрузок, но платит >= 10%
            is_genuine_partial = (
                debit == 0
                and credit >= debt * self.IMITATION_THRESHOLD
                and debt > 0
            )

            # Исторические данные (счётчик мог быть сброшен оплатой в 1С)
            _raw_hist = (historical_map or {}).get(client['client']) if historical_map else None
            _has_residual = client.get("residual_debt_age_days") is not None
            _counter_reset = bool((not _has_residual) and _raw_hist and _raw_hist > days + 2)
            effective_days = (_raw_hist + days) if _counter_reset else days

            base = dict(
                client,
                is_imitation=is_imitation,
                historical_days=_raw_hist if _counter_reset else None,
                effective_days=effective_days,
                age_days=days,
            )

            if days >= self.CRITICAL_DAYS:
                categorized['critical'].append(base)
            elif days >= self.ALARM_DAYS:
                categorized['alarm'].append(base)
            elif days >= self.SILENCE_DAYS:
                categorized['silence'].append(base)
            elif days >= self.OVERDUE_DAYS:
                # Еженедельные клиенты пропускаем в overdue (7-9 дн)
                if client['client'] in weekly_set:
                    continue
                categorized['overdue'].append(base)
            else:
                # days < 7
                if is_genuine_partial:
                    categorized['partial_payment'].append(dict(base, partial_payment=True))
                elif is_imitation:
                    categorized['on_stop'].append(dict(base, partial_payment=True))
                # else: debit > 0, days < 7 — нормальный активный клиент, пропускаем

        return categorized
    
    def format_manager_alert(self, manager_name: str, categorized: Dict[str, List[Dict]], report_date: str = "") -> str:
        """
        v1.7: Формирует текст уведомления для менеджера.
        Категории: critical / alarm / silence / overdue / partial_payment / on_stop.
        on_stop — в конце сводки каждого менеджера.
        """
        total_silent = sum(
            len(categorized.get(k, []))
            for k in ('critical', 'alarm', 'silence', 'overdue', 'partial_payment', 'on_stop')
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

        def _imitation_suffix(c: Dict) -> str:
            return " ⚠️ имитация" if c.get('is_imitation') else ""

        def _violation_suffix(c: Dict) -> str:
            return " 🚨 нарушение отгрузки" if c.get('shipment_violation') else ""

        def _append_block(label: str, clients: List[Dict], limit: int = 10) -> float:
            if not clients:
                return 0.0
            msg_lines.append(label)
            total = 0.0
            for c in clients[:limit]:
                suffix = _imitation_suffix(c) + _violation_suffix(c)
                msg_lines.append(
                    f"  • {c['client']} — {c['debt_str']} ({self._age_text(c)}){suffix}"
                )
                total += c['debt']
            if len(clients) > limit:
                rest_debt = sum(x['debt'] for x in clients[limit:])
                msg_lines.append(
                    f"  ... и ещё {len(clients) - limit} клиент(ов) на {self.format_amount(rest_debt)} ₸"
                )
                total += rest_debt
            return total

        if categorized.get('critical'):
            debt = _append_block("🔴 КРИТИЧНО (30+ дней):", categorized['critical'])
            msg_lines.append(f"  💰 Итого: {self.format_amount(debt)} ₸")
            msg_lines.append("")

        if categorized.get('alarm'):
            debt = _append_block("🟠 ТРЕВОГА (15-29 дней):", categorized['alarm'])
            msg_lines.append(f"  💰 Итого: {self.format_amount(debt)} ₸")
            msg_lines.append("")

        if categorized.get('silence'):
            debt = _append_block("🟡 МОЛЧАНИЕ (10-14 дней):", categorized['silence'])
            msg_lines.append(f"  💰 Итого: {self.format_amount(debt)} ₸")
            msg_lines.append("")

        if categorized.get('overdue'):
            debt = _append_block("⚡ ПРОСРОЧКА (7-9 дней):", categorized['overdue'])
            msg_lines.append(f"  💰 Итого: {self.format_amount(debt)} ₸")
            msg_lines.append("")

        if categorized.get('partial_payment'):
            msg_lines.append("💛 ЧАСТИЧНАЯ ОПЛАТА (долг не закрыт):")
            total_partial = 0.0
            for c in categorized['partial_payment'][:5]:
                paid_str    = c.get('paid_str', '') or self.format_amount(c.get('paid_amount', 0))
                initial_str = c.get('initial_str', '') or self.format_amount(c.get('initial_amount', 0))
                msg_lines.append(
                    f"  • {c['client']} — "
                    f"нач: {initial_str} → оплачено: {paid_str} → остаток: {c['debt_str']} ({self._age_text(c)})"
                )
                total_partial += c['debt']
            if len(categorized['partial_payment']) > 5:
                rest = len(categorized['partial_payment']) - 5
                rest_debt = sum(x['debt'] for x in categorized['partial_payment'][5:])
                msg_lines.append(f"  ... и ещё {rest} клиент(ов) на {self.format_amount(rest_debt)} ₸")
                total_partial += rest_debt
            msg_lines.append(f"  💰 Остаток: {self.format_amount(total_partial)} ₸")
            msg_lines.append("")

        if categorized.get('on_stop'):
            msg_lines.append("🛑 СТОП / ИМИТАЦИЯ ОПЛАТЫ:")
            total_stop = 0.0
            for c in categorized['on_stop'][:5]:
                paid_str = c.get('paid_str', '') or self.format_amount(c.get('paid_amount', 0))
                credit = c.get('paid_amount', 0.0)
                pct = (credit / c['debt'] * 100) if c['debt'] > 0 else 0
                msg_lines.append(
                    f"  • {c['client']} — долг: {c['debt_str']}, "
                    f"оплата: {paid_str} ({pct:.0f}%)"
                )
                total_stop += c['debt']
            if len(categorized['on_stop']) > 5:
                rest = len(categorized['on_stop']) - 5
                rest_debt = sum(x['debt'] for x in categorized['on_stop'][5:])
                msg_lines.append(f"  ... и ещё {rest} клиент(ов) на {self.format_amount(rest_debt)} ₸")
                total_stop += rest_debt
            msg_lines.append(f"  💰 Заморожено: {self.format_amount(total_stop)} ₸")
            msg_lines.append("")

        total_debt = sum(
            c['debt']
            for k in ('critical', 'alarm', 'silence', 'overdue', 'partial_payment', 'on_stop')
            for c in categorized.get(k, [])
        )
        msg_lines.append(f"💰 Общий долг: {self.format_amount(total_debt)} ₸")
        msg_lines.append("")
        msg_lines.append("📊 Открыть детальный отчёт → /debt")

        return "\n".join(msg_lines)
    
    def format_admin_summary(self, all_managers_data: Dict[str, Dict]) -> str:
        """v1.7: Краткая сводка для админа по всем менеджерам."""
        msg_lines = [
            "⚠️ СВОДКА: ДНИ МОЛЧАНИЯ ПО ВСЕМ МЕНЕДЖЕРАМ",
            ""
        ]

        total_overall_debt = 0.0
        managers_with_issues = 0

        _CATS = ('critical', 'alarm', 'silence', 'overdue', 'partial_payment', 'on_stop')

        for manager_name, categorized in sorted(all_managers_data.items()):
            total_count = sum(len(categorized.get(k, [])) for k in _CATS)
            if total_count == 0:
                continue

            managers_with_issues += 1
            msg_lines.append(f"👨‍💼 {manager_name}:")

            for key, icon, label in [
                ('critical',        '🔴', '30+ дн'),
                ('alarm',           '🟠', '15-29 дн'),
                ('silence',         '🟡', '10-14 дн'),
                ('overdue',         '⚡', '7-9 дн'),
                ('partial_payment', '💛', 'частичная оплата'),
                ('on_stop',         '🛑', 'стоп/имитация'),
            ]:
                clients = categorized.get(key, [])
                if not clients:
                    continue
                debt = sum(c['debt'] for c in clients)
                msg_lines.append(
                    f"  {icon} {len(clients)} кл. ({label}) — {self.format_amount(debt)} ₸"
                )
                total_overall_debt += debt

            msg_lines.append("")

        if managers_with_issues == 0:
            return "✅ У всех менеджеров нет критичных дней молчания!"

        msg_lines.append(f"💰 Всего: {self.format_amount(total_overall_debt)} ₸")
        msg_lines.append(f"📊 Менеджеров с проблемами: {managers_with_issues}")

        return "\n".join(msg_lines)

    def format_admin_detailed(self, all_managers_data: Dict[str, Dict], manager_dates: Dict[str, str] = None) -> str:
        """
        v1.7: ДЕТАЛЬНАЯ сводка для админа.
        Категории: critical / alarm / silence / overdue / partial_payment / on_stop.
        on_stop — в конце раздела каждого менеджера.
        """
        msg_lines = [
            "⚠️ ДЕТАЛЬНАЯ СВОДКА: ДНИ МОЛЧАНИЯ",
            ""
        ]

        total_overall_debt = 0.0
        managers_with_issues = 0
        total_overall_clients = 0

        _CATS = ('critical', 'alarm', 'silence', 'overdue', 'partial_payment', 'on_stop')

        def _imitation_suffix(c: Dict) -> str:
            return " ⚠️ имитация" if c.get('is_imitation') else ""

        def _append_std_block(label: str, clients: List[Dict], limit: int = 10) -> float:
            msg_lines.append(label)
            total = 0.0
            for c in clients[:limit]:
                suffix = _imitation_suffix(c)
                msg_lines.append(
                    f"  • {c['client']} — {c['debt_str']} ({self._age_text(c)}){suffix}"
                )
                total += c['debt']
            if len(clients) > limit:
                rest = len(clients) - limit
                rest_debt = sum(x['debt'] for x in clients[limit:])
                msg_lines.append(
                    f"  ... и ещё {rest} клиент(ов) на {self.format_amount(rest_debt)} ₸"
                )
                total += rest_debt
            return total

        for manager_name, categorized in sorted(all_managers_data.items()):
            total_count = sum(len(categorized.get(k, [])) for k in _CATS)
            if total_count == 0:
                continue

            managers_with_issues += 1
            total_overall_clients += total_count

            report_date = (manager_dates or {}).get(manager_name, "")
            date_suffix = f" | 📅 {report_date}" if report_date else ""

            msg_lines.append(f"👨‍💼 {manager_name.upper()}{date_suffix}")
            msg_lines.append("━" * 50)

            if categorized.get('critical'):
                d = _append_std_block("🔴 КРИТИЧНО (30+ дней):", categorized['critical'], 20)
                msg_lines.append(f"  💰 Итого критично: {self.format_amount(d)} ₸")
                msg_lines.append("")
                total_overall_debt += d

            if categorized.get('alarm'):
                d = _append_std_block("🟠 ТРЕВОГА (15-29 дней):", categorized['alarm'])
                msg_lines.append(f"  💰 Итого тревога: {self.format_amount(d)} ₸")
                msg_lines.append("")
                total_overall_debt += d

            if categorized.get('silence'):
                d = _append_std_block("🟡 МОЛЧАНИЕ (10-14 дней):", categorized['silence'])
                msg_lines.append(f"  💰 Итого молчание: {self.format_amount(d)} ₸")
                msg_lines.append("")
                total_overall_debt += d

            if categorized.get('overdue'):
                d = _append_std_block("⚡ ПРОСРОЧКА (7-9 дней):", categorized['overdue'])
                msg_lines.append(f"  💰 Итого просрочка: {self.format_amount(d)} ₸")
                msg_lines.append("")
                total_overall_debt += d

            if categorized.get('partial_payment'):
                msg_lines.append("💛 ЧАСТИЧНАЯ ОПЛАТА (долг не закрыт):")
                partial_total = 0.0
                for c in categorized['partial_payment'][:10]:
                    paid_str    = c.get('paid_str', '') or self.format_amount(c.get('paid_amount', 0))
                    initial_str = c.get('initial_str', '') or self.format_amount(c.get('initial_amount', 0))
                    msg_lines.append(
                        f"  • {c['client']} — "
                        f"нач: {initial_str} → оплачено: {paid_str} → остаток: {c['debt_str']} ({self._age_text(c)})"
                    )
                    partial_total += c['debt']
                rest = categorized['partial_payment'][10:]
                if rest:
                    rest_debt = sum(x['debt'] for x in rest)
                    msg_lines.append(f"  ... и ещё {len(rest)} клиент(ов) на {self.format_amount(rest_debt)} ₸")
                    partial_total += rest_debt
                msg_lines.append(f"  💰 Остаток долга: {self.format_amount(partial_total)} ₸")
                msg_lines.append("")
                total_overall_debt += partial_total

            if categorized.get('on_stop'):
                msg_lines.append("🛑 СТОП / ИМИТАЦИЯ ОПЛАТЫ:")
                stop_total = 0.0
                for c in categorized['on_stop'][:10]:
                    paid_str = c.get('paid_str', '') or self.format_amount(c.get('paid_amount', 0))
                    credit = c.get('paid_amount', 0.0)
                    pct = (credit / c['debt'] * 100) if c['debt'] > 0 else 0
                    msg_lines.append(
                        f"  • {c['client']} — долг: {c['debt_str']}, "
                        f"оплата: {paid_str} ({pct:.0f}%)"
                    )
                    stop_total += c['debt']
                rest = categorized['on_stop'][10:]
                if rest:
                    rest_debt = sum(x['debt'] for x in rest)
                    msg_lines.append(f"  ... и ещё {len(rest)} клиент(ов) на {self.format_amount(rest_debt)} ₸")
                    stop_total += rest_debt
                msg_lines.append(f"  💰 Заморожено: {self.format_amount(stop_total)} ₸")
                msg_lines.append("")
                total_overall_debt += stop_total

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
        return (0, 0, _safe_mtime(p))

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
            _safe_mtime(p)
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

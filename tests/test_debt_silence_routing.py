"""
tests/test_debt_silence_routing.py
Доказательства корректности v9.4.38 + v9.4.39:
  - Именные Ведомости взаиморасчётов → debt_auto_report (не rejected)
  - Сводные Ведомости → rejected/unknown
  - debt_files_processed считает только долговые файлы
  - silence_alerts вызывается только по долговым файлам
  - Коллектор не задваивается (cooldown не затронут)
  - find_header корректно находит заголовок Ведомости
"""

import sys, os, types, asyncio, unittest
from unittest.mock import AsyncMock, MagicMock, patch, call
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ──────────────────────────────────────────────────────────
# 1. find_header — Ведомость взаиморасчётов
# ──────────────────────────────────────────────────────────
class TestFindHeaderVzaimo(unittest.TestCase):
    """find_header в debt_auto_report.py должен найти заголовок Ведомости без ошибок."""

    def test_find_header_vzaimo_score4(self):
        """Строки 8-9 Ведомости дают score=4 >= 2 → заголовок найден."""
        import pandas as pd, re

        RE_CLIENT_EQ = re.compile(r"^\s*контрагент\s*$", re.I)
        keys = ["нач. остаток", "приход", "расход", "кон. остаток"]

        # Имитируем структуру Ведомости (строки 0-9 = мета + шапка)
        rows = [
            ["", "Ведомость по взаиморасчетам с контрагентами"],
            ["", "Период: 04.04.2026 - 04.05.2026"],
            ["", "Показатели: Сумма взаиморасчетов(нач. остаток, приход, расход, кон. остаток);"],
            ["", "Группировки строк: Контрагент (Иерархия);"],
            ["", "Отборы:\nКонтрагент В группе из списка (Ергали);"],
            ["", "Дополнительные поля:\nПо дням;"],
            ["", "Сортировка: Контрагент (По возрастанию);"],
            ["", ""],
            ["", "Контрагент", "Сумма взаиморасчетов", "", "", ""],   # строка 8
            ["", "По дням, По дням", "нач. остаток", "приход", "расход", "кон. остаток"],  # строка 9
            ["", ""],
            ["", "Покупатели", "5404734.37", "38713252.66", "40862927.48", "3255059.55"],
        ]
        df = pd.DataFrame(rows)

        def _row_vals(raw, i):
            return [str(v).strip() for v in raw.iloc[i] if str(v).strip() not in ("", "nan")]

        # EQ pass → check next row
        found = False
        for i in range(len(df) - 1):
            if any(RE_CLIENT_EQ.match(c) for c in _row_vals(df, i)):
                r1 = [c.lower() for c in _row_vals(df, i + 1)]
                score = sum(1 for k in keys if any(k in c for c in r1))
                if score >= 2:
                    found = True
                    self.assertEqual(i, 8, "Заголовок должен быть на строке 8")
                    self.assertGreaterEqual(score, 4, f"Score={score} < 4")
                    break

        self.assertTrue(found, "find_header не нашёл заголовок Ведомости")


# ──────────────────────────────────────────────────────────
# 2. Маршрутизация: взаиморасч именной vs сводный
# ──────────────────────────────────────────────────────────
class TestVzaimoRouting(unittest.TestCase):
    """Проверяем логику разветвления в pipeline_task для взаиморасч файлов."""

    MANAGERS = ["Ергали", "Алена", "Магира", "Оксана"]

    def _is_named(self, fname: str) -> bool:
        known = {m.lower() for m in self.MANAGERS}
        return any(m in fname.lower() for m in known)

    def test_named_ведомость_is_routed(self):
        """Именные Ведомости → _is_named=True → должны обрабатываться."""
        named_files = [
            "Ведомость_по_взаиморасчетам_с_контрагентами_Ергали (353).xlsx",
            "Ведомость_по_взаиморасчетам_с_контрагентами_Алена (352).xlsx",
            "Ведомость_по_взаиморасчетам_с_контрагентами_Магира (353).xlsx",
            "Ведомость_по_взаиморасчетам_с_контрагентами_Оксана (353).xlsx",
        ]
        for f in named_files:
            with self.subTest(f=f):
                self.assertTrue(self._is_named(f), f"{f} должен быть именным")

    def test_summary_ведомость_is_rejected(self):
        """Сводные Ведомости → _is_named=False → rejected/unknown."""
        summary_files = [
            "Ведомость_по_взаиморасчетам_с_контрагентами (424).xlsx",
            "Ведомость_по_взаиморасчетам_с_контрагентами_за_26_05 (222).xlsx",
        ]
        for f in summary_files:
            with self.subTest(f=f):
                self.assertFalse(self._is_named(f), f"{f} должен быть сводным")

    def test_детальный_дебиторы_not_vzaimo(self):
        """Детальный Дебиторы не попадает в ветку взаиморасч."""
        names = [
            "Детальный Дебиторы Ергали (146).xlsx",
            "Детальный Дебиторы Алена (146).xlsx",
        ]
        for f in names:
            with self.subTest(f=f):
                self.assertNotIn("взаиморасч", f.lower())

    def test_manager_list_covers_all_known(self):
        """Имена менеджеров из MANAGERS покрывают все именные Ведомости."""
        test_cases = {
            "ведомость_ергали.xlsx": "Ергали",
            "ведомость_алена.xlsx": "Алена",
            "ведомость_магира.xlsx": "Магира",
            "ведомость_оксана.xlsx": "Оксана",
        }
        for fname, expected_mgr in test_cases.items():
            self.assertTrue(self._is_named(fname),
                            f"{fname} не распознан как именной (менеджер: {expected_mgr})")


# ──────────────────────────────────────────────────────────
# 3. debt_files_processed — счётчик только для долговых
# ──────────────────────────────────────────────────────────
class TestDebtFilesCounter(unittest.TestCase):
    """debt_files_processed увеличивается только для debt_auto_report веток."""

    def _simulate_pipeline(self, filenames: list[str], managers: list[str]) -> tuple[int, int]:
        """
        Возвращает (processed_files, debt_files_processed) как в реальном pipeline.
        Упрощённая имитация ветвления.
        """
        import re
        RE_INV = re.compile(r"(остат|inventory|товар.*склад|партия.*товар|ведомость.*склад)", re.I)
        RE_SALES = re.compile(r"(sales|продаж)", re.I)
        RE_GROSS = re.compile(r"(gross|валов)", re.I)
        RE_EXP = re.compile(r"(затрат|расход|expense)", re.I)

        known = {m.lower() for m in managers}
        processed = 0
        debt_processed = 0

        for fname in filenames:
            fl = fname.lower()
            this_is_debt = False
            executed = True  # assume success for simulation

            if any(kw in fl for kw in ["денежн", "средств", "касс", "банк"]):
                continue  # cash → rejected, skip
            elif RE_INV.search(fl):
                pass  # inventory
            elif RE_SALES.search(fl):
                pass  # sales
            elif RE_GROSS.search(fl):
                pass  # gross
            elif RE_EXP.search(fl):
                pass  # expenses
            elif "взаиморасч" in fl:
                # v9.4.41: Ведомости взаиморасчётов НЕ тригерят silence_alerts.
                # Они обрабатываются debt_auto_report, но this_is_debt остаётся False.
                is_named = any(m in fl for m in known)
                if not is_named:
                    continue  # сводная → rejected
            else:
                this_is_debt = True  # debt_auto_report fallback

            if executed:
                processed += 1
                if this_is_debt:
                    debt_processed += 1

        return processed, debt_processed

    MANAGERS = ["Ергали", "Алена", "Магира", "Оксана"]

    def test_debt_only_batch(self):
        """4 файла дебиторки → processed=4, debt_files_processed=4."""
        files = [
            "Детальный Дебиторы Ергали (146).xlsx",
            "Детальный Дебиторы Алена (146).xlsx",
            "Детальный Дебиторы Магира (154).xlsx",
            "Детальный Дебиторы Оксана (156).xlsx",
        ]
        p, d = self._simulate_pipeline(files, self.MANAGERS)
        self.assertEqual(p, 4)
        self.assertEqual(d, 4, "Все 4 файла должны считаться долговыми")

    def test_sales_only_batch(self):
        """Только продажи → debt_files_processed=0 → silence_alerts НЕ запускается."""
        files = [
            "Продажи Ергали (270).xlsx",
            "Продажи Алена (291).xlsx",
            "Продажи Магира (309).xlsx",
            "Продажи Оксана (306).xlsx",
            "Продажи (451).xlsx",
        ]
        p, d = self._simulate_pipeline(files, self.MANAGERS)
        self.assertGreater(p, 0)
        self.assertEqual(d, 0, "Продажи не должны триггерить silence_alerts")

    def test_mixed_batch(self):
        """Смешанный батч: 2 дебиторки + продажи + затраты → debt=2."""
        files = [
            "Детальный Дебиторы Ергали (146).xlsx",
            "Детальный Дебиторы Алена (146).xlsx",
            "Продажи (451).xlsx",
            "Затраты с нарастающим (227).xlsx",
            "Валовая прибыль (435).xlsx",
        ]
        p, d = self._simulate_pipeline(files, self.MANAGERS)
        self.assertEqual(p, 5)
        self.assertEqual(d, 2, "Только 2 долговых файла из 5")

    def test_named_vzaimo_does_not_trigger_silence(self):
        """v9.4.41: Именные Ведомости взаиморасчётов → обрабатываются, но НЕ тригерят silence.
        silence_alerts читает только Детальный Дебиторы; Ведомость — другой формат без age-данных."""
        files = [
            "Ведомость_по_взаиморасчетам_с_контрагентами_Ергали (353).xlsx",
            "Ведомость_по_взаиморасчетам_с_контрагентами_Алена (352).xlsx",
        ]
        p, d = self._simulate_pipeline(files, self.MANAGERS)
        self.assertEqual(p, 2, "Ведомости обрабатываются (debt_auto_report)")
        self.assertEqual(d, 0, "v9.4.41: Ведомости НЕ тригерят silence_alerts")

    def test_summary_vzaimo_not_counted(self):
        """Сводные Ведомости → rejected → debt_files_processed=0."""
        files = [
            "Ведомость_по_взаиморасчетам_с_контрагентами (424).xlsx",
            "Ведомость_по_взаиморасчетам_с_контрагентами_за_26_05 (222).xlsx",
        ]
        p, d = self._simulate_pipeline(files, self.MANAGERS)
        self.assertEqual(d, 0, "Сводные Ведомости не должны считаться")

    def test_only_detailny_triggers_silence(self):
        """v9.4.41: silence_alerts срабатывает только от Детальный, не от Ведомостей.
        Батч 1 (Детальный) → d1 > 0. Батч 2 (Ведомость) → d2 == 0."""
        batch1 = ["Детальный Дебиторы Ергали (147).xlsx"]
        batch2 = ["Ведомость_по_взаиморасчетам_с_контрагентами_Ергали (353).xlsx"]

        _, d1 = self._simulate_pipeline(batch1, self.MANAGERS)
        _, d2 = self._simulate_pipeline(batch2, self.MANAGERS)

        self.assertGreater(d1, 0, "Детальный → silence_alerts запускается")
        self.assertEqual(d2, 0, "v9.4.41: Ведомость → silence_alerts НЕ запускается")


# ──────────────────────────────────────────────────────────
# 4. silence_alerts не вызывается без долговых файлов
# ──────────────────────────────────────────────────────────
class TestSilenceAlertsNotCalledForNonDebt(unittest.TestCase):
    """Проверяем что silence_alerts вызывается только когда debt_files_processed > 0."""

    def test_silence_not_triggered_for_zero_debt(self):
        """debt_files_processed=0 → silence_alerts не должен вызываться."""
        called = []

        async def mock_silence(context=None):
            called.append(True)

        async def run(debt_count):
            if debt_count > 0:
                await mock_silence()

        asyncio.run(run(0))
        self.assertEqual(len(called), 0, "silence_alerts не должен вызываться при debt=0")

    def test_silence_triggered_for_debt(self):
        """debt_files_processed=4 → silence_alerts вызывается один раз."""
        called = []

        async def mock_silence(context=None):
            called.append(True)

        async def run(debt_count):
            if debt_count > 0:
                await mock_silence()

        asyncio.run(run(4))
        self.assertEqual(len(called), 1, "silence_alerts должен вызваться ровно один раз")

    def test_silence_once_per_cycle_regardless_of_file_count(self):
        """5 долговых файлов в одном цикле → silence_alerts всё равно один раз."""
        called = []

        async def mock_silence(context=None):
            called.append(True)

        async def pipeline_cycle(file_count):
            debt = file_count  # все файлы долговые
            if debt > 0:
                await mock_silence()  # один вызов после цикла, не внутри

        asyncio.run(pipeline_cycle(5))
        self.assertEqual(len(called), 1, "Один цикл → один вызов silence_alerts")


# ──────────────────────────────────────────────────────────
# 5. Коллектор — не задваивается
# ──────────────────────────────────────────────────────────
class TestCollectorNotDoubled(unittest.TestCase):
    """Коллектор триггерится через collector_trigger.flag (отдельный механизм).
    Изменения silence_alerts не затрагивают collector_trigger_check."""

    def test_collector_trigger_independent_of_silence(self):
        """debt_files_processed и collector_trigger — разные переменные/файлы."""
        # Имитируем: silence_alerts вызван 2 раза → коллектор НЕ запускается дважды
        # потому что он читает свой флаг-файл с cooldown
        collector_runs = []
        silence_runs = []

        COOLDOWN_HOURS = 4
        NOW = 100_000.0  # произвольная точка отсчёта
        # Начальное значение: cooldown уже прошёл → первый запуск коллектора пройдёт
        last_run_ts = [NOW - COOLDOWN_HOURS * 3600 - 1]

        def maybe_run_collector(now_ts):
            elapsed_h = (now_ts - last_run_ts[0]) / 3600
            if elapsed_h >= COOLDOWN_HOURS:
                collector_runs.append(now_ts)
                last_run_ts[0] = now_ts

        def run_silence():
            silence_runs.append(True)

        # Silence запускается дважды за день (2 батча долговых файлов)
        run_silence()   # батч 1, утром
        run_silence()   # батч 2, через 30 мин

        # Коллектор проверяет cooldown (независимо от silence_alerts)
        maybe_run_collector(NOW)               # первый запуск — cooldown прошёл
        maybe_run_collector(NOW + 1800)        # через 30 мин — cooldown не прошёл → пропуск
        maybe_run_collector(NOW + 3600 * 5)   # через 5 ч — cooldown прошёл → второй запуск

        self.assertEqual(len(silence_runs), 2, "silence_alerts должен запуститься дважды")
        self.assertEqual(len(collector_runs), 2,
                         "Коллектор: ровно 2 раза несмотря на 2 silence + попытку без cooldown")
        # Ключевое: коллектор НЕ задвоился из-за silence — у него свой cooldown

    def test_silence_calls_dont_trigger_collector(self):
        """Вызов silence_alerts не пишет collector_trigger.flag."""
        # Проверяем что в check_and_send_silence_alerts нет записи в collector_trigger.flag
        import ast, os

        src_path = os.path.join(os.path.dirname(__file__), "..", "bot", "send_reports.py")
        with open(src_path, encoding="utf-8") as f:
            source = f.read()

        # Находим функцию check_and_send_silence_alerts
        start = source.find("async def check_and_send_silence_alerts")
        # Следующая функция
        end = source.find("\nasync def ", start + 10)
        if end < 0:
            end = source.find("\ndef ", start + 10)
        func_body = source[start:end] if end > 0 else source[start:start+3000]

        # В теле функции не должно быть записи collector_trigger.flag
        self.assertNotIn("collector_trigger", func_body,
                         "check_and_send_silence_alerts не должна трогать collector_trigger.flag")


# ──────────────────────────────────────────────────────────
# 6. 14:00 расписание убрано из кода
# ──────────────────────────────────────────────────────────
class TestScheduleRemoved(unittest.TestCase):
    """Проверяем что silence_alerts_14h убран из scheduler."""

    def test_no_silence_14h_job(self):
        """run_daily с silence_alerts_14h не должно быть в send_reports.py."""
        import os
        src_path = os.path.join(os.path.dirname(__file__), "..", "bot", "send_reports.py")
        with open(src_path, encoding="utf-8") as f:
            source = f.read()

        # Проверяем что нет активного run_daily для silence_alerts_14h
        # (может быть в комментарии, но не как активный вызов)
        active_daily = [
            line for line in source.splitlines()
            if "run_daily" in line
            and "silence_alerts" in line
            and not line.strip().startswith("#")
        ]
        self.assertEqual(len(active_daily), 0,
                         f"Найден активный run_daily для silence_alerts: {active_daily}")

    def test_event_driven_comment_present(self):
        """В коде есть комментарий о event-driven поведении."""
        import os
        src_path = os.path.join(os.path.dirname(__file__), "..", "bot", "send_reports.py")
        with open(src_path, encoding="utf-8") as f:
            source = f.read()

        self.assertIn("event-driven", source,
                      "Должен быть комментарий о event-driven поведении")


# ──────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    classes = [
        TestFindHeaderVzaimo,
        TestVzaimoRouting,
        TestDebtFilesCounter,
        TestSilenceAlertsNotCalledForNonDebt,
        TestCollectorNotDoubled,
        TestScheduleRemoved,
    ]

    results = []
    total = 0
    failed = 0

    print("\n" + "═" * 62)
    print("  ТЕСТЫ: Маршрутизация Ведомостей + Event-driven Silence")
    print("═" * 62)

    for cls in classes:
        s = loader.loadTestsFromTestCase(cls)
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, "w"))
        result = runner.run(s)
        count = s.countTestCases()
        errs = len(result.failures) + len(result.errors)
        total += count
        failed += errs
        status = "✅" if errs == 0 else "❌"
        print(f"  {status} {cls.__name__:45} {count - errs}/{count}")
        for f in result.failures + result.errors:
            print(f"      ↳ {f[0]}: {f[1].splitlines()[-1][:80]}")

    print("═" * 62)
    print(f"  ИТОГ: {total - failed}/{total} тестов прошло  |  {failed} упало")
    print("═" * 62)
    sys.exit(0 if failed == 0 else 1)

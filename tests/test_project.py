#!/usr/bin/env python
# coding: utf-8
"""
tests/test_project.py — комплексный тест проекта GPT1C_Processor
Запуск: python tests/test_project.py
Не требует pytest, не требует .env, не требует Telegram.
"""
import sys
import os
import json
import tempfile
import traceback
from pathlib import Path

# Добавляем корень проекта в путь
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bot"))

PASS = "✅"
FAIL = "❌"
results = []

def check(name: str, ok: bool, detail: str = ""):
    icon = PASS if ok else FAIL
    msg = f"  {icon} {name}"
    if detail and not ok:      # detail только при падении
        msg += f"\n       > {detail}"
    print(msg)
    results.append((name, ok))

def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

# ═══════════════════════════════════════════════════════════════
# 1. ИМПОРТЫ — все модули должны загружаться без исключений
# ═══════════════════════════════════════════════════════════════
section("1. Импорты модулей")

def try_import(mod_name, from_path=None):
    try:
        if from_path:
            sys.path.insert(0, str(from_path))
        __import__(mod_name)
        check(f"import {mod_name}", True)
        return True
    except Exception as e:
        check(f"import {mod_name}", False, str(e)[:80])
        return False

try_import("utils_common")
try_import("utils_excel")
try_import("cleanup_cache")
try_import("expenses_report")   # должен работать без DeprecationWarning
try_import("expenses_parser")
try_import("send_tg")

# bot-модули требуют sys.path на ROOT
try_import("silence_alerts")
try_import("sales_summary")
try_import("gross_summary")
try_import("inventory_summary")
try_import("user_tracker")

# ═══════════════════════════════════════════════════════════════
# 2. utils_common
# ═══════════════════════════════════════════════════════════════
section("2. utils_common — slugify_safe, normalize_client_name")

from utils_common import slugify_safe, normalize_client_name

cases_slug = [
    # Кириллица сохраняется (транслитерация не выполняется)
    ("Ергали",            "ергали"),
    ("ТОО Рога и Копыта", "тоо_рога_и_копыта"),
    ("",                  "row"),   # fallback для пустой строки
    ("  пробелы  ",       "пробелы"),
    ("123abc",            "123abc"),
    ("Валовая / Прибыль", "валовая_прибыль"),
]
for inp, expected in cases_slug:
    got = slugify_safe(inp)
    check(f"slugify_safe({inp!r})", got == expected, f"got={got!r} expected={expected!r}")

cases_norm = [
    ("  ТОО Рога и Копыта  ", "тоо рога и копыта"),
    ("ИП АЛИНА",              "ип алина"),
    ("",                      ""),
]
for inp, expected in cases_norm:
    got = normalize_client_name(inp)
    check(f"normalize_client_name({inp!r})", got == expected, f"got={got!r}")

# ═══════════════════════════════════════════════════════════════
# 3. send_tg — _chunk_text, send_text parse_mode fix
# ═══════════════════════════════════════════════════════════════
section("3. send_tg — _chunk_text, parse_mode fix")

import send_tg

# _chunk_text: короткий текст не разбивается
t = "Привет"
chunks = send_tg._chunk_text(t, 100)
check("_chunk_text — короткий текст = 1 кусок", len(chunks) == 1)

# _chunk_text: длинный текст разбивается
long_text = ("Слово " * 1000).strip()
chunks = send_tg._chunk_text(long_text, 100)
check("_chunk_text — длинный текст разбивается", len(chunks) > 1)
# восстановление текста
rejoined = "".join(chunks)
check("_chunk_text — текст восстанавливается без потерь",
      rejoined.replace(" ", "") == long_text.replace(" ", ""))

# Проверяем что send_text с parse_html=False НЕ включает parse_mode в data
# (тестируем через monkey-patch requests)
import unittest.mock as mock
captured_data = {}
def fake_post(url, data=None, **kwargs):
    captured_data.update(data or {})
    class FakeResp:
        def raise_for_status(self): pass
    return FakeResp()

# Подменяем токен и chat_id чтобы _assert_ready() не упал
orig_token = send_tg.TG_BOT_TOKEN
orig_chat  = send_tg.ADMIN_CHAT_ID
send_tg.TG_BOT_TOKEN  = "fake_token"
send_tg.ADMIN_CHAT_ID = "12345"

with mock.patch.object(send_tg.requests, "post", side_effect=fake_post):
    send_tg.send_text("тест без HTML", parse_html=False)
check("send_text(parse_html=False) — parse_mode НЕ в data",
      "parse_mode" not in captured_data,
      f"keys={list(captured_data.keys())}")

captured_data.clear()
with mock.patch.object(send_tg.requests, "post", side_effect=fake_post):
    send_tg.send_text("тест с HTML", parse_html=True)
check("send_text(parse_html=True) — parse_mode='HTML' в data",
      captured_data.get("parse_mode") == "HTML")

send_tg.TG_BOT_TOKEN  = orig_token
send_tg.ADMIN_CHAT_ID = orig_chat

# ═══════════════════════════════════════════════════════════════
# 4. gender_emoji (bot/send_reports)
# ═══════════════════════════════════════════════════════════════
section("4. gender_emoji — эвристика по окончанию имени")

# Импортируем функцию напрямую из файла без запуска всего бота
import importlib.util
spec = importlib.util.spec_from_file_location(
    "send_reports_partial",
    ROOT / "bot" / "send_reports.py"
)
# Не загружаем полностью (слишком тяжело), ищем функцию через grep
import re as _re
src = (ROOT / "bot" / "send_reports.py").read_text(encoding="utf-8")
# Извлекаем только функцию gender_emoji
m = _re.search(r"(def gender_emoji\(.*?\n(?:    .+\n)+)", src)
if m:
    exec(m.group(1))
    cases_gender = [
        ("Алена",   "👩"),
        ("Оксана",  "👩"),
        ("Магира",  "👩"),
        ("Ергали",  "👨"),
        ("Надежда", "👩"),
        ("Игорь",   "👤"),  # ends in р — fallback
        ("",        "👤"),
    ]
    for name, expected in cases_gender:
        got = gender_emoji(name)  # noqa
        check(f"gender_emoji({name!r})", got == expected, f"got={got!r}")
else:
    check("gender_emoji — нашли функцию в send_reports.py", False, "функция не найдена")

# ═══════════════════════════════════════════════════════════════
# 5. silence_alerts — categorize_by_silence, format_manager_alert
# ═══════════════════════════════════════════════════════════════
section("5. silence_alerts — categorize, format (расшифровка скобок)")

from silence_alerts import SilenceAlert

alert = SilenceAlert()

# Тестовые данные
clients = [
    # критичный молчун
    {"client": "ТОО Альфа",   "debt": 500_000, "debt_str": "500 000,00", "silence_days": 35,
     "debit_amount": 0, "paid_amount": 0},
    # тревога
    {"client": "ИП Бета",     "debt": 200_000, "debt_str": "200 000,00", "silence_days": 20,
     "debit_amount": 0, "paid_amount": 50_000},
    # внимание
    {"client": "ИП Гамма",    "debt": 150_000, "debt_str": "150 000,00", "silence_days": 10,
     "debit_amount": 0, "paid_amount": 0},
    # частичная оплата (days < 7, нет товара, платит, крупный долг)
    {"client": "ТОО Дельта",  "debt": 300_000, "debt_str": "300 000,00", "silence_days": 0,
     "debit_amount": 0, "paid_amount": 100_000},
    # мелкий долг — должен быть пропущен
    {"client": "Мелкий",      "debt": 5_000,   "debt_str": "5 000,00",   "silence_days": 60,
     "debit_amount": 0, "paid_amount": 0},
]

cat = alert.categorize_by_silence(clients)

check("critical содержит ТОО Альфа",
      any(c["client"] == "ТОО Альфа" for c in cat["critical"]))
check("alarm содержит ИП Бета",
      any(c["client"] == "ИП Бета" for c in cat["alarm"]))
check("warning содержит ИП Гамма",
      any(c["client"] == "ИП Гамма" for c in cat["warning"]))
check("partial_payment содержит ТОО Дельта",
      any(c["client"] == "ТОО Дельта" for c in cat["partial_payment"]))
check("Мелкий долг пропущен",
      not any(c["client"] == "Мелкий" for c in cat.get("critical", []) +
              cat.get("alarm", []) + cat.get("warning", []) + cat.get("partial_payment", [])))

# Проверяем формат строки с историческими днями
clients_pp = [
    {"client": "ТОО Тест",  "debt": 200_000, "debt_str": "200 000,00", "silence_days": 2,
     "debit_amount": 0, "paid_amount": 100_000, "paid_str": "100 000,00",
     "partial_payment": True, "historical_days": 19},
    {"client": "ИП Тест2",  "debt": 150_000, "debt_str": "150 000,00", "silence_days": 0,
     "debit_amount": 0, "paid_amount": 50_000, "paid_str": "50 000,00",
     "partial_payment": True, "historical_days": None},
]
cat_pp = {"critical": [], "alarm": [], "warning": [], "partial_payment": clients_pp}
msg = alert.format_manager_alert("Ергали", cat_pp, report_date="01.03.2026 - 13.03.2026")

check("format — 'было 19 дн молчания' присутствует", "было 19 дн молчания" in msg,
      f"фрагмент не найден в:\n{msg[:300]}")
check("format — 'сейчас 2 дн' присутствует", "сейчас 2 дн" in msg)
check("format — 'сейчас 0 дн' присутствует для клиента без hist", "сейчас 0 дн" in msg)
check("format — старый формат '(19 дн)' НЕ присутствует",
      "(19 дн)" not in msg, "старый нечитабельный формат всё ещё есть!")

# ═══════════════════════════════════════════════════════════════
# 6. _mark_notified_today / _should_notify_manager_today
# ═══════════════════════════════════════════════════════════════
section("6. _mark_notified_today — запись только после успешной отправки")

# Тестируем через прямое чтение и exec нужных функций
import json as _json
import tempfile as _tempfile

# Найдём нужные функции в send_reports.py
src = (ROOT / "bot" / "send_reports.py").read_text(encoding="utf-8")

# Вытаскиваем SALES_NOTIFY_DECADE_PATH и обе функции
with _tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tf:
    tf.write("{}")
    test_state_path = Path(tf.name)

# Патчим путь к файлу состояния
SALES_NOTIFY_DECADE_PATH = test_state_path  # noqa

# Готовим минимальное окружение для выполнения функций
env = {
    "SALES_NOTIFY_DECADE_PATH": test_state_path,
    "json": _json,
    "Path": Path,
    "logger": __import__("logging").getLogger("test"),
}

# Извлекаем _mark_notified_today
m_mark = _re.search(r"(def _mark_notified_today\(.*?)\ndef _should_notify", src, _re.DOTALL)
m_check = _re.search(r"(def _should_notify_manager_today\(.*?)\n\n\n", src, _re.DOTALL)

if m_mark and m_check:
    exec(m_mark.group(1), env)
    exec(m_check.group(1), env)

    _mark  = env["_mark_notified_today"]
    _check = env["_should_notify_manager_today"]

    period = "01.03.2026 - 13.03.2026"

    # Изначально функция должна вернуть True (не отправляли)
    ok1 = _check("Ергали", period)
    check("_should_notify — первый вызов возвращает True", ok1 is True)

    # Без _mark — повторный вызов всё ещё True
    ok2 = _check("Ергали", period)
    check("_should_notify — без _mark повторный вызов тоже True (не записали)", ok2 is True)

    # После _mark — вызов вернёт False
    _mark("Ергали", period)
    ok3 = _check("Ергали", period)
    check("_should_notify — после _mark возвращает False", ok3 is False)

    # Другой менеджер — всё ещё True
    ok4 = _check("Алена", period)
    check("_should_notify — другой менеджер не заблокирован", ok4 is True)
else:
    check("_mark_notified_today и _should_notify_manager_today найдены", False, "не найдены в коде")

test_state_path.unlink(missing_ok=True)

# ═══════════════════════════════════════════════════════════════
# 7. expenses_report — НЕТ DeprecationWarning при импорте
# ═══════════════════════════════════════════════════════════════
section("7. expenses_report — нет DeprecationWarning")

import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    import importlib
    # reimport чтобы сработало
    if "expenses_report" in sys.modules:
        del sys.modules["expenses_report"]
    importlib.import_module("expenses_report")
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)
                    and "expenses_report" in str(x.message).lower()]
    check("expenses_report — нет DeprecationWarning при импорте", len(dep_warnings) == 0,
          f"найдено {len(dep_warnings)} предупреждений")

# ═══════════════════════════════════════════════════════════════
# 8. cleanup_cache — _mtime безопасен для несуществующего файла
# ═══════════════════════════════════════════════════════════════
section("8. cleanup_cache — _mtime defensive")

import cleanup_cache
result_mtime = cleanup_cache._mtime(Path("/nonexistent/file/path.html"))
check("_mtime(несуществующий файл) = 0.0", result_mtime == 0.0, f"got={result_mtime}")

# ═══════════════════════════════════════════════════════════════
# 9. config.py — загружается без исключений
# ═══════════════════════════════════════════════════════════════
section("9. config.py — базовая загрузка")

try:
    import config
    check("config импортируется", True)
    check("config.TZ определён", hasattr(config, "TZ") and config.TZ is not None)
    check("config.ROOT существует", hasattr(config, "ROOT") and Path(config.ROOT).exists())
except Exception as e:
    check("config импортируется", False, str(e)[:80])

# ═══════════════════════════════════════════════════════════════
# 10. user_tracker — атомарная запись JSON
# ═══════════════════════════════════════════════════════════════
section("10. user_tracker — атомарная запись")

import user_tracker as ut

# user_tracker использует ANALYTICS_FILE из модуля — подменяем через монки-патч
orig_file = ut.ANALYTICS_FILE
with _tempfile.TemporaryDirectory() as td:
    test_path = Path(td) / "user_analytics.json"
    ut.ANALYTICS_FILE = test_path

    try:
        # track_user создаёт нового пользователя
        ut.track_user(99999, "ТестЮзер", "test_user")
        check("user_tracker.track_user — не падает", True)
        check("user_tracker — файл создался после track_user", test_path.exists())

        # track_action
        ut.track_action(99999, "debt", "test_detail")
        check("user_tracker.track_action — не падает", True)

        # _load_analytics — данные валидны
        data = ut._load_analytics()
        check("user_tracker._load_analytics — возвращает dict", isinstance(data, dict))
        check("user_tracker — users содержит тестового юзера",
              "99999" in data.get("users", {}))
        check("user_tracker — actions непустые", len(data.get("actions", [])) > 0)

        # _save_analytics атомарен — пишем и перечитываем
        data["_test_key"] = "ok"
        ut._save_analytics(data)
        data2 = ut._load_analytics()
        check("user_tracker._save_analytics — данные сохраняются",
              data2.get("_test_key") == "ok")

        # get_stats работает
        stats = ut.get_stats()
        check("user_tracker.get_stats — total_users >= 1",
              stats.get("total_users", 0) >= 1)

    except Exception as e:
        check("user_tracker — тест", False, str(e)[:80])
    finally:
        ut.ANALYTICS_FILE = orig_file

# ═══════════════════════════════════════════════════════════════
# 11. silence_alerts — parse_debt_amount
# ═══════════════════════════════════════════════════════════════
section("11. SilenceAlert.parse_debt_amount — различные форматы")

cases_debt = [
    ("3 708 816,58",    3708816.58),
    ("1 073 616,00",    1073616.00),
    ("0",               0.0),
    ("",                0.0),
    ("100₸",            100.0),
    ("1 234 567,89 ₸",  1234567.89),
]
for s, expected in cases_debt:
    got = SilenceAlert.parse_debt_amount(s)
    check(f"parse_debt_amount({s!r})",
          abs(got - expected) < 0.01, f"got={got} expected={expected}")

# ═══════════════════════════════════════════════════════════════
# ИТОГ
# ═══════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
total   = len(results)
passed  = sum(1 for _, ok in results if ok)
failed  = total - passed
print(f"  ИТОГ: {passed}/{total} тестов прошло  |  {failed} упало")
print(f"{'═'*60}")

if failed:
    print("\nУпавшие тесты:")
    for name, ok in results:
        if not ok:
            print(f"  {FAIL} {name}")
    sys.exit(1)
else:
    print(f"\n  {PASS} Все тесты прошли успешно!")
    sys.exit(0)

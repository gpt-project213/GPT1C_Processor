#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collections/collection_agent.py
AI-диалоговый агент взыскания долгов через DeepSeek.

Версия: 1.0.7 (2026-04-13)

Функции:
  generate_message()  — генерирует текст сообщения должнику
  analyze_response()  — анализирует ответ должника, определяет намерение

Тон по уровням:
  1: вежливо, партнёрский тон, без давления
  2: нейтрально, показать что срок прошёл
  3: настойчиво, чёткий запрос даты
  4: строго, упомянуть возможные последствия
  5: жёстко, последнее предупреждение перед юристом

OpenClaw:
  Если OPENCLAW_ENABLED=true — регистрировать skill через WebSocket Gateway
  Если false — standalone режим (callback от Green API / Telegram)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env",
            encoding="utf-8-sig", override=False)

TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))

DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL    = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

OPENCLAW_ENABLED  = os.getenv("OPENCLAW_ENABLED", "false").lower() == "true"
OPENCLAW_GATEWAY  = os.getenv("OPENCLAW_GATEWAY", "ws://127.0.0.1:18789")
COMPANY_NAME      = os.getenv("COMPANY_NAME", "Минбаракат")

logger = logging.getLogger(__name__)

_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "config" / "collector_prompts.json"

# Встроенные defaults — используются если файл недоступен
_TONE_DEFAULTS = {
    1: "Вежливый партнёрский тон. Мягкое напоминание, без давления. Выражай уважение и готовность помочь. Попроси уточнить дату следующего платежа.",
    2: "Нейтральный деловой тон. Укажи что срок оплаты прошёл. Попроси сообщить причину задержки и конкретную дату оплаты.",
    3: "Настойчивый деловой тон. Чётко укажи на серьёзность просрочки. Прямо запроси конкретную дату и сумму оплаты. Без угроз.",
    4: "Строгий официальный тон. Укажи что ситуация требует срочного решения. Напомни что отгрузки приостановлены до закрытия задолженности. Попроси подтвердить дату оплаты. Без угроз судом или юристами.",
    5: "Строгий официальный тон. Укажи что остаток задолженности давно не закрыт и отгрузки ограничены до оплаты. Попроси связаться с менеджером или сообщить дату оплаты. Без угроз судом или юристами — только деловая позиция.",
}

_LANG_INSTRUCTION = {
    "ru": "Пиши ТОЛЬКО на русском языке.",
    "kz": "Тек қазақ тілінде жаз. (Пиши ТОЛЬКО на казахском языке.)",
}


def load_prompts() -> dict:
    """Загружает промты из config/collector_prompts.json.

    При ошибке чтения или отсутствии файла возвращает пустой dict —
    generate_message() автоматически применит встроенные defaults.
    Функция публична для использования в тестах.
    """
    try:
        with open(_PROMPTS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        logger.debug("Промты коллектора загружены из %s", _PROMPTS_PATH)
        return data
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Не удалось загрузить collector_prompts.json: %s — используются defaults", e)
        return {}


_PROMPTS: dict = load_prompts()


def _get_tone(level: int) -> str:
    tones = _PROMPTS.get("tones", {})
    return tones.get(str(level)) or _TONE_DEFAULTS.get(level, _TONE_DEFAULTS[1])


def _get_lang_inst(language: str) -> str:
    lang = _PROMPTS.get("lang_instructions", {})
    return lang.get(language) or _LANG_INSTRUCTION.get(language, _LANG_INSTRUCTION["ru"])


def _get_fallback_template(msg_type: str, **kwargs) -> str:
    """Возвращает fallback-шаблон (из JSON или встроенный) с подставленными переменными.

    kwargs: client_name, manager_name, amount, days, company — для подстановки в шаблон.
    """
    ftpl = _PROMPTS.get("fallback_templates", {})
    key = msg_type if msg_type in ftpl else "strict_reminder"
    if key in ftpl:
        template = ftpl[key]
    else:
        # Аварийный fallback если файл отсутствует
        template = _FALLBACK_TEMPLATES_DEFAULT.get(key, _FALLBACK_TEMPLATES_DEFAULT["strict_reminder"])
    if not kwargs:
        return template
    # Форматируем сумму с пробелами если передана
    if "amount" in kwargs:
        try:
            kwargs = {**kwargs, "amount": f"{int(kwargs['amount']):,}".replace(",", " ")}
        except (ValueError, TypeError):
            pass
    kwargs = {
        **kwargs,
        "report_date_part": _report_date_part(kwargs.get("report_date", "")),
        "report_date": _format_report_date(kwargs.get("report_date", "")),
        "days_text": _format_days_text(kwargs.get("days", 0)),
    }
    try:
        return template.format(**kwargs)
    except KeyError:
        return template


def _format_report_date(raw: str) -> str:
    if not raw:
        return ""
    s = str(raw).strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(s[:10], fmt).strftime("%d.%m.%Y")
        except ValueError:
            pass
    return s


def _report_date_part(raw: str) -> str:
    formatted = _format_report_date(raw)
    return f" на {formatted}" if formatted else ""


def _format_days_text(days: int) -> str:
    try:
        n = abs(int(days))
    except (TypeError, ValueError):
        return f"{days} дней"
    if 11 <= n % 100 <= 14:
        word = "дней"
    elif n % 10 == 1:
        word = "день"
    elif 2 <= n % 10 <= 4:
        word = "дня"
    else:
        word = "дней"
    return f"{days} {word}"


def _call_deepseek(system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
    """Вызывает DeepSeek API синхронно. Возвращает текст ответа или пустую строку при ошибке."""
    if not DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY не задан")
        return ""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    try:
        resp = httpx.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("choices"):
            logger.warning("DeepSeek вернул пустой choices: %s", str(data)[:200])
            return ""
        return data["choices"][0]["message"]["content"].strip()
    except (httpx.HTTPError, KeyError, json.JSONDecodeError) as e:
        logger.error("DeepSeek API ошибка: %s", e)
        return ""


_FALLBACK_TEMPLATES_DEFAULT = {
    "soft_reminder": (
        "Здравствуйте! Это {company}, отдел по работе с клиентами.\n\n"
        "Пишем по имеющейся задолженности: {client_name}.\n"
        "Ответственный менеджер: {manager_name}.\n\n"
        "По нашим данным, остаток задолженности{report_date_part} составляет {amount} тг. "
        "Остаток не закрыт уже {days_text}.\n"
        "Подскажите, пожалуйста, когда планируете следующий платёж?\n\n"
        "Если удобнее обсудить с менеджером — напишите 1, передадим {manager_name}."
    ),
    "payment_plan_control": (
        "Здравствуйте! Это {company}, отдел по работе с клиентами.\n\n"
        "Пишем по имеющейся задолженности: {client_name}.\n"
        "Ответственный менеджер: {manager_name}.\n\n"
        "Видим, что оплаты поступают, но остаток задолженности{report_date_part} составляет {amount} тг. "
        "Остаток не закрыт уже {days_text}.\n"
        "Подскажите, по какому графику планируете закрыть остаток?\n\n"
        "Если удобнее — напишите 1, передадим {manager_name}."
    ),
    "strict_reminder": (
        "Здравствуйте! Это {company}, отдел по работе с клиентами.\n\n"
        "Пишем по имеющейся задолженности: {client_name}.\n"
        "Ответственный менеджер: {manager_name}.\n\n"
        "По нашим данным, остаток задолженности{report_date_part} составляет {amount} тг. "
        "Остаток не закрыт уже {days_text}.\n"
        "Пожалуйста, сообщите, когда сможете оплатить остаток.\n\n"
        "Если удобнее обсудить детали — напишите 1, передадим {manager_name}."
    ),
    "stoplist_reminder": (
        "Здравствуйте! Это {company}, отдел по работе с клиентами.\n\n"
        "Пишем по имеющейся задолженности: {client_name}.\n"
        "Ответственный менеджер: {manager_name}.\n\n"
        "По нашим данным, остаток задолженности{report_date_part} составляет {amount} тг. "
        "Остаток не закрыт уже {days_text}, поэтому дальнейшие отгрузки ограничены до его закрытия.\n"
        "Пожалуйста, сообщите, когда сможете оплатить остаток.\n\n"
        "Если удобнее обсудить с менеджером — напишите 1, передадим {manager_name}."
    ),
}


def generate_message(
    client_name: str,
    debt_amount: float,
    days_overdue: int,
    level: int,
    language: str = "ru",
    previous_promise: Optional[str] = None,
    manager_name: str = "",
    msg_type: str = "",
    report_date: str = "",
) -> str:
    """Генерирует персонализированный текст первого сообщения должнику через DeepSeek.

    Args:
        client_name:      Название клиента / торговой точки
        debt_amount:      Сумма долга в тенге
        days_overdue:     Дней просрочки
        level:            Уровень давления 1–5
        language:         "ru" или "kz"
        previous_promise: Дата предыдущего обещания если было (ISO строка)
        manager_name:     Имя менеджера
        msg_type:         Тип сообщения: soft_reminder / payment_plan_control /
                          strict_reminder / stoplist_reminder
        report_date:      Дата отчёта 1С, по которому рассчитан остаток.

    Returns:
        Текст сообщения. Завершается вопросом о дате оплаты и фразой про менеджера.
    """
    tone      = _get_tone(level)
    lang_inst = _get_lang_inst(language)
    amount_str = f"{debt_amount:,.0f}".replace(",", " ")
    mgr     = manager_name or "вашим менеджером"
    company = COMPANY_NAME

    if msg_type:
        return _get_fallback_template(msg_type,
            company=company, client_name=client_name,
            manager_name=mgr, amount=amount_str, days=days_overdue,
            report_date=report_date,
        )

    promise_note = ""
    if previous_promise:
        pn_tpl = _PROMPTS.get("promise_note", "")
        if pn_tpl:
            promise_note = "\n" + pn_tpl.format(date=previous_promise).lstrip("\n")
        else:
            promise_note = (
                f"\nВАЖНО: Клиент ранее обещал оплатить {previous_promise}, "
                f"но обещание не было выполнено. Упомяни это деликатно."
            )

    sys_tpl  = _PROMPTS.get("system_prompt", "")
    user_tpl = _PROMPTS.get("user_prompt", "")

    if sys_tpl and user_tpl:
        system_prompt = sys_tpl.format(
            company=company, client_name=client_name,
            manager_name=mgr, lang_inst=lang_inst,
        )
        user_prompt = user_tpl.format(
            client_name=client_name, amount=amount_str,
            days=days_overdue, tone=tone, promise_note=promise_note,
        )
    else:
        # Аварийный fallback если файл не загружен
        system_prompt = (
            f"Ты — сотрудник отдела по работе с клиентами компании {company} "
            f"(оптовые поставки продуктов питания, Алматы, Казахстан). "
            f"Пишешь от имени компании, НЕ от имени ИИ или бота. "
            f"Первая строка: «Здравствуйте! Это {company}, отдел по работе с клиентами.» "
            f"НЕ используй слова «ИИ», «бот», «робот». {lang_inst}"
        )
        user_prompt = (
            f"Напиши сообщение клиенту:\nКлиент: {client_name}\n"
            f"Долг: {amount_str} тенге\nПросрочка: {days_overdue} дней\n"
            f"Тон: {tone}{promise_note}"
        )

    result = _call_deepseek(system_prompt, user_prompt, max_tokens=400)
    if not result:
        result = _get_fallback_template(msg_type).format(
            company=company, client_name=client_name,
            manager_name=mgr, amount=amount_str, days=days_overdue,
        )
    return result


def analyze_response(
    response_text: str,
    manager_name: str = "",
    conversation_history: Optional[list] = None,
) -> dict:
    """Анализирует ответ должника через DeepSeek.

    Args:
        response_text:         Текст ответа клиента.
        manager_name:          Имя менеджера (для контекста промпта).
        conversation_history:  Список предыдущих обменов (последние 4 включаются в промпт).

    Возвращает:
        intent:           promise | refusal | delay_request | question | unclear
        promise_date:     ISO дата если есть обещание, иначе None
        promise_amount:   сумма обещания если указана, иначе None
        requires_human:   True если нужно подключить живого человека
        suggested_reply:  рекомендуемый ответ агента
    """
    if not response_text.strip():
        return {
            "intent": "unclear",
            "promise_date": None,
            "promise_amount": None,
            "requires_human": False,
            "suggested_reply": "Уточните, пожалуйста, Ваше решение по оплате.",
        }

    company_ctx = f"для компании {COMPANY_NAME}"
    mgr_ctx = f" (менеджер: {manager_name})" if manager_name else ""

    today = datetime.now(TZ).strftime("%Y-%m-%d")
    today_display = datetime.now(TZ).strftime("%d.%m.%Y")
    system_prompt = (
        f"Ты — аналитик сообщений должников {company_ctx}{mgr_ctx}. "
        f"Сегодняшняя дата: {today} ({today_display}). "
        "Твоя задача: проанализировать ответ клиента и вернуть ТОЛЬКО JSON без пояснений. "
        "Формат ответа строго:\n"
        '{"intent":"...", "promise_date":"...", "promise_amount":..., '
        '"requires_human":..., "suggested_reply":"..."}\n'
        "intent: одно из [promise, promise_without_date, refusal, delay_request, question, identity_question, unclear]\n"
        "  - promise: клиент обещает оплатить — с конкретной датой ИЛИ с относительной "
        "('завтра', 'послезавтра', 'до пятницы', 'в пятницу', 'через 2 дня', 'на этой неделе', '17 числа' и т.д.).\n"
        "    Для относительных дат ВЫЧИСЛИ конкретную дату YYYY-MM-DD от сегодня:\n"
        "    завтра=+1д, послезавтра=+2д, через N дней=+Nд, 'до пятницы'/'в пятницу'=ближайшая пятница,\n"
        "    'на этой неделе'=пятница текущей недели, 'до конца недели'=пятница текущей недели,\n"
        "    'в следующий понедельник'=+7д от ближайшего понедельника, 'до X числа'=X число текущего месяца.\n"
        "  - promise_without_date: клиент готов платить, но говорит 'скоро', 'не знаю когда', "
        "'постараюсь' без любой временной привязки\n"
        "  - delay_request: клиент просит отсрочку, называет причину почему не может сейчас\n"
        "  - identity_question: клиент спрашивает кто пишет, откуда номер, кто вы такие\n"
        "promise_date: YYYY-MM-DD — ОБЯЗАТЕЛЬНО вычисли если клиент назвал относительную дату. "
        "null только если дата вообще не упоминается\n"
        "promise_amount: число или null\n"
        "requires_human: true если агрессия, юридические угрозы или неоднозначность\n"
        "suggested_reply при promise (с датой): подтверди дату — 'Принято, фиксируем оплату до ДД.ММ.ГГГГ. "
        "Как оплатите — пришлите чек.'\n"
        "Если клиент говорит, что уже оплатил, QR/куар/киар уже прошёл, деньги скоро упадут "
        "или оплата ещё не разнесена: это promise_without_date, requires_human=false. "
        "В suggested_reply обязательно укажи, что задолженность взята по данным отчёта 1С "
        "на дату сообщения бота/отчёта, и попроси точную дату и сумму оплаты.\n"
        "Не обсуждай темы не связанные с задолженностью. "
        "Если клиент уходит от темы — это off_topic, set requires_human=true "
        "после второго off_topic."
    )

    # Включаем последние 4 обмена как контекст
    history_block = ""
    if conversation_history:
        last_exchanges = conversation_history[-4:]
        lines = []
        for ex in last_exchanges:
            role = ex.get("role", "")
            txt = ex.get("text", "")
            if role == "bot":
                lines.append(f"Бот: {txt[:120]}")
            elif role == "client":
                lines.append(f"Клиент: {txt[:120]}")
        if lines:
            history_block = "История переписки (последние обмены):\n" + "\n".join(lines) + "\n\n"

    user_prompt = f"{history_block}Ответ клиента:\n{response_text}"

    raw = _call_deepseek(system_prompt, user_prompt, max_tokens=300)
    try:
        # Извлекаем JSON из ответа (DeepSeek может добавить markdown)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(raw[start:end])
            # Валидируем поля
            valid_intents = {"promise", "promise_without_date", "refusal", "delay_request", "question", "identity_question", "unclear"}
            if data.get("intent") not in valid_intents:
                data["intent"] = "unclear"
            data.setdefault("promise_date", None)
            data.setdefault("promise_amount", None)
            data.setdefault("requires_human", False)
            data.setdefault("suggested_reply", "")
            return data
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning("Ошибка парсинга ответа DeepSeek: %s | raw=%s", e, raw[:200])

    return {
        "intent": "unclear",
        "promise_date": None,
        "promise_amount": None,
        "requires_human": True,
        "suggested_reply": "Не удалось распознать намерение. Требуется уточнение.",
    }

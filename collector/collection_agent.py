#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collections/collection_agent.py
AI-диалоговый агент взыскания долгов через DeepSeek.

Версия: 1.1.1 (2026-05-11)

v1.1.1 (2026-05-11): уточнено определение soft_positive в AI-промпте —
  теперь требуется хотя бы одно платёжное слово; чистые приветствия
  (Здравствуйте, Добрый день и пр.) без платёжного контекста → unclear.

v1.1.0 (2026-04-29): добавлены fallback-шаблоны для старых хвостовых
  долгов без торговли и для частично погашаемых старых хвостов, без фразы
  про ограничение отгрузок.

v1.0.8 (2026-04-22): analyze_response() теперь безопасно переживает пустой
  или `None`-ответ от AI и уходит в fallback вместо падения на `.get`.
v1.0.9 (2026-04-23): смягчены клиентские ответы, убраны цифровые команды,
  добавлены intent-ветки soft_positive / promise_schedule / paid_claim.

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
from collector.logging_utils import get_collector_logger
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

logger = get_collector_logger(__name__)

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
        logger.debug("collector_prompts.json не найден: %s — используются defaults", e)
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
    if msg_type in ftpl:
        template = ftpl[msg_type]
    elif msg_type in _FALLBACK_TEMPLATES_DEFAULT:
        template = _FALLBACK_TEMPLATES_DEFAULT[msg_type]
    elif "strict_reminder" in ftpl:
        template = ftpl["strict_reminder"]
    else:
        template = _FALLBACK_TEMPLATES_DEFAULT["strict_reminder"]
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
        "Остаток задолженности{report_date_part} составляет {amount} тг. "
        "Он не закрыт уже {days_text}.\n"
        "Подскажите, пожалуйста, когда планируете ближайший платёж?"
    ),
    "payment_plan_control": (
        "Здравствуйте! Это {company}, отдел по работе с клиентами.\n\n"
        "Пишем по имеющейся задолженности: {client_name}.\n"
        "Ответственный менеджер: {manager_name}.\n\n"
        "Видим, что оплаты поступают, но остаток задолженности{report_date_part} составляет {amount} тг.\n"
        "Спасибо, что закрываете его частями. Когда планируете первый ближайший платёж и примерно какая будет сумма?"
    ),
    "strict_reminder": (
        "Здравствуйте! Это {company}, отдел по работе с клиентами.\n\n"
        "Пишем по имеющейся задолженности: {client_name}.\n"
        "Ответственный менеджер: {manager_name}.\n\n"
        "Остаток задолженности{report_date_part} составляет {amount} тг и не закрыт уже {days_text}.\n"
        "Подскажите, пожалуйста, когда сможете внести ближайший платёж."
    ),
    "stoplist_reminder": (
        "Здравствуйте! Это {company}, отдел по работе с клиентами.\n\n"
        "Пишем по имеющейся задолженности: {client_name}.\n"
        "Ответственный менеджер: {manager_name}.\n\n"
        "Остаток задолженности{report_date_part} составляет {amount} тг и не закрыт уже {days_text}, "
        "поэтому дальнейшие отгрузки ограничены до его закрытия.\n"
        "Подскажите, пожалуйста, когда сможете закрыть остаток или внести ближайший платёж."
    ),
    "legacy_tail_reminder": (
        "Здравствуйте! Это {company}, отдел по работе с клиентами.\n\n"
        "Пишем по имеющейся задолженности: {client_name}.\n"
        "Ответственный менеджер: {manager_name}.\n\n"
        "По нашим данным, задолженность{report_date_part} составляет {amount} тг и остаётся незакрытой уже {days_text}.\n"
        "Подскажите, пожалуйста, когда ожидается ближайшая оплата по закрытию остатка."
    ),
    "partial_tail_reminder": (
        "Здравствуйте! Это {company}, отдел по работе с клиентами.\n\n"
        "Пишем по имеющейся задолженности: {client_name}.\n"
        "Ответственный менеджер: {manager_name}.\n\n"
        "Видим частичное погашение, однако задолженность{report_date_part} всё ещё составляет {amount} тг.\n"
        "Подскажите, пожалуйста, когда планируете ближайший платёж и примерно какую сумму сможете внести."
    ),
    "no_movement_reminder": (
        "Здравствуйте! Это {company}, отдел по работе с клиентами.\n\n"
        "Обращаемся по задолженности: {client_name}.\n"
        "Ответственный менеджер: {manager_name}.\n\n"
        "Остаток задолженности{report_date_part} составляет {amount} тг и не закрыт уже {days_text}. "
        "За этот период оплат и отгрузок не поступало.\n"
        "Требуется незамедлительное подтверждение даты и суммы ближайшего платежа."
    ),
    "promise_broken_reminder": (
        "Здравствуйте! Это {company}, отдел по работе с клиентами.\n\n"
        "Обращаемся по задолженности: {client_name}.\n"
        "Ответственный менеджер: {manager_name}.\n\n"
        "Ранее вы обещали погасить задолженность, однако обещание не выполнено.\n"
        "Текущий остаток{report_date_part} составляет {amount} тг, просрочка — {days_text}.\n"
        "Укажите, пожалуйста, конкретную дату и сумму ближайшего платежа."
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
            f"(оптовые поставки продуктов питания, Астана, Казахстан). "
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
        "intent: одно из [promise, promise_without_date, promise_schedule, paid_claim, soft_positive, "
        "refusal, delay_request, question, identity_question, cash_pickup, dispute, "
        "doc_request, complaint, unclear]\n"
        "  - promise: клиент обещает оплатить и даёт конкретную дату или относительный срок "
        "('завтра', 'послезавтра', 'до пятницы', 'в пятницу', 'через 2 дня', 'на этой неделе', '17 числа' и т.д.).\n"
        "  - promise_schedule: клиент описывает график частичных платежей: 'ежедневно', 'частями', "
        "'по немного', 'по определённой сумме', 'буду закрывать постепенно'.\n"
        "  - paid_claim: клиент говорит, что уже оплатил, QR/куар/киар прошёл, чек есть, "
        "деньги скоро упадут или оплата ещё не разнесена.\n"
        "  - soft_positive: клиент ЯВНО сигнализирует о намерении платить, но без конкретной даты/суммы. "
        "Требуется хотя бы одно платёжное слово: 'оплачу', 'закрою', 'постараюсь', 'сегодня будет', 'переведу', 'оплатим'. "
        "Чистые приветствия ('Здравствуйте', 'Добрый день', 'Ок', 'Хорошо', 'Понял') без платёжного контекста — НЕ soft_positive, а unclear.\n"
        "    Для относительных дат ВЫЧИСЛИ конкретную дату YYYY-MM-DD от сегодня:\n"
        "    завтра=+1д, послезавтра=+2д, через N дней=+Nд, 'до пятницы'/'в пятницу'=ближайшая пятница,\n"
        "    'на этой неделе'=пятница текущей недели, 'до конца недели'=пятница текущей недели,\n"
        "    'в следующий понедельник'=+7д от ближайшего понедельника, 'до X числа'=X число текущего месяца.\n"
        "  - promise_without_date: клиент готов платить, но говорит 'скоро', 'не знаю когда', "
        "'постараюсь' без любой временной привязки\n"
        "  - delay_request: клиент просит отсрочку, называет причину почему не может сейчас\n"
        "  - identity_question: клиент спрашивает кто пишет, откуда номер, кто вы такие\n"
        "  - cash_pickup: клиент предлагает забрать оплату наличными у него "
        "('зайдут заберут', 'из кассы возьмите', 'самовывоз оплаты', 'налом отдам', "
        "'приезжайте за деньгами', 'у меня в магазине заберёте'). "
        "Наличку забирает МЕНЕДЖЕР, не бухгалтер. ВСЕГДА requires_human=true.\n"
        "  - dispute: клиент оспаривает сумму или сам факт долга "
        "('у меня по моим данным меньше', 'я уже всё оплатил, проверьте', 'это не моя задолженность', "
        "'сверим, у меня другие цифры'). ВСЕГДА requires_human=true.\n"
        "  - doc_request: клиент просит документы — акт сверки, счёт-фактуру, накладную, договор. "
        "ВСЕГДА requires_human=true.\n"
        "  - complaint: клиент жалуется на качество товара, доставку, сервис, менеджера, "
        "просрочку или порчу товара. ВСЕГДА requires_human=true.\n"
        "  - unclear: ни одна категория не подходит и смысл не считывается. ВСЕГДА requires_human=true.\n"
        "promise_date: YYYY-MM-DD — ОБЯЗАТЕЛЬНО вычисли если клиент назвал относительную дату. "
        "null только если дата вообще не упоминается\n"
        "promise_amount: число или null\n"
        "requires_human: true если агрессия, юридические угрозы, неоднозначность, "
        "или intent ∈ {cash_pickup, dispute, doc_request, complaint, unclear}\n"
        "suggested_reply при promise с датой и суммой: коротко подтверди договорённость.\n"
        "suggested_reply при promise с датой, но без суммы: подтверди срок без фразы 'фиксируем' "
        "и попроси чек после оплаты.\n"
        "suggested_reply при promise_schedule: признай график, зафиксируй частичные/ежедневные платежи "
        "и попроси чек после первого платежа.\n"
        "suggested_reply при paid_claim: поблагодари и попроси чек либо дату и сумму платежа. "
        "Не повторяй ссылку на 1С.\n"
        "suggested_reply при soft_positive: один мягкий короткий вопрос про первый платёж и примерную сумму.\n"
        "Никогда не используй механические команды с цифрами. Не подтверждай договорённость, если клиент не дал достаточно конкретики.\n"
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
    if not isinstance(raw, str):
        logger.warning("analyze_response: DeepSeek вернул %s вместо строки", type(raw).__name__)
        raw = ""
    try:
        # Извлекаем JSON из ответа (DeepSeek может добавить markdown)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(raw[start:end])
            # Валидируем поля
            valid_intents = {
                "promise", "promise_without_date", "promise_schedule", "paid_claim",
                "soft_positive", "refusal", "delay_request", "question",
                "identity_question",
                "cash_pickup", "dispute", "doc_request", "complaint",
                "unclear",
            }
            if data.get("intent") not in valid_intents:
                data["intent"] = "unclear"
            data.setdefault("promise_date", None)
            data.setdefault("promise_amount", None)
            data.setdefault("requires_human", False)
            data.setdefault("suggested_reply", "")
            # Жёсткое правило: при перечисленных intent-ах ответа от бота быть
            # не должно — диалог уходит к менеджеру/Саиде. Подавляем suggested_reply
            # и принудительно поднимаем requires_human, даже если модель забыла.
            if data["intent"] in {"cash_pickup", "dispute", "doc_request", "complaint", "unclear"}:
                data["requires_human"] = True
                data["suggested_reply"] = ""
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

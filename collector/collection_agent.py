#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collections/collection_agent.py
AI-диалоговый агент взыскания долгов через DeepSeek.

Версия: 1.0.0 (2026-03-16)

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

_TONE = {
    1: "Вежливый партнёрский тон. Мягкое напоминание, без давления. Выражай уважение и готовность помочь.",
    2: "Нейтральный деловой тон. Укажи что срок оплаты прошёл. Попроси уточнить причину задержки.",
    3: "Настойчивый тон. Чётко укажи на серьёзность просрочки. Прямо запроси конкретную дату оплаты.",
    4: "Строгий официальный тон. Укажи что данная ситуация требует немедленного решения. "
       "Упомяни что дальнейшая просрочка может повлечь правовые меры.",
    5: "Жёсткий официальный тон. Последнее предупреждение. Чётко укажи что дело будет "
       "передано в юридический отдел если оплата не поступит в течение 3 рабочих дней.",
}

_LANG_INSTRUCTION = {
    "ru": "Пиши ТОЛЬКО на русском языке.",
    "kz": "Тек қазақ тілінде жаз. (Пиши ТОЛЬКО на казахском языке.)",
}


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
        return data["choices"][0]["message"]["content"].strip()
    except (httpx.HTTPError, KeyError, json.JSONDecodeError) as e:
        logger.error("DeepSeek API ошибка: %s", e)
        return ""


def generate_message(
    client_name: str,
    debt_amount: float,
    days_overdue: int,
    level: int,
    language: str = "ru",
    previous_promise: Optional[str] = None,
    manager_name: str = "",
) -> str:
    """Генерирует персонализированный текст сообщения должнику через DeepSeek.

    Args:
        client_name:      Название клиента (как в базе дебиторки)
        debt_amount:      Сумма долга в тенге
        days_overdue:     Дней просрочки
        level:            Уровень давления 1–5
        language:         "ru" или "kz"
        previous_promise: Дата предыдущего обещания если было (ISO строка)
        manager_name:     Имя менеджера (для персонализации системного промпта)

    Returns:
        Текст сообщения (4–6 предложений, завершается вопросом о дате оплаты).
    """
    tone = _TONE.get(level, _TONE[1])
    lang_inst = _LANG_INSTRUCTION.get(language, _LANG_INSTRUCTION["ru"])
    amount_str = f"{debt_amount:,.0f}".replace(",", " ")

    promise_note = ""
    if previous_promise:
        promise_note = (
            f"\nВАЖНО: Клиент ранее обещал оплатить {previous_promise}, "
            f"но обещание не было выполнено. Упомяни это деликатно."
        )

    if manager_name:
        persona = (
            f"Ты — ИИ-помощник менеджера {manager_name} из компании {COMPANY_NAME} "
            f"(оптовые поставки продуктов питания, Алматы, Казахстан). "
            f"В начале сообщения представься: «Здравствуйте! Я — ИИ-помощник "
            f"вашего менеджера {manager_name}.»"
        )
    else:
        persona = (
            f"Ты — официальный представитель компании {COMPANY_NAME} "
            f"(оптовые поставки продуктов питания, Алматы, Казахстан). "
            f"В начале сообщения поздоровайся от имени компании."
        )

    system_prompt = (
        f"{persona} Ты ведёшь переписку по вопросу "
        "дебиторской задолженности. НЕ угрожаешь, НЕ давишь эмоционально — "
        "ты официальный и профессиональный представитель компании. "
        f"{lang_inst}"
    )
    user_prompt = (
        f"Напиши сообщение клиенту:\n"
        f"Клиент: {client_name}\n"
        f"Сумма долга: {amount_str} тенге\n"
        f"Дней просрочки: {days_overdue}\n"
        f"Тон: {tone}"
        f"{promise_note}\n\n"
        f"Требования:\n"
        f"- 4–6 предложений\n"
        f"- Живая речь, не шаблон\n"
        f"- В конце — конкретный вопрос о дате оплаты\n"
        f"- Без лишних заголовков, только текст сообщения"
    )
    result = _call_deepseek(system_prompt, user_prompt, max_tokens=400)
    if not result:
        # Fallback шаблон если DeepSeek недоступен
        result = (
            f"Добрый день! Напоминаем о задолженности в размере {amount_str} тенге "
            f"({days_overdue} дней). Просим сообщить планируемую дату оплаты."
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

    system_prompt = (
        f"Ты — аналитик сообщений должников {company_ctx}{mgr_ctx}. "
        "Твоя задача: проанализировать ответ клиента и вернуть ТОЛЬКО JSON без пояснений. "
        "Формат ответа строго:\n"
        '{"intent":"...", "promise_date":"...", "promise_amount":..., '
        '"requires_human":..., "suggested_reply":"..."}\n'
        "intent: одно из [promise, refusal, delay_request, question, unclear]\n"
        "promise_date: дата в формате YYYY-MM-DD или null\n"
        "promise_amount: число или null\n"
        "requires_human: true если агрессия, юридические угрозы или неоднозначность\n"
        "suggested_reply: короткий ответ агента на русском (1–2 предложения)\n"
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
            valid_intents = {"promise", "refusal", "delay_request", "question", "unclear"}
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

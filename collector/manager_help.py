#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collector/manager_help.py
Жёсткий DeepSeek-помощник для менеджеров внутри Telegram-запросов бота.

Помощник объясняет только текущий запрос и кнопки. Решений за менеджера
не принимает и на посторонние темы не отвечает.
"""
from __future__ import annotations

import html
import logging
import os
from typing import Any, Dict, Iterable

import httpx
from dotenv import load_dotenv

try:
    load_dotenv(encoding="utf-8-sig", override=False)
except Exception:
    pass

LOG = logging.getLogger(__name__)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

MANAGER_HELP_SYSTEM_PROMPT = """
Ты внутренний помощник менеджера в Telegram-боте компании Минбаракат.

ЖЁСТКИЕ ПРАВИЛА:
1. Отвечай только по текущему запросу бота и только о том, какую кнопку нажать
   или какие данные отправить.
2. Не обсуждай посторонние темы: политику, религию, новости, личные вопросы,
   программирование, развлечения, медицину, юриспруденцию, финсоветы.
3. Если вопрос не относится к текущему запросу бота, ответь: "Я могу помочь
   только с этим запросом бота: какую кнопку нажать и что написать."
4. Не принимай решение за менеджера. Объясни последствия вариантов.
5. Не обещай оплату, не разрешай отгрузку и не отменяй стоп сам.
6. Пиши простым русским языком, коротко, без канцелярита.
7. Не ругай менеджера, но прямо предупреждай: если не ответить, бот будет
   напоминать каждые 30 минут, затем запрос уйдёт руководителю со статистикой
   игнора.
8. Обязательно объясняй: по каждому менеджеру ведётся статистика игнора.
   Эта статистика видна руководителю и может повлиять на отношения менеджера
   с руководителем.
9. Не используй Markdown-таблицы. Не используй HTML-теги.

ОСНОВНЫЕ ПРИНЦИПЫ КНОПОК:
- "Договорились" означает: менеджер берёт ответственность, что с клиентом есть
  понятная договорённость. После этого надо написать детали: дату оплаты, сумму,
  условия. Без деталей запрос не закрыт.
- "Нет, стоп" означает: договорённости нет, клиента надо передать руководителю
  на решение по стопу/отгрузке.
- "Ввести имя" означает: менеджер пишет нормальное имя клиента для CRM и
  сообщений.
- "Оставить как в системе" означает: текущее имя из 1С подходит.
- "Позже" означает: имя можно уточнить позже, но запрос не исчезает.
- "Указать другой номер" означает: менеджер должен прислать актуальный WhatsApp.
- "Записать номер" означает: выбранный номер будет сохранён в CRM.
- "Отправить" означает: менеджер подтверждает данные и разрешает подготовку
  WhatsApp-уведомления по текущему клиенту.
- "Не отправлять" означает: менеджер должен объяснить причину; причина уйдёт
  руководителю.
- Админские кнопки по отгрузке объясняй только если запрос адресован руководителю:
  "Разрешить сейчас", "После оплаты с лимитом", "Запретить до оплаты",
  "Запретить".

ФОРМАТ ОТВЕТА:
1. "Что от вас хотят" — 1-2 предложения.
2. "Что нажать" — список вариантов по кнопкам.
3. "Что будет если молчать" — одно предупреждение.
""".strip()


def _fallback_help(area: str, buttons: Iterable[str], state: str = "") -> str:
    button_text = "\n".join(f"• {b}" for b in buttons) or "• Ответьте по смыслу запроса."
    state_line = f"\nЭтап: {state}." if state else ""
    return (
        "Что от вас хотят:\n"
        f"Нужно закрыть текущий запрос бота.{state_line}\n\n"
        "Что нажать:\n"
        f"{button_text}\n\n"
        "Что будет если молчать:\n"
        "Бот будет напоминать каждые 30 минут. После повторного игнора запрос "
        "уйдёт руководителю со статистикой. Статистика игнора ведётся по каждому "
        "менеджеру и может повлиять на отношения с руководителем."
    )


async def build_manager_help(
    *,
    area: str,
    manager: str = "",
    client: str = "",
    state: str = "",
    buttons: Iterable[str] = (),
    context: Dict[str, Any] | None = None,
) -> str:
    """Возвращает HTML-safe текст помощи для менеджера."""
    buttons = list(buttons)
    if not DEEPSEEK_API_KEY:
        return html.escape(_fallback_help(area, buttons, state))

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": MANAGER_HELP_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Раздел: {area}\n"
                    f"Менеджер: {manager or 'не указан'}\n"
                    f"Клиент: {client or 'не указан'}\n"
                    f"Этап/состояние: {state or 'не указано'}\n"
                    f"Доступные кнопки: {', '.join(buttons) or 'нет'}\n"
                    f"Контекст: {context or {}}\n\n"
                    "Объясни менеджеру, что делать именно в этом запросе."
                ),
            },
        ],
        "temperature": 0.0,
        "max_tokens": 450,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=25) as client:
            resp = await client.post(DEEPSEEK_URL, json=payload, headers=headers)
        if resp.status_code != 200:
            LOG.warning("manager_help DeepSeek error %s: %s", resp.status_code, resp.text[:200])
            return html.escape(_fallback_help(area, buttons, state))
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if not content:
            return html.escape(_fallback_help(area, buttons, state))
        return html.escape(content[:2500])
    except Exception as e:
        LOG.warning("manager_help DeepSeek failed: %s", e)
        return html.escape(_fallback_help(area, buttons, state))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Analyzer v9.4.18 - Генерация ИИ-анализа по типу отчёта
Fix #AI-1: AI_MAX_INPUT_CHARS 80000 → 15000 (экономия ~96% токенов)
Полная интеграция с .env настройками
Поддержка типов: DEBT, SALES, GROSS, INVENTORY, EXPENSES
Поддержка провайдеров: OpenAI (GPT) и DeepSeek

Изменения v9.4.17:
- Добавлен параметр --type (DEBT/SALES/GROSS/INVENTORY/EXPENSES)
- Загрузка промпта по типу: AI_PROMPT_DEBT, AI_PROMPT_SALES и т.д. из .env
- Имя файла содержит тип: ai_{type}_{manager}_{timestamp}.txt
- Обратная совместимость: --type не задан = DEBT (старое поведение)
"""

import logging
import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from openai import OpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Версия
VERSION = "v9.4.18"

# Поддерживаемые типы отчётов
REPORT_TYPES = ("DEBT", "SALES", "GROSS", "INVENTORY", "EXPENSES")

# Загрузка конфигурации
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env", encoding="utf-8-sig", override=True)

# Настройки из .env
TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))
AI_DIR = ROOT_DIR / "reports" / "ai"
AI_DIR.mkdir(parents=True, exist_ok=True)

# AI конфигурация из .env
AI_PROVIDER = os.getenv("AI_PROVIDER", "deepseek").lower()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Параметры генерации из .env
AI_PROMPT_PATH = os.getenv("AI_PROMPT_PATH", "")  # Legacy — используется как DEBT если нет AI_PROMPT_DEBT

# Промпты по типам отчётов (v9.4.17)
AI_PROMPT_DEBT      = os.getenv("AI_PROMPT_DEBT",      AI_PROMPT_PATH)  # fallback на старый путь
AI_PROMPT_SALES     = os.getenv("AI_PROMPT_SALES",     "")
AI_PROMPT_GROSS     = os.getenv("AI_PROMPT_GROSS",     "")
AI_PROMPT_INVENTORY = os.getenv("AI_PROMPT_INVENTORY", "")
AI_PROMPT_EXPENSES  = os.getenv("AI_PROMPT_EXPENSES",  "")

PROMPT_PATHS = {
    "DEBT":      AI_PROMPT_DEBT,
    "SALES":     AI_PROMPT_SALES,
    "GROSS":     AI_PROMPT_GROSS,
    "INVENTORY": AI_PROMPT_INVENTORY,
    "EXPENSES":  AI_PROMPT_EXPENSES,
}

AI_TEMPERATURE = float(os.getenv("AI_TEMPERATURE", "1.0"))
AI_MAX_TOKENS = int(os.getenv("AI_MAX_TOKENS", "1500"))
AI_MAX_INPUT_CHARS = int(os.getenv("AI_MAX_INPUT_CHARS", "15000"))  # Fix #AI-1: было 80000 (~3.8M токенов/мес → ~130K)
AI_SAVE_ALWAYS = os.getenv("AI_SAVE_ALWAYS", "true").lower() == "true"

# Telegram отправка (для обратной совместимости)
AI_TG_SEND_HTML = os.getenv("AI_TG_SEND_HTML", "false").lower() == "true"
AI_TG_SPLIT = os.getenv("AI_TG_SPLIT", "true").lower() == "true"
AI_TG_CHUNK = int(os.getenv("AI_TG_CHUNK", "3500"))
AI_TG_SLEEP_MS = int(os.getenv("AI_TG_SLEEP_MS", "400"))
AI_ASCII_STRICT = os.getenv("AI_ASCII_STRICT", "false").lower() == "true"
AI_TG_PRE = os.getenv("AI_TG_PRE", "false").lower() == "true"


def get_ai_client():
    """Возвращает клиента и модель на основе AI_PROVIDER из .env"""
    
    if AI_PROVIDER == "openai":
        if not OPENAI_API_KEY or OPENAI_API_KEY == "ВСТАВЬ_СЮДА_КЛЮЧ_GPT":
            raise ValueError(
                "AI_PROVIDER=openai но OPENAI_API_KEY не установлен!\n"
                "Получите ключ: https://platform.openai.com/api-keys"
            )
        logger.info("AI Provider: OpenAI, Model: %s", OPENAI_MODEL)
        return OpenAI(api_key=OPENAI_API_KEY), OPENAI_MODEL
    
    elif AI_PROVIDER == "deepseek":
        if not DEEPSEEK_API_KEY:
            raise ValueError(
                "AI_PROVIDER=deepseek но DEEPSEEK_API_KEY не установлен!\n"
                "Получите ключ: https://platform.deepseek.com"
            )
        logger.info("AI Provider: DeepSeek, Model: %s", DEEPSEEK_MODEL)
        return OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        ), DEEPSEEK_MODEL
    
    else:
        # Автовыбор: пробуем DeepSeek, потом OpenAI
        if DEEPSEEK_API_KEY:
            logger.info("AI Provider: DeepSeek (автовыбор), Model: %s", DEEPSEEK_MODEL)
            return OpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com"
            ), DEEPSEEK_MODEL
        elif OPENAI_API_KEY and OPENAI_API_KEY != "ВСТАВЬ_СЮДА_КЛЮЧ_GPT":
            logger.info("AI Provider: OpenAI (автовыбор), Model: %s", OPENAI_MODEL)
            return OpenAI(api_key=OPENAI_API_KEY), OPENAI_MODEL
        else:
            raise ValueError(
                "Не найдены API ключи!\n"
                "Установите в .env:\n"
                "  OPENAI_API_KEY=sk-proj-... или\n"
                "  DEEPSEEK_API_KEY=sk-..."
            )


# Дефолтные промпты по типам (если файл не найден)
DEFAULT_PROMPTS = {
    "DEBT": """Ты — финансовый аналитик компании. Проанализируй данные о дебиторской задолженности.
Обязательно укажи в отчете имя менеджера из данных.
Структура: 1. КРАТКАЯ СВОДКА  2. ГЛАВНЫЕ ДОЛЖНИКИ (топ-3)  3. СРОЧНЫЕ ДЕЙСТВИЯ""",

    "SALES": """Ты — коммерческий директор. Проанализируй данные о продажах.
Обязательно укажи имя менеджера.
Структура: 1. ОБЩИЕ ПОКАЗАТЕЛИ  2. ТОП КЛИЕНТЫ (топ-3)  3. ТОП ТОВАРЫ  4. РЕКОМЕНДАЦИИ""",

    "GROSS": """Ты — финансовый директор. Проанализируй маржинальность.
Структура: 1. ОБЩАЯ МАРЖА (%)  2. ПРИБЫЛЬНЫЕ ПОЗИЦИИ  3. УБЫТОЧНЫЕ ПОЗИЦИИ  4. РЕКОМЕНДАЦИИ""",

    "INVENTORY": """Ты — директор по закупкам. Проанализируй складские остатки (скоропортящиеся продукты).
Структура: 1. ОБЩИЙ СКЛАД  2. КРИТИЧЕСКИЕ ОСТАТКИ (>500К ₸)  3. СРОЧНО ПРОДАТЬ  4. ПЛАН""",

    "EXPENSES": """Ты — финансовый контролёр. Проанализируй расходы компании.
Структура: 1. ИТОГ ЗАТРАТ  2. ТОП СТАТЬИ (топ-5)  3. АНОМАЛИИ  4. ОПТИМИЗАЦИЯ""",
}


def load_prompt(report_type: str = "DEBT") -> str:
    """Загружает промпт по типу отчёта из .env или дефолтный (v9.4.17)"""
    
    rtype = report_type.upper() if report_type else "DEBT"
    prompt_path = PROMPT_PATHS.get(rtype, "")
    
    if prompt_path and Path(prompt_path).exists():
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            logger.info("Промпт [%s] загружен: %s", rtype, Path(prompt_path).name)
            return prompt
        except Exception as e:
            logger.warning("Ошибка загрузки промпта [%s]: %s", rtype, e)
    else:
        if prompt_path:
            logger.warning("Файл промпта [%s] не найден: %s", rtype, prompt_path)
        else:
            logger.info("AI_PROMPT_%s не задан в .env — используется дефолтный", rtype)
    
    # Дефолтный промпт по типу
    return DEFAULT_PROMPTS.get(rtype, DEFAULT_PROMPTS["DEBT"])


def extract_manager_from_data(data: dict) -> str:
    """Извлекает имя менеджера из JSON данных"""
    
    # Пробуем найти в aggregates
    if isinstance(data, dict):
        aggregates = data.get("aggregates", {})
        manager = aggregates.get("manager", "")
        if manager:
            return manager
    
    # Пробуем найти в первом контрагенте
    if isinstance(data, dict):
        contractors = data.get("contractors", [])
        if contractors and isinstance(contractors, list):
            first = contractors[0]
            if isinstance(first, dict):
                manager = first.get("manager", "")
                if manager:
                    return manager
    
    return "Unknown"


def _load_manager_names() -> list:
    """Читает имена менеджеров из config/managers.json. Fallback: пустой список."""
    cfg = ROOT_DIR / "config" / "managers.json"
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            data = json.load(f)
        return list(data.keys())
    except (OSError, json.JSONDecodeError):
        return []


def extract_manager_from_filename(path: str) -> str:
    """Извлекает имя менеджера из имени файла (запасной вариант).
    Читает список менеджеров из config/managers.json, не из хардкода."""
    filename = Path(path).name.lower()
    managers = _load_manager_names()
    for manager in managers:
        if manager.lower() in filename:
            return manager
    return "Unknown"


def analyze(path: str, chat_id: str, send_mode: bool = False, report_type: str = "DEBT",
            send_html: bool = False):
    """Основная функция анализа"""
    
    logger.info("AI ANALYZER %s | Файл: %s | Тип: %s | Chat: %s",
                VERSION, path, report_type, chat_id)
    
    # Проверка файла
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    
    # Загрузка данных
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data_str = json.dumps(data, ensure_ascii=False, indent=2)
        logger.info("Данные загружены: %d символов", len(data_str))
    except Exception as e:
        raise RuntimeError(f"Ошибка чтения JSON: {e}") from e
    
    # Ограничение размера входных данных
    if len(data_str) > AI_MAX_INPUT_CHARS:
        truncated = data_str[:AI_MAX_INPUT_CHARS]
        # Обрезаем до последнего переноса строки, чтобы не рвать JSON-значение посередине.
        # json.dumps с indent=2 ставит каждое значение на отдельную строку.
        last_nl = truncated.rfind('\n')
        if last_nl > AI_MAX_INPUT_CHARS // 2:
            truncated = truncated[:last_nl]
        data_str = truncated + "\n... [данные обрезаны]"
        logger.warning("Данные обрезаны: %d → %d символов", len(data_str), AI_MAX_INPUT_CHARS)
    
    # Определяем менеджера
    manager = extract_manager_from_data(data)
    if manager == "Unknown":
        manager = extract_manager_from_filename(path)
    logger.info("Менеджер: %s", manager)
    
    # Загружаем промпт по типу отчёта
    system_prompt = load_prompt(report_type)
    
    # Формируем запрос
    type_labels = {
        "DEBT": "дебиторской задолженности",
        "SALES": "продаж",
        "GROSS": "валовой прибыли и маржинальности",
        "INVENTORY": "складских остатков",
        "EXPENSES": "затрат компании",
    }
    type_label = type_labels.get(report_type.upper(), "отчёта")
    
    manager_note = f"\n\nМенеджер: {manager}" if manager and manager != "Unknown" else ""
    user_content = f"""Данные {type_label}:

{data_str}

Проанализируй эти данные согласно инструкции.{manager_note}"""
    
    # Получаем AI клиента
    try:
        client, model = get_ai_client()
    except ValueError as e:
        raise RuntimeError(f"Ошибка конфигурации: {e}") from e
    
    logger.info("Температура: %s, Max tokens: %d", AI_TEMPERATURE, AI_MAX_TOKENS)
    logger.info("Генерация анализа...")
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=AI_TEMPERATURE,
            max_tokens=AI_MAX_TOKENS
        )
        
        answer = response.choices[0].message.content
        elapsed = time.time() - start_time
        
        logger.info("Анализ готов! (%.1f сек)", elapsed)
        if hasattr(response, 'usage'):
            logger.info("Токены: %d (prompt: %d, completion: %d)",
                        response.usage.total_tokens,
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens)
        
    except Exception as e:
        raise RuntimeError(f"Ошибка API: {e}") from e
    
    # Сохранение результата
    if AI_SAVE_ALWAYS or send_mode:
        timestamp = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
        rtype_lower = report_type.lower() if report_type else "debt"
        output_file = AI_DIR / f"ai_{rtype_lower}_{manager}_{timestamp}.txt"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(answer)
            logger.info("Результат сохранён: %s", output_file)
            print(f"AI saved: {output_file}")  # sentinel: парсится subprocess-caller'ом в send_reports.py
            
        except Exception as e:
            raise RuntimeError(f"Ошибка сохранения: {e}") from e
    
    # Отправка в Telegram (если требуется)
    if send_mode:
        logger.info("Отправка в Telegram...")
        try:
            from send_tg import send_long_text, send_file

            if AI_TG_SEND_HTML or send_html:
                from tools.txt_to_html import build_html
                html_path = build_html(output_file)
                send_file(html_path, caption="AI-анализ дебиторки", chat_id=chat_id)
                logger.info("HTML отправлен: %s", html_path.name)
            else:
                send_long_text(answer, chat_id=chat_id)
                send_file(output_file, caption="AI-анализ (.txt)", chat_id=chat_id)
                logger.info("Текст отправлен")

        except Exception as e:
            logger.warning("Ошибка отправки в Telegram: %s", e)

    logger.info("ГОТОВО")


def main(argv=None):
    """CLI точка входа"""
    
    parser = argparse.ArgumentParser(
        description=f'AI Debt Analysis Generator {VERSION}',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  
  # Генерация анализа (только сохранение)
  python ai_analyzer.py --path reports/json/debt_Ергали_20250101.json --chat-id 123456789
  
  # Генерация + отправка в Telegram
  python ai_analyzer.py --path reports/json/debt_Ергали_20250101.json --chat-id 123456789 --send
  
Настройки в .env:
  AI_PROVIDER=deepseek|openai   - выбор провайдера
  AI_PROMPT_PATH=путь/к/промпту - кастомный промпт
  AI_TEMPERATURE=0.7-2.0        - креативность
  AI_MAX_TOKENS=1500            - длина ответа
        """
    )
    
    parser.add_argument('--path', required=True, 
                       help='Путь к JSON файлу с данными')
    parser.add_argument('--chat-id', required=True, 
                       help='Telegram chat ID')
    parser.add_argument('--type', default='DEBT',
                       choices=['DEBT', 'SALES', 'GROSS', 'INVENTORY', 'EXPENSES'],
                       help='Тип отчёта (default: DEBT)')
    parser.add_argument('--send', action='store_true',
                       help='Отправить результат в Telegram')
    parser.add_argument('--send-html', action='store_true',
                       help='Отправить как HTML (устарело, используйте AI_TG_SEND_HTML в .env)')
    
    args = parser.parse_args(argv)
    
    # --send-html устарел, но поддерживается для обратной совместимости
    if args.send_html:
        logger.warning("--send-html устарел, используйте AI_TG_SEND_HTML=true в .env")

    try:
        analyze(args.path, args.chat_id, send_mode=args.send, report_type=args.type,
                send_html=args.send_html)
    except Exception as e:
        logger.error("FAILED: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
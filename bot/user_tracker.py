"""
bot/user_tracker.py
Система отслеживания активности пользователей

Версия: 1.0.0
Дата: 2026-02-02
"""

import json
import logging
import os
from pathlib import Path
from datetime import datetime
from tempfile import NamedTemporaryFile
from zoneinfo import ZoneInfo
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Путь к файлу с данными
ANALYTICS_FILE = Path(__file__).parent.parent / "logs" / "user_analytics.json"
TZ = ZoneInfo(os.getenv("TZ", "Asia/Almaty"))

def _load_analytics() -> Dict[str, Any]:
    """Загружает данные аналитики из JSON"""
    try:
        if ANALYTICS_FILE.exists():
            with open(ANALYTICS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"users": {}, "actions": []}
    except Exception as e:
        logger.error(f"Ошибка загрузки аналитики: {e}")
        return {"users": {}, "actions": []}

def _save_analytics(data: Dict[str, Any]):
    """Атомарно сохраняет данные аналитики в JSON через tmp + replace."""
    tmp = None
    try:
        ANALYTICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", delete=False, encoding="utf-8",
                                dir=ANALYTICS_FILE.parent, suffix=".tmp") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            tmp = f.name
        os.replace(tmp, ANALYTICS_FILE)
    except Exception as e:
        logger.error(f"Ошибка сохранения аналитики: {e}")
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass

def track_user(user_id: int, first_name: str, username: str = None):
    """
    Регистрирует или обновляет пользователя
    
    Args:
        user_id: Telegram ID пользователя
        first_name: Имя пользователя
        username: Username пользователя (если есть)
    """
    try:
        data = _load_analytics()
        user_key = str(user_id)
        now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
        
        if user_key not in data["users"]:
            # Новый пользователь
            data["users"][user_key] = {
                "user_id": user_id,
                "first_name": first_name,
                "username": username,
                "first_seen": now,
                "last_seen": now,
                "total_actions": 0
            }
            logger.info(f"👤 Новый пользователь зарегистрирован: {first_name} (ID: {user_id})")
        else:
            # Обновляем данные существующего пользователя
            data["users"][user_key]["last_seen"] = now
            data["users"][user_key]["first_name"] = first_name  # Обновляем имя (могло измениться)
            if username:
                data["users"][user_key]["username"] = username
        
        _save_analytics(data)
    except Exception as e:
        logger.error(f"Ошибка отслеживания пользователя {user_id}: {e}")

def track_action(user_id: int, action: str, details: str = None):
    """
    Записывает действие пользователя
    
    Args:
        user_id: Telegram ID пользователя
        action: Тип действия (debt, sales, ai, back, etc.)
        details: Дополнительные детали действия
    """
    try:
        data = _load_analytics()
        user_key = str(user_id)
        now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
        
        # Добавляем действие в историю
        action_record = {
            "user_id": user_id,
            "action": action,
            "timestamp": now
        }
        
        if details:
            action_record["details"] = details
        
        data["actions"].append(action_record)
        
        # Обновляем счётчик действий пользователя
        if user_key in data["users"]:
            data["users"][user_key]["total_actions"] += 1
            data["users"][user_key]["last_seen"] = now
        
        # Ограничиваем историю действий (храним последние 10000)
        if len(data["actions"]) > 10000:
            data["actions"] = data["actions"][-10000:]
        
        _save_analytics(data)
        
        logger.debug(f"📊 Действие записано: user={user_id}, action={action}")
    except Exception as e:
        logger.error(f"Ошибка записи действия для {user_id}: {e}")

def get_stats() -> Dict[str, Any]:
    """
    Получает статистику использования бота
    
    Returns:
        Dict с полной статистикой
    """
    try:
        data = _load_analytics()
        
        # Подсчитываем статистику по действиям
        action_counts = {}
        for action in data.get("actions", []):
            action_type = action.get("action", "unknown")
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        # Сортируем пользователей по активности
        users_list = []
        for user_id, user_data in data.get("users", {}).items():
            users_list.append({
                "user_id": user_data.get("user_id"),
                "first_name": user_data.get("first_name"),
                "username": user_data.get("username"),
                "first_seen": user_data.get("first_seen"),
                "last_seen": user_data.get("last_seen"),
                "total_actions": user_data.get("total_actions", 0)
            })
        
        users_list.sort(key=lambda x: x["total_actions"], reverse=True)
        
        return {
            "total_users": len(data.get("users", {})),
            "total_actions": len(data.get("actions", [])),
            "action_breakdown": action_counts,
            "users": users_list
        }
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        return {
            "total_users": 0,
            "total_actions": 0,
            "action_breakdown": {},
            "users": []
        }

def format_stats_message(stats: Dict[str, Any]) -> str:
    """
    Форматирует статистику для отправки в Telegram
    
    Args:
        stats: Данные статистики из get_stats()
    
    Returns:
        Отформатированное сообщение для Telegram
    """
    message_parts = ["📊 *СТАТИСТИКА БОТА AI 1C PRO*\n"]
    
    # Общая информация
    message_parts.append(f"👥 Всего пользователей: *{stats['total_users']}*")
    message_parts.append(f"⚡ Всего действий: *{stats['total_actions']}*")
    
    if stats['total_users'] > 0:
        avg = stats['total_actions'] / stats['total_users']
        message_parts.append(f"📈 Среднее действий/пользователь: *{avg:.1f}*\n")
    
    # ТОП-5 активных пользователей
    if stats['users']:
        message_parts.append("*🏆 ТОП-5 активных пользователей:*")
        for i, user in enumerate(stats['users'][:5], 1):
            username_str = f" (@{user['username']})" if user['username'] else ""
            message_parts.append(
                f"{i}. {user['first_name']}{username_str} — {user['total_actions']} действий"
            )
        message_parts.append("")
    
    # Популярные действия
    if stats['action_breakdown']:
        message_parts.append("*📋 Популярные действия:*")
        sorted_actions = sorted(
            stats['action_breakdown'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for action, count in sorted_actions[:5]:
            action_emoji = {
                "debt": "📊",
                "sales": "🛒",
                "gross": "💰",
                "inventory": "📦",
                "ai": "🤖",
                "back": "◀️",
                "menu": "📋"
            }.get(action, "•")
            message_parts.append(f"  {action_emoji} {action}: {count}")
    
    return "\n".join(message_parts)

def get_user_info(user_id: int) -> Dict[str, Any]:
    """
    Получает информацию о конкретном пользователе
    
    Args:
        user_id: Telegram ID пользователя
    
    Returns:
        Dict с информацией о пользователе
    """
    try:
        data = _load_analytics()
        user_key = str(user_id)
        
        if user_key in data["users"]:
            return data["users"][user_key]
        
        return None
    except Exception as e:
        logger.error(f"Ошибка получения информации о пользователе {user_id}: {e}")
        return None


# Пример использования
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Тестовые данные
    track_user(12345, "Иван", "ivan_test")
    track_action(12345, "debt")
    track_action(12345, "sales")
    track_action(12345, "ai")
    
    track_user(67890, "Мария", "maria_test")
    track_action(67890, "inventory")
    track_action(67890, "gross")
    
    # Получаем статистику
    stats = get_stats()
    message = format_stats_message(stats)
    
    print("\n" + "="*60)
    print(message)
    print("="*60)
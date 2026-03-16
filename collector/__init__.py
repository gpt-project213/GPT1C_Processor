"""
collections/ — AI-Коллектор долгов
Версия: 1.0.0 (2026-03-16)

Модули:
  debt_monitor       — классификация должников по уровням риска
  collections_db     — хранение истории, обещаний, статусов (logs/collector_state.json)
  communications     — gateway: WhatsApp (Green API) + Telegram + эскалация admin
  collection_agent   — AI-диалог через DeepSeek (генерация + анализ ответа)
  voice_calls        — голосовые звонки через Retell AI
  collections_engine — главный оркестратор + CLI
"""

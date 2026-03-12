import os, time, json
from pathlib import Path
from datetime import datetime
import asyncio
from telegram import Bot
from send_reports import (
    _load_json_safe, _read_full, _extract_manager,
    ROLES, MANAGERS_MAP, ADMIN_CHAT_ID, TZ,
    HTML_DIR, AI_DIR, NOTIFY_STATE_PATH, SCAN_INTERVAL_MIN, kb_open_menu
)

async def main():
    token = os.getenv("TG_BOT_TOKEN") or os.getenv("BOT_TOKEN")
    if not token:
        print("TG_BOT_TOKEN/BOT_TOKEN не задан"); return
    bot = Bot(token)

    # 1) Обнулим state (чтобы считаło как новые)
    try:
        if Path(NOTIFY_STATE_PATH).exists():
            Path(NOTIFY_STATE_PATH).unlink()
    except OSError:
        pass

    watermark_time = time.time() - (SCAN_INTERVAL_MIN * 60 * 2)
    all_files = list(HTML_DIR.glob("*.html")) + list(AI_DIR.glob("*.html"))
    notifications = {}
    for file in all_files:
        try:
            mtime = file.stat().st_mtime
            if mtime < watermark_time:
                continue
            full_text = _read_full(file)
            manager = _extract_manager(full_text, file.name)
            # менеджеру
            for mgr_name, chat_id in (MANAGERS_MAP or {}).items():
                if mgr_name.strip().lower() == manager.strip().lower():
                    notifications.setdefault(chat_id, []).append(f"📊 {file.name}")
            # субадминам по скоупам
            subadmin_scopes = ROLES.get("subadmin_scopes", {})
            for str_chat_id, scope_list in subadmin_scopes.items():
                if any(manager.strip().lower()==s.strip().lower() for s in scope_list):
                    notifications.setdefault(int(str_chat_id), []).append(f"📊 {manager}: {file.name}")
            # админу
            if ADMIN_CHAT_ID:
                notifications.setdefault(ADMIN_CHAT_ID, []).append(f"📊 {manager}: {file.name}")
        except (OSError, ValueError, KeyError, AttributeError, TypeError):
            continue

    sent_total = 0
    for chat_id, reports in notifications.items():
        if not isinstance(chat_id, int) or chat_id <= 0:
            continue
        current_time = datetime.now(TZ).strftime('%H:%M')
        if len(reports) > 10:
            message = f"🔔 Обновлено {len(reports)} отчетов ({current_time})"
        else:
            message = f"🔔 Новые отчеты ({current_time}):\n\n"
            for report in reports[:10]:
                clean_report = report.replace("📊 ", "") if report.startswith("📊 ") else report
                message += f"📊 {clean_report}\n"
            if len(reports) > 10:
                message += f"\n... и еще {len(reports)-10} отчетов"
        try:
            await bot.send_message(chat_id=chat_id, text=message, reply_markup=kb_open_menu())
            sent_total += 1
        except Exception as e:
            print(f"send_error chat_id={chat_id}: {e}")

    print(f"done: delivered_to={sent_total} chats")

if __name__ == "__main__":
    asyncio.run(main())

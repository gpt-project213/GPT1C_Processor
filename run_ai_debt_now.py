#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Перегенерация AI анализа дебиторки по каждому менеджеру + отправка в Telegram."""

import subprocess, sys, os, re, time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

TZ = ZoneInfo("Asia/Almaty")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID", "7422963573")

MANAGERS = {
    "Алена":  {"chat_id": "188939016",  "json": "debt_ext_Детальный Дебиторы Алена (34).json"},
    "Ергали": {"chat_id": "756622791",  "json": "debt_ext_Детальный Дебиторы Ергали (34).json"},
    "Магира": {"chat_id": "735574334",  "json": "debt_ext_Детальный Дебиторы Магира (38).json"},
    "Оксана": {"chat_id": "1446255940", "json": "debt_ext_Детальный Дебиторы Оксана (34).json"},
}

JSON_DIR = ROOT / "reports" / "json"
AI_DIR   = ROOT / "reports" / "ai"
AI_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from send_tg import send_file, send_long_text
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", encoding="utf-8-sig", override=True)


def run_analyzer(json_path: Path, chat_id: str) -> Path | None:
    """Запускает ai_analyzer.py, возвращает путь к сохранённому TXT."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "ai_analyzer.py"),
         "--path", str(json_path),
         "--chat-id", chat_id,
         "--type", "DEBT"],
        capture_output=True, text=True, encoding="utf-8", timeout=180
    )
    print(result.stdout[-500:] if result.stdout else "")
    if result.returncode != 0:
        print(f"  ОШИБКА ai_analyzer: {result.stderr[-300:]}", flush=True)
        return None
    m = re.search(r"AI saved:\s*(.+)", result.stdout)
    if not m:
        print("  ai_analyzer не вернул путь к файлу", flush=True)
        return None
    return Path(m.group(1).strip())


def build_html(manager: str, txt_content: str) -> Path:
    ts = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
    html_path = AI_DIR / f"ai_debt_{manager}_{ts}.html"
    html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Анализ - {manager}</title>
<style>
body {{font-family:system-ui,Arial,sans-serif;padding:20px;max-width:800px;margin:0 auto;line-height:1.6}}
pre {{white-space:pre-wrap;word-wrap:break-word;background:#f5f5f5;padding:15px;border-radius:8px}}
h1 {{color:#2563eb}}
</style>
</head>
<body>
<h1>AI Анализ дебиторки: {manager}</h1>
<pre>{txt_content}</pre>
</body>
</html>"""
    html_path.write_text(html_content, encoding="utf-8")
    return html_path


def main():
    results = []
    for manager, info in MANAGERS.items():
        print(f"\n{'='*50}", flush=True)
        print(f"[{manager}] Генерация...", flush=True)

        json_path = JSON_DIR / info["json"]
        if not json_path.exists():
            print(f"  JSON не найден: {json_path}", flush=True)
            continue

        txt_file = run_analyzer(json_path, ADMIN_CHAT_ID)
        if not txt_file or not txt_file.exists():
            print(f"  Не удалось получить TXT файл", flush=True)
            continue

        txt_content = txt_file.read_text(encoding="utf-8")
        html_path = build_html(manager, txt_content)
        print(f"  HTML: {html_path.name}", flush=True)

        # Отправить только админу (AI-анализ содержит "косяки" — не для менеджеров)
        ok_admin = send_file(html_path, chat_id=ADMIN_CHAT_ID,
                             caption=f"AI Дебиторка - {manager}")
        print(f"  >> Админу ({ADMIN_CHAT_ID}): {'OK' if ok_admin else 'FAIL'}", flush=True)

        results.append({"manager": manager, "html": html_path, "ok": ok_admin})
        time.sleep(5)

    print(f"\n{'='*50}")
    print(f"Готово: {sum(1 for r in results if r['ok'])}/{len(MANAGERS)} отправлено", flush=True)


if __name__ == "__main__":
    main()

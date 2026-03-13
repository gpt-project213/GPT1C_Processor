#!/usr/bin/env python
# coding: utf-8
"""
tools/patch_mobile_click.py · v1.1 · Asia/Almaty
------------------------------------------------
Патч HTML-отчёта для удобного тапа на iOS/Android.

✓ Добавляет CSS: увеличенная зона нажатия + touch-action:manipulation
  (fix: iOS WKWebView требует touch-action для кнопок и ссылок)
✓ JS-блок убран: он конфликтовал с собственными обработчиками шаблонов
  (вызывал e.preventDefault() дважды → ссылки не работали на iOS)
✓ Не вносит изменений, если патч уже присутствует (идемпотентен)

Использование CLI:
    python tools/patch_mobile_click.py path/to/file.html

Использование из кода:
    from tools.patch_mobile_click import patch
    patch(Path_to_html)
"""

from __future__ import annotations
import sys, re
from pathlib import Path

# ---- вставляемые фрагменты ---------------------------------------------------
# JS-блок удалён в v1.1: он добавлял e.preventDefault() + scrollIntoView,
# что конфликтовало с обработчиками вкладок в шаблонах и блокировало ссылки
# на iOS WKWebView (Telegram). CSS достаточен для корректной работы тапов.
CSS_BLOCK = """
<style id="mobile-patch">
/* iOS WKWebView: touch-action убирает задержку 300мс и делает элементы кликабельными */
a,button,.tab{touch-action:manipulation;-webkit-tap-highlight-color:rgba(0,0,0,0.05)}
a[href^="#"]{display:inline-block;min-height:44px;min-width:44px;line-height:44px}
</style>
""".strip()

# ------------------------------------------------------------------------------
def patch(path: Path) -> bool:
    """Патчит файл на месте. Возвращает True, если был изменён."""
    if not path.exists():
        raise FileNotFoundError(path)
    html = path.read_text(encoding="utf-8", errors="ignore")
    if "id=\"mobile-patch\"" in html:        # уже патчено
        return False
    html = re.sub(r"</head>", CSS_BLOCK + "\n</head>", html, count=1, flags=re.I)
    path.write_text(html, encoding="utf-8")
    return True

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: patch_mobile_click.py file.html"); sys.exit(1)
    p = Path(sys.argv[1])
    changed = patch(p)
    print("patched" if changed else "already ok")

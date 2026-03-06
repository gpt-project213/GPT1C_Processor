#!/usr/bin/env python
# coding: utf-8
"""
tools/patch_mobile_click.py · v1.0 · Asia/Almaty
------------------------------------------------
Патч HTML-отчёта для удобного тапа на iOS/Android.

✓ Добавляет CSS .tap-area (увеличивает “зону нажатия” ссылок-таба)
✓ Добавляет JS, который плавно скроллит к anchor и не блокирует клики
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
CSS_BLOCK = """
<style id="mobile-patch">
.tap-area{display:inline-block;padding:14px 22px;margin:-14px -22px}
</style>
""".strip()

JS_BLOCK = """
<script id="mobile-patch-js">
document.addEventListener("DOMContentLoaded",()=>{const anchors=[...document.querySelectorAll('a[href^="#"]')];
anchors.forEach(a=>{if(!a.classList.contains("tap-area")){const w=document.createElement("span");
w.className="tap-area";a.parentNode.insertBefore(w,a);w.appendChild(a);}
a.addEventListener("click",e=>{e.preventDefault();
const id=a.getAttribute("href").substring(1);
const el=document.getElementById(id);
if(el){el.scrollIntoView({behavior:"smooth",block:"start"});}
});});});
</script>
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
    html = re.sub(r"</body>", JS_BLOCK + "\n</body>", html, count=1, flags=re.I)
    path.write_text(html, encoding="utf-8")
    return True

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: patch_mobile_click.py file.html"); sys.exit(1)
    p = Path(sys.argv[1])
    changed = patch(p)
    print("patched" if changed else "already ok")

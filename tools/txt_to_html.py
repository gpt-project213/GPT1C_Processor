#!/usr/bin/env python
# coding: utf-8
"""
tools/txt_to_html.py · v1.0.1 · Asia/Almaty
Конвертирует текстовый отчёт ИИ (.txt/.json) в аккуратный HTML и возвращает путь.
v1.0.1: TZ timezone(timedelta(hours=5)) → ZoneInfo("Asia/Almaty") (Bug TZ)
        OUT_DIR: relative path → absolute (Bug M6)
"""

from __future__ import annotations
import json, re, html as _html
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Union

TZ = ZoneInfo("Asia/Almaty")
OUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "html"

_CSS = """
body{margin:0;font-family:Consolas,monospace,Menlo,Monaco,monospace;background:#fff;color:#111}
h1{font-size:20px;margin:16px 0;padding:0}
pre{white-space:pre-wrap;word-wrap:break-word;background:#f8f8f8;border:1px solid #eee;
     border-radius:6px;padding:12px;font-size:14px;line-height:1.35}
footer{margin:24px 0 16px 0;font-size:12px;color:#555;text-align:center}
.container{max-width:950px;margin:0 auto;padding:0 12px}
"""
_HTML_WRAP = """<!DOCTYPE html>
<html lang="ru"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{css}</style></head>
<body><div class="container">
<h1>🤖 Анализ ИИ</h1>
{body}
<footer>Сформировано: {stamp} (Asia/Almaty)</footer>
</div></body></html>"""

def _render_json(obj) -> str:
    if isinstance(obj, dict):
        items = "".join(f"<li><b>{_html.escape(str(k))}</b>: {_render_json(v)}</li>" for k, v in obj.items())
        return f"<ul>{items}</ul>"
    if isinstance(obj, list):
        items = "".join(f"<li>{_render_json(v)}</li>" for v in obj)
        return f"<ol>{items}</ol>"
    return _html.escape(str(obj))

def _txt_to_html(txt: str) -> str:
    safe = _html.escape(txt)
    safe = re.sub(r"^# (.+)$", r"<h2>\1</h2>", safe, flags=re.MULTILINE)
    safe = re.sub(r"^## (.+)$", r"<h3>\1</h3>", safe, flags=re.MULTILINE)
    safe = re.sub(r"^(?:- |• )(.*)$", r"• \1", safe, flags=re.MULTILINE)
    return f"<pre>{safe}</pre>"

def build_html(txt_path: Union[str, Path]) -> Path:
    p = Path(txt_path)
    if not p.exists():
        raise FileNotFoundError(p)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = p.read_text(encoding="utf-8", errors="ignore")

    try:
        body = _render_json(json.loads(raw))
    except Exception:
        body = _txt_to_html(raw)

    stamp = datetime.now(TZ).strftime("%d.%m.%Y %H:%M")
    html_text = _HTML_WRAP.format(title=f"AI-report · {p.stem}", css=_CSS, body=body, stamp=stamp)
    out_path = OUT_DIR / f"ai_{p.stem}.html"
    out_path.write_text(html_text, encoding="utf-8")
    return out_path

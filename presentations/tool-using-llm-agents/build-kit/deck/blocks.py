"""Constructors for slide content. Every prose field is bilingual: en + zh.

Conventions checked by tools/lint.py:
  * prose sentences: 12-25 words (hard limit 35); label=True exempts fragments
  * no em dashes, no question marks, no 'Quotient', no 'Next:' pointers
  * '(my synthesis)' / '（个人归纳）' at most once per slide, same count in both languages
  * math ($...$) and numbers must match between en and zh
"""


def T(en, zh):
    return {"en": en, "zh": zh}


def P(en, zh, cls="", label=False):
    return {"t": "p", "en": en, "zh": zh, "cls": cls, "label": label}


def UL(*items, label=False, cls=""):
    out = []
    for it in items:
        if len(it) == 2:
            out.append({"en": it[0], "zh": it[1]})
        else:
            out.append({"lead": {"en": it[0], "zh": it[1]}, "en": it[2], "zh": it[3]})
    return {"t": "ul", "items": out, "label": label, "cls": cls}


def STEPS(*items):
    return {"t": "steps", "items": [i if isinstance(i, str) else {"en": i[0], "zh": i[1]} for i in items]}


def MATH(tex, cls=""):
    return {"t": "math", "tex": tex, "cls": cls}


def CALL(kind, en, zh):
    return {"t": "callout", "kind": kind, "en": en, "zh": zh}


def QUOTE(en, zh):
    return {"t": "quote", "en": en, "zh": zh, "label": True}


def TABLE(head, rows, num_cols=(), hl_rows=(), cls="", style=""):
    return {"t": "table", "head": head, "rows": rows, "num_cols": list(num_cols),
            "hl_rows": list(hl_rows), "cls": cls, "style": style}


def TERMS(rows, head=None, cls=""):
    return {"t": "terms", "rows": rows, "head": head, "cls": cls}


def SVG(fn, caption=None, style=""):
    return {"t": "svg", "fn": fn, "caption": caption, "style": style}


def COLS(*cols, ratio=None, cls=""):
    return {"t": "cols", "cols": list(cols), "ratio": ratio, "cls": cls}


def CARDS(*items, n=None):
    return {"t": "cards", "items": list(items), "n": n or len(items)}


def CARD(title_en, title_zh, *blocks, cls=""):
    return {"title": T(title_en, title_zh), "blocks": list(blocks), "cls": cls}


def KV(*rows):
    return {"t": "kv", "rows": list(rows)}


def CALLLINE(text):
    return {"t": "callline", "text": text}


def CODE(text):
    return {"t": "code", "text": text}


def FLOW(*lanes):
    return {"t": "flow", "lanes": list(lanes)}


def LANE(title_en, title_zh, *steps, per=3):
    return {"title": T(title_en, title_zh), "per": per,
            "steps": [{"en": s[0], "zh": s[1], "hl": (len(s) > 2 and s[2])} for s in steps]}


def HTML(html):
    return {"t": "html", "html": html}


def slide(id, section, title_en, title_zh, blocks, source=None, status=(), layout=None):
    s = {"id": id, "section": section, "title": T(title_en, title_zh), "blocks": blocks,
         "status": list(status)}
    if source:
        s["source"] = T(*source) if isinstance(source, tuple) else source
    if layout:
        s["layout"] = layout
    return s

#!/usr/bin/env python3
"""Build the bilingual slide deck 'Tool-Using LLM Agents'.

Usage:
    npm ci                      # installs pinned KaTeX and font packages
    python3 build.py            # writes dist/ and evidence/computed_values.json
    python3 tools/lint.py       # language and consistency checks
    node tools/check_layout.mjs # overflow check in both languages (Playwright)

Outputs:
    dist/tool-using-llm-agents.html           standalone, fully self-contained
    dist/tool-using-llm-agents.artifact.html  same page without the outer
                                              <html>/<head>/<body> skeleton
"""

from __future__ import annotations

import base64
import html
import importlib
import json
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from deck import diagrams  # noqa: E402
from deck.numbers import VALUES  # noqa: E402

DIST = ROOT / "dist"
NM = ROOT / "node_modules"
TITLE = "Tool-Using LLM Agents"
METHOD_NAME = "QET"  # the draft's acronym; never expanded in the slides

MACROS = {
    r"\Reduce": r"\operatorname{Reduce}",
    r"\Render": r"\operatorname{Render}",
    r"\Exec": r"\operatorname{Exec}",
    r"\LinExt": r"\operatorname{LinExt}",
    r"\State": r"\operatorname{State}",
    r"\pa": r"\operatorname{pa}",
    r"\sg": r"\operatorname{sg}",
    r"\KL": r"D_{\mathrm{KL}}",
    r"\aend": r"a_{\mathrm{end}}^{\star}",
    r"\Lroot": r"\mathcal{L}_{\mathrm{root}}",
    r"\LQET": r"\mathcal{L}_{\mathrm{%s}}" % METHOD_NAME,
    r"\LSFT": r"\mathcal{L}_{\mathrm{SFT}}",
    r"\Laug": r"\mathcal{L}_{\mathrm{aug}}",
    r"\CanExecute": r"\operatorname{CanExecute}",
    r"\Valid": r"\operatorname{Valid}",
}

SECTIONS = {
    "intro": ("Overview", "概览"),
    "mot": ("Motivation", "研究动机"),
    "setup": ("Problem setup", "问题设定"),
    "graph": ("Execution graph", "执行图"),
    "state": ("Execution state", "执行状态"),
    "dp": ("Shared-state computation", "共享状态计算"),
    "obj": ("Training objective", "训练目标"),
    "alg": ("Algorithms", "算法流程"),
    "pos": ("Positioning and status", "定位与现状"),
    "end": ("Summary", "总结"),
    "app": ("Appendix", "附录"),
}

CHIPS = {
    "draft": ("From the v2 draft", "来自 v2 草稿"),
    "draft1": ("From the earlier draft", "来自较早草稿"),
    "est": ("Established result", "已有结论"),
    "hyp": ("Hypothesis", "假设"),
    "ill": ("Illustrative numbers", "示意数值"),
    "derived": ("Derived computation", "推导计算"),
}

CALLOUT_TAGS = {
    "hyp": ("Hypothesis.", "假设。"),
    "est": ("Established.", "已有结论。"),
    "draft": ("From the v2 draft.", "来自 v2 草稿。"),
    "draft1": ("From the earlier draft.", "来自较早草稿。"),
    "note": ("Note.", "说明。"),
    "warn": ("Caveat.", "注意。"),
    "syn": ("Interpretation.", "解读。"),
    "corr": ("Correction.", "更正。"),
}

# ---------------------------------------------------------------------------
# Loading content
# ---------------------------------------------------------------------------


def load_slides():
    import deck.content as pkg

    slides = []
    for mod in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda m: m.name):
        m = importlib.import_module(f"deck.content.{mod.name}")
        for s in m.SLIDES:
            s.setdefault("module", mod.name)
            slides.append(s)
    ids = [s["id"] for s in slides]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise SystemExit(f"duplicate slide ids: {dup}")
    return substitute(slides)


_PH = re.compile(r"@@([A-Za-z0-9_]+)@@")


def substitute(obj):
    if isinstance(obj, str):
        def rep(m):
            k = m.group(1)
            if k not in VALUES:
                raise SystemExit(f"unknown value placeholder @@{k}@@")
            return VALUES[k]
        return _PH.sub(rep, obj.replace("@@METHOD@@", METHOD_NAME))
    if isinstance(obj, list):
        return [substitute(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(substitute(x) for x in obj)
    if isinstance(obj, dict):
        return {k: substitute(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Math extraction: $...$ inline and $$...$$ display, rendered in one node call
# ---------------------------------------------------------------------------


class Math:
    def __init__(self):
        self.items: list[dict] = []

    def token(self, tex, display):
        self.items.append({"tex": tex.strip(), "display": display})
        return f"\x00M{len(self.items) - 1}\x00"

    def text(self, s: str) -> str:
        """Replace math segments in prose with tokens."""
        s = re.sub(r"\$\$(.+?)\$\$", lambda m: self.token(m.group(1), True), s, flags=re.S)
        s = re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$", lambda m: self.token(m.group(1), False), s, flags=re.S)
        return s

    def render(self):
        payload = json.dumps({"macros": MACROS, "items": self.items})
        res = subprocess.run(["node", str(ROOT / "tools" / "render_math.mjs")], input=payload.encode(),
                             capture_output=True, cwd=ROOT)
        if res.returncode != 0:
            sys.stderr.write(res.stderr.decode())
            raise SystemExit("math rendering failed")
        return json.loads(res.stdout)


M = Math()

# ---------------------------------------------------------------------------
# Block renderers
# ---------------------------------------------------------------------------


def bi(obj, tag="span", cls="", inline=True):
    """Bilingual element: obj is {'en','zh'} or a shared string."""
    c = f" {cls}" if cls else ""
    if isinstance(obj, str):
        return f'<{tag} class="{cls}">{M.text(obj)}</{tag}>' if cls or tag != "span" else M.text(obj)
    return (f'<{tag} class="l-en{c}">{M.text(obj["en"])}</{tag}>'
            f'<{tag} class="l-zh{c}" lang="zh-CN">{M.text(obj["zh"])}</{tag}>')


def cell(c):
    if isinstance(c, dict) and "en" in c:
        return bi(c)
    return M.text(str(c))


def r_p(b):
    return bi(b, "p", b.get("cls", ""))


def r_ul(b):
    tag = "ol" if b.get("ordered") else "ul"
    lis = []
    for it in b["items"]:
        lead_en = f'<span class="lead">{M.text(it["lead"]["en"])}</span> ' if "lead" in it else ""
        lead_zh = f'<span class="lead">{M.text(it["lead"]["zh"])}</span>' if "lead" in it else ""
        lis.append(f'<li class="l-en">{lead_en}{M.text(it["en"])}</li>'
                   f'<li class="l-zh" lang="zh-CN">{lead_zh}{M.text(it["zh"])}</li>')
    return f'<{tag} class="bl {b.get("cls", "")}">{"".join(lis)}</{tag}>'


def r_steps(b):
    lis = []
    for it in b["items"]:
        if isinstance(it, str):
            lis.append(f"<li>{M.text(it)}</li>")
        else:
            lis.append(f'<li>{bi(it, "span")}</li>')
    return f'<ol class="steps">{"".join(lis)}</ol>'


def r_math(b):
    return f'<div class="mblock {b.get("cls", "")}">{M.token(b["tex"], True)}</div>'


def r_table(b):
    head = ""
    if b.get("head"):
        ths = []
        for i, h in enumerate(b["head"]):
            num = " class=\"num\"" if i in b.get("num_cols", []) else ""
            ths.append(f"<th{num}>{cell(h)}</th>")
        head = f"<thead><tr>{''.join(ths)}</tr></thead>"
    rows = []
    for ri, r in enumerate(b["rows"]):
        tds = []
        for i, c in enumerate(r):
            num = " class=\"num\"" if i in b.get("num_cols", []) else ""
            tds.append(f"<td{num}>{cell(c)}</td>")
        hl = ' class="hl"' if ri in b.get("hl_rows", []) else ""
        rows.append(f"<tr{hl}>{''.join(tds)}</tr>")
    style = f' style="{b["style"]}"' if b.get("style") else ""
    return f'<div class="tblwrap"><table class="tbl {b.get("cls", "")}"{style}>{head}<tbody>{"".join(rows)}</tbody></table></div>'


def r_terms(b):
    rows = [[f"${sym}$", meaning] for sym, meaning in b["rows"]]
    return r_table({"head": b.get("head"), "rows": rows, "cls": "terms " + b.get("cls", "")})


def r_callout(b):
    k = b["kind"]
    ten, tzh = CALLOUT_TAGS[k]
    return (f'<div class="callout {k}"><p class="l-en"><span class="tag">{ten}</span>{M.text(b["en"])}</p>'
            f'<p class="l-zh" lang="zh-CN"><span class="tag">{tzh}</span>{M.text(b["zh"])}</p></div>')


def r_quote(b):
    return bi(b, "p", "quote")


def r_code(b):
    if "en" in b:
        return (f'<pre class="code l-en">{html.escape(b["en"])}</pre>'
                f'<pre class="code l-zh" lang="zh-CN">{html.escape(b["zh"])}</pre>')
    return f'<pre class="code">{html.escape(b["text"])}</pre>'


def r_svg(b):
    svg = diagrams.render(b["fn"], VALUES)
    cap = ""
    if b.get("caption"):
        cap = f'<div class="figcap">{bi(b["caption"])}</div>'
    style = f' style="{b["style"]}"' if b.get("style") else ""
    return f'<div class="fig"{style}>{svg}{cap}</div>'


def r_cols(b):
    style = f' style="--cols:{b["ratio"]}"' if b.get("ratio") else ""
    inner = "".join(f"<div>{render_blocks(col)}</div>" for col in b["cols"])
    return f'<div class="cols {b.get("cls", "")}"{style}>{inner}</div>'


def r_cards(b):
    out = []
    for c in b["items"]:
        out.append(f'<div class="card {c.get("cls", "")}"><h3>{bi(c["title"])}</h3>{render_blocks(c["blocks"])}</div>')
    return f'<div class="cards" style="--n:{b.get("n", len(b["items"]))}">{"".join(out)}</div>'


def r_kv(b):
    out = []
    for k, v in b["rows"]:
        out.append(f'<span class="k">{cell(k)}</span><span>{cell(v)}</span>')
    return f'<div class="kv">{"".join(out)}</div>'


def r_callline(b):
    return f'<div class="callline">{html.escape(b["text"])} <span class="x">×</span></div>'


def r_flow(b):
    lanes = []
    for lane in b["lanes"]:
        steps = []
        for s in lane["steps"]:
            hl = " hl" if s.get("hl") else ""
            steps.append(f'<div class="fstep{hl}"><span>{bi(s)}</span></div>')
        per = lane.get("per", 3)
        lanes.append(f'<div class="lane"><div class="lane-title">{bi(lane["title"])}</div>'
                     f'<div class="lane-steps" style="--per:{per}">{"".join(steps)}</div></div>')
    return f'<div class="flow">{"".join(lanes)}</div>'


def r_html(b):
    return b["html"]


RENDER = {
    "p": r_p, "ul": r_ul, "steps": r_steps, "math": r_math, "table": r_table, "terms": r_terms,
    "callout": r_callout, "quote": r_quote, "code": r_code, "svg": r_svg, "cols": r_cols,
    "cards": r_cards, "kv": r_kv, "callline": r_callline, "flow": r_flow, "html": r_html,
}


def render_blocks(blocks):
    return "".join(RENDER[b["t"]](b) for b in blocks)


def render_slide(s, i, n):
    sec_en, sec_zh = SECTIONS[s["section"]]
    chips = "".join(f'<span class="chip {c}">{bi({"en": CHIPS[c][0], "zh": CHIPS[c][1]})}</span>'
                    for c in s.get("status", []))
    if s.get("layout") == "title":
        body = s["html_body"](M) if callable(s.get("html_body")) else render_blocks(s["blocks"])
        head = ""
        cls = "slide title-slide"
    else:
        body = render_blocks(s["blocks"])
        head = (f'<header class="s-head"><div class="eyebrow"><span>{bi({"en": sec_en, "zh": sec_zh})}</span>'
                f'<span class="chips">{chips}</span></div>'
                f'<h2 class="s-title">{bi(s["title"])}</h2></header>')
        cls = "slide"
    src = bi(s["source"]) if s.get("source") else ""
    foot = f'<footer class="s-foot"><span class="src">{src}</span><span class="num">{i + 1} / {n}</span></footer>'
    return (f'<section class="{cls}" id="s{i + 1}" data-id="{s["id"]}" '
            f'data-sec-en="{html.escape(sec_en)}" data-sec-zh="{html.escape(sec_zh)}">'
            f'{head}<div class="s-body"><div class="s-fit">{body}</div></div>{foot}</section>')


# ---------------------------------------------------------------------------
# Assets: fonts and KaTeX CSS as data URIs
# ---------------------------------------------------------------------------


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def font_css():
    faces = [
        ("Source Serif 4", "@fontsource/source-serif-4/files/source-serif-4-latin-400-normal.woff2", 400, "normal"),
        ("Source Serif 4", "@fontsource/source-serif-4/files/source-serif-4-latin-600-normal.woff2", 600, "normal"),
        ("Source Serif 4", "@fontsource/source-serif-4/files/source-serif-4-latin-400-italic.woff2", 400, "italic"),
        ("Source Sans 3", "@fontsource/source-sans-3/files/source-sans-3-latin-400-normal.woff2", 400, "normal"),
        ("Source Sans 3", "@fontsource/source-sans-3/files/source-sans-3-latin-400-italic.woff2", 400, "italic"),
        ("Source Sans 3", "@fontsource/source-sans-3/files/source-sans-3-latin-600-normal.woff2", 600, "normal"),
        ("JetBrains Mono", "@fontsource/jetbrains-mono/files/jetbrains-mono-latin-400-normal.woff2", 400, "normal"),
    ]
    out = []
    for fam, rel, wt, style in faces:
        p = NM / rel
        out.append(f"@font-face{{font-family:'{fam}';font-style:{style};font-weight:{wt};font-display:swap;"
                   f"src:url(data:font/woff2;base64,{b64(p)}) format('woff2');"
                   f"unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,"
                   f"U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD;}}")
    return "\n".join(out)


def katex_css():
    css = (NM / "katex/dist/katex.min.css").read_text()
    fonts = NM / "katex/dist/fonts"

    def rep(m):
        name = m.group(1)
        return f"src:url(data:font/woff2;base64,{b64(fonts / (name + '.woff2'))}) format(\"woff2\")"

    css, n = re.subn(r"src:url\(fonts/(KaTeX_[A-Za-z0-9_-]+)\.woff2\) format\(\"woff2\"\)"
                     r"(?:,url\(fonts/[^)]+\) format\(\"[a-z]+\"\))*", rep, css)
    if n < 15:
        raise SystemExit(f"unexpected KaTeX CSS format ({n} font faces replaced)")
    return css


# ---------------------------------------------------------------------------


def toolbar():
    return """
<div class="progress" aria-hidden="true"></div>
<nav class="bar" aria-label="Slide controls">
  <div class="grp">
    <button id="prev" type="button" aria-label="Previous slide">&#8592;</button>
    <span id="count" class="count" aria-live="polite">1 / 1</span>
    <button id="next" type="button" aria-label="Next slide">&#8594;</button>
    <span id="sec" class="sec"></span>
  </div>
  <div class="grp">
    <span class="hint"><span class="l-en">Keys: &#8592; &#8594; and L</span><span class="l-zh" lang="zh-CN">按键：&#8592; &#8594; 与 L</span></span>
    <button id="lang-en" type="button" aria-pressed="true">EN</button>
    <button id="lang-zh" type="button" aria-pressed="false" lang="zh-CN">中文</button>
    <button id="theme" type="button" data-mode="auto"><span class="l-en">Theme: auto</span><span class="l-zh" lang="zh-CN">主题：自动</span></button>
  </div>
</nav>"""


def build():
    slides = load_slides()
    n = len(slides)
    sections_html = "\n".join(render_slide(s, i, n) for i, s in enumerate(slides))
    rendered = M.render()
    body_html = re.sub("\x00M(\\d+)\x00", lambda m: rendered[int(m.group(1))], sections_html)
    if "\x00" in body_html:
        raise SystemExit("unreplaced math token")

    css = (ROOT / "deck/theme.css").read_text()
    js = (ROOT / "deck/runtime.js").read_text()
    head = (f"<title>{TITLE}</title>\n"
            f'<meta name="description" content="Bilingual research slides on training tool-using LLM agents '
            f'with dependency structure, canonical execution state and dynamic programming over completed sets.">\n'
            f"<style>\n{css}\n</style>\n"
            f"<style>\n{font_css()}\n</style>\n<style>\n{katex_css()}\n</style>\n")
    page = (f'<div class="app"><main class="viewport"><div class="deck" id="deck">\n{body_html}\n'
            f"</div></main>{toolbar()}</div>\n<script>\n{js}\n</script>\n")

    DIST.mkdir(exist_ok=True)
    standalone = ("<!doctype html>\n<html lang=\"en\" data-lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                  "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
                  "<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n"
                  f"{head}</head>\n<body>\n{page}</body>\n</html>\n")
    (DIST / "tool-using-llm-agents.html").write_text(standalone, encoding="utf-8")
    (DIST / "tool-using-llm-agents.artifact.html").write_text(head + page, encoding="utf-8")

    ev = ROOT / "evidence"
    ev.mkdir(exist_ok=True)
    (ev / "computed_values.json").write_text(json.dumps(VALUES, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    index = [{"n": i + 1, "id": s["id"], "section": s["section"], "title_en": s["title"]["en"],
              "title_zh": s["title"]["zh"], "status": s.get("status", []), "module": s["module"]}
             for i, s in enumerate(slides)]
    (ev / "slide_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    size = (DIST / "tool-using-llm-agents.html").stat().st_size
    print(f"built {n} slides, {len(M.items)} math expressions, {size / 1024:.0f} KiB")


if __name__ == "__main__":
    build()

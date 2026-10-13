#!/usr/bin/env python3
"""Language and consistency checks for the deck content.

Errors fail the run (exit 1); warnings are printed for review.
  E1 banned words: 'quotient' (any case), the Chinese term for it, buzzwords
  E2 em dashes; question marks outside code; 'Next:' pointers
  E3 English prose sentence longer than 35 words; Chinese sentence longer than 80 units
  E4 '(my synthesis)' more than once per slide, or different counts in EN and ZH
  E5 math segments differ between EN and ZH of one item
  E6 numbers outside math differ between EN and ZH of one item
  W1 English prose sentence outside 12-25 words; Chinese sentence above 60 units
Run after build.py; it also scans the built HTML for banned words.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build import load_slides  # noqa: E402

BANNED_EN = [r"quotient", r"leverag", r"unlock", r"seamless", r"cutting-edge", r"game-chang", r"revolutioni",
             r"\bdelve", r"paradigm shift", r"holistic", r"synerg", r"empower", r"\bcrucial", r"\bnotably",
             r"\bimportantly", r"worth noting", r"in conclusion", r"\bnext:", r"groundbreaking", r"state-of-the-art"]
BANNED_ZH = ["商", "赋能", "助力", "打造", "全方位", "值得注意的是", "不难看出", "总而言之", "至关重要", "颠覆",
             "革命性", "范式转变", "无缝", "深度融合", "显著提升", "极大地", "一站式", "闭环", "下一页", "接下来我们"]
SYN_EN, SYN_ZH = "(my synthesis)", "（个人归纳）"
MATH_RE = re.compile(r"\$\$.+?\$\$|(?<!\\)\$.+?(?<!\\)\$", re.S)
TAG_RE = re.compile(r"<[^>]+>")
NUM_RE = re.compile(r"(?<![A-Za-z_])\d+(?:[.,]\d+)*%?")

errors: list[str] = []
warnings: list[str] = []


def err(sid, msg):
    errors.append(f"[{sid}] {msg}")


def warn(sid, msg):
    warnings.append(f"[{sid}] {msg}")


def strip(s):
    return TAG_RE.sub("", s)


def maths(s):
    return Counter(re.sub(r"\s+", "", m) for m in MATH_RE.findall(s))


def numbers(s):
    s = MATH_RE.sub(" ", strip(s))
    return Counter(n.replace(",", "") for n in NUM_RE.findall(s))


def en_sentences(s):
    s = MATH_RE.sub(" X ", strip(s))
    s = re.sub(r"\b(e\.g|i\.e|et al|Eq|Sec|vs)\.", r"\1", s)
    parts = re.split(r"(?<=[.!])\s+(?=[A-Z$X(])", s.strip())
    return [p for p in parts if p.strip()]


def word_count(s):
    return len([w for w in re.split(r"\s+", s) if re.search(r"[A-Za-z0-9X]", w)])


def zh_sentences(s):
    s = MATH_RE.sub("X", strip(s))
    return [p for p in re.split(r"(?<=[。！；])", s) if p.strip()]


def zh_units(s):
    cjk = len(re.findall(r"[一-鿿]", s))
    latin = len(re.findall(r"[A-Za-z0-9X]+", s))
    return cjk + latin


def check_text(sid, en, zh, prose):
    for pat in BANNED_EN:
        if re.search(pat, en, re.I):
            err(sid, f"E1 banned English term /{pat}/ in: {en[:70]}")
    for w in BANNED_ZH:
        if w in zh:
            err(sid, f"E1 banned Chinese term '{w}' in: {zh[:40]}")
    for s, lang in ((en, "en"), (zh, "zh")):
        if "—" in strip(s):
            err(sid, f"E2 em dash ({lang}): {s[:60]}")
        if re.search(r"[?？]", MATH_RE.sub("", strip(s))):
            err(sid, f"E2 question mark ({lang}): {s[:60]}")
    if maths(en) != maths(zh):
        err(sid, f"E5 math differs:\n    en={sorted(maths(en))}\n    zh={sorted(maths(zh))}")
    if numbers(en) != numbers(zh):
        err(sid, f"E6 numbers differ: en={dict(numbers(en))} zh={dict(numbers(zh))} :: {strip(en)[:60]}")
    if prose:
        for sent in en_sentences(en):
            n = word_count(sent)
            if n > 35:
                err(sid, f"E3 {n} words: {sent}")
            elif n > 25 or n < 12:
                warn(sid, f"W1 {n} words: {sent}")
        for sent in zh_sentences(zh):
            u = zh_units(sent)
            if u > 80:
                err(sid, f"E3 zh {u} units: {sent}")
            elif u > 60:
                warn(sid, f"W1 zh {u} units: {sent}")
    else:
        for sent in en_sentences(en):
            if word_count(sent) > 35:
                err(sid, f"E3 label sentence over 35 words: {sent}")


def walk(sid, blocks, items):
    for b in blocks:
        t = b["t"]
        label = b.get("label", False)
        if t in ("p", "quote"):
            items.append((b["en"], b["zh"], not label and t == "p"))
        elif t == "callout":
            items.append((b["en"], b["zh"], True))
        elif t == "ul":
            for it in b["items"]:
                items.append((it["en"], it["zh"], not label))
                if "lead" in it:
                    items.append((it["lead"]["en"], it["lead"]["zh"], False))
        elif t == "steps":
            for it in b["items"]:
                if isinstance(it, dict):
                    items.append((it["en"], it["zh"], False))
                else:
                    items.append((it, it, False))
        elif t in ("table", "terms"):
            rows = b["rows"] if t == "table" else [[f"${s}$", m] for s, m in b["rows"]]
            for row in ([b["head"]] if b.get("head") else []) + rows:
                for c in row:
                    if isinstance(c, dict):
                        items.append((c["en"], c["zh"], False))
                    else:
                        items.append((str(c), str(c), False))
        elif t == "cols":
            for col in b["cols"]:
                walk(sid, col, items)
        elif t == "cards":
            for c in b["items"]:
                items.append((c["title"]["en"], c["title"]["zh"], False))
                walk(sid, c["blocks"], items)
        elif t == "kv":
            for k, v in b["rows"]:
                for c in (k, v):
                    items.append((c["en"], c["zh"], False) if isinstance(c, dict) else (c, c, False))
        elif t == "flow":
            for lane in b["lanes"]:
                items.append((lane["title"]["en"], lane["title"]["zh"], False))
                for s in lane["steps"]:
                    items.append((s["en"], s["zh"], False))
        elif t in ("code", "callline"):
            txt = b.get("text", "")
            if re.search("quotient", txt, re.I):
                err(sid, "E1 banned term in code")
        elif t == "html":
            h = b["html"]
            if re.search("quotient", h, re.I) or "—" in h:
                err(sid, "E1/E2 in raw HTML block")
            en = " ".join(re.findall(r'class="l-en[^"]*">(.*?)</', h))
            zh = " ".join(re.findall(r'class="l-zh[^"]*" lang="zh-CN">(.*?)</', h))
            if en or zh:
                items.append((en, zh, False))


def main():
    slides = load_slides()
    for s in slides:
        sid = s["id"]
        items = []
        t = s["title"]
        if re.search(r"[?？]", t["en"] + t["zh"]):
            err(sid, "E2 question in title")
        items.append((t["en"], t["zh"], False))
        if s.get("source"):
            items.append((s["source"]["en"], s["source"]["zh"], False))
        walk(sid, s["blocks"], items)
        syn_en = sum(en.count(SYN_EN) for en, _, _ in items)
        syn_zh = sum(zh.count(SYN_ZH) for _, zh, _ in items)
        if syn_en > 1 or syn_zh > 1 or syn_en != syn_zh:
            err(sid, f"E4 synthesis markers en={syn_en} zh={syn_zh}")
        for en, zh, prose in items:
            check_text(sid, en, zh, prose)

    html_path = ROOT / "dist" / "tool-using-llm-agents.html"
    if html_path.exists():
        body = html_path.read_text(encoding="utf-8")
        body = body[body.find('<div class="app">'):]
        visible = TAG_RE.sub(" ", re.sub(r"<script.*?</script>", " ", body, flags=re.S))
        if re.search("quotient", visible, re.I):
            err("dist", "E1 'quotient' appears in built HTML")
        if "—" in visible:
            err("dist", "E2 em dash in built HTML")
    else:
        warn("dist", "built HTML not found; run build.py first")

    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print(f"{len(slides)} slides checked: {len(errors)} errors, {len(warnings)} warnings")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()

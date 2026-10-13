"""Inline SVG diagrams. Colors come from CSS tokens (see theme.css):
path1 = order A then B, path2 = order B then A, merge = shared state, bad = error.
Bilingual labels are emitted twice with classes l-en / l-zh; CSS shows one.
"""

from __future__ import annotations

from html import escape

_uid = [0]


def _id(prefix):
    _uid[0] += 1
    return f"{prefix}{_uid[0]}"


def _defs(mid):
    out = ["<defs>"]
    for name in ("ink", "p1", "p2", "mg", "bad", "mut"):
        out.append(
            f'<marker id="{mid}-{name}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" class="ah ah-{name}"/></marker>'
        )
    out.append("</defs>")
    return "".join(out)


def _svg(w, h, body, label_en, label_zh, cls=""):
    return (
        f'<svg class="dg {cls}" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
        f'aria-label="{escape(label_en)}">{body}</svg>'
    )


def t(x, y, s, cls="", anchor="middle", zh=None, size=None):
    """Text element. If zh is given, emit a bilingual pair."""
    st = f' style="font-size:{size}px"' if size else ""
    if zh is None:
        return f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls}"{st}>{s}</text>'
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls} l-en"{st}>{s}</text>'
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls} l-zh" lang="zh-CN"{st}>{zh}</text>'
    )


def line(x1, y1, x2, y2, cls, mid, mk="ink", dash=False):
    d = ' stroke-dasharray="6 5"' if dash else ""
    m = f' marker-end="url(#{mid}-{mk})"' if mk else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="edge {cls}"{d}{m}/>'


def path(d, cls, mid, mk="ink", dash=False):
    da = ' stroke-dasharray="6 5"' if dash else ""
    m = f' marker-end="url(#{mid}-{mk})"' if mk else ""
    return f'<path d="{d}" class="edge {cls}"{da}{m}/>'


def box(x, y, w, h, cls="node", r=8):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" class="{cls}"/>'


def circle(cx, cy, r, cls="node"):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" class="{cls}"/>'


# ---------------------------------------------------------------------------


def dag_abc():
    """Slide: running example. DAG a,b -> c and its two linear extensions."""
    mid = _id("dabc")
    b = [_defs(mid)]
    # DAG
    b.append(circle(70, 60, 24))
    b.append(t(70, 67, "a", "mth big"))
    b.append(circle(70, 170, 24))
    b.append(t(70, 177, "b", "mth big"))
    b.append(circle(230, 115, 24))
    b.append(t(230, 122, "c", "mth big"))
    b.append(line(94, 67, 205, 106, "", mid))
    b.append(line(94, 163, 205, 124, "", mid))
    b.append(t(70, 120, "no edge", "small mut", zh="无边"))
    b.append(t(150, 225, "dependency graph", "cap", zh="依赖图"))
    # two orders
    x0, y1, y2 = 320, 70, 160
    for y, seq, cls, mk in ((y1, "abc", "p1", "p1"), (y2, "bac", "p2", "p2")):
        for i, ch in enumerate(seq):
            x = x0 + i * 80
            b.append(box(x, y - 24, 48, 48, f"node {cls}f"))
            b.append(t(x + 24, y + 7, ch, "mth big"))
            if i < 2:
                b.append(line(x + 50, y, x + 78, y, cls, mid, mk))
    b.append(t(x0 + 104, 225, "two valid executions", "cap", zh="两种合法执行"))
    return _svg(560, 245, "".join(b), "Dependency graph a, b to c and its two valid orders", "")


def dag_flight():
    """Slide: target execution graph for the flight/hotel task."""
    mid = _id("dfl")
    b = [_defs(mid)]
    nodes = [("A", 10, 30, "get_flight_price", "flight_price = 1200", "p1f"),
             ("B", 10, 170, "get_hotel_price", "hotel_price = 600", "p2f"),
             ("C", 320, 100, "calculate_total", "total", "mgf")]
    for name, x, y, tool, out, cls in nodes:
        b.append(box(x, y, 232, 70, f"node {cls}", r=8))
        b.append(t(x + 14, y + 28, name, "mth big", anchor="start"))
        b.append(t(x + 44, y + 27, tool, "mono", anchor="start"))
        b.append(t(x + 44, y + 52, out, "mono mut", anchor="start"))
    b.append(line(244, 65, 316, 122, "", mid))
    b.append(line(244, 205, 316, 150, "", mid))
    b.append(t(282, 270, "E = {(A, C), (B, C)}; no edge between A and B", "cap",
               zh="E = {(A, C), (B, C)}；A 与 B 之间无边"))
    return _svg(564, 285, "".join(b), "Target execution graph for the flight and hotel task", "")


def seq_target():
    """Slide: a sequence target encodes required and arbitrary order alike."""
    mid = _id("seq")
    b = [_defs(mid)]
    xs = [60, 230, 400]
    for x, ch in zip(xs, "abc"):
        b.append(box(x, 150, 70, 56, "node"))
        b.append(t(x + 35, 186, ch, "mth big"))
    # arrows along the sequence
    b.append(line(132, 178, 226, 178, "mut", mid, "mut"))
    b.append(line(302, 178, 396, 178, "mut", mid, "mut"))
    # arcs: required a->c (above), b->c (above small), arbitrary a->b (below dashed)
    b.append(path("M95,148 C150,40 380,40 432,146", "mg", mid, "mg"))
    b.append(path("M265,148 C300,105 395,105 425,146", "mg", mid, "mg"))
    b.append(path("M95,208 C130,275 230,275 262,210", "p2", mid, "p2", dash=True))
    b.append(t(265, 62, "required: a before c, b before c", "small mg-t", zh="必要：a 先于 c，b 先于 c"))
    b.append(t(178, 283, "arbitrary: a before b", "small p2-t", zh="偶然：a 先于 b"))
    b.append(t(265, 312, "target sequence (a, b, c)", "cap", zh="目标序列 (a, b, c)"))
    return _svg(540, 325, "".join(b), "Required and arbitrary orderings in one target sequence", "")


def two_histories():
    """Slide: two histories, one canonical state."""
    mid = _id("hist")
    b = [_defs(mid)]
    rows = [
        (40, ["A", "o_A", "B", "o_B"], "p1", "h₁"),
        (150, ["B", "o_B", "A", "o_A"], "p2", "h₂"),
    ]
    for y, toks, cls, name in rows:
        b.append(t(28, y + 30, name, "mth", anchor="middle"))
        for i, tok in enumerate(toks):
            x = 52 + i * 66
            is_obs = tok.startswith("o_")
            b.append(box(x, y + 6, 58, 40, f"node {cls}f" if not is_obs else "node obs", r=6))
            label = tok if not is_obs else f"o<tspan baseline-shift='sub' font-size='12'>{tok[2]}</tspan>"
            b.append(t(x + 29, y + 32, label, "mth"))
    # arrows into state box
    b.append(path("M318,66 C360,66 370,110 404,118", "p1", mid, "p1"))
    b.append(path("M318,176 C360,176 370,140 404,132", "p2", mid, "p2"))
    b.append(t(360, 112, "Reduce", "small mut", anchor="middle"))
    # state card
    b.append(box(408, 58, 210, 140, "node mgf", r=10))
    b.append(t(424, 86, "Completed: {A, B}", "small", anchor="start", zh="已完成：{A, B}"))
    b.append(t(424, 114, "flight_price = 1200", "mono", anchor="start"))
    b.append(t(424, 140, "hotel_price = 600", "mono", anchor="start"))
    b.append(t(424, 172, "same state Hₜ", "small mg-t", anchor="start", zh="同一状态 Hₜ"))
    b.append(t(330, 236, "different token orders, identical task progress", "cap",
               zh="token 顺序不同，任务进度相同"))
    return _svg(640, 250, "".join(b), "Two histories reduced to one canonical state", "")


# ---------------------------------------------------------------------------
# Lattice of completed sets for the A, B -> C example

_LAT_POS = {  # centre of each node
    "e": (44, 165),
    "A": (190, 66),
    "B": (190, 264),
    "AB": (336, 165),
    "ABC": (506, 165),
}
_LAT_W = {"e": 54, "A": 66, "B": 66, "AB": 84, "ABC": 104}
_LAT_H = 40
_LAT_LABEL = {"e": "∅", "A": "{A}", "B": "{B}", "AB": "{A, B}", "ABC": "{A, B, C}"}
_LAT_EDGES = [  # (from, to, action, class, side)  side: -1 label above the edge, +1 below
    ("e", "A", "A", "p1", -1),
    ("A", "AB", "B", "p1", -1),
    ("e", "B", "B", "p2", 1),
    ("B", "AB", "A", "p2", 1),
    ("AB", "ABC", "C", "mg", -1),
]


def _border(k, toward):
    """Point where the segment from node k's centre toward `toward` leaves its box."""
    (cx, cy), (tx, ty) = _LAT_POS[k], toward
    hw, hh = _LAT_W[k] / 2 + 3, _LAT_H / 2 + 3
    dx, dy = tx - cx, ty - cy
    sx = hw / abs(dx) if dx else 1e9
    sy = hh / abs(dy) if dy else 1e9
    f = min(sx, sy)
    return cx + dx * f, cy + dy * f


def lattice(edge_labels=None, node_notes=None, show_invalid=False, caption=None):
    """Hasse diagram of order ideals. edge_labels: {(from,to): str};
    node_notes: {node: text} drawn outside the node (above {A}, below the others)."""
    mid = _id("lat")
    b = [_defs(mid)]
    for f, to, act, cls, side in _LAT_EDGES:
        x1, y1 = _border(f, _LAT_POS[to])
        x2, y2 = _border(to, _LAT_POS[f])
        b.append(line(round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1), cls, mid, cls))
        tp = 0.62 if f == "e" else 0.5   # keep labels on edges out of the root node's note
        mx, my = x1 + (x2 - x1) * tp, y1 + (y2 - y1) * tp
        dx, dy = x2 - x1, y2 - y1
        L = (dx * dx + dy * dy) ** 0.5
        nx, ny = -dy / L, dx / L          # unit normal
        if ny * side < 0:
            nx, ny = -nx, -ny               # make the normal point to the requested side
        off = 15
        lx, ly = mx + nx * off, my + ny * off + 5
        anchor = "middle"
        if abs(dy) > 1:
            anchor = "end" if nx < 0 else "start"
        lab = act if not edge_labels else edge_labels.get((f, to), act)
        b.append(t(round(lx, 1), round(ly, 1), lab, f"mth {cls}-t", anchor=anchor, size=17))
    for k, (x, y) in _LAT_POS.items():
        w = _LAT_W[k]
        cls = {"AB": "node mgf", "ABC": "node endf"}.get(k, "node")
        b.append(box(x - w / 2, y - _LAT_H / 2, w, _LAT_H, cls, r=20))
        b.append(t(x, y + 6, _LAT_LABEL[k], "mth", size=18))
        if node_notes and k in node_notes:
            txt, strong = node_notes[k]
            ny = y - _LAT_H / 2 - 10 if k == "A" else y + _LAT_H / 2 + 22
            b.append(t(x, ny, txt, "note strong" if strong else "note"))
    h = 322
    if show_invalid:
        x, y = 506, 264
        b.append(box(x - 32, y - 20, 64, 40, "node invalid", r=20))
        b.append(t(x, y + 6, "{C}", "mth mut", size=17))
        b.append(line(x - 22, y - 14, x + 22, y + 14, "bad", mid, None))
        b.append(t(x, y + 42, "not down-closed", "small bad-t", zh="不满足向下封闭"))
    if caption:
        h += 22
        b.append(t(283, h - 4, caption[0], "cap", zh=caption[1]))
    return _svg(566, h, "".join(b), "Lattice of completed sets for A, B before C", "")


def lattice_plain():
    return lattice(show_invalid=True,
                   caption=("valid completed sets 𝒥(G), ordered by inclusion",
                            "合法已完成集合 𝒥(G)，按包含关系排列"))


def _wlabels(v):
    return {("e", "A"): f"w = {v['wA']}", ("A", "AB"): f"w = {v['wAB']}",
            ("e", "B"): f"w = {v['wB']}", ("B", "AB"): f"w = {v['wBA']}",
            ("AB", "ABC"): f"w = {v['wC']}"}


def lattice_backward(v):
    notes = {"e": (f"B = {v['b0']}", True), "A": (f"B = {v['b_a']}", False),
             "B": (f"B = {v['b_b']}", False), "AB": (f"B = {v['b_ab']}", False),
             "ABC": (f"B = e = {v['eV']}", False)}
    return lattice(edge_labels=_wlabels(v), node_notes=notes)


def lattice_forward(v):
    notes = {"e": ("F = 1", False), "A": (f"F = {v['F_a']}", False), "B": (f"F = {v['F_b']}", False),
             "AB": (f"F = {v['F_ab']}", False), "ABC": (f"F = {v['F_v']}", True)}
    return lattice(edge_labels=_wlabels(v), node_notes=notes)


def per_path_vs_shared():
    """Slide: separate path scoring repeats the suffix; a shared state does not."""
    mid = _id("pps")
    b = [_defs(mid)]
    W, H = 80, 38

    def node(x, y, label, cls="node"):
        b.append(box(x - W / 2, y - H / 2, W, H, cls, r=19))
        b.append(t(x, y + 6, label, "mth", size=16))

    xs = [50, 170, 290, 410]
    for y, seq, cls in ((44, "AB", "p1"), (124, "BA", "p2")):
        labels = ["∅", "{" + seq[0] + "}", "{A, B}", "end"]
        for i, x in enumerate(xs):
            node(x, y, labels[i], "node endf" if i == 3 else "node")
        for i in range(3):
            act = [seq[0], seq[1], "C"][i]
            c = cls if i < 2 else "bad"
            b.append(line(xs[i] + W / 2 + 2, y, xs[i + 1] - W / 2 - 4, y, c, mid, c))
            b.append(t((xs[i] + xs[i + 1]) / 2, y - 8, act, f"mth {c}-t", size=16))
    b.append(t(230, 180, "per-path scoring: C is scored twice", "cap", zh="逐路径计分：C 被计分两次"))
    b.append(line(10, 200, 450, 200, "mut", mid, None, dash=True))
    pos = {"e": (50, 300), "A": (170, 240), "B": (170, 360), "AB": (290, 300), "end": (410, 300)}
    lab = {"e": "∅", "A": "{A}", "B": "{B}", "AB": "{A, B}", "end": "end"}
    for k, (x, y) in pos.items():
        node(x, y, lab[k], {"AB": "node mgf", "end": "node endf"}.get(k, "node"))
    for f, to, act, cls, lx, ly in (("e", "A", "A", "p1", 92, 258), ("A", "AB", "B", "p1", 248, 258),
                                    ("e", "B", "B", "p2", 92, 358), ("B", "AB", "A", "p2", 248, 358)):
        (x1, y1), (x2, y2) = pos[f], pos[to]
        sx, ex = x1 + (24 if f == "e" else 30), x2 - (30 if to == "AB" else 24)
        sy = y1 + (-H / 2 if y2 < y1 else H / 2) if f == "e" else y1 + (H / 2 if y2 > y1 else -H / 2)
        ey = y2 + (H / 2 if y2 < y1 else -H / 2) if to != "AB" else y2 + (-H / 2 + 6 if y2 > y1 else H / 2 - 6)
        b.append(line(sx, sy, ex, ey, cls, mid, cls))
        b.append(t(lx, ly, act, f"mth {cls}-t", size=16))
    b.append(line(pos["AB"][0] + W / 2 + 2, 300, pos["end"][0] - W / 2 - 4, 300, "mg", mid, "mg"))
    b.append(t(350, 292, "C", "mth mg-t", size=16))
    b.append(t(230, 410, "shared state: C is scored once", "cap", zh="共享状态：C 只计分一次"))
    return _svg(460, 420, "".join(b), "Per-path scoring versus shared-state scoring", "")


REGISTRY = {
    "dag_abc": dag_abc,
    "dag_flight": dag_flight,
    "seq_target": seq_target,
    "two_histories": two_histories,
    "lattice_plain": lattice_plain,
    "lattice_backward": lattice_backward,
    "lattice_forward": lattice_forward,
    "per_path_vs_shared": per_path_vs_shared,
}


def render(name, values):
    fn = REGISTRY[name]
    if name in ("lattice_backward", "lattice_forward"):
        return fn(values)
    return fn()

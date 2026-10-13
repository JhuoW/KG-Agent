"""Every number that appears on a slide is computed here.

Slides reference values with @@key@@ placeholders. build.py substitutes them,
so a number on a slide can never drift from its definition. The values are
also written to evidence/computed_values.json for auditing.

Sources of the input numbers:
  * Worked example policy scores: earlier draft (v1), Section 13.1.
  * Two-model example: v2 draft, "Class-marginal objective" table.
  * Order-collapsed policy: illustrative numbers chosen for this deck
    (marked "my synthesis" on the slide that uses them).
  * FuncBenchGen statistics: Maekawa et al., ICLR 2026, Table 3 and abstract.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction
from functools import lru_cache

# ---------------------------------------------------------------------------
# Generic machinery: a DAG, its order ideals, and the completion recursion.
# ---------------------------------------------------------------------------


def ideals(nodes, parents):
    """All down-closed subsets (order ideals) of a DAG, as frozensets."""
    out = []
    for r in range(len(nodes) + 1):
        for combo in itertools.combinations(nodes, r):
            s = frozenset(combo)
            if all(parents[v] <= s for v in s):
                out.append(s)
    return out


def ready(nodes, parents, done):
    return [v for v in nodes if v not in done and parents[v] <= done]


def completion(nodes, parents, w, e_end):
    """Backward recursion B(I) = sum_v w(I,v) B(I+{v}), B(V) = e(V)."""
    full = frozenset(nodes)

    @lru_cache(maxsize=None)
    def B(done):
        if done == full:
            return e_end
        return sum(w[(done, v)] * B(done | {v}) for v in ready(nodes, parents, done))

    return B


def forward(nodes, parents, w):
    """Forward recursion F(I+{v}) += F(I) w(I,v), F(empty) = 1."""
    order = sorted(ideals(nodes, parents), key=len)
    F = {s: 0.0 for s in order}
    F[frozenset()] = 1.0
    for s in order:
        for v in ready(nodes, parents, s):
            F[s | {v}] += F[s] * w[(s, v)]
    return F


def path_sum(nodes, parents, w, e_end):
    """Brute-force check: sum of path probabilities over all linear extensions."""
    total = 0.0
    for perm in itertools.permutations(nodes):
        done, p, ok = frozenset(), 1.0, True
        for v in perm:
            if not parents[v] <= done:
                ok = False
                break
            p *= w[(done, v)]
            done = done | {v}
        if ok:
            total += p * e_end
    return total


E = frozenset()
A, Bn, C = "A", "B", "C"
NODES = [A, Bn, C]
PARENTS = {A: frozenset(), Bn: frozenset(), C: frozenset({A, Bn})}
fs = frozenset


def scores(wA, wB, wAB, wBA, wC):
    return {
        (E, A): wA,
        (E, Bn): wB,
        (fs({A}), Bn): wAB,
        (fs({Bn}), A): wBA,
        (fs({A, Bn}), C): wC,
    }


# Worked example (earlier draft, Section 13.1)
W_EX = scores(0.60, 0.30, 0.50, 0.90, 0.80)
E_EX = 0.95
# Order-collapsed policy (illustrative, this deck)
W_COL = scores(0.85, 0.05, 0.90, 0.10, 0.80)
E_COL = 0.95


def fmt(x, nd):
    s = f"{x:.{nd}f}"
    return s


def thousands(n: int) -> str:
    return f"{n:,}"


def sci(x: float, nd=2) -> str:
    """Scientific notation as LaTeX, e.g. 2.43\\times10^{18}."""
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / 10**exp
    return f"{mant:.{nd}f}\\times10^{{{exp}}}"


def compute():
    N: dict[str, str] = {}
    raw: dict[str, float] = {}

    # ---- worked example -----------------------------------------------------
    B = completion(NODES, PARENTS, W_EX, E_EX)
    b_full, b_ab, b_a, b_b, b0 = B(fs(NODES)), B(fs({A, Bn})), B(fs({A})), B(fs({Bn})), B(E)
    p_abc = W_EX[(E, A)] * W_EX[(fs({A}), Bn)] * W_EX[(fs({A, Bn}), C)] * E_EX
    p_bac = W_EX[(E, Bn)] * W_EX[(fs({Bn}), A)] * W_EX[(fs({A, Bn}), C)] * E_EX
    assert abs(path_sum(NODES, PARENTS, W_EX, E_EX) - b0) < 1e-12
    assert abs(p_abc + p_bac - b0) < 1e-12
    F = forward(NODES, PARENTS, W_EX)
    assert abs(F[fs(NODES)] * E_EX - b0) < 1e-12

    raw.update(b_full=b_full, b_ab=b_ab, b_a=b_a, b_b=b_b, b0=b0)
    N.update(
        wA="0.60", wB="0.30", wAB="0.50", wBA="0.90", wC="0.80", eV="0.95",
        b_ab=fmt(b_ab, 2), b_a=fmt(b_a, 2), b_b=fmt(b_b, 3), b0=fmt(b0, 4),
        p_abc=fmt(p_abc, 3), p_bac=fmt(p_bac, 4),
        leak0=fmt(1 - W_EX[(E, A)] - W_EX[(E, Bn)], 2),
        F_a=fmt(F[fs({A})], 2), F_b=fmt(F[fs({Bn})], 2), F_ab=fmt(F[fs({A, Bn})], 2),
        F_v=fmt(F[fs(NODES)], 3),
        F_ab_t1=fmt(W_EX[(E, A)] * W_EX[(fs({A}), Bn)], 2),
        F_ab_t2=fmt(W_EX[(E, Bn)] * W_EX[(fs({Bn}), A)], 2),
        L_root_ex=fmt(-math.log(b0), 3),
    )

    # posterior and occupancy (earlier draft, Sections 9.1 and 9.2)
    q_a = W_EX[(E, A)] * b_a / b0
    q_b = W_EX[(E, Bn)] * b_b / b0
    renorm = W_EX[(E, A)] / (W_EX[(E, A)] + W_EX[(E, Bn)])
    N.update(q_a=fmt(q_a, 3), q_b=fmt(q_b, 3), renorm_a=fmt(renorm, 3),
             q_a_num=fmt(W_EX[(E, A)] * b_a, 3), wsum0=fmt(W_EX[(E, A)] + W_EX[(E, Bn)], 2))

    mu = {E: 0.5, fs({A}): 0.25, fs({Bn}): 0.25}
    m = {s: mu.get(s, 0.0) for s in ideals(NODES, PARENTS)}
    rho = {}
    for s in sorted(m, key=len):
        if s == fs(NODES):
            continue
        for v in ready(NODES, PARENTS, s):
            q = W_EX[(s, v)] * B(s | {v}) / B(s)
            rho[(s, v)] = m[s] * q
            m[s | {v}] += m[s] * q
    total_rho = sum(rho.values())
    expected = sum(p * (3 - len(s)) for s, p in mu.items())
    assert abs(total_rho - expected) < 1e-12 and abs(m[fs(NODES)] - 1) < 1e-12
    N.update(
        mu0="0.5", muA="0.25", muB="0.25",
        rho_0A=fmt(rho[(E, A)], 3), rho_0B=fmt(rho[(E, Bn)], 3),
        rho_AB=fmt(rho[(fs({A}), Bn)], 3), rho_BA=fmt(rho[(fs({Bn}), A)], 3),
        rho_C=fmt(rho[(fs({A, Bn}), C)], 3), rho_sum=fmt(total_rho, 1),
        m_A=fmt(m[fs({A})], 3), m_B=fmt(m[fs({Bn})], 3),
    )

    # prefix-mixture losses: balanced (worked example) vs order-collapsed
    Bc = completion(NODES, PARENTS, W_COL, E_COL)
    L = lambda x: -math.log(x)
    mix_ex = 0.5 * L(b0) + 0.25 * L(b_a) + 0.25 * L(b_b)
    mix_col = 0.5 * L(Bc(E)) + 0.25 * L(Bc(fs({A}))) + 0.25 * L(Bc(fs({Bn})))
    assert mix_ex < mix_col and L(Bc(E)) < L(b0)
    N.update(
        c_wA="0.85", c_wB="0.05", c_wAB="0.90", c_wBA="0.10",
        c_b0=fmt(Bc(E), 4), c_bA=fmt(Bc(fs({A})), 3), c_bB=fmt(Bc(fs({Bn})), 3),
        c_L0=fmt(L(Bc(E)), 3), c_LA=fmt(L(Bc(fs({A}))), 3), c_LB=fmt(L(Bc(fs({Bn}))), 3),
        e_L0=fmt(L(b0), 3), e_LA=fmt(L(b_a), 3), e_LB=fmt(L(b_b), 3),
        mix_ex=fmt(mix_ex, 3), mix_col=fmt(mix_col, 3),
    )

    # ---- two-model example (v2) --------------------------------------------
    def losses(p1, p2):
        tot = p1 + p2
        post = (p1 / tot, p2 / tot)
        kl = sum(0.5 * math.log(0.5 / q) for q in post)
        return dict(root=-math.log(tot), aug=-0.5 * (math.log(p1) + math.log(p2)),
                    sft=-math.log(p1), kl=kl)

    m1, m2 = losses(0.45, 0.05), losses(0.25, 0.25)
    for mm in (m1, m2):  # decomposition: L_aug = L_root + log 2 + KL
        assert abs(mm["aug"] - (mm["root"] + math.log(2) + mm["kl"])) < 1e-12
    N.update(
        m1_root=fmt(m1["root"], 3), m1_aug=fmt(m1["aug"], 3), m1_sft=fmt(m1["sft"], 3), m1_kl=fmt(m1["kl"], 3),
        m2_root=fmt(m2["root"], 3), m2_aug=fmt(m2["aug"], 3), m2_sft=fmt(m2["sft"], 3), m2_kl=fmt(m2["kl"], 3),
        ln2=fmt(math.log(2), 3),
    )

    # ---- counting: orders, ordered prefixes, completed sets, DP edges -------
    def counts_independent(n):
        orders = math.factorial(n)
        prefixes = sum(math.factorial(n) // math.factorial(n - k) for k in range(n + 1))
        states = 2**n
        edges = n * 2 ** (n - 1)
        return orders, prefixes, states, edges

    def counts_chains(lengths):
        n = sum(lengths)
        orders = math.factorial(n)
        for l in lengths:
            orders //= math.factorial(l)
        states = 1
        for l in lengths:
            states *= l + 1
        # edges: for each ideal (tuple of chain progress), one edge per unfinished chain
        edges = 0
        prefixes = 0
        for prog in itertools.product(*[range(l + 1) for l in lengths]):
            edges += sum(1 for p, l in zip(prog, lengths) if p < l)
            k = sum(prog)
            ways = math.factorial(k)
            for p in prog:
                ways //= math.factorial(p)
            prefixes += ways  # distinct ordered prefixes that reach this ideal
        return orders, prefixes, states, edges

    def put(tag, tup):
        o, p, s, e = tup
        for name, val in zip(("o", "p", "s", "e"), (o, p, s, e)):
            N[f"cnt_{tag}_{name}"] = thousands(val) if val < 10**13 else sci(val)

    for n in (3, 5, 10, 20):
        put(f"ind{n}", counts_independent(n))
    put("chain20", counts_chains([20]))
    put("ch4x5", counts_chains([5, 5, 5, 5]))
    assert counts_chains([5, 5, 5, 5])[2] == 1296  # earlier draft, Section 10.1
    assert counts_independent(10)[0] == 3_628_800  # v2 draft

    # ---- FuncBenchGen (Maekawa et al., ICLR 2026) ----------------------------
    N.update(
        fbg_vnyk_lo="66.8%", fbg_vnyk_hi="81.3%",
        fbg_iv_lo="17.9%", fbg_iv_hi="33.2%",
        fbg_fnf_hi="2.4%", fbg_wni_hi="0.9%",
        fbg_gpt5_before="62.5%", fbg_gpt5_after="81.3%",
        fbg_models="7",
    )

    # ---- decoding example (earlier draft, Section 11.3) ------------------------
    N.update(dec_each="0.09", dec_bad="0.10", dec_succ=fmt(10 * 0.09, 2))

    # ---- earlier draft numerical self-check (Section 13.2) -------------------
    N.update(selfcheck_p="5.6\\times10^{-17}", selfcheck_g="2.2\\times10^{-16}")

    # sanity: exact arithmetic for the headline value
    b0_exact = (Fraction(60, 100) * Fraction(50, 100) + Fraction(30, 100) * Fraction(90, 100)) \
        * Fraction(80, 100) * Fraction(95, 100)
    assert float(b0_exact) == round(b0, 12) or abs(float(b0_exact) - b0) < 1e-12
    return N


VALUES = compute()

if __name__ == "__main__":
    for k, v in VALUES.items():
        print(f"{k:14s} {v}")

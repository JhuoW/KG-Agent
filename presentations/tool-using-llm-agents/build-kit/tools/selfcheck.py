#!/usr/bin/env python3
"""Independent reproduction of the earlier draft's numerical self-check (Section 13.2).

For random DAGs with 3-7 nodes and a softmax policy over {ready actions, one wrong
action} at every completed set, it compares
  1. B(empty) from the lattice recursion with brute-force enumeration of linear extensions;
  2. the posterior-weighted gradient of log B(empty) with central finite differences.
Pure Python, double precision. Prints the maximum absolute differences.
"""

from __future__ import annotations

import itertools
import math
import random


def random_dag(n, p, rng):
    nodes = list(range(n))
    parents = {v: frozenset(u for u in range(v) if rng.random() < p) for v in nodes}
    return nodes, parents


def ideals(nodes, parents):
    out = []
    for r in range(len(nodes) + 1):
        for c in itertools.combinations(nodes, r):
            s = frozenset(c)
            if all(parents[v] <= s for v in s):
                out.append(s)
    return out


def ready(nodes, parents, s):
    return [v for v in nodes if v not in s and parents[v] <= s]


def policy(logits):
    """logits: {state: {action: z}} with action 'wrong' always present; returns probs."""
    probs = {}
    for s, zs in logits.items():
        m = max(zs.values())
        e = {a: math.exp(z - m) for a, z in zs.items()}
        tot = sum(e.values())
        probs[s] = {a: x / tot for a, x in e.items()}
    return probs


def completion(nodes, parents, probs):
    full = frozenset(nodes)
    order = sorted(probs, key=len, reverse=True)
    B = {}
    for s in order:
        if s == full:
            B[s] = probs[s]["end"]
        else:
            B[s] = sum(probs[s][v] * B[s | {v}] for v in ready(nodes, parents, s))
    return B


def enumerate_paths(nodes, parents, probs):
    total = 0.0
    for perm in itertools.permutations(nodes):
        s, p = frozenset(), 1.0
        ok = True
        for v in perm:
            if not parents[v] <= s:
                ok = False
                break
            p *= probs[s][v]
            s = s | {v}
        if ok:
            total += p * probs[s]["end"]
    return total


def posterior_gradient(nodes, parents, probs, B):
    """d log B(empty) / d z[s][a] via forward-backward edge posteriors."""
    full = frozenset(nodes)
    F = {s: 0.0 for s in probs}
    F[frozenset()] = 1.0
    for s in sorted(probs, key=len):
        if s != full:
            for v in ready(nodes, parents, s):
                F[s | {v}] += F[s] * probs[s][v]
    b0 = B[frozenset()]
    grad = {}
    for s, ps in probs.items():
        if s == full:
            gam = {"end": F[s] * ps["end"] / b0}
        else:
            gam = {v: F[s] * ps[v] * B[s | {v}] / b0 for v in ready(nodes, parents, s)}
        occ = sum(gam.values())  # posterior probability of visiting s
        for a in ps:
            grad[(s, a)] = gam.get(a, 0.0) - occ * ps[a]
    return grad


def main(n_graphs=30, seed=0):
    rng = random.Random(seed)
    max_p, max_g = 0.0, 0.0
    for _ in range(n_graphs):
        n = rng.randint(3, 7)
        nodes, parents = random_dag(n, rng.uniform(0.1, 0.6), rng)
        J = ideals(nodes, parents)
        full = frozenset(nodes)
        logits = {}
        for s in J:
            acts = ["end"] if s == full else ready(nodes, parents, s)
            logits[s] = {a: rng.gauss(0, 1) for a in acts}
            logits[s]["wrong"] = rng.gauss(0, 1)
        probs = policy(logits)
        B = completion(nodes, parents, probs)
        max_p = max(max_p, abs(B[frozenset()] - enumerate_paths(nodes, parents, probs)))
        g = posterior_gradient(nodes, parents, probs, B)
        h = 1e-6
        for key in rng.sample(sorted(g, key=str), min(12, len(g))):
            s, a = key
            logits[s][a] += h
            up = math.log(completion(nodes, parents, policy(logits))[frozenset()])
            logits[s][a] -= 2 * h
            dn = math.log(completion(nodes, parents, policy(logits))[frozenset()])
            logits[s][a] += h
            max_g = max(max_g, abs((up - dn) / (2 * h) - g[key]))
    print(f"{n_graphs} random DAGs, 3-7 nodes")
    print(f"max |B_DP - B_enum|            = {max_p:.2e}")
    print(f"max |grad_posterior - grad_FD| = {max_g:.2e}  (finite differences, h=1e-6)")
    assert max_p < 1e-12 and max_g < 1e-6
    return max_p, max_g


if __name__ == "__main__":
    main()

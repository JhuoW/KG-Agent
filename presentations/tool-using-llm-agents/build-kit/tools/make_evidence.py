#!/usr/bin/env python3
"""Writes evidence/sources.json and evidence/evidence.json.

Each claim on the slides is mapped to its source location, its epistemic status, and
how it was checked. Status values:
  source-v2          stated in the reviewed draft (QET v2, Notion, 2026-10-13)
  source-v1          stated only in the earlier draft (Notion, 2026-10-08)
  established        published or textbook result, checked at the cited source
  derived            arithmetic on the drafts' definitions, recomputed by deck/numbers.py
  illustrative       example values chosen for teaching
  synthesis          interpretation added for this deck, marked "(my synthesis)"
  hypothesis         claim the drafts propose to test
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCES = [
    {"id": "v2", "title": "Working draft, method name acronym QET, version 2",
     "where": "Notion page 3f3730d9-83f2-80d0-a2fe-cdffe2def0ef (parent: Graph For LLM)",
     "version": "last edited 2026-10-13T11:32Z", "read": "full text",
     "notes": "Primary source. Two inline SVG figures were not retrievable as images; their content is "
              "described in the surrounding text."},
    {"id": "v1", "title": "Working draft, earlier version",
     "where": "Notion page 3ec730d9-83f2-80d8-955c-c91df4d83d65 (parent: Graph For LLM)",
     "version": "last edited 2026-10-08T05:43Z", "read": "full text",
     "notes": "Used for definitions that v2 uses without stating (q, m, rho, B-, B+, C, epsilon), the "
              "worked-example policy scores, the evaluation plan, and the scope statements."},
    {"id": "fbg", "title": "Maekawa et al. Towards Reliable Benchmarking: A Contamination Free, Controllable "
                          "Evaluation Framework for Multi-step LLM Function Calling. ICLR 2026.",
     "where": "https://arxiv.org/pdf/2509.26553v2", "read": "full text (19 pages, pdftotext)",
     "notes": "Table 3 numbers read from text. Figure 8 bar heights are not machine-readable; the 62.5% to "
              "81.3% GPT-5 figure is quoted from the abstract and Section 1."},
    {"id": "he2025", "title": "He et al. Order Doesn't Matter, But Reasoning Does: Training LLMs with "
                             "Order-Centric Augmentation. arXiv:2502.19907.",
     "where": "https://arxiv.org/abs/2502.19907", "read": "abstract page only"},
    {"id": "li2026", "title": "Li, Cheng, Nicholls, Kong. De-Linearizing Agent Traces: Bayesian Inference of "
                             "Latent Partial Orders for Efficient Execution. arXiv:2602.02806.",
     "where": "https://arxiv.org/abs/2602.02806", "read": "abstract page only"},
    {"id": "chen2021", "title": "Chen et al. Order Matters: Probabilistic Modeling of Node Sequence for Graph "
                               "Generation. ICML 2021, PMLR 139:1630-1639.",
     "where": "https://proceedings.mlr.press/v139/chen21j.html", "read": "abstract page only"},
    {"id": "goldberg2013", "title": "Goldberg and Nivre. Training Deterministic Parsers with Non-Deterministic "
                                   "Oracles. TACL 1:403-414, 2013.",
     "where": "https://aclanthology.org/Q13-1033/", "read": "abstract page only"},
    {"id": "kim2024", "title": "Kim et al. An LLM Compiler for Parallel Function Calling. ICML 2024, PMLR "
                              "235:24370-24391.",
     "where": "https://proceedings.mlr.press/v235/kim24y.html", "read": "search-result snippets only"},
    {"id": "graves2006", "title": "Graves et al. Connectionist Temporal Classification. ICML 2006, 369-376.",
     "where": "doi:10.1145/1143844.1143891", "read": "Crossref metadata and secondary descriptions"},
    {"id": "godefroid1996", "title": "Godefroid. Partial-Order Methods for the Verification of Concurrent "
                                    "Systems. LNCS 1032, Springer, 1996.",
     "where": "doi:10.1007/3-540-60761-7", "read": "library catalog record and table of contents"},
    {"id": "bernstein1966", "title": "Bernstein. Analysis of Programs for Parallel Processing. IEEE Trans. "
                                    "Electronic Computers EC-15(5):757-763, 1966.",
     "where": "doi:10.1109/PGEC.1966.264565", "read": "Crossref metadata and course notes stating the conditions"},
    {"id": "bw1991", "title": "Brightwell and Winkler. Counting Linear Extensions. Order 8(3):225-242, 1991.",
     "where": "doi:10.1007/BF00383444", "read": "Crossref metadata and citing abstracts"},
    {"id": "appworld", "title": "Trivedi et al. AppWorld. ACL 2024, 16022-16076.",
     "where": "doi:10.18653/v1/2024.acl-long.850", "read": "Crossref metadata only (cited by v1)"},
]


def C(id, slides, claim, status, sources, check):
    return {"id": id, "slides": slides, "claim": claim, "status": status, "sources": sources, "check": check}


CLAIMS = [
    C("C01", ["running-example"], "For load A, load B, compare: a before c and b before c are the only required "
      "orders; (a,b,c) and (b,a,c) are two executions of one computation.", "source-v2",
      ["v2: motivation"], "Consistent with v1 Section 1.1."),
    C("C02", ["semantics-vs-serialization"], "A sequence target mixes task semantics with serialization choices.",
      "source-v2", ["v2: motivation"], "Read in source."),
    C("C03", ["consequences"], "Three consequences: repeated data and compute, resumption, argument availability "
      "and provenance.", "source-v2", ["v2: motivation callout"], "Read in source."),
    C("C04", ["funcbenchgen"], "Value Not Yet Known accounts for 66.8% to 81.3% of errors across 7 models.",
      "established", ["fbg: Table 3, p.9"], "Read in full text; min 66.8% (GPT-5-mini, GPT-4.1-mini), max 81.3% "
      "(Gemini-2.5-Flash)."),
    C("C05", ["funcbenchgen"], "Incorrect Value accounts for 17.9% to 33.2% of errors.", "established",
      ["fbg: Table 3"], "Read in full text."),
    C("C06", ["funcbenchgen"], "Function Not Found at most 2.4%; Wrong Number of Inputs at most 0.9%.",
      "established", ["fbg: Table 3"], "Read in full text."),
    C("C07", ["funcbenchgen", "baselines"], "Restating known variable values after each call raised GPT-5 success "
      "from 62.5% to 81.3%.", "established", ["fbg: abstract; Section 1; Section 4.6.1; Figure 8"],
      "Numbers from abstract and Section 1; the configuration is shown only in Figure 8 (5 core nodes, averaged "
      "over connected-distractor settings), which could not be read numerically."),
    C("C08", ["failure-illustration"], "Prices 1200, 600 and the wrong value 900 illustrate the two failure types.",
      "illustrative", ["v2: motivation"], "The draft's own illustration, not FuncBenchGen data; the paper's case "
      "study (Figure 9a) shows the same pattern with a value from an irrelevant function."),
    C("C09", ["hypothesis"], "Central hypothesis and its testable version.", "hypothesis",
      ["v2: central research hypothesis", "v1: abstract, Section 14.1"], "Stated as hypotheses in both drafts."),
    C("C10", ["hypothesis", "claim-status"], "No training results; self-check on 30 random DAGs (3-7 nodes) with "
      "differences 5.6e-17 (probability) and 2.2e-16 (gradient).", "source-v1", ["v1: Section 13.2"],
      "Reproduced independently by tools/selfcheck.py: 5.55e-17 for the probability; gradient checked against "
      "finite differences (4.6e-10, limited by finite-difference precision) instead of autodiff."),
    C("C11", ["augmentation-limits"], "Order augmentation fits a sampling distribution and rescores shared "
      "states; one mastered order does not ensure resumption.", "source-v2", ["v2: motivation"], "Read in source."),
    C("C12", ["setup"], "x=(u,T,o0); (s',o)=Exec_zeta(s,a); tau=(a1,o1,...,an,on,a_end*).", "source-v2",
      ["v2: Section 2"], "Notation preserved."),
    C("C13", ["setup"], "The snapshot zeta is never given to the policy as privileged input.", "source-v1",
      ["v1: Section 3.1"], "Read in source."),
    C("C14", ["target-graph"], "G=(V,E,A,O,K) with the flight and hotel example.", "source-v2", ["v2: Section 3"],
      "v1 Section 4.1 defines O and K differently; see conflicts in REPORT.md."),
    C("C15", ["non-interference"], "W_u ∩ (R_v ∪ W_v) = ∅ and W_v ∩ (R_u ∪ W_u) = ∅ are sufficient, not "
      "necessary, for safe reordering (Bernstein conditions).", "established",
      ["bernstein1966", "v1: Section 4.4", "v2: Section 3 (states it as necessary)"],
      "Corrected: v2 phrases the condition as necessary."),
    C("C16", ["edge-sources"], "lookup_address then send_message is a data dependency without write conflict.",
      "source-v1", ["v1: Section 4.3"], "update_budget/get_budget row is an illustration added for this deck."),
    C("C17", ["edge-sources"], "All relevant pairs must be checked (a writes x, b unrelated, c reads x keeps a<c).",
      "source-v1", ["v1: Section 4.5"], "Read in source."),
    C("C18", ["accepted-set"], "LinExt(G) and C(G) for the example; C(G) can be smaller than all truly equivalent "
      "executions.", "source-v2", ["v2: Section 3", "v1: Section 4.6"], "Read in source."),
    C("C19", ["set-target"], "SFT on one trajectory versus -log sum over C(G).", "source-v2", ["v2: solution 1"],
      "Typo corrected: the second trajectory is tau_2."),
    C("C20", ["history-dependence"], "A Transformer may assign P(C|h1) != P(C|h2) for equivalent histories.",
      "source-v2", ["v2: Section 3.1"], "Follows from order-dependent inputs; no empirical claim."),
    C("C21", ["reduce-render"], "Reduce/Render definitions and their constraints.", "source-v2",
      ["v2: Section 3.1", "v1: Sections 5.2, 5.3"], "Read in source."),
    C("C22", ["action-probability"], "Action log probability is the sum of token log probabilities including the "
      "end marker; no renormalization over correct actions.", "source-v2",
      ["v2: Section 3.1", "v1: Sections 6.1, 6.2"], "Read in source."),
    C("C23", ["order-ideals"], "Order ideals of a finite poset form a distributive lattice.", "established",
      ["textbook (Birkhoff's representation theorem)"], "Standard result; not checked against a specific text."),
    C("C24", ["sharing-condition"], "State sharing requires deterministic accepted paths, order-invariant "
      "reduction, and no hidden memory.", "source-v1", ["v1: Sections 5.2, 5.5, 7.2"], "Read in source."),
    C("C25", ["ready-set", "recursion"], "Ready set, step scores, recursion B(I)=sum w B(I+v) with B(V)=e(V).",
      "source-v2", ["v2: Section 3", "v1: Eqs. (17)-(18)"], "Base case is implicit in v2."),
    C("C26", ["worked-example"], "B(empty)=0.4332 with B({A,B})=0.76, B({A})=0.38, B({B})=0.684.", "derived",
      ["v1: Section 13.1"], "Recomputed by deck/numbers.py; brute-force path sum agrees."),
    C("C27", ["forward-view"], "F({A,B})=0.57, F(V)=0.456, F(V)e(V)=0.4332=B(empty).", "derived",
      ["v2: solution 3"], "Recomputed; v2's forward form omits e(V)."),
    C("C28", ["why-sharing"], "Counts of orders, prefixes, completed sets, and lattice edges; 1,296 for four "
      "chains of five; counting linear extensions is #P-complete.", "derived",
      ["v1: Section 10.1", "bw1991"], "Counts computed exactly; 1,296 matches v1."),
    C("C29", ["why-sharing"], "Prefix caching alone cannot merge A-then-B with B-then-A.", "synthesis", [],
      "Marked (my synthesis)."),
    C("C30", ["marginal"], "L_root = E[-log B(empty)] = -log P(C(G)); B(I) <= Pr(success) when every accepted "
      "execution passes the task check.", "source-v2", ["v2: class-marginal objective", "v1: Eq. (19)"],
      "Read in source."),
    C("C31", ["three-objectives"], "Two-model example: L_root 0.693 for both; L_aug 1.897 versus 1.386.",
      "derived", ["v2: class-marginal objective"], "Recomputed; SFT row (0.799, 1.386) added for comparison."),
    C("C32", ["decomposition"], "E_omega[-log p] = -log P(C) + H(omega) + KL(omega || p(.|C)).", "established",
      ["v1: Eq. (21)"], "Verified algebraically (cross-entropy = entropy + KL) and numerically (KL 0.511 and 0)."),
    C("C33", ["posterior-gradient"], "Gradient identity and q(A|empty)=0.526, q(B|empty)=0.474 versus 0.667.",
      "derived", ["v1: Sections 9.1, 13.1"], "Recomputed; matches v1's 0.5263."),
    C("C34", ["posterior-gradient"], "Marginal gradient follows the model's own success-conditioned preference.",
      "synthesis", [], "Marked (my synthesis); consistent with v1 Section 15.3."),
    C("C35", ["transition-weights"], "rho values and their sum 2.5 for mu = (0.5, 0.25, 0.25); stop-gradient "
      "surrogate value differs from the marginal loss.", "derived", ["v1: Sections 9.2-9.5, 13.1"],
      "Recomputed; sum matches v1."),
    C("C36", ["root-only-limit", "prefix-effect"], "Order-collapsed policy: root loss 0.536 but resume loss "
      "2.577; mixture losses 0.755 versus 1.007.", "illustrative", ["v2: valid-prefix coverage", "v1: Section 8.3"],
      "Collapsed-policy scores chosen for this deck and marked (my synthesis); losses recomputed."),
    C("C37", ["prefix-training"], "mu_G = (1-lambda) delta_empty + lambda mu_cut; L = E E[-log B(I)]; K samples.",
      "source-v2", ["v2: valid-prefix coverage"], "Read in source."),
    C("C38", ["prefix-effect"], "Coverage bound: nu <= kappa mu implies E_nu[1-B] <= kappa E_mu[-log B].",
      "source-v1", ["v1: Proposition 3, Section 12.3"], "Proof checked: 1 - z <= -log z for z in (0, 1]."),
    C("C39", ["training-procedure"], "Algorithm 1 steps.", "source-v2", ["v2: Algorithm 1"], "Read in source."),
    C("C40", ["bounded-computation"], "B- <= B <= B+, log gap bound, seeding, O(w N_G), N_G <= prod(l_j+1).",
      "source-v1", ["v1: Section 10"], "Bounds follow from monotonicity of the recursion in nonnegative terms."),
    C("C41", ["inference-procedure"], "Algorithm 2; no access to G, lattice, B, q at inference; decoding example "
      "0.09 x 10 versus 0.10.", "source-v2", ["v2: Algorithm 2", "v1: Sections 11.2, 11.3"], "Read in source."),
    C("C42", ["related-methods"], "Descriptions of He et al. 2025, Li et al. 2026, Chen et al. 2021, Goldberg "
      "and Nivre 2013, Kim et al. 2024.", "established",
      ["he2025", "li2026", "chen2021", "goldberg2013", "kim2024", "v1: Section 2.2"],
      "Checked against abstracts only; LLMCompiler row marked as added for this deck."),
    C("C43", ["foundations"], "Bernstein conditions, partial-order reduction, #P-completeness of counting linear "
      "extensions, CTC forward-backward.", "established", ["bernstein1966", "godefroid1996", "bw1991", "graves2006"],
      "Bibliographic data verified via Crossref; content checked through secondary sources. CTC analogy marked "
      "(my synthesis)."),
    C("C44", ["related-methods"], "Novelty is claimed only for the combination.", "source-v1",
      ["v1: Section 2.2"], "Read in source."),
    C("C45", ["assumptions"], "Scope: deterministic accepted paths, finite demonstrations, conservative graphs, "
      "successful prefixes only, one accepted set per task.", "source-v1",
      ["v1: Sections 3.2, 15.1, 15.4, 15.6", "fbg: Section 3.4"],
      "FuncBenchGen returns random incorrect outputs for invalid inputs (read in full text)."),
    C("C46", ["claim-status"], "Hypotheses H1-H5.", "hypothesis", ["v1: Section 14.1"], "Read in source."),
    C("C47", ["baselines", "metrics-falsification"], "Planned baselines, controls, metrics, testbeds, and "
      "falsification criteria.", "source-v1", ["v1: Sections 14.2-14.10"], "Read in source."),
    C("C48", ["open-questions"], "Open questions on certification, mu_cut, posterior concentration, cost, "
      "stochastic tools.", "source-v1", ["v1: Sections 8.4, 15, 16", "v2: Algorithm 1"], "Read in source."),
]


def main():
    ev = ROOT / "evidence"
    ev.mkdir(exist_ok=True)
    (ev / "sources.json").write_text(json.dumps(SOURCES, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ev / "evidence.json").write_text(json.dumps({"status_legend": __doc__.split("Status values:")[1].strip(),
                                                  "claims": CLAIMS}, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    print(f"wrote {len(SOURCES)} sources and {len(CLAIMS)} claims")


if __name__ == "__main__":
    main()

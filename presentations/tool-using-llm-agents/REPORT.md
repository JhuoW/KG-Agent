# Report: "Tool-Using LLM Agents" slide deck

## Deliverables
| File | Content |
|---|---|
| `tool-using-llm-agents.html` | Self-contained deck, 47 slides, English and Simplified Chinese with a toggle (1.3 MB, works offline) |
| `tool-using-llm-agents-build-kit.zip` | Generator, content modules, evidence JSON, change logs, checks (also unpacked in `build-kit/`) |
| `REPORT.md` | This report |

**Slide count:** 47. Overview 2, motivation 8, problem setup 1, execution graph 5, execution state 3,
shared-state computation 7, training objective 8, algorithms 3, positioning and status 7, summary 1,
appendix 2 (notation, references).

**Checks at release:**
- Linter: 0 errors and 0 warnings. All 141 English prose sentences are 13 to 25 words long.
  Math segments and numbers match between the English and Chinese versions of every item.
- Layout: no overflow in any slide in either language at 1280×720. 46 of 47 slides scale their
  content up to fill the slide; one Chinese slide (25) scales down to 0.974.
- Other renders: no horizontal scroll at 400 px width; dark theme checked.
- The word "quotient" appears nowhere in the HTML.
- The build is byte-for-byte reproducible.

## Sources read in full text
1. **v2 draft** (Notion, edited 2026-10-13 11:32 UTC): the primary source. Two embedded SVG figures
   could not be retrieved as images; the surrounding text describes them.
2. **Earlier draft** (Notion, edited 2026-10-08). v2 uses q, m, ρ, B⁻, B⁺, C and ε without defining
   them, and only this version defines them. It also supplies the worked-example scores, the scope
   statements and the evaluation plan. Every slide using it is labelled "From the earlier draft".
3. **FuncBenchGen**, Maekawa et al., ICLR 2026, arXiv 2509.26553v2, all 19 pages.

Read only partially (abstract, metadata or secondary descriptions): He et al. 2025, Li et al. 2026,
Chen et al. 2021, Goldberg and Nivre 2013, LLMCompiler 2024, CTC 2006, Godefroid 1996,
Bernstein 1966, Brightwell and Winkler 1991, AppWorld 2024. Bibliographic data for the last five
was confirmed through Crossref. `build-kit/evidence/sources.json` records how deeply each source
was read.

## Corrections made (full table: `build-kit/changelog/source_corrections.md`)
1. **Non-interference condition.** v2 states it as necessary ("必须满足"). It is the Bernstein
   condition, which is sufficient but not necessary, as the earlier draft (§4.4) says. Shown on
   slide 13.
2. **Missing base case.** B_θ(V) = e_θ(V) is implicit in v2; the slides now state it (slide 23).
3. **Forward and backward forms.** v2's forward recursion omits the terminal action. The slides
   reconcile it: F(V)·e_θ(V) = B_θ(∅) = 0.4332 (slide 25).
4. **Notation clashes.** v2 uses s_t for both the canonical state and the environment state, and 𝒜
   for both the acceptance class and the action specification. The deck uses H_t and 𝒞(G) throughout.
5. **Typos.** τ₁ is repeated where τ₂ is meant; "AB and BC" should read "AB and BA"; "augment"
   should be "argument"; "第二部" should be "第二步"; stray "人" in the §3.1 heading.
6. **Terminology.**
   - "Execution acceptance class" became "accepted execution set".
   - "Probability flow" became "forward probability".
   - The method appears only as the acronym QET and is never expanded.

## Unverified or partially verified claims
- **FuncBenchGen GPT-5 gain (62.5% → 81.3%).** Quoted from the paper's abstract and Section 1. The
  exact configuration appears only in Figure 8, whose values cannot be read from the text.
- **Related-method descriptions** (slide 38). Checked against abstracts only. The "difference" column
  restates the earlier draft's own claims and was not tested against the full papers.
- **Gradient half of the earlier draft's self-check** (30 random DAGs).
  - The probability half reproduced: 5.55e-17 versus the reported 5.6e-17.
  - The gradient was checked against finite differences (agreement 4.6e-10), not against autodiff.
    The draft's 2.2e-16 figure is therefore not reproduced exactly.
- **Birkhoff lattice statement** (slide 20). A standard result, not checked against a specific text.
- **Interpretations, each marked (my synthesis):**
  - the prefix-tree argument (slide 26);
  - "the marginal gradient follows the model's own preference" (slide 30);
  - the order-collapsed policy numbers (slides 32 and 34);
  - the LLMCompiler row (slide 38);
  - the CTC analogy (slide 39).
- **Illustrative examples added for this deck.** The update_budget pair (slide 14) and the snapshot
  example (slide 11) are labelled as illustrations.
- **No performance results.** Neither draft reports LLM training results. Every performance
  statement is presented as a hypothesis (slides 8, 41, 45).

## Unresolved conflicts between sources
1. **Definitions of 𝒪 and 𝒦.**
   - v2: 𝒪 is the observation specification, and 𝒦 is the set of constraints on legality, data flow
     and result checks.
   - Earlier draft (§4.1): 𝒪 is action outputs and versioned artifacts, and 𝒦 is the contracts or
     evidence that support edges and swaps.
   - The deck follows v2 and names the other reading in slide 12's source line.
2. **Cut distribution μ_cut.** v2 leaves it unspecified. The earlier draft samples a depth and then
   ready nodes (§8.4). This is listed as an open question (slide 44).
3. **Is augmentation's order preference a defect?**
   - v2 presents the preference for Model 2 as a drawback of augmentation.
   - The earlier draft (§8.2) says the extra KL term may act as a useful regularizer and that this is
     an empirical question.
   - The deck shows the decomposition and the earlier draft's caution (slide 29). It does not
     resolve the question, which needs experiments.

Two apparent conflicts were resolved:
- **Bernstein's paper title.** A course note gives "Program Analysis for Parallel Processing";
  Crossref confirms "Analysis of Programs for Parallel Processing".
- **"Incorrect Value" example.** v2's illustration passes a wrong value that may come from another
  tool. This matches FuncBenchGen's definition and its Figure 9a case study.

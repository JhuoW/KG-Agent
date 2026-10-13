# Corrections and harmonizations relative to the v2 draft

Each entry lists the location in the v2 draft (Notion, edited 2026-10-13), what it says,
what the deck shows instead, and the evidence for the change. Entries marked
"shown on slide" are visible to the audience in a Correction callout or a source line.

| # | v2 location | v2 text | Deck | Evidence | Shown on slide |
|---|---|---|---|---|---|
| 1 | Section 3, operation contracts | Non-interference: two non-interfering operations "必须满足" (must satisfy) the two write/read conditions, which reads as a necessary condition | The conditions are sufficient for safe reordering, not necessary, and assume complete read and write sets | Bernstein (1966); the earlier draft, Section 4.4, also calls them sufficient. Counterexample: two writes of the same value commute but violate the condition | Yes (slide 13) |
| 2 | Solution item 1 | Both trajectories are labelled τ₁ | The second trajectory is τ₂; the deck writes (A,B,C) and (B,A,C) explicitly | Typo | Yes (slide 16 source line) |
| 3 | Class-marginal objective | "实际上 AB 和 BC 到达的状态是一样的" | AB and BA reach the same state | Typo; BC is not a valid two-action prefix | No (the deck never repeats the sentence) |
| 4 | Section 3, completion probability | Defines e_θ(V) but never states the base case | B_θ(V) = e_θ(V) | Earlier draft, Eq. (17) | Yes (slide 23) |
| 5 | Section 1 vs Sections 2-3 | s_t denotes the canonical state σ(h_t) in Section 1 and the environment state in Sections 2-3 | s denotes the environment state; H_t denotes the canonical observation state throughout | Notation clash inside v2 | Yes (slide 17 source line) |
| 6 | Section 1 | 𝒜(x) denotes the acceptance class, while 𝒜 is the action specification inside G | 𝒞(G) for the accepted execution set; 𝒜 only for actions | Notation clash inside v2; v2 Section 3 itself uses 𝒞(G) | No |
| 7 | Solution item 3 | Forward recursion F(s') += F(s)π(a|s) ends at F(s_ABC) without the terminal action, while the method section's B_θ includes e_θ(V) | F(V)·e_θ(V) = B_θ(∅) = 0.4332, shown numerically | Derived; both recursions recomputed in deck/numbers.py | Yes (slide 25) |
| 8 | Algorithm 1 | Uses q, m, ρ, B⁻, B⁺, C and ε without defining them | Definitions taken from the earlier draft, Sections 9 and 10, and labelled as such | Earlier draft | Yes (slides 30, 31, 36) |
| 9 | Solution item 3 | "概率流" (probability flow) | "Forward probability" F(I) and "completion probability" B_θ(I) | Established terms (forward-backward algorithm); the earlier draft calls B_θ the completion probability | No |
| 10 | Section 3.1 heading | "根据人“task进行到什么状态”" (stray character 人) | Not quoted | Typo | No |
| 11 | Section 3 | "参数绑定到 C 的 augment 中" | "argument" | Typo | No |
| 12 | Valid-prefix coverage | "第二部" | "第二步" (step 2) | Typo | No |
| 13 | Whole draft | Method name expands to a term containing "Quotient" | The deck uses only the acronym QET (configurable as METHOD_NAME in build.py) and never expands it | User requirement | Not applicable |
| 14 | Section 3 | "execution acceptance class" (执行接受类) | "accepted execution set" (接受执行集合), with the draft's term named once in a source line | Plain wording per user requirement | Yes (slide 15 source line) |

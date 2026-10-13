from deck.blocks import CALL, CODE, COLS, MATH, P, SVG, T, TERMS, UL, slide

RENDERED = """Completed operations:
- flight_query: price = 1200
- hotel_query: price = 600

Available values:
- flight_price: 1200, source = flight_query
- hotel_price: 600, source = hotel_query"""

ACTION_JSON = """{
  "tool": "calculate_total",
  "arguments": {
    "flight_price": 1200,
    "hotel_price": 600
  }
}"""

SLIDES = [
    slide(
        "history-dependence", "state", "History dependence of a Transformer policy", "Transformer 策略对历史顺序的依赖",
        status=["draft"],
        blocks=[
            MATH(r"h_1=(A,o_A,B,o_B),\qquad h_2=(B,o_B,A,o_A)", cls="key"),
            COLS(
                [UL(
                    (r"If $A$ and $B$ are independent and return the same results, both histories describe exactly "
                     r"the same task progress.",
                     r"若 $A$、$B$ 相互独立且返回结果相同，两段历史描述的任务进度完全一样。"),
                    (r"A Transformer reads different token orders and positions, so it may still assign "
                     r"$P_\theta(C\mid h_1)\neq P_\theta(C\mid h_2)$.",
                     r"Transformer 读入的 token 顺序和位置不同，因此仍可能给出 $P_\theta(C\mid h_1)\neq P_\theta(C\mid h_2)$。"),
                    ("When next-action probabilities differ, the two histories cannot be merged into one state for "
                     "exact probability computation.",
                     "下一步动作的概率一旦不同，两段历史就不能合并为同一状态来做精确的概率计算。"),
                )],
                [SVG("two_histories"),
                 TERMS([
                     ("s_t", T("Environment state at step $t$, held by the executor", "第 $t$ 步的环境真实状态，由执行器保存")),
                     ("h_t", T("Full interaction history up to step $t$", "截至第 $t$ 步的完整交互历史")),
                     ("H_t", T("Canonical observation state given to the LLM", "提供给 LLM 的规范化观测状态")),
                 ], cls="compact")],
                ratio="1fr 1.1fr",
            ),
        ],
        source=("v2 draft, Sections 1 and 3.1. v2 also uses $s_t$ for the canonical state in its introduction; this deck "
                "uses $H_t$ throughout.",
                "v2 草稿第 1 节与第 3.1 节。v2 在引言中也用 $s_t$ 表示规范化状态；本报告统一使用 $H_t$。"),
    ),
    slide(
        "reduce-render", "state", "Reduce and Render", "Reduce 与 Render",
        status=["draft", "draft1"],
        blocks=[
            COLS(
                [
                    MATH(r"\Reduce(h_1)=\Reduce(h_2)=H_t"),
                    MATH(r"\pi_\theta(a\mid x,H_t)=p_\theta\big(a\mid x,\Render(H_t)\big)", cls="key"),
                    UL(
                        ("Reduce maps every history with the same task progress to one canonical state built from "
                         "completed actions and their observations.",
                         "Reduce 把任务进度相同的所有历史映射为同一个规范化状态，该状态只由已完成动作及其观察构成。"),
                        ("Render writes that state as fixed text, so both histories yield identical prompts and "
                         "identical next-action distributions.",
                         "Render 把该状态写成固定文本，因此两段历史得到完全相同的提示词和下一步动作分布。"),
                        ("Reduce may use only past actions, actual observations, and runtime contracts, never the "
                         "target graph or future actions.",
                         "Reduce 只能使用已发生的动作、实际观察和运行时可访问的契约，不能使用目标图或未来动作。"),
                    ),
                ],
                [
                    CODE(RENDERED),
                    P(r"Rendered state after either history, with the same text for $h_1$ and $h_2$.",
                      r"两段历史之后的渲染状态，$h_1$ 与 $h_2$ 得到的文本完全相同。", cls="small muted", label=True),
                    CALL("draft1",
                         "Canonicalization drops only order certified as irrelevant, while overwrites of one variable "
                         "and orders required by the task remain.",
                         "规范化只去除经认证与任务无关的顺序；对同一变量的覆盖写入以及任务要求的先后顺序都会保留。"),
                ],
                ratio="1.05fr 1fr",
            ),
        ],
        source=("v2 draft, Section 3.1; the constraints on Reduce are from the earlier draft, Sections 5.2 and 5.3.",
                "v2 草稿第 3.1 节；对 Reduce 的约束来自较早草稿第 5.2 与 5.3 节。"),
    ),
    slide(
        "action-probability", "state", "Probability of a complete action", "完整动作的概率",
        status=["draft", "draft1"],
        blocks=[
            MATH(r"\log\pi_\theta(a\mid x,H)=\sum_{\ell=1}^{L}\log p_\theta(a_\ell\mid x,H,a_{<\ell})", cls="key"),
            COLS(
                [CODE(ACTION_JSON),
                 P("The complete action $a$ for node $C$, including its closing token.",
                   "节点 $C$ 的完整动作 $a$，包括结尾的结束标记。", cls="small muted", label=True)],
                [UL(
                    ("The action probability covers every token of the call: tool name, arguments, format, and the "
                     "explicit end marker.",
                     "动作概率覆盖调用中的每个 token，包括工具名称、参数、格式以及明确的结束标记。"),
                    ("A call with <code>hotel_price = 900</code> is a different token sequence, so the objective never "
                     "rewards that wrong argument.",
                     "传入 <code>hotel_price = 900</code> 的调用是另一个 token 序列，因此训练目标不会奖励这个错误参数。"),
                    ("Tool outputs enter as observations, not prediction targets, so deterministic accepted paths need "
                     "no environment probability factor.",
                     "工具返回内容作为观察输入而不是预测目标，因此在确定性的接受路径上不需要额外的环境概率因子。"),
                )],
                ratio="0.8fr 1.2fr", cls="mid",
            ),
            CALL("draft1",
                 "Scores use the actual policy probability, because renormalizing over the correct actions would "
                 "discard the mass placed on wrong actions.",
                 "得分使用策略的实际概率，因为若只在正确动作之间重新归一化，就会丢弃模型分配给错误动作的概率。"),
        ],
        source=("v2 draft, Section 3.1; the last two points are from the earlier draft, Sections 6.1 and 6.2.",
                "v2 草稿第 3.1 节；最后两点来自较早草稿第 6.1 与 6.2 节。"),
    ),
]

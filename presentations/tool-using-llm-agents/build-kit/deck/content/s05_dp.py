from deck.blocks import CALL, COLS, MATH, P, STEPS, SVG, T, TABLE, TERMS, UL, slide

SLIDES = [
    slide(
        "order-ideals", "dp", "Completed sets and order ideals", "已完成集合与序理想",
        status=["draft", "est"],
        blocks=[
            COLS(
                [
                    MATH(r"v\in I\;\Longrightarrow\;\pa_G(v)\subseteq I", cls="key"),
                    MATH(r"\mathcal{J}(G)=\big\{\varnothing,\{A\},\{B\},\{A,B\},\{A,B,C\}\big\}"),
                    UL(
                        (r"A completed set $I\subseteq V$ lists the operations already done, and it is valid only if it "
                         r"contains every prerequisite of its members.",
                         r"已完成集合 $I\subseteq V$ 列出已经完成的操作；只有包含其中每个操作的全部前驱时，它才合法。"),
                        ("Such down-closed sets are called order ideals, and under inclusion they form a lattice that "
                         "serves as the training state space.",
                         "这类向下封闭的集合称为序理想，它们按包含关系构成一个格，作为训练时的状态空间。"),
                        (r"The set $\{C\}$ is invalid because $C$ cannot be complete while its parents $A$ and $B$ are "
                         r"still missing.",
                         r"$\{C\}$ 不合法，因为在父节点 $A$、$B$ 都未完成时，$C$ 不可能已经完成。"),
                    ),
                ],
                [SVG("lattice_plain")],
                ratio="1fr 0.95fr", cls="mid",
            ),
        ],
        source=("v2 draft, Section 3; the order ideals of a finite partial order form a distributive lattice (Birkhoff).",
                "v2 草稿第 3 节；有限偏序集的序理想构成分配格（Birkhoff）。"),
    ),
    slide(
        "sharing-condition", "dp", "Condition for sharing a state", "状态共享的条件",
        status=["draft", "draft1"],
        blocks=[
            MATH(r"H_I=\State(x,I)", cls="key center lg"),
            P(r"Merging is exact only when every accepted route to $I$ yields the same canonical state, and therefore "
              r"the same model input.",
              r"只有当所有到达 $I$ 的接受路径都产生相同的规范化状态，从而产生相同的模型输入时，合并才是精确的。"),
            UL(
                ("Deterministic accepted paths.", "接受路径确定：",
                 "Tool results on accepted paths are deterministic, or they have a verified equivalent canonical form.",
                 "接受路径上的工具结果是确定的，或者具有经过验证的等价规范形式。"),
                ("Order-invariant reduction.", "归约与顺序无关：",
                 r"Swapping two certified independent actions leaves the reduced state unchanged, written as "
                 r"$\Reduce(h\,a\,o_a\,b\,o_b)=\Reduce(h\,b\,o_b\,a\,o_a)$.",
                 r"交换两个经认证独立的动作不会改变归约后的状态，记为 $\Reduce(h\,a\,o_a\,b\,o_b)=\Reduce(h\,b\,o_b\,a\,o_a)$。"),
                ("No hidden memory.", "无隐藏记忆：",
                 r"The policy carries no scratchpad across turns, so the same $H_I$ always produces the same model "
                 r"input.",
                 r"策略在回合之间不保留额外的草稿文本，因此相同的 $H_I$ 总是产生相同的模型输入。"),
            ),
            CALL("note",
                 r"Sharing saves computation only, since different routes to $I$ keep their own probabilities, which "
                 r"are summed rather than assumed equal.",
                 r"共享只是节省计算：到达 $I$ 的不同路径各自保留自己的概率，这些概率被相加，而不是被假设为相等。"),
        ],
        source=("v2 draft, Section 3.1; the three conditions are from the earlier draft, Sections 5.2, 5.5, and 7.2.",
                "v2 草稿第 3.1 节；三项条件来自较早草稿第 5.2、5.5 与 7.2 节。"),
    ),
    slide(
        "ready-set", "dp", "Ready set and step scores", "就绪集合与单步得分",
        status=["draft"],
        blocks=[
            MATH(r"\mathcal{R}_G(I)=\{v\in V\setminus I:\ \pa_G(v)\subseteq I\}"),
            MATH(r"w_\theta(I,v)=\pi_\theta(a_v\mid x,H_I),\qquad e_\theta(V)=\pi_\theta(\aend\mid x,H_V)", cls="key"),
            COLS(
                [TABLE(
                    ["$I$", r"$\mathcal{R}_G(I)$", T("Scores needed", "所需得分")],
                    [
                        [r"$\varnothing$", r"$\{A,B\}$", r"$w_\theta(\varnothing,A),\ w_\theta(\varnothing,B)$"],
                        [r"$\{A\}$", r"$\{B\}$", r"$w_\theta(\{A\},B)$"],
                        [r"$\{B\}$", r"$\{A\}$", r"$w_\theta(\{B\},A)$"],
                        [r"$\{A,B\}$", r"$\{C\}$", r"$w_\theta(\{A,B\},C)$"],
                        [r"$\{A,B,C\}$", r"$\varnothing$", r"$e_\theta(V)$"],
                    ], cls="compact")],
                [UL(
                    ("The ready set holds the unfinished operations whose prerequisites are all complete, so each one "
                     "is executable now.",
                     "就绪集合包含所有前提均已完成、但自身尚未执行的操作，因此其中每个操作现在都可以执行。"),
                    (r"$w_\theta(I,v)$ is the probability that the model emits exactly the call $a_v$ at the state "
                     r"$I$.",
                     r"$w_\theta(I,v)$ 是模型在状态 $I$ 下生成的调用恰好为 $a_v$ 的概率。"),
                    (r"$e_\theta(V)$ is the probability of ending the task correctly once every operation is complete.",
                     r"$e_\theta(V)$ 是所有操作完成之后正确结束任务的概率。"),
                    ("This example needs 6 scores, one for each lattice edge plus one for the terminal action.",
                     "本例共需 6 个得分：格中每条边一个，再加上终止动作一个。"),
                )],
                ratio="1fr 1.1fr",
            ),
        ],
        source=("v2 draft, Section 3.", "v2 草稿第 3 节。"),
    ),
    slide(
        "recursion", "dp", "Completion probability recursion", "完成概率的递推",
        status=["draft", "draft1"],
        blocks=[
            MATH(r"B_\theta(I)=\sum_{v\in\mathcal{R}_G(I)}w_\theta(I,v)\,B_\theta(I\cup\{v\}),\qquad "
                 r"B_\theta(V)=e_\theta(V)", cls="key center lg"),
            TERMS([
                (r"B_\theta(I)", T(r"Probability of finishing all remaining operations in some accepted order and then "
                                   r"ending correctly, starting from $I$",
                                   r"从 $I$ 出发，按任一接受顺序完成全部剩余操作并正确结束的概率")),
                (r"w_\theta(I,v)", T(r"Probability of executing $v$ next at state $I$", r"在状态 $I$ 下接着执行 $v$ 的概率")),
                (r"B_\theta(I\cup\{v\})", T(r"Completion probability after $v$ is done", r"执行 $v$ 之后的完成概率")),
                (r"B_\theta(V)=e_\theta(V)", T("Base case: every operation is done and only the terminal action remains",
                                               "边界条件：所有操作均已完成，只剩终止动作")),
            ], head=[T("Term", "项"), T("Meaning", "含义")], cls="compact"),
            P("Every accepted suffix begins with exactly one ready action, so summing over ready actions counts each "
              "suffix exactly once.",
              "每条接受后缀都恰好以一个就绪动作开头，因此对就绪动作求和时，每条后缀只被计算一次。"),
            CALL("corr",
                 r"v2 defines $e_\theta(V)$ but leaves the base case implicit, and the earlier draft states "
                 r"$B_\theta(V)=e_\theta(V)$ explicitly.",
                 r"v2 定义了 $e_\theta(V)$ 但没有写出边界条件，较早的草稿则明确给出了 $B_\theta(V)=e_\theta(V)$。"),
        ],
        source=("v2 draft, Section 3; correctness argument and its requirement of unambiguous action labels from the "
                "earlier draft, Section 7.3.",
                "v2 草稿第 3 节；正确性论证及其对动作标签无歧义的要求来自较早草稿第 7.3 节。"),
    ),
    slide(
        "worked-example", "dp", "Worked example, step by step", "逐步计算示例",
        status=["draft1", "ill"],
        blocks=[
            COLS(
                [
                    STEPS(
                        r"$B(V)=e(V)=@@eV@@$",
                        r"$B(\{A,B\})=@@wC@@\times@@eV@@=@@b_ab@@$",
                        (r"$B(\{A\})=@@wAB@@\times@@b_ab@@=@@b_a@@$, and $B(\{B\})=@@wBA@@\times@@b_ab@@=@@b_b@@$",
                         r"$B(\{A\})=@@wAB@@\times@@b_ab@@=@@b_a@@$，$B(\{B\})=@@wBA@@\times@@b_ab@@=@@b_b@@$"),
                        r"$B(\varnothing)=@@wA@@\times@@b_a@@+@@wB@@\times@@b_b@@=@@p_abc@@+@@p_bac@@=@@b0@@$",
                    ),
                    P(r"The total equals the sum over both accepted paths, while the shared suffix through "
                      r"$\{A,B\}$ is scored only once.",
                      r"总概率等于两条接受路径的概率之和，而经过 $\{A,B\}$ 的共享后缀只计分一次。"),
                    P(r"At $\varnothing$ the remaining @@leak0@@ of probability falls on actions outside the accepted "
                      r"set, such as calling $C$ too early.",
                      r"在 $\varnothing$ 处，剩余的 @@leak0@@ 概率落在接受集合之外的动作上，例如过早调用 $C$。"),
                ],
                [SVG("lattice_backward")],
                ratio="1fr 0.95fr", cls="mid",
            ),
        ],
        source=(r"Policy scores from the earlier draft, Section 13.1; values computed by the build script; $\theta$ "
                r"omitted.",
                r"策略得分取自较早草稿第 13.1 节；数值由构建脚本计算；省略 $\theta$。"),
    ),
    slide(
        "forward-view", "dp", "Forward view of the same sum", "同一求和的前向视角",
        status=["draft", "derived"],
        blocks=[
            MATH(r"F(I\cup\{v\})\mathrel{+}=F(I)\,w_\theta(I,v),\qquad F(\varnothing)=1", cls="key"),
            COLS(
                [
                    STEPS(
                        r"$F(\{A\})=@@F_a@@,\quad F(\{B\})=@@F_b@@$",
                        r"$F(\{A,B\})=@@wA@@\times@@wAB@@+@@wB@@\times@@wBA@@=@@F_ab_t1@@+@@F_ab_t2@@=@@F_ab@@$",
                        r"$F(V)=@@F_ab@@\times@@wC@@=@@F_v@@$",
                        r"$F(V)\,e_\theta(V)=@@F_v@@\times@@eV@@=@@b0@@=B_\theta(\varnothing)$",
                    ),
                    P(r"$F(I)$ is the total probability of reaching $I$ along accepted routes, and v2 introduces the "
                      r"method with this forward form.",
                      r"$F(I)$ 是沿接受路径到达 $I$ 的总概率；v2 正是用这种前向形式引入方法的。"),
                    P(r"Both directions give @@b0@@, which is the forward–backward identity for path sums on a "
                      r"directed acyclic graph.",
                      r"两个方向都得到 @@b0@@，这正是有向无环图上路径求和的前向–后向恒等式。"),
                ],
                [SVG("lattice_forward")],
                ratio="1fr 0.95fr", cls="mid",
            ),
            CALL("corr",
                 r"The v2 introduction omits the terminal action in $F$, and multiplying by $e_\theta(V)$ reconciles "
                 r"it with $B_\theta(\varnothing)$.",
                 r"v2 引言中的 $F$ 省略了终止动作，乘以 $e_\theta(V)$ 之后即与 $B_\theta(\varnothing)$ 一致。"),
        ],
        source=("v2 draft, solution item 3; numbers continue the worked example.",
                "v2 草稿，解决方案第 3 点；数值延续逐步计算示例。"),
    ),
    slide(
        "why-sharing", "dp", "Why shared states matter", "共享状态的作用",
        status=["derived", "est"],
        blocks=[
            TABLE(
                [T("Dependency structure", "依赖结构"), T("Accepted orders", "接受顺序数"),
                 T("Ordered prefixes", "有序前缀数"), T("Completed sets", "已完成集合数"),
                 T("Lattice edges", "格的边数")],
                [
                    [T("3 independent operations", "3 个独立操作"), "@@cnt_ind3_o@@", "@@cnt_ind3_p@@",
                     "@@cnt_ind3_s@@", "@@cnt_ind3_e@@"],
                    [T("10 independent operations", "10 个独立操作"), "@@cnt_ind10_o@@", "@@cnt_ind10_p@@",
                     "@@cnt_ind10_s@@", "@@cnt_ind10_e@@"],
                    [T("20 independent operations", "20 个独立操作"), "$@@cnt_ind20_o@@$", "$@@cnt_ind20_p@@$",
                     "@@cnt_ind20_s@@", "@@cnt_ind20_e@@"],
                    [T("4 chains of 5 operations", "4 条长度为 5 的链"), "@@cnt_ch4x5_o@@", "@@cnt_ch4x5_p@@",
                     "@@cnt_ch4x5_s@@", "@@cnt_ch4x5_e@@"],
                    [T("1 chain of 20 operations", "1 条长度为 20 的链"), "@@cnt_chain20_o@@", "@@cnt_chain20_p@@",
                     "@@cnt_chain20_s@@", "@@cnt_chain20_e@@"],
                ], num_cols=[1, 2, 3, 4], hl_rows=[1], cls="compact"),
            UL(
                (r"With $n$ independent operations there are $n!$ orders but only $2^n$ completed sets, so 10 "
                 r"operations give @@cnt_ind10_o@@ orders and @@cnt_ind10_s@@ states.",
                 r"$n$ 个独立操作有 $n!$ 种顺序，但只有 $2^n$ 个已完成集合；10 个操作对应 @@cnt_ind10_o@@ 种顺序和 "
                 r"@@cnt_ind10_s@@ 个状态。"),
                (r"Ordered prefixes count the nodes of a prefix tree, so prefix caching alone cannot merge $A$ then "
                 r"$B$ with $B$ then $A$ (my synthesis).",
                 r"有序前缀数即前缀树的节点数，因此仅靠前缀缓存无法合并“先 $A$ 后 $B$”与“先 $B$ 后 $A$”（个人归纳）。"),
                ("The state count can still grow exponentially, and the saving exists only when mergeable states "
                 "actually occur, as v2 notes.",
                 "状态数仍可能指数增长；正如 v2 所指出的，只有在确实存在可合并的状态时，才会有压缩收益。"),
            ),
        ],
        source=("Counts computed by the build script; 1,296 matches the earlier draft, Section 10.1. Counting linear "
                "extensions is #P-complete (Brightwell and Winkler, 1991).",
                "计数由构建脚本得出；1,296 与较早草稿第 10.1 节一致。线性扩展计数是 #P 完全问题（Brightwell 与 "
                "Winkler，1991）。"),
    ),
]

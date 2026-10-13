from deck.blocks import CALL, CARD, CARDS, P, T, TABLE, UL, slide

SLIDES = [
    slide(
        "related-methods", "pos", "Related methods", "相关方法",
        status=["est", "draft1"],
        blocks=[
            TABLE(
                [T("Work", "工作"), T("What it does", "主要内容"), T("Difference stated in the earlier draft", "较早草稿所述的区别")],
                [
                    [T("Order-centric augmentation (He et al., 2025)", "顺序中心增强（He 等，2025）"),
                     T("Generates valid reorderings from a dependency DAG as extra training data",
                       "依据依赖 DAG 生成合法重排，作为额外训练数据"),
                     T("Optimizes the marginal probability of the accepted set and covers intermediate states",
                       "优化接受集合的边际概率，并覆盖中间状态")],
                    [T("De-linearizing agent traces (Li et al., 2026)", "智能体轨迹去线性化（Li 等，2026）"),
                     T("Infers latent partial orders from linear traces for faster execution",
                       "从线性轨迹推断潜在偏序，用于更高效的执行"),
                     T("Uses a certified partial order as supervision to update the LLM policy",
                       "把已认证的偏序作为监督信号，用于更新 LLM 策略")],
                    [T("Order marginalization in graph generation (Chen et al., ICML 2021)",
                       "图生成中的顺序边际化（Chen 等，ICML 2021）"),
                     T("Treats node order as latent and trains a variational bound", "把节点顺序视为潜变量，训练其变分界"),
                     T("Handles actions, observations, and executability in an interactive environment",
                       "处理交互环境中的动作、观察与可执行性")],
                    [T("Dynamic oracles (Goldberg and Nivre, TACL 2013)", "动态 oracle（Goldberg 与 Nivre，TACL 2013）"),
                     T("Allow several correct transitions when training structured parsers",
                       "在结构化解析器训练中允许多个正确转移"),
                     T("Scores whole accepted suffixes with exact state merging", "对整条接受后缀计分，并精确合并状态")],
                    [T("LLMCompiler (Kim et al., ICML 2024) †", "LLMCompiler（Kim 等，ICML 2024）†"),
                     T("Plans and runs independent function calls in parallel at inference time",
                       "在推理时规划并并行执行相互独立的函数调用"),
                     T("Changes the training objective rather than the executor", "改变的是训练目标，而不是执行器")],
                ], cls="compact"),
            P("The earlier draft claims novelty only for the combination of certified equivalence, an online state "
              "interface, and completion likelihood with intermediate starts.",
              "较早的草稿只把创新点限定为三者的组合：经认证的执行等价、在线状态接口，以及带中间起点的完成似然训练。"),
            P("It explicitly does not claim to be the first to use partial orders, to sum over orders, or to allow "
              "several correct next steps.",
              "它明确不主张自己是首个使用偏序、首个对顺序求和或首个允许多个正确下一步的工作。"),
        ],
        source=("The first four rows follow the earlier draft, Section 2.2; † marks a row added for this deck (my synthesis).",
                "前四行依据较早草稿第 2.2 节；† 表示本报告补充的一行（个人归纳）。"),
    ),
    slide(
        "foundations", "pos", "Foundations from concurrency and sequence modeling", "并发理论与序列建模中的基础",
        status=["est"],
        blocks=[
            TABLE(
                [T("Result", "结论"), T("Statement", "内容"), T("Role in this work", "在本工作中的作用")],
                [
                    [T("Bernstein conditions (1966)", "Bernstein 条件（1966）"),
                     T("Disjoint read and write sets let two program segments run in either order",
                       "读写集合互不相交时，两个程序段可以按任意顺序执行"),
                     T("Certifies which tool calls commute", "认证哪些工具调用可以交换")],
                    [T("Partial-order reduction (Godefroid, 1996)", "偏序归约（Godefroid，1996）"),
                     T("Explores one representative of each class of equivalent interleavings",
                       "对每一类等价交错只探索一个代表"),
                     T("Same equivalence, used to define a training target instead of pruning a search",
                       "同样的等价关系，在此用于定义训练目标而非剪枝搜索")],
                    [T("Order ideals and linear extensions (Brightwell and Winkler, 1991)",
                       "序理想与线性扩展（Brightwell 与 Winkler，1991）"),
                     T("Counting linear extensions is #P-complete, while dynamic programming over ideals is exact",
                       "线性扩展计数是 #P 完全问题，而在序理想上的动态规划可以精确求解"),
                     T("Explains why the lattice of completed sets is the state space",
                       "说明为何以已完成集合构成的格作为状态空间")],
                    [T("CTC (Graves et al., ICML 2006)", "CTC（Graves 等，ICML 2006）"),
                     T("Sums over all alignments of one label sequence with forward–backward recursions",
                       "用前向–后向递推对同一标签序列的所有对齐求和"),
                     T("Same dynamic-programming pattern, with completed sets in place of alignments",
                       "同样的动态规划模式，只是把对齐换成了已完成集合")],
                ]),
            P("The results in the first two columns are established, and the CTC analogy in the last row is an "
              "interpretation (my synthesis).",
              "前两列中的结论均为已有结论，最后一行与 CTC 的类比属于解读（个人归纳）。"),
        ],
        source=("Sources listed on the references slide; the drafts cite partial-order methods (Godefroid) but not "
                "Bernstein or CTC.",
                "出处见参考文献页；草稿引用了偏序方法（Godefroid），但没有引用 Bernstein 或 CTC。"),
    ),
    slide(
        "assumptions", "pos", "Assumptions and scope", "假设与适用范围",
        status=["draft1", "draft"],
        blocks=[
            UL(
                ("Deterministic accepted paths.", "接受路径确定：",
                 "Accepted actions return fixed results under the snapshot, while invalid inputs may still produce "
                 "random errors, as in FuncBenchGen.",
                 "在快照下，接受动作返回确定的结果；无效输入仍可能产生随机错误，FuncBenchGen 即是如此。"),
                ("Finite demonstrations.", "有限示范：",
                 "Loops are unrolled to the number of iterations actually observed, and the graph does not cover other "
                 "iteration counts.",
                 "循环按实际发生的次数展开，目标图不覆盖其他循环次数。"),
                ("Conservative graphs.", "保守构图：",
                 "A wrong independence claim adds illegal executions to positive supervision, which the earlier draft "
                 "names the highest-priority risk.",
                 "错误的独立性判断会把非法执行加入正向监督，较早的草稿把它列为最高优先级的风险。"),
                ("Successful prefixes only.", "只含成功前缀：",
                 "The objective covers legal partial executions and does not teach recovery after tool failures or "
                 "wrong writes.",
                 "该目标只覆盖合法的部分执行，不学习工具失败或错误写入之后的恢复。"),
                ("One accepted set per task.", "每个任务一个接受集合：",
                 r"Solutions that use other tools or algorithms lie outside $\mathcal{C}(G)$, so evaluation must use "
                 r"environment success.",
                 r"使用其他工具或算法的解不属于 $\mathcal{C}(G)$，因此评估必须以环境中的任务成功为准。"),
            ),
        ],
        source=("Earlier draft, Sections 3.2, 15.1, 15.4, and 15.6; FuncBenchGen, Section 3.4.",
                "较早草稿第 3.2、15.1、15.4 与 15.6 节；FuncBenchGen 第 3.4 节。"),
    ),
    slide(
        "claim-status", "pos", "Status of the claims", "各项主张的现状",
        status=["est", "hyp"],
        blocks=[
            CARDS(
                CARD("Established", "已有结论",
                     UL(("FuncBenchGen failure statistics and the reported gain from restating known variable values.",
                         "FuncBenchGen 的错误统计，以及重述已知变量值带来的收益。"),
                        ("Bernstein conditions as a sufficient test for safe reordering.",
                         "Bernstein 条件作为安全交换的充分判据。"),
                        ("Exactness of dynamic programming over order ideals for path sums.",
                         "在序理想上用动态规划求路径概率和的精确性。"),
                        ("The marginal-likelihood gradient identity and the augmentation decomposition.",
                         "边际似然梯度恒等式与顺序增强损失的分解。"),
                        label=True)),
                CARD("Derived in the drafts", "草稿中的推导",
                     UL(("Correctness of the recursion under determinism, consistent states, and unambiguous actions.",
                         "在确定性、状态一致与动作无歧义条件下递推的正确性。"),
                        (r"Bounds $B^-\le B\le B^+$ and the logarithmic gap bound.",
                         r"上下界 $B^-\le B\le B^+$ 及其对数间隙界。"),
                        (r"Coverage bound for resume distributions dominated by $\mu_G$.",
                         r"接续分布受 $\mu_G$ 控制时的覆盖界。"),
                        (r"Self-check on 30 random DAGs: differences of at most $@@selfcheck_p@@$ and $@@selfcheck_g@@$.",
                         r"在 30 个随机 DAG 上的自检：差异分别不超过 $@@selfcheck_p@@$ 与 $@@selfcheck_g@@$。"),
                        label=True)),
                CARD("Hypotheses, not yet tested", "假设，尚未检验",
                     UL(("H1: removing irrelevant order constraints helps learning under a limited budget.",
                         "H1：去除无关的顺序约束有助于有限预算下的学习。"),
                        ("H2: training from intermediate states improves resumption.",
                         "H2：从中间状态训练能提高接续能力。"),
                        ("H3: shared states make marginal training affordable.",
                         "H3：共享状态使边际训练的成本可以接受。"),
                        ("H4: certification quality determines effectiveness.",
                         "H4：认证质量决定方法是否有效。"),
                        ("H5: gains grow with the real ordering freedom of tasks.",
                         "H5：收益随任务真实的顺序自由度增大。"),
                        label=True)),
            ),
            P("No LLM training or benchmark results exist yet, so every performance statement in this deck remains a "
              "hypothesis.",
              "目前还没有任何 LLM 训练或基准测试结果，因此本报告中所有关于性能的表述都仍是假设。"),
        ],
        source=("Earlier draft, Sections 12, 13.2, and 14.1; FuncBenchGen, Table 3 and Section 4.6.1.",
                "较早草稿第 12、13.2 与 14.1 节；FuncBenchGen 表 3 与第 4.6.1 节。"),
    ),
    slide(
        "baselines", "pos", "Planned baselines and controls", "计划中的基线与控制变量",
        status=["draft1"],
        blocks=[
            TABLE(
                [T("Baseline", "基线"), T("Definition", "定义"), T("Controls for", "控制的因素")],
                [
                    ["Canonical-State SFT", T("Same state interface as the method, trained on one order",
                                              "与所提方法相同的状态接口，只训练一种顺序"),
                     T("Gain from the state interface", "状态接口带来的收益")],
                    ["Order-Augmented SFT", T("Dependency-preserving reorderings as extra sequences",
                                              "使用保持依赖的重排作为额外序列"),
                     T("Gain from seeing more orders", "多看顺序带来的收益")],
                    ["Prefix-Matched SFT", T("Same start prefixes as the method", "使用与所提方法相同的起始前缀"),
                     T("Gain from extra state coverage", "额外状态覆盖带来的收益")],
                    [T("Frontier supervision", "前沿集合监督"),
                     r"$-\log\sum_{v\in\mathcal{R}_G(I)}\pi_\theta(a_v\mid x,H_I)$",
                     T("Need for marginalizing whole suffixes", "对整条后缀做边际化是否必要")],
                    [T("Sampled-order marginalization", "采样顺序边际化"),
                     T("Probability sum over sampled, deduplicated accepted orders", "对采样并去重后的接受顺序求概率和"),
                     T("Value of the shared-state algorithm", "共享状态算法的价值")],
                ], hl_rows=[3], cls="compact"),
            UL(
                ("All baselines receive the same variable table, because FuncBenchGen shows that restating known "
                 "values alone changes success rates.",
                 "所有基线都获得相同的变量表，因为 FuncBenchGen 表明，仅重述已知变量值就会改变成功率。"),
                ("Argument interfaces must also match across methods, either all handles or all literal values, "
                 "otherwise the task itself changes.",
                 "各方法的参数接口也必须一致，要么都用句柄，要么都直接输出数值，否则任务本身就变了。"),
            ),
        ],
        source=("Earlier draft, Sections 14.5 and 14.6.", "较早草稿第 14.5 与 14.6 节。"),
    ),
    slide(
        "metrics-falsification", "pos", "Metrics and falsification criteria", "评价指标与证伪标准",
        status=["draft1"],
        blocks=[
            P("Planned testbeds are FuncBenchGen extended with effect contracts and restorable snapshots, and an "
              "instrumented subset of AppWorld tasks.",
              "计划的测试环境包括：加入作用契约与可恢复快照的扩展版 FuncBenchGen，以及经过插桩的 AppWorld 任务子集。"),
            CARDS(
                CARD("Metrics", "评价指标",
                     UL(("Task success rate in the real environment, judged by a programmatic state check.",
                         "真实环境中的任务成功率，由程序化的状态检查判定。"),
                        ("Resume success from valid prefixes, with the backend restored to the matching state.",
                         "从合法前缀接续的成功率，后端需恢复到对应状态。"),
                        ("Auxiliary errors: argument provenance, unknown-variable use, redundant calls, early "
                         "termination, and side effects.",
                         "辅助错误：参数来源、使用未知变量、冗余调用、提前终止与副作用。"),
                        ("Full cost: training tokens, GPU time, scoring calls, environment calls, and graph "
                         "construction.",
                         "完整成本：训练 token 数、GPU 时间、计分次数、环境调用与构图开销。"),
                        label=True)),
                CARD("Falsification criteria", "证伪标准",
                     UL(("Simple methods suffice: a canonical order or order augmentation matches the method at equal "
                         "budget.",
                         "简单方法已足够：在相同预算下，规范顺序或顺序增强达到同等表现。"),
                        ("Local supervision suffices: frontier supervision reaches the same result at much lower cost.",
                         "局部监督已足够：前沿集合监督以低得多的成本达到同等效果。"),
                        ("Too little structure: most real task graphs reduce to chains, or certification costs too "
                         "much.",
                         "可用结构太少：大多数真实任务图退化为链，或认证成本过高。"),
                        ("Representation dominates: gains come almost entirely from the variable table or canonical "
                         "state.",
                         "表示占主导：收益几乎全部来自变量表或规范化状态。"),
                        ("Decoding gap: likelihood gains do not turn into success under the deployed decoding strategy.",
                         "解码落差：似然的改善不能转化为实际部署解码策略下的成功。"),
                        label=True)),
            ),
        ],
        source=("Earlier draft, Sections 14.2, 14.3, 14.7, and 14.10.", "较早草稿第 14.2、14.3、14.7 与 14.10 节。"),
    ),
    slide(
        "open-questions", "pos", "Open questions", "待解决的问题",
        status=["draft1", "draft"],
        blocks=[
            UL(
                ("Certification at scale.", "大规模认证：",
                 "Reliable read and write contracts for real tools are hard to obtain, and their coverage limits "
                 "which tasks can enter training.",
                 "真实工具可靠的读写契约难以获得，其覆盖范围决定了哪些任务能进入训练。"),
                (r"Choice of $\mu_{\mathrm{cut}}$.", r"$\mu_{\mathrm{cut}}$ 的选择：",
                 "v2 leaves the cut distribution unspecified, while the earlier draft samples a depth and then ready "
                 "nodes up to it.",
                 "v2 没有规定截断分布，较早的草稿则先采样截断深度，再逐步采样就绪节点直到该深度。"),
                ("Posterior concentration.", "后验集中：",
                 "If the model starts with one preferred order, the success posterior concentrates on it, and prefix "
                 "coverage only partly counters this.",
                 "若模型初始时偏好某一种顺序，成功后验就会集中到这一顺序上，前缀覆盖只能部分缓解这一问题。"),
                ("Cost of canonical states.", "规范化状态的成本：",
                 "Rendered states need not share a text prefix, so reusing the key–value cache across states is not "
                 "free.",
                 "渲染后的状态文本未必共享前缀，因此跨状态复用键值缓存并非没有代价。"),
                ("Stochastic tools and several strategies.", "随机工具与多种策略：",
                 "Random tool outputs and multiple solution graphs need explicit outcome branches or unions of "
                 "accepted sets.",
                 "随机的工具输出与多种解法需要显式建模结果分支，或者对多个接受集合取并集。"),
            ),
        ],
        source=("Earlier draft, Sections 8.4, 15.1, 15.3, 15.5, 16.1, and 16.3.",
                "较早草稿第 8.4、15.1、15.3、15.5、16.1 与 16.3 节。"),
    ),
]

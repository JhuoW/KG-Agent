from deck.blocks import CALL, COLS, FLOW, LANE, MATH, UL, slide

SLIDES = [
    slide(
        "training-procedure", "alg", "Training procedure (Algorithm 1)", "训练流程（算法 1）",
        status=["draft"],
        blocks=[
            FLOW(
                LANE("Offline preparation", "离线准备",
                     ("Record actions, argument sources, observations, state versions, and control conditions",
                      "记录动作、参数来源、观察、状态版本与控制条件"),
                     (r"Check all relevant action pairs and build a conservative target graph $G$",
                      r"检查所有相关动作对，构造保守的目标图 $G$"),
                     ("Verify unambiguous action events, demonstration success, and swap evidence",
                      "验证动作事件无歧义、示范成功与交换证据"),
                     ("Split train, validation, and test sets by task and template",
                      "按底层任务与模板划分训练、验证与测试集"), per=4),
                LANE("Each parameter update", "每次参数更新",
                     (r"Sample tasks with fixed graphs, then root and valid prefixes from $\mu_G$",
                      r"采样任务及其固定目标图，再从 $\mu_G$ 采样根状态与合法前缀"),
                     (r"Build $H_I$ from actual observations of completed actions",
                      r"仅用已完成动作的实际观察构造 $H_I$"),
                     ("Small graph: full lattice. Large graph: bounded expansion",
                      "小图：完整的格。大图：有界展开", True),
                     (r"Compute $B$ or a labeled lower bound $B^-$, then $q$, $m$, $\rho$",
                      r"计算 $B$ 或明确标记的下界 $B^-$，再计算 $q$、$m$、$\rho$"),
                     ("Fix posterior weights and recompute weighted token log probabilities",
                      "固定后验权重，重算加权的 token 对数概率"),
                     ("Accumulate gradients over micro-batches, then take one optimizer step",
                      "跨微批次累积梯度，然后执行一次优化器更新")),
            ),
            UL(
                ("Only the LLM or its parameter-efficient adapter is trained, while the graph compiler and the state "
                 "interface stay fixed.",
                 "只训练 LLM 或其参数高效适配器，图编译器与状态接口保持固定。"),
                ("All compared methods share format warm-up, tool descriptions, state templates, and comparable "
                 "optimization, which isolates the effect of the objective.",
                 "所有对比方法共享格式预热、工具说明、状态模板和可比的优化配置，以便单独识别目标函数的作用。"),
            ),
        ],
        source=("v2 draft, Algorithm 1.", "v2 草稿，算法 1。"),
    ),
    slide(
        "bounded-computation", "alg", "Exact and bounded computation", "精确计算与有界近似",
        status=["draft1"],
        blocks=[
            COLS(
                [
                    MATH(r"B^-_\theta(I)\;\le\;B_\theta(I)\;\le\;B^+_\theta(I)", cls="key center lg"),
                    MATH(r"0\;\le\;-\log B^-_\theta(I)+\log B_\theta(I)\;\le\;\log\frac{B^+_\theta(I)}{B^-_\theta(I)}",
                         cls="center"),
                    MATH(r"O(w\,N_G),\qquad N_G=|\mathcal{J}(G)|\le\prod_{j=1}^{k}(\ell_j+1)",
                         cls="sm center"),
                    CALL("note",
                         r"v2 names $B^-$, $B^+$, $C$, and $\varepsilon$ in Algorithm 1, but only the earlier draft "
                         r"defines them, in Section 10.",
                         r"v2 在算法 1 中提到了 $B^-$、$B^+$、$C$ 与 $\varepsilon$，但只有较早草稿的第 10 节给出了定义。"),
                ],
                [UL(
                    (r"Small graphs use the full lattice, and large graphs expand only part of it within the budget "
                     r"$C$.",
                     r"小图使用完整的格，大图只在预算 $C$ 内展开其中一部分。"),
                    (r"Unexpanded states take $B^-=0$ and $B^+=1$, and the same recursion then yields guaranteed bounds "
                     r"on $B_\theta(I)$.",
                     r"未展开状态取 $B^-=0$、$B^+=1$，再用同一递推即可得到 $B_\theta(I)$ 的确定上下界。"),
                    (r"Each sampled start is seeded with one complete accepted suffix, so $B^-$ stays positive and the "
                     r"loss stays finite.",
                     r"每个采样起点先播种一条完整的接受后缀，使 $B^-$ 保持为正，损失保持有限。"),
                    (r"When the budget runs out, training uses $-\log B^-$, an upper bound on the true loss, and "
                     r"reports it as such.",
                     r"预算耗尽时，训练使用真实损失的上界 $-\log B^-$，并须如实标注。"),
                    (r"Exact computation needs $O(w\,N_G)$ action scores, where $w$ is the largest ready-set size "
                     r"and $\ell_j$ are the chain lengths of a chain decomposition of $G$.",
                     r"精确计算需要 $O(w\,N_G)$ 次动作计分，其中 $w$ 是最大就绪集合的大小，$\ell_j$ 是 $G$ 的某个链分解中各条链的长度。"),
                )],
                ratio="1fr 1fr",
            ),
        ],
        source=("Earlier draft, Sections 10.1 to 10.5.", "较早草稿第 10.1 至 10.5 节。"),
    ),
    slide(
        "inference-procedure", "alg", "Inference procedure (Algorithm 2)", "推理流程（算法 2）",
        status=["draft", "draft1"],
        blocks=[
            FLOW(
                LANE("Loop until the terminal action or the task budget", "循环，直到终止动作或任务预算耗尽",
                     (r"Initialize $H_0$", r"初始化 $H_0$"),
                     (r"Generate a complete action $a_t$ from $x$ and $\Render(H_t)$",
                      r"根据 $x$ 与 $\Render(H_t)$ 生成完整动作 $a_t$"),
                     (r"If $a_t$ is the terminal action, output the result", r"若 $a_t$ 为终止动作，则输出结果"),
                     (r"Otherwise execute the call and receive the observation $o_t$",
                      r"否则执行该调用，获得观察 $o_t$"),
                     (r"Update $H_{t+1}$ with runtime information", r"用运行时信息更新 $H_{t+1}$"), per=5),
            ),
            UL(
                (r"At inference the agent sees only $x$ and $\Render(H_t)$, and it never accesses $G$, the lattice, "
                 r"$B_\theta$, or $q_\theta$.",
                 r"推理时智能体只看到 $x$ 与 $\Render(H_t)$，不会访问 $G$、完成格、$B_\theta$ 或 $q_\theta$。"),
                ("Training and inference share one implementation of Reduce and Render, so the model always sees the "
                 "same state format.",
                 "训练与推理使用同一套 Reduce 与 Render 实现，因此模型看到的状态格式始终一致。"),
            ),
            CALL("warn",
                 "The class probability is defined under sampling, and greedy decoding can behave quite differently. "
                 "Suppose ten correct first actions have probability @@dec_each@@ each and one wrong action has "
                 "@@dec_bad@@. Sampling then picks a correct first action with probability @@dec_succ@@, but greedy "
                 "decoding always picks the wrong one.",
                 "类概率是在采样策略下定义的，贪心解码的表现可能截然不同。设十个正确首动作的概率各为 @@dec_each@@，一个错误"
                 "首动作的概率为 @@dec_bad@@。此时采样以 @@dec_succ@@ 的概率选中正确首动作，而贪心解码总是选中错误动作。"),
        ],
        source=("v2 draft, Algorithm 2; the access rule and the decoding example are from the earlier draft, Sections "
                "11.2 and 11.3.",
                "v2 草稿，算法 2；访问限制与解码示例来自较早草稿第 11.2 与 11.3 节。"),
    ),
]

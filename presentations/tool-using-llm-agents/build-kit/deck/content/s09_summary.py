from deck.blocks import CALL, COLS, HTML, T, TABLE, UL, slide

REFS = [
    "Maekawa, S., Hassell, J., Pezeshkpour, P., Mitchell, T., Hruschka, E. Towards Reliable Benchmarking: A "
    "Contamination Free, Controllable Evaluation Framework for Multi-step LLM Function Calling. ICLR 2026. "
    "arXiv:2509.26553.",
    "He, Q., He, Q., Liang, J., Xiao, Y., Zhou, W., Sun, Z., Yu, F. Order Doesn't Matter, But Reasoning Does: "
    "Training LLMs with Order-Centric Augmentation. arXiv:2502.19907, 2025.",
    "Li, D., Cheng, Z., Nicholls, G. K., Kong, Q. De-Linearizing Agent Traces: Bayesian Inference of Latent Partial "
    "Orders for Efficient Execution. arXiv:2602.02806, 2026.",
    "Chen, X., Han, X., Hu, J., Ruiz, F., Liu, L. Order Matters: Probabilistic Modeling of Node Sequence for Graph "
    "Generation. ICML 2021, PMLR 139:1630–1639.",
    "Goldberg, Y., Nivre, J. Training Deterministic Parsers with Non-Deterministic Oracles. TACL 1:403–414, 2013.",
    "Kim, S., Moon, S., Tabrizi, R., Lee, N., Mahoney, M., Keutzer, K., Gholami, A. An LLM Compiler for Parallel "
    "Function Calling. ICML 2024, PMLR 235:24370–24391.",
    "Graves, A., Fernández, S., Gomez, F., Schmidhuber, J. Connectionist Temporal Classification: Labelling "
    "Unsegmented Sequence Data with Recurrent Neural Networks. ICML 2006, 369–376.",
    "Godefroid, P. Partial-Order Methods for the Verification of Concurrent Systems. LNCS 1032, Springer, 1996.",
    "Bernstein, A. J. Analysis of Programs for Parallel Processing. IEEE Transactions on Electronic Computers "
    "EC-15(5):757–763, 1966.",
    "Brightwell, G., Winkler, P. Counting Linear Extensions. Order 8(3):225–242, 1991.",
    "Trivedi, H., Khot, T., Hartmann, M., Manku, R., et al. AppWorld: A Controllable World of Apps and People for "
    "Benchmarking Interactive Coding Agents. ACL 2024, 16022–16076.",
]

SLIDES = [
    slide(
        "summary", "end", "Summary", "总结",
        blocks=[
            UL(
                (r"Sequence targets mix necessary dependencies with arbitrary order, so the proposed target is the set "
                 r"of accepted executions $\mathcal{C}(G)$.",
                 r"序列目标把必要依赖与偶然顺序混在一起，因此所提方法的训练目标是接受执行集合 $\mathcal{C}(G)$。"),
                ("A canonical execution state turns equivalent histories into identical model inputs, which makes "
                 "probability sharing across histories exact.",
                 "规范化执行状态让等价历史成为完全相同的模型输入，从而使跨历史的概率共享成为精确计算。"),
                ("Dynamic programming over completed sets computes the marginal likelihood, and its gradient weights "
                 "each path by its success posterior.",
                 "在已完成集合上的动态规划给出边际似然，其梯度按成功后验对每条路径加权。"),
                ("Training from valid intermediate states targets resumption, a failure mode that root-only training "
                 "cannot observe.",
                 "从合法中间状态开始训练针对的是接续能力，而仅从根状态训练无法观察到这类失败。"),
            ),
            CALL("hyp",
                 "Whether these changes improve agents at a fixed budget is still untested, and the answer depends on "
                 "reliable certification of independence.",
                 "这些改变能否在固定预算下提升智能体仍未经检验，其结论取决于能否可靠地认证动作之间的独立性。"),
        ],
    ),
    slide(
        "notation", "app", "Notation", "符号表",
        blocks=[
            COLS(
                [TABLE(
                    [T("Symbol", "符号"), T("Meaning", "含义")],
                    [
                        [r"$x=(u,\mathcal{T},o_0)$", T("Task input: request, tools, initial observation", "任务输入：请求、工具、初始观察")],
                        [r"$\zeta,\ \Exec_\zeta$", T("Environment snapshot and its executor", "环境快照及其执行器")],
                        [r"$\tau,\ \aend$", T("Demonstration and terminal action", "示范轨迹与终止动作")],
                        [r"$G=(V,E,\mathcal{A},\mathcal{O},\mathcal{K})$", T("Target execution graph", "目标执行图")],
                        [r"$k_v=(a_v,R_v,W_v,P_v,O_v)$", T("Operation record", "操作记录")],
                        [r"$\LinExt(G),\ \mathcal{C}(G)$", T("Linear extensions; accepted executions", "线性扩展；接受执行集合")],
                        [r"$h_t,\ H_t$", T("Raw history; canonical state", "原始历史；规范化状态")],
                        [r"$\Reduce,\ \Render$", T("History-to-state map; state-to-text map", "历史到状态的映射；状态到文本的映射")],
                    ], cls="compact terms")],
                [TABLE(
                    [T("Symbol", "符号"), T("Meaning", "含义")],
                    [
                        [r"$I,\ \mathcal{J}(G)$", T("Completed set; all order ideals", "已完成集合；全部序理想")],
                        [r"$\pa_G(v),\ \mathcal{R}_G(I)$", T("Parents of $v$; ready set at $I$", "$v$ 的父节点；$I$ 处的就绪集合")],
                        [r"$w_\theta(I,v),\ e_\theta(V)$", T("Step score; terminal score", "单步得分；终止得分")],
                        [r"$B_\theta(I),\ F(I)$", T("Completion probability; forward probability", "完成概率；前向概率")],
                        [r"$q_\theta,\ m_\theta,\ \rho_\theta$", T("Success posterior; occupancy; transition weight", "成功后验；占用量；转移权重")],
                        [r"$\mu_G,\ \lambda,\ K$", T("Start distribution; non-root weight; sample count", "起点分布；非根权重；采样数")],
                        [r"$B^-_\theta,\ B^+_\theta,\ C,\ \varepsilon$", T("Lower and upper bounds; budget; tolerance", "下界与上界；预算；误差阈值")],
                        [r"$\Lroot,\ \LQET$", T("Root loss; loss with intermediate starts", "根状态损失；含中间起点的损失")],
                    ], cls="compact terms")],
                ratio="1fr 1fr",
            ),
        ],
        source=("Notation follows the v2 method section and the earlier draft, Section 3.3.",
                "符号沿用 v2 方法部分与较早草稿第 3.3 节。"),
    ),
    slide(
        "references", "app", "References", "参考文献",
        blocks=[
            HTML('<ol class="refs">' + "".join(f"<li>{r}</li>" for r in REFS) + "</ol>"),
        ],
        source=("Working drafts: @@METHOD@@ v2 (Notion, edited 2026-10-13) and the earlier draft (Notion, edited 2026-10-08).",
                "工作稿：@@METHOD@@ v2（Notion，2026-10-13 编辑）及较早草稿（Notion，2026-10-08 编辑）。"),
    ),
]

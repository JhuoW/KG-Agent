from deck.blocks import (CALL, CALLLINE, CARD, CARDS, COLS, KV, MATH, P, QUOTE, SVG, T, TABLE, UL,
                         slide)

SRC_V2_MOT = ("v2 draft, motivation section.", "v2 草稿，动机部分。")

SLIDES = [
    slide(
        "running-example", "mot", "Running example: two loads and one comparison", "示例：两次读取与一次比较",
        status=["draft"],
        blocks=[
            QUOTE("“Load dataset A and dataset B, then compare their statistics.”",
                  "“分别读取数据集 A 和数据集 B，然后比较二者的统计结果。”"),
            MATH(r"a=\operatorname{load}(\mathrm{A}),\qquad b=\operatorname{load}(\mathrm{B}),\qquad "
                 r"c=\operatorname{compare}\big(\operatorname{out}(a),\operatorname{out}(b)\big)"),
            COLS(
                [
                    MATH(r"a\prec c,\qquad b\prec c", cls="key"),
                    P(r"With fixed data snapshots and non-interfering loads, the only required orderings are $a$ "
                      r"before $c$ and $b$ before $c$.",
                      r"在数据快照固定、两次读取互不干扰的条件下，唯一必须满足的顺序是 $a$ 先于 $c$、$b$ 先于 $c$。"),
                    P(r"Nothing requires $a$ to precede $b$, so $(a,b,c)$ and $(b,a,c)$ are two executions of the same "
                      r"dependency-structured computation.",
                      r"任务并不要求 $a$ 先于 $b$，因此 $(a,b,c)$ 与 $(b,a,c)$ 是同一个依赖结构计算的两种执行。"),
                ],
                [SVG("dag_abc")],
                ratio="1fr 1.1fr", cls="mid",
            ),
        ],
        source=SRC_V2_MOT,
    ),
    slide(
        "semantics-vs-serialization", "mot", "Task semantics and serialization choice", "任务语义与序列化选择",
        status=["draft"],
        blocks=[
            MATH(r"\LSFT=-\log P_\theta(a,b,c\mid x)"),
            COLS(
                [
                    UL(
                        ("Task semantics.", "任务语义：",
                         "They specify which actions and data dependencies are necessary, for example both loads must "
                         "finish before the comparison runs.",
                         "规定哪些动作和数据依赖是必要的，例如两次读取都必须在比较之前完成。"),
                        ("Serialization choice.", "序列化选择：",
                         "It records which independent action the demonstrator happened to write first, here the load "
                         "of A before the load of B.",
                         "记录示范者恰好先写下了哪个独立动作，这里是先读取 A、再读取 B。"),
                    ),
                    P("The SFT target encodes both as the same relation, one span preceding another, so the loss "
                      "cannot separate them.",
                      "SFT 目标把两者编码成同一种关系，即一段内容出现在另一段之前，因此损失函数无法将二者区分开。"),
                ],
                [SVG("seq_target")],
                ratio="1fr 1fr", cls="mid",
            ),
            CALL("draft",
                 "For tool-using agents the problem is that the supervision signal may impose constraints that the "
                 "task itself does not require.",
                 "对工具智能体而言，问题在于监督信号可能施加了任务本身并不需要的约束。"),
        ],
        source=SRC_V2_MOT,
    ),
    slide(
        "consequences", "mot", "Consequences for tool-using agents", "对工具智能体的影响",
        status=["draft"],
        blocks=[
            CARDS(
                CARD("Data and compute", "数据与算力",
                     P("One computation can admit many valid orders, so teaching each order as a separate example "
                       "spends data and compute repeatedly on one computation.",
                       "同一计算可能有许多合法顺序；若把每种顺序都当作单独样本来教，数据和算力就会被重复花在同一个计算上。")),
                CARD("Resuming partial work", "接续部分完成的工作",
                     P("An agent may need to continue work that another executor partly completed, and one memorized "
                       "order from scratch does not guarantee this ability.",
                       "智能体可能需要接续其他执行者已部分完成的工作，而从头记住一种顺序并不能保证具备这种能力。")),
                CARD("Argument availability", "参数可用性",
                     P("An agent must know which arguments are already available, which prerequisites are unmet, and "
                       "which call produced each value.",
                       "智能体需要知道哪些参数已经可用、哪些前提尚未满足，以及每个值由哪次调用产生。")),
            ),
            CALL("draft",
                 "The draft concludes that execution progress and argument provenance should become explicit objects "
                 "of post-training.",
                 "草稿据此认为，执行进度与参数来源应当成为后训练中的显式对象。"),
            P("Learning a correct call order is different from learning to manage execution state across several "
              "dependent tools.",
              "学会正确的调用顺序，并不等于学会在多个相互依赖的工具之间管理执行状态。"),
        ],
        source=SRC_V2_MOT,
    ),
    slide(
        "funcbenchgen", "mot", "Evidence from FuncBenchGen", "FuncBenchGen 中的证据",
        status=["est"],
        blocks=[
            P("FuncBenchGen builds tasks on a hidden function-dependency DAG, and the model must call functions until "
              "it can report a target value.",
              "FuncBenchGen 在隐藏的函数依赖 DAG 上构造任务，模型需要不断调用函数，直到能够给出目标变量的值。"),
            TABLE(
                [T("Failure type", "错误类型"), T("Definition in the paper", "论文中的定义"),
                 T("Share of errors across @@fbg_models@@ models", "在 @@fbg_models@@ 个模型中的错误占比")],
                [
                    [T("Value Not Yet Known", "Value Not Yet Known（值尚未获得）"),
                     T("Uses a variable whose value no prior call or initial input has established",
                       "使用了尚未由先前调用或初始输入确定的变量值"),
                     T("@@fbg_vnyk_lo@@ to @@fbg_vnyk_hi@@", "@@fbg_vnyk_lo@@ 至 @@fbg_vnyk_hi@@")],
                    [T("Incorrect Value", "Incorrect Value（值错误）"),
                     T("Uses an established variable but with a wrong value", "使用了已确定的变量，但取值错误"),
                     T("@@fbg_iv_lo@@ to @@fbg_iv_hi@@", "@@fbg_iv_lo@@ 至 @@fbg_iv_hi@@")],
                    [T("Function Not Found; Wrong Number of Inputs", "Function Not Found；Wrong Number of Inputs"),
                     T("Schema-level errors", "接口层面的错误"),
                     T("at most @@fbg_fnf_hi@@ and @@fbg_wni_hi@@", "分别不超过 @@fbg_fnf_hi@@ 和 @@fbg_wni_hi@@")],
                ], num_cols=[2], hl_rows=[0, 1]),
            P("Most models write syntactically valid calls, yet they lose track of which values exist and which value "
              "belongs to which argument.",
              "多数模型能写出语法正确的调用，却难以跟踪哪些值已经存在，以及哪个值应当填入哪个参数。"),
            P("Restating all known variable values after every call raised the success rate of GPT-5 from "
              "@@fbg_gpt5_before@@ to @@fbg_gpt5_after@@, according to the paper.",
              "论文报告，在每次调用后重述所有已知变量值，可使 GPT-5 的成功率从 @@fbg_gpt5_before@@ 提升到 "
              "@@fbg_gpt5_after@@。"),
        ],
        source=("Maekawa et al., ICLR 2026 (arXiv 2509.26553): Table 3, Section 4.6, and the abstract.",
                "Maekawa 等，ICLR 2026（arXiv 2509.26553）：表 3、第 4.6 节与摘要。"),
    ),
    slide(
        "failure-illustration", "mot", "Two failure types in the running example", "示例中的两类错误",
        status=["draft", "ill"],
        blocks=[
            CARDS(
                CARD("Value Not Yet Known", "Value Not Yet Known（值尚未获得）",
                     KV((T("Completed", "已完成"), "A → flight_price = 1200"),
                        (T("Missing", "缺失"), "B → hotel_price")),
                     CALLLINE("calculate_total(flight_price=1200, hotel_price=?)"),
                     P("The agent called A but not B and already tries C, so it never checked the preconditions "
                       "of C.",
                       "智能体调用了 A 却还没有调用 B，就已尝试执行 C，说明它没有检查 C 的前置条件。"),
                     cls="bad"),
                CARD("Incorrect Value", "Incorrect Value（值错误）",
                     KV((T("Completed", "已完成"), "A → 1200,  B → 600"),
                        (T("Passed", "传入"), "hotel_price = 900")),
                     CALLLINE("calculate_total(flight_price=1200, hotel_price=900)"),
                     P("The correct value 600 is already available, but the agent passes 900, possibly a value "
                       "returned by another tool.",
                       "正确的值 600 已经可用，智能体却传入了 900，这个值可能来自环境中另一个工具的返回结果。"),
                     cls="bad"),
            ),
            P("In both cases the link between a variable, the call that produced it, and its actual value has been "
              "lost.",
              "两种情况下，变量、产生它的调用与其实际取值之间的对应关系都已经丢失。"),
            P("Both errors concern execution state: which call is executable now, and which real arguments it "
              "should receive.",
              "两类错误都与执行状态有关：现在可以执行哪个调用，以及它应当接收哪些真实参数。"),
        ],
        source=("v2 draft, motivation section; the prices are the draft's illustration, not FuncBenchGen data.",
                "v2 草稿，动机部分；其中价格是草稿给出的示意数值，并非 FuncBenchGen 数据。"),
    ),
    slide(
        "hypothesis", "mot", "Central research hypothesis", "核心研究假设",
        status=["hyp"],
        blocks=[
            CALL("hyp",
                 "An agent can learn the whole set of semantically valid tool-call orders and complete the remaining "
                 "work from any valid intermediate state.",
                 "智能体可以学会全部语义合法的工具调用顺序，并能从任意合法的中间状态完成剩余工作。"),
            P("The earlier draft states a testable version for tasks whose ordering freedom is large and can be "
              "identified reliably.",
              "较早的草稿针对顺序自由度足够大、且可以被可靠识别的任务，给出了可检验的表述。"),
            CALL("hyp",
                 "Under these conditions, training on the set of valid executions gives a better capability–compute "
                 "trade-off than fixed-order supervision, order augmentation, or frontier supervision.",
                 "在这些条件下，基于合法执行集合的训练，比固定顺序监督、顺序增强或前沿集合监督具有更好的能力与计算成本权衡。"),
            P("Neither draft reports training results, and the only numerical evidence is a self-check that the "
              "probability recursion and gradients are implemented consistently.",
              "两版草稿都没有报告训练结果；仅有的数值证据是一项自检，用来确认概率递推与梯度的实现彼此一致。"),
        ],
        source=("v2 draft, central research hypothesis; earlier draft, abstract and Sections 13.2 and 14.1.",
                "v2 草稿，核心研究假设；较早草稿，摘要及第 13.2、14.1 节。"),
    ),
    slide(
        "augmentation-limits", "mot", "Limits of order augmentation", "顺序增强的局限",
        status=["draft"],
        blocks=[
            P("The direct fix samples more valid reorderings from the dependency graph and trains on each one as a "
              "separate sequence.",
              "直接的做法是从依赖图中采样更多合法重排，并把每一种重排都作为单独的序列来训练。"),
            COLS(
                [UL(
                    ("Sampling distribution.", "采样分布：",
                     "Augmentation must choose a distribution over orders, and the model is then also trained to "
                     "match that distribution.",
                     "增强必须指定一个顺序分布，于是模型还要被训练去拟合这个分布。"),
                    ("Repeated scoring.", "重复计分：",
                     "Trajectories that reach the same state after independent actions are still scored as separate "
                     "sequences from start to end.",
                     "经过一组独立动作后到达同一状态的轨迹，仍被当作不同序列从头到尾分别计分。"),
                    ("No resumption guarantee.", "无法保证接续：",
                     r"A model that reliably completes $A\to B\to C$ may still fail when another agent has already "
                     r"completed $B$.",
                     r"一个能稳定完成 $A\to B\to C$ 的模型，在 $B$ 已由另一个智能体完成时仍可能失败。"),
                )],
                [SVG("per_path_vs_shared")],
                ratio="1.15fr 1fr", cls="mid",
            ),
        ],
        source=SRC_V2_MOT,
    ),
    slide(
        "three-changes", "mot", "Three changes to the training setup", "训练设定的三处改变",
        status=["draft"],
        blocks=[
            TABLE(
                [T("", ""), T("Standard SFT", "标准 SFT"), T("Proposed method (@@METHOD@@)", "所提方法（@@METHOD@@）")],
                [
                    [T("Supervision target", "监督目标"), T(r"One demonstrated trajectory $\tau$", r"一条示范轨迹 $\tau$"),
                     T(r"Set of accepted executions $\mathcal{C}(G)$", r"接受执行集合 $\mathcal{C}(G)$")],
                    [T("Policy input", "策略输入"), T(r"Full chronological history $h_t$", r"完整的时间顺序历史 $h_t$"),
                     T(r"Canonical execution state $H_t=\Reduce(h_t)$", r"规范化执行状态 $H_t=\Reduce(h_t)$")],
                    [T("Probability computation", "概率计算"), T("Each path scored on its own", "每条路径单独计分"),
                     T("Dynamic programming over shared completed sets", "在共享的已完成集合上做动态规划")],
                    [T("Training start states", "训练起点"), T("Task start only", "仅任务起点"),
                     T("Task start and valid intermediate states", "任务起点与合法中间状态")],
                ]),
            P("The model learns a space of valid solutions instead of imitating one reference trajectory, and it "
              "learns to continue from partial progress.",
              "模型学习的是合法解构成的空间，而不是模仿某一条参考轨迹，同时还学习从部分进度继续执行。"),
        ],
        source=("v2 draft, solution items 1 to 3; the fourth row comes from the method section on valid prefixes.",
                "v2 草稿，解决方案第 1 至 3 点；第四行来自方法部分关于合法前缀的内容。"),
    ),
]

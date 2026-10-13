from deck.blocks import CALL, COLS, MATH, P, SVG, T, TABLE, TERMS, UL, slide

SLIDES = [
    slide(
        "target-graph", "graph", "Target execution graph", "目标执行图",
        status=["draft"],
        blocks=[
            MATH(r"G=(V,E,\mathcal{A},\mathcal{O},\mathcal{K})", cls="key center lg"),
            COLS(
                [TABLE(
                    [T("Part", "组成"), T("Meaning", "含义"), T("Flight and hotel example", "机票与酒店示例")],
                    [
                        ["$V$", T("Operation instances the task must complete", "任务需要完成的操作实例"),
                         r"$\{A,B,C\}$"],
                        ["$E$", T("Dependencies that must hold", "必须满足的依赖关系"), r"$\{(A,C),(B,C)\}$"],
                        [r"$\mathcal{A}$", T("Tool call and argument binding of each node", "每个节点的工具调用与参数绑定"),
                         r"$a_C=\texttt{calculate\_total}(p_f,p_h)$"],
                        [r"$\mathcal{O}$", T("Fields that each call returns", "每次调用返回的字段"),
                         r"$o_A:\{\texttt{flight\_price}\}$"],
                        [r"$\mathcal{K}$", T("Constraints on legality, data flow, and result checks",
                                             "执行合法性、数据流与结果验证的约束"),
                         r"$\CanExecute(C)\Leftrightarrow\Valid(o_A)\wedge\Valid(o_B)$"],
                    ], cls="compact")],
                [SVG("dag_flight")],
                ratio="1.2fr 1fr", cls="mid",
            ),
            P("Nodes are operation instances rather than tool types, so calling the same tool twice creates two "
              "distinct nodes.",
              "节点是操作实例而不是工具类型，因此同一工具被调用两次就对应两个不同的节点。"),
        ],
        source=("v2 draft, Section 3. The earlier draft describes the last component as contracts or evidence for edges "
                "and swaps.",
                "v2 草稿第 3 节。较早草稿把最后一项描述为支持依赖边与交换的契约或证据。"),
    ),
    slide(
        "non-interference", "graph", "Operation record and non-interference", "操作记录与非干扰条件",
        status=["draft", "est"],
        blocks=[
            COLS(
                [
                    MATH(r"k_v=(a_v,R_v,W_v,P_v,O_v)", cls="key"),
                    TERMS([
                        ("a_v", T("Executed action, such as <code>get_flight_price()</code>",
                                  "实际执行的动作，例如 <code>get_flight_price()</code>")),
                        ("R_v", T("Read set: database fields, files, and other state it reads",
                                  "读取集合：读取的数据库字段、文件等状态")),
                        ("W_v", T("Write set: state it modifies", "写入集合：修改的状态")),
                        ("P_v", T("Argument sources and preconditions", "参数来源与前置条件")),
                        ("O_v", T("Output returned by the tool", "工具返回的输出")),
                    ], cls="compact"),
                ],
                [
                    MATH(r"W_u\cap(R_v\cup W_v)=\varnothing,\qquad W_v\cap(R_u\cup W_u)=\varnothing", cls="key"),
                    P("When both hold, neither operation writes anything the other reads or writes, so swapping them "
                      "changes neither result.",
                      "两个条件同时成立时，任一操作写入的内容都不会被另一操作读取或写入，因此交换二者不会改变任何结果。"),
                    P(r"Two pure queries have $W_u=W_v=\varnothing$ and always satisfy both conditions, so the flight "
                      r"and hotel lookups commute.",
                      r"两个纯查询满足 $W_u=W_v=\varnothing$，必然满足这两个条件，因此机票查询与酒店查询可以交换顺序。"),
                    CALL("corr",
                         "These are the Bernstein conditions (1966), which are sufficient for safe reordering but not "
                         "necessary, and they assume complete read and write sets.",
                         "这正是 Bernstein 条件（1966），它们是安全交换的充分条件而非必要条件，并且要求读写集合完整。"),
                ],
                ratio="1fr 1.05fr",
            ),
        ],
        source=("v2 draft, Section 3, states the condition as necessary; the earlier draft, Section 4.4, correctly calls "
                "it sufficient.",
                "v2 草稿第 3 节把该条件表述为必要条件；较早草稿第 4.4 节正确地称其为充分条件。"),
    ),
    slide(
        "edge-sources", "graph", "Sources of dependency edges", "依赖边的来源",
        status=["draft1", "ill"],
        blocks=[
            TABLE(
                [T("Operation pair", "操作对"), T("Check", "检查"), T("Result", "结论")],
                [
                    ["<code>get_flight_price</code>, <code>get_hotel_price</code>", r"$W_u=W_v=\varnothing$",
                     T("Commute, no edge", "可交换，无边")],
                    ["<code>update_budget(+100)</code>, <code>get_budget</code>",
                     r"$W_u\cap R_v=\{\texttt{budget}\}$",
                     T("Write–read conflict, keep the order", "写读冲突，保留顺序")],
                    ["<code>lookup_address</code>, <code>send_message</code>",
                     T("No write conflict, but sending needs the address as an argument",
                       "无写冲突，但发送消息需要地址作为参数"),
                     T("Data dependency, keep the order", "数据依赖，保留顺序")],
                ], hl_rows=[2]),
            UL(
                ("Non-interference alone does not decide independence, because a later call may need an earlier "
                 "result as its argument.",
                 "仅凭非干扰条件不能判定独立，因为后一个调用可能需要前一个调用的结果作为参数。"),
                (r"The earlier draft therefore builds $E$ from data, guard, effect, and task edges, and keeps the "
                 r"demonstrated order for any unverified pair.",
                 r"因此较早的草稿由数据、控制条件、状态作用和任务要求四类边构造 $E$，并对任何未经验证的动作对保留示范顺序。"),
                (r"All relevant pairs must be checked: if $a$ writes $x$, $b$ is unrelated, and $c$ reads $x$, the "
                 r"edge $a\prec c$ must stay.",
                 r"必须检查所有相关动作对：若 $a$ 写入 $x$、$b$ 与之无关、$c$ 读取 $x$，则边 $a\prec c$ 必须保留。"),
            ),
        ],
        source=("Earlier draft, Sections 4.3 to 4.5; the budget pair is an illustration added for this deck.",
                "较早草稿第 4.3 至 4.5 节；预算操作对是本报告补充的示意例子。"),
    ),
    slide(
        "accepted-set", "graph", "Linear extensions and accepted executions", "线性扩展与接受执行集合",
        status=["draft"],
        blocks=[
            MATH(r"\LinExt(G)=\{(A,B,C),\ (B,A,C)\}"),
            MATH(r"\mathcal{C}(G)=\big\{(a_A,a_B,a_C,\aend),\ \ (a_B,a_A,a_C,\aend)\big\}", cls="key"),
            UL(
                (r"A linear extension of $G$ orders all nodes so that each edge starts at an earlier node; $\LinExt(G)$ "
                 r"collects all of them.",
                 r"$G$ 的线性扩展是全部节点的一种排列，其中每条边的起点都排在终点之前；$\LinExt(G)$ 是所有这类排列的集合。"),
                (r"The accepted execution set $\mathcal{C}(G)$ writes each linear extension as full tool calls and "
                 r"appends the terminal action.",
                 r"接受执行集合 $\mathcal{C}(G)$ 把每个线性扩展写成完整的工具调用序列，并在末尾加上终止动作。"),
                (r"Any generated sequence inside $\mathcal{C}(G)$ counts as a correct execution, even when it differs "
                 r"from the demonstration.",
                 r"模型生成的序列只要属于 $\mathcal{C}(G)$，就算作正确执行，即使它与示范轨迹不同。"),
                (r"Uncertain pairs keep their demonstrated order, so $\mathcal{C}(G)$ can be smaller than the set of "
                 r"all truly equivalent executions.",
                 r"不确定的动作对会保留示范顺序，因此 $\mathcal{C}(G)$ 可能小于全部真正等价的执行集合。"),
            ),
        ],
        source=("v2 draft, Section 3, which calls this set the execution acceptance class; last point from the earlier "
                "draft, Section 4.6.",
                "v2 草稿第 3 节，草稿称该集合为执行接受类；最后一点来自较早草稿第 4.6 节。"),
    ),
    slide(
        "set-target", "graph", "From one target to a set of targets", "从单一目标到目标集合",
        status=["draft"],
        blocks=[
            COLS(
                [MATH(r"\LSFT=-\log P_\theta(A,B,C\mid x)", cls="key")],
                [MATH(r"\mathcal{L}=-\log\sum_{\tau\in\mathcal{C}(G)}P_\theta(\tau\mid x)", cls="key")],
                ratio="1fr 1fr", cls="mid",
            ),
            UL(
                (r"Standard SFT raises the probability of the single demonstrated trajectory $(A,B,C)$ and gives "
                 r"$(B,A,C)$ no positive signal.",
                 r"标准 SFT 只提高示范轨迹 $(A,B,C)$ 的概率，对 $(B,A,C)$ 不给任何正向信号。"),
                ("The set objective gives a positive signal whenever the model places probability on any accepted "
                 "execution.",
                 "集合目标只要模型把概率放在任一接受执行上，就会给出正向训练信号。"),
                ("The model is no longer forced to follow the path that the demonstrator happened to fix in advance.",
                 "模型不再被强制沿着示范者恰好事先确定的那条路径执行。"),
            ),
            P("The objective moves from imitating one reference answer to learning a space of valid solutions.",
              "训练目标从模仿某一个标准答案，转变为学习一个合法解空间。"),
        ],
        source=("v2 draft, solution item 1. The draft labels the second trajectory with a repeated index; it is "
                "the second one.",
                "v2 草稿，解决方案第 1 点。草稿中第二条轨迹的下标重复，此处应为第二条。"),
    ),
]

from deck.blocks import MATH, P, T, TABLE, slide

SLIDES = [
    slide(
        "setup", "setup", "Task input, environment, and demonstration", "任务输入、环境与示范轨迹",
        status=["draft", "draft1"],
        blocks=[
            MATH(r"x=(u,\mathcal{T},o_0)\qquad (s',o)=\Exec_\zeta(s,a)\qquad "
                 r"\tau=(a_1,o_1,\ldots,a_n,o_n,\aend)", cls="key center"),
            TABLE(
                [T("Symbol", "符号"), T("Meaning", "含义"), T("Running example", "示例")],
                [
                    ["$u$", T("Natural-language request", "自然语言请求"),
                     T("Query flight and hotel prices from NYC to SG, then compute the total cost",
                       "查询 NYC 到 SG 的机票与酒店价格，然后计算总费用")],
                    [r"$\mathcal{T}$", T("Tool descriptions", "工具说明"),
                     "<code>get_flight_price</code>, <code>get_hotel_price</code>, <code>calculate_total</code>"],
                    ["$o_0$", T("Information the agent may use at the start", "任务开始时智能体可直接获得的信息"),
                     T("Not specified in the draft", "草稿中未给出")],
                    [r"$\zeta$", T("Fixed environment snapshot: what is stored and how it is restored",
                                   "固定环境快照：环境保存了什么以及如何恢复"),
                     T("Stored price data, restorable for every run (illustrative)", "可在每次运行时恢复的价格数据（示意）")],
                    ["$s,\\ s'$", T("Environment state before and after action $a$, held by the executor",
                                    "动作 $a$ 前后的环境真实状态，由执行器保存"), "–"],
                    ["$o$", T("Observation returned to the agent by this call", "本次调用返回给智能体的观察"),
                     "<code>flight_price = 1200</code>"],
                    [r"$\aend$", T("Terminal action after successful completion", "任务成功完成后的终止动作"),
                     T("Return the total cost", "返回总费用")],
                ], cls="compact"),
            P(r"The snapshot $\zeta$ belongs to the training executor and is never passed to the policy as "
              r"privileged input.",
              r"快照 $\zeta$ 由训练执行器持有，不会作为额外的特权信息提供给策略。"),
        ],
        source=("v2 draft, Section 2; the privileged-input rule is from the earlier draft, Section 3.1.",
                "v2 草稿第 2 节；关于特权信息的规定来自较早草稿第 3.1 节。"),
    ),
]

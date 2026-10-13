from deck.blocks import CALL, COLS, MATH, P, T, TABLE, UL, slide

SLIDES = [
    slide(
        "marginal", "obj", "Marginal likelihood over accepted executions", "接受执行集合上的边际似然",
        status=["draft", "draft1"],
        blocks=[
            MATH(r"\Lroot(\theta)=\mathbb{E}_{(x,G)\sim\mathcal{D}}\big[-\log B_\theta(\varnothing)\big]",
                 cls="key center lg"),
            MATH(r"B_\theta(\varnothing)=P_\theta\big(\mathcal{C}(G)\big)=\sum_{\sigma\in\LinExt(G)}p_\theta(\sigma)",
                 cls="center"),
            UL(
                ("The root loss is the negative log probability that the agent completes the task through any "
                 "accepted execution.",
                 "根状态损失是智能体通过任一接受执行完成任务的概率的负对数。"),
                (r"In the worked example, the root loss for this single task equals $-\ln @@b0@@=@@L_root_ex@@$ "
                 r"nats.",
                 r"在逐步计算示例中，这一个任务的根状态损失为 $-\ln @@b0@@=@@L_root_ex@@$ nat。"),
                (r"If every accepted execution passes the task check, $B_\theta(I)$ is a lower bound on the task "
                 r"success probability from $I$.",
                 r"若每条接受执行都能通过任务检查，则 $B_\theta(I)$ 是从 $I$ 出发任务成功概率的下界。"),
                ("The objective therefore targets a certified success event, not every strategy that might also "
                 "solve the task.",
                 "因此，该目标优化的是一个经过认证的成功事件，而不是所有可能完成任务的策略。"),
            ),
        ],
        source=("v2 draft, class-marginal objective; the lower-bound remark is from the earlier draft, Section 7.4.",
                "v2 草稿，类边际目标；下界的说明来自较早草稿第 7.4 节。"),
    ),
    slide(
        "three-objectives", "obj", "Three objectives on two example models", "两个示例模型上的三种目标",
        status=["draft", "derived"],
        blocks=[
            COLS(
                [
                    MATH(r"\LSFT=-\log p_\theta(ABC)", cls="sm"),
                    MATH(r"\Laug=-\tfrac12\big[\log p_\theta(ABC)+\log p_\theta(BAC)\big]", cls="sm"),
                    MATH(r"\Lroot=-\log\big[p_\theta(ABC)+p_\theta(BAC)\big]", cls="sm key"),
                    P("Both models reach the completed state with total probability 0.50, so the marginal objective "
                      "treats them as equivalent.",
                      "两个模型到达完成状态的总概率都是 0.50，因此边际目标认为二者等价。"),
                ],
                [TABLE(
                    [T("", ""), T("Model 1", "模型 1"), T("Model 2", "模型 2")],
                    [
                        [r"$p_\theta(ABC)$", "0.45", "0.25"],
                        [r"$p_\theta(BAC)$", "0.05", "0.25"],
                        [T("Total over valid paths", "合法路径总概率"), "0.50", "0.50"],
                        [r"$\Lroot$", "@@m1_root@@", "@@m2_root@@"],
                        [r"$\Laug$", "@@m1_aug@@", "@@m2_aug@@"],
                        [r"$\LSFT$", "@@m1_sft@@", "@@m2_sft@@"],
                    ], num_cols=[1, 2], hl_rows=[3])],
                ratio="1.1fr 1fr",
            ),
            UL(
                ("Order augmentation prefers Model 2 because it also rewards an even split between the two valid "
                 "orders.",
                 "顺序增强更偏好模型 2，因为它还会奖励两种合法顺序之间的均匀分配。"),
                (r"SFT on $(A,B,C)$ prefers Model 1, while the marginal objective has no preference between orders "
                 r"that are equally valid.",
                 r"在 $(A,B,C)$ 上做 SFT 则偏好模型 1，而边际目标对同样合法的顺序没有偏好。"),
            ),
        ],
        source=("Probabilities and the first two loss rows from v2; the SFT row is derived for comparison. Losses in "
                "nats.",
                "概率及前两行损失来自 v2；SFT 一行为对比而推导。损失单位为 nat。"),
    ),
    slide(
        "decomposition", "obj", "Decomposition of the augmentation loss", "顺序增强损失的分解",
        status=["draft1", "est"],
        blocks=[
            MATH(r"\mathbb{E}_{\sigma\sim\omega}\big[-\log p_\theta(\sigma)\big]=-\log P_\theta\big(\mathcal{C}(G)\big)"
                 r"+\mathcal{H}(\omega)+\KL\big(\omega\,\big\|\,p_\theta(\cdot\mid\mathcal{C}(G))\big)",
                 cls="key center"),
            COLS(
                [UL(
                    (r"Training on orders drawn from $\omega$ equals the marginal loss, plus the entropy of $\omega$, "
                     r"plus a term that fits $\omega$.",
                     r"在按 $\omega$ 抽取的顺序上训练，等于边际损失，加上 $\omega$ 的熵，再加上一个拟合 $\omega$ 的项。"),
                    (r"With a uniform $\omega$ over $M$ orders, $\mathcal{H}(\omega)=\log M$ is constant, so only the "
                     r"KL term separates the two models.",
                     r"当 $\omega$ 在 $M$ 个顺序上均匀分布时，$\mathcal{H}(\omega)=\log M$ 为常数，因此只有 KL 项能区分两个模型。"),
                    (r"Averaging probabilities before the logarithm only adds $\log M$ to the marginal loss, so the "
                     r"gradient stays the same.",
                     r"先对概率取平均再取对数，只会给边际损失加上常数 $\log M$，因此梯度不变。"),
                ),
                    MATH(r"-\log\Big(\tfrac{1}{M}\sum_\sigma p_\theta(\sigma)\Big)=-\log\sum_\sigma p_\theta(\sigma)"
                         r"+\log M", cls="sm")],
                [TABLE(
                    [T("", ""), r"$\Laug$", r"$\Lroot$", r"$\mathcal{H}(\omega)$", r"$D_{\mathrm{KL}}$"],
                    [
                        [T("Model 1", "模型 1"), "@@m1_aug@@", "@@m1_root@@", "@@ln2@@", "@@m1_kl@@"],
                        [T("Model 2", "模型 2"), "@@m2_aug@@", "@@m2_root@@", "@@ln2@@", "@@m2_kl@@"],
                    ], num_cols=[1, 2, 3, 4]),
                 P(r"Each row satisfies $\Laug=\Lroot+\mathcal{H}(\omega)+D_{\mathrm{KL}}$ with uniform $\omega$.",
                   r"每一行都满足 $\Laug=\Lroot+\mathcal{H}(\omega)+D_{\mathrm{KL}}$，其中 $\omega$ 为均匀分布。",
                   cls="small muted", label=True),
                 CALL("draft1",
                      "The extra KL term is not necessarily harmful and may act as a regularizer, so the comparison "
                      "is an empirical question.",
                      "额外的 KL 项未必有害，它也可能起到正则化作用，因此孰优孰劣需要由实验回答。")],
                ratio="1.05fr 1fr",
            ),
        ],
        source=("Earlier draft, Section 8.2, Eq. (21); v2 states the log-M identity without the decomposition.",
                "较早草稿第 8.2 节式 (21)；v2 只给出了 log M 恒等式，没有给出分解。"),
    ),
    slide(
        "posterior-gradient", "obj", "Posterior-weighted gradient", "后验加权梯度",
        status=["draft1", "derived"],
        blocks=[
            MATH(r"\nabla_\theta\log\sum_{\sigma\in\mathcal{C}}p_\theta(\sigma)=\mathbb{E}_{\sigma\sim p_\theta(\cdot\mid"
                 r"\mathcal{C})}\big[\nabla_\theta\log p_\theta(\sigma)\big]", cls="center"),
            MATH(r"q_\theta(v\mid I,G)=\frac{w_\theta(I,v)\,B_\theta(I\cup\{v\})}{B_\theta(I)}", cls="key center lg"),
            UL(
                ("The marginal gradient is an ordinary log-likelihood gradient, with each accepted path weighted by "
                 "its posterior probability given success.",
                 "边际似然的梯度就是普通的对数似然梯度，只是每条接受路径按其在成功条件下的后验概率加权。"),
                (r"In the worked example, the posteriors for the first action are "
                 r"$q(A\mid\varnothing)=@@wA@@\times@@b_a@@/@@b0@@=@@q_a@@$ and $q(B\mid\varnothing)=@@q_b@@$.",
                 r"在逐步计算示例中，第一个动作的后验为 $q(A\mid\varnothing)=@@wA@@\times@@b_a@@/@@b0@@=@@q_a@@$ 与 "
                 r"$q(B\mid\varnothing)=@@q_b@@$。"),
                (r"They differ from the policy's own share $@@wA@@/@@wsum0@@=@@renorm_a@@$, because completion after "
                 r"$B$ is more likely.",
                 r"它们不同于策略自身的比例 $@@wA@@/@@wsum0@@=@@renorm_a@@$，因为先做 $B$ 之后完成的可能性更高。"),
            ),
            CALL("syn",
                 r"Augmentation weights every order by the fixed $\omega$, while the marginal gradient follows the "
                 r"success-conditioned preference of the model itself (my synthesis).",
                 r"顺序增强按固定的 $\omega$ 给每种顺序加权，而边际梯度跟随模型自身在成功条件下的偏好（个人归纳）。"),
        ],
        source=(r"$q_\theta$ is defined in the earlier draft, Section 9.1; v2 uses it in Algorithm 1 without a "
                r"definition.",
                r"$q_\theta$ 的定义见较早草稿第 9.1 节；v2 在算法 1 中使用了它，但没有给出定义。"),
    ),
    slide(
        "transition-weights", "obj", "Transition weights and the two-stage update", "转移权重与两阶段更新",
        status=["draft1", "derived"],
        blocks=[
            COLS(
                [
                    MATH(r"m_\theta(I)\leftarrow\mu_G(I),\qquad m_\theta(I\cup\{v\})\mathrel{+}=m_\theta(I)\,"
                         r"q_\theta(v\mid I,G)", cls="sm"),
                    MATH(r"\rho_\theta(I,v)=m_\theta(I)\,q_\theta(v\mid I,G)", cls="sm"),
                    MATH(r"\widetilde{\mathcal{L}}_G=-\sum_{I,v}\sg\big[\rho_\theta(I,v)\big]\log w_\theta(I,v)"
                         r"-\log e_\theta(V)", cls="key"),
                    UL(
                        (r"$\rho_\theta(I,v)$ is the expected number of times the success posterior takes action $v$ "
                         r"at state $I$.",
                         r"$\rho_\theta(I,v)$ 是成功后验在状态 $I$ 下选择动作 $v$ 的期望次数。"),
                        ("The stop-gradient fixes the weights while token log probabilities are recomputed, which "
                         "gives the exact marginal gradient at the current parameters.",
                         "停止梯度先固定权重，再重新计算 token 对数概率，从而在当前参数处得到精确的边际似然梯度。"),
                        ("The value of this surrogate differs from the marginal loss, so training logs should report "
                         "the dynamic-programming value.",
                         "该代理损失的数值不同于边际损失，因此训练日志应当记录由动态规划计算的损失值。"),
                    ),
                ],
                [
                    TABLE(
                        [T("Transition", "转移"), r"$\rho_\theta$"],
                        [
                            [r"$(\varnothing,A)$", "@@rho_0A@@"],
                            [r"$(\varnothing,B)$", "@@rho_0B@@"],
                            [r"$(\{A\},B)$", "@@rho_AB@@"],
                            [r"$(\{B\},A)$", "@@rho_BA@@"],
                            [r"$(\{A,B\},C)$", "@@rho_C@@"],
                            [T("Sum", "合计"), "@@rho_sum@@"],
                        ], num_cols=[1], cls="compact"),
                    P(r"Start weights are $\mu(\varnothing)=@@mu0@@$ and $\mu(\{A\})=\mu(\{B\})=@@muA@@$, as in the "
                      r"earlier draft, and the sum equals $\mathbb{E}[n-|S|]$.",
                      r"起点权重取 $\mu(\varnothing)=@@mu0@@$、$\mu(\{A\})=\mu(\{B\})=@@muA@@$，与较早草稿一致；权重之和等于 "
                      r"$\mathbb{E}[n-|S|]$。", cls="small"),
                ],
                ratio="1.3fr 0.9fr",
            ),
        ],
        source=("Earlier draft, Sections 9.2 to 9.5 and 13.1; v2 Algorithm 1 computes these quantities without "
                "defining them.",
                "较早草稿第 9.2 至 9.5 节及第 13.1 节；v2 的算法 1 计算了这些量，但没有给出定义。"),
    ),
    slide(
        "root-only-limit", "obj", "Limitation of root-only training", "仅从根状态训练的局限",
        status=["draft", "ill"],
        blocks=[
            P("A policy that masters one order and rarely completes the task from other valid states can still reach "
              "a low root loss.",
              "一个只掌握一种顺序、很少能从其他合法状态完成任务的策略，仍然可以得到很低的根状态损失。"),
            TABLE(
                [T("", ""), T("Worked-example policy", "示例策略"), T("Order-collapsed policy", "顺序坍缩策略")],
                [
                    [r"$w(\varnothing,A),\ w(\varnothing,B)$", "@@wA@@, @@wB@@", "@@c_wA@@, @@c_wB@@"],
                    [r"$w(\{A\},B),\ w(\{B\},A)$", "@@wAB@@, @@wBA@@", "@@c_wAB@@, @@c_wBA@@"],
                    [T(r"Root loss $-\log B(\varnothing)$", r"根状态损失 $-\log B(\varnothing)$"), "@@e_L0@@", "@@c_L0@@"],
                    [T(r"Resume after $B$: $-\log B(\{B\})$", r"在 $B$ 之后接续：$-\log B(\{B\})$"), "@@e_LB@@", "@@c_LB@@"],
                ], num_cols=[1, 2], hl_rows=[2, 3]),
            UL(
                (r"The order-collapsed policy has the lower root loss, yet from $\{B\}$ it completes the task with "
                 r"probability only @@c_bB@@.",
                 r"顺序坍缩策略的根状态损失更低，但从 $\{B\}$ 出发时，它完成任务的概率只有 @@c_bB@@。"),
                (r"A root-only objective cannot see this failure, because no training example starts after another "
                 r"agent has completed $B$.",
                 r"仅从根状态训练的目标看不到这一缺陷，因为没有任何训练样本从另一个智能体已完成 $B$ 的状态开始。"),
            ),
        ],
        source=(r"Argument from v2 and the earlier draft, Section 8.3. The collapsed policy uses illustrative numbers "
                r"(my synthesis); both share $w(\{A,B\},C)=@@wC@@$ and $e(V)=@@eV@@$.",
                r"论证来自 v2 与较早草稿第 8.3 节。顺序坍缩策略使用示意数值（个人归纳）；两者均取 $w(\{A,B\},C)=@@wC@@$、"
                r"$e(V)=@@eV@@$。"),
    ),
    slide(
        "prefix-training", "obj", "Training from intermediate states", "从中间状态开始训练",
        status=["draft"],
        blocks=[
            COLS(
                [MATH(r"\mu_G=(1-\lambda)\,\delta_{\varnothing}+\lambda\,\mu_{\mathrm{cut}}", cls="sm")],
                [MATH(r"\widehat{\mu}_G=(1-\lambda)\,\delta_{\varnothing}+\frac{\lambda}{K}\sum_{k=1}^{K}"
                      r"\delta_{I_k}", cls="sm")],
                ratio="1fr 1fr", cls="mid",
            ),
            MATH(r"\LQET(\theta)=\mathbb{E}_{(x,G)\sim\mathcal{D}}\,\mathbb{E}_{I\sim\mu_G}\big[-\log B_\theta(I)\big]",
                 cls="key center lg"),
            COLS(
                [UL(
                    (r"Each update samples a task, then a valid start state $I$ from $\mu_G$, and trains completion "
                     r"from that state.",
                     r"每次更新先采样一个任务，再从 $\mu_G$ 中采样一个合法起始状态 $I$，并训练从该状态完成任务。"),
                    (r"$\lambda$ sets the weight of non-root starts, and $K$ sampled cut states replace enumeration of "
                     r"all valid completed sets.",
                     r"$\lambda$ 决定非根起点的权重，并用 $K$ 个采样得到的截断状态代替对全部合法已完成集合的枚举。"),
                    (r"$H_I$ is built only from the actual observations of actions in $I$, never from later steps of "
                     r"the demonstration.",
                     r"$H_I$ 只由 $I$ 中动作的实际观察构造，不使用示范中后续步骤的信息。"),
                )],
                [TABLE(
                    [T("Start $I$", "起点 $I$"), T("Done", "已完成"), T("Remaining work", "剩余工作")],
                    [
                        [r"$\varnothing$", T("none", "无"),
                         T(r"$A\to B\to C$ or $B\to A\to C$", r"$A\to B\to C$ 或 $B\to A\to C$")],
                        [r"$\{A\}$", "$A$", r"$B\to C$"],
                        [r"$\{B\}$", "$B$", r"$A\to C$"],
                        [r"$\{A,B\}$", "$A,B$", "$C$"],
                    ], cls="compact")],
                ratio="1.15fr 1fr",
            ),
        ],
        source=("v2 draft, valid-prefix coverage; the observation rule is from v2 Algorithm 1.",
                "v2 草稿，合法前缀覆盖；关于观察的规定来自 v2 的算法 1。"),
    ),
    slide(
        "prefix-effect", "obj", "Effect of intermediate starts", "中间起点的作用",
        status=["draft1", "ill", "derived"],
        blocks=[
            MATH(r"\mu(\varnothing)=@@mu0@@,\qquad\mu(\{A\})=\mu(\{B\})=@@muA@@", cls="center"),
            TABLE(
                [T("", ""), T("Worked-example policy", "示例策略"), T("Order-collapsed policy", "顺序坍缩策略")],
                [
                    [T(r"Root loss $-\log B(\varnothing)$", r"根状态损失 $-\log B(\varnothing)$"), "@@e_L0@@", "@@c_L0@@"],
                    [T(r"Mixture loss $\LQET$", r"混合损失 $\LQET$"), "@@mix_ex@@", "@@mix_col@@"],
                ], num_cols=[1, 2], hl_rows=[1]),
            MATH(r"\begin{aligned}&@@mu0@@\times@@e_L0@@+@@muA@@\times@@e_LA@@+@@muB@@\times@@e_LB@@=@@mix_ex@@\\"
                 r"&@@mu0@@\times@@c_L0@@+@@muA@@\times@@c_LA@@+@@muB@@\times@@c_LB@@=@@mix_col@@\end{aligned}",
                 cls="sm center"),
            UL(
                ("With a quarter of the weight on each one-action start, the worked-example policy now has the lower "
                 "loss.",
                 "在两个单动作起点上各放四分之一的权重之后，示例策略的损失反而更低。"),
                (r"If the actual resume distribution satisfies $\nu_G(I)\le\kappa\,\mu_G(I)$, the expected failure is "
                 r"bounded by $\mathbb{E}_{\nu_G}[1-B_\theta(I)]\le\kappa\,\mathbb{E}_{\mu_G}[-\log B_\theta(I)]$.",
                 r"若实际接续起点分布满足 $\nu_G(I)\le\kappa\,\mu_G(I)$，则期望失败概率满足 "
                 r"$\mathbb{E}_{\nu_G}[1-B_\theta(I)]\le\kappa\,\mathbb{E}_{\mu_G}[-\log B_\theta(I)]$。"),
                (r"The bound follows from $1-z\le-\log z$ and holds only for the covered tasks and valid resume states.",
                 r"该界由 $1-z\le-\log z$ 推出，只对所覆盖的任务与合法接续状态成立。"),
            ),
        ],
        source=("Start weights from the earlier draft, Section 13.1; coverage bound from its Proposition 3, Section "
                "12.3. Losses in nats.",
                "起点权重取自较早草稿第 13.1 节；覆盖界来自其第 12.3 节命题 3。损失单位为 nat。"),
    ),
]

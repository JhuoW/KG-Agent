from deck.blocks import COLS, HTML, P, T, TABLE, slide

SLIDES = [
    slide(
        "title", "intro", "Tool-Using LLM Agents", "工具使用型大语言模型智能体",
        layout="title",
        blocks=[
            HTML('<div class="t-rule"></div>'
                 '<h1 class="t-main"><span class="l-en">Tool-Using LLM Agents</span>'
                 '<span class="l-zh" lang="zh-CN">工具使用型大语言模型智能体</span></h1>'),
            P("Learning dependency structure and execution state instead of one fixed tool-call order",
              "学习依赖结构与执行状态，而非某一种固定的工具调用顺序", cls="t-sub", label=True),
            P("Research presentation based on the working draft @@METHOD@@ v2 (Notion, 2026-10-13) "
              "and its earlier version (2026-10-08)",
              "研究报告，依据工作稿 @@METHOD@@ v2（Notion，2026-10-13）及其较早版本（2026-10-08）",
              cls="t-meta", label=True),
            P("Use the arrow keys to move between slides and press L to switch between English and Chinese.",
              "使用方向键翻页，按 L 键在英文与中文之间切换。", cls="t-meta"),
        ],
        source=("Both drafts are design documents; they report no LLM training results.",
                "两版草稿均为方法设计稿，没有报告任何 LLM 训练结果。"),
    ),
    slide(
        "outline", "intro", "Outline and status labels", "报告结构与状态标签",
        blocks=[
            COLS(
                [TABLE(
                    [T("Part", "部分"), T("Content", "内容")],
                    [
                        [T("1 Motivation", "1 研究动机"),
                         T("Why one demonstrated order is a poor training target", "为什么单一示范顺序不适合作为训练目标")],
                        [T("2 Problem setup", "2 问题设定"),
                         T("Task input, environment snapshot, and demonstration", "任务输入、环境快照与示范轨迹")],
                        [T("3 Execution graph", "3 执行图"),
                         T("Dependencies, non-interference, and accepted executions", "依赖、非干扰条件与接受执行集合")],
                        [T("4 Execution state", "4 执行状态"),
                         T("Mapping equivalent histories to one model input", "把等价历史映射为同一模型输入")],
                        [T("5 Shared-state computation", "5 共享状态计算"),
                         T("Dynamic programming over completed sets", "在已完成集合上做动态规划")],
                        [T("6 Training objective", "6 训练目标"),
                         T("Marginal likelihood, gradients, and intermediate starts", "边际似然、梯度与中间起点")],
                        [T("7 Algorithms", "7 算法流程"),
                         T("Training and inference procedures", "训练与推理流程")],
                        [T("8 Positioning and status", "8 定位与现状"),
                         T("Related work, assumptions, evidence, and open questions", "相关工作、假设、证据与待解决问题")],
                    ], cls="compact")],
                [TABLE(
                    [T("Label", "标签"), T("Meaning", "含义")],
                    [
                        ['<span class="chip draft"><span class="l-en">From the v2 draft</span><span class="l-zh" lang="zh-CN">来自 v2 草稿</span></span>',
                         T("Content of the reviewed draft", "所评审草稿中的内容")],
                        ['<span class="chip draft1"><span class="l-en">From the earlier draft</span><span class="l-zh" lang="zh-CN">来自较早草稿</span></span>',
                         T("Definitions v2 uses but only the earlier draft states", "v2 使用、但只在较早草稿中给出的定义")],
                        ['<span class="chip est"><span class="l-en">Established result</span><span class="l-zh" lang="zh-CN">已有结论</span></span>',
                         T("Published or textbook result, checked at its source", "已发表或教科书结论，已核对原始出处")],
                        ['<span class="chip derived"><span class="l-en">Derived computation</span><span class="l-zh" lang="zh-CN">推导计算</span></span>',
                         T("Arithmetic on the drafts' definitions, computed by script", "基于草稿定义的计算，由脚本求得")],
                        ['<span class="chip ill"><span class="l-en">Illustrative numbers</span><span class="l-zh" lang="zh-CN">示意数值</span></span>',
                         T("Example values for teaching, not measurements", "用于讲解的示例数值，不是测量结果")],
                        ['<span class="chip hyp"><span class="l-en">Hypothesis</span><span class="l-zh" lang="zh-CN">假设</span></span>',
                         T("A claim the drafts propose to test", "草稿提出、有待检验的主张")],
                    ], cls="compact")],
                ratio="1.1fr 1fr",
            ),
            P("Interpretations that go beyond both drafts carry the marker (my synthesis), used at most once per slide.",
              "超出两版草稿的解读标注为（个人归纳），每页至多出现一次。"),
        ],
        source=("Method name: the drafts call the method @@METHOD@@; this deck uses the acronym only.",
                "方法名称：草稿将该方法称为 @@METHOD@@，本报告只使用该缩写。"),
    ),
]

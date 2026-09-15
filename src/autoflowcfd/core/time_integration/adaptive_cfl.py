"""
AutoFlowCFD V2.0 - 稳态求解器自适应 CFL 控制器

稳态伪时间迭代中，固定 CFL 数（如此前硬编码 0.1）会导致收敛速度过慢
（实测 20 步仅下降 4 倍）。本模块根据残差历史自动调节 CFL 数，在保持
稳定性的前提下加速收敛。

设计原则（2026-08-24）：
    1. 保证不发散：CFL 上限保守（0.3，远低于 SSP-RK3 稳定极限 ~1.0），
       残差恶化时立即缩小 CFL，爬升阶段固定 CFL 不动。
    2. 五级调节策略：
       - grow：ratio < 0.9（残差快速下降），连续 5 步确认后 ×1.1
       - crawl：0.9 ≤ ratio < 0.95（缓慢收敛），连续 5 步确认后 ×1.05
       - 死区：0.95 ≤ ratio ≤ 1.0 —— 单步比值看不出进展，改由
         **窗口趋势判据**接管（见下方第 4 条），不再是"CFL 不变"
       - shrink(轻)：1.0 < ratio ≤ 1.1（轻微恶化），立即 ×0.9
       - shrink(重)：ratio > 1.1（明显恶化），立即 ×0.8
       所有调节均有 5 步冷却期，避免 CFL 振荡。
    4. 窗口趋势放大（2026-09-14 新增，修一个真实的收敛速度缺陷）：
       上面五个区间是按**瞬态**式的快速残差下降标定的（grow 要求单步
       降 >10%）。但显式稳态迭代进入渐近段后，每步残差只降千分之几是
       **正常且健康**的——79 万单元 cube_demo P1 实测每步 ratio≈0.99916
       并持续单调下降，正好落在死区里，于是 CFL 被永久钉在 cfl_start
       （实测 30 步全程 0.100），cfl_max=0.5 从来到不了，白白丢掉约 5 倍
       步长。这直接是用户反馈"从开始计算到收敛要数万步"的一个成因。
       修法：死区里改看**累计**趋势——若最近 trend_window 步的残差累计
       下降超过 trend_threshold（默认 20 步累计降到 0.995 以下），说明
       确实在收敛，只是慢，于是放大 CFL。任何一步恶化（ratio>1.0）都会
       立即走 shrink 分支并清空趋势窗口，所以放大是单向试探、随时可退。
       为什么放大 CFL 不会反过来推高残差：残差范数 ‖R(U)‖ 只依赖解 U、
       不显含 dt；在稳定极限内加大 dt 只是沿同一下降方向走得更远，残差
       降得更快，超出稳定极限才会上升——反馈符号天然正确，控制器因此能
       自己找到稳定边界。
    5. 轻度恶化需要连续确认（2026-09-14 新增，与第 4 条同一批、修的是
       同一个"CFL 到不了上限"症状的另一半成因）：`shrink_mild` 原先对
       **单步** ratio > 1.0 就立刻 ×0.9（只受 5 步冷却约束）。但稳态显式
       迭代的残差范数并不单调——即使迭代稳定且在收敛，也约有一半的步
       会出现小幅上升。于是 CFL 在任何带噪声的算例里单向棘轮下滑到
       cfl_min 并卡住。真实实测（128 单元压力扰动算例，4000 步，已启用
       第 4 条的趋势放大）：grow 触发 114 次、shrink_mild 触发 112 次，
       净效果 1.1^114 × 0.9^112 ≈ 0.39，CFL 从 0.1 被压到下限 0.05；此前
       79 万单元真实网格上 CFL 30 步内 0.1 → 0.053 的持续节流、以及另一
       个算例里 CFL 卡在 0.05 再也回不来，都是同一机制。
       现在轻度恶化要连续 mild_shrink_confirm_steps 步才收缩（与同函数
       里 grow/crawl 一直要求的"连续 N 步确认"对齐，原先唯独这一支不
       要求，是不对称的疏漏）。**重度**恶化（ratio > shrink_threshold，
       NaN/inf 被规约成 +inf 也走这里）不受确认约束、仍然立即收缩——
       真实失稳必须第一时间响应，这一条安全性不能让。
       同批的第二处调整：趋势窗口只在 CFL **真的被调节**时清空，不再
       对每个 ratio>1 的噪声步都清空——否则噪声会让窗口永远攒不满、
       第 4 条的趋势放大形同废置；窗口判据本身用的是 20 步**累计**比值，
       窗口内的个别上升步已经计入其中，不需要另外剔除。
    6. 两个方向都由同一个窗口判据仲裁（2026-09-14 同批第三处）：只做完
       第 4、5 条之后实测仍然不够——128 单元算例里 CFL 先爬到 0.2975
       （残差 4000 步降 150 倍，修复前同步数只有 67 倍），随后触发 18 次
       **重度** shrink（CFL 真的越界了）又塌回下限 0.05 卡住，净效果
       1.1^183 × 0.9^138 × 0.8^18 ≈ 0.32 仍是负的。根源是**方向不对称**：
       收缩最快每 cooldown_steps(5) 步一次、幅度 ×0.8/×0.9；放大最快每
       trend_window(20) 步一次、幅度 ×1.1。收缩的"事件速率 × 幅度"是放大
       的数倍，于是任何带噪声的区段都单向棘轮下滑，最终卡在 cfl_min
       （比 cfl_start 还低）再也回不来。
       改法：**轻度**恶化除了连续确认，还必须窗口内没有累计下降才收缩
       ——两个方向于是共用同一个时间尺度和同一个判据，棘轮效应从机制上
       消失。窗口未攒满时退回纯连续确认，保留调节后头 20 步的快速保护。
       **重度**恶化始终不受任何窗口/确认约束（安全底线）。
    8. 软上限：记住失败过的 CFL（2026-09-14，由 79 万单元真实网格长程
       对照的数据逼出来的，是本批最后一处）。做完第 4~7 条之后，真实
       网格上出现了一个新的、更根本的问题：控制器**不记得自己在哪个
       CFL 上失败过**，于是每隔约 trend_window 步就重新探过稳定边界一次，
       每次探过都要付一段残差过冲的代价。实测（同一个 P1 检查点、
       预处理开启、400 步）：CFL 由 0.100 经 7 次 grow_trend 爬到 0.195，
       在第 161 步起失稳、残差累计涨 9% 后于第 168 步被拉回 0.175；
       随后又爬到 0.174 再次过冲、退到 0.156……残差长期在 3.2e8 附近
       churn，而**CFL 冻结在 0.100 的对照**同期稳步降到 2.961e8。也就是
       说，只有第 4~7 条时这套机制在这个算例上是**净负**的——每次探过
       边界的损失超过大步长的收益。
       修法：每次发生收缩时把"软上限"设为 `ceiling_backoff * 收缩前的
       CFL`（默认 0.95），此后放大不得越过它；上限随每次失败几何式下降，
       因此从上方收敛到稳定边界并停住，不再周期性穿越。取"收缩前"而不是
       "收缩后"的值，是为了让上限贴着边界而不是一步退到远低于边界处。
       阶数切换（`reset()`）时软上限作废——那是换了一个离散问题，旧的
       稳定边界不再适用。
       还有一个必须堵住的漏洞：窗口在每次 CFL 调节后清空，于是放大之后
       有 trend_window 步的"盲区"。第一版让盲区退回"连续 3 步确认"，实测
       噪声在盲区内稳定凑出 3 连升，每次放大都被紧随其后的收缩抵消，
       CFL 在 0.050 <-> 0.055 之间无限往复（单元测试直接复现）。现在盲区
       内只对**持续**恶化反应（连续 trend_window//2 步）——噪声连不出
       这么多步，真实的单调上升 10 步内仍会被抓到，比这更快的失稳本来
       就走 severe 分支。
    7. 迟滞带（2026-09-14 同批第四处，也是最后一处）：第 6 条让两个方向
       共用窗口判据后，实测 CFL 不再塌到下限，但会在稳定边界上形成
       **极限环**——128 单元算例稳定后 CFL 在 0.106 <-> 0.120 之间每约
       26 步往复一次（grow_trend 与 shrink_mild 交替），因为两者用的是
       同一个阈值 0.995（累计 < 0.995 放大、>= 0.995 收缩），边界上必然
       抖动。现在收缩改用独立的 trend_shrink_threshold（默认 1.0，即窗口
       **确实变差**才收缩），与放大阈值之间留出 0.995~1.0 的迟滞带：带内
       （窗口在下降但幅度不够）CFL 保持不动。这既消掉了极限环，也让 CFL
       调节日志从"每 26 步一条"回到只在真正需要时才出现。
    9. 绝对基准：记住"自己到过的最好残差"（2026-09-15，由 79 万单元真实
       网格一次**已跑完**的发散逼出来的）。第 4~8 条全部看的都是**相对
       最近过去**的量——单步比值、20 步累计窗口——于是一条"缓慢离开收敛
       盆"的轨迹可以穿过它们**全部**判据：实测（同一 P1 检查点、模态
       滤波器关闭即真实 P1 内容、cfl_start=0.1）残差在 step 2 到达全程
       最低 2.744e8，随后冲到 5.26e9（最低点的 **19.2 倍**），再缓慢衰减
       到 1.02e9（仍是 **3.7 倍**），最后在 step 174 发散成 NaN。控制器
       在这全程里**两次放大 CFL**：step 45 在 15.8 倍处 0.0576 -> 0.0608、
       step 129 在 4.6 倍处 0.0608 -> 0.064。两次都"合规"——因为
       5.26e9 -> 1.26e9 这段衰减确实让 20 步窗口累计比值低于 0.995，是一个
       全局已失败轨迹里的真实局部下降。

       换句话说，第 4 条论证放大反馈符号天然正确所依赖的前提（"残差降得
       更快说明还在稳定极限内"）只在**解还在收敛盆里**时成立；一旦已经
       离开，局部斜率就不再携带稳定性信息。

       修法是加一个**绝对**基准 `_res_best`（自构造/reset 以来见过的最小
       有限残差），并只用它做两件事，两件都只会**阻止**动作、不会主动
       收缩：

         (a) `res / res_best > grow_block_ratio`（默认 2.0）时**禁止任何
             放大**（grow / crawl / grow_trend 三条路径）。"比自己到过的
             最好成绩还差一倍以上却要加大步长"没有任何情形下是对的。
         (b) 软上限的时间性释放（第 8 条的 `ceiling_release_steps`）额外
             要求"当前残差不差于上限设定时的残差"。释放的理由是"问题随
             收敛变容易了"；残差比当时更差时这个理由不成立，释放就只是
             去重探一个已知失败的 CFL。实测 step 129 那次放大正是释放
             造成的——它回到了 step 24 已经失败过的 0.064。

       阈值的依据（同一批日志，逐条算例实测 `res/res_best` 的上界）：

         健康算例（legacy 档、以及固定 CFL 0.03 的真实 P1 档）：全程恒为
         **1.00**——残差严格单调下降，当前值**就是**历史最好值；
         失败算例（off / sensor / 自适应真实 P1 各档）：20 步内就达到
         **17.5**，之后 20.47 起跳。

       1.00 与 17.5 之间隔着一个数量级以上，所以 2.0 这个阈值离两侧都
       很远，不是需要调参的量。**默认值不变、健康轨迹逐位不变**：健康
       轨迹上比值恒为 1.00，(a)(b) 两条永不触发。

       **本条不做的事**：没有加"绝对回退即判定重度恶化并强制收缩"。那
       在上面这条轨迹上会在 step 3~20 的启动暂态里触发（续算 P0 检查点
       进入真实 P1，解本来就还不在 P1 的盆里），而那一段是不是"发散"并
       没有数据支撑。本项目已经有三次"短窗口得出符号相反结论"的先例，
       所以只落地有真实长程数据支撑的那部分。同样要说清楚的是：(a)(b)
       两条**可证地**消掉了上述两次不该发生的放大，但由此停住的 CFL
       （0.0576）是否落在稳定边界内**尚无数据**——现有数据只确定 0.03
       稳定、0.0608 不稳定。

   11. `cfl_min` 会把收缩变成**放大**、并越过调用方给定的 `cfl_max`
       （2026-09-15 晚，真实运行日志直接暴露）。收缩分支写的是

           self.cfl_number = max(self.cfl_number * factor, self.cfl_min)

       只钳下界、完全不看 `cfl_max`。于是当调用方刻意把 `cfl_max`（连带
       `cfl_start`）设到默认 `cfl_min=0.05` **以下**时——做低 CFL 稳定边界
       扫描正需要这样——第一次收缩就会

           max(0.045 * 0.9, 0.05) = 0.05 > cfl_max = 0.045

       把 CFL **调高**，而且日志照旧标成 `shrink_mild`。真实日志原文：

           [AdaptiveCFL] Step 25: CFL 0.045 → 0.050 (shrink_mild, ratio=1.057)

       一个名为"收缩"的分支把步长变大、并突破了调用方声明的硬上限，
       这在任何情形下都不是可接受的行为。后果不只是标签不对：软上限
       （第 8 条）只约束放大路径，所以这条通道能绕过全部越界保护，把
       CFL 推到一个调用方明确禁止的值上——本轮的 CFL 0.045 探针就是这样
       从 step 25 起变成了 0.05 的运行，整段数据作废。

       与 2026-08-31 那次修复是**同一个表达式的两种病**：那次修的是
       `cfl_number` 已经等于 `cfl_min` 时结果被 clip 成原值、却照样打印
       "已调节"日志（空操作）；这次是 `cfl_number` 低于 `cfl_min` 时结果
       被 clip 成**更大**的值（反向操作）。当时只补了打印判断，没有回头
       问"为什么 clip 的结果可以不等于收缩的意图"。

       修法分三处，都是收紧而不是放松：
         (a) `__init__` 里协调三个参数——`cfl_min > cfl_max` 是自相矛盾
             的配置，`cfl_max` 是调用方更强的意图声明，所以把 `cfl_min`
             钳到 `cfl_max` 并**打 warning**（不静默）；`cfl_start` 同样
             钳进 `[cfl_min, cfl_max]`。
         (b) 收缩分支改成 `min(max(cfl*factor, cfl_min), cfl_max)`——即便
             (a) 被绕过（比如外部直接改属性），也不可能产出高于 cfl_max
             的值。
         (c) 软上限的 `ceiling_floor` 同样用 `min(..., cfl_max)` 收口，
             否则上限本身可以被顶到 cfl_max 之上。

    3. 与 dual.py 的双时间步自适应逻辑独立——两者面向不同的迭代结构
       （稳态每步一次 RK3 vs 双时间每步多次内迭代），参数和策略不同。

与 dual.py 自适应 CFL 的关键差异：
    - dual.py 有步拒绝 + 重试（内层迭代代价低），本控制器不做步拒绝
      （稳态一步代价高 ~8s，保存/恢复状态 + 重算残差不划算）
    - dual.py 对单步残差变化立即反应（内迭代中），本控制器要求连续
      多步确认后才调节（跨步反馈，天然滞后但更平滑）
    - dual.py 的 CFL 范围 [1e-6, 10.0]，本控制器 [0.05, 0.3]（保守）
"""

from __future__ import annotations

import math
import os
from collections import deque
from typing import Deque, List, Tuple

from loguru import logger


class AdaptiveCFLController:
    """稳态求解器自适应 CFL 控制器。

    根据连续两步残差范数的变化率，动态调节 CFL 数。

    调节逻辑：
        1. 爬升阶段（前 ramp_steps 步）：CFL 固定为 cfl_start，不调节。
           建立稳定的残差基线，避免初始暂态触发误调节。
        2. 爬升结束后，根据残差比 ratio = res_current / res_previous：
           - ratio < 0.9（残差快速下降 >10%）：
             连续 5 步确认后 CFL ×1.1（grow）
           - 0.9 ≤ ratio < 0.95（缓慢收敛）：
             连续 5 步确认后 CFL ×1.05（crawl）
           - 0.95 ≤ ratio ≤ 1.0（死区）：单步比值无信息，交给窗口趋势
             判据（最近 trend_window 步累计下降 < trend_threshold 则
             ×trend_factor），见模块文档第 4 条
           - 1.0 < ratio ≤ 1.1（轻微恶化）：立即 CFL ×0.9（shrink）
           - ratio > 1.1（明显恶化）：立即 CFL ×0.8（shrink）
        3. 冷却期：两次 CFL 调节之间至少间隔 cooldown_steps 步，
           避免 CFL 在相邻两步间来回振荡。

    Attributes:
        cfl_number: 当前 CFL 数（cfl.py 读取此值替代硬编码 0.1）
    """

    def __init__(
        self,
        cfl_start: float = 0.1,
        cfl_max: float = 0.5,
        cfl_min: float = 0.05,
        growth_factor: float = 1.1,
        shrink_factor: float = 0.8,
        growth_threshold: float = 0.9,
        shrink_threshold: float = 1.1,
        growth_confirm_steps: int = 5,
        cooldown_steps: int = 5,
        ramp_steps: int = 3,
        crawl_threshold: float = 0.95,
        crawl_factor: float = 1.05,
        crawl_confirm_steps: int = 5,
        mild_shrink_factor: float = 0.9,
        trend_window: int = 20,
        trend_threshold: float = 0.995,
        trend_factor: float = 1.1,
        mild_shrink_confirm_steps: int = 3,
        trend_shrink_threshold: float = 1.0,
        ceiling_backoff: float = 0.95,
        ceiling_release_steps: int = 100,
        grow_block_ratio: float = 2.0,
    ):
        """初始化自适应 CFL 控制器。

        Args:
            cfl_start: 初始 CFL 数（调用方认为已验证稳定的保守值）。
                **注意它只是一个断言、不是事实**：软上限（第 8 条）原先
                无条件把 cfl_start 当下界，结果在真实稳定 CFL 低于
                cfl_start 的工况下那道保护被完全抵消（79 万单元 cube_demo
                实测：稳定 CFL 约 0.063、cfl_start=0.1，控制器爬到 0.0697
                失败后又爬回 0.0690 并在 2 步内发散）。现在只有**收缩发生
                在 cfl_start 之上**时才用它当下界；一旦在它以下发生收缩，
                就说明这个断言对当前问题不成立，下界改用收缩后的 CFL，
                详见 `_shrink` 分支里该处的完整说明。
            cfl_max: CFL 上限。SSP-RK3 线性稳定极限 ~1.0，实际可用上限
                取决于网格/物理问题（AUSM+up 低马赫预处理激活时更低）。
                默认 0.5（2026-09-07 从 0.3 上调——0.3 对本项目多数网格
                过于保守、稳态收敛慢）；不稳定时用 CLI `--cfl-max` 回调
                到 0.3 或更低。
            cfl_min: CFL 下限。低于此值说明问题本身很难，继续缩小意义不大。
            growth_factor: 快速放大因子（1.1 = 每次放大 10%）。
            shrink_factor: 重度缩小因子（0.8 = 每次缩小 20%，ratio > 1.1）。
            growth_threshold: 快速下降阈值（ratio < 0.9 即残差下降 >10%）。
            shrink_threshold: 重度恶化阈值（ratio > 1.1 即残差增长 >10%）。
            growth_confirm_steps: 快速放大前需连续满足条件的步数。
            cooldown_steps: 两次 CFL 调节之间的最小间隔步数。
            ramp_steps: 初始爬升步数。此期间 CFL 固定为 cfl_start，不调节。
            crawl_threshold: 慢速爬升阈值上界（ratio < 0.95 视为缓慢收敛）。
            crawl_factor: 慢速放大因子（1.05 = 每次放大 5%）。
            crawl_confirm_steps: 慢速放大前需连续满足条件的步数。
            mild_shrink_factor: 轻度缩小因子（0.9 = 每次缩小 10%，1.0 < ratio ≤ 1.1）。
            trend_window: 窗口趋势判据的窗口长度（步）。取 20 是因为渐近段
                每步只降千分之几，单步比值被死区吞掉，必须跨足够多步才
                能把真实趋势与噪声分开。
            trend_threshold: 窗口内累计残差比的上界。0.995 对应"20 步至少
                累计下降 0.5%"，比实测的健康值（0.99916^20≈0.983）宽松
                一个量级，因此不会把停滞误判成进展。
            trend_factor: 窗口趋势成立时的放大因子（1.1，与 growth_factor
                同——每次只放大 10%，且每次放大后窗口清空重新积累，所以
                从 0.1 爬到 0.5 约需 17 次、~340 步，属于单向温和试探）。
            mild_shrink_confirm_steps: **轻度**恶化（1.0 < ratio <= 1.1）需要
                连续满足的步数。取 3 的理由见模块文档第 5 条：单步残差上升
                在稳态显式迭代里是正常噪声，据此立刻收缩会让 CFL 在任何
                残差略带噪声的算例里单向棘轮下滑到下限。重度恶化
                （ratio > shrink_threshold，含 NaN/inf）**不受这个确认约束**，
                仍然立即收缩——那是真实失稳信号，必须第一时间响应。
            ceiling_release_steps: 连续这么多步没有发生任何收缩时，把软
                上限**放开一档**（乘 1/ceiling_backoff）。没有这条释放
                机制，带噪声的收敛轨迹里偶发的收缩会把上限不断压低、
                放大再也回不去——实测（噪声 sigma=0.4%、均值每步降 0.05%
                的 6000 步轨迹）CFL 会一路滑到 0.034，比 cfl_start=0.1
                还低，等于把第 5/6 条修掉的棘轮效应换了个形式又引回来。
                默认 100（= 5 倍 trend_window）：贴着稳定边界时收缩频繁、
                上限稳稳压住；真正安静的区段才缓慢放开。
            ceiling_backoff: 每次发生收缩时，把"软上限"设为
                `ceiling_backoff * 收缩前的 CFL`（默认 0.95），此后放大
                不得越过这个上限。理由见模块文档第 8 条：没有它，控制器
                会周期性地重新探过稳定边界，每次探过都要付一段残差过冲
                的代价——79 万单元真实网格实测因此变成净负收益。
            trend_shrink_threshold: 轻度收缩的**窗口**阈值，与 trend_threshold
                之间构成迟滞带（默认 0.995 ~ 1.0）。两个方向共用一个阈值时
                CFL 会在稳定边界上无限抖动——实测 128 单元算例稳定后 CFL
                在 0.106 <-> 0.120 之间每约 26 步往复一次（见模块文档第 7
                条）。落在带内（窗口有下降但幅度不够）时 CFL 保持不动。
        """
        # 参数
        self.cfl_start = cfl_start
        self.cfl_max = cfl_max
        self.cfl_min = cfl_min
        self.growth_factor = growth_factor
        self.shrink_factor = shrink_factor
        self.growth_threshold = growth_threshold
        self.shrink_threshold = shrink_threshold
        self.growth_confirm_steps = growth_confirm_steps
        self.cooldown_steps = cooldown_steps
        self.ramp_steps = ramp_steps
        self.crawl_threshold = crawl_threshold
        self.crawl_factor = crawl_factor
        self.crawl_confirm_steps = crawl_confirm_steps
        self.mild_shrink_factor = mild_shrink_factor
        self.trend_window = trend_window
        self.trend_threshold = trend_threshold
        self.trend_factor = trend_factor
        self.mild_shrink_confirm_steps = mild_shrink_confirm_steps
        self.trend_shrink_threshold = trend_shrink_threshold
        self.ceiling_backoff = ceiling_backoff
        self.ceiling_release_steps = ceiling_release_steps
        self.grow_block_ratio = grow_block_ratio

        # 环境变量 `AFCFD_CFL_LEGACY=1`：整体退回 2026-09-14 之前的行为
        # （死区里 CFL 完全不动 + 轻度恶化单步立即收缩），供现场排查与
        # 受控 A/B 用。与 `AFCFD_LOW_MACH_PRECOND` 同一原则："把这个新
        # 机制单独关掉再跑一遍"必须是一条随时可用的路径、不需要改代码。
        # 这条口子不是可选的锦上添花：本次做真实网格 CFL 对照时，"修复
        # 前"的基线只能靠"进程在改动落盘之前启动"这种巧合来获得，没有
        # 干净的 A/B 手段——这就是需要它的直接证据。
        # 三个 CFL 边界参数的协调（见模块文档第 11 条）：`cfl_min > cfl_max`
        # 是自相矛盾的配置，而 `cfl_max` 是调用方更强的意图声明（"这个问题
        # 在这个值以上不稳定"），所以以它为准把 `cfl_min` 钳下去。不静默：
        # 打 warning 说明发生了什么、为什么。
        if self.cfl_min > self.cfl_max:
            logger.warning(
                f"[AdaptiveCFL] cfl_min={self.cfl_min} 高于 cfl_max="
                f"{self.cfl_max}，是自相矛盾的配置；以 cfl_max 为准把 "
                f"cfl_min 钳到 {self.cfl_max}。低 CFL 稳定边界扫描时请显式"
                f"传 cfl_min，否则默认下限会顶住你想测的值。"
            )
            self.cfl_min = self.cfl_max
        if not (self.cfl_min <= self.cfl_start <= self.cfl_max):
            _clamped = min(max(self.cfl_start, self.cfl_min), self.cfl_max)
            logger.warning(
                f"[AdaptiveCFL] cfl_start={self.cfl_start} 不在 "
                f"[{self.cfl_min}, {self.cfl_max}] 内，钳到 {_clamped}。"
            )
            self.cfl_start = _clamped

        self.legacy_mode = os.environ.get("AFCFD_CFL_LEGACY") == "1"
        if self.legacy_mode:
            logger.warning(
                "[AdaptiveCFL] AFCFD_CFL_LEGACY=1：已退回 2026-09-14 之前的"
                "调节策略（死区不调节 + 轻度恶化单步立即收缩）。稳态收敛"
                "步数会明显增多，仅用于对照/排查。"
            )
    
        # 状态
        # 用**钳制后**的 self.cfl_start，不是局部形参 cfl_start——第 11 条的
        # 协调逻辑改的是 self.cfl_start，若这里仍读形参，初值就会绕过协调
        # （cfl_start=0.5 / cfl_max=0.08 时第一步就跑在 0.5 上）。
        self.cfl_number: float = self.cfl_start
        self._prev_residual: float = 0.0
        self._step_count: int = 0
        self._consecutive_good: int = 0      # 连续"快速下降"步数计数
        self._consecutive_crawl: int = 0     # 连续"缓慢收敛"步数计数
        self._consecutive_mild_bad: int = 0  # 连续"轻度恶化"步数计数
        # 软上限：放大不得越过它。None 表示尚未发生过收缩、只受 cfl_max
        # 约束。见模块文档第 8 条。
        self._cfl_ceiling = None
        self._steps_since_shrink: int = 0
        # 绝对基准（见模块文档第 9 条）：自构造/reset 以来见过的最小
        # **有限**残差，以及"软上限设定当时"的残差（用于给时间性释放
        # 加条件）。None 表示尚无有限残差样本。
        self._res_best = None
        self._res_at_ceiling: float = float("inf")
        self._steps_since_last_change: int = 0  # 距上次 CFL 调节的步数
        self._history: List[Tuple[int, float, float]] = []  # (step, cfl, residual)
        # 窗口趋势判据的残差滑动窗口。长度 trend_window+1，使 [0] 与 [-1]
        # 正好相隔 trend_window 步。任何恶化或一次放大都会清空它。
        self._res_window: Deque[float] = deque(maxlen=max(2, trend_window + 1))

    def update(self, current_residual: float) -> float:
        """根据当前步残差更新 CFL，返回下一步使用的 CFL 值。

        调用时机：step() 末尾，残差范数已计算出来之后。

        流程：
            1. 记录历史
            2. 首步 / 爬升阶段：仅记录，不调节
            3. 计算残差比 ratio = current / previous
            4. 判断是否满足调节条件（死区 + 冷却期 + 连续确认）
            5. 执行调节（放大 / 缩小 / 不变）

        Args:
            current_residual: 当前步的残差 RMS 范数

        Returns:
            cfl_number: 下一步使用的 CFL 值
        """
        self._step_count += 1
        self._steps_since_last_change += 1
        self._steps_since_shrink += 1
        # 绝对基准（见模块文档第 9 条）：先更新"到过的最好残差"，本步
        # 后续所有判断都能用上它。非有限值不入基准（那是发散本身，
        # 由 ratio=+inf 的 severe 分支处理）。
        if math.isfinite(current_residual) and current_residual > 0:
            if self._res_best is None or current_residual < self._res_best:
                self._res_best = current_residual

        # 软上限的缓慢释放（见模块文档第 8 条与 ceiling_release_steps）。
        # 释放额外要求"当前残差不差于上限设定当时"（第 9 条 (b)）：释放的
        # 理由是"问题随收敛变容易了"，残差比当时更差时这个理由不成立，
        # 释放就只是去重探一个已知失败的 CFL——实测 step 129 那次正是
        # 这样回到了 step 24 已经失败过的 0.064，并在 45 步后发散。
        _release_justified = (
            not math.isfinite(self._res_at_ceiling)
            or (math.isfinite(current_residual)
                and current_residual <= self._res_at_ceiling))
        if (self._cfl_ceiling is not None
                and _release_justified
                and self._steps_since_shrink >= self.ceiling_release_steps):
            self._cfl_ceiling = min(
                self.cfl_max, self._cfl_ceiling / self.ceiling_backoff)
            self._steps_since_shrink = 0
            if self._cfl_ceiling >= self.cfl_max - 1e-12:
                self._cfl_ceiling = None   # 已放开到 cfl_max，不再需要
        self._history.append((self._step_count, self.cfl_number, current_residual))
        self._res_window.append(current_residual)

        # 首步：无前值可比，仅记录
        if self._prev_residual <= 0:
            self._prev_residual = current_residual
            return self.cfl_number

        # 爬升阶段：CFL 固定不动，建立残差基线
        if self._step_count <= self.ramp_steps:
            self._prev_residual = current_residual
            return self.cfl_number

        ratio = current_residual / max(self._prev_residual, 1e-30)
        # NaN/inf 防护（2026-08-25 代码审查）：残差发散成 NaN/inf 时，
        # 五区间判据的所有比较对 NaN 都为 False，会永久落入死区分支、
        # 控制器对发散完全无反应。基线或当前值非有限时一律视为最严重
        # 恶化（ratio=+inf → 立即重度缩小，CFL 快速降至下限）。
        prev_ok = math.isfinite(self._prev_residual) and self._prev_residual > 0
        if not math.isfinite(current_residual) or not prev_ok:
            ratio = float("inf")

        # --- 调节判断 ---

        if ratio < self.growth_threshold:
            # 快速下降（ratio < 0.9）→ grow
            self._consecutive_good += 1
            self._consecutive_crawl = 0
            self._consecutive_mild_bad = 0
        elif ratio < self.crawl_threshold:
            # 缓慢收敛（0.9 ≤ ratio < 0.95）→ crawl
            self._consecutive_crawl += 1
            self._consecutive_good = 0
            self._consecutive_mild_bad = 0
            if (self._consecutive_crawl >= self.crawl_confirm_steps
                    and self._steps_since_last_change >= self.cooldown_steps
                    and not self._growth_blocked(current_residual)):
                old_cfl = self.cfl_number
                self.cfl_number = min(
                    self.cfl_number * self.crawl_factor, self._growth_cap()
                )
                self._consecutive_crawl = 0
                self._steps_since_last_change = 0
                self._res_window.clear()
                if abs(self.cfl_number - old_cfl) > 1e-10:
                    logger.info(
                        f"[AdaptiveCFL] Step {self._step_count}: "
                        f"CFL {old_cfl:.3f} → {self.cfl_number:.3f} "
                        f"(crawl, ratio={ratio:.3f})"
                    )
            self._prev_residual = current_residual
            return self.cfl_number
        elif ratio > 1.0:
            # 恶化：分轻度 (1.0 < ratio ≤ 1.1) 和重度 (ratio > 1.1)
            self._consecutive_good = 0
            self._consecutive_crawl = 0
            severe = ratio > self.shrink_threshold
            factor = self.shrink_factor if severe else self.mild_shrink_factor
            label = "shrink" if severe else "shrink_mild"
            # **轻度恶化需要连续确认**（2026-09-14，见模块文档第 5 条）：
            # 单步残差上升在稳态显式迭代里是正常噪声，旧实现对它立刻
            # ×0.9，使 CFL 在任何残差略带噪声的算例里单向棘轮下滑到下限
            # ——真实实测（128 单元压力扰动算例，4000 步）：grow 触发 114
            # 次、shrink_mild 触发 112 次，净效果 1.1^114 × 0.9^112 ≈ 0.39，
            # CFL 从 0.1 一路被压到下限 0.05。同函数内 grow/crawl 本来都
            # 要求连续 5 步确认，唯独这一支不要求，是不对称的疏漏。
            # 重度恶化（含 NaN/inf，ratio=+inf）不受确认约束、仍然立即
            # 收缩——那是真实失稳，必须第一时间响应。
            if severe:
                self._consecutive_mild_bad = 0
            else:
                self._consecutive_mild_bad += 1
            # 轻度恶化还要过**窗口判据**（2026-09-14 同批第三处，见模块
            # 文档第 6 条）：只要窗口内（自上次 CFL 调节以来的
            # trend_window 步）残差**确有累计下降**，就说明当前 CFL 整体
            # 是在起作用的，个别连续几步的小幅上升不构成收缩理由。窗口
            # 还没攒满时退回纯连续确认（保留调节后头 20 步的快速保护）。
            window_full = (self._res_window.maxlen is not None
                           and len(self._res_window) >= self._res_window.maxlen)
            # 迟滞带（见模块文档第 7 条）：放大要求窗口累计 < trend_threshold
            # (0.995)，收缩要求窗口累计 > trend_shrink_threshold (1.0)，两者
            # 之间留出空档。共用一个阈值时 CFL 必然在稳定边界上无限抖动。
            window_worsened = False
            if window_full:
                r_old, r_new = self._res_window[0], self._res_window[-1]
                window_worsened = (
                    math.isfinite(r_old) and math.isfinite(r_new) and r_old > 0
                    and r_new / r_old > self.trend_shrink_threshold)
            if severe or self.legacy_mode:
                confirmed = True
            elif window_full:
                # 与放大同一个时间尺度、同一个窗口量，只是阈值留了迟滞
                confirmed = (window_worsened
                             and self._consecutive_mild_bad
                             >= self.mild_shrink_confirm_steps)
            else:
                # 窗口盲区（刚调节过、还没攒满 trend_window 步）：这里如果
                # 退回"连续 3 步确认"，噪声会稳定地在盲区内凑出 3 连升，
                # 于是每次放大后马上被收缩抵消——实测就是 CFL 在
                # 0.050 <-> 0.055 之间无限往复（见模块文档第 6 条）。
                # 盲区内只对**持续**恶化反应：连续 trend_window//2 步。
                # 噪声几乎不可能连出这么多步，而真实的单调上升 10 步内
                # 就会被抓到；更快的失稳本来就走 severe 分支。
                confirmed = (self._consecutive_mild_bad
                             >= max(self.mild_shrink_confirm_steps,
                                    self.trend_window // 2))
            if not confirmed:
                self._prev_residual = current_residual
                return self.cfl_number
            self._consecutive_mild_bad = 0
            if self._steps_since_last_change >= self.cooldown_steps:
                old_cfl = self.cfl_number
                # 软上限：记住"这个 CFL 失败过"，此后放大不得越过它
                # （见模块文档第 8 条）。取收缩前的值而不是收缩后的值，
                # 这样上限随每次失败几何式下降、从上方收敛到稳定边界，
                # 而不是一步跳到远低于边界的位置。
                # 软上限的下界**只有在 cfl_start 本身尚未被证伪时**才取
                # cfl_start：那是调用方断言过稳定的保守起始值（见
                # cfl_start 参数文档），把"能否回到起点"也禁掉就是把第
                # 5/6 条修掉的棘轮换个形式引回来（实测会让带噪声的收敛
                # 轨迹滑到 0.034 << cfl_start=0.1）。
                #
                # **真实 bug 修复（2026-09-15）**：但一旦收缩发生在**低于**
                # cfl_start 的 CFL 上，就已经有直接证据说明 cfl_start 对
                # 这个问题不稳定，再用它兜底等于把上限抬到一个**已知不稳**
                # 的值上，第 8 条这道保护被完全抵消。真实复现（79 万单元
                # cube_demo，P1 真实内容、模态滤波器关闭，稳定 CFL 约
                # 0.063 而 cfl_start=0.1）：上限被钉在 0.1 形同不存在，
                # 控制器 step 66 爬到 0.0697 -> step 79 被判定过高退回
                # 0.0627 -> step 100 **又爬回 0.0690** -> step 102 残差
                # 一步从 2.17e9 跳到 1.87e25，step 103 发散。CFL 历史在
                # 修复前后的两次运行里逐位相同，死亡步号也相同（103），
                # 这正是第 8 条要消灭的"周期性穿越稳定边界"本身。
                #
                # 改法是最小的、有证据支撑的：`old_cfl >= cfl_start` 时
                # 行为逐位不变（那正是第 8 条原本设计的工况）；
                # `old_cfl < cfl_start` 时下界改用**收缩后的 CFL**——它
                # 保证上限不会挡住"停在当前值"（不构成棘轮），同时绝不
                # 高于刚刚失败的那个值。恢复仍由 ceiling_release_steps
                # 这条时间性释放负责，不依赖 cfl_start。
                #
                # 注意这只约束**放大**：真实持续恶化时 cfl_number 仍会
                # 被收缩到 cfl_min 以下界限，不受软上限影响。
                if old_cfl >= self.cfl_start:
                    ceiling_floor = self.cfl_start
                else:
                    ceiling_floor = min(
                        max(old_cfl * factor, self.cfl_min), self.cfl_max)
                ceiling = max(old_cfl * self.ceiling_backoff, ceiling_floor)
                self._cfl_ceiling = (ceiling if self._cfl_ceiling is None
                                     else min(self._cfl_ceiling, ceiling))
                # 记住"上限是在多差的残差上设的"，供时间性释放做条件判断
                # （见模块文档第 9 条 (b)）。上限只会下降，所以这里也只
                # 记录更严格（更小）的那个残差基准。
                if math.isfinite(current_residual):
                    self._res_at_ceiling = min(self._res_at_ceiling,
                                               current_residual)
                self._steps_since_shrink = 0
                # 双向钳制（见模块文档第 11 条）：下界 cfl_min 之外还必须
                # 钳住上界 cfl_max——否则 cfl_number 低于 cfl_min 时这一行
                # 会把"收缩"变成放大、并突破调用方给定的硬上限。
                self.cfl_number = min(
                    max(self.cfl_number * factor, self.cfl_min), self.cfl_max)
                self._steps_since_last_change = 0
                self._res_window.clear()
                # 真实 bug 修复（2026-08-31，用户直接观察到真实日志报出
                # "CFL 0.050 → 0.050 (shrink_mild, ratio=1.001)"发现）：
                # CFL 已经触到下限 cfl_min 时，`max(cfl*factor, cfl_min)`
                # 会把结果 clip 回与 old_cfl 完全相同的值——数值上什么都
                # 没变，却无条件打印出"已调节"的日志，具有误导性（看起来
                # 像是控制器在正常工作、CFL 却诡异地不降低，实际上是已经
                # 到底、调节动作是空操作）。crawl（180行）/grow（222行）
                # 两个分支早就有 `abs(new-old)>1e-10` 才打印的判断，唯独
                # shrink 分支遗漏了同一道防护，是同一函数内三个并列分支
                # 彼此不一致的真实疏漏，不是三处案例中特意如此设计。
                if abs(self.cfl_number - old_cfl) > 1e-10:
                    logger.info(
                        f"[AdaptiveCFL] Step {self._step_count}: CFL {old_cfl:.3f} → "
                        f"{self.cfl_number:.3f} ({label}, ratio={ratio:.3f})"
                    )
            self._prev_residual = current_residual
            return self.cfl_number
        else:
            # 死区（0.95 ≤ ratio ≤ 1.0）：单步比值分不出"慢速收敛"和
            # "停滞"，所以不按单步比值动 CFL；但也不能就此不动——显式
            # 稳态迭代的渐近段本来就长期停在这个区间（实测每步
            # ratio≈0.999 且持续单调下降），旧实现在这里直接 return，
            # CFL 于是被永久钉在 cfl_start，是一个真实的收敛速度缺陷。
            # 改由窗口趋势判据接管，见 `_maybe_grow_on_trend`。
            # 连续计数照旧复位：否则 good/crawl 步隔着死区步交替也能凑满
            # 确认步数，与文档的"连续 N 步确认"语义不符。
            self._consecutive_good = 0
            self._consecutive_crawl = 0
            self._consecutive_mild_bad = 0
            self._maybe_grow_on_trend()
            self._prev_residual = current_residual
            return self.cfl_number

        # --- 快速放大 CFL（需要连续确认 + 冷却期）---

        if (self._consecutive_good >= self.growth_confirm_steps
                and self._steps_since_last_change >= self.cooldown_steps
                and not self._growth_blocked(current_residual)):
            old_cfl = self.cfl_number
            self.cfl_number = min(self.cfl_number * self.growth_factor,
                                  self._growth_cap())
            self._consecutive_good = 0
            self._steps_since_last_change = 0
            self._res_window.clear()
            if abs(self.cfl_number - old_cfl) > 1e-10:
                logger.info(
                    f"[AdaptiveCFL] Step {self._step_count}: CFL {old_cfl:.3f} → "
                    f"{self.cfl_number:.3f} (grow, ratio={ratio:.3f})"
                )

        self._prev_residual = current_residual
        return self.cfl_number

    def _growth_blocked(self, current_residual: float) -> bool:
        """当前残差比"自己到过的最好成绩"差 `grow_block_ratio` 倍以上时
        禁止放大（模块文档第 9 条 (a)）。

        这只**阻止放大**，从不主动收缩——所以它不可能引入第 5/6 条修掉的
        那种棘轮效应。健康轨迹上残差严格单调下降、当前值就是历史最好值，
        比值恒为 1.00，本函数恒返回 False，行为逐位不变（实测见第 9 条
        列出的逐算例数据：健康档 1.00，失败档 20 步内 17.5）。
        """
        if self._res_best is None or self._res_best <= 0:
            return False
        if not math.isfinite(current_residual):
            return True   # 非有限值绝不是放大的时机
        return current_residual / self._res_best > self.grow_block_ratio

    def _growth_cap(self) -> float:
        """放大的实际上限：`cfl_max` 与软上限取小。

        软上限（`_cfl_ceiling`）由每次收缩时设置，见模块文档第 8 条。
        """
        if self._cfl_ceiling is None:
            return self.cfl_max
        return min(self.cfl_max, self._cfl_ceiling)

    def _maybe_grow_on_trend(self) -> None:
        """窗口趋势判据：最近 `trend_window` 步累计确有下降则放大 CFL。

        只在死区（单步比值 0.95~1.0）里调用——其余四个区间本来就已经
        各自动作。完整动机见模块文档第 4 条。

        要求窗口必须**攒满**才判断：窗口每次放大后清空，所以两次放大
        之间至少相隔 trend_window 步，放大速率因此自带上限（默认 20 步
        最多 ×1.1），不需要额外的振荡抑制。冷却期仍然叠加生效，保证与
        shrink/crawl 之间也不会挤在一起。
        """
        if self.legacy_mode:
            return   # 见 __init__ 里 AFCFD_CFL_LEGACY 的说明
        if self._res_window.maxlen is None or len(self._res_window) < self._res_window.maxlen:
            return
        if self._steps_since_last_change < self.cooldown_steps:
            return
        r_old, r_new = self._res_window[0], self._res_window[-1]
        if not (math.isfinite(r_old) and math.isfinite(r_new)) or r_old <= 0:
            return
        if r_new / r_old >= self.trend_threshold:
            return   # 窗口内没有实质进展：停滞或原地震荡，不放大
        # 绝对基准闸门（模块文档第 9 条 (a)）：窗口内确有下降、但整条轨迹
        # 已经比自己到过的最好残差差 grow_block_ratio 倍以上时不放大。
        # `_res_window[-1]` 就是本步刚 append 进去的当前残差。这一条正是
        # 第 9 条实测那两次不该发生的放大（15.8 倍处、4.6 倍处）的阻断点。
        if self._growth_blocked(r_new):
            return

        old_cfl = self.cfl_number
        self.cfl_number = min(self.cfl_number * self.trend_factor, self._growth_cap())
        self._steps_since_last_change = 0
        self._res_window.clear()
        if abs(self.cfl_number - old_cfl) > 1e-10:
            logger.info(
                f"[AdaptiveCFL] Step {self._step_count}: CFL {old_cfl:.3f} → "
                f"{self.cfl_number:.3f} (grow_trend, {self.trend_window} 步累计 "
                f"ratio={r_new / r_old:.5f})"
            )

    def reset(self):
        """重置控制器状态。

        调用时机：Order Continuation 阶数切换时。阶数变化导致残差跳变
        （插值误差），不应触发 CFL 缩小。重置后重新开始爬升。
        """
        logger.info(
            f"[AdaptiveCFL] Reset at step {self._step_count} "
            f"(CFL was {self.cfl_number:.3f})"
        )
        self.cfl_number = self.cfl_start
        self._res_window.clear()
        self._consecutive_mild_bad = 0
        # 阶数切换 = 换了一个离散问题，旧的稳定边界不再适用，软上限作废
        self._cfl_ceiling = None
        self._steps_since_shrink = 0
        # 阶数切换 = 换了一个离散问题，旧的残差量级不可比，绝对基准同样
        # 作废（否则新阶数的第一步残差会被拿去和旧阶数的最好值比）
        self._res_best = None
        self._res_at_ceiling = float("inf")
        self._prev_residual = 0.0
        self._step_count = 0
        self._consecutive_good = 0
        self._consecutive_crawl = 0
        self._steps_since_last_change = 0
        # 不清空 _history（保留诊断信息）

    @property
    def step_count(self) -> int:
        """已处理的步数。"""
        return self._step_count

    @property
    def history(self) -> List[Tuple[int, float, float]]:
        """CFL 调节历史：(step, cfl_used, residual) 列表。"""
        return list(self._history)

    def summary(self) -> str:
        """返回当前状态的单行摘要（供日志输出）。"""
        return (
            f"CFL={self.cfl_number:.3f} "
            f"(steps={self._step_count}, good_streak={self._consecutive_good}, "
            f"crawl_streak={self._consecutive_crawl}, "
            f"since_change={self._steps_since_last_change})"
        )

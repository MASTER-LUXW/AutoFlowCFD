"""自适应 CFL 控制器的**绝对基准**（`adaptive_cfl.py` 模块文档第 9 条）。

## 修的是什么

第 4~8 条的全部判据看的都是**相对最近过去**的量——单步残差比值、20 步
累计窗口。于是一条"缓慢离开收敛盆"的轨迹可以穿过它们全部：79 万单元
cube_demo 上一次**已跑完**的真实运行（同一 P1 检查点、模态滤波器关闭
即真实 P1 内容、`cfl_start=0.1`）里，残差在 step 2 到达全程最低
2.744e8，随后冲到 5.26e9（最低点的 19.2 倍），再缓慢衰减到 1.02e9
（仍是 3.7 倍），最后在 step 174 发散成 NaN。

控制器在这全程里**两次放大 CFL**：

    step  45: 0.0576 -> 0.0608   此时残差 = 最低点的 15.8 倍
    step 129: 0.0608 -> 0.0640   此时残差 = 最低点的  4.6 倍

两次都"合规"：5.26e9 -> 1.26e9 这段衰减确实让 20 步窗口累计比值低于
0.995，是一个全局已失败轨迹里的真实局部下降。step 129 那次还叠加了
`ceiling_release_steps` 的时间性释放——它把上限放开，让控制器回到
step 24 已经失败过的 0.0640。

## 判据与数据

阈值的依据是同一批日志里逐条算例实测的 `res/res_best` 上界：

    健康档（legacy；以及固定 CFL 0.03 的真实 P1 档）  全程恒为  1.00
    失败档（off / sensor / 自适应真实 P1）            20 步内达 17.50

两者隔着一个数量级以上，所以默认阈值 2.0 离两侧都很远。本文件同时
钉住两个方向：失败轨迹上必须阻断放大（`TestReplayRealDivergedRun`），
健康轨迹上必须**逐位不变**（`TestHealthyTrajectoryUnchanged`）。

## 这些测试**不**声称什么

`TestReplayRealDivergedRun` 是**开环**回放：喂进去的是原始运行记录下来
的残差序列，而原始序列本身是在"控制器做了那两次放大"的条件下产生的。
所以它证明的是**决策逻辑在那些时刻不会再放大**，不是"这条运行会因此
存活"。由此停住的 CFL（0.0576）是否落在稳定边界内尚无数据——现有真实
数据只确定 0.03 稳定、0.0608 不稳定（后者即本文件回放的这条运行）。
本项目已有三次"短窗口得出符号相反结论"的先例，所以只断言有数据支撑
的那部分。
"""

import math

import pytest

from autoflowcfd.core.time_integration.adaptive_cfl import AdaptiveCFLController


#: 79 万单元 cube_demo 真实运行（FILTER off = 真实 P1 内容，cfl_start=0.1）
#: 逐步残差，step 1..173，第 174 步发散成 NaN。取自
#: scratchpad/pc/filt/f_true_adaptive.log。
REAL_DIVERGED_RESIDUALS = [
    3.265703e+08, 2.744303e+08, 4.765396e+08, 7.149991e+08, 1.085485e+09,
    1.698761e+09, 2.130358e+09, 2.438276e+09, 2.697886e+09, 3.124516e+09,
    3.450629e+09, 4.102334e+09, 4.155052e+09, 4.138862e+09, 4.259945e+09,
    4.334958e+09, 4.478568e+09, 4.565423e+09, 4.707240e+09, 4.802526e+09,
    4.936377e+09, 5.035149e+09, 5.158325e+09, 5.259983e+09, 5.364119e+09,
    5.617937e+09, 5.583936e+09, 5.540923e+09, 5.470164e+09, 5.443102e+09,
    5.360822e+09, 5.335757e+09, 5.241458e+09, 5.211041e+09, 5.115215e+09,
    5.070820e+09, 4.974956e+09, 4.921964e+09, 4.824008e+09, 4.764184e+09,
    4.666533e+09, 4.603639e+09, 4.498208e+09, 4.435578e+09, 4.328538e+09,
    4.260892e+09, 4.129286e+09, 4.077832e+09, 4.007989e+09, 3.964087e+09,
    3.896596e+09, 3.848322e+09, 3.779282e+09, 3.725957e+09, 3.654884e+09,
    3.596788e+09, 3.525150e+09, 3.463666e+09, 3.392241e+09, 3.333056e+09,
    3.260021e+09, 3.204476e+09, 3.132333e+09, 3.078202e+09, 3.009963e+09,
    2.957189e+09, 2.892447e+09, 2.841299e+09, 2.780315e+09, 2.731512e+09,
    2.672252e+09, 2.626179e+09, 2.569228e+09, 2.525187e+09, 2.471336e+09,
    2.429455e+09, 2.379200e+09, 2.339893e+09, 2.292703e+09, 2.255810e+09,
    2.212048e+09, 2.177607e+09, 2.136728e+09, 2.104472e+09, 2.066415e+09,
    2.035758e+09, 2.000115e+09, 1.970276e+09, 1.938489e+09, 1.908854e+09,
    1.880254e+09, 1.850528e+09, 1.825274e+09, 1.796730e+09, 1.774347e+09,
    1.747305e+09, 1.727006e+09, 1.701425e+09, 1.683083e+09, 1.658410e+09,
    1.642066e+09, 1.618408e+09, 1.604107e+09, 1.581513e+09, 1.568296e+09,
    1.546947e+09, 1.532933e+09, 1.514297e+09, 1.499491e+09, 1.482992e+09,
    1.468120e+09, 1.453112e+09, 1.439608e+09, 1.424357e+09, 1.411853e+09,
    1.397412e+09, 1.385977e+09, 1.372571e+09, 1.362017e+09, 1.349661e+09,
    1.340021e+09, 1.328356e+09, 1.319228e+09, 1.308384e+09, 1.299692e+09,
    1.289586e+09, 1.281446e+09, 1.271996e+09, 1.264354e+09, 1.255386e+09,
    1.248982e+09, 1.245719e+09, 1.245526e+09, 1.245572e+09, 1.247505e+09,
    1.249355e+09, 1.252261e+09, 1.255490e+09, 1.259071e+09, 1.263384e+09,
    1.268155e+09, 1.272134e+09, 1.275685e+09, 1.279858e+09, 1.278681e+09,
    1.265615e+09, 1.248177e+09, 1.231573e+09, 1.215834e+09, 1.200566e+09,
    1.183816e+09, 1.169478e+09, 1.155246e+09, 1.142901e+09, 1.130810e+09,
    1.120182e+09, 1.109759e+09, 1.100640e+09, 1.091472e+09, 1.083747e+09,
    1.075622e+09, 1.068899e+09, 1.061859e+09, 1.055678e+09, 1.049567e+09,
    1.043810e+09, 1.039307e+09, 1.035252e+09, 1.031943e+09, 1.028471e+09,
    1.025951e+09, 1.022954e+09, 1.021035e+09,
]

#: 该运行的真实参数（与 f_true_adaptive 启动时一致）
REAL_RUN_KWARGS = dict(cfl_start=0.1, cfl_max=0.5, cfl_min=0.05)

#: 关掉绝对基准闸门：阈值取一个任何残差比都到不了的值，等价于第 9 条
#: 之前的行为。用于 fail-then-pass 对照。
GATE_OFF = 1e30


def _replay(residuals, **kwargs):
    """开环回放一条残差序列，返回逐步的 (cfl_before_update, cfl_after)。"""
    c = AdaptiveCFLController(**kwargs)
    trace = []
    for r in residuals:
        before = c.cfl_number
        after = c.update(r)
        trace.append((before, after))
    return c, trace


def _growth_steps(trace):
    """回放轨迹里 CFL 被**放大**的步号（1-based）。"""
    return [i + 1 for i, (b, a) in enumerate(trace) if a > b + 1e-12]


class TestReplayRealDivergedRun:
    """真实已发散运行的开环回放：两次不该发生的放大必须被阻断。"""

    def test_unfixed_controller_reproduces_both_bad_growths(self):
        """先确认缺陷本身可复现（fail-then-pass 的 fail 半边）。

        闸门关掉时控制器必须在残差远高于历史最好值的区段放大 CFL——
        否则本文件测的就不是真实缺陷。
        """
        _, trace = _replay(REAL_DIVERGED_RESIDUALS,
                           grow_block_ratio=GATE_OFF, **REAL_RUN_KWARGS)
        grew = _growth_steps(trace)
        assert grew, "闸门关掉时应当复现出放大动作，否则回放没有覆盖到缺陷"
        best = min(REAL_DIVERGED_RESIDUALS)
        # 每一次放大都发生在残差远高于历史最好值处
        ratios = [REAL_DIVERGED_RESIDUALS[s - 1] / best for s in grew]
        assert max(ratios) > 3.0, (
            f"放大发生处的 res/res_best 最大只有 {max(ratios):.2f}，"
            f"与日志记录的 15.8x / 4.6x 不符")

    def test_fix_blocks_every_growth_above_the_threshold(self):
        """修复后：任何 `res/res_best > 2.0` 的步都不得放大。"""
        _, trace = _replay(REAL_DIVERGED_RESIDUALS, **REAL_RUN_KWARGS)
        best_so_far = math.inf
        offenders = []
        for i, (before, after) in enumerate(trace):
            r = REAL_DIVERGED_RESIDUALS[i]
            best_so_far = min(best_so_far, r)
            if after > before + 1e-12 and r / best_so_far > 2.0:
                offenders.append((i + 1, r / best_so_far, before, after))
        assert not offenders, (
            f"仍在残差远高于历史最好值处放大 CFL：{offenders}")

    def test_fix_removes_all_growth_events_on_this_trajectory(self):
        """这条轨迹上修复后不应再有**任何**放大事件。

        它自 step 2 之后全程都在历史最好值的 3.7 倍以上、再没回去过，
        所以闸门对每一次放大尝试都成立。
        """
        _, fixed = _replay(REAL_DIVERGED_RESIDUALS, **REAL_RUN_KWARGS)
        _, unfixed = _replay(REAL_DIVERGED_RESIDUALS,
                             grow_block_ratio=GATE_OFF, **REAL_RUN_KWARGS)
        assert _growth_steps(fixed) == []
        assert len(_growth_steps(unfixed)) >= 2

    def test_fix_does_not_raise_the_peak_cfl(self):
        """闸门只**阻止**放大、从不主动收缩，所以整条回放的 CFL 峰值
        不可能被它抬高。

        注意不能逐步比较两条轨迹：一旦分叉，收缩就是在不同基数上做乘法
        ——实测 step 143 对照档从 0.0608 收缩到 0.0547，而修复档一直停在
        0.0576，于是那一步"修复后反而更高"。那是对照档先爬得更高又跌
        下来的结果，不是闸门造成的收缩差异。峰值比较才是闸门单向性的
        正确判据。
        """
        _, tf = _replay(REAL_DIVERGED_RESIDUALS, **REAL_RUN_KWARGS)
        _, tu = _replay(REAL_DIVERGED_RESIDUALS,
                        grow_block_ratio=GATE_OFF, **REAL_RUN_KWARGS)
        assert max(a for _, a in tf) <= max(a for _, a in tu) + 1e-12

    def test_gate_is_only_consulted_in_growth_paths(self):
        """结构判据：闸门谓词只出现在放大路径上，绝不影响收缩分支——
        这是"不可能引入模块文档第 5/6 条修掉的反向棘轮"的依据。
        """
        import inspect

        from autoflowcfd.core.time_integration import adaptive_cfl
        lines = inspect.getsource(adaptive_cfl).splitlines()
        hits = [i for i, ln in enumerate(lines)
                if "_growth_blocked(" in ln and "def _growth_blocked" not in ln]
        assert len(hits) == 3, (
            f"闸门调用点应为 grow / crawl / grow_trend 三处，实为 {len(hits)}")
        for i in hits:
            ctx = "\n".join(lines[max(0, i - 12):i + 1])
            assert ("growth_confirm_steps" in ctx
                    or "crawl_confirm_steps" in ctx
                    or "trend_threshold" in ctx), (
                f"第 {i + 1} 行的闸门调用不在放大路径的上下文里：\n{ctx}")

    def test_final_cfl_is_strictly_lower_than_unfixed(self):
        """净效果必须可见：末步 CFL 严格低于关掉闸门的对照。"""
        cf, _ = _replay(REAL_DIVERGED_RESIDUALS, **REAL_RUN_KWARGS)
        cu, _ = _replay(REAL_DIVERGED_RESIDUALS,
                        grow_block_ratio=GATE_OFF, **REAL_RUN_KWARGS)
        assert cf.cfl_number < cu.cfl_number


class TestHealthyTrajectoryUnchanged:
    """健康轨迹（残差严格单调下降）上必须逐位不变。

    实测依据见模块文档第 9 条：legacy 档与固定 CFL 0.03 的真实 P1 档
    全程 `res/res_best` 恒为 1.00——当前值**就是**历史最好值，闸门恒不
    触发。
    """

    @pytest.mark.parametrize("per_step", [0.999, 0.99, 0.95, 0.9, 0.8])
    def test_monotone_decay_is_bit_identical(self, per_step):
        res = [1.0e9 * per_step ** k for k in range(300)]
        cf, tf = _replay(res, **REAL_RUN_KWARGS)
        cu, tu = _replay(res, grow_block_ratio=GATE_OFF, **REAL_RUN_KWARGS)
        assert tf == tu, "单调下降轨迹上闸门改变了 CFL 轨迹"
        assert cf.cfl_number == cu.cfl_number

    def test_monotone_decay_still_grows_cfl(self):
        """反向确认：健康轨迹上 CFL 仍然会被放大（闸门没有把机制废掉）。"""
        res = [1.0e9 * 0.999 ** k for k in range(300)]
        _, trace = _replay(res, **REAL_RUN_KWARGS)
        assert _growth_steps(trace), "健康轨迹上 CFL 应当仍会放大"

    def test_noisy_but_converging_is_unchanged(self):
        """带噪声但在收敛：残差偶有上升，但从不偏离历史最好值 2 倍以上，
        闸门同样不该介入（这正是第 5/6 条针对的工况）。"""
        import random
        rng = random.Random(3)
        res, v = [], 1.0e9
        for _ in range(400):
            v *= 0.999 * (1.0 + rng.uniform(-0.004, 0.004))
            res.append(v)
        assert max(r / min(res[:i + 1]) for i, r in enumerate(res)) < 2.0
        _, tf = _replay(res, **REAL_RUN_KWARGS)
        _, tu = _replay(res, grow_block_ratio=GATE_OFF, **REAL_RUN_KWARGS)
        assert tf == tu


class TestGrowthBlockedPredicate:
    """闸门谓词本身。"""

    def test_no_baseline_yet_does_not_block(self):
        c = AdaptiveCFLController(**REAL_RUN_KWARGS)
        assert c._growth_blocked(1.0e9) is False

    def test_blocks_exactly_above_the_ratio(self):
        c = AdaptiveCFLController(**REAL_RUN_KWARGS)
        c._res_best = 1.0
        assert c._growth_blocked(1.9) is False
        assert c._growth_blocked(2.0) is False   # 阈值本身不算越界
        assert c._growth_blocked(2.001) is True

    def test_non_finite_always_blocks(self):
        c = AdaptiveCFLController(**REAL_RUN_KWARGS)
        c._res_best = 1.0
        assert c._growth_blocked(float("nan")) is True
        assert c._growth_blocked(float("inf")) is True

    def test_baseline_tracks_the_minimum_not_the_last_value(self):
        c = AdaptiveCFLController(**REAL_RUN_KWARGS)
        for r in (1.0e9, 5.0e8, 9.0e8, 7.0e8):
            c.update(r)
        assert c._res_best == pytest.approx(5.0e8)

    def test_non_finite_residual_does_not_poison_the_baseline(self):
        c = AdaptiveCFLController(**REAL_RUN_KWARGS)
        c.update(1.0e9)
        c.update(5.0e8)
        c.update(float("nan"))
        assert c._res_best == pytest.approx(5.0e8)

    def test_reset_clears_the_baseline(self):
        """阶数切换换了一个离散问题，旧阶数的残差量级不可比。"""
        c = AdaptiveCFLController(**REAL_RUN_KWARGS)
        for r in (1.0e9, 5.0e8):
            c.update(r)
        c.reset()
        assert c._res_best is None
        assert not math.isfinite(c._res_at_ceiling)


class TestCeilingReleaseIsGatedOnResidual:
    """软上限的时间性释放必须要求"残差不差于上限设定当时"。

    释放（模块文档第 8 条的 `ceiling_release_steps`）的理由是"问题随收敛
    变容易了"。残差比设定当时更差时这个理由不成立，释放就只是去重探一个
    已知失败的 CFL——真实运行里 step 129 那次放大正是这样回到了 step 24
    已经失败过的 0.0640，45 步后发散。
    """

    def _armed(self, release=10):
        """构造一个"上限已在残差 1.0e9 处设定"的控制器。"""
        c = AdaptiveCFLController(ceiling_release_steps=release,
                                  **REAL_RUN_KWARGS)
        c._cfl_ceiling = 0.0608
        c._res_at_ceiling = 1.0e9
        c._steps_since_shrink = release   # 时间条件已满足
        return c

    def test_release_happens_when_residual_improved(self):
        c = self._armed()
        c.update(9.0e8)      # 优于设定当时
        assert c._cfl_ceiling > 0.0608

    def test_release_blocked_while_residual_is_worse(self):
        c = self._armed()
        c.update(1.1e9)      # 比设定当时更差
        assert c._cfl_ceiling == pytest.approx(0.0608)

    def test_release_allowed_at_exactly_the_same_residual(self):
        """判据是"不差于"，相等应当放行（不留一个只能靠严格改善才动的
        死角）。"""
        c = self._armed()
        c.update(1.0e9)
        assert c._cfl_ceiling > 0.0608

    def test_non_finite_residual_never_releases(self):
        c = self._armed()
        c.update(float("inf"))
        assert c._cfl_ceiling == pytest.approx(0.0608)

    def test_release_time_condition_still_required(self):
        """残差改善也不能绕过时间条件。"""
        c = self._armed()
        c._steps_since_shrink = 0
        c.update(1.0e8)
        assert c._cfl_ceiling == pytest.approx(0.0608)

    def test_unset_baseline_keeps_legacy_release_behaviour(self):
        """从未记录过基准（`_res_at_ceiling` 为 inf）时行为不变——这是
        `reset()` 之后、以及旧状态反序列化场景下的形态。"""
        c = AdaptiveCFLController(ceiling_release_steps=10, **REAL_RUN_KWARGS)
        c._cfl_ceiling = 0.0608
        c._steps_since_shrink = 10
        assert not math.isfinite(c._res_at_ceiling)
        c.update(9.9e99)     # 任意差的残差
        assert c._cfl_ceiling > 0.0608

"""轻度收缩的绝对基准闸门（自适应 CFL 第 13 条缺陷，2026-09-24 回归）。

## 钉的是什么

`adaptive_cfl.py` 第 9 条给"放大"加了绝对基准
（`res/res_best > grow_block_ratio` 时禁止放大），但**没有给"轻度收缩"
加对称的一半**。于是启动暂态里的残差回升会把 CFL 一路压到下限，而解其实
一直在改善。

真实数据（179,237 单元网格 P1+SST 长程运行，全默认参数）：

    iter 1980..2420   残差 2.5280e8 -> 2.6739e8   **升 5.77%**
                      Cd    4.7945  ->  4.7183    降 0.0762
                      Cd 在这 441 条记录里**严格单调不增**
    iter  900..3240   Cd 5.5679 -> 4.5310，2340 步零次反转

残差是时间导数的范数、对"当前哪些单元在调整"极其敏感；气动系数是积分量。
暂态里不同区域先后进入调整，前者会升、后者仍在单调收敛。控制器只看残差
比值，就把这件事误判成失稳：全程 3240 步里 CFL 停在下限 0.010 的有 1529
步，step 1819 收缩到下限之后**连续 915 步没有任何调节**。

## 判据

1. `test_gate_blocks_near_the_best_and_allows_real_divergence` —— 闸门在
   实测的三个收缩点（res/res_best ≈ 1.00 / 1.00 / 1.086）全部拦住，在
   真实失稳的量级（17.5）与阈值处（1.5）全部放行。
2. `test_severe_worsening_is_never_gated` —— 重度恶化（含 NaN/inf）不经过
   闸门，必须立即收缩。这是安全性的另一半。
3. `test_replay_of_the_real_trajectory_no_longer_collapses` —— 用一条
   **合成**但形状取自真实运行的残差序列（缓慢下降 + 一段 5.8% 的暂态
   回升）回放控制器：关闸门会塌到下限，开闸门不会。
4. `test_monotone_healthy_trajectory_is_bit_identical` —— 健康轨迹（残差
   严格单调下降）上 `ratio > 1.0` 根本不成立，闸门永不被求值，两档 CFL
   轨迹必须**逐位相同**。这条保证新闸门不改变任何既有已验证结果。
"""

import math

import pytest

from autoflowcfd.core.time_integration.adaptive_cfl import AdaptiveCFLController


def _controller(block_ratio=None, **kw):
    args = dict(cfl_start=0.03, cfl_max=0.06, cfl_min=0.01)
    if block_ratio is not None:
        args["mild_shrink_block_ratio"] = block_ratio
    args.update(kw)
    return AdaptiveCFLController(**args)


def test_default_block_ratio_is_the_measured_value():
    assert _controller().mild_shrink_block_ratio == 1.5, (
        "默认阈值变了 —— 1.5 的两侧依据是实测的（本次运行三次收缩点 "
        "1.00/1.00/1.086，真实失稳 20 步内 17.5），改它要同步更新"
        "adaptive_cfl.py 模块文档第 13 条")


@pytest.mark.parametrize("res_over_best,blocked", [
    (1.000, True),    # 实测收缩点 1（iter 52 / 477 附近）
    (1.086, True),    # 实测收缩点 3（iter 2400，残差升 5.77% 之后）
    (1.499, True),
    (1.500, True),    # 阈值本身：<= 才拦，等于仍拦（闭区间）
    (1.501, False),
    (17.52, False),   # 真实失稳实测量级
])
def test_gate_blocks_near_the_best_and_allows_real_divergence(
        res_over_best, blocked):
    c = _controller()
    c._res_best = 2.46e8
    assert c._mild_shrink_blocked(res_over_best * c._res_best) is blocked


def test_non_finite_residual_is_not_gated():
    """NaN/inf 是真实失稳，必须交给重度分支，不能被闸门拦住。"""
    c = _controller()
    c._res_best = 1.0
    for bad in (float("nan"), float("inf")):
        assert c._mild_shrink_blocked(bad) is False


def test_gate_is_inert_before_any_best_is_recorded():
    c = _controller()
    assert c._res_best is None
    assert c._mild_shrink_blocked(1.0) is False


def test_severe_worsening_is_never_gated():
    """重度恶化必须立即收缩，哪怕残差离历史最好很近（判据 2）。

    构造：先喂一串单调下降把 `_res_best` 压低，再喂一个只比它高 20% 的
    值 —— `res/res_best = 1.2 < 1.5` 会被闸门拦住*轻度*收缩，但单步
    `ratio > shrink_threshold=1.1` 走重度分支，必须照样收缩。
    """
    c = _controller()
    r = 1.0e8
    for _ in range(30):
        c.update(r)
        r *= 0.97
    cfl_before = c.cfl_number
    best = c._res_best
    # 单步跳升 20%（重度），但绝对水平只有历史最好的 1.2 倍
    c.update(best * 1.2)
    assert c._res_best is not None and best * 1.2 / c._res_best < 1.5
    assert c.cfl_number < cfl_before, (
        f"重度恶化没有收缩（{cfl_before:.4f} -> {c.cfl_number:.4f}）——"
        f"绝对基准闸门只允许约束**轻度**分支")


def _transient_trajectory():
    """形状取自真实运行的残差序列：缓慢下降 + 一段暂态回升 + 继续下降。

    真实运行 iter 1980..2420 是 2.5280e8 -> 2.6739e8（升 5.77%），
    之前之后都在缓慢下降（每 20 步累计约 0.99）。这里按同一形状合成，
    步数压到 900 以便单元测试快速跑完。
    """
    seq = []
    r = 7.87e8
    for _ in range(300):          # 缓慢下降段
        r *= 0.9985
        seq.append(r)
    for _ in range(300):          # 暂态回升段（总计约 +5.8%）
        r *= 1.000188
        seq.append(r)
    for _ in range(300):          # 恢复下降
        r *= 0.9985
        seq.append(r)
    return seq


@pytest.mark.parametrize("block_ratio,expect_floor", [(0.0, True), (1.5, False)])
def test_replay_of_the_real_trajectory_no_longer_collapses(
        block_ratio, expect_floor):
    """关闸门会塌到下限、开闸门不会（判据 3）。"""
    c = _controller(block_ratio=block_ratio)
    traj = [c.update(r) for r in _transient_trajectory()]
    at_floor = sum(1 for v in traj if v <= c.cfl_min * 1.05)
    if expect_floor:
        assert at_floor > 0, (
            "关闸门（block_ratio=0）本应复现旧行为、在暂态回升段塌到下限"
            " —— 若这条不再成立，本文件的对照就失去意义了")
    else:
        assert at_floor == 0, (
            f"开闸门后仍有 {at_floor}/{len(traj)} 步停在下限 —— 暂态回升"
            f"不应该把 CFL 压到下限")
        assert max(traj) > c.cfl_start, (
            f"开闸门后 CFL 最大值 {max(traj):.4f} 没有超过起始值 "
            f"{c.cfl_start:.4f}，放大通道被别的东西堵住了")


def test_monotone_healthy_trajectory_is_bit_identical():
    """健康轨迹上两档必须**逐位相同**（判据 4）。

    残差严格单调下降时 `ratio > 1.0` 根本不成立，轻度收缩分支不会进入，
    闸门永不被求值 —— 这条保证新闸门不改变任何既有已验证结果。
    """
    seq = []
    r = 1.0e9
    for _ in range(500):
        r *= 0.995
        seq.append(r)

    c_off = _controller(block_ratio=0.0)
    c_on = _controller(block_ratio=1.5)
    a = [c_off.update(x) for x in seq]
    b = [c_on.update(x) for x in seq]
    assert len(a) == len(b) == len(seq)
    for i, (x, y) in enumerate(zip(a, b)):
        assert x == y, (
            f"第 {i} 步两档 CFL 不同（{x!r} vs {y!r}）—— 健康轨迹上闸门"
            f"本应完全不被求值")
    assert math.isfinite(a[-1]) and a[-1] > 0.0
